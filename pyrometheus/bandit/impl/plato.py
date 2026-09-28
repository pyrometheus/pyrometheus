import numpy as np
import pymbolic.primitives as p
from typing import Dict, List, Union, Tuple
from pymbolic.primitives import Variable
from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.kinetics import (
    RateCoefficient,
    make_arrhenius,
    reaction_progress_rate_expr,
    species_production_rate_expr,
    pairwise_relaxation_time_expr,
    vt_mean_relaxation_rate_expr,
    vt_energy_transfer_expr,
    conc, k_fwd, log_k_eq, exp,
)
from pyrometheus.bandit.chem_expr.thermo import (
    PolynomialParameters,
    SpeciesNASAThermo,
    SpeciesVibrationalThermo,
    make_species_nasa_thermo,
    make_species_vibrational_thermo,
    equilibrium_constant_expr,
    translational_rotational_energy_expr,
    nasa_polynomial_vibrational_energy_expr,
    nasa_polynomial_vibrational_specific_heat_expr,
)

# Standard thermodynamic reference temperature [K], matching the convention
# NASA9 polynomials are fit against (h(T_ref) equals the standard enthalpy
# of formation).
standard_reference_temperature = 298.15

# {{{ Map reaction type from human-readable to plato naming convention

_reaction_dict = {
    "heavy_particle_dissociation": "dh_cat_mt",
    "exchange": "exc",
    "electron_impact_dissociation": "de",
    "electron_impact_ionization": "ie",
    "associative_ionization": "ai",
}

# }}}


# {{{ Temperature substitutions. Convention taken from plato::kinetics_rate.F90

_temp_map = {
    "translational": Variable("temperature"),
    "electron": Variable("temperature")[-1],
    "geometric_ttv": Variable("sqrt")(
        Variable("temperature")[0] *
        Variable("temperature")[1]
    )
}


_rxn_type_temp = {
    "dh_cat_mt": "geometric_ttv",
    "exc": "translational",
    "de": "electron",
    "ie": "electron",
    "ai": "translational",
}

# }}}


class Plato(BaseNamespace):

    one_atm = 101325.0
    ref_pressure = 1.0e5
    avogadro = 6.02214076e23
    boltzmann = 1.380649e-23
    # Plato defines R = k_B N_A (utils_constants.F90); in J/(kmol K).
    gas_constant = 1e3 * boltzmann * avogadro

    def __init__(self, mixture, reaction_set, transfer, plato_db_path):
        from plato import Thermochemistry
        self.thermochem = Thermochemistry(
            solver="thermo",
            mixture=mixture,
            reaction=reaction_set,
            transfer=transfer,
            db_path=plato_db_path,
            quiet=True
        )

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.thermochem, name)(*args)
        else:
            return getattr(self.thermochem, name)

    def finalize(self):
        self.thermochem.finalize()

    @property
    def nonequil_thermo(self):
        return self.thermochem.vt_active


class PlatoMechanism(BaseMechanism):

    def __init__(self,
                 mixture,
                 reaction_set,
                 transfer,
                 plato_db_path,
                 pyro_np=np,
                 hardcode_params=True):
        self.hardcode_params = hardcode_params
        self.namespace = Plato(mixture, reaction_set, transfer, plato_db_path)
        self.nonequil_thermo = self.namespace.nonequil_thermo
        self._build_reaction_list()
        self.make_rates(hardcode_params)
        self.make_thermo()

    def finalize(self):
        self.namespace.finalize()

    # {{{ Assemble reactions data
    def _num_reactions_by_type(self,
                               rxn_type: str):
        rxn_label = _reaction_dict[rxn_type]
        method_name = f"n_reactions_{rxn_label}"
        return self.namespace.__getattr__(method_name)

    def _reaction_getter_by_type(self,
                                 rxn_access: str,
                                 rxn_type: str,
                                 rxn_index: int):
        rxn_label = _reaction_dict[rxn_type]
        method_name = f"get_{rxn_label}_{rxn_access}"
        return self.namespace.__getattr__(method_name, rxn_index)

    def _build_reaction_list(self):
        self._reactions = []

        num_rxn = self._num_reactions_by_type(
            rxn_type="heavy_particle_dissociation",
        )
        for r in range(1, num_rxn + 1):
            i, j, k = self._reaction_getter_by_type(
                rxn_access="species",
                rxn_type="heavy_particle_dissociation",
                rxn_index=r
            )
            a, b, c = self._reaction_getter_by_type(
                rxn_access="arrhenius",
                rxn_type="heavy_particle_dissociation",
                rxn_index=r
            )
            num_p = self._reaction_getter_by_type(
                rxn_access="partners",
                rxn_type="heavy_particle_dissociation",
                rxn_index=r
            )
            partners = [p - 1 for p in num_p]
            self._reactions.append({
                "type": "dh_cat_mt",
                "plato_index": r,
                "reactant_species": [i - 1],
                "product_species": [j - 1, k - 1],
                "partners": partners,
                "arrhenius": (a, b, c),
                "reversible": True,
                "temperature": _rxn_type_temp["dh_cat_mt"],
            })

        num_rxn = self._num_reactions_by_type(
            rxn_type="exchange",
        )
        for r in range(1, num_rxn + 1):
            i, j, k, ell = self._reaction_getter_by_type(
                rxn_access="species",
                rxn_type="exchange",
                rxn_index=r
            )
            a, b, c = self._reaction_getter_by_type(
                rxn_access="arrhenius",
                rxn_type="exchange",
                rxn_index=r
            )
            self._reactions.append({
                "type": "exc",
                "plato_index": r,
                "reactant_species": [i - 1, j - 1],
                "product_species": [k - 1, ell - 1],
                "partners": [],
                "arrhenius": (a, b, c),
                "reversible": True,
                "temperature": _rxn_type_temp["exc"],
            })

        num_rxn = self._num_reactions_by_type(
            rxn_type="electron_impact_dissociation"
        )
        for r in range(1, num_rxn + 1):
            i, j, k = self._reaction_getter_by_type(
                rxn_access="species",
                rxn_type="electron_impact_dissociation",
                rxn_index=r
            )
            a, b, c = self._reaction_getter_by_type(
                rxn_access="arrhenius",
                rxn_type="electron_impact_dissociation",
                rxn_index=r,
            )
            self._reactions.append({
                "type": "de",
                "plato_index": r,
                "reactant_species": [i - 1],
                "product_species": [j - 1, k - 1],
                "partners": [],
                "arrhenius": (a, b, c),
                "reversible": True,
                "temperature": _rxn_type_temp["de"],
            })

        num_rxn = self._num_reactions_by_type(
            rxn_type="electron_impact_ionization"
        )
        for r in range(1, num_rxn + 1):
            i, j = self._reaction_getter_by_type(
                rxn_access="species",
                rxn_type="electron_impact_ionization",
                rxn_index=r
            )
            a, b, c = self._reaction_getter_by_type(
                rxn_access="arrhenius",
                rxn_type="electron_impact_ionization",
                rxn_index=r
            )
            self._reactions.append({
                "type": "ie",
                "plato_index": r,
                "reactant_species": [i - 1],
                "product_species": [j - 1],
                "partners": [],
                "arrhenius": (a, b, c),
                "reversible": True,
                "temperature": _rxn_type_temp["ie"],
            })

        num_rxn = self._num_reactions_by_type(
            rxn_type="associative_ionization",
        )
        for r in range(1, num_rxn + 1):
            i, j, k = self._reaction_getter_by_type(
                rxn_access="species",
                rxn_type="associative_ionization",
                rxn_index=r
            )
            a, b, c = self._reaction_getter_by_type(
                rxn_access="arrhenius",
                rxn_type="associative_ionization",
                rxn_index=r
            )
            self._reactions.append({
                "type": "ai",
                "plato_index": r,
                "reactant_species": [i - 1, j - 1],
                "product_species": [k - 1],
                "partners": [],
                "arrhenius": (a, b, c),
                "reversible": True,
                "temperature": _rxn_type_temp["ai"],
            })

    # }}}

    # {{{ Abstract interface

    @property
    def num_temp(self):
        """Return number of temperatures."""
        return self.namespace.n_temp

    @property
    def num_species(self):
        """Return number of species."""
        return self.namespace.n_species

    @property
    def num_reactions(self):
        """Return number of reactions."""
        return len(self._reactions)

    @property
    def num_vt_molecules(self):
        """Return number of VT-active molecules."""
        return self.namespace.n_vt_molecules

    @property
    def molecular_weights(self):
        """Return species molecular weights in kg/kmol."""
        return np.array([1e3 * m for m in self.namespace.molar_masses])

    @property
    def translational_rotational_specific_heat_cv(self):
        """Return per-species translational-rotational Cv [J/(kg K)]:
        equipartition theorem, 3/2 R for atoms (0 rotational degrees of
        freedom) or 5/2 R for linear molecules (2 rotational degrees of
        freedom, inferred from whether the species has vibrational modes).
        """
        specific_gas_constant = (
            self.namespace.gas_constant
            / self.molecular_weights
        )
        rotational_degrees_of_freedom = np.array([
            2 if len(self.species_vibrational_temperature(species_index))
            else 0
            for species_index in range(self.num_species)
        ])
        return (
            0.5
            * (3 + rotational_degrees_of_freedom)
            * specific_gas_constant
        )

    @property
    def translational_rotational_specific_heat_cp(self):
        """Return per-species translational-rotational Cp [J/(kg K)]."""
        specific_gas_constant = (
            self.namespace.gas_constant
            / self.molecular_weights
        )
        return (
            self.translational_rotational_specific_heat_cv
            + specific_gas_constant
        )

    @property
    def standard_enthalpy_of_formation(self):
        """Return per-species standard enthalpy of formation [J/kg] at the
        standard reference temperature.
        """
        specific_gas_constant = (
            self.namespace.gas_constant
            / self.molecular_weights
        )
        return (
            self.standard_energy_of_formation
            + specific_gas_constant * standard_reference_temperature
        )

    @property
    def standard_energy_of_formation(self):
        """Return per-species standard energy of formation [J/kg] at the
        standard reference temperature.
        """
        # Plato's tabulated value, not the NASA9 fit at 298.15 K: the two
        # differ by up to ~200 J/kg, and Plato splits energy with the former.
        return np.array(self.namespace.__getattr__("formation_energies"))

    @property
    def species_names(self):
        """Return species names."""
        return self.namespace.species_names

    @property
    def reactions(self):
        """Return all reactions in the mechanism as a `typing:List`."""
        return self._reactions

    def species_vibrational_temperature(self, species_index) -> np.ndarray:
        # theta_vib expects a 1-based component index but species_index is
        # zero-based
        return self.namespace.__getattr__("theta_vib", species_index + 1)

    def _nasa_polynomial_interval_bounds(self, species_index):
        """Return NASA poly interval temperature bounds."""
        return self.namespace.__getattr__("nasa_temp_bounds", species_index)

    def _nasa_polynomial_coefficients(self, species_index):
        """Return NASA poly coefficients."""
        return self.namespace.__getattr__("nasa_cp_coeffs", species_index)

    def _nasa_polynomial_offsets(self, species_index, interval_index):
        """Return NASA poly offsets for enthalpy and entropy."""
        return self.namespace.__getattr__(
            "nasa_b_consts", species_index, interval_index
        )

    def nasa_polynomial_parameterization(self, species_index):
        t_bounds = self._nasa_polynomial_interval_bounds(species_index)
        cp_coeffs = self._nasa_polynomial_coefficients(species_index)
        num_intervals = len(cp_coeffs)
        _coeffs_list = []
        for interval in range(num_intervals):
            b1, b2 = self._nasa_polynomial_offsets(species_index, interval + 1)
            _coeffs_list.append(
                np.append(cp_coeffs[interval], [b1, b2])
            )
        coeffs = np.column_stack(_coeffs_list)
        num_coeff, num_intervals = coeffs.shape
        return t_bounds, coeffs

    def species_thermo_params(self, species_index) -> PolynomialParameters:
        t_bounds, coeffs = self.nasa_polynomial_parameterization(
            species_index + 1
        )
        num_coeff, num_intervals = coeffs.shape
        # Plato's interval search (nasa_thermo_utils.F90) assigns T equal
        # to a break to the upper interval.
        return PolynomialParameters(
            num_intervals=num_intervals,
            num_coeff=num_coeff,
            t_bounds=t_bounds,
            coeffs=coeffs,
            break_comparison=">=",
        )

    def reaction(self, reaction_index: int) -> Dict:
        return self._reactions[reaction_index]

    def is_reversible(self, reaction_index: int) -> bool:
        return self._reactions[reaction_index]["reversible"]

    def species_name(self, species_index: int) -> str:
        return self.species_names[species_index]

    def species_index(self, species_name: str) -> int:
        return self.species_names.index(species_name)

    def reactants(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index)["reactant_species"]

    def products(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index)["product_species"]

    def stoichiometric_coefficients(
            self, reaction_index: int) -> Tuple[List[int], List[int]]:
        rxn = self.reaction(reaction_index)
        return (
            [1] * len(rxn["reactant_species"]),
            [1] * len(rxn["product_species"]),
        )

    def participation_set(
            self, species_id: Union[int, str]) -> Tuple[List[int], List[int]]:
        if isinstance(species_id, str):
            sp_idx = self.species_index(species_id)
        else:
            sp_idx = species_id

        fwd_set = [r for r, rxn in enumerate(self.reactions)
                   if sp_idx in rxn["reactant_species"]]
        rev_set = [r for r, rxn in enumerate(self.reactions)
                   if sp_idx in rxn["product_species"]]
        return fwd_set, rev_set

    def production_balance(
            self, species_index: int
    ) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        fwd_set, rev_set = self.participation_set(species_index)
        # Count actual occurrences: homoatomic dissociation (e.g. N2->N+N)
        # gives stoich 2
        stoich_fwd = [
            self._reactions[r]["reactant_species"].count(species_index)
            for r in fwd_set
        ]
        stoich_rev = [
            self._reactions[r]["product_species"].count(species_index)
            for r in rev_set
        ]
        return (fwd_set, rev_set), (stoich_fwd, stoich_rev)

    # }}}

    # {{{ Make methods

    def make_rate_coefficient(self,
                              reaction_index,
                              hardcode_params,
                              temperature=None) -> RateCoefficient:
        """:returns: the forward Arrhenius coefficient and its parameters,
            evaluated at the reaction type's controlling temperature
            unless *temperature* overrides it.
        """
        a, b, c = self.reaction(reaction_index)["arrhenius"]
        temp_type = self.reaction(reaction_index)["temperature"]

        if temperature is not None:
            temp_var = temperature
        elif self.num_temp == 1:
            temp_var = Variable("temperature")
        elif temp_type == "translational":
            temp_var = _temp_map[temp_type][0]
        else:
            temp_var = _temp_map[temp_type]

        _mol_to_kmol = 1e3
        if hardcode_params:
            params = {
                "a": np.log(a * self.namespace.avogadro * _mol_to_kmol),
                "b": b,
                "t_a": c
            }
            k_fwd = make_arrhenius(
                reaction_index=reaction_index,
                params=params,
                temperature=temp_var
            )
        else:
            params = np.array([np.log(a), b, c])
            k_fwd = make_arrhenius(
                reaction_index=reaction_index,
                temperature=temp_var
            )

        return k_fwd, params

    def make_mass_action_rate(self, reaction_index):
        rxn = self.reaction(reaction_index)
        rxn_type = rxn["type"]
        reac = rxn["reactant_species"]
        prod = rxn["product_species"]
        stoich_reac, stoich_prod = self.stoichiometric_coefficients(
            reaction_index
        )

        if rxn_type == "dh_cat_mt":
            # Third-body: sum over collision-partner concentrations
            m_sum = sum(conc[m] for m in rxn["partners"])
            r_fwd = k_fwd[reaction_index] * conc[reac[0]]
            r_rev = (
                exp(log_k_eq[reaction_index])
                * self._backward_dissociation_base_rate(reaction_index)
                * conc[prod[0]] * conc[prod[1]]
            )
            return m_sum * (r_fwd - r_rev)
        else:
            return reaction_progress_rate_expr(
                reaction_index,
                rxn["reversible"],
                (reac, prod),
                (stoich_reac, stoich_prod),
            )

    def _backward_dissociation_base_rate(self, reaction_index):
        """:returns: the rate coefficient that the equilibrium constant
            divides to give the recombination rate.
        """
        if self.num_temp == 1:
            return k_fwd[reaction_index]
        # Park's model drives only dissociation by sqrt(Th Tv);
        # recombination stays at Th (kinetics_rate.F90, Dh_cat_mt).
        k_translational, _ = self.make_rate_coefficient(
            reaction_index, self.hardcode_params,
            temperature=_temp_map["translational"][0]
        )
        return k_translational.expr

    def make_species_production_rate(self, species_index):
        part_sets, stoich_coeffs = self.production_balance(species_index)
        return species_production_rate_expr(
            species_index,
            part_sets[0],
            part_sets[1],
            stoich_coeffs[0],
            stoich_coeffs[1],
        )

    def make_species_nasa_thermo(self, species_index) -> SpeciesNASAThermo:
        sp_name = self.species_names[species_index]
        poly_params = self.species_thermo_params(species_index)
        t_var = (
            Variable("temperature") if self.num_temp == 1 else
            (
                Variable("temperature")[-1] if sp_name == 'em' else
                Variable("temperature")[0]
            )
        )
        return make_species_nasa_thermo(
            poly_params,
            t_var
        )

    def make_species_vibrational_thermo(
            self, species_index
    ) -> SpeciesVibrationalThermo:
        vib_temp = self.species_vibrational_temperature(species_index)
        return make_species_vibrational_thermo(
            (self.namespace.gas_constant
             / self.molecular_weights[species_index]),
            vib_temp
        )

    def make_vt_relaxation_time_exprs(self, vt_molecule_index):
        heavy_temperature = Variable("temperature")[0]
        millikan_white_a = self.namespace.vt_mw_a()[vt_molecule_index, :]
        millikan_white_b = self.namespace.vt_mw_b()[vt_molecule_index, :]
        park_cross_section = self.namespace.vt_park_sigma()[vt_molecule_index]
        reduced_molar_mass_sqrt = (
            self.namespace.vt_sqrmu()[vt_molecule_index, :]
        )
        return [
            pairwise_relaxation_time_expr(
                millikan_white_a[heavy_partner_index],
                millikan_white_b[heavy_partner_index],
                park_cross_section,
                reduced_molar_mass_sqrt[heavy_partner_index],
                self.namespace.one_atm,
                heavy_temperature,
            )
            for heavy_partner_index in range(self.namespace.n_heavy)
        ]

    def make_vt_energy_transfer_expr(self,
                                     vt_molecule_index) -> p.ExpressionNode:
        from pymbolic import substitute
        mass_fractions = Variable("mass_fractions")
        heavy_species_indices = [
            self.namespace.n_e + heavy_partner_index
            for heavy_partner_index in range(self.namespace.n_heavy)
        ]
        heavy_partner_mole_ratios = [
            mass_fractions[heavy_species_index]
            / self.molecular_weights[heavy_species_index]
            for heavy_species_index in heavy_species_indices
        ]
        mean_relaxation_rate = vt_mean_relaxation_rate_expr(
            Variable("pressure"),
            heavy_partner_mole_ratios,
            self.make_vt_relaxation_time_exprs(vt_molecule_index)
        )

        species_index = (
            int(self.namespace.vt_molecule_ids()[vt_molecule_index])
            - 1
        )
        vibrational_thermo = self.species_vib_thermo_expressions[species_index]
        vibrational_energy_at_vib_temperature = vibrational_thermo.energy_expr
        vibrational_energy_at_heavy_temperature = substitute(
            vibrational_thermo.energy_expr,
            {Variable("temperature")[1]: Variable("temperature")[0]}
        )

        return vt_energy_transfer_expr(
            Variable("density"),
            mass_fractions[species_index],
            vibrational_energy_at_heavy_temperature,
            vibrational_energy_at_vib_temperature,
            mean_relaxation_rate,
        )

    def make_translational_rotational_energy_expr(
            self,
            species_index) -> p.ExpressionNode:
        heavy_temperature = Variable("temperature")[0]
        return translational_rotational_energy_expr(
            self.translational_rotational_specific_heat_cv[species_index],
            standard_reference_temperature,
            self.standard_energy_of_formation[species_index],
            heavy_temperature,
        )

    def make_nasa_polynomial_vibrational_energy_expr(
            self,
            species_index) -> p.ExpressionNode:
        from pymbolic import substitute
        vibrational_temperature = Variable("temperature")[1]
        specific_gas_constant = (
            self.namespace.gas_constant / self.molecular_weights[species_index]
        )
        nasa_poly = self.species_nasa_thermo_polynomials
        enthalpy_rt_at_vibrational_temperature = substitute(
            nasa_poly[species_index].enthalpy_poly.expr,
            {Variable("temperature")[0]: vibrational_temperature}
        )
        return nasa_polynomial_vibrational_energy_expr(
            enthalpy_rt_at_vibrational_temperature,
            specific_gas_constant,
            self.translational_rotational_specific_heat_cp[species_index],
            standard_reference_temperature,
            self.standard_enthalpy_of_formation[species_index],
            vibrational_temperature,
        )

    def make_nasa_polynomial_vibrational_specific_heat_expr(
            self,
            species_index) -> p.ExpressionNode:
        from pymbolic import substitute
        vibrational_temperature = Variable("temperature")[1]
        specific_gas_constant = (
            self.namespace.gas_constant / self.molecular_weights[species_index]
        )
        cp_r_at_vibrational_temperature = substitute(
            self.species_nasa_thermo_polynomials[species_index].cp_poly.expr,
            {Variable("temperature")[0]: vibrational_temperature}
        )
        return nasa_polynomial_vibrational_specific_heat_expr(
            cp_r_at_vibrational_temperature,
            specific_gas_constant,
            self.translational_rotational_specific_heat_cp[species_index],
        )

    def make_equilibrium_constant(self, reaction_index):
        rxn = self.reaction(reaction_index)
        reac = rxn["reactant_species"]
        prod = rxn["product_species"]
        stoich_reac, stoich_prod = self.stoichiometric_coefficients(
            reaction_index
        )
        expr = equilibrium_constant_expr(
            reaction_index,
            (reac, prod),
            (stoich_reac, stoich_prod),
            self.namespace.ref_pressure,
            self.namespace.gas_constant
        )
        if self.num_temp > 1:
            from pymbolic import substitute
            expr = substitute(
                expr,
                {"temperature": Variable("temperature")[0]}
            )

        return expr

    # }}}
