import numpy as np
import mutationpp as mpp
from pymbolic.primitives import Variable
from typing import Dict, List, Tuple, Union
from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.thermo import (
    PolynomialParameters,
    SpeciesElectronicThermo,
    SpeciesNASAThermo,
    SpeciesVibrationalThermo,
    equilibrium_constant_expr,
    make_species_electronic_thermo,
    make_species_nasa_thermo,
    make_species_vibrational_thermo,
    nasa_polynomial_vibrational_energy_expr,
    nasa_polynomial_vibrational_specific_heat_expr,
    translational_rotational_energy_expr,
)
from pyrometheus.bandit.chem_expr.kinetics import (
    RateCoefficient,
    exp,
    k_fwd,
    log_k_eq,
    make_arrhenius,
    mass_action_rxn_progress_rate_expr,
    reaction_progress_rate_expr,
    species_production_rate_expr,
    third_body_concentration_expr,
)


# {{{ Temperature substitutions, keyed by the strings Mutation++ reports
# from Reaction::fwdRateTemperature (see kinetics/ReactionType.h)

_temp_map = {
    "translational": Variable("temperature")[0],
    "electron": Variable("temperature")[1],
    "geometric_ttv": Variable("sqrt")(
        Variable("temperature")[0]
        * Variable("temperature")[1]
    ),
}

# }}}


# Standard thermodynamic reference temperature [K], at which the
# vibronic part of the NASA energy split vanishes.
standard_reference_temperature = 298.15


# {{{ Unit conversion
#
# Mutation++ works in mol; bandit concentrations are kmol/m^3.

_kmol_per_mol = 1e3

# }}}


class Mutationpp(BaseNamespace):

    def __init__(self, mixture, state_model, thermo_database):
        options = mpp.MixtureOptions(mixture)
        options.setStateModel(state_model)
        options.setThermodynamicDatabase(thermo_database)
        self.mix = mpp.Mixture(options)
        # Taken from the library rather than hardcoded: Mutation++
        # carries an older CODATA gas constant, and a mismatch would
        # shift every equilibrium constant.
        self.gas_constant = _kmol_per_mol * self.mix.RU()
        self.one_atm = self.mix.ONEATM()

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.mix, name)(*args)
        else:
            return getattr(self.mix, name)


class MutationMechanism(BaseMechanism):

    def __init__(self,
                 mixture,
                 state_model="ChemNonEqTTv",
                 thermo_database="NASA-9",
                 pyro_np=np,
                 hardcode_params=True):
        self.hardcode_params = hardcode_params
        self.namespace = Mutationpp(mixture, state_model, thermo_database)
        self.nonequil_thermo = self.num_temp > 1
        self.make_rates(hardcode_params)
        self.make_thermo()

    # {{{ Abstract interface

    @property
    def num_temp(self):
        """Return number of temperatures."""
        return self.namespace.__getattr__("nEnergyEqns").__call__()

    @property
    def num_species(self):
        """Return number of species."""
        return self.namespace.__getattr__("nSpecies").__call__()

    @property
    def num_reactions(self):
        """Return number of reactions."""
        return self.namespace.__getattr__("nReactions").__call__()

    @property
    def molecular_weights(self):
        """Return species molecular weights in kg/kmol."""
        return _kmol_per_mol * np.array(
            self.namespace.__getattr__("speciesMw").__call__()
        )

    @property
    def species_names(self):
        """Return species names."""
        return [
            self.species_name(species_index)
            for species_index in range(self.num_species)
        ]

    def species_name(self, species_index: int) -> str:
        return self.namespace.__getattr__("speciesName", species_index)

    def species_index(self, species_name: str) -> int:
        return self.namespace.__getattr__("speciesIndex", species_name)

    def reactions(self) -> List:
        return [
            self.reaction(reaction_index)
            for reaction_index in range(self.num_reactions)
        ]

    def reaction(self, reaction_index: int):
        return self.namespace.__getattr__("reaction", reaction_index)

    def is_reversible(self, reaction_index: int) -> bool:
        return self.reaction(reaction_index).is_reversible

    def is_third_body(self, reaction_index: int) -> bool:
        return self.reaction(reaction_index).is_thirdbody

    def third_body_efficiencies(
            self, reaction_index: int) -> Dict[int, float]:
        """:returns: Collision efficiencies keyed by species index. Only
        species that Mutation++ lists explicitly appear, and the electron,
        which never acts as a third body (ThirdBodyManager.h); every
        other heavy species takes the default efficiency of one.
        """
        efficiencies = {
            species_index: float(efficiency)
            for species_index, efficiency
            in self.reaction(reaction_index).efficiencies
        }
        if self.namespace.__getattr__("hasElectrons").__call__():
            efficiencies[self.species_index("e-")] = 0.0
        return efficiencies

    def rate_temperature(self, reaction_index: int, direction: str):
        """:returns: The temperature the rate coefficient of the
        reaction with index *reaction_index* is evaluated at, as a
        :class:`pymbolic.primitives.ExpressionNode`. Dissociation runs
        forward on the geometric mean of the heavy and vibrational
        temperatures but reverse on the heavy temperature alone, so the
        two directions are asked for separately.
        """
        reaction = self.reaction(reaction_index)
        if direction not in ("fwd", "rev"):
            raise ValueError(f"unknown rate direction '{direction}'")
        # Every role collapses onto the one temperature, which generated
        # code takes as a scalar.
        if self.num_temp == 1:
            return Variable("temperature")
        if direction == "fwd":
            return _temp_map[reaction.fwd_rate_coeff_temperature]
        elif direction == "rev":
            return _temp_map[reaction.rev_rate_coeff_temperature]
        else:
            raise ValueError(f"unknown rate direction '{direction}'")

    def reactants(self, reaction_index: int) -> List[int]:
        return self._unique_indices(self.reaction(reaction_index).reactants)

    def products(self, reaction_index: int) -> List[int]:
        return self._unique_indices(self.reaction(reaction_index).products)

    def stoichiometric_coefficients(
            self, reaction_index: int) -> Tuple[List[int], List[int]]:
        reaction = self.reaction(reaction_index)
        return (
            self._count_indices(reaction.reactants),
            self._count_indices(reaction.products),
        )

    def participation_set(
            self, species_id: Union[int, str]) -> Tuple[List[int], List[int]]:
        if isinstance(species_id, str):
            species_index = self.species_index(species_id)
        else:
            species_index = species_id

        forward_set = [
            reaction_index for reaction_index in range(self.num_reactions)
            if species_index in self.reactants(reaction_index)
        ]
        reverse_set = [
            reaction_index for reaction_index in range(self.num_reactions)
            if species_index in self.products(reaction_index)
        ]
        return forward_set, reverse_set

    def production_balance(
            self, species_index: int
    ) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        forward_set, reverse_set = self.participation_set(species_index)
        stoich_forward = [
            self.reaction(reaction_index).reactants.count(species_index)
            for reaction_index in forward_set
        ]
        stoich_reverse = [
            self.reaction(reaction_index).products.count(species_index)
            for reaction_index in reverse_set
        ]
        return (forward_set, reverse_set), (stoich_forward, stoich_reverse)

    # }}}

    # {{{ Make methods

    def make_rate_coefficient(self,
                              reaction_index,
                              hardcode_params,
                              temperature=None) -> RateCoefficient:
        """:returns: The forward rate coefficient, with its third body
        folded in, and its parameters. It is evaluated at the reaction's
        forward rate temperature unless *temperature* overrides it.
        """
        if temperature is None:
            temperature = self.rate_temperature(reaction_index, "fwd")
        reaction = self.reaction(reaction_index)
        rate_law = reaction.rate_law()
        # Mutation++ reports ln(A) in mol-based SI units; a rate of
        # order p carries p-1 concentration factors to convert.
        log_pre_exponential = (
            rate_law.log_pre_exponential
            + (reaction.order - 1) * np.log(_kmol_per_mol)
        )
        if hardcode_params:
            params = {
                "a": log_pre_exponential,
                "b": rate_law.exponent,
                "t_a": rate_law.activation_temperature,
            }
            rate_coeff = make_arrhenius(
                reaction_index=reaction_index,
                params=params,
                temperature=temperature,
            )
        else:
            params = np.array([
                log_pre_exponential,
                rate_law.exponent,
                rate_law.activation_temperature,
            ])
            rate_coeff = make_arrhenius(
                reaction_index=reaction_index,
                temperature=temperature,
            )

        # Mutation++ keeps the third body out of the rate constant and
        # applies it to the rate of progress; folding it in here lets
        # generated code treat every reaction's coefficient uniformly.
        if self.is_third_body(reaction_index):
            rate_coeff.expr = (
                rate_coeff.expr
                * third_body_concentration_expr(
                    self.num_species,
                    self.third_body_efficiencies(reaction_index),
                )
            )
        return rate_coeff, params

    def species_translational_temperature(self, species_index: int):
        """:returns: The electron temperature for the electron, which
        Mutation++ also gives the electron's partial pressure; the
        heavy-particle temperature for every other species.
        """
        if (self.num_temp > 1
                and self.species_name(species_index) == "e-"):
            return _temp_map["electron"]
        return super().species_translational_temperature(species_index)

    def species_rrho(self, species_index: int):
        """:returns: The rigid-rotor harmonic-oscillator parameters of
        the species with index *species_index*.
        """
        return self.namespace.__getattr__("species_rrho", species_index)

    def specific_gas_constant(self, species_index: int) -> float:
        return (
            self.namespace.gas_constant
            / self.molecular_weights[species_index]
        )

    def make_species_vibrational_thermo(
            self, species_index) -> SpeciesVibrationalThermo:
        return make_species_vibrational_thermo(
            self.specific_gas_constant(species_index),
            np.array(self.species_rrho(species_index)
                     .vibrational_temperatures),
        )

    def make_species_electronic_thermo(
            self, species_index) -> SpeciesElectronicThermo:
        return make_species_electronic_thermo(
            self.specific_gas_constant(species_index),
            np.array(self.species_rrho(species_index).electronic_levels),
        )

    @property
    def thermo_temperature(self):
        """:returns: The temperature the standard-state thermodynamic
        properties are evaluated at. Mutation++ builds equilibrium
        constants from Gibbs energies with every temperature set equal,
        so this is the heavy-particle temperature.
        """
        if self.num_temp == 1:
            return Variable("temperature")
        return Variable("temperature")[0]

    def species_thermo_params(self, species_index) -> PolynomialParameters:
        polynomial = self.namespace.__getattr__(
            "species_nasa_polynomial", species_index
        )
        # The binding hands back one row per temperature interval,
        # lowest first; PolynomialParameters wants one column each.
        coeffs = np.array(polynomial["coefficients"]).T
        num_coeff, num_intervals = coeffs.shape
        return PolynomialParameters(
            num_intervals=num_intervals,
            num_coeff=num_coeff,
            t_bounds=np.array(polynomial["t_bounds"]),
            coeffs=coeffs,
        )

    def make_species_nasa_thermo(self, species_index) -> SpeciesNASAThermo:
        return make_species_nasa_thermo(
            self.species_thermo_params(species_index),
            self.thermo_temperature,
        )

    def make_equilibrium_constant(self, reaction_index):
        """:returns: The equilibrium constant that sets the reverse rate.
        Mutation++ evaluates it, with every species' Gibbs energy, at the
        reverse rate temperature (RateLawGroup.h), which for reactions
        that reverse on the electron temperature is not the heavy one.
        """
        from pymbolic import substitute
        expr = equilibrium_constant_expr(
            reaction_index,
            (self.reactants(reaction_index), self.products(reaction_index)),
            self.stoichiometric_coefficients(reaction_index),
            self.namespace.one_atm,
            self.namespace.gas_constant,
        )
        if self.num_temp == 1:
            return expr
        temperature = self.rate_temperature(reaction_index, "rev")
        expr = substitute(expr, {Variable("temperature"): temperature})
        if temperature != self.thermo_temperature:
            # gibbs_rt holds Gibbs energies at the heavy temperature.
            gibbs_rt = Variable("gibbs_rt")
            expr = substitute(expr, {
                gibbs_rt[species_index]: make_species_nasa_thermo(
                    self.species_thermo_params(species_index), temperature
                ).gibbs_poly.expr
                for species_index in range(self.num_species)
            })
        return expr

    def make_mass_action_rate(self, reaction_index):
        indices = (self.reactants(reaction_index),
                   self.products(reaction_index))
        stoich_coeffs = self.stoichiometric_coefficients(reaction_index)
        reverse_rate_coeff = self._reverse_base_rate(reaction_index)
        if reverse_rate_coeff is None:
            return reaction_progress_rate_expr(
                reaction_index,
                self.is_reversible(reaction_index),
                indices,
                stoich_coeffs,
            )
        return (
            k_fwd[reaction_index]
            * mass_action_rxn_progress_rate_expr(
                reaction_index, indices[0], stoich_coeffs[0])
            - exp(log_k_eq[reaction_index]) * reverse_rate_coeff
            * mass_action_rxn_progress_rate_expr(
                reaction_index, indices[1], stoich_coeffs[1])
        )

    def _reverse_base_rate(self, reaction_index):
        """:returns: The rate coefficient that the equilibrium constant
        scales into the reverse rate coefficient, when it differs from
        ``k_fwd``; otherwise *None*. Dissociation runs forward on
        sqrt(Th Tv) but recombines on Th (ReactionType.h), so its
        reverse rate needs the forward Arrhenius law evaluated at Th.
        """
        if not self.is_reversible(reaction_index):
            return None
        reverse_temperature = self.rate_temperature(reaction_index, "rev")
        if reverse_temperature == self.rate_temperature(reaction_index, "fwd"):
            return None
        rate_coeff, _ = self.make_rate_coefficient(
            reaction_index, self.hardcode_params,
            temperature=reverse_temperature,
        )
        return rate_coeff.expr

    def make_species_production_rate(self, species_index):
        part_sets, stoich_coeffs = self.production_balance(species_index)
        return species_production_rate_expr(
            species_index,
            part_sets[0], part_sets[1],
            stoich_coeffs[0], stoich_coeffs[1],
        )

    # }}}

    # {{{ Two-temperature energy exchange
    #
    # Mutation++'s OmegaVT (transfer/OmegaVT.cpp, MillikanWhite.cpp):
    # Landau-Teller relaxation of harmonic-oscillator energies, with a
    # Millikan-White time averaged over heavy partners plus Park's
    # high-temperature correction. The energy equations split the NASA
    # energy into a translational-rotational part at Th and a vibronic
    # remainder at Tv, as the Plato adapter does.

    def _has_electrons(self):
        return self.namespace.__getattr__("hasElectrons").__call__()

    @property
    def _heavy_species_indices(self):
        """:returns: Every species but the electron, which Mutation++
        lists first: the Millikan-White collision partners.
        """
        return list(range(int(self._has_electrons()), self.num_species))

    @property
    def _vt_species_indices(self):
        return [
            species_index for species_index in range(self.num_species)
            if len(self.species_rrho(species_index).vibrational_temperatures)
        ]

    @property
    def num_vt_molecules(self):
        """Return number of VT-active molecules."""
        return len(self._vt_species_indices)

    def _millikan_white_data(self, vt_molecule_index):
        return self.namespace.__getattr__(
            "millikan_white_data", self._vt_species_indices[vt_molecule_index]
        )

    def make_vt_relaxation_time_exprs(self, vt_molecule_index):
        """:returns: The Millikan-White relaxation time, times pressure
        [Pa s], of the VT-active molecule with index *vt_molecule_index*
        against each heavy partner.
        """
        data = self._millikan_white_data(vt_molecule_index)
        temperature = self.heavy_temperature
        return [
            exp(a * (temperature ** (-1 / 3) - b) - 18.42)
            * self.namespace.one_atm
            for a, b in zip(data.a, data.b)
        ]

    def make_vt_energy_transfer_expr(self, vt_molecule_index):
        from pymbolic import substitute
        data = self._millikan_white_data(vt_molecule_index)
        species_index = self._vt_species_indices[vt_molecule_index]
        temperature = self.heavy_temperature
        density = Variable("density")
        mass_fractions = Variable("mass_fractions")
        partner_moles = [
            mass_fractions[partner_index]
            / self.molecular_weights[partner_index]
            for partner_index in self._heavy_species_indices
        ]
        # Mutation++ averages the times, not the rates, over partners.
        millikan_white_time = sum(
            moles * pressure_time for moles, pressure_time in zip(
                partner_moles,
                self.make_vt_relaxation_time_exprs(vt_molecule_index))
        ) / (sum(partner_moles) * Variable("pressure"))

        molar_mass = data.molecular_weight
        number_density = (
            density * mass_fractions[species_index] / molar_mass
            * self.namespace.__getattr__("NA").__call__()
        )
        mean_speed = Variable("sqrt")(
            8 * self.namespace.__getattr__("RU").__call__() * temperature
            / (np.pi * molar_mass)
        )
        cross_section = (
            data.reference_cross_section * 50000.0 ** 2 / temperature ** 2
        )
        # Park's time is 1 / (n c sigma), infinite for an absent molecule;
        # multiplying through by that collision frequency keeps the source
        # and its derivatives finite there.
        collision_frequency = number_density * mean_speed * cross_section

        vibrational_energy = (
            self.species_vib_thermo_expressions[species_index].energy_expr
        )
        vibrational_energy_at_heavy_temperature = substitute(
            vibrational_energy,
            {Variable("temperature")[1]: Variable("temperature")[0]}
        )
        return (
            density * mass_fractions[species_index]
            * (vibrational_energy_at_heavy_temperature - vibrational_energy)
            * collision_frequency
            / (millikan_white_time * collision_frequency + 1)
        )

    @property
    def _specific_gas_constants(self):
        return self.namespace.gas_constant / self.molecular_weights

    @property
    def translational_rotational_specific_heat_cv(self):
        """Return per-species translational-rotational Cv [J/(kg K)]:
        3/2 R for atoms and electrons, 5/2 R for the linear molecules
        that carry a vibrational temperature.
        """
        rotational_degrees_of_freedom = np.array([
            2 if species_index in self._vt_species_indices else 0
            for species_index in range(self.num_species)
        ])
        return (
            0.5 * (3 + rotational_degrees_of_freedom)
            * self._specific_gas_constants
        )

    @property
    def translational_rotational_specific_heat_cp(self):
        """Return per-species translational-rotational Cp [J/(kg K)]."""
        return (
            self.translational_rotational_specific_heat_cv
            + self._specific_gas_constants
        )

    @property
    def standard_enthalpy_of_formation(self):
        """Return per-species NASA enthalpy [J/kg] at the standard
        reference temperature.
        """
        from pymbolic import evaluate
        temperature = (
            standard_reference_temperature if self.num_temp == 1
            else np.full(self.num_temp, standard_reference_temperature)
        )
        context = {"temperature": temperature, "log": np.log, "exp": np.exp}
        return np.array([
            evaluate(thermo.enthalpy_poly.expr, context)
            for thermo in self.species_nasa_thermo_polynomials
        ]) * self._specific_gas_constants * standard_reference_temperature

    @property
    def standard_energy_of_formation(self):
        """Return per-species NASA energy [J/kg] at the standard
        reference temperature.
        """
        return (
            self.standard_enthalpy_of_formation
            - self._specific_gas_constants * standard_reference_temperature
        )

    def make_translational_rotational_energy_expr(self, species_index):
        return translational_rotational_energy_expr(
            self.translational_rotational_specific_heat_cv[species_index],
            standard_reference_temperature,
            self.standard_energy_of_formation[species_index],
            self.species_translational_temperature(species_index),
        )

    def make_nasa_polynomial_vibrational_energy_expr(self, species_index):
        from pymbolic import substitute
        vibrational_temperature = Variable("temperature")[1]
        enthalpy_rt = substitute(
            self.species_nasa_thermo_polynomials[
                species_index].enthalpy_poly.expr,
            {self.thermo_temperature: vibrational_temperature}
        )
        return nasa_polynomial_vibrational_energy_expr(
            enthalpy_rt,
            self._specific_gas_constants[species_index],
            self.translational_rotational_specific_heat_cp[species_index],
            standard_reference_temperature,
            self.standard_enthalpy_of_formation[species_index],
            vibrational_temperature,
        )

    def make_nasa_polynomial_vibrational_specific_heat_expr(
            self, species_index):
        from pymbolic import substitute
        cp_r = substitute(
            self.species_nasa_thermo_polynomials[species_index].cp_poly.expr,
            {self.thermo_temperature: Variable("temperature")[1]}
        )
        return nasa_polynomial_vibrational_specific_heat_expr(
            cp_r,
            self._specific_gas_constants[species_index],
            self.translational_rotational_specific_heat_cp[species_index],
        )

    # }}}

    # {{{ Stoichiometry helpers
    #
    # Mutation++ stores stoichiometry as a multiset of species indices,
    # repeating an index once per unit coefficient, so 2N is [i, i].

    @staticmethod
    def _unique_indices(indices: List[int]) -> List[int]:
        return list(dict.fromkeys(indices))

    @staticmethod
    def _count_indices(indices: List[int]) -> List[int]:
        return [
            indices.count(species_index)
            for species_index in dict.fromkeys(indices)
        ]

    # }}}
