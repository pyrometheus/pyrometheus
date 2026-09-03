from __future__ import annotations
import numpy as np
import pymbolic.primitives as p
from dataclasses import field
from typing import List, Union, Tuple, ForwardRef
from pyrometheus.bandit.chem_expr.kinetics import RateCoefficient
from pyrometheus.bandit.chem_expr.thermo import (SpeciesNASAThermo,
                                                 SpeciesVibrationalThermo)


class BaseNamespace:
    pass


class BaseMechanism:
    """
    .. attribute:: mass_action_rates
    .. attribute:: species_prod_rates

    .. automethod:: reactant_indices
    .. automethod:: product_indices
    .. automethod:: stoichiometric_coefficients
    .. automethod:: participation_set
    .. automethod:: make_mass_action_rate
    .. automethod:: make_rates
    """

    nonequil_thermo: bool = False
    pyro_generated: bool = False
    pyro_compiled: bool = False
    pyro_code: str = None
    pyro_engine: ForwardRef('Thermochemistry') = None  # noqa: F821
    rate_coeffs: np.ndarray = np.empty(shape=(0,), dtype=RateCoefficient)
    equil_constants: np.ndarray = np.empty(shape=(0,), dtype=p.ExpressionNode)
    mass_action_rates: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    species_prod_rates: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    species_nasa_thermo_polynomials: np.ndarray = np.empty(
        shape=(0,), dtype=SpeciesNASAThermo
    )
    species_vib_thermo_expressions: np.ndarray = np.empty(
        shape=(0,), dtype=SpeciesVibrationalThermo
    )
    pressure_relaxation_time_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    vt_energy_transfer_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    translational_rotational_energy_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    nasa_polynomial_vibrational_energy_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    nasa_polynomial_vibrational_specific_heat_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=p.ExpressionNode
    )
    species_production_rate_jacobian_exprs: np.ndarray = np.empty(
        shape=(0,), dtype=object
    )
    param_vals: np.ndarray = np.empty(shape=(0,), dtype=np.float64)

    # Composing the full chemistry graph and differentiating it (see
    # make_species_production_rate_jacobian) gets slower than the staged
    # primal evaluation as a mechanism grows; this is only a hint used to
    # warn callers, not a hard limit.
    _jac_warning_threshold = 50

    def __init__(self):
        pass

    @property
    def num_species(self):
        """
        :returns: The number of species in the mechanism.
        """
        raise NotImplementedError

    @property
    def num_reactions(self):
        """
        :returns: The number of reactions in the mechanism.
        """
        raise NotImplementedError

    @property
    def num_temp(self):
        """
        :returns: The number of temperatures carried by the mechanism (1
            for single-temperature thermochemistry, >1 for multi-
            temperature/nonequilibrium mechanisms).
        """
        raise NotImplementedError

    @property
    def molecular_weights(self):
        """
        :returns: The molecular weights for the species in the mechanism.
        """
        raise NotImplementedError

    def has_nonequilibrium_energy_modes(self) -> bool:
        return self.nonequil_thermo

    @property
    def num_vt_molecules(self):
        """
        :returns: The number of VT-active molecules in the mechanism.
        """
        raise NotImplementedError

    def reactant_indices(self, reaction_index: int):
        """:returns: The indices for reactants in reaction with index
        *reaction_index*.

        """
        raise NotImplementedError

    def product_indices(self, reaction_index: int):
        raise NotImplementedError

    def stoichiometric_coefficients(self, reaction_index: int):
        """:returns: A list of stoichiometric coefficients for the
        reaction with index *reaction_index*.
        """
        raise NotImplementedError

    def reaction_orders(self, reaction_index: int):
        """:returns: The concentration exponents of the forward rate of
        the reaction with index *reaction_index*. Defaults to the
        reactant stoichiometric coefficients; libraries that let a
        mechanism declare orders independently of stoichiometry
        override this.
        """
        return self.stoichiometric_coefficients(reaction_index)[0]

    def participation_set(self,
                          species_id: Union[int, str]) -> Tuple[List[int]]:
        """:return: A tuple of lists of indices for the reactions in
        which the species with ID *species_id* participates. The first
        list in the tuple corresponds to reactions where *specie_id*
        appears as a reactant, and the second list to reactions where
        it appears as a product.

        """
        raise NotImplementedError

    def make_rate_coefficient(self,
                              reaction_index,
                              hardcode_params) -> RateCoefficient:
        """:return: A rate coefficient expression as a
        :class:`chem_expr.kinetics.RateCoefficient`.
        """
        raise NotImplementedError

    def make_species_nasa_thermo(self, species_index) -> SpeciesNASAThermo:
        """:return: NASA polynomial expressions as a
        "class:`chem_expr.thermo.SpeciesNASAThermo`
        """
        raise NotImplementedError

    def make_species_vibrational_thermo(self, species_index) -> SpeciesVibrationalThermo:
        """:return: Harmonic-oscillator expressions as a
        "class:`chem_expr.thermo.SpeciesVibrationalThermo`
        """
        raise NotImplementedError

    def make_vt_relaxation_time_exprs(self, vt_molecule_index):
        """:return: A list of pairwise VT relaxation-time expressions
        (one per heavy collision partner) for the VT-active molecule with
        index *vt_molecule_index*, as
        :class:`pymbolic.primitives.ExpressionNode`.
        """
        raise NotImplementedError

    def make_vt_energy_transfer_expr(self, vt_molecule_index) -> p.ExpressionNode:
        """:return: The Landau-Teller energy-transfer contribution of the
        VT-active molecule with index *vt_molecule_index* to Omega_VT, as a
        :class:`pymbolic.primitives.ExpressionNode`.
        """
        raise NotImplementedError

    @property
    def translational_rotational_specific_heat_cv(self):
        """
        :returns: Per-species translational-rotational Cv [J/(kg K)]
        (equipartition theorem: 3/2 R for atoms, 5/2 R for linear
        molecules).
        """
        raise NotImplementedError

    @property
    def translational_rotational_specific_heat_cp(self):
        """
        :returns: Per-species translational-rotational Cp [J/(kg K)].
        """
        raise NotImplementedError

    @property
    def standard_enthalpy_of_formation(self):
        """
        :returns: Per-species standard enthalpy of formation [J/kg]: the
        NASA9 enthalpy evaluated at the standard reference temperature.
        """
        raise NotImplementedError

    @property
    def standard_energy_of_formation(self):
        """
        :returns: Per-species standard energy of formation [J/kg].
        """
        raise NotImplementedError

    def make_translational_rotational_energy_expr(self, species_index) -> p.ExpressionNode:
        """:return: The translational-rotational internal energy
        expression for species with index *species_index*, as a
        :class:`pymbolic.primitives.ExpressionNode`.
        """
        raise NotImplementedError

    def make_nasa_polynomial_vibrational_energy_expr(self, species_index) -> p.ExpressionNode:
        """:return: The NASA-polynomial-based vibronic energy expression
        for species with index *species_index*, as a
        :class:`pymbolic.primitives.ExpressionNode`.
        """
        raise NotImplementedError

    def make_nasa_polynomial_vibrational_specific_heat_expr(self, species_index) -> p.ExpressionNode:
        """:return: The NASA-polynomial-based vibronic specific heat
        expression for species with index *species_index*, as a
        :class:`pymbolic.primitives.ExpressionNode`.
        """
        raise NotImplementedError


    def make_mass_action_rate(self, reaction_index, hardcode_params=True):
        """
        :returns: mass action rate for *reaction_index* as a
        :class:`pymbolic.primitives.Expression`. The resulting
        symbolic rate is a product between a
        :class:`rate_coeff.RateCoefficient` and a
        :class:`pymbolic.primitives.Product` of reactant
        concentrations.
        """
        conc = p.Variable('concentrations')
        indices = self.reactants(reaction_index)
        stoich = self.stoichiometric_coefficients(reaction_index)
        rate_coeff, param_vals = self.make_rate_coefficient(
            reaction_index, hardcode_params
        )
        if isinstance(rate_coeff, RateCoefficient):
            if not hardcode_params:
                self.param_vals = np.vstack(
                    (self.param_vals, param_vals)
                ) if self.param_vals.size else param_vals

            return rate_coeff.expr * np.prod([
                conc[i] ** nu for i, nu in zip(indices, stoich)
            ])
        else:
            return 0

    def make_rates(self, hardcode_params=True):
        """Loop over reactions to create their corresponding mass
        action rate expression by invoking
        :class:`BaseMechanism.make_mass_action_rate`.
        """
        assert not self.rate_coeffs.size
        assert not self.mass_action_rates.size
        for irxn in range(self.num_reactions):
            self.rate_coeffs = np.append(
                self.rate_coeffs,
                self.make_rate_coefficient(irxn, hardcode_params)[0]
            )
            self.mass_action_rates = np.append(
                self.mass_action_rates,
                self.make_mass_action_rate(irxn)
            )

        assert not self.species_prod_rates.size
        for isp in range(self.num_species):
            self.species_prod_rates = np.append(
                self.species_prod_rates,
                self.make_species_production_rate(isp,)
            )

    def make_thermo(self,):
        # Make NASA thermo first
        assert not self.species_nasa_thermo_polynomials.size
        for isp in range(self.num_species):
            self.species_nasa_thermo_polynomials = np.append(
                self.species_nasa_thermo_polynomials,
                self.make_species_nasa_thermo(isp)
            )

        # Make equilibrium constnats
        assert not self.equil_constants.size
        for irxn in range(self.num_reactions):
            self.equil_constants = np.append(
                self.equil_constants,
                self.make_equilibrium_constant(irxn)
            )

        # Now check for vibrational nonequlibrium
        if self.has_nonequilibrium_energy_modes():
            for isp in range(self.num_species):
                self.species_vib_thermo_expressions = np.append(
                    self.species_vib_thermo_expressions,
                    self.make_species_vibrational_thermo(isp)
                )
            self.make_vt_transfer()
            self.make_nasa_polynomial_vibrational_thermo()

    def make_nasa_polynomial_vibrational_thermo(self):
        """Loop over species to build their translational-rotational and
        NASA-polynomial-based vibronic energy/specific-heat expressions by
        invoking :class:`BaseMechanism.make_translational_rotational_energy_expr`,
        :class:`BaseMechanism.make_nasa_polynomial_vibrational_energy_expr`, and
        :class:`BaseMechanism.make_nasa_polynomial_vibrational_specific_heat_expr`.
        """
        assert not self.translational_rotational_energy_exprs.size
        assert not self.nasa_polynomial_vibrational_energy_exprs.size
        assert not self.nasa_polynomial_vibrational_specific_heat_exprs.size
        for species_index in range(self.num_species):
            self.translational_rotational_energy_exprs = np.append(
                self.translational_rotational_energy_exprs,
                self.make_translational_rotational_energy_expr(species_index)
            )
            self.nasa_polynomial_vibrational_energy_exprs = np.append(
                self.nasa_polynomial_vibrational_energy_exprs,
                self.make_nasa_polynomial_vibrational_energy_expr(species_index)
            )
            self.nasa_polynomial_vibrational_specific_heat_exprs = np.append(
                self.nasa_polynomial_vibrational_specific_heat_exprs,
                self.make_nasa_polynomial_vibrational_specific_heat_expr(species_index)
            )

    def make_vt_transfer(self):
        """Loop over VT-active molecules to build their pairwise
        relaxation-time expressions and Landau-Teller energy-transfer
        contributions by invoking
        :class:`BaseMechanism.make_vt_relaxation_time_exprs` and
        :class:`BaseMechanism.make_vt_energy_transfer_expr`.
        """
        assert not self.pressure_relaxation_time_exprs.size
        assert not self.vt_energy_transfer_exprs.size
        for vt_molecule_index in range(self.num_vt_molecules):
            self.pressure_relaxation_time_exprs = np.append(
                self.pressure_relaxation_time_exprs,
                self.make_vt_relaxation_time_exprs(vt_molecule_index)
            )
            self.vt_energy_transfer_exprs = np.append(
                self.vt_energy_transfer_exprs,
                self.make_vt_energy_transfer_expr(vt_molecule_index)
            )

    def _compose_species_production_rate_graph(self):
        """Fully substitute the staged placeholder arrays
        (``concentrations``, ``k_fwd``, ``log_k_eq``, ``r_net``) with the
        actual upstream expressions already stored on this mechanism --
        the same expressions codegen renders, just composed into one
        graph per species instead of staged, so the whole thing can be
        differentiated in one pass.

        :returns: A list of :class:`pymbolic.primitives.ExpressionNode`,
            one per species, in terms of ``density``, ``temperature``
            (bare, or indexed if :attr:`num_temp` > 1), and
            ``mass_fractions`` only.
        """
        from pymbolic import substitute

        conc = p.Variable("concentrations")
        k_fwd = p.Variable("k_fwd")
        log_k_eq = p.Variable("log_k_eq")
        gibbs_rt = p.Variable("gibbs_rt")
        r_net = p.Variable("r_net")
        density = p.Variable("density")
        mass_fractions = p.Variable("mass_fractions")

        conc_subst = {
            conc[k]: density * mass_fractions[k] / self.molecular_weights[k]
            for k in range(self.num_species)
        }
        # equil_constants[j] is itself staged one level deeper, in terms
        # of per-species Gibbs energy placeholders (see
        # chem_expr/thermo.py:equilibrium_constant_expr) -- substitute
        # those with the actual NASA-polynomial Gibbs expressions before
        # folding equil_constants into rate_subst below.
        gibbs_subst = {
            gibbs_rt[i]: self.species_nasa_thermo_polynomials[i].gibbs_poly.expr
            for i in range(self.num_species)
        }
        rate_subst = {
            k_fwd[j]: self.rate_coeffs[j].expr
            for j in range(self.num_reactions)
        }
        rate_subst.update({
            log_k_eq[j]: substitute(self.equil_constants[j], gibbs_subst)
            for j in range(self.num_reactions)
        })
        r_net_subst = {
            r_net[j]: substitute(
                substitute(self.mass_action_rates[j], rate_subst),
                conc_subst
            )
            for j in range(self.num_reactions)
        }

        return [
            substitute(self.species_prod_rates[i], r_net_subst)
            for i in range(self.num_species)
        ]

    def _species_production_rate_jacobian_wrt_vars(self):
        """:returns: The ordered list of state-variable leaves the
        species-production-rate Jacobian is differentiated against:
        density, then each temperature, then each mass fraction. This
        ordering is the column order of
        :attr:`species_production_rate_jacobian_exprs` and of
        ``get_net_production_rates_jacobian`` in generated code.
        """
        density = p.Variable("density")
        mass_fractions = p.Variable("mass_fractions")
        if self.num_temp == 1:
            temperature_vars = [p.Variable("temperature")]
        else:
            temperature_vars = [
                p.Variable("temperature")[t] for t in range(self.num_temp)
            ]
        return (
            [density] + temperature_vars
            + [mass_fractions[k] for k in range(self.num_species)]
        )

    def make_species_production_rate_jacobian(self):
        """Compose the full species-production-rate graph and
        differentiate it symbolically, w.r.t. density, each
        temperature, and each mass fraction. Populates
        :attr:`species_production_rate_jacobian_exprs` with shape
        ``(num_species, 1 + num_temp + num_species)``.

        This is opt-in: unlike :meth:`make_rates`/:meth:`make_thermo`,
        it is not called automatically, because composing collapses the
        staged/shared intermediate arrays the rest of this class keeps
        separate, and symbolic differentiation cost grows with mechanism
        size -- callers (or code generators, via
        ``CodeGenerationOptions.compute_jacobian``) ask for this
        explicitly.
        """
        if self.num_species > self._jac_warning_threshold:
            import warnings
            warnings.warn(
                f"Composing and differentiating the analytic "
                f"species-production-rate Jacobian for "
                f"{self.num_species} species may be slow -- symbolic "
                f"differentiation cost grows with mechanism size.",
                stacklevel=2,
            )

        from pyrometheus.bandit.chem_expr.jacobian import jacobian_row

        composed = self._compose_species_production_rate_graph()
        wrt_vars = self._species_production_rate_jacobian_wrt_vars()
        self.species_production_rate_jacobian_exprs = np.array(
            [jacobian_row(w_dot_i, wrt_vars) for w_dot_i in composed],
            dtype=object,
        )

    def make_pyro(self, pyro_np=np):
        """
        Generate the computational engine using Pyrometheus.
        """
        from minipyro.codegen.python import get_thermochem_class
        pyro_class, self.pyro_code = get_thermochem_class(
            self,
            self.hardcode_params
        )
        self.pyro_engine = pyro_class(pyro_np)
        self.pyro_generated = True

    def compile_pyro(self, pyro_name, wg_size, *args):
        assert self.pyro_generated
        self.pyro_graph = getattr(self.pyro_engine, pyro_name)(*args)
        self.pyro_graph.compile(pyro_name, wg_size)
        self.pyro_compiled = True

    def compute(self, pyro_name, *args):
        assert self.pyro_compiled
        return self.pyro_graph.evaluate(*args)

    def print_pyro_code(self):
        print(20*"==" + "Pyro Code" + 20*"==")
        print(self.pyro_code)
        print(40*"==")

    def print_dev_code(self):
        print(20*"==" + "CUDA Code" + 20*"==")
        print(self.pyro_graph.cuda_code)
        print(40*"==")
