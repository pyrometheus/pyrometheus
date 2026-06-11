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
        shape=(0,), type=p.ExpressionNode
    )
    species_nasa_thermo_polynomials: np.ndarray = np.empty(
        shape=(0,), dtype=SpeciesNASAThermo
    )
    species_vib_thermo_expressions: np.ndarray = np.empty(
        shape=(0,), dtype=SpeciesVibrationalThermo
    )
    param_vals: np.ndarray = np.empty(shape=(0,), dtype=np.float64)

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
    def molecular_weights(self):
        """
        :returns: The molecular weights for the species in the mechanism.
        """
        raise NotImplementedError

    def has_nonequilibrium_energy_modes(self):
        return self.nonequil_thermo

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
        assert not self.species_thermo_polynomials.size
        for isp in range(self.num_species):
            self.species_thermo_polynomials = np.append(
                self.species_thermo_polynomials,
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
        if self.has_nonequilibrium_energy_modes:
            for isp in range(self.num_species):
                self.species_vib_thermo_expressions = np.append(
                    self.species_vib_thermo_expressions,
                    self.make_species_vibrational_thermo(isp)
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
