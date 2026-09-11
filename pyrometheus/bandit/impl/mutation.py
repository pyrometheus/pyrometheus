import numpy as np
import mutationpp as mpp
from pymbolic.primitives import Variable
from typing import Dict, List, Tuple, Union
from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.kinetics import (
    RateCoefficient,
    make_arrhenius,
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


# {{{ Unit conversion
#
# Mutation++ works in mol; bandit concentrations are kmol/m^3.

_kmol_per_mol = 1e3

# }}}


class Mutationpp(BaseNamespace):

    gas_constant = 8314.462618
    one_atm = 101325.0

    def __init__(self, mixture, state_model, thermo_database):
        options = mpp.MixtureOptions(mixture)
        options.setStateModel(state_model)
        options.setThermodynamicDatabase(thermo_database)
        self.mix = mpp.Mixture(options)

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.mix, name)(*args)
        else:
            return getattr(self.mix, name)


class MutationMechanism(BaseMechanism):

    def __init__(self,
                 mixture,
                 state_model="ChemNonEqTTv",
                 thermo_database="RRHO",
                 pyro_np=np,
                 hardcode_params=True):
        self.hardcode_params = hardcode_params
        self.namespace = Mutationpp(mixture, state_model, thermo_database)
        if self.namespace.__getattr__("hasElectrons").__call__():
            raise NotImplementedError(
                f"mixture '{mixture}' carries electrons, whose separate "
                f"temperature and third-body exclusion are not supported"
            )
        self.nonequil_thermo = self.num_temp > 1
        self.make_rates(hardcode_params)

    # {{{ Abstract interface

    @property
    def num_temp(self):
        """Return number of temperatures."""
        return self.namespace.__getattr__("num_energy_eqns")

    @property
    def num_species(self):
        """Return number of species."""
        return self.namespace.__getattr__("num_species")

    @property
    def num_reactions(self):
        """Return number of reactions."""
        return self.namespace.__getattr__("num_reactions")

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
        species that Mutation++ lists explicitly appear; every other
        heavy species takes the default efficiency of one.
        """
        return {
            species_index: float(efficiency)
            for species_index, efficiency
            in self.reaction(reaction_index).efficiencies
        }

    def rate_temperature(self, reaction_index: int, direction: str):
        """:returns: The temperature the rate coefficient of the
        reaction with index *reaction_index* is evaluated at, as a
        :class:`pymbolic.primitives.ExpressionNode`. Dissociation runs
        forward on the geometric mean of the heavy and vibrational
        temperatures but reverse on the heavy temperature alone, so the
        two directions are asked for separately.
        """
        reaction = self.reaction(reaction_index)
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
                              hardcode_params) -> RateCoefficient:
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
                temperature=self.rate_temperature(reaction_index, "fwd"),
            )
        else:
            params = np.array([
                log_pre_exponential,
                rate_law.exponent,
                rate_law.activation_temperature,
            ])
            rate_coeff = make_arrhenius(
                reaction_index=reaction_index,
                temperature=self.rate_temperature(reaction_index, "fwd"),
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

    def make_mass_action_rate(self, reaction_index):
        return reaction_progress_rate_expr(
            reaction_index,
            self.is_reversible(reaction_index),
            (self.reactants(reaction_index), self.products(reaction_index)),
            self.stoichiometric_coefficients(reaction_index),
        )

    def make_species_production_rate(self, species_index):
        part_sets, stoich_coeffs = self.production_balance(species_index)
        return species_production_rate_expr(
            species_index,
            part_sets[0], part_sets[1],
            stoich_coeffs[0], stoich_coeffs[1],
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
