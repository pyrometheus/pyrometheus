import numpy as np
import cantera as ct
from typing import List, Union, Tuple, Dict
from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.kinetics import (
    make_arrhenius,
    reaction_progress_rate_expr,
    species_production_rate_expr
)


class Cantera(BaseNamespace):

    def __init__(self, file_name):
        self.one_atm = ct.one_atm
        self.gas_constant = ct.gas_constant
        self.sol = ct.Solution(file_name)

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.sol, name)(*args)
        else:
            return getattr(self.sol, name)


class CanteraMechanism(BaseMechanism):

    num_temp = 1

    def __init__(self, file_name, pyro_np=np, hardcode_params=True):
        self.hardcode_params = hardcode_params
        self.namespace = Cantera(file_name)
        self.make_rates(hardcode_params)
        # self.make_pyro(pyro_np)

    @property
    def num_species(self):
        return self.namespace.__getattr__("n_species")

    @property
    def num_reactions(self):
        return self.namespace.__getattr__("n_reactions")

    @property
    def molecular_weights(self):
        return self.namespace.__getattr__("molecular_weights")

    @property
    def species_names(self):
        return self.namespace.__getattr__("species_names")

    def reactions(self) -> List[ct.Reaction]:
        return self.namespace.__getattr__("reactions").__call__()

    def reaction(self, reaction_index: int) -> ct.Reaction:
        return self.namespace.__getattr__("reaction", reaction_index)

    def is_reversible(self, reaction_index: int) -> bool:
        return self.reaction(reaction_index).reversible
    
    def species(self, species_index: int) -> ct.Species:
        return self.namespace.__getattr__("species", species_index)

    def species_index(self, species_name: str) -> int:
        return self.namespace.__getattr__("species_index", species_name)

    def species_name(self, species_index: int) -> str:
        return self.species_names[species_index]

    def reactants(self, reaction_index: int) -> List[int]:
        reac_dict = self.reaction(reaction_index).reactants
        return [self.species_index(sp) for sp in reac_dict]

    def products(self, reaction_index: int) -> List[int]:
        prod_dict = self.reaction(reaction_index).products
        return [self.species_index(sp) for sp in prod_dict]

    def reaction_stoichiometry(self, reaction_index: int) -> Tuple[Dict]:
        return tuple((
            self.reaction(reaction_index).reactants,
            self.reaction(reaction_index).products
        ))

    def stoichiometric_coefficient(self,
                                   reaction_index: int,
                                   species_index: int,
                                   direction: str) -> int:
        species_name = self.species_name(species_index)
        reac_dict, prod_dict = self.reaction_stoichiometry(reaction_index)
        if direction == "fwd":
            return int(reac_dict[species_name])
        elif direction == "rev":
            return int(prod_dict[species_name])
        else:
            raise ValueError

    def stoichiometric_coefficients(
            self, reaction_index: int
    ) -> Tuple[List[int], List[int]]:
        reac_dict, prod_dict = self.reaction_stoichiometry(reaction_index)
        return tuple((
            list([int(v) for v in reac_dict.values()]),
            list([int(v) for v in prod_dict.values()])
        ))

    def participation_set(
            self, species_id: Union[int, str]
    ) -> Tuple[List[int]]:

        if isinstance(species_id, int):
            assert species_id < self.num_species
            species_name = self.species_name(species_id)
        elif isinstance(species_id, str):
            assert species_id in self.species_names
            species_name = species_id

        fwd_set = [i for i, r in enumerate(self.reactions())
                   if species_name in r.reactants]
        rev_set = [i for i, r in enumerate(self.reactions())
                   if species_name in r.products]

        return (fwd_set, rev_set)

    def production_balance(
            self, species_index: int
    ) -> Tuple[List[int], List[int]]:

        fwd_set, rev_set = self.participation_set(species_index)

        fwd_coeff = []
        rev_coeff = []
        for reaction_index in fwd_set:
            fwd_coeff += [self.stoichiometric_coefficient(
                reaction_index, species_index, "fwd"
            )]

        for reaction_index in rev_set:
            rev_coeff += [self.stoichiometric_coefficient(
                reaction_index, species_index, "rev"
            )]
        return (
            (fwd_set, rev_set),
            (fwd_coeff, rev_coeff)
        )

    def make_rate_coefficient(self, reaction_index, hardcode_params):
        rate = self.reaction(reaction_index).rate
        if isinstance(rate, ct.reaction.ArrheniusRate):
            if hardcode_params:
                params = {
                    "a": np.log(rate.pre_exponential_factor),
                    "b": rate.temperature_exponent,
                    "t_a": rate.activation_energy / self.namespace.gas_constant
                }
                return make_arrhenius(
                    reaction_index=reaction_index, params=params
                ), params
            else:
                params = np.array([
                    np.log(rate.pre_exponential_factor),
                    rate.temperature_exponent,
                    rate.activation_energy / self.namespace.gas_constant
                ])
                return make_arrhenius(reaction_index=reaction_index), params
        else:
            return 0, 0

    def make_mass_action_rate(self, reaction_index):
        reac_indices = self.reactants(reaction_index)
        prod_indices = self.products(reaction_index)
        stoich_coeff = self.stoichiometric_coefficients(reaction_index)
        return reaction_progress_rate_expr(
            reaction_index,
            self.reaction(reaction_index).reversible,
            tuple((reac_indices, prod_indices)),
            stoich_coeff
        )

    def make_species_production_rate(self, species_index):
        part_sets, stoich_coeffs = self.production_balance(species_index)
        return species_production_rate_expr(
            species_index,
            part_sets[0],
            part_sets[1],
            stoich_coeffs[0],
            stoich_coeffs[1]
        )
