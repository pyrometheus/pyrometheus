import numpy as np
import pymbolic.primitives as p
from pymbolic import substitute
from dataclasses import dataclass, field, InitVar
from typing import Union, Optional, List, Tuple


# {{{

def _ones_like(arg):
    return 0 * arg + 1

# }}}


# {{{ Common variables

log = p.Variable("log")
exp = p.Variable("exp")
k_fwd = p.Variable("k_fwd")
log_k_eq = p.Variable("log_k_eq")
r_net = p.Variable("r_net")
conc = p.Variable("concentrations")
_temp = p.Variable("temperature")

# }}}


# {{{ Rate Coefficients

@dataclass
class RateCoefficient:
    reaction_index: Union[int, p.Variable]
    expr: p.ExpressionNode = field(init=False)
    params: Optional[dict] = None


@dataclass
class ArrheniusCoefficient(RateCoefficient):
    def __post_init__(self,):
        self.a = p.Variable("a")[self.reaction_index]
        self.b = p.Variable("b")[self.reaction_index]
        self.t_a = p.Variable("t_a")[self.reaction_index]
        self.expr = exp(self.a + self.b * log(_temp) -
                        self.t_a / _temp)

        if self.params:
            self.hardcore_parameters(self.params)
        else:
            self.standarize_parameters()

    def hardcore_parameters(self, params: dict):
        from pymbolic import substitute
        self.expr = substitute(self.expr, {
            self.a: params["a"],
            self.b: params["b"],
            self.t_a: params["t_a"]
        })

    def standardize_parameters(self):
        from pymbolic import substitute
        self.expr = substitute(self.expr, {
            self.a: p.Variable("params")[self.reaction_index, 0],
            self.b: p.Variable("params")[self.reaction_index, 1],
            self.t_a: p.Variable("params")[self.reaction_index, 2],
        })


def make_arrhenius(reaction_index: int,
                   temperature: Optional[p.Variable] = None,
                   **kwargs) -> ArrheniusCoefficient:
    coeff = ArrheniusCoefficient(
        reaction_index=reaction_index,
        **kwargs
    )
    if temperature:
        coeff.expr = substitute(
            coeff.expr,
            {_temp: temperature}
        )
    else:
        pass

    return coeff

# }}}


# {{{ Species-production and Reaction-progress Rates

def mass_action_rxn_progress_rate_expr(
        rxn_index: int,
        indices: List[int],
        stoich_coeff: List[float]) -> p.ExpressionNode:
    """Return the mass-action expression, as a pymbolic ExpressionNode for
    the rate of progress.
    """
    return np.prod([
        conc[i]**nu for i, nu in zip(indices, stoich_coeff)
    ])


def reaction_progress_rate_expr(
        rxn_index: int,
        reversible: bool,
        indices: Tuple[List[int], ...],
        stoich_coeff: Tuple[List[int], ...]) -> p.ExpressionNode:
    """Return the net rate of progress of reaction with index *rxn_index*
    as a pymbolic ExpressionNode.
    """
    r_fwd = mass_action_rxn_progress_rate_expr(
        rxn_index, indices[0], stoich_coeff[0]
    )
    if reversible:
        assert len(indices) == 2 and len(stoich_coeff) == 2
        r_rev = mass_action_rxn_progress_rate_expr(
            rxn_index, indices[1], stoich_coeff[1]
        )
        return k_fwd[rxn_index] * r_fwd - exp(log_k_eq[rxn_index]) * r_rev
    else:
        return k_fwd[rxn_index] * r_fwd


def species_production_rate_expr(sp_index: int,
                                 fwd_part_set: List[int],
                                 rev_part_set: List[int],
                                 stoich_fwd: List[float],
                                 stoich_rev: List[float]) -> p.ExpressionNode:
    """Return the production rate for species with index *sp_index* as
    a pymbolic ExpressionNode
    """
    ones = _ones_like(r_net[0])
    sum_fwd = sum(nu * r_net[i] for nu, i in zip(stoich_fwd, fwd_part_set))
    sum_rev = sum(nu * r_net[i] for nu, i in zip(stoich_rev, rev_part_set))
    return (sum_rev - sum_fwd) * ones

# }}}
