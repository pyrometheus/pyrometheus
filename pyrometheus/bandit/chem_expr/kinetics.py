import numpy as np
import pymbolic.primitives as p
from pymbolic import substitute
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List, Tuple


# {{{

def _ones_like(arg):
    return 0 * arg + 1

# }}}


# {{{ Constants

_boltzmann = 1.380649e-23
_avogadro = 6.02214076e23

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


# {{{ Third bodies

def third_body_concentration_expr(
        num_species: int,
        efficiencies: Dict[int, float],
        default_efficiency: float = 1.0) -> p.ExpressionNode:
    """Return the efficiency-weighted sum of species concentrations that
    a third body contributes to a rate coefficient.

    :arg efficiencies: Collision efficiencies keyed by species index.
        Every species absent from it takes *default_efficiency*.
    """
    weighted_terms = [
        efficiency * conc[species_index]
        for species_index, efficiency in efficiencies.items()
    ]
    default_terms = [
        conc[species_index] for species_index in range(num_species)
        if species_index not in efficiencies
    ]
    if default_terms:
        default_sum = np.sum(default_terms)
        weighted_terms.append(
            default_sum if default_efficiency == 1
            else default_efficiency * default_sum
        )
    return np.sum(weighted_terms)

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
        return k_fwd[rxn_index] * (r_fwd - exp(log_k_eq[rxn_index]) * r_rev)
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


# {{{ Vibrational-translational (VT) energy transfer

def pairwise_relaxation_time_expr(
        millikan_white_a: float,
        millikan_white_b: float,
        park_cross_section: float,
        reduced_molar_mass_sqrt: float,
        atmospheric_pressure: float,
        temperature: p.ExpressionNode) -> p.ExpressionNode:
    """Return the pressure-scaled Millikan-White + Park VT relaxation time
    for one (VT-active molecule, heavy collision partner) pair, as a
    pymbolic ExpressionNode in *temperature*.
    """
    millikan_white_term = exp(
        millikan_white_a * (temperature ** (-1 / 3) - millikan_white_b)
        - 18.42
    ) * atmospheric_pressure

    park_high_temperature_factor = p.If(
        p.Comparison(temperature, "<=", 20000),
        (temperature / 50000) ** 2,
        1 / 6.25,
    )
    # sqrt(pi * boltzmann_constant / (8 * avogadro_number)), matching
    # PLATO's VT_FAC_PARK (general_transfer.F90)

    park_relaxation_constant = np.sqrt(
        np.pi * _boltzmann / (8 * _avogadro)
    )
    park_term = (
        park_relaxation_constant * temperature ** 0.5
        * park_high_temperature_factor
        / park_cross_section * reduced_molar_mass_sqrt
    )
    return millikan_white_term + park_term


def vt_mean_relaxation_rate_expr(
        pressure: p.ExpressionNode,
        heavy_partner_mole_ratios: List[p.ExpressionNode],
        pairwise_relaxation_times: List[p.ExpressionNode]) -> p.ExpressionNode:
    """Return the inverse mean VT relaxation time (SSH frequency average)
    for one VT-active molecule, given the mass-fraction-over-molar-mass
    weight of every heavy collision partner and the corresponding pairwise
    relaxation times from :func:`pairwise_relaxation_time_expr`.

    *pairwise_relaxation_times* are pressure-normalized (Pa*s, a function of
    temperature only, per the Millikan-White scaling p*tau = f(T)); the
    actual mean relaxation rate 1/tau at the current gas state requires
    multiplying the SSH-averaged frequency by the current *pressure*
    (matching PLATO's ``ov_tau_VT = (p*Dm)/N`` in ``add_Omega_VT``).
    """
    heavy_partner_weight_sum = sum(heavy_partner_mole_ratios)
    relaxation_rate_sum = sum(
        weight / relaxation_time
        for weight, relaxation_time in zip(
            heavy_partner_mole_ratios, pairwise_relaxation_times)
    )
    return pressure * relaxation_rate_sum / heavy_partner_weight_sum


def vt_energy_transfer_expr(
        density: p.ExpressionNode,
        molecule_mass_fraction: p.ExpressionNode,
        vibrational_energy_at_heavy_temperature: p.ExpressionNode,
        vibrational_energy_at_vibrational_temperature: p.ExpressionNode,
        mean_relaxation_rate: p.ExpressionNode) -> p.ExpressionNode:
    """Return the Landau-Teller VT energy-transfer contribution of one
    VT-active molecule to Omega_VT [W/m^3].
    """
    return (
        density * molecule_mass_fraction
        * (vibrational_energy_at_heavy_temperature
           - vibrational_energy_at_vibrational_temperature)
        * mean_relaxation_rate
    )

# }}}
