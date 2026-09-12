import pymbolic.primitives as p
import numpy as np
from dataclasses import dataclass, field, fields, InitVar
from functools import singledispatch
from typing import List
from pymbolic import substitute


# {{{ polynomial processing

t = p.Variable("temperature")
log = p.Variable("log")
exp = p.Variable("exp")


@dataclass
class PolynomialParameters:
    num_intervals: int
    num_coeff: int
    t_bounds: np.ndarray   # shape (num_intv + 1,)
    coeffs: np.ndarray     # shape (num_coeff, num_intv)


@dataclass
class NasaPoly:
    poly_params: InitVar[PolynomialParameters]
    expr: p.ExpressionNode = field(init=False)
    variable: str

    def __post_init__(self, poly_params: PolynomialParameters):
        self.expr = _nasa_poly_expr[self.variable](poly_params)


@dataclass
class SpeciesNASAThermo:
    poly_params: InitVar[PolynomialParameters]
    cp_poly: NasaPoly = field(init=False)
    enthalpy_poly: NasaPoly = field(init=False)
    entropy_poly: NasaPoly = field(init=False)
    gibbs_poly: NasaPoly = field(init=False)

    def __post_init__(self, poly_params):
        self.cp_poly = NasaPoly(
            poly_params=poly_params,
            variable="cp"
        )
        self.enthalpy_poly = NasaPoly(
            poly_params=poly_params,
            variable="enthalpy"
        )
        self.entropy_poly = NasaPoly(
            poly_params=poly_params,
            variable="entropy"
        )
        self.gibbs_poly = NasaPoly(
            poly_params=poly_params,
            variable="gibbs"
        )


@dataclass
class SpeciesElectronicThermo:
    specific_gas_constant: InitVar[np.float64]
    electronic_levels: InitVar[np.ndarray]
    specific_heat_expr: p.ExpressionNode = field(init=False)
    energy_expr: p.ExpressionNode = field(init=False)

    def __post_init__(self, specific_gas_constant, electronic_levels):
        self.specific_heat_expr = electronic_specific_heat_expr(
            specific_gas_constant, electronic_levels
        )
        self.energy_expr = electronic_energy_expr(
            specific_gas_constant, electronic_levels
        )


@dataclass
class SpeciesVibrationalThermo:
    specific_gas_constant: InitVar[np.float64]
    vibrational_temperatures: InitVar[np.ndarray]
    specific_heat_expr: p.ExpressionNode = field(init=False)
    energy_expr: p.ExpressionNode = field(init=False)

    def __post_init__(self, specific_gas_constant, vibrational_temperatures):
        self.specific_heat_expr = vibrational_specific_heat_expr(
            specific_gas_constant, vibrational_temperatures
        )
        self.energy_expr = vibrational_energy_expr(
            specific_gas_constant, vibrational_temperatures
        )

# }}}


# {{{ polynomial processing

def nasa_conditional(poly_params: PolynomialParameters,
                     part_gen):
    num_int = poly_params.num_intervals
    bounds = poly_params.t_bounds
    # Build from inside out: start with lowest-T interval, wrap upward
    result = part_gen(poly_params.coeffs[:, 0], t)
    for i in range(1, num_int):
        result = p.If(
            p.Comparison(t, ">", bounds[i]),
            part_gen(poly_params.coeffs[:, i], t),
            result,
        )
    return result


@singledispatch
def nasa_poly_specific_heat_cp_expr(poly_params):
    raise TypeError("unexpected argument type in poly_to_expr: "
                    f"{type(poly_params)}")


@nasa_poly_specific_heat_cp_expr.register
def _(poly_params: PolynomialParameters):
    def gen_nasa7(c, t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] * t
            + c[2] * t ** 2
            + c[3] * t ** 3
            + c[4] * t ** 4
        )

    def gen_nasa9(c, t):
        assert len(c) == 9
        return (
            c[0] / (t**2)
            + c[1] / t
            + c[2]
            + c[3] * t
            + c[4] * t ** 2
            + c[5] * t ** 3
            + c[6] * t ** 4
        )

    if poly_params.num_coeff == 7:
        return nasa_conditional(poly_params, gen_nasa7)
    elif poly_params.num_coeff == 9:
        return nasa_conditional(poly_params, gen_nasa9)
    else:
        raise ValueError("Wrong number of coefficients "
                         "{poly_params.num_coeff}")


@singledispatch
def nasa_poly_enthalpy_expr(poly_params):
    raise TypeError("unexpected argument type in poly_to_enthalpy_expr: "
                    f"{type(poly_params)}")


@nasa_poly_enthalpy_expr.register
def _(poly_params: PolynomialParameters):
    def gen_nasa7(c, t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] / 2 * t
            + c[2] / 3 * t ** 2
            + c[3] / 4 * t ** 3
            + c[4] / 5 * t ** 4
            + c[5] / t
        )

    def gen_nasa9(c, t):
        assert len(c) == 9
        return (
            -c[0] / (t**2)
            + c[1] * log(t) / t
            + c[2]
            + c[3] / 2 * t
            + c[4] / 3 * t ** 2
            + c[5] / 4 * t ** 3
            + c[6] / 5 * t ** 4
            + c[7] / t
        )

    if poly_params.num_coeff == 7:
        return nasa_conditional(poly_params, gen_nasa7)
    elif poly_params.num_coeff == 9:
        return nasa_conditional(poly_params, gen_nasa9)
    else:
        raise ValueError("Wrong number of coefficients "
                         f"{poly_params.num_coeff}")


@singledispatch
def nasa_poly_entropy_expr(poly_params):
    raise TypeError("unexpected argument type in poly_to_entropy_expr: "
                    f"{type(poly_params)}")


@nasa_poly_entropy_expr.register
def _(poly_params: PolynomialParameters):
    def gen_nasa7(c, t):
        assert len(c) == 7
        return (
            c[0] * log(t)
            + c[1] * t
            + c[2] / 2 * t ** 2
            + c[3] / 3 * t ** 3
            + c[4] / 4 * t ** 4
            + c[6]
        )

    def gen_nasa9(c, t):
        assert len(c) == 9
        return (
            -c[0] / (2 * t ** 2)
            - c[1] / t
            + c[2] * log(t)
            + c[3] * t
            + c[4] / 2 * t ** 2
            + c[5] / 3 * t ** 3
            + c[6] / 4 * t ** 4
            + c[8]
        )

    if poly_params.num_coeff == 7:
        return nasa_conditional(poly_params, gen_nasa7)
    elif poly_params.num_coeff == 9:
        return nasa_conditional(poly_params, gen_nasa9)
    else:
        raise ValueError("Wrong number of coefficients "
                         f"{poly_params.num_coeff}")


@singledispatch
def nasa_poly_gibbs_expr(poly_params):
    raise TypeError("unexpected argument type in poly_to_gibbs_expr: "
                    f"{type(poly)}")


@nasa_poly_gibbs_expr.register
def _(poly_params: PolynomialParameters):
    def gen_nasa7(c, t):
        assert len(c) == 7
        h = (
            c[0]
            + c[1]/2*t
            + c[2]/3*t**2
            + c[3]/4*t**3
            + c[4]/5*t**4
            + c[5]/t
        )
        s = (
            c[0]*log(t)
            + c[1]*t
            + c[2]/2*t**2
            + c[3]/3*t**3
            + c[4]/4*t**4
            + c[6]
        )
        return h - s

    def gen_nasa9(c, t):
        assert len(c) == 9
        h = (
            -c[0]/t**2
            + c[1]*log(t)/t
            + c[2]
            + c[3]/2*t
            + c[4]/3*t**2
            + c[5]/4*t**3
            + c[6]/5*t**4
            + c[7]/t
        )
        s = (
            -c[0]/(2*t**2)
            - c[1]/t
            + c[2]*log(t)
            + c[3]*t
            + c[4]/2*t**2
            + c[5]/3*t**3
            + c[6]/4*t**4
            + c[8]
        )
        return h - s

    if poly_params.num_coeff == 7:
        return nasa_conditional(poly_params, gen_nasa7)
    elif poly_params.num_coeff == 9:
        return nasa_conditional(poly_params, gen_nasa9)
    else:
        raise ValueError("Wrong number of coefficients "
                         f"{poly_params.num_coeff}")


_nasa_poly_expr = {
    "cp": nasa_poly_specific_heat_cp_expr,
    "enthalpy": nasa_poly_enthalpy_expr,
    "entropy": nasa_poly_entropy_expr,
    "gibbs": nasa_poly_gibbs_expr,
}


def make_species_nasa_thermo(poly_params: PolynomialParameters,
                             temperature: p.Variable) -> SpeciesNASAThermo:

    thermo_container = SpeciesNASAThermo(poly_params)
    for f in fields(thermo_container):
        poly = getattr(thermo_container, f.name)
        poly.expr = substitute(poly.expr, {t: temperature})

    return thermo_container

# }}}


# {{{ Equilibrium Constants

def equilibrium_constant_expr(reaction_index: int,
                              indices: List[int],
                              stoich_coeff: List[float],
                              p_not: float,
                              gas_constant: float):
    g = p.Variable("gibbs_rt")
    sum_reac = sum(
        nu * g[i] for nu, i in zip(stoich_coeff[0], indices[0])
    )
    sum_prod = sum(
        nu * g[i] for nu, i in zip(stoich_coeff[1], indices[1])
    )
    sum_nu_net = sum(stoich_coeff[1]) - sum(stoich_coeff[0])
    if sum_nu_net:
        c = log(p_not / gas_constant / p.Variable("temperature"))
        return sum_prod - sum_reac - sum_nu_net * c
    else:
        return sum_prod - sum_reac

# }}}


# {{{ Vibrational nonequlibrium

def make_species_vibrational_thermo(
        specific_gas_constant: np.float64,
        vibrational_temperatures: np.ndarray
) -> SpeciesVibrationalThermo:
    return SpeciesVibrationalThermo(specific_gas_constant,
                                    vibrational_temperatures)


def vibrational_specific_heat_expr(
        specific_gas_constant: np.float64,
        vibrational_temperatures: np.ndarray
) -> p.ExpressionNode:
    return np.sum([
        specific_gas_constant
        * exp(t_vib / t[1])
        * (t_vib / t[1])**2
        / (exp(t_vib / t[1]) - 1)**2
        for t_vib in vibrational_temperatures
    ])


def vibrational_energy_expr(specific_gas_constant: np.float64,
                            vibrational_temperatures: np.ndarray) -> p.ExpressionNode:
    return np.sum([
        specific_gas_constant
        * t_vib
        / (exp(t_vib / t[1]) - 1)
        for t_vib in vibrational_temperatures
    ])


def _electronic_boltzmann_sums(electronic_levels: np.ndarray,
                               temperature: p.ExpressionNode):
    """Return the partition function and its first two energy moments,
    ``sum g exp(-theta/T)``, ``sum g theta exp(-theta/T)`` and
    ``sum g theta**2 exp(-theta/T)``, over the (degeneracy,
    characteristic temperature) pairs in *electronic_levels*.
    """
    weights = [
        degeneracy * exp(-level_temperature / temperature)
        for degeneracy, level_temperature in electronic_levels
    ]
    return (
        np.sum(weights),
        np.sum([
            level_temperature * weight
            for (_, level_temperature), weight
            in zip(electronic_levels, weights)
        ]),
        np.sum([
            level_temperature ** 2 * weight
            for (_, level_temperature), weight
            in zip(electronic_levels, weights)
        ]),
    )


def electronic_energy_expr(
        specific_gas_constant: np.float64,
        electronic_levels: np.ndarray) -> p.ExpressionNode:
    """Return the electronic energy: the Boltzmann average of the level
    energies over the electronic partition function. For atoms this is
    the only internal energy mode.
    """
    partition, first_moment, _ = _electronic_boltzmann_sums(
        electronic_levels, t[1]
    )
    return specific_gas_constant * first_moment / partition


def electronic_specific_heat_expr(
        specific_gas_constant: np.float64,
        electronic_levels: np.ndarray) -> p.ExpressionNode:
    """Return the derivative of :func:`electronic_energy_expr` with
    respect to temperature: the variance of the level energies over the
    electronic partition function, scaled by ``R/T**2``.
    """
    partition, first_moment, second_moment = _electronic_boltzmann_sums(
        electronic_levels, t[1]
    )
    return specific_gas_constant * (
        second_moment / partition - (first_moment / partition) ** 2
    ) / t[1] ** 2


def make_species_electronic_thermo(
        specific_gas_constant: np.float64,
        electronic_levels: np.ndarray) -> SpeciesElectronicThermo:
    return SpeciesElectronicThermo(specific_gas_constant, electronic_levels)

# }}}


# {{{ Translational-rotational / NASA-polynomial vibronic thermodynamics
#
# Two-temperature energy bookkeeping splits a species' total NASA9 enthalpy
# into a translational-rotational piece (a classical, constant-Cp ideal-gas
# formula, evaluated at the heavy-particle temperature) and a residual
# "vibronic" piece (whatever the NASA9 fit has left over once the tr-rot
# baseline is removed, capturing vibration, anharmonicity, and electronic
# excitation, evaluated at the vibrational temperature). The two pieces are
# constructed so they sum back to exactly the full NASA9 enthalpy when both
# temperatures coincide.

def translational_rotational_energy_expr(
        cv_translational_rotational: float,
        reference_temperature: float,
        reference_energy_of_formation: float,
        temperature: p.ExpressionNode) -> p.ExpressionNode:
    """Translational-rotational internal energy: a classical (constant-Cv)
    ideal-gas formula referenced at *reference_temperature*.
    """
    return (
        cv_translational_rotational * (temperature - reference_temperature)
        + reference_energy_of_formation
    )


def nasa_polynomial_vibrational_energy_expr(
        enthalpy_rt_expr: p.ExpressionNode,
        specific_gas_constant: float,
        cp_translational_rotational: float,
        reference_temperature: float,
        reference_enthalpy_of_formation: float,
        temperature: p.ExpressionNode) -> p.ExpressionNode:
    """Vibronic (vibrational + electronic) energy: the full NASA9 enthalpy
    at *temperature*, minus the translational-rotational baseline. *
    enthalpy_rt_expr* is the species' dimensionless NASA9 enthalpy (h/RT)
    expression, already substituted so it is a function of *temperature*.
    """
    nasa_polynomial_enthalpy = enthalpy_rt_expr * specific_gas_constant * temperature
    return (
        nasa_polynomial_enthalpy
        - cp_translational_rotational * (temperature - reference_temperature)
        - reference_enthalpy_of_formation
    )


def nasa_polynomial_vibrational_specific_heat_expr(
        cp_r_expr: p.ExpressionNode,
        specific_gas_constant: float,
        cp_translational_rotational: float) -> p.ExpressionNode:
    """Derivative of :func:`nasa_polynomial_vibrational_energy_expr` with
    respect to temperature: the full NASA9 Cp, minus the same tr-rot
    baseline. *cp_r_expr* is the species' dimensionless NASA9 Cp (Cp/R)
    expression, already substituted so it is a function of the target
    temperature.
    """
    nasa_polynomial_specific_heat_cp = cp_r_expr * specific_gas_constant
    return nasa_polynomial_specific_heat_cp - cp_translational_rotational

# }}}
