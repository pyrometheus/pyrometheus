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

def relaxation_rate_expr() -> p.ExpressionNode:
    pass

# }}}
