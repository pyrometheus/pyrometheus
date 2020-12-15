__copyright__ = """
Copyright (C) 2020 Esteban Cisneros
Copyright (C) 2020 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from numbers import Number
from functools import singledispatch

import pymbolic.primitives as p
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
import cantera as ct
import numpy as np  # noqa: F401

from itertools import compress
from mako.template import Template


# {{{ code generation helpers


class CodeGenerationMapper(StringifyMapper):
    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    def map_if(self, expr, enclosing_prec, *args, **kwargs):
        return "self.npctx.where(%s, %s, %s)" % (
            self.rec(expr.condition, PREC_NONE, *args, **kwargs),
            self.rec(expr.then, PREC_NONE, *args, **kwargs),
            self.rec(expr.else_, PREC_NONE, *args, **kwargs),
        )

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "self.npctx.%s(%s)",
            self.rec(expr.function, PREC_CALL, *args, **kwargs),
            self.join_rec(", ", expr.parameters, PREC_NONE, *args, **kwargs),
        )


def str_np_inner(ary):
    if isinstance(ary, Number):
        return repr(ary)
    elif ary.shape:
        return "[%s]" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return "self.npctx.array(%s)" % str_np_inner(ary)


# }}}


# {{{ polynomial processing


def nasa7_conditional(t, poly, part_gen):
    # FIXME: Should check minTemp, maxTemp
    return p.If(
        p.Comparison(t, ">", poly.coeffs[0]),
        part_gen(poly.coeffs[1:8], t),
        part_gen(poly.coeffs[8:15], t),
    )


@singledispatch
def poly_to_expr(poly):
    raise TypeError(f"unexpected argument type in poly_to_expr: {type(poly)}")


@poly_to_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


@singledispatch
def poly_to_enthalpy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_enthalpy_expr: "
            f"{type(poly)}")


@poly_to_enthalpy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] / 2 * t
            + c[2] / 3 * t ** 2
            + c[3] / 4 * t ** 3
            + c[4] / 5 * t ** 4
            + c[5] / t
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


@singledispatch
def poly_to_entropy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_entropy_expr: "
            f"{type(poly)}")


@poly_to_entropy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    log = p.Variable("log")

    def gen(c, t):
        assert len(c) == 7
        return (
            c[0] * log(t)
            + c[1] * t
            + c[2] / 2 * t ** 2
            + c[3] / 3 * t ** 3
            + c[4] / 4 * t ** 4
            + c[6]
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


# }}}


# {{{ Equilibrium constants


def equilibrium_constants_expr(sol: ct.Solution, react: ct.Reaction, gibbs_rt):
    """:returns: Equilibrium constant expression for reaction *react* in terms of
    the species Gibbs functions *gibbs_rt* as a
    :class:`pymbolic.primitives.Expression`"""
    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    # Stoichiometric coefficients
    #nu_reac = [react.reactants[sp] for sp in react.reactants]
    #nu_prod = [react.products[sp] for sp in react.products]
    nu_reac = [sol.reactant_stoich_coeff(sol.species_index(sp), int(react.ID)-1)
               for sp in react.reactants]
    nu_prod = [sol.product_stoich_coeff(sol.species_index(sp), int(react.ID)-1)
               for sp in react.products]

    sum_r = sum(nu_reac_i * gibbs_rt[indices_reac_i]
            for indices_reac_i, nu_reac_i in zip(indices_reac, nu_reac))
    sum_p = sum(nu_prod_i * gibbs_rt[indices_prod_i]
            for indices_prod_i, nu_prod_i in zip(indices_prod, nu_prod))

    # Check if reaction is termolecular
    sum_nu_net = sum(nu_prod) - sum(nu_reac)
    if sum_nu_net < 0:
        # Three species on reactants side
        return sum_p - sum_nu_net*p.Variable("C0") - sum_r
    elif sum_nu_net > 0:
        # Three species on products side
        return sum_p - (sum_r - sum_nu_net*p.Variable("C0"))
    else:
        return sum_p - sum_r

# }}}


# {{{ Rate coefficients

def rate_coefficient_expr(rate_coeff: ct.Arrhenius, t):
    """:returns: The rate coefficient expression for *rate_coeff* in terms
    of the temperature *t* as a :class:`pymbolic.primitives.Expression`"""
    # Rate parameters
    a = rate_coeff.pre_exponential_factor
    b = rate_coeff.temperature_exponent
    t_a = rate_coeff.activation_energy/ct.gas_constant
    if b == 0 and t_a == 0:
        # Constant rate
        return a
    elif b != 0 and t_a == 0:
        # Weakly temperature-dependent rate
        return a * t**b
    elif b == 0 and a != 0 and t_a != 0:
        # Classic Arrhenius rate
        return p.Variable("exp")(np.log(a)-t_a/t)
    else:
        # Modified Arrhenius
        return p.Variable("exp")(np.log(a)+b*p.Variable("log")(t)-t_a/t)


def third_body_efficiencies_expr(sol: ct.Solution, react: ct.Reaction, c):
    """:returns: The third-body concentration expression for reaction *react* in terms
    of the species concentraions *c* as a :class:`pymbolic.primitives.Expression`"""
    efficiencies = [react.efficiencies[sp] for sp in react.efficiencies]
    indices_nondef = [sol.species_index(sp) for sp in react.efficiencies]
    indices_default = [i for i in range(sol.n_species) if i not in indices_nondef]
    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(efficiencies, indices_nondef))
    sum_default = react.default_efficiency * sum(c[i] for i in indices_default)
    return sum_nondef + sum_default


def troe_falloff_expr(react: ct.Reaction, t):
    """:returns: The Troe falloff center expression for reaction *react* in terms of the
    temperature *t* as a :class:`pymbolic.primitives.Expression`"""
    troe_params = react.falloff.parameters
    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[1])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[2])
    if troe_params[3] > 1.0e-16:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return troe_1 + troe_2 + troe_3
    else:
        return troe_1 + troe_2


def falloff_function_expr(react: ct.Reaction, i, t, red_pressure, falloff_center):
    """:returns: Falloff function expression for reaction *react* in terms
    of the temperature *t*, reduced pressure *red_pressure*, and falloff center
    *falloff_center* as a :class:`pymbolic.primitives.Expression`"""
    if react.falloff.falloff_type == "Troe":
        log_rp = p.Variable("log10")(red_pressure[i])
        c = -0.4-0.67*falloff_center[i]
        n = 0.75-1.27*falloff_center[i]
        f = (log_rp+c)/(n-0.14*(log_rp+c))
        return 10**((falloff_center[i])/(1+f**2))
    else:
        return 1

# }}}


# {{{ Rates of progress

def rate_of_progress_expr(sol: ct.Solution, react: ct.Reaction, c, k_fwd, k_eq):
    """:returns: Rate of progress expression for reaction *react* in terms of
    species concentrations *c* with rate coefficients *k_fwd* and equilbrium
    constants *k_eq* as a :class:`pymbolic.primitives.Expression`"""
    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    if react.orders:
        nu_reac = [react.orders[sp] for sp in react.orders]
    else:
        nu_reac = [react.reactants[sp] for sp in react.reactants]

    r_fwd = np.prod([c[index]**nu for index, nu in zip(indices_reac, nu_reac)])

    if react.reversible:
        nu_prod = [react.products[sp] for sp in react.products]
        r_rev = np.prod([c[index]**nu for index, nu in zip(indices_prod, nu_prod)])
        return k_fwd[int(react.ID)-1] * (r_fwd - k_eq[int(react.ID)-1] * r_rev)
    else:
        return k_fwd[int(react.ID)-1] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """:returns: Species production rate for species *species* in terms of
    the net reaction rates of progress *r_net* as a
    :class:`pymbolic>primitives.Expression`"""
    ones = r_net[0] / r_net[0]
    indices_fwd = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.reactants]
    indices_rev = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.products]
    nu_fwd = [sol.reactant_stoich_coeff(sol.species_index(species), react_index)
              for react_index in indices_fwd]
    nu_rev = [sol.product_stoich_coeff(sol.species_index(species), prod_index)
              for prod_index in indices_rev]
    sum_fwd = sum(nu*r_net[index] for nu, index in zip(nu_fwd, indices_fwd))
    sum_rev = sum(nu*r_net[index] for nu, index in zip(nu_rev, indices_rev))
    return (sum_rev - sum_fwd) * ones

# }}}


# {{{ main code template

code_tpl = Template(
    """
import numpy as np
from pytools.obj_array import make_obj_array


class Thermochemistry:
    def __init__(self, npctx=np):
        self.npctx = npctx
        self.model_name = ${repr(sol.source)}
        self.num_elements = ${sol.n_elements}
        self.num_species = ${sol.n_species}
        self.num_reactions = ${sol.n_reactions}
        self.num_falloff   = ${
            sum(1 if isinstance(r, ct.FalloffReaction) else 0
            for r in sol.reactions())}

        self.one_atm = ${ct.one_atm}
        self.gas_constant = ${ct.gas_constant}
        self.big_number = 1.0e300

        self.species_names = ${sol.species_names}
        self.species_indices = ${
            dict([[sol.species_name(i), i]
                for i in range(sol.n_species)])}

        self.wts = ${str_np(sol.molecular_weights)}
        self.iwts = 1/self.wts

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, Y):
        return self.gas_constant * np.dot( self.iwts, Y )

    def get_density(self, p, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return p * mmw / RT

    def get_pressure(self, rho, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return rho * RT / mmw

    def get_mix_molecular_weight(self, Y):
        return 1/np.dot( self.iwts, Y )

    def get_concentrations(self, rho, Y):
        conctest = self.iwts * rho * Y
        zero = 0 * conctest[0]
        for i, conc in enumerate(conctest):
            conctest[i] = self.npctx.where(conctest[i] > 0, conctest[i], zero)
        return conctest

    def get_mixture_specific_heat_cp_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature)
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_specific_heat_cv_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature) - 1.0
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_enthalpy_mass(self, temperature, massfractions):
        h0_rt = self.get_species_enthalpies_RT(temperature)
        hsum = sum([massfractions[i] * h0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * hsum

    def get_mixture_internal_energy_mass(self, temperature, massfractions):

        e0_rt = self.get_species_enthalpies_RT(temperature) - 1.0
        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * esum

    def get_species_specific_heats_R(self, temperature):
        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2

        cp_high = (
            2.036111e00
            + 1.464542e-02 * tt0
            - 6.710779e-06 * tt1
            + 1.472229e-09 * tt2
            - 1.257061e-13 * tt3
        )
        cp_low = (
            3.959201e00
            - 7.570522e-03 * tt0
            + 5.709903e-05 * tt1
            - 6.915888e-08 * tt2
            + 2.698844e-11 * tt3
        )
        cpr0 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.282538e00
            + 1.483088e-03 * tt0
            - 7.579667e-07 * tt1
            + 2.094706e-10 * tt2
            - 2.167178e-14 * tt3
        )
        cp_low = (
            3.782456e00
            - 2.996734e-03 * tt0
            + 9.847302e-06 * tt1
            - 9.681295e-09 * tt2
            + 3.243728e-12 * tt3
        )
        cpr1 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.857460e00
            + 4.414370e-03 * tt0
            - 2.214814e-06 * tt1
            + 5.234902e-10 * tt2
            - 4.720842e-14 * tt3
        )
        cp_low = (
            2.356774e00
            + 8.984597e-03 * tt0
            - 7.123563e-06 * tt1
            + 2.459190e-09 * tt2
            - 1.436995e-13 * tt3
        )
        cpr2 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.715186e00
            + 2.062527e-03 * tt0
            - 9.988258e-07 * tt1
            + 2.300530e-10 * tt2
            - 2.036477e-14 * tt3
        )
        cp_low = (
            3.579533e00
            - 6.103537e-04 * tt0
            + 1.016814e-06 * tt1
            + 9.070059e-10 * tt2
            - 9.044245e-13 * tt3
        )
        cpr3 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.033992e00
            + 2.176918e-03 * tt0
            - 1.640725e-07 * tt1
            - 9.704199e-11 * tt2
            + 1.682010e-14 * tt3
        )
        cp_low = (
            4.198641e00
            - 2.036434e-03 * tt0
            + 6.520402e-06 * tt1
            - 5.487971e-09 * tt2
            + 1.771978e-12 * tt3
        )
        cpr4 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.337279e00
            - 4.940247e-05 * tt0
            + 4.994568e-07 * tt1
            - 1.795664e-10 * tt2
            + 2.002554e-14 * tt3
        )
        cp_low = (
            2.344331e00
            + 7.980521e-03 * tt0
            - 1.947815e-05 * tt1
            + 2.015721e-08 * tt2
            - 7.376118e-12 * tt3
        )
        cpr5 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.926640e00
            + 1.487977e-03 * tt0
            - 5.684760e-07 * tt1
            + 1.009704e-10 * tt2
            - 6.753351e-15 * tt3
        )
        cp_low = (
            3.298677e00
            + 1.408240e-03 * tt0
            - 3.963222e-06 * tt1
            + 5.641515e-09 * tt2
            - 2.444854e-12 * tt3
        )
        cpr6 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        return make_obj_array([cpr0, cpr1, cpr2, cpr3, cpr4, cpr5, cpr6])

    def get_species_enthalpies_RT(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt4 = 1.0 / temperature

        h_high = (
            2.036111e00
            + 1.464542e-02 * 0.50 * tt0
            - 6.710779e-06 * tt1 / 3.0
            + 1.472229e-09 * 0.25 * tt2
            - 1.257061e-13 * 0.20 * tt3
            + 4.939886e03 * tt4
        )
        h_low = (
            3.959201e00
            - 7.570522e-03 * 0.50 * tt0
            + 5.709903e-05 * tt1 / 3.0
            - 6.915888e-08 * 0.25 * tt2
            + 2.698844e-11 * 0.20 * tt3
            + 5.089776e03 * tt4
        )
        hrt0 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.282538e00
            + 1.483088e-03 * 0.50 * tt0
            - 7.579667e-07 * tt1 / 3.0
            + 2.094706e-10 * 0.25 * tt2
            - 2.167178e-14 * 0.20 * tt3
            - 1.088458e03 * tt4
        )
        h_low = (
            3.782456e00
            - 2.996734e-03 * 0.50 * tt0
            + 9.847302e-06 * tt1 / 3.0
            - 9.681295e-09 * 0.25 * tt2
            + 3.243728e-12 * 0.20 * tt3
            - 1.063944e03 * tt4
        )
        hrt1 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.857460e00
            + 4.414370e-03 * 0.50 * tt0
            - 2.214814e-06 * tt1 / 3.0
            + 5.234902e-10 * 0.25 * tt2
            - 4.720842e-14 * 0.20 * tt3
            - 4.875917e04 * tt4
        )
        h_low = (
            2.356774e00
            + 8.984597e-03 * 0.50 * tt0
            - 7.123563e-06 * tt1 / 3.0
            + 2.459190e-09 * 0.25 * tt2
            - 1.436995e-13 * 0.20 * tt3
            - 4.837197e04 * tt4
        )
        hrt2 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.715186e00
            + 2.062527e-03 * 0.50 * tt0
            - 9.988258e-07 * tt1 / 3.0
            + 2.300530e-10 * 0.25 * tt2
            - 2.036477e-14 * 0.20 * tt3
            - 1.415187e04 * tt4
        )
        h_low = (
            3.579533e00
            - 6.103537e-04 * 0.50 * tt0
            + 1.016814e-06 * tt1 / 3.0
            + 9.070059e-10 * 0.25 * tt2
            - 9.044245e-13 * 0.20 * tt3
            - 1.434409e04 * tt4
        )
        hrt3 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.033992e00
            + 2.176918e-03 * 0.50 * tt0
            - 1.640725e-07 * tt1 / 3.0
            - 9.704199e-11 * 0.25 * tt2
            + 1.682010e-14 * 0.20 * tt3
            - 3.000430e04 * tt4
        )
        h_low = (
            4.198641e00
            - 2.036434e-03 * 0.50 * tt0
            + 6.520402e-06 * tt1 / 3.0
            - 5.487971e-09 * 0.25 * tt2
            + 1.771978e-12 * 0.20 * tt3
            - 3.029373e04 * tt4
        )
        hrt4 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.337279e00
            - 4.940247e-05 * 0.50 * tt0
            + 4.994568e-07 * tt1 / 3.0
            - 1.795664e-10 * 0.25 * tt2
            + 2.002554e-14 * 0.20 * tt3
            - 9.501589e02 * tt4
        )
        h_low = (
            2.344331e00
            + 7.980521e-03 * 0.50 * tt0
            - 1.947815e-05 * tt1 / 3.0
            + 2.015721e-08 * 0.25 * tt2
            - 7.376118e-12 * 0.20 * tt3
            - 9.179352e02 * tt4
        )
        hrt5 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.926640e00
            + 1.487977e-03 * 0.50 * tt0
            - 5.684760e-07 * tt1 / 3.0
            + 1.009704e-10 * 0.25 * tt2
            - 6.753351e-15 * 0.20 * tt3
            - 9.227977e02 * tt4
        )
        h_low = (
            3.298677e00
            + 1.408240e-03 * 0.50 * tt0
            - 3.963222e-06 * tt1 / 3.0
            + 5.641515e-09 * 0.25 * tt2
            - 2.444854e-12 * 0.20 * tt3
            - 1.020900e03 * tt4
        )
        hrt6 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        return make_obj_array([hrt0, hrt1, hrt2, hrt3, hrt4, hrt5, hrt6])

    def get_species_entropies_R(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt6 = self.npctx.log(tt0)

        s_high = (
            2.036111e00 * tt6
            + 1.464542e-02 * tt0
            - 6.710779e-06 * 0.50 * tt1
            + 1.472229e-09 * tt2 / 3.0
            - 1.257061e-13 * 0.25 * tt3
            + 1.030537e01
        )
        s_low = (
            3.959201e00 * tt6
            - 7.570522e-03 * tt0
            + 5.709903e-05 * 0.50 * tt1
            - 6.915888e-08 * tt2 / 3.0
            + 2.698844e-11 * 0.25 * tt3
            + 4.097331e00
        )
        sr0 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.282538e00 * tt6
            + 1.483088e-03 * tt0
            - 7.579667e-07 * 0.50 * tt1
            + 2.094706e-10 * tt2 / 3.0
            - 2.167178e-14 * 0.25 * tt3
            + 5.453231e00
        )
        s_low = (
            3.782456e00 * tt6
            - 2.996734e-03 * tt0
            + 9.847302e-06 * 0.50 * tt1
            - 9.681295e-09 * tt2 / 3.0
            + 3.243728e-12 * 0.25 * tt3
            + 3.657676e00
        )
        sr1 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.857460e00 * tt6
            + 4.414370e-03 * tt0
            - 2.214814e-06 * 0.50 * tt1
            + 5.234902e-10 * tt2 / 3.0
            - 4.720842e-14 * 0.25 * tt3
            + 2.271638e00
        )
        s_low = (
            2.356774e00 * tt6
            + 8.984597e-03 * tt0
            - 7.123563e-06 * 0.50 * tt1
            + 2.459190e-09 * tt2 / 3.0
            - 1.436995e-13 * 0.25 * tt3
            + 9.901052e00
        )
        sr2 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.715186e00 * tt6
            + 2.062527e-03 * tt0
            - 9.988258e-07 * 0.50 * tt1
            + 2.300530e-10 * tt2 / 3.0
            - 2.036477e-14 * 0.25 * tt3
            + 7.818688e00
        )
        s_low = (
            3.579533e00 * tt6
            - 6.103537e-04 * tt0
            + 1.016814e-06 * 0.50 * tt1
            + 9.070059e-10 * tt2 / 3.0
            - 9.044245e-13 * 0.25 * tt3
            + 3.508409e00
        )
        sr3 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.033992e00 * tt6
            + 2.176918e-03 * tt0
            - 1.640725e-07 * 0.50 * tt1
            - 9.704199e-11 * tt2 / 3.0
            + 1.682010e-14 * 0.25 * tt3
            + 4.966770e00
        )
        s_low = (
            4.198641e00 * tt6
            - 2.036434e-03 * tt0
            + 6.520402e-06 * 0.50 * tt1
            - 5.487971e-09 * tt2 / 3.0
            + 1.771978e-12 * 0.25 * tt3
            - 8.490322e-01
        )
        sr4 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.337279e00 * tt6
            - 4.940247e-05 * tt0
            + 4.994568e-07 * 0.50 * tt1
            - 1.795664e-10 * tt2 / 3.0
            + 2.002554e-14 * 0.25 * tt3
            - 3.205023e00
        )
        s_low = (
            2.344331e00 * tt6
            + 7.980521e-03 * tt0
            - 1.947815e-05 * 0.50 * tt1
            + 2.015721e-08 * tt2 / 3.0
            - 7.376118e-12 * 0.25 * tt3
            + 6.830102e-01
        )
        sr5 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.926640e00 * tt6
            + 1.487977e-03 * tt0
            - 5.684760e-07 * 0.50 * tt1
            + 1.009704e-10 * tt2 / 3.0
            - 6.753351e-15 * 0.25 * tt3
            + 5.980528e00
        )
        s_low = (
            3.298677e00 * tt6
            + 1.408240e-03 * tt0
            - 3.963222e-06 * 0.50 * tt1
            + 5.641515e-09 * tt2 / 3.0
            - 2.444854e-12 * 0.25 * tt3
            + 3.950372e00
        )
        sr6 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        return make_obj_array([sr0, sr1, sr2, sr3, sr4, sr5, sr6])

    def get_species_gibbs_RT(self, T):
        h0_RT = self.get_species_enthalpies_RT(T)
        s0_R  = self.get_species_entropies_R(T)
        return h0_RT - s0_R

    def get_equilibrium_constants(self, T):
        RT = self.gas_constant * T
        C0 = self.npctx.log( self.one_atm / RT )

        g0_RT = self.get_species_gibbs_RT( T )
        return make_obj_array([
            %for react in sol.reactions():
                %if react.reversible:
                    ${cgm(equilibrium_constants_expr(
                        sol, react, Variable("g0_RT")))},
                %else:
                    -86*T,
                %endif
            %endfor
            ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy == False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = (1 + enthalpy_or_energy) - enthalpy_or_energy
        t_i = t_guess * ones

        for iter in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self.npctx.linalg.norm(dt, np.inf) < tol:
                break

        return t_i

    def get_falloff_rates(self, T, C, k_fwd):
        k_high = make_obj_array([
        %for react in falloff_reactions:
            ${cgm(rate_coefficient_expr(react.high_rate, Variable("T")))},
        %endfor
        ])

        k_low = make_obj_array([
        %for react in falloff_reactions:
            ${cgm(rate_coefficient_expr(react.low_rate, Variable("T")))},
        %endfor
        ])

        reduced_pressure = make_obj_array([
        %for i, react in enumerate(falloff_reactions):
            (${cgm(third_body_efficiencies_expr(
                sol, react, Variable("C")))})*k_low[${i}]/k_high[${i}],
        %endfor
        ])

        falloff_center = make_obj_array([
        %for react in falloff_reactions:
            %if react.falloff.falloff_type == "Troe":
            self.npctx.log10(${cgm(troe_falloff_expr(react, Variable("T")))}),
            %else:
            1,
            %endif
        %endfor
        ])

        falloff_function = make_obj_array([
        %for i, react in enumerate(falloff_reactions):
            ${cgm(falloff_function_expr(
                react, i, Variable("T"), Variable("reduced_pressure"),
                Variable("falloff_center")))},
        %endfor
        ])*reduced_pressure/(1+reduced_pressure)

        %for i, react in enumerate(falloff_reactions):
        k_fwd[${int(react.ID)-1}] = k_high[${i}]*falloff_function[${i}]
        %endfor
        return


    def get_fwd_rate_coefficients(self, T, C):
        k_fwd = make_obj_array([
        %for react in sol.reactions():
        %if isinstance(react, ct.FalloffReaction):
            0*T,
        %else:
            ${cgm(rate_coefficient_expr(react.rate, Variable("T")))},
        %endif
        %endfor
        ])
        %if falloff_reactions:
        self.get_falloff_rates(T, C, k_fwd)
        %endif

        %for react in three_body_reactions:
        k_fwd[${int(react.ID)-1}] *= (${cgm(third_body_efficiencies_expr(
            sol, react, Variable("C")))})
        %endfor

        return k_fwd

    def get_net_rates_of_progress(self, T, C):
        k_fwd = self.get_fwd_rate_coefficients(T, C)
        log_k_eq = self.get_equilibrium_constants(T)
        k_eq = self.npctx.exp(log_k_eq)
        return make_obj_array([
                %for react in sol.reactions():
                    ${cgm(rate_of_progress_expr(sol, react, Variable("C"),
                        Variable("k_fwd"), Variable("k_eq")))},
                %endfor
               ])

    def get_net_production_rates(self, rho, T, Y):
        C = self.get_concentrations(rho, Y)
        r_net = self.get_net_rates_of_progress(T, C)
        ones = r_net[0] + 1.0 - r_net[0]
        return make_obj_array([
            %for sp in sol.species():
                ${cgm(production_rate_expr(sol, sp.name, Variable("r_net")))} * ones,
            %endfor
            ])

""", strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution) -> str:
    return code_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        poly_to_expr=poly_to_expr,
        poly_to_enthalpy_expr=poly_to_enthalpy_expr,
        poly_to_entropy_expr=poly_to_entropy_expr,
        equilibrium_constants_expr=equilibrium_constants_expr,
        rate_coefficient_expr=rate_coefficient_expr,
        third_body_efficiencies_expr=third_body_efficiencies_expr,
        troe_falloff_expr=troe_falloff_expr,
        falloff_function_expr=falloff_function_expr,
        rate_of_progress_expr=rate_of_progress_expr,
        production_rate_expr=production_rate_expr,

        falloff_reactions=list(compress(sol.reactions(),
                                        [isinstance(r, ct.FalloffReaction)
                                         for r in sol.reactions()])),
        non_falloff_reactions=list(compress(sol.reactions(),
                                            [not isinstance(r, ct.FalloffReaction)
                                             for r in sol.reactions()])),
        three_body_reactions=list(compress(sol.reactions(),
                                           [isinstance(r, ct.ThreeBodyReaction)
                                            for r in sol.reactions()])),
    )


def compile_class(code_str, class_name="Thermochemistry"):
    exec_dict = {}
    exec(compile(code_str, "<generated code>", "exec"), exec_dict)
    exec_dict["_MODULE_SOURCE_CODE"] = code_str

    return exec_dict[class_name]


def get_thermochem_class(sol: ct.Solution):
    return compile_class(gen_thermochem_code(sol))


# vim: foldmethod=marker

#    def get_concentrations(self, rho, Y):
#        return self.iwts * rho * Y
#
#    def get_mixture_specific_heat_cp_mass(self, T, Y):
#        return self.gas_constant * sum(
#            self.get_species_specific_heats_R(T)* Y * self.iwts)
#
#    def get_mixture_specific_heat_cv_mass(self, T, Y):
#        cp0_R = self.get_species_specific_heats_R( T ) - 1.0
#        return self.gas_constant * sum(Y * cp0_R * self.iwts)
#
#    def get_mixture_enthalpy_mass(self, T, Y):
#        h0_RT = self.get_species_enthalpies_RT( T )
#        return self.gas_constant * T * sum(Y * h0_RT * self.iwts)
#
#    def get_mixture_internal_energy_mass(self, T, Y):
#        e0_rt = self.get_species_enthalpies_RT( T ) - 1.0
#        esum = sum([Y[i] * e0_rt[i] * self.iwts[i]
#                    for i in range(self.num_species)])
#        return self.gas_constant * T * esum
