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

"""
Internal Functionality
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: equilibrium_constants_expr
.. autofunction:: rate_coefficient_expr
.. autofunction:: third_body_efficiencies_expr
.. autofunction:: troe_falloff_expr
.. autofunction:: falloff_function_expr
.. autofunction:: rate_of_progress_expr
.. autofunction:: production_rate_expr
"""

import pymbolic.primitives as p
from functools import singledispatch
import cantera as ct
import numpy as np


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
    

@singledispatch
def poly_deriv_to_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_deriv_to_expr: "
            f"{type(poly)}")


@poly_deriv_to_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7        
        return c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3

    return nasa7_conditional(p.Variable(arg_name), poly, gen)
        
    

@singledispatch
def poly_deriv_to_enthalpy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_deriv_to_enthalpy_expr: "
            f"{type(poly)}")


@poly_deriv_to_enthalpy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return (
            c[1] / 2
            + 2 * c[2] / 3 * t
            + 3 * c[3] / 4 * t ** 2
            + 4 * c[4] / 5 * t ** 3
            - c[5] / (t ** 2)
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


@singledispatch
def poly_deriv_to_entropy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_deriv_to_entropy_expr: "
            f"{type(poly)}")


@poly_deriv_to_entropy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return (
            c[0] / t
            + c[1]
            + c[2] * t
            + c[3] * t ** 2
            + c[4] * t ** 3
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)

# }}}


# {{{ Data-handling helper

def _zeros_like(argument):
    # FIXME: This mishandles NaNs.
    return 0 * argument

# }}}

# {{{ Equilibrium constants


def equilibrium_constants_expr(sol: ct.Solution, react: ct.Reaction, gibbs_rt):
    """Generate code for equilibrium constants.

    :returns: Equilibrium constant expression for reaction *react* in terms of
        the species Gibbs functions *gibbs_rt* as a
        :class:`pymbolic.primitives.Expression`
    """

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
    if sum_nu_net != 0:
        return sum_p - sum_r - sum_nu_net*p.Variable("c0")    
    else:
        return sum_p - sum_r

# }}}


# {{{ Rate coefficients

def rate_coefficient_expr(rate_coeff: ct.Arrhenius, t):
    """
    :returns: The rate coefficient expression for *rate_coeff* in terms
        of the temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    # Rate parameters
    a = rate_coeff.pre_exponential_factor
    b = rate_coeff.temperature_exponent
    t_a = rate_coeff.activation_energy/ct.gas_constant
    if t_a == 0:
        # Weakly temperature-dependent rate
        return a * t**b
    else:
        # Modified Arrhenius
        return p.Variable("exp")(np.log(a)+b*p.Variable("log")(t)-t_a/t)


def third_body_efficiencies_expr(sol: ct.Solution, react: ct.Reaction, c):
    """
    :returns: The third-body concentration expression for reaction *react* in terms
        of the species concentrations *c* as a
        :class:`pymbolic.primitives.Expression`
    """
    efficiencies = [react.efficiencies[sp] for sp in react.efficiencies]
    indices_nondef = [sol.species_index(sp) for sp in react.efficiencies]
    indices_default = [i for i in range(sol.n_species) if i not in indices_nondef]
    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(np.array(efficiencies), indices_nondef))
    sum_default = react.default_efficiency * sum(c[i] for i in indices_default)
    return sum_nondef + sum_default


def troe_falloff_expr(react: ct.Reaction, t):
    """
    :returns: The Troe falloff center expression for reaction *react* in terms of the
        temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    troe_params = react.falloff.parameters
    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[1])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[2])
    if troe_params[3]:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return troe_1 + troe_2 + troe_3
    else:
        return troe_1 + troe_2


def falloff_function_expr(react: ct.Reaction, i, t, red_pressure, falloff_center):
    """
    :returns: Falloff function expression for reaction *react* in terms
        of the temperature *t*, reduced pressure *red_pressure*, and falloff center
        *falloff_center* as a :class:`pymbolic.primitives.Expression`
    """
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

def rate_of_progress_expr(sol: ct.Solution, react: ct.Reaction, c, k_fwd, log_k_eq):
    """
    :returns: Rate of progress expression for reaction *react* in terms of
        species concentrations *c* with rate coefficients *k_fwd* and equilbrium
        constants *k_eq* as a :class:`pymbolic.primitives.Expression`
    """
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
        # FIXME: It's not clear that this is available other than by this clunky,
        # string-parsing route
        reaction_index = int(react.ID)-1
        return k_fwd[reaction_index] * (r_fwd -
                                        p.Variable("exp")(log_k_eq[reaction_index]) * r_rev)
    else:
        return k_fwd[int(react.ID)-1] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """
    :returns: Species production rate for species *species* in terms of
        the net reaction rates of progress *r_net* as a
        :class:`pymbolic.primitives.Expression`
    """
    ones = _zeros_like(r_net[0]) + 1.0
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

# vim
