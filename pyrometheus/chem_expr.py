__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
.. autofunction:: viscosity_polynomial_expr
.. autofunction:: conductivity_polynomial_expr
.. autofunction:: diffusivity_polynomial_expr
.. autofunction:: viscosity_mixture_rule_wilke_expr
.. autofunction:: diffusivity_mixture_rule_denom_expr
.. autofunction:: equilibrium_constants_expr
.. autofunction:: rate_coefficient_expr
.. autofunction:: third_body_efficiencies_expr
.. autofunction:: troe_falloff_center_expr
.. autofunction:: troe_falloff_factor_expr
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


@poly_to_enthalpy_expr.register
def _(poly: ct.ConstantCp, arg_name):
    """Constant heat capacity: h(T) = h0 + cp0*(T - T0), normalised by RT.

    Common for condensed-phase and surface-site species, whose thermo is often
    given as a single reference point instead of a NASA fit.
    """
    t = p.Variable(arg_name)
    t0, h0, _s0, cp0 = poly.coeffs
    return (h0 + cp0*(t - t0))/(ct.gas_constant*t)


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


@poly_to_entropy_expr.register
def _(poly: ct.ConstantCp, arg_name):
    """Constant heat capacity: s(T) = s0 + cp0*ln(T/T0), normalised by R."""
    t = p.Variable(arg_name)
    t0, _h0, s0, cp0 = poly.coeffs
    return (s0 + cp0*p.Variable("log")(t/t0))/ct.gas_constant


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


# {{{ Transport polynomials & mixture rules

def viscosity_polynomial_expr(c, t):
    """Generate code for viscosity polynomials

    :returns: Viscosity polynomial expression with coefficients c in terms of
    the temperature t as a :class:`pymbolic.primitives.Expression`.
    """
    assert len(c) == 5
    return (
        p.Variable("sqrt")(t) * (
            c[0]
            + c[1] * p.Variable("log")(t)
            + c[2] * p.Variable("log")(t) ** 2
            + c[3] * p.Variable("log")(t) ** 3
            + c[4] * p.Variable("log")(t) ** 4
        )**2
    )


def conductivity_polynomial_expr(c, t):
    """Generate code for conductivity polynomials

    :returns: Conductivity polynomial expression with coefficients c in terms
    of the temperature t as a :class:`pymbolic.primitives.Expression`.
    """
    assert len(c) == 5
    return (
        p.Variable("sqrt")(t) * (
            c[0]
            + c[1] * p.Variable("log")(t)
            + c[2] * p.Variable("log")(t) ** 2
            + c[3] * p.Variable("log")(t) ** 3
            + c[4] * p.Variable("log")(t) ** 4
        )
    )


def diffusivity_polynomial_expr(c, t):
    """Generate code for diffusivity polynomials

    :returns: Diffusivity polynomial expression with coefficients c in terms
    of the temperature t as a :class:`pymbolic.primitives.Expression`.
    """
    assert len(c) == 5
    return (
        p.Variable("sqrt")(t) * t * (
            c[0]
            + c[1] * p.Variable("log")(t)
            + c[2] * p.Variable("log")(t) ** 2
            + c[3] * p.Variable("log")(t) ** 3
            + c[4] * p.Variable("log")(t) ** 4
        )
    )


def viscosity_mixture_rule_wilke_expr(sol: ct.Solution, sp, x, mu):
    """Generate code for species mixture rule. See [Kee_2003]_, chapter 12.

    :returns: Expression for the Wilke viscosity mixture rule
        for species *sp* in terms of species mole fractions *w*
        and viscosities *mu* as a :class:`pymbolic.primitives.Expression`
    """
    w = sol.molecular_weights
    sqrt = p.Variable("sqrt")
    return sum([x[j]*(
        1 + sqrt((mu[sp]/mu[j])*np.sqrt(w[j]/w[sp]))
    )**2 / np.sqrt(
        8*(1 + (w[sp]/w[j]))
    ) for j in range(sol.n_species)])


def diffusivity_mixture_rule_denom_expr(sol: ct.Solution, j_sp, x, bdiff):
    """ See [Kee_2003]_, chapter 12 for details.
    :returns: The denominator expression to the mixture rule
    for mixture-averaged species diffusivities in terms
    of the species mole fractions *x* and binary diffusivities *bdiff* as a
    :class:`pymbolic.primitives.Expression`
    """
    return sum(x[i_sp] / bdiff[i_sp][j_sp] for i_sp in range(sol.n_species))

# }}}


# {{{ Equilibrium constants

def equilibrium_constants_expr(sol: ct.Solution, reaction_index, gibbs_rt):
    """Generate code for equilibrium constants.

    :returns: Equilibrium constant expression for reaction with
        index *reaction_index* in terms of the species Gibbs
        functions *gibbs_rt* as a :class:`pymbolic.primitives.Expression`
    """
    indices_reac = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).reactants]
    indices_prod = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).products]

    # Stoichiometric coefficients
    nu_reac = [sol.reactant_stoich_coeff(sol.species_index(sp), reaction_index)
               for sp in sol.reaction(reaction_index).reactants]
    nu_prod = [sol.product_stoich_coeff(sol.species_index(sp), reaction_index)
               for sp in sol.reaction(reaction_index).products]

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
    :returns: The third-body concentration expression for reaction *react* in
    terms of the species concentrations *c* as a
        :class:`pymbolic.primitives.Expression`
    """

    efficiencies = [react.third_body.efficiencies[sp]
                    for sp in react.third_body.efficiencies]
    indices_nondef = [sol.species_index(sp) for sp
                      in react.third_body.efficiencies]
    indices_default = [i for i in range(sol.n_species)
                       if i not in indices_nondef]
    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(np.array(efficiencies), indices_nondef))
    sum_default = react.third_body.default_efficiency * sum(
        c[i] for i in indices_default
    )
    return sum_nondef + sum_default


def troe_falloff_center_expr(react: ct.Reaction, t):
    """
    :returns: The Troe falloff center expression for reaction *react* in
    terms of the temperature *t* as a
    :class:`pymbolic.primitives.Expression`
    """

    if isinstance(react.rate, ct.TroeRate):
        troe_params = react.rate.falloff_coeffs
    elif isinstance(react.rate, ct.LindemannRate):
        return 1
    else:
        raise ValueError("Unexpected value of 'rate.type': "
                         f" '{react.rate.type}'")

    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[1])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[2])
    if len(troe_params) == 3:
        return p.Variable("log10")(troe_1 + troe_2)
    elif len(troe_params) == 4:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return p.Variable("log10")(troe_1 + troe_2 + troe_3)
    else:
        raise ValueError("Unexpected length of 'troe_params': "
                         f" '{len(troe_params)}'")
    return


def troe_falloff_factor_expr(react: ct.Reaction, i,
                             red_pressure, falloff_center):
    """
    :returns: The Troe falloff factor expression for reaction
    *react* in terms of reduced pressure *red_pressure* and the
    falloff center *falloff_center* as a
    :class:`pymbolic.primitives.Expression`

    """
    if isinstance(react.rate, ct.TroeRate):
        log_rp = p.Variable("log10")(red_pressure[i])
        c = -0.4 - 0.67 * falloff_center[i]
        n = 0.75 - 1.27 * falloff_center[i]
        return p.If(
            p.Comparison(red_pressure[i], ">", 0),
            (log_rp + c) / (n - 0.14 * (log_rp + c)),
            -1/0.14
        )
    elif isinstance(react.rate, ct.LindemannRate):
        return 0
    else:
        raise ValueError("Unexpected value of 'rate.type': "
                         f" '{react.rate.type}'")


def falloff_function_expr(react: ct.Reaction, i,
                          falloff_factor, falloff_center):
    """
    :returns: Falloff function expression for reaction *react* in
    terms of the temperature *t*, falloff width factor
    *falloff_factor*, and falloff center *falloff_center* as a
    :class:`pymbolic.primitives.Expression`

    """

    falloff_type = react.reaction_type.split("-")[1]

    if falloff_type == "Troe":
        return 10**(
            falloff_center[i] / (1+falloff_factor[i]**2)
        )
    elif falloff_type == "Lindemann":
        return 1
    else:
        raise ValueError("Unexpected value of 'falloff_type': "
                         f" '{falloff_type}'")

# }}}


# {{{ Rates of progress

def rate_of_progress_expr(sol: ct.Solution, reaction_index, c,
                          k_fwd, log_k_eq):
    """
    :returns: Rate of progress expression for reaction with
    index *reaction_index* in terms of species concentrations *c*
    with rate coefficients *k_fwd* and equilbrium constants *k_eq*
    as a :class:`pymbolic.primitives.Expression`
    """
    indices_reac = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).reactants]
    indices_prod = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).products]

    if sol.reaction(reaction_index).orders:
        nu_reac = [sol.reaction(reaction_index).orders[sp]
                   for sp in sol.reaction(reaction_index).orders]
    else:
        nu_reac = [sol.reaction(reaction_index).reactants[sp]
                   for sp in sol.reaction(reaction_index).reactants]

    r_fwd = np.prod([c[index]**nu for index, nu in zip(indices_reac, nu_reac)])

    if sol.reaction(reaction_index).reversible:
        nu_prod = [sol.reaction(reaction_index).products[sp]
                   for sp in sol.reaction(reaction_index).products]
        r_rev = np.prod([
            c[index]**nu for index, nu in zip(indices_prod, nu_prod)
        ])
        return k_fwd[reaction_index] * (
                r_fwd
                - p.Variable("exp")(log_k_eq[reaction_index]) * r_rev)
    else:
        return k_fwd[reaction_index] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """
    :returns: Species production rate for species *species* in terms of
        the net reaction rates of progress *r_net* as a
        :class:`pymbolic.primitives.Expression`
    """
    ones = _zeros_like(r_net[0]) + 1.0
    indices_fwd = [i for i, react in enumerate(sol.reactions())
                   if species in react.reactants]
    indices_rev = [i for i, react in enumerate(sol.reactions())
                   if species in react.products]
    nu_fwd = [
        sol.reactant_stoich_coeff(
            sol.species_index(species), react_index
        ) for react_index in indices_fwd
    ]
    nu_rev = [sol.product_stoich_coeff(sol.species_index(species), prod_index)
              for prod_index in indices_rev]
    sum_fwd = sum(nu*r_net[index] for nu, index in zip(nu_fwd, indices_fwd))
    sum_rev = sum(nu*r_net[index] for nu, index in zip(nu_rev, indices_rev))
    return (sum_rev - sum_fwd) * ones


def fwd_rate_of_progress_expr(sol: ct.Solution, reaction_index, c, k_fwd):
    """
    :returns: Forward rate of progress (non-negative) for reaction
        *reaction_index*, i.e. the forward part of
        :func:`rate_of_progress_expr`.
    """
    indices_reac = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).reactants]
    if sol.reaction(reaction_index).orders:
        nu_reac = [sol.reaction(reaction_index).orders[sp]
                   for sp in sol.reaction(reaction_index).orders]
    else:
        nu_reac = [sol.reaction(reaction_index).reactants[sp]
                   for sp in sol.reaction(reaction_index).reactants]
    r_fwd = np.prod([c[index]**nu for index, nu in zip(indices_reac, nu_reac)])
    return k_fwd[reaction_index] * r_fwd


def rev_rate_of_progress_expr(sol: ct.Solution, reaction_index, c,
                              k_fwd, log_k_eq):
    """
    :returns: Reverse rate of progress (non-negative) for reaction
        *reaction_index*; zero for irreversible reactions.
    """
    if not sol.reaction(reaction_index).reversible:
        return _zeros_like(c[0])
    indices_prod = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).products]
    nu_prod = [sol.reaction(reaction_index).products[sp]
               for sp in sol.reaction(reaction_index).products]
    r_rev = np.prod([c[index]**nu for index, nu in zip(indices_prod, nu_prod)])
    return k_fwd[reaction_index] \
        * p.Variable("exp")(log_k_eq[reaction_index]) * r_rev


def _species_reaction_stoich(sol: ct.Solution, species):
    """
    :returns: ``(idx_reactant, idx_product, nu_reactant, nu_product)``: the
        reaction indices in which *species* appears as a reactant and as a
        product, with the matching stoichiometric coefficients. Shared by the
        creation/destruction splits, which differ only in how these are summed.
    """
    si = sol.species_index(species)
    idx_reactant = [i for i, react in enumerate(sol.reactions())
                    if species in react.reactants]
    idx_product = [i for i, react in enumerate(sol.reactions())
                   if species in react.products]
    nu_reactant = [sol.reactant_stoich_coeff(si, i) for i in idx_reactant]
    nu_product = [sol.product_stoich_coeff(si, i) for i in idx_product]
    return idx_reactant, idx_product, nu_reactant, nu_product


def creation_rate_expr(sol: ct.Solution, species, r_fwd, r_rev):
    """
    :returns: Species creation rate for *species*: created as a product by
        forward reactions and as a reactant by reverse reactions. Mirrors
        Cantera's ``creation_rates``.
    """
    ones = _zeros_like(r_fwd[0]) + 1.0
    idx_reactant, idx_product, nu_reactant, nu_product = \
        _species_reaction_stoich(sol, species)
    made = sum(nu*r_fwd[i] for nu, i in zip(nu_product, idx_product)) \
        + sum(nu*r_rev[i] for nu, i in zip(nu_reactant, idx_reactant))
    return made * ones


def destruction_rate_expr(sol: ct.Solution, species, r_fwd, r_rev):
    """
    :returns: Species destruction rate for *species*: consumed as a reactant by
        forward reactions and as a product by reverse reactions. Mirrors
        Cantera's ``destruction_rates``. creation - destruction == net.
    """
    ones = _zeros_like(r_fwd[0]) + 1.0
    idx_reactant, idx_product, nu_reactant, nu_product = \
        _species_reaction_stoich(sol, species)
    lost = sum(nu*r_fwd[i] for nu, i in zip(nu_reactant, idx_reactant)) \
        + sum(nu*r_rev[i] for nu, i in zip(nu_product, idx_product))
    return lost * ones

# }}}


# {{{ surface (heterogeneous) kinetics

# Sticking and Blowers-Masel are siblings in Cantera, not subclasses, so the
# isinstance check is exact. StickingArrheniusRate is not an
# InterfaceArrheniusRate and needs its own entry.
_SUPPORTED_SURFACE_RATES = (
    ct.ArrheniusRate, ct.InterfaceArrheniusRate, ct.StickingArrheniusRate)


def surface_rate_coefficient_expr(interface: ct.Interface, react: ct.Reaction, t,
                                  coverages=None):
    """
    :returns: The forward rate coefficient expression for the heterogeneous
        reaction *react* on *interface*, in terms of the temperature *t* and,
        where the rate depends on them, the surface *coverages*, as a
        :class:`pymbolic.primitives.Expression`.

    Three rate forms are handled, all verified against
    :attr:`cantera.Interface.forward_rate_constants`:

    * ``interface-Arrhenius``: the modified Arrhenius form, as in the gas phase.

    * ``sticking-Arrhenius``: the Arrhenius parameters give a dimensionless
      sticking probability :math:`\\gamma`, not a rate coefficient. The coefficient
      follows from kinetic theory,

      .. math:: k = \\frac{\\gamma}{\\Gamma_0^n}\\sqrt{\\frac{RT}{2\\pi W}},

      with :math:`\\Gamma_0` the site density, *n* the sticking order and *W* the
      molar mass of the sticking species. With the Motz-Wise correction the
      leading factor is :math:`\\gamma/(1 - \\gamma/2)`, which matters once
      :math:`\\gamma` is no longer small.

    * coverage dependence: any of the above multiplied by

      .. math:: 10^{a\\theta_k}\\,\\theta_k^{m}\\,e^{-E\\theta_k/RT}

      for each species *k* the rate declares a dependence on.
    """
    rate = react.rate

    # Allowlist rather than a hasattr check: every Cantera surface rate class
    # exposes pre_exponential_factor, temperature_exponent and activation_energy,
    # so an unsupported rate would read as a plausible Arrhenius one. A
    # Blowers-Masel activation energy is an intrinsic barrier that Cantera shifts
    # by the reaction enthalpy at run time, and its sticking variant is a
    # StickRateBase, which the sticking branch below would otherwise accept.
    if not isinstance(rate, _SUPPORTED_SURFACE_RATES):
        raise ValueError(
            f"reaction '{react.equation}' uses rate type '{rate.type}', which "
            "heterogeneous kinetics does not handle; only interface-Arrhenius and "
            "sticking-Arrhenius rates are translated")

    t_a = rate.activation_energy/ct.gas_constant

    # Sticking probability, or the rate coefficient for a plain rate.
    if t_a == 0:
        base = rate.pre_exponential_factor * t**rate.temperature_exponent
    else:
        base = p.Variable("exp")(
            np.log(rate.pre_exponential_factor)
            + rate.temperature_exponent*p.Variable("log")(t) - t_a/t)

    if isinstance(rate, ct.StickRateBase):
        if rate.motz_wise_correction:
            base = base/(1 - base/2)
        flux = p.Variable("sqrt")(
            (ct.gas_constant/(2*np.pi*rate.sticking_weight)) * t)
        base = base * flux / interface.site_density**rate.sticking_order

    coverage = dict(getattr(rate, "coverage_dependencies", {}) or {})
    for name, dep in coverage.items():
        if coverages is None:
            raise ValueError(
                f"reaction '{react.equation}' has a coverage-dependent rate, so "
                "surface_rate_coefficient_expr needs the coverages")
        theta = coverages[interface.species_index(name)]
        if dep["a"]:
            base = base * 10**(dep["a"]*theta)
        if dep["m"]:
            base = base * theta**dep["m"]
        if dep["E"]:
            base = base * p.Variable("exp")(-(dep["E"]/ct.gas_constant)*theta/t)

    return base


def surface_equilibrium_constant_expr(interface: ct.Interface, reaction_index,
                                      g0_rt):
    """
    :returns: Log of the equilibrium constant for heterogeneous reaction
        *reaction_index*, in terms of the standard-state Gibbs energies *g0_rt* of
        every species the interface couples, as a
        :class:`pymbolic.primitives.Expression`.

    Each species carries its own phase's standard concentration,
    :math:`\\log K = -\\Delta G/RT + \\sum_k \\nu_k \\log c^{0}_{k}`, and the three
    phases an interface couples do not share one: a gas species has
    :math:`p_0/RT`, a surface species occupying :math:`\\sigma_k` sites has
    :math:`\\Gamma_0/\\sigma_k`, and a bulk species has unit activity and so
    contributes nothing. Written in logs, so the caller exponentiates once.
    """
    d_g = sum(
        nu*g0_rt[k]
        for k, nu in enumerate(_surface_net_stoich(interface, reaction_index))
        if nu != 0)

    expr = -d_g
    dn_gas, log_c0_surface = _surface_standard_concentrations(
        interface, reaction_index)
    if dn_gas:
        expr = expr + dn_gas*p.Variable("c0")
    if log_c0_surface:
        expr = expr + log_c0_surface
    return expr


def _surface_net_stoich(interface: ct.Interface, reaction_index):
    """Net stoichiometric coefficient of every kinetics species in a reaction."""
    return [interface.product_stoich_coeff(k, reaction_index)
            - interface.reactant_stoich_coeff(k, reaction_index)
            for k in range(interface.n_total_species)]


# Length dimension of the standard concentration: kmol/m^3 in a volume,
# kmol/m^2 on a surface, dimensionless in a bulk solid. Classifying on the
# thermo model would mean enumerating every model name Cantera has.
_PHASE_KIND_BY_LENGTH_DIMENSION = {-3.0: "gas", -2.0: "surface", 0.0: "bulk"}


def surface_phase_kind(phase):
    """Which of ``"gas"``, ``"surface"`` or ``"bulk"`` *phase* behaves as.

    Raises for anything else, such as a one-dimensional edge phase.
    """
    units = phase.standard_concentration_units
    try:
        return _PHASE_KIND_BY_LENGTH_DIMENSION[units.dimensions["length"]]
    except KeyError:
        raise ValueError(
            f"phase '{phase.name}' has standard concentration units '{units}', "
            "which heterogeneous kinetics does not handle") from None


def _surface_kinetics_phases(interface: ct.Interface):
    """The interface and its adjacent phases, paired with their kinetics ranges.

    The interface's own species occupy the first n_species slots of the kinetics
    ordering and the adjacent phases follow, in the order Cantera lists them.
    """
    phases = [interface, *interface.adjacent.values()]

    offset = 0
    for phase in phases:
        yield phase, offset
        offset += phase.n_species

    assert offset == interface.n_total_species


def surface_phase_blocks(interface: ct.Interface):
    """Where the standard-state Gibbs energy of each kinetics species comes from.

    An equilibrium constant needs a Gibbs energy for every species the interface
    couples, but they do not all come from the same place: the surface species and
    any bulk species are generated with the surface code, while the gas-phase ones
    come from the separately generated gas class. This returns one block per phase,
    in kinetics order, as ``(kind, start, stop, source_start)``: *kind* is
    ``"surface"``, ``"gas"`` or ``"bulk"``, ``start:stop`` the block's span in the
    kinetics ordering, and *source_start* its offset within its own source array,
    which differs from *start* only for bulk phases, since they share one array
    while not necessarily being adjacent in the kinetics ordering.

    Raises if the interface couples more than one gas phase, which the generated
    code has no way to name.
    """
    blocks = []
    bulk_offset = 0
    n_gas_phases = 0

    for phase, offset in _surface_kinetics_phases(interface):
        stop = offset + phase.n_species

        if phase is interface:
            blocks.append(("surface", offset, stop, 0))
        elif surface_phase_kind(phase) == "gas":
            n_gas_phases += 1
            blocks.append(("gas", offset, stop, 0))
        else:
            blocks.append(("bulk", offset, stop, bulk_offset))
            bulk_offset += phase.n_species

    if n_gas_phases > 1:
        raise ValueError(
            f"interface '{interface.name}' couples {n_gas_phases} gas phases; the "
            "generated code takes a single gas-phase class")

    return blocks


def _surface_needs_gas_standard_concentration(interface: ct.Interface):
    """Whether any equilibrium constant of *interface* carries a :math:`p_0/RT`.

    A reversible reaction that leaves the moles of gas unchanged does not, so a
    mechanism whose reversible reactions are all of that kind never references
    the factor.
    """
    return any(
        react.reversible
        and _surface_standard_concentrations(interface, i)[0]
        for i, react in enumerate(interface.reactions()))


def surface_bulk_species(interface: ct.Interface):
    """Species of the interface's bulk phases, in the order the blocks expect.

    Their thermodynamics is generated with the surface code: unlike the gas phase,
    a bulk phase has no Pyrometheus class of its own to defer to.
    """
    return [sp
            for phase, _offset in _surface_kinetics_phases(interface)
            if phase is not interface and surface_phase_kind(phase) != "gas"
            for sp in phase.species()]


def _surface_standard_concentrations(interface: ct.Interface, reaction_index):
    """Standard-concentration contribution of a reaction, split by phase kind.

    :returns: the net change in moles of gas-phase species, whose standard
        concentration is temperature-dependent and so stays symbolic, and the log
        of the surface contribution, which is a number.

    Bulk species have unit activity and drop out; lumping them in with the gas
    species would leave a spurious factor of :math:`(p_0/RT)^{\\Delta n_b}`, which
    is two orders of magnitude per mole of bulk at combustion temperatures.
    """
    nu = _surface_net_stoich(interface, reaction_index)

    dn_gas = 0.0
    log_c0_surface = 0.0

    for phase, offset in _surface_kinetics_phases(interface):
        kind = surface_phase_kind(phase)
        nu_phase = nu[offset:offset + phase.n_species]

        if kind == "gas":
            dn_gas += sum(nu_phase)
        elif kind == "surface":
            # A species occupying several sites has a proportionally smaller
            # standard concentration, so the factor is per species.
            log_c0_surface += sum(
                nu_k*np.log(phase.site_density/phase.species(k).size)
                for k, nu_k in enumerate(nu_phase) if nu_k != 0)

    return dn_gas, log_c0_surface


def surface_concentrations_expr(interface: ct.Interface, coverages):
    """
    :returns: Site concentration of every surface species, as a list of
        :class:`pymbolic.primitives.Expression`.

    A species occupying *size* sites is present at ``coverage*site_density/size``.
    """
    return [coverages[k]*(interface.site_density/interface.species(k).size)
            for k in range(interface.n_species)]


def surface_rate_of_progress_expr(interface: ct.Interface, reaction_index, k_fwd,
                                  concentrations):
    """
    :returns: Forward rate of progress of heterogeneous reaction *reaction_index*,
        in terms of its rate coefficient *k_fwd* and the concentrations of every
        species the interface couples, as a
        :class:`pymbolic.primitives.Expression`.

    *concentrations* is in the interface's kinetics ordering: its own surface
    species first, then those of the adjacent phases. They are activity
    concentrations: molar concentration for a gas species, coverage*site_density
    /size for a surface one, and an activity of one for a pure solid. This keeps
    a rate of progress dimensionally coherent across phases of different
    dimensionality. Explicit reaction orders, where a mechanism gives them,
    override the reactant stoichiometry.
    """
    reaction = interface.reaction(reaction_index)
    orders = dict(reaction.orders)

    expr = k_fwd
    for k in range(interface.n_total_species):
        order = orders.get(interface.kinetics_species_name(k),
                           interface.reactant_stoich_coeff(k, reaction_index))
        if order:
            expr = expr * concentrations[k]**order
    return expr


def surface_reverse_rate_of_progress_expr(interface: ct.Interface, reaction_index,
                                          k_fwd, k_eq, concentrations):
    """
    :returns: Reverse rate of progress of heterogeneous reaction *reaction_index*,
        as a :class:`pymbolic.primitives.Expression`.

    The reverse coefficient is the forward one over the equilibrium constant, and
    the concentration product runs over the products rather than the reactants.
    Explicit reaction orders are a property of the forward direction only, so they
    do not appear here.
    """
    expr = k_fwd/k_eq
    for k in range(interface.n_total_species):
        order = interface.product_stoich_coeff(k, reaction_index)
        if order:
            expr = expr * concentrations[k]**order
    return expr


def surface_production_rate_expr(interface: ct.Interface, species, r_net):
    """
    :returns: Production rate of *species* from the heterogeneous reactions of
        *interface*, in terms of the net rates of progress *r_net*, as a
        :class:`pymbolic.primitives.Expression`.

    *species* may be a gas-phase, surface-site or bulk species: the interface's
    own stoichiometry covers all the phases it couples, and a species absent from
    every reaction yields zero.
    """
    ones = _zeros_like(r_net[0]) + 1.0
    index = interface.kinetics_species_index(species)

    nu = [interface.product_stoich_coeff(index, i)
          - interface.reactant_stoich_coeff(index, i)
          for i in range(interface.n_reactions)]

    terms = [coeff*r_net[i] for i, coeff in enumerate(nu) if coeff != 0]
    if not terms:
        return 0.0 * ones

    return sum(terms) * ones

# }}}

# vim
