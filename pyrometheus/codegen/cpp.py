"""
C++ code generation
----------------------

.. autofunction:: gen_thermochem_code
"""


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

import pymbolic.primitives as p
from pymbolic.mapper.c_code import CCodeMapper
import cantera as ct
import numpy as np  # noqa: F401

from itertools import compress
from mako.template import Template
import pyrometheus.chem_expr


file_extension = "hpp"


# {{{ code generation helpers

class CodeGenerationMapper(CCodeMapper):
    def map_constant(self, expr, enclosing_prec):
        return repr(expr)


def str_np_inner(ary):
    if isinstance(ary, Number):
        return repr(ary)
    elif ary.shape:
        return "[%s]" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return ", ".join(repr(entry) for entry in ary)

# }}}


# {{{ main code template

header_tpl = Template("""
#include <cmath>


template <typename ContainerT, typename ScalarT>
struct thermochemistry
{
    typedef ContainerT container_type;
    typedef ScalarT scalar_type;

    constexpr static int num_species = ${sol.n_species};
    constexpr static int num_reactions = ${sol.n_reactions};
    constexpr static int num_falloff = ${
        sum(1 if isinstance(r, ct.FalloffReaction) else 0
        for r in sol.reactions())};

    constexpr static ScalarT mol_weights[] = {${str_np(sol.molecular_weights)}};
    constexpr static ScalarT inv_weights[] = {${str_np(1/sol.molecular_weights)}};
    constexpr static ScalarT gas_constant = ${repr(ct.gas_constant)};
    constexpr static ScalarT one_atm = ${repr(ct.one_atm)};

    static ScalarT get_specific_gas_constant(ContainerT const &mass_fractions)
    {
        return gas_constant * (
                %for i in range(sol.n_species):
                    + inv_weights[${i}]*mass_fractions[${i}]
                %endfor
                );
    }

    static ScalarT get_mix_molecular_weight(ContainerT const &mass_fractions)
    {
        return 1.0/(
        %for i in range(sol.n_species):
            + inv_weights[${i}]*mass_fractions[${i}]
        %endfor
        );
    }

    static ContainerT get_concentrations(ScalarT rho, ContainerT const &mass_fractions)
    {
        ContainerT concentrations = {
            %for i in range(sol.n_species):
                inv_weights[${i}]*mass_fractions[${i}]*rho,
            %endfor
        };
        return concentrations;
    }

    static ScalarT get_mass_average_property(ContainerT const &mass_fractions, ContainerT const &spec_property)
    {
        return (
            %for i in range(sol.n_species):
                + mass_fractions[${i}]*spec_property[${i}]*inv_weights[${i}]
            %endfor
        );
    }

    static ScalarT get_mixture_specific_heat_cp_mass(ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT cp0_r = get_species_specific_heats_r(temperature);
        ScalarT cpmix = get_mass_average_property(mass_fractions, cp0_r);
        return cpmix;
    }

    static ScalarT get_mixture_enthalpy_mass(ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT h0_rt = get_species_enthalpies_rt(temperature);
        ScalarT hmix = get_mass_average_property(mass_fractions, h0_rt);
        return hmix;
    }

    static ScalarT get_density(ScalarT p, ScalarT temperature,
            ContainerT const &mass_fractions)
    {
        ScalarT mmw = get_mix_molecular_weight(mass_fractions);
        ScalarT rt = gas_constant * temperature;
        return p * mmw / rt;
    }

    static ContainerT get_species_specific_heats_r(ScalarT temperature)
    {
        ContainerT cp0_r = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return cp0_r;
    }

    static ContainerT get_species_enthalpies_rt(ScalarT temperature)
    {
        ContainerT h0_rt = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return h0_rt;
    }

    static ContainerT get_species_entropies_r(ScalarT temperature)
    {
        ContainerT s0_r = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_entropy_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return s0_r;
    }

    static ContainerT get_species_gibbs_rt(ScalarT temperature)
    {
        ContainerT h0_rt = get_species_enthalpies_rt(temperature);
        ContainerT s0_r = get_species_entropies_r(temperature);
        ContainerT g0_rt = {
        %for sp in range(sol.n_species):
        h0_rt[${sp}] - s0_r[${sp}],
        %endfor
        };
        return g0_rt;
    }

    static ContainerT get_equilibrium_constants(ScalarT temperature)
    {
        ScalarT rt = gas_constant * temperature;
        ScalarT c0 = std::log(one_atm/rt);
        ContainerT g0_rt = get_species_gibbs_rt(temperature);
        ContainerT k_eq = {
        %for react in sol.reactions():
        %if react.reversible:
        ${cgm(ce.equilibrium_constants_expr(sol, react, Variable("g0_rt")))},
        %else:
        -0.17364695002734*temperature,
        %endif
        %endfor
        };
        return k_eq;
    }

    static ScalarT get_temperature(ScalarT enthalpy, ScalarT t_guess,
                                   ContainerT const &mass_fractions)
    {
        int iter = 0;
        int num_iter = 500;
        double tol = 1.0e-06;
        ScalarT iter_temp = t_guess;

        for(int iter = 0; iter < num_iter; ++iter){
            ScalarT cp = get_mixture_specific_heat_cp_mass(iter_temp, mass_fractions);
            ScalarT h = get_mixture_enthalpy_mass(iter_temp, mass_fractions);
            ScalarT iter_rhs = enthalpy - h;
            iter_temp -= iter_rhs/cp;
            if(std::fabs(iter_rhs/cp) < tol){ break; }
        }
        return iter_temp;
    }

    static ContainerT get_fwd_rate_coefficients(ScalarT temperature, 
                                                ContainerT const &concentrations)
    {
        ContainerT k_fwd = {
        %for react in sol.reactions():
        %if isinstance(react, ct.FalloffReaction):
        0.0, 
        %else:
        ${cgm(ce.rate_coefficient_expr(react.rate, Variable("temperature")))},
        %endif
        %endfor
        };

        %for react in three_body_reactions:
        k_fwd[${int(react.ID)-1}] *= (${cgm(ce.third_body_efficiencies_expr(
        sol, react, Variable("concentrations")))});
        %endfor
        return k_fwd;
    }

    static ContainerT get_net_rates_of_progress(ScalarT rho, ScalarT temperature, ContainerT const &concentrations)
    {
        ContainerT k_fwd = get_fwd_rate_coefficients(temperature, concentrations);
        ContainerT log_k_eq = get_equilibrium_constants(temperature);
        ContainerT r_net = {
        %for react in sol.reactions():
        ${cgm(ce.rate_of_progress_expr(sol, react, Variable("concentrations"),
        Variable("k_fwd"), Variable("log_k_eq")))}, 
        %endfor
        };
        return r_net;
    }

    static ContainerT get_net_production_rates(ScalarT rho, ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT concentrations = get_concentrations(rho, mass_fractions);
        ContainerT r_net = get_net_rates_of_progress(rho, temperature, concentrations);
        ContainerT omega = {
        %for sp in sol.species():
        ${cgm(ce.production_rate_expr(sol, sp.name, Variable("r_net")))},
        %endfor
        };
        return omega;
    }
};
""", strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution) -> str:
    """For the mechanism given by *sol*, return Python source code for a class conforming
    to a module containing a class called ``Thermochemistry`` adhering to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
    return header_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        ce=pyrometheus.chem_expr,

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


# vim: foldmethod=marker
