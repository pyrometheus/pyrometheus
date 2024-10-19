"""
C++ code generation
----------------------

.. autofunction:: gen_thermochem_code
"""


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


from numbers import Number

import pymbolic.primitives as p
from pymbolic.mapper.c_code import CCodeMapper
import cantera as ct
import numpy as np  # noqa: F401

from mako.template import Template
import pyrometheus.chem_expr


file_extension = "hpp"


# {{{ code generation helpers

class CodeGenerationMapper(CCodeMapper):
    def map_constant(self, expr, enclosing_prec):
        if isinstance(expr, np.float64):
            return expr.astype(str)
        return super().map_constant(expr, enclosing_prec)


def str_np_inner(ary):
    if isinstance(ary, Number):
        return repr(ary)
    elif ary.shape:
        return "[%s]" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    def _mk_entry(entry):
        if isinstance(entry, np.float64):
            return entry.astype(str)

        return repr(entry)

    return ", ".join([_mk_entry(entry) for entry in ary])

# }}}


# {{{ main code template

header_tpl = Template("""
#include <cmath>
#include <string>

namespace ${namespace}{
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

    constexpr static const char* species_names[] = {
        ${", ".join(f'"{name}"' for name in sol.species_names)}
    };

    constexpr static const char* element_names[] = {
        ${", ".join(f'"{name}"' for name in sol.element_names)}
    };

    constexpr static ScalarT mol_weights[] = {${str_np(sol.molecular_weights)}};
    constexpr static ScalarT inv_weights[] = {${str_np(1/sol.molecular_weights)}};
    constexpr static ScalarT gas_constant = ${repr(ct.gas_constant)};
    constexpr static ScalarT one_atm = ${repr(ct.one_atm)};

    static int nSpecies() { return num_species; };

    static std::string get_species_name(int index) { return species_names[index]; }
    static std::string get_element_name(int index) { return element_names[index]; }

    static int get_species_index(const std::string& name) {
        %for i, sp in enumerate(sol.species()):
            if (name == "${sp.name}") return ${i};
        %endfor

        return -1;
    }

    static int get_element_index(const std::string& name) {
        %for i, el in enumerate(sol.element_names):
            if (name == "${el}") return ${i};
        %endfor

        return -1;
    }

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

    static ContainerT get_concentrations(
        ScalarT rho, ContainerT const &mass_fractions)
    {
        ContainerT concentrations = {
            %for i in range(sol.n_species):
                inv_weights[${i}]*mass_fractions[${i}]*rho,
            %endfor
        };
        return concentrations;
    }

    static ScalarT get_mass_average_property(
        ContainerT const &mass_fractions, ContainerT const &spec_property)
    {
        return (
            %for i in range(sol.n_species):
                + mass_fractions[${i}]*spec_property[${i}]*inv_weights[${i}]
            %endfor
        );
    }

    static ScalarT get_mixture_specific_heat_cv_mass(
        ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT cp0_r = get_species_specific_heats_r(temperature);

        for (int i = 0; i < num_species; ++i)
            cp0_r[i] -= 1.0;

        const ScalarT cpmix = get_mass_average_property(mass_fractions, cp0_r);

        return gas_constant * cpmix;
    }

    static ScalarT get_mixture_specific_heat_cp_mass(
        ScalarT temperature, ContainerT const &mass_fractions)
    {
        const ContainerT cp0_r = get_species_specific_heats_r(temperature);
        const ScalarT cpmix = get_mass_average_property(mass_fractions, cp0_r);
        return gas_constant * cpmix;
    }

    static ScalarT get_mixture_enthalpy_mass(
        ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT h0_rt = get_species_enthalpies_rt(temperature);
        ScalarT hmix = get_mass_average_property(mass_fractions, h0_rt);
        return gas_constant * hmix * temperature;
    }

    static ScalarT get_mixture_internal_energy_mass(
        ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT e0_rt = get_species_enthalpies_rt(temperature);

        for (int i = 0; i < num_species; ++i)
            e0_rt[i] -= 1.0;

        const ScalarT emix = get_mass_average_property(mass_fractions, e0_rt);
        return gas_constant * emix * temperature;
    }

    static ScalarT get_density(ScalarT p, ScalarT temperature,
            ContainerT const &mass_fractions)
    {
        ScalarT mmw = get_mix_molecular_weight(mass_fractions);
        ScalarT rt = gas_constant * temperature;
        return p * mmw / rt;
    }

    static ScalarT get_pressure(
        ScalarT density, ScalarT temperature, ContainerT const &mass_fractions)
    {
        const double mmw = get_mix_molecular_weight(mass_fractions);
        const double rt = gas_constant * temperature;
        return density * rt / mmw;
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
        %for i, react in enumerate(sol.reactions()):
        %if react.reversible:
        ${cgm(ce.equilibrium_constants_expr(sol, i, Variable("g0_rt")))},
        %else:
        -0.17364695002734*temperature,
        %endif
        %endfor
        };
        return k_eq;
    }

    static ScalarT get_temperature(ScalarT energy_or_enthalpy, ScalarT t_guess,
                                   ContainerT const &mass_fractions,
                                   bool const do_energy = true)
    {
        ScalarT (*pv_fun)(ScalarT, ContainerT const &);
        ScalarT (*he_fun)(ScalarT, ContainerT const &);

        if (do_energy) {
            pv_fun = get_mixture_specific_heat_cv_mass;
            he_fun = get_mixture_internal_energy_mass;
        } else {
            pv_fun = get_mixture_specific_heat_cp_mass;
            he_fun = get_mixture_enthalpy_mass;
        }

        int iter = 0;
        int num_iter = 500;
        double tol = 1.0e-06;
        ScalarT iter_temp = t_guess;

        for(int iter = 0; iter < num_iter; ++iter){
            auto iter_rhs   = energy_or_enthalpy - he_fun(iter_temp, mass_fractions);
            auto iter_deriv = -pv_fun(iter_temp, mass_fractions);
            auto iter_dt    = -iter_rhs/iter_deriv;
            iter_temp += iter_dt;
            if(std::fabs(iter_dt) < tol){ break; }
        }
        return iter_temp;
    }

    %if falloff_reactions:
    static void get_falloff_rates(
        ScalarT temperature, ContainerT const &concentrations, ContainerT &k_fwd)
    {
        ContainerT k_high = {
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.high_rate, Variable("temperature")))},
        %endfor
        };

        ContainerT k_low = {
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.low_rate, Variable("temperature")))},
        %endfor
        };

        ContainerT reduced_pressure = {
        %for i, (_, react) in enumerate(falloff_reactions):
            (${cgm(ce.third_body_efficiencies_expr(
                sol, react, Variable("concentrations")))})*k_low[${i}]/k_high[${i}],
        %endfor
        };

        ContainerT falloff_center = {
        %for _, react in falloff_reactions:
            ${cgm(ce.troe_falloff_center_expr(react, Variable("temperature")))},
        %endfor
        };

        ContainerT falloff_factor = {
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.troe_falloff_factor_expr(react, i,
            Variable("reduced_pressure"), Variable("falloff_center")))},
        %endfor
        };

        ContainerT falloff_function = {
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.falloff_function_expr(react, i,
            Variable("falloff_factor"), Variable("falloff_center")))},
        %endfor
        };

        %for j, (i, react) in enumerate(falloff_reactions):
            k_fwd[${i}] = k_high[${j}]*falloff_function[${j}] *
            reduced_pressure[${j}]/(1.0 + reduced_pressure[${j}]);
        %endfor
    };
    %endif

    static ContainerT get_fwd_rate_coefficients(ScalarT temperature,
                                                ContainerT const &concentrations)
    {
        ContainerT k_fwd = {
        %for react in sol.reactions():
        %if react.equation in [r.equation for _, r in falloff_reactions]:
        0.0,
        %else:
        ${cgm(ce.rate_coefficient_expr(react.rate, Variable("temperature")))},
        %endif
        %endfor
        };

        %if falloff_reactions:
        get_falloff_rates(temperature, concentrations, k_fwd);
        %endif

        %for i, react in three_body_reactions:
        k_fwd[${i}] *= (${cgm(ce.third_body_efficiencies_expr(
        sol, react, Variable("concentrations")))});
        %endfor
        return k_fwd;
    }

    static ContainerT get_net_rates_of_progress(
        ScalarT temperature, ContainerT const &concentrations)
    {
        ContainerT k_fwd = get_fwd_rate_coefficients(temperature, concentrations);
        ContainerT log_k_eq = get_equilibrium_constants(temperature);
        ContainerT r_net = {
        %for i in range(sol.n_reactions):
        ${cgm(ce.rate_of_progress_expr(sol, i, Variable("concentrations"),
            Variable("k_fwd"), Variable("log_k_eq")))},
        %endfor
        };
        return r_net;
    }

    static ContainerT get_net_production_rates(
        ScalarT rho, ScalarT temperature, ContainerT const &mass_fractions)
    {
        ContainerT concentrations = get_concentrations(rho, mass_fractions);
        ContainerT r_net = get_net_rates_of_progress(temperature, concentrations);
        ContainerT omega = {
        %for sp in sol.species():
        ${cgm(ce.production_rate_expr(sol, sp.name, Variable("r_net")))},
        %endfor
        };
        return omega;
    }
};
}
""", strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution, namespace="pyrometheus") -> str:
    """ For the mechanism given by *sol*, return Python source code for a class
    conforming to a module containing a class called ``Thermochemistry`` adhering
    to the :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """

    falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                   if r.reaction_type.startswith("falloff")]
    three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                      if r.reaction_type == "three-body-Arrhenius"]

    return header_tpl.render(
        ct=ct,
        sol=sol,

        namespace=namespace,

        str_np=str_np,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        ce=pyrometheus.chem_expr,

        falloff_reactions=falloff_rxn,
        three_body_reactions=three_body_rxn,
    )


# vim: foldmethod=marker
