"""
C++ code generation
----------------------

.. autoclass:: CppCodeGenerator
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

from . import CodeGenerator, CodeGenerationOptions


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
#pragma once

#if __cplusplus < 202002L
#error "Pyrometheus requires C++20 or later."
#endif

#include <array>
#include <cmath>
#include <string>
#include <concepts>

namespace pyro {

template <typename T>
concept IntegralLike = requires(T a, T b) {
    { T(1.0)  } -> std::same_as<T>;    { a         } -> std::convertible_to<double>;
    { a + b   } -> std::same_as<T>;    { a - b     } -> std::same_as<T>;
    { a * b   } -> std::same_as<T>;    { a / b     } -> std::same_as<T>;
    { -a      } -> std::same_as<T>;    { +a        } -> std::same_as<T>;
    { a += b  } -> std::same_as<T&>;   { a -= b    } -> std::same_as<T&>;
    { a *= b  } -> std::same_as<T&>;   { a /= b    } -> std::same_as<T&>;
    { a == b  } -> std::same_as<bool>; { a != b    } -> std::same_as<bool>;
    { a < b   } -> std::same_as<bool>; { a > b     } -> std::same_as<bool>;
    { a <= b  } -> std::same_as<bool>; { a >= b    } -> std::same_as<bool>;
    { log(a)  } -> std::same_as<T>;    { exp(a)    } -> std::same_as<T>;
    { sqrt(a) } -> std::same_as<T>;    { pow(a, b) } -> std::same_as<T>;
};

template <IntegralLike _DataTypeT = double, typename _ContainerT = _DataTypeT>
struct ${name}
{
    constexpr static int num_species = ${sol.n_species};
    constexpr static int num_reactions = ${sol.n_reactions};
    constexpr static int num_falloff = ${
        sum(1 if isinstance(r.rate, ct.FalloffRate) else 0
        for r in sol.reactions())};

    constexpr static const char* species_names[] = {
        ${", ".join(f'"{name}"' for name in sol.species_names)}
    };

    constexpr static const char* element_names[] = {
        ${", ".join(f'"{name}"' for name in sol.element_names)}
    };

    using DataTypeT  = _DataTypeT;
    using ContainerT = _ContainerT;
    using SpeciesT   = std::array<ContainerT, num_species>;
    using Species2T  = std::array<SpeciesT, num_species>;
    using ReactionsT = std::array<ContainerT, num_reactions>;

    constexpr static DataTypeT molecular_weights[] =
        {${str_np(sol.molecular_weights)}};
    constexpr static DataTypeT inv_molecular_weights[] =
        {${str_np(1/sol.molecular_weights)}};
    constexpr static DataTypeT gas_constant = ${repr(ct.gas_constant)};
    constexpr static DataTypeT one_atm = ${repr(ct.one_atm)};

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

    static ContainerT get_specific_gas_constant(SpeciesT const &mass_fractions)
    {
        return gas_constant * (
                %for i in range(sol.n_species):
                    + inv_molecular_weights[${i}]*mass_fractions[${i}]
                %endfor
                );
    }

    static ContainerT get_mix_molecular_weight(SpeciesT const &mass_fractions)
    {
        return 1.0/(
        %for i in range(sol.n_species):
            + inv_molecular_weights[${i}]*mass_fractions[${i}]
        %endfor
        );
    }

    static SpeciesT get_concentrations(
        ContainerT rho, SpeciesT const &mass_fractions)
    {
        SpeciesT concentrations = {
            %for i in range(sol.n_species):
                inv_molecular_weights[${i}]*mass_fractions[${i}]*rho,
            %endfor
        };
        return concentrations;
    }

    static SpeciesT get_mole_fractions(
        ContainerT mix_mol_weight, SpeciesT mass_fractions)
    {
        return SpeciesT{
            %for i in range(sol.n_species):
            inv_molecular_weights[${i}] * mass_fractions[${i}] * mix_mol_weight,
            %endfor
        };
    }

    static ContainerT get_mass_average_property(
        SpeciesT const &mass_fractions, SpeciesT const &spec_property)
    {
        return (
            %for i in range(sol.n_species):
                + mass_fractions[${i}]*
                  spec_property[${i}]*
                  inv_molecular_weights[${i}]
            %endfor
        );
    }

    static ContainerT get_mixture_specific_heat_cv_mass(
        ContainerT temperature, SpeciesT const &mass_fractions)
    {
        SpeciesT cp0_r = get_species_specific_heats_r(temperature);

        for (int i = 0; i < num_species; ++i)
            cp0_r[i] -= 1.0;

        const ContainerT cpmix = get_mass_average_property(mass_fractions, cp0_r);

        return gas_constant * cpmix;
    }

    static ContainerT get_mixture_specific_heat_cp_mass(
        ContainerT temperature, SpeciesT const &mass_fractions)
    {
        const SpeciesT cp0_r = get_species_specific_heats_r(temperature);
        const ContainerT cpmix = get_mass_average_property(mass_fractions, cp0_r);
        return gas_constant * cpmix;
    }

    static ContainerT get_mixture_enthalpy_mass(
        ContainerT temperature, SpeciesT const &mass_fractions)
    {
        SpeciesT h0_rt = get_species_enthalpies_rt(temperature);
        ContainerT hmix = get_mass_average_property(mass_fractions, h0_rt);
        return gas_constant * hmix * temperature;
    }

    static ContainerT get_mixture_internal_energy_mass(
        ContainerT temperature, SpeciesT const &mass_fractions)
    {
        SpeciesT e0_rt = get_species_enthalpies_rt(temperature);

        for (int i = 0; i < num_species; ++i)
            e0_rt[i] -= 1.0;

        const ContainerT emix = get_mass_average_property(mass_fractions, e0_rt);
        return gas_constant * emix * temperature;
    }

    static ContainerT get_density(ContainerT p, ContainerT temperature,
            SpeciesT const &mass_fractions)
    {
        ContainerT mmw = get_mix_molecular_weight(mass_fractions);
        ContainerT rt = gas_constant * temperature;
        return p * mmw / rt;
    }

    static ContainerT get_pressure(
        ContainerT density, ContainerT temperature, SpeciesT const &mass_fractions)
    {
        const double mmw = get_mix_molecular_weight(mass_fractions);
        const double rt = gas_constant * temperature;
        return density * rt / mmw;
    }

    static ContainerT get_mixture_molecular_weight(SpeciesT mass_fractions) {
        return 1.0/(
            %for i in range(sol.n_species):
            + inv_molecular_weights[${i}]*mass_fractions[${i}]
            %endfor
        );
    }

    static SpeciesT get_species_specific_heats_r(ContainerT temperature)
    {
        SpeciesT cp0_r = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return cp0_r;
    }

    static SpeciesT get_species_enthalpies_rt(ContainerT temperature)
    {
        SpeciesT h0_rt = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return h0_rt;
    }

    static SpeciesT get_species_entropies_r(ContainerT temperature)
    {
        SpeciesT s0_r = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_entropy_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return s0_r;
    }

    static SpeciesT get_species_gibbs_rt(ContainerT temperature)
    {
        SpeciesT h0_rt = get_species_enthalpies_rt(temperature);
        SpeciesT s0_r = get_species_entropies_r(temperature);
        SpeciesT g0_rt = {
        %for sp in range(sol.n_species):
        h0_rt[${sp}] - s0_r[${sp}],
        %endfor
        };
        return g0_rt;
    }

    static ReactionsT get_equilibrium_constants(ContainerT temperature)
    {
        ContainerT rt = gas_constant * temperature;
        ContainerT c0 = std::log(one_atm/rt);
        SpeciesT g0_rt = get_species_gibbs_rt(temperature);
        ReactionsT k_eq = {
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

    static ContainerT get_temperature(
        ContainerT energy_or_enthalpy,
        ContainerT t_guess,
        SpeciesT const &mass_fractions,
        bool const do_energy = true)
    {
        ContainerT (*pv_fun)(ContainerT, SpeciesT const &);
        ContainerT (*he_fun)(ContainerT, SpeciesT const &);

        if (do_energy) {
            pv_fun = get_mixture_specific_heat_cv_mass;
            he_fun = get_mixture_internal_energy_mass;
        } else {
            pv_fun = get_mixture_specific_heat_cp_mass;
            he_fun = get_mixture_enthalpy_mass;
        }

        int num_iter = 500;
        double tol = 1.0e-06;
        ContainerT iter_temp = t_guess;

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
        ContainerT temperature, SpeciesT const &concentrations, ReactionsT &k_fwd)
    {
        SpeciesT k_high = {
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.high_rate, Variable("temperature")))},
        %endfor
        };

        SpeciesT k_low = {
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.low_rate, Variable("temperature")))},
        %endfor
        };

        SpeciesT reduced_pressure = {
        %for i, (_, react) in enumerate(falloff_reactions):
            (${cgm(ce.third_body_efficiencies_expr(
                sol, react, Variable("concentrations")))})*k_low[${i}]/k_high[${i}],
        %endfor
        };

        SpeciesT falloff_center = {
        %for _, react in falloff_reactions:
            ${cgm(ce.troe_falloff_center_expr(react, Variable("temperature")))},
        %endfor
        };

        SpeciesT falloff_factor = {
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.troe_falloff_factor_expr(react, i,
            Variable("reduced_pressure"), Variable("falloff_center")))},
        %endfor
        };

        SpeciesT falloff_function = {
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

    static ReactionsT get_fwd_rate_coefficients(ContainerT temperature,
                                                SpeciesT const &concentrations)
    {
        ReactionsT k_fwd = {
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

    static ReactionsT get_net_rates_of_progress(
        ContainerT temperature, SpeciesT const &concentrations)
    {
        ReactionsT k_fwd = get_fwd_rate_coefficients(temperature, concentrations);
        ReactionsT log_k_eq = get_equilibrium_constants(temperature);
        ReactionsT r_net = {
        %for i in range(sol.n_reactions):
        ${cgm(ce.rate_of_progress_expr(sol, i, Variable("concentrations"),
            Variable("k_fwd"), Variable("log_k_eq")))},
        %endfor
        };
        return r_net;
    }

    static SpeciesT get_net_production_rates(
        ContainerT rho, ContainerT temperature, SpeciesT const &mass_fractions)
    {
        SpeciesT concentrations = get_concentrations(rho, mass_fractions);
        ReactionsT r_net = get_net_rates_of_progress(temperature, concentrations);
        SpeciesT omega = {
        %for sp in sol.species():
        ${cgm(ce.production_rate_expr(sol, sp.name, Variable("r_net")))},
        %endfor
        };
        return omega;
    }

    static SpeciesT get_species_viscosities(ContainerT temperature)
    {
        return SpeciesT{
            % for sp in range(sol.n_species):
            ${cgm(ce.viscosity_polynomial_expr(
                sol.get_viscosity_polynomial(sp),
                Variable("temperature")))},
            % endfor
        };
    }

    static SpeciesT get_species_thermal_conductivities(ContainerT temperature)
    {
        return SpeciesT{
            % for sp in range(sol.n_species):
            ${cgm(ce.conductivity_polynomial_expr(
                sol.get_thermal_conductivity_polynomial(sp),
                Variable("temperature")))},
            % endfor
        };
    }

    static Species2T get_species_binary_mass_diffusivities(ContainerT temperature)
    {
        return Species2T{
            %for i in range(sol.n_species):
            SpeciesT{
                %for j in range(sol.n_species):
                ${cgm(ce.diffusivity_polynomial_expr(
                      sol.get_binary_diff_coeffs_polynomial(i, j),
                      Variable("temperature")))},
                %endfor
            },
            %endfor
        };
    }

    static ContainerT get_mixture_viscosity_mixavg(
        ContainerT temperature, SpeciesT mass_fractions)
    {
        ContainerT mmw = get_mixture_molecular_weight(mass_fractions);
        SpeciesT mole_fractions = get_mole_fractions(mmw, mass_fractions);
        SpeciesT viscosities = get_species_viscosities(temperature);
        SpeciesT mix_rule_f = {
            %for sp in range(sol.n_species):
            ${cgm(ce.viscosity_mixture_rule_wilke_expr(sol, sp,
                Variable("mole_fractions"), Variable("viscosities")))},
            %endfor
        };

        return (0.0 +
        %for i in range(sol.n_species):
            + mole_fractions[${i}]*viscosities[${i}]/mix_rule_f[${i}]
        %endfor
        );
    }

    static ContainerT get_mixture_thermal_conductivity_mixavg(
        ContainerT temperature, SpeciesT mass_fractions)
    {
        ContainerT mmw = get_mixture_molecular_weight(mass_fractions);
        SpeciesT mole_fractions = get_mole_fractions(mmw, mass_fractions);
        SpeciesT conductivities = get_species_thermal_conductivities(temperature);

        ContainerT lhs = (0.0 +
        %for i in range(sol.n_species):
            + mole_fractions[${i}]*conductivities[${i}]
        %endfor
        );

        ContainerT rhs = 1.0 / (0.0 +
        %for i in range(sol.n_species):
            + mole_fractions[${i}]/conductivities[${i}]
        %endfor
        );

        return 0.5*(lhs + rhs);
    }

    static SpeciesT get_species_mass_diffusivities_mixavg(
        ContainerT pressure, ContainerT temperature, SpeciesT mass_fractions)
    {
        ContainerT mmw = get_mixture_molecular_weight(mass_fractions);
        SpeciesT mole_fractions = get_mole_fractions(mmw, mass_fractions);
        Species2T bdiff_ij = get_species_binary_mass_diffusivities(temperature);

        SpeciesT x_sum = {
            %for sp in range(sol.n_species):
            ${cgm(ce.diffusivity_mixture_rule_denom_expr(
                sol, sp, Variable("mole_fractions"), Variable("bdiff_ij")))},
            %endfor
        };

        SpeciesT denom = {
            %for s in range(sol.n_species):
            x_sum[${s}] - mole_fractions[${s}]/bdiff_ij[${s}][${s}],
            %endfor
        };

        return SpeciesT{
            %for sp in range(sol.n_species):
                denom[${sp}] > 0.0 ?
                    (mmw - mole_fractions[${sp}] * molecular_weights[${sp}])
                      / (pressure * mmw * denom[${sp}])
                  : (bdiff_ij[${sp}][${sp}] / pressure),
            %endfor
        };
    }

};
}
""", strict_undefined=True)

# }}}


class CppCodeGenerator(CodeGenerator):
    @staticmethod
    def get_name() -> str:
        return "cpp"

    @staticmethod
    def supports_overloading() -> bool:
        return False

    @staticmethod
    def generate(name: str,
                 sol: ct.Solution,
                 opts: CodeGenerationOptions = None) -> str:
        if opts is None:
            opts = CodeGenerationOptions()

        if opts.directive_offload is not None:
            raise TypeError(
                "OpenMP/ACC directive based offloading is not supported for "
                "C++ code generation"
                )

        falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                    if r.reaction_type.startswith("falloff")]
        three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                        if r.reaction_type == "three-body-Arrhenius"]

        return header_tpl.render(
            ct=ct,
            sol=sol,

            name=name,

            str_np=str_np,
            cgm=CodeGenerationMapper(),
            Variable=p.Variable,

            ce=pyrometheus.chem_expr,

            falloff_reactions=falloff_rxn,
            three_body_reactions=three_body_rxn,
        )


# vim: foldmethod=marker
