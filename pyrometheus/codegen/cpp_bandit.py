"""
C++ code generation (Bandit)
-----------------------------

.. autoclass:: CppBanditCodeGenerator
"""

from numbers import Number

import numpy as np
import pymbolic.primitives as p
from mako.template import Template
from pymbolic.mapper.stringifier import (
    StringifyMapper, PREC_NONE, PREC_CALL, PREC_PRODUCT, PREC_UNARY,
    PREC_LOGICAL_AND, PREC_LOGICAL_OR, PREC_IF,
)

from pyrometheus.bandit.general_thermochem import BaseMechanism
from pyrometheus.bandit.chem_expr.transport import (
    _boltzmann as transport_boltzmann
)

from . import CodeGenerator, CodeGenerationOptions


file_extension = "hpp"


# {{{ code generation helpers

def float_to_cpp(number):
    """:returns: A C++ double literal for *number* that round-trips, so
    integer-valued constants never fall into integer division.
    """
    value = float(number)
    if not np.isfinite(value):
        raise ValueError(f"cannot generate the non-finite constant {value}")
    literal = repr(value)
    if value < 0:
        literal = f"({literal})"
    return literal


def str_np(ary):
    return ", ".join(float_to_cpp(entry) for entry in ary)


def temperature_type(bandit_mech):
    """:returns: The C++ type of the *temperature* argument: a scalar
    for single-temperature mechanisms, an array of ``num_temperatures``
    otherwise, matching how expressions index it.
    """
    if bandit_mech.num_temp > 1:
        return "TemperaturesT"
    return "ScalarT"


def temperature_leading_term(bandit_mech):
    """:returns: The heavy-particle temperature that scales mixture
    enthalpy and energy.
    """
    return "temperature[0]" if bandit_mech.num_temp > 1 else "temperature"

# }}}


# {{{ C++ expression generation

class CppExpressionMapper(StringifyMapper):
    """Converts expressions to C++, keeping the evaluation order of the
    Python and Fortran backends so that the three agree to round-off.
    """

    def map_constant(self, expr, enclosing_prec):
        if isinstance(expr, (bool, np.bool_)):
            return "true" if expr else "false"
        if isinstance(expr, Number):
            return float_to_cpp(expr)
        raise TypeError(f"cannot generate the constant {expr!r}")

    def map_variable(self, expr, enclosing_prec):
        return expr.name

    def map_subscript(self, expr, enclosing_prec):
        indices = expr.index if isinstance(expr.index, tuple) else (expr.index,)

        def index_to_cpp(index):
            if isinstance(index, (int, np.integer)):
                return str(int(index))
            return self.rec(index, PREC_NONE)

        return self.parenthesize_if_needed(
            self.rec(expr.aggregate, PREC_CALL) + "".join(
                f"[{index_to_cpp(index)}]" for index in indices),
            enclosing_prec, PREC_CALL)

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" * ", expr.children, PREC_PRODUCT, *args, **kwargs),
            enclosing_prec, PREC_PRODUCT)

    def map_power(self, expr, enclosing_prec, *args, **kwargs):
        # Squares become products, always parenthesized: as a divisor,
        # a / x * x would otherwise parse as (a / x) * x.
        if p.is_constant(expr.exponent) and expr.exponent == 2:
            base = self.rec(expr.base, PREC_PRODUCT, *args, **kwargs)
            return f"({base} * {base})"
        return self.format(
            "pow(%s, %s)",
            self.rec(expr.base, PREC_NONE, *args, **kwargs),
            self.rec(expr.exponent, PREC_NONE, *args, **kwargs))

    def map_logical_not(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            "!" + self.rec(expr.child, PREC_UNARY, *args, **kwargs),
            enclosing_prec, PREC_UNARY)

    def map_logical_and(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" && ", expr.children, PREC_LOGICAL_AND,
                          *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_AND)

    def map_logical_or(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" || ", expr.children, PREC_LOGICAL_OR,
                          *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_OR)

    def map_if(self, expr, enclosing_prec, *args, **kwargs):
        return "(%s ? %s : %s)" % (
            self.rec(expr.condition, PREC_IF, *args, **kwargs),
            self.rec(expr.then, PREC_IF, *args, **kwargs),
            self.rec(expr.else_, PREC_IF, *args, **kwargs),
        )

# }}}


# {{{ header template

header_tpl = Template("""
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pyro {

using std::exp;
using std::log;
using std::pow;
using std::sqrt;

template <typename ScalarT = double>
struct ${name}
{
    static constexpr int num_species = ${bandit_mech.num_species};
    static constexpr int num_reactions = ${bandit_mech.num_reactions};
    static constexpr int num_temperatures = ${bandit_mech.num_temp};

    using SpeciesT = std::array<ScalarT, num_species>;
    using Species2T = std::array<SpeciesT, num_species>;
    using ReactionsT = std::array<ScalarT, num_reactions>;
    using TemperaturesT = std::array<ScalarT, num_temperatures>;
    using TemperatureT = ${temperature_type(bandit_mech)};

    static constexpr double one_atm = ${float_to_cpp(bandit_mech.namespace.one_atm)};
    static constexpr double gas_constant = ${float_to_cpp(bandit_mech.namespace.gas_constant)};
    static constexpr std::array<double, num_species> molecular_weights = {
        ${str_np(bandit_mech.molecular_weights)}
    };
    static constexpr std::array<double, num_species> inv_molecular_weights = {
        ${str_np(1/bandit_mech.molecular_weights)}
    };
    static constexpr std::array<const char*, num_species> species_names = {
        ${", ".join(f'"{name}"' for name in bandit_mech.species_names)}
    };

    static std::string get_species_name(int species_index)
    {
        return species_names.at(species_index);
    }

    static int get_species_index(std::string const &species_name)
    {
        for (int species_index = 0; species_index < num_species; ++species_index)
            if (species_name == species_names[species_index])
                return species_index;
        return -1;
    }

    static ScalarT get_specific_gas_constant(SpeciesT const &mass_fractions)
    {
        return gas_constant * (
            %for i in range(bandit_mech.num_species):
            + inv_molecular_weights[${i}] * mass_fractions[${i}]
            %endfor
        );
    }

    static ScalarT get_mixture_molecular_weight(SpeciesT const &mass_fractions)
    {
        return 1.0 / (
            %for i in range(bandit_mech.num_species):
            + inv_molecular_weights[${i}] * mass_fractions[${i}]
            %endfor
        );
    }

    static SpeciesT get_mole_fractions(SpeciesT const &mass_fractions)
    {
        const ScalarT mix_mol_weight = get_mixture_molecular_weight(mass_fractions);
        SpeciesT mole_fractions;
        %for i in range(bandit_mech.num_species):
        mole_fractions[${i}] = inv_molecular_weights[${i}] * mass_fractions[${i}] * mix_mol_weight;
        %endfor
        return mole_fractions;
    }

    static ScalarT get_density(
        ScalarT pressure, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        return pressure / (${cgm(bandit_mech.pressure_over_density_expr)});
    }

    static ScalarT get_pressure(
        ScalarT density, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        return density * (${cgm(bandit_mech.pressure_over_density_expr)});
    }

    static SpeciesT get_concentrations(
        ScalarT density, SpeciesT const &mass_fractions)
    {
        SpeciesT concentrations;
        %for i in range(bandit_mech.num_species):
        concentrations[${i}] = density * mass_fractions[${i}] * inv_molecular_weights[${i}];
        %endfor
        return concentrations;
    }

    static ScalarT get_mass_average_property(
        SpeciesT const &mass_fractions, SpeciesT const &spec_property)
    {
        return (
            %for i in range(bandit_mech.num_species):
            + inv_molecular_weights[${i}] * mass_fractions[${i}] * spec_property[${i}]
            %endfor
        );
    }

    static SpeciesT get_species_specific_heats_cp_r(TemperatureT const &temperature)
    {
        SpeciesT cp_r;
        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        cp_r[${i}] = ${cgm(sp_thermo.cp_poly.expr)};
        %endfor
        return cp_r;
    }

    static SpeciesT get_species_specific_heats_cv_r(TemperatureT const &temperature)
    {
        SpeciesT cv_r = get_species_specific_heats_cp_r(temperature);
        for (auto &entry : cv_r)
            entry -= 1.0;
        return cv_r;
    }

    static SpeciesT get_species_enthalpies_rt(TemperatureT const &temperature)
    {
        SpeciesT h_rt;
        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        h_rt[${i}] = ${cgm(sp_thermo.enthalpy_poly.expr)};
        %endfor
        return h_rt;
    }

    static SpeciesT get_species_internal_energies_rt(TemperatureT const &temperature)
    {
        SpeciesT e_rt = get_species_enthalpies_rt(temperature);
        for (auto &entry : e_rt)
            entry -= 1.0;
        return e_rt;
    }

    static SpeciesT get_species_gibbs_rt(TemperatureT const &temperature)
    {
        SpeciesT g_rt;
        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        g_rt[${i}] = ${cgm(sp_thermo.gibbs_poly.expr)};
        %endfor
        return g_rt;
    }

    static ScalarT get_mixture_specific_heat_cp_mass(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        return gas_constant * get_mass_average_property(
            mass_fractions, get_species_specific_heats_cp_r(temperature));
    }

    static ScalarT get_mixture_specific_heat_cv_mass(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        return gas_constant * get_mass_average_property(
            mass_fractions, get_species_specific_heats_cv_r(temperature));
    }

    static ScalarT get_mixture_enthalpy_mass(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        return gas_constant * ${temperature_leading_term(bandit_mech)} * get_mass_average_property(
            mass_fractions, get_species_enthalpies_rt(temperature));
    }

    static ScalarT get_mixture_internal_energy_mass(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        return gas_constant * ${temperature_leading_term(bandit_mech)} * get_mass_average_property(
            mass_fractions, get_species_internal_energies_rt(temperature));
    }

    %if bandit_mech.num_temp == 1:
    static ScalarT get_temperature(
        ScalarT enthalpy_or_energy, ScalarT t_guess,
        SpeciesT const &mass_fractions, bool do_energy = false)
    {
        using std::abs;
        constexpr int num_iter = 500;
        constexpr double tol = 1.0e-6;
        ScalarT iter_temp = t_guess;
        for (int iter = 0; iter < num_iter; ++iter) {
            const ScalarT iter_rhs = enthalpy_or_energy - (do_energy
                ? get_mixture_internal_energy_mass(iter_temp, mass_fractions)
                : get_mixture_enthalpy_mass(iter_temp, mass_fractions));
            const ScalarT iter_deriv = -(do_energy
                ? get_mixture_specific_heat_cv_mass(iter_temp, mass_fractions)
                : get_mixture_specific_heat_cp_mass(iter_temp, mass_fractions));
            const ScalarT iter_dt = -iter_rhs / iter_deriv;
            iter_temp += iter_dt;
            if (abs(iter_dt) < tol)
                return iter_temp;
        }
        throw std::runtime_error("Temperature iteration failed to converge");
    }

    %endif
    static ReactionsT get_equilibrium_constants(TemperatureT const &temperature)
    {
        const SpeciesT gibbs_rt = get_species_gibbs_rt(temperature);
        ReactionsT log_k_eq;
        %for i, expr in enumerate(bandit_mech.equil_constants):
        %if bandit_mech.is_reversible(i):
        log_k_eq[${i}] = ${cgm(expr)};
        %else:
        log_k_eq[${i}] = -0.17364695002734 * ${temperature_leading_term(bandit_mech)};
        %endif
        %endfor
        return log_k_eq;
    }

    %if bandit_mech.has_falloff_reactions():
<% num_falloff = len(bandit_mech.falloff_reactions) %>
    using FalloffT = std::array<ScalarT, ${num_falloff}>;

    static FalloffT get_falloff_rates(
        TemperatureT const &temperature, SpeciesT const &concentrations)
    {
        FalloffT k_high, k_low, falloff_center, reduced_pressure;
        FalloffT falloff_factor, falloff_function, falloff_rate_coefficients;
        %for attribute, target in [("high_rate_expr", "k_high"), ("low_rate_expr", "k_low"), ("falloff_center", "falloff_center"), ("reduced_pressure", "reduced_pressure"), ("falloff_factor", "falloff_factor"), ("falloff_function", "falloff_function"), ("rate_coefficient", "falloff_rate_coefficients")]:
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        ${target}[${i}] = ${cgm(getattr(falloff, attribute))};
        %endfor
        %endfor
        return falloff_rate_coefficients;
    }

    %endif
    static ReactionsT get_fwd_rate_coefficients(
        TemperatureT const &temperature,
        [[maybe_unused]] SpeciesT const &concentrations)
    {
        %if bandit_mech.has_falloff_reactions():
        const FalloffT falloff_rate_coefficients = get_falloff_rates(
            temperature, concentrations);
        %endif
        ReactionsT k_fwd;
        %for i, rate_coeff in enumerate(bandit_mech.rate_coeffs):
        k_fwd[${i}] = ${cgm(rate_coeff.expr)};
        %endfor
        return k_fwd;
    }

    static ReactionsT get_net_rates_of_progress(
        TemperatureT const &temperature, SpeciesT const &concentrations)
    {
        const ReactionsT k_fwd = get_fwd_rate_coefficients(temperature, concentrations);
        const ReactionsT log_k_eq = get_equilibrium_constants(temperature);
        ReactionsT r_net;
        %for i, expr in enumerate(bandit_mech.mass_action_rates):
        r_net[${i}] = ${cgm(expr)};
        %endfor
        return r_net;
    }

    static SpeciesT get_net_production_rates(
        ScalarT density, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        const SpeciesT concentrations = get_concentrations(density, mass_fractions);
        const ReactionsT r_net = get_net_rates_of_progress(temperature, concentrations);
        SpeciesT omega;
        %for i, expr in enumerate(bandit_mech.species_prod_rates):
        omega[${i}] = ${cgm(expr)};
        %endfor
        return omega;
    }

    %if opts.compute_jacobian:
<% num_variables = 1 + bandit_mech.num_temp + bandit_mech.num_species %>
    using JacobianT = std::array<std::array<ScalarT, ${num_variables}>, num_species>;

    // jacobian[i][j] = d omega[i] / d x[j], with x = [density,
    // temperature[0..num_temperatures), mass_fractions[0..num_species)].
    static JacobianT get_net_production_rates_jacobian(
        ScalarT density, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        JacobianT jacobian;
        %for i, row in enumerate(bandit_mech.species_production_rate_jacobian_exprs):
        %for j, entry in enumerate(row):
        jacobian[${i}][${j}] = ${cgm(entry)};
        %endfor
        %endfor
        return jacobian;
    }

    %endif
    %if bandit_mech.has_nonequilibrium_energy_modes():
<% num_vt = len(bandit_mech.pressure_relaxation_time_exprs) %>
    static SpeciesT get_species_vibrational_energies(TemperatureT const &temperature)
    {
        SpeciesT energies;
        %for i, sp_thermo in enumerate(bandit_mech.species_vib_thermo_expressions):
        energies[${i}] = ${cgm(sp_thermo.energy_expr)};
        %endfor
        return energies;
    }

    static SpeciesT get_species_vibrational_specific_heats(TemperatureT const &temperature)
    {
        SpeciesT specific_heats;
        %for i, sp_thermo in enumerate(bandit_mech.species_vib_thermo_expressions):
        specific_heats[${i}] = ${cgm(sp_thermo.specific_heat_expr)};
        %endfor
        return specific_heats;
    }

    static std::array<ScalarT, ${num_vt}> get_pressure_relaxation_times(
        TemperatureT const &temperature)
    {
        std::array<ScalarT, ${num_vt}> relaxation_times;
        %for i, expr in enumerate(bandit_mech.pressure_relaxation_time_exprs):
        relaxation_times[${i}] = ${cgm(expr)};
        %endfor
        return relaxation_times;
    }

    static ScalarT get_vt_energy_transfer_source(
        ScalarT density, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        [[maybe_unused]] const ScalarT pressure = get_pressure(
            density, temperature, mass_fractions);
        return 0.0
            %for expr in bandit_mech.vt_energy_transfer_exprs:
            + ${cgm(expr)}
            %endfor
            ;
    }

    static SpeciesT get_translational_rotational_energy(TemperatureT const &temperature)
    {
        SpeciesT energies;
        %for i, expr in enumerate(bandit_mech.translational_rotational_energy_exprs):
        energies[${i}] = ${cgm(expr)};
        %endfor
        return energies;
    }

    static SpeciesT get_nasa_polynomial_vibrational_energy(TemperatureT const &temperature)
    {
        SpeciesT energies;
        %for i, expr in enumerate(bandit_mech.nasa_polynomial_vibrational_energy_exprs):
        energies[${i}] = ${cgm(expr)};
        %endfor
        return energies;
    }

    static SpeciesT get_nasa_polynomial_vibrational_specific_heat(
        TemperatureT const &temperature)
    {
        SpeciesT specific_heats;
        %for i, expr in enumerate(bandit_mech.nasa_polynomial_vibrational_specific_heat_exprs):
        specific_heats[${i}] = ${cgm(expr)};
        %endfor
        return specific_heats;
    }

    %endif
    %if bandit_mech.has_transport():
<%
    transport = bandit_mech.mixture_transport
    heavy = bandit_mech.heavy_temperature
%>
    static SpeciesT get_transport_mole_fractions(SpeciesT const &mass_fractions)
    {
        SpeciesT mole_fractions = get_mole_fractions(mass_fractions);
        ScalarT total = 0.0;
        for (auto &entry : mole_fractions) {
            entry += ${float_to_cpp(bandit_mech.mole_fraction_tolerance)};
            total += entry;
        }
        for (auto &entry : mole_fractions)
            entry /= total;
        return mole_fractions;
    }

    // Transport matrices are symmetric positive definite, so Gaussian
    // elimination needs no pivoting.
    static SpeciesT solve_transport_system(Species2T matrix, SpeciesT solution)
    {
        for (int k = 0; k < num_species - 1; ++k) {
            for (int i = k + 1; i < num_species; ++i) {
                const ScalarT multiplier = matrix[i][k] / matrix[k][k];
                for (int j = k + 1; j < num_species; ++j)
                    matrix[i][j] -= multiplier * matrix[k][j];
                solution[i] -= multiplier * solution[k];
            }
        }
        for (int i = num_species - 1; i >= 0; --i) {
            for (int j = i + 1; j < num_species; ++j)
                solution[i] -= matrix[i][j] * solution[j];
            solution[i] /= matrix[i][i];
        }
        return solution;
    }

    static Species2T get_species_binary_mass_diffusivities(
        TemperatureT const &temperature)
    {
        Species2T binary_diffusivities;
        %for i, row in enumerate(transport.inverse_number_density_diffusivities):
        %for j, expr in enumerate(row):
        binary_diffusivities[${i}][${j}] = ${cgm(boltzmann * heavy / expr)};
        %endfor
        %endfor
        return binary_diffusivities;
    }

    static SpeciesT get_species_effective_diffusivities(
        ScalarT pressure, TemperatureT const &temperature,
        SpeciesT const &mass_fractions)
    {
        const SpeciesT mole_fractions = get_transport_mole_fractions(mass_fractions);
        const ScalarT number_density = pressure / (${float_to_cpp(boltzmann)} * ${cgm(heavy)});
        SpeciesT diffusivities;
        %for i, expr in enumerate(transport.number_density_effective_diffusivities):
        diffusivities[${i}] = (${cgm(expr)}) / number_density;
        %endfor
        return diffusivities;
    }

    static SpeciesT get_species_viscosities(TemperatureT const &temperature)
    {
        SpeciesT viscosities;
        %for i, expr in enumerate(transport.species_viscosities):
        viscosities[${i}] = ${cgm(expr)};
        %endfor
        return viscosities;
    }

    static ScalarT get_mixture_viscosity(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        const SpeciesT mole_fractions = get_transport_mole_fractions(mass_fractions);
        Species2T matrix;
        %for i, row in enumerate(transport.viscosity_matrix):
        %for j, expr in enumerate(row):
        matrix[${i}][${j}] = ${cgm(expr)};
        %endfor
        %endfor
        const SpeciesT solution = solve_transport_system(matrix, mole_fractions);
        return 0.0
            %for i in range(bandit_mech.num_species):
            + mole_fractions[${i}] * solution[${i}]
            %endfor
            ;
    }

    static TemperaturesT get_mixture_thermal_conductivities(
        TemperatureT const &temperature, SpeciesT const &mass_fractions)
    {
        const SpeciesT mole_fractions = get_transport_mole_fractions(mass_fractions);
        Species2T matrix;
        %for i, row in enumerate(transport.conductivity_matrix):
        %for j, expr in enumerate(row):
        matrix[${i}][${j}] = ${cgm(expr)};
        %endfor
        %endfor
        const SpeciesT solution = solve_transport_system(matrix, mole_fractions);
        TemperaturesT conductivities;
        %for k, expr in enumerate(transport.internal_conductivities):
        conductivities[${k}] = ${cgm(expr)};
        %endfor
        %for i in range(bandit_mech.num_species):
        conductivities[0] += mole_fractions[${i}] * solution[${i}];
        %endfor
        return conductivities;
    }

    %endif
};

}  // namespace pyro
""", strict_undefined=True)

# }}}


class CppBanditCodeGenerator(CodeGenerator):
    @staticmethod
    def get_name() -> str:
        return "cpp-bandit"

    @staticmethod
    def load_mechanism(mech_path: str, phase: str = None) -> BaseMechanism:
        from pyrometheus.bandit.impl.cantera import CanteraMechanism
        return CanteraMechanism(mech_path)

    @staticmethod
    def supports_overloading() -> bool:
        return True

    @staticmethod
    def generate(name: str,
                 bandit_mech: BaseMechanism,
                 opts: CodeGenerationOptions = None) -> str:
        if opts is None:
            opts = CodeGenerationOptions()

        if opts.directive_offload is not None:
            raise TypeError(
                "OpenMP/ACC directive based offloading is not supported for "
                "C++ code generation"
            )

        if opts.compute_jacobian:
            bandit_mech.make_species_production_rate_jacobian()

        return header_tpl.render(
            name=name,
            bandit_mech=bandit_mech,
            opts=opts,
            cgm=CppExpressionMapper(),
            str_np=str_np,
            float_to_cpp=float_to_cpp,
            boltzmann=transport_boltzmann,
            temperature_type=temperature_type,
            temperature_leading_term=temperature_leading_term,
        ).strip() + "\n"


# vim: foldmethod=marker
