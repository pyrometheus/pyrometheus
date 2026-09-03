"""
Fortran code generation (Bandit)
---------------------------------

.. autoclass:: FortranBanditCodeGenerator
"""

import shlex
from functools import partial

import pymbolic.primitives as p
from mako.template import Template
from pyrometheus.bandit.general_thermochem import BaseMechanism
from pymbolic.mapper.stringifier import (
        StringifyMapper, PREC_NONE, PREC_CALL, PREC_PRODUCT)

from . import CodeGenerator, CodeGenerationOptions


file_extension = "f90"


# {{{ code generation helpers

def pad_fortran(line, width):
    line += " " * (width - 1 - len(line))
    line += "&"
    return line


def wrap_line_base(line, level=0, width=80, indentation="    ",
                   pad_func=lambda string, amount: string,
                   lex_func=None):
    """
    The input is a line of code at the given indentation level. Return the list
    of lines that results from wrapping the line to the given width. Lines
    subsequent to the first line in the returned list are padded with extra
    indentation. The initial indentation level is not included in the input or
    output lines.

    The `pad_func` argument is a function that adds line continuations. The
    `lex_func` argument returns the list of tokens in the line.
    """
    if lex_func is None:
        lex_func = partial(shlex.split, posix=False)

    tokens = lex_func(line)
    resulting_lines = []
    at_line_start = True
    indentation_len = len(level * indentation)
    current_line = ""
    padding_width = width - indentation_len
    for index, word in enumerate(tokens):
        has_next_word = index < len(tokens) - 1
        word_len = len(word)
        if not at_line_start:
            next_len = indentation_len + len(current_line) + 1 + word_len
            if next_len < width or (not has_next_word and next_len == width):
                # The word goes on the same line.
                current_line += " " + word
            else:
                # The word goes on the next line.
                resulting_lines.append(pad_func(current_line, padding_width))
                at_line_start = True
                current_line = indentation
        if at_line_start:
            current_line += word
            at_line_start = False
    resulting_lines.append(current_line)
    return resulting_lines


def count_leading_spaces(s):
    n = 0
    while n < len(s) and s[n] == " ":
        n += 1
    return n


def wrap_code(s, indent=4):
    lines = s.split("\n")
    result_lines = []
    for ln in lines:
        nspaces = count_leading_spaces(ln)
        level, remainder = divmod(nspaces, indent)

        if remainder != 0:
            raise ValueError(f"indentation of '{ln}' is not a multiple of "
                    f"{indent}")

        result_lines.extend(
                (level * indent) * " " + subln
                for subln in
                wrap_line_base(ln, level=level, indentation=" "*indent,
                    pad_func=pad_fortran))

    return "\n".join(result_lines)


def float_to_fortran(num):
    result = f"{num}".replace("e", "d")
    if "d" not in result:
        result = result+"d0"
    if num < 0:
        result = "(%s)" % result
    return result


def str_np(ary):
    return ", ".join(float_to_fortran(entry) for entry in ary)


def temperature_decl(real_type, bandit_mech, intent="in"):
    """Return the Fortran dummy-argument declaration for *temperature*: a
    scalar when the mechanism tracks a single temperature, or an array of
    length ``num_temperatures`` for multi-temperature (e.g. TTv) mechanisms.
    Per-species/per-reaction expressions already reference the correct slot
    (``temperature`` or ``temperature(k)``) baked in at code-generation time,
    so every subroutine that forwards temperature to those expressions must
    declare it with this same shape.
    """
    if bandit_mech.num_temp > 1:
        return (f"{real_type}, intent({intent}), "
                f"dimension({bandit_mech.num_temp}) :: temperature")
    else:
        return f"{real_type}, intent({intent}) :: temperature"


def temperature_leading_term(bandit_mech):
    """Return the Fortran expression for the temperature used to scale
    mixture-averaged quantities (enthalpy, energy) that are not already
    fully expressed by per-species polynomials: the heavy-particle/
    translational temperature for multi-temperature mechanisms, or the
    single temperature otherwise.
    """
    return "temperature(1)" if bandit_mech.num_temp > 1 else "temperature"

# }}}


# {{{ fortran expression generation

class FortranExpressionMapper(StringifyMapper):
    """Converts expressions to Fortran code."""

    def map_constant(self, expr, enclosing_prec):
        if isinstance(expr, bool):
            if expr:
                return ".true."
            else:
                return ".false."
        else:
            return float_to_fortran(expr)

    def map_variable(self, expr, enclosing_prec):
        return expr.name

    def map_lookup(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.format("%s%%%s",
                    self.rec(expr.aggregate, PREC_CALL),
                    expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec):
        def get_base_and_indices(expr):
            if not hasattr(expr, "aggregate") or not hasattr(expr, "index"):
                return expr, []

            # Get current level indices
            if isinstance(expr.index, tuple):
                current_indices = [self.rec(i, PREC_NONE) for i in expr.index]
            else:
                current_indices = [self.rec(expr.index, PREC_NONE)]

            # Only recurse if aggregate is another subscript
            if (hasattr(expr.aggregate, "aggregate")
            and hasattr(expr.aggregate, "index")):
                base, prev_indices = get_base_and_indices(expr.aggregate)
                return base, prev_indices + current_indices
            else:
                return expr.aggregate, current_indices

        # Get base array and all indices
        base_array, all_indices = get_base_and_indices(expr)

        # Convert float indices (ending with 'd0') to integers and add 1
        def convert_index(idx):
            idx_str = str(idx)
            if idx_str.endswith("d0"):
                # Remove 'd0' suffix and convert to int
                num = int(float(idx_str.replace("d0", "")))
                return str(num + 1)
            try:
                # Try to convert to int and add 1
                return str(int(idx_str) + 1)
            except ValueError:
                # If it's not a simple number, wrap in a +1
                return f"({idx_str} + 1)"

        # Format indices, converting floats to integers and adding 1
        index_str = ", ".join(convert_index(idx) for idx in all_indices)

        # Format the final expression
        return self.parenthesize_if_needed(
            self.format("%s(%s)", self.rec(base_array, PREC_CALL), index_str),
            enclosing_prec,
            PREC_CALL
        )

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        # This differs from the superclass only by adding spaces
        # around the operator, which provide an opportunity for
        # line breaking.
        return self.parenthesize_if_needed(
                self.join_rec(" * ", expr.children, PREC_PRODUCT, *args, **kwargs),
                enclosing_prec, PREC_PRODUCT)

    def map_logical_not(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_UNARY
        return self.parenthesize_if_needed(
                ".not. " + self.rec(expr.child, PREC_UNARY),
                enclosing_prec, PREC_UNARY)

    def map_logical_or(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_OR
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .or. ", expr.children, PREC_LOGICAL_OR),
                enclosing_prec, PREC_LOGICAL_OR)

    def map_logical_and(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_AND
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .and. ", expr.children, PREC_LOGICAL_AND),
                enclosing_prec, PREC_LOGICAL_AND)

    def map_if(self, expr, enclosing_prec):
        return self.format(
            "merge(%s)" % self.join_rec(", ",
                                        [expr.then, expr.else_,
                                         expr.condition],
                                        PREC_NONE))
# }}}


# {{{ module template

module_tpl = Template("""
#ifndef PYROMETHEUS_CALLER_INDEXING
#define PYROMETHEUS_CALLER_INDEXING 0
#endif

${gpu_routine}

module ${module_name}

    implicit none

    integer, parameter :: sp = selected_real_kind(6,37)   ! Single precision
    integer, parameter :: dp = selected_real_kind(15,307) ! Double precision

    integer, parameter :: num_species = ${bandit_mech.num_species}
    integer, parameter :: num_reactions = ${bandit_mech.num_reactions}
    integer, parameter :: num_temperatures = ${bandit_mech.num_temp}
    ${real_type}, parameter :: one_atm = &
        ${float_to_fortran(bandit_mech.namespace.one_atm)}
    ${real_type}, parameter :: gas_constant = &
        ${float_to_fortran(bandit_mech.namespace.gas_constant)}
    ${real_type}, parameter :: molecular_weights(${bandit_mech.num_species}) = &
        (/ ${str_np(bandit_mech.molecular_weights)} /)
    ${real_type}, parameter :: inv_molecular_weights(${bandit_mech.num_species}) = &
        (/ ${str_np(1/bandit_mech.molecular_weights)} /)

    character(len=12), parameter :: species_names(${bandit_mech.num_species}) = &
        (/ ${", ".join('"'+'{0: <12}'.format(s)+'"'
                       for s in bandit_mech.species_names)} /)

contains

    subroutine get_species_name(sp_index, sp_name)

        integer, intent(in) :: sp_index
        character(len=*), intent(out) :: sp_name

        sp_name = species_names(sp_index + PYROMETHEUS_CALLER_INDEXING)

    end subroutine get_species_name

    subroutine get_species_index(sp_name, sp_index)

        character(len=*), intent(in) :: sp_name
        integer, intent(out) :: sp_index

        integer :: idx

        sp_index = 0
        loop:do idx = 1, num_species
            if(trim(adjustl(sp_name)) .eq. trim(species_names(idx))) then
                sp_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_species_index

    subroutine get_specific_gas_constant(mass_fractions, specific_gas_constant)

        GPU_ROUTINE(get_specific_gas_constant)

        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: specific_gas_constant

        specific_gas_constant = gas_constant * ( &
                %for i in range(bandit_mech.num_species):
                    + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

    end subroutine get_specific_gas_constant

    subroutine get_density(pressure, temperature, mass_fractions, density)

        GPU_ROUTINE(get_density)

        ${real_type}, intent(in) :: pressure
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: density

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        density = pressure * mix_mol_weight / (gas_constant * temperature)

    end subroutine get_density

    subroutine get_pressure(density, temperature, mass_fractions, pressure)

        GPU_ROUTINE(get_pressure)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: pressure

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        pressure = density * gas_constant * temperature / mix_mol_weight

    end subroutine get_pressure

    subroutine get_mixture_molecular_weight(mass_fractions, mix_mol_weight)

        GPU_ROUTINE(get_mixture_molecular_weight)

        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: mix_mol_weight

        mix_mol_weight = 1.0d0 / ( &
                %for i in range(bandit_mech.num_species):
                    + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

    end subroutine get_mixture_molecular_weight

    subroutine get_concentrations(density, mass_fractions, concentrations)

        GPU_ROUTINE(get_concentrations)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in),  dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: &
            concentrations

        %for i in range(bandit_mech.num_species):
            concentrations(${i+1}) = density * &
                inv_molecular_weights(${i+1}) * mass_fractions(${i+1})
        %endfor

    end subroutine get_concentrations

    subroutine get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)

        GPU_ROUTINE(get_mole_fractions)

        ${real_type}, intent(in) :: mix_mol_weight
        ${real_type}, intent(in),  dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: &
            mole_fractions

        %for i in range(bandit_mech.num_species):
            mole_fractions(${i+1}) = inv_molecular_weights(${i+1}) * &
                mass_fractions(${i+1}) * mix_mol_weight
        %endfor

    end subroutine get_mole_fractions

    subroutine get_mass_averaged_property(&
        & mass_fractions, spec_property, mix_property)

        GPU_ROUTINE(get_mass_averaged_property)

        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            spec_property
        ${real_type}, intent(out) :: mix_property

        mix_property =  ( &
            %for i in range(bandit_mech.num_species):
                + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                *spec_property(${i+1}) &
            %endfor
        )

    end subroutine get_mass_averaged_property

    subroutine get_mixture_specific_heat_cp_mass(temperature, mass_fractions, cp_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cp_mass)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: cp_mix

        ${real_type}, dimension(${bandit_mech.num_species}) :: cp0_r

        call get_species_specific_heats_cp_r(temperature, cp0_r)
        call get_mass_averaged_property(mass_fractions, cp0_r, cp_mix)
        cp_mix = cp_mix * gas_constant

    end subroutine get_mixture_specific_heat_cp_mass

    subroutine get_mixture_specific_heat_cv_mass(temperature, mass_fractions, cv_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cv_mass)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: cv_mix

        ${real_type}, dimension(${bandit_mech.num_species}) :: cv0_r

        call get_species_specific_heats_cv_r(temperature, cv0_r)
        call get_mass_averaged_property(mass_fractions, cv0_r, cv_mix)
        cv_mix = cv_mix * gas_constant

    end subroutine get_mixture_specific_heat_cv_mass

    subroutine get_mixture_enthalpy_mass(temperature, mass_fractions, h_mix)

        GPU_ROUTINE(get_mixture_enthalpy_mass)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: h_mix

        ${real_type}, dimension(${bandit_mech.num_species}) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_mass_averaged_property(mass_fractions, h0_rt, h_mix)
        h_mix = h_mix * gas_constant * ${temperature_leading_term(bandit_mech)}

    end subroutine get_mixture_enthalpy_mass

    subroutine get_mixture_energy_mass(temperature, mass_fractions, e_mix)

        GPU_ROUTINE(get_mixture_energy_mass)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: e_mix

        ${real_type}, dimension(${bandit_mech.num_species}) :: e0_rt

        call get_species_internal_energies_rt(temperature, e0_rt)
        call get_mass_averaged_property(mass_fractions, e0_rt, e_mix)
        e_mix = e_mix * gas_constant * ${temperature_leading_term(bandit_mech)}

    end subroutine get_mixture_energy_mass

    subroutine get_species_specific_heats_cp_r(temperature, cp0_r)

        GPU_ROUTINE(get_species_specific_heats_cp_r)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: cp0_r

        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        cp0_r(${i+1}) = ${cgm(sp_thermo.cp_poly.expr)}
        %endfor

    end subroutine get_species_specific_heats_cp_r

    subroutine get_species_specific_heats_cv_r(temperature, cv0_r)

        GPU_ROUTINE(get_species_specific_heats_cv_r)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: cv0_r

        call get_species_specific_heats_cp_r(temperature, cv0_r)
        %for i in range(bandit_mech.num_species):
            cv0_r(${i+1}) = cv0_r(${i+1}) - 1.d0
        %endfor

    end subroutine get_species_specific_heats_cv_r

    subroutine get_species_enthalpies_rt(temperature, h0_rt)

        GPU_ROUTINE(get_species_enthalpies_rt)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: h0_rt

        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        h0_rt(${i+1}) = ${cgm(sp_thermo.enthalpy_poly.expr)}
        %endfor

    end subroutine get_species_enthalpies_rt

    subroutine get_species_internal_energies_rt(temperature, e0_rt)

        GPU_ROUTINE(get_species_internal_energies_rt)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: e0_rt

        call get_species_enthalpies_rt(temperature, e0_rt)
        %for i in range(bandit_mech.num_species):
            e0_rt(${i+1}) = e0_rt(${i+1}) - 1.d0
        %endfor

    end subroutine get_species_internal_energies_rt

    subroutine get_species_gibbs_rt(temperature, g0_rt)

        GPU_ROUTINE(get_species_gibbs_rt)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: g0_rt

        %for i, sp_thermo in enumerate(bandit_mech.species_nasa_thermo_polynomials):
        g0_rt(${i+1}) = ${cgm(sp_thermo.gibbs_poly.expr)}
        %endfor

    end subroutine get_species_gibbs_rt

    subroutine get_equilibrium_constants(temperature, k_eq)

        GPU_ROUTINE(get_equilibrium_constants)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_reactions}) :: k_eq

        ${real_type}, dimension(${bandit_mech.num_species}) :: gibbs_rt

        call get_species_gibbs_rt(temperature, gibbs_rt)

        %for i, expr in enumerate(bandit_mech.equil_constants):
        %if bandit_mech.is_reversible(i):
        k_eq(${i+1}) = ${cgm(expr)}
        %else:
        k_eq(${i+1}) = -0.17364695002734d0*${temperature_leading_term(bandit_mech)}
        %endif
        %endfor

    end subroutine get_equilibrium_constants

    %if bandit_mech.num_temp == 1:
    subroutine get_temperature( &
        & enthalpy_or_energy, t_guess, mass_fractions, do_energy, temperature)

        GPU_ROUTINE(get_temperature)

        logical, intent(in) :: do_energy
        ${real_type}, intent(in)  :: enthalpy_or_energy
        ${real_type}, intent(in)  :: t_guess
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out) :: temperature

        integer :: iter
        integer,      parameter :: num_iter = 500
        ${real_type}, parameter :: tol = 1.0d-06

        ${real_type} :: iter_temp
        ${real_type} :: iter_energy
        ${real_type} :: iter_energy_deriv
        ${real_type} :: iter_rhs
        ${real_type} :: iter_deriv

        iter_rhs = 0.d0
        iter_deriv = 1.d0
        iter_temp = t_guess

        do iter = 1, num_iter
            if(do_energy) then
                call get_mixture_specific_heat_cv_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_energy_mass(iter_temp, mass_fractions, iter_energy)
            else
                call get_mixture_specific_heat_cp_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_enthalpy_mass(&
                    & iter_temp, mass_fractions, iter_energy)
            endif
            iter_rhs = enthalpy_or_energy - iter_energy
            iter_deriv = (-1.d0)*iter_energy_deriv
            iter_temp = iter_temp - iter_rhs / iter_deriv
            if(abs(iter_rhs/iter_deriv) .lt. tol) exit
        end do

        temperature = iter_temp

    end subroutine get_temperature

    %endif
    %if bandit_mech.has_falloff_reactions():
    subroutine get_falloff_rates(&
        & temperature, concentrations, falloff_rate_coefficients)

        GPU_ROUTINE(get_falloff_rates)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            concentrations
        ${real_type}, intent(out), &
            dimension(${len(bandit_mech.falloff_reactions)}) :: &
            falloff_rate_coefficients

        ${real_type}, dimension(${len(bandit_mech.falloff_reactions)}) :: &
            k_high, k_low, falloff_center
        ${real_type}, dimension(${len(bandit_mech.falloff_reactions)}) :: &
            reduced_pressure, falloff_factor, falloff_function

        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        k_high(${i+1}) = ${cgm(falloff.high_rate_expr)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        k_low(${i+1}) = ${cgm(falloff.low_rate_expr)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        falloff_center(${i+1}) = ${cgm(falloff.falloff_center)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        reduced_pressure(${i+1}) = ${cgm(falloff.reduced_pressure)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        falloff_factor(${i+1}) = ${cgm(falloff.falloff_factor)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        falloff_function(${i+1}) = ${cgm(falloff.falloff_function)}
        %endfor
        %for i, falloff in enumerate(bandit_mech.falloff_reactions):
        falloff_rate_coefficients(${i+1}) = ${cgm(falloff.rate_coefficient)}
        %endfor

    end subroutine get_falloff_rates

    %endif
    subroutine get_fwd_rate_coefficients(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_fwd_rate_coefficients)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            concentrations
        ${real_type}, intent(out), dimension(${bandit_mech.num_reactions}) :: k_fwd
        %if bandit_mech.has_falloff_reactions():

        ${real_type}, dimension(${len(bandit_mech.falloff_reactions)}) :: &
            falloff_rate_coefficients

        call get_falloff_rates(&
            & temperature, concentrations, falloff_rate_coefficients)
        %endif

        %for i, rate_coeff in enumerate(bandit_mech.rate_coeffs):
        k_fwd(${i+1}) = ${cgm(rate_coeff.expr)}
        %endfor

    end subroutine get_fwd_rate_coefficients

    subroutine get_net_rates_of_progress(temperature, concentrations, r_net)

        GPU_ROUTINE(get_net_rates_of_progress)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            concentrations
        ${real_type}, intent(out), dimension(${bandit_mech.num_reactions}) :: r_net

        ${real_type}, dimension(${bandit_mech.num_reactions}) :: k_fwd
        ${real_type}, dimension(${bandit_mech.num_reactions}) :: log_k_eq

        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        %for i, expr in enumerate(bandit_mech.mass_action_rates):
        r_net(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_net_rates_of_progress

    subroutine get_net_production_rates(density, temperature, mass_fractions, omega)

        GPU_ROUTINE(get_net_production_rates)

        ${real_type}, intent(in) :: density
        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in),  dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: omega

        ${real_type}, dimension(${bandit_mech.num_species})   :: concentrations
        ${real_type}, dimension(${bandit_mech.num_reactions}) :: r_net

        call get_concentrations(density, mass_fractions, concentrations)
        call get_net_rates_of_progress(temperature, concentrations, r_net)

        %for i, expr in enumerate(bandit_mech.species_prod_rates):
        omega(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_net_production_rates

    %if opts.compute_jacobian:
    subroutine get_net_production_rates_jacobian(&
        & density, temperature, mass_fractions, jacobian)

        GPU_ROUTINE(get_net_production_rates_jacobian)

        ! jacobian(i, j) = d(omega(i)) / d(x(j)), with x ordered as
        ! [density, temperature(1..num_temperatures),
        ! mass_fractions(1..num_species)] -- see
        ! BaseMechanism._species_production_rate_jacobian_wrt_vars.
        ${real_type}, intent(in) :: density
        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(in),  dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${real_type}, intent(out), dimension( &
            ${bandit_mech.num_species}, &
            ${1 + bandit_mech.num_temp + bandit_mech.num_species}) :: jacobian

        %for i, row in enumerate(bandit_mech.species_production_rate_jacobian_exprs):
        %for j, entry in enumerate(row):
        jacobian(${i+1}, ${j+1}) = ${cgm(entry)}
        %endfor
        %endfor

    end subroutine get_net_production_rates_jacobian

    %endif
    %if bandit_mech.has_nonequilibrium_energy_modes():
    subroutine get_species_vibrational_energies(temperature, e_v)

        GPU_ROUTINE(get_species_vibrational_energies)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: e_v

        %for i, sp_thermo in enumerate(bandit_mech.species_vib_thermo_expressions):
        e_v(${i+1}) = ${cgm(sp_thermo.energy_expr)}
        %endfor

    end subroutine get_species_vibrational_energies

    subroutine get_species_vibrational_specific_heats(temperature, cv_v)

        GPU_ROUTINE(get_species_vibrational_specific_heats)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), dimension(${bandit_mech.num_species}) :: cv_v

        %for i, sp_thermo in enumerate(bandit_mech.species_vib_thermo_expressions):
        cv_v(${i+1}) = ${cgm(sp_thermo.specific_heat_expr)}
        %endfor

    end subroutine get_species_vibrational_specific_heats

    subroutine get_pressure_relaxation_times(temperature, ptau_vt)

        GPU_ROUTINE(get_pressure_relaxation_times)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), &
            dimension(${len(bandit_mech.pressure_relaxation_time_exprs)}) :: ptau_vt

        %for i, expr in enumerate(bandit_mech.pressure_relaxation_time_exprs):
        ptau_vt(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_pressure_relaxation_times

    subroutine get_vt_energy_transfer_source(&
        & density, mass_fractions, temperature, omega_vt)

        GPU_ROUTINE(get_vt_energy_transfer_source)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in), dimension(${bandit_mech.num_species}) :: &
            mass_fractions
        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out) :: omega_vt

        omega_vt = 0.d0
        %for expr in bandit_mech.vt_energy_transfer_exprs:
        omega_vt = omega_vt + ${cgm(expr)}
        %endfor

    end subroutine get_vt_energy_transfer_source

    subroutine get_translational_rotational_energy(temperature, tr_rot_energy)

        GPU_ROUTINE(get_translational_rotational_energy)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), &
            dimension(${bandit_mech.num_species}) :: tr_rot_energy

        %for i, expr in enumerate(bandit_mech.translational_rotational_energy_exprs):
        tr_rot_energy(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_translational_rotational_energy

    subroutine get_nasa_polynomial_vibrational_energy(temperature, vibe_energy)

        GPU_ROUTINE(get_nasa_polynomial_vibrational_energy)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), &
            dimension(${bandit_mech.num_species}) :: vibe_energy

        %for i, expr in enumerate(bandit_mech.nasa_polynomial_vibrational_energy_exprs):
        vibe_energy(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_nasa_polynomial_vibrational_energy

    subroutine get_nasa_polynomial_vibrational_specific_heat(&
        & temperature, vibe_specific_heat)

        GPU_ROUTINE(get_nasa_polynomial_vibrational_specific_heat)

        ${temperature_decl(real_type, bandit_mech)}
        ${real_type}, intent(out), &
            dimension(${bandit_mech.num_species}) :: vibe_specific_heat

        %for i, expr in enumerate(bandit_mech.nasa_polynomial_vibrational_specific_heat_exprs):
        vibe_specific_heat(${i+1}) = ${cgm(expr)}
        %endfor

    end subroutine get_nasa_polynomial_vibrational_specific_heat

    %endif
end module ${module_name}
""")

# }}}


class FortranBanditCodeGenerator(CodeGenerator):
    @staticmethod
    def get_name() -> str:
        return "fortran"

    @staticmethod
    def supports_overloading() -> bool:
        return False

    @staticmethod
    def generate(name: str,
                 bandit_mech: BaseMechanism,
                 opts: CodeGenerationOptions = None) -> str:
        if opts is None:
            opts = CodeGenerationOptions()

        if opts.directive_offload == "acc":
            gpu_routine_str = """
#ifdef _CRAYFTN
#define GPU_ROUTINE(name) !DIR$ INLINEALWAYS name
#else
#define GPU_ROUTINE(name) !$acc routine seq
#endif
"""
        elif opts.directive_offload == "mp":
            gpu_routine_str = """
#define GPU_ROUTINE(name) !$omp declare target
"""
        else:
            gpu_routine_str = """
#define GPU_ROUTINE(name) ! name
"""

        if opts.compute_jacobian:
            bandit_mech.make_species_production_rate_jacobian()

        return wrap_code(module_tpl.render(
            bandit_mech=bandit_mech,
            opts=opts,

            str_np=str_np,
            cgm=FortranExpressionMapper(),
            Variable=p.Variable,
            float_to_fortran=float_to_fortran,
            temperature_decl=temperature_decl,
            temperature_leading_term=temperature_leading_term,

            real_type=opts.scalar_type or "real(dp)",
            gpu_routine=gpu_routine_str,

            module_name=name,
        ))


# vim: foldmethod=marker
