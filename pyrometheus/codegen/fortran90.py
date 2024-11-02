"""
Fortran code generation
-----------------------

.. autofunction:: gen_thermochem_code
"""

import shlex
from functools import partial
from numbers import Number

import numpy as np  # noqa: F401
import cantera as ct
import pymbolic.primitives as p
from mako.template import Template
import pyrometheus.chem_expr
from pymbolic.mapper.stringifier import (
        StringifyMapper, PREC_NONE, PREC_CALL, PREC_PRODUCT)


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


def str_np_inner(ary):
    if isinstance(ary, Number):
        return float_to_fortran(ary)
    elif ary.shape:
        return "(%s)" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return ", ".join(float_to_fortran(entry) for entry in ary)

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
        if isinstance(expr.index, tuple):
            index_str = ", ".join(
                    "int(%s)" % self.rec(i, PREC_NONE)
                    for i in expr.index)
        else:
            index_str = "%s" % (expr.index+1)  # self.rec(expr.index, PREC_NONE)

        return self.parenthesize_if_needed(
                self.format("%s(%s)",
                    self.rec(expr.aggregate, PREC_CALL),
                    index_str),
                enclosing_prec, PREC_CALL)

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

#ifdef _CRAYFTN
#define GPU_ROUTINE(name) !DIR$ INLINEALWAYS name
#else
#define GPU_ROUTINE(name) !$acc routine seq
#endif


module ${module_name}

    implicit none

    integer, parameter :: sp = selected_real_kind(6,37)   ! Single precision
    integer, parameter :: dp = selected_real_kind(15,307) ! Double precision

    integer, parameter :: num_elements = ${sol.n_elements}
    integer, parameter :: num_species = ${sol.n_species}
    integer, parameter :: num_reactions = ${sol.n_reactions}
    integer, parameter :: num_falloff = ${len(falloff_reactions)}
    ${real_type}, parameter :: one_atm = ${float_to_fortran(ct.one_atm)}
    ${real_type}, parameter :: gas_constant = ${float_to_fortran(ct.gas_constant)}
    ${real_type}, parameter :: mol_weights(${sol.n_species}) = &
        (/ ${str_np(sol.molecular_weights)} /)
    ${real_type}, parameter :: inv_weights(${sol.n_species}) = &
        (/ ${str_np(1/sol.molecular_weights)} /)

    !$acc declare create(mol_weights, inv_weights)

    character(len=12), parameter :: species_names(${sol.n_species}) = &
        (/ ${", ".join('"'+'{0: <12}'.format(s)+'"' for s in sol.species_names)} /)

    character(len=4), parameter :: element_names(${sol.n_elements}) = &
        (/ ${", ".join('"'+'{0: <4}'.format(e)+'"' for e in sol.element_names)} /)

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

    subroutine get_element_index(el_name, el_index)

        character(len=*), intent(in) :: el_name
        integer, intent(out) :: el_index

        integer :: idx

        el_index = 0
        loop:do idx = 1, num_elements
            if(trim(adjustl(el_name)) .eq. trim(element_names(idx))) then
                el_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_element_index

    subroutine get_specific_gas_constant(mass_fractions, specific_gas_constant)

        GPU_ROUTINE(get_specific_gas_constant)

        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: specific_gas_constant

        specific_gas_constant = gas_constant * ( &
                %for i in range(sol.n_species):
                    + inv_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

    end subroutine get_specific_gas_constant

    subroutine get_density(pressure, temperature, mass_fractions, density)

        GPU_ROUTINE(get_density)

        ${real_type}, intent(in) :: pressure
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: density

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        density = pressure * mix_mol_weight / (gas_constant * temperature)

    end subroutine get_density

    subroutine get_pressure(density, temperature, mass_fractions, pressure)

        GPU_ROUTINE(get_pressure)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: pressure

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        pressure = density * gas_constant * temperature / mix_mol_weight

    end subroutine get_pressure

    subroutine get_mixture_molecular_weight(mass_fractions, mix_mol_weight)

        GPU_ROUTINE(get_mixture_molecular_weight)

        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: mix_mol_weight

        mix_mol_weight = 1.0d0 / ( &
                %for i in range(sol.n_species):
                    + inv_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

    end subroutine get_mixture_molecular_weight

    subroutine get_concentrations(density, mass_fractions, concentrations)

        GPU_ROUTINE(get_concentrations)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in),  dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out), dimension(${sol.n_species}) :: concentrations

        concentrations = density * inv_weights * mass_fractions

    end subroutine get_concentrations

    subroutine get_mass_averaged_property(&
        & mass_fractions, spec_property, mix_property)

        GPU_ROUTINE(get_mass_averaged_property)

        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(in), dimension(${sol.n_species}) :: spec_property
        ${real_type}, intent(out) :: mix_property

        mix_property = ( &
            %for i in range(sol.n_species):
                + inv_weights(${i+1})*mass_fractions(${i+1})*spec_property(${i+1}) &
            %endfor
        )

    end subroutine get_mass_averaged_property

    subroutine get_mixture_specific_heat_cp_mass(temperature, mass_fractions, cp_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cp_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: cp_mix

        ${real_type}, dimension(${sol.n_species}) :: cp0_r

        call get_species_specific_heats_r(temperature, cp0_r)
        call get_mass_averaged_property(mass_fractions, cp0_r, cp_mix)
        cp_mix = cp_mix * gas_constant

    end subroutine get_mixture_specific_heat_cp_mass

    subroutine get_mixture_specific_heat_cv_mass(temperature, mass_fractions, cv_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cv_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: cv_mix

        ${real_type}, dimension(${sol.n_species}) :: cp0_r

        call get_species_specific_heats_r(temperature, cp0_r)
        cp0_r(:) = cp0_r(:) - 1.d0
        call get_mass_averaged_property(mass_fractions, cp0_r, cv_mix)
        cv_mix = cv_mix * gas_constant

    end subroutine get_mixture_specific_heat_cv_mass

    subroutine get_mixture_enthalpy_mass(temperature, mass_fractions, h_mix)

        GPU_ROUTINE(get_mixture_enthalpy_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: h_mix

        ${real_type}, dimension(${sol.n_species}) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_mass_averaged_property(mass_fractions, h0_rt, h_mix)
        h_mix = h_mix * gas_constant * temperature

    end subroutine get_mixture_enthalpy_mass

    subroutine get_mixture_energy_mass(temperature, mass_fractions, e_mix)

        GPU_ROUTINE(get_mixture_energy_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out) :: e_mix

        ${real_type}, dimension(${sol.n_species}) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        h0_rt(:) = h0_rt - 1.d0
        call get_mass_averaged_property(mass_fractions, h0_rt, e_mix)
        e_mix = e_mix * gas_constant * temperature

    end subroutine get_mixture_energy_mass

    subroutine get_species_specific_heats_r(temperature, cp0_r)

        GPU_ROUTINE(get_species_specific_heats_r)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${sol.n_species}) :: cp0_r

        %for i, sp in enumerate(sol.species()):
        cp0_r(${i+1}) = ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))}
        %endfor

    end subroutine get_species_specific_heats_r

    subroutine get_species_enthalpies_rt(temperature, h0_rt)

        GPU_ROUTINE(get_species_enthalpies_rt)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${sol.n_species}) :: h0_rt

        %for i, sp in enumerate(sol.species()):
        h0_rt(${i+1}) = ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))}
        %endfor

    end subroutine get_species_enthalpies_rt

    subroutine get_species_entropies_r(temperature, s0_r)

        GPU_ROUTINE(get_species_entropies_r)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${sol.n_species}) :: s0_r

        %for i, sp in enumerate(sol.species()):
        s0_r(${i+1}) = ${cgm(ce.poly_to_entropy_expr(sp.thermo, "temperature"))}
        %endfor

    end subroutine get_species_entropies_r

    subroutine get_species_gibbs_rt(temperature, g0_rt)

        GPU_ROUTINE(get_species_gibbs_rt)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${sol.n_species}) :: g0_rt

        ${real_type}, dimension(${sol.n_species}) :: h0_rt
        ${real_type}, dimension(${sol.n_species}) :: s0_r

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_species_entropies_r(temperature, s0_r)
        g0_rt(:) = h0_rt(:) - s0_r(:)

    end subroutine get_species_gibbs_rt

    subroutine get_equilibrium_constants(temperature, k_eq)

        !$acc routine seq

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${sol.n_reactions}) :: k_eq

        ${real_type} :: rt
        ${real_type} :: c0

        ${real_type}, dimension(${sol.n_species}) :: g0_rt

        rt = gas_constant * temperature
        c0 = log(one_atm/rt)

        call get_species_gibbs_rt(temperature, g0_rt)

        %for i, react in enumerate(sol.reactions()):
        %if react.reversible:
        k_eq(${i+1}) = ${cgm(
            ce.equilibrium_constants_expr(sol, i, Variable("g0_rt")))}
        %else:
        k_eq(${i+1}) = -0.1d0*temperature
        %endif
        %endfor

    end subroutine get_equilibrium_constants

    subroutine get_temperature( &
        & enthalpy_or_energy, t_guess, mass_fractions, do_energy, temperature)

        GPU_ROUTINE(get_temperature)

        logical, intent(in) :: do_energy
        ${real_type}, intent(in)  :: enthalpy_or_energy
        ${real_type}, intent(in)  :: t_guess
        ${real_type}, intent(in), dimension(${sol.n_species}) :: mass_fractions
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

    %if falloff_reactions:
    subroutine get_falloff_rates(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_falloff_rates)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: concentrations
        ${real_type}, intent(out), dimension(${sol.n_reactions}) :: k_fwd

        ${real_type}, dimension(${len(falloff_reactions)}) :: k_high
        ${real_type}, dimension(${len(falloff_reactions)}) :: k_low
        ${real_type}, dimension(${len(falloff_reactions)}) :: reduced_pressure
        ${real_type}, dimension(${len(falloff_reactions)}) :: falloff_center
        ${real_type}, dimension(${len(falloff_reactions)}) :: falloff_factor
        ${real_type}, dimension(${len(falloff_reactions)}) :: falloff_function

        %for i, (_, react) in enumerate(falloff_reactions):
        k_high(${i+1}) = ${cgm(ce.rate_coefficient_expr(
                                react.rate.high_rate,
                                Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        k_low(${i+1}) = ${cgm(ce.rate_coefficient_expr(
                                react.rate.low_rate,
                                Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        reduced_pressure(${i+1}) = (${cgm(
            ce.third_body_efficiencies_expr(sol,
                                            react,
                                            Variable("concentrations")))})*k_low(${i+1})/k_high(${i+1})
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_center(${i+1}) = ${cgm(ce.troe_falloff_center_expr(
            react, Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_factor(${i+1}) = ${cgm(ce.troe_falloff_factor_expr(react, i,
            Variable("reduced_pressure"), Variable("falloff_center")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_function(${i+1}) = ${cgm(ce.falloff_function_expr(
            react, i,
            Variable("falloff_factor"),
            Variable("falloff_center")))}
        %endfor

        %for i, (j, react) in enumerate(falloff_reactions):
        k_fwd(${j+1}) = k_high(${i+1})*falloff_function(${i+1}) * &
            reduced_pressure(${i+1})/(1.d0 + reduced_pressure(${i+1}))
        %endfor

    end subroutine get_falloff_rates

    %endif
    subroutine get_fwd_rate_coefficients(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_fwd_rate_coefficients)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: concentrations
        ${real_type}, intent(out), dimension(${sol.n_reactions}) :: k_fwd

        %if falloff_reactions:
        ${real_type}, dimension(${len(falloff_reactions)}) :: k_falloff
        %endif

        %for i, react in enumerate(sol.reactions()):
        %if react.equation in [r.equation for _, r in falloff_reactions]:
        k_fwd(${i+1}) = 0.d0
        %else:
        k_fwd(${i+1}) = ${cgm(ce.rate_coefficient_expr(react.rate,
                            Variable("temperature")))}
        %endif
        %endfor

        %for j, react in three_body_reactions:
        k_fwd(${j+1}) = k_fwd(${j+1}) * ( &
            ${cgm(ce.third_body_efficiencies_expr(
            sol, react, Variable("concentrations")))})
        %endfor

        %if falloff_reactions:
        call get_falloff_rates(temperature, concentrations, k_fwd)
        %endif

    end subroutine get_fwd_rate_coefficients

    subroutine get_net_rates_of_progress(temperature, concentrations, r_net)

        GPU_ROUTINE(get_net_rates_of_progress)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${sol.n_species}) :: concentrations
        ${real_type}, intent(out), dimension(${sol.n_reactions}) :: r_net

        ${real_type}, dimension(${sol.n_reactions}) :: k_fwd
        ${real_type}, dimension(${sol.n_reactions}) :: log_k_eq

        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        %for i in range(sol.n_reactions):
        r_net(${i+1}) = ${cgm(ce.rate_of_progress_expr(sol, i,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("log_k_eq")))}
        %endfor

    end subroutine get_net_rates_of_progress

    subroutine get_net_production_rates(density, temperature, mass_fractions, omega)

        GPU_ROUTINE(get_net_production_rates)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in),  dimension(${sol.n_species}) :: mass_fractions
        ${real_type}, intent(out), dimension(${sol.n_species}) :: omega

        ${real_type}, dimension(${sol.n_species})   :: concentrations
        ${real_type}, dimension(${sol.n_reactions}) :: r_net

        call get_concentrations(density, mass_fractions, concentrations)
        call get_net_rates_of_progress(temperature, concentrations, r_net)

        %for i, sp in enumerate(sol.species()):
        omega(${i+1}) = ${cgm(ce.production_rate_expr(sol,
            sp.name, Variable("r_net")))}
        %endfor

    end subroutine get_net_production_rates

end module
""")

# }}}


def gen_thermochem_code(sol: ct.Solution, real_type="real(dp)",
        module_name="thermochem") -> str:
    """For the mechanism given by *sol*, return Fortran source code.
    """

    falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                   if r.reaction_type.startswith("falloff")]
    three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                      if r.reaction_type == "three-body-Arrhenius"]

    return wrap_code(module_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        cgm=FortranExpressionMapper(),
        Variable=p.Variable,
        float_to_fortran=float_to_fortran,

        real_type=real_type,
        module_name=module_name,

        ce=pyrometheus.chem_expr,

        falloff_reactions=falloff_rxn,
        three_body_reactions=three_body_rxn
    ))


# vim: foldmethod=marker
