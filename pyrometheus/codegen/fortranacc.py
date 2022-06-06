"""
Fortran-acc code generation
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
from itertools import compress, product


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
    result = repr(num).replace("e", "d")
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
module ${module_name}

    implicit none
    integer, parameter :: num_elements = ${sol.n_elements}
    integer, parameter :: num_species = ${sol.n_species}
    integer, parameter :: num_reactions = ${sol.n_reactions}
    integer, parameter :: num_falloff = ${
        sum(1 if isinstance(r, ct.FalloffReaction) else 0
        for r in sol.reactions())}
    ${real_type}, parameter :: one_atm = ${float_to_fortran(ct.one_atm)}
    ${real_type}, parameter :: gas_constant = ${float_to_fortran(ct.gas_constant)}
    ${real_type}, parameter :: mol_weights(*) = &
        (/ ${str_np(sol.molecular_weights)} /)
    ${real_type}, parameter :: inv_weights(*) = &
        (/ ${str_np(1/sol.molecular_weights)} /)

    ${real_type}, parameter :: arrhenius_params(num_reactions, 3) = reshape(&
        (/ ${str_np(arrhenius_coefficients)} /), &
        (/${sol.n_reactions}, 3/))

    ${real_type}, parameter :: stoich_matrix_fwd(num_species, num_reactions) = reshape(&
        (/ ${str_np(nu_fwd)} /), &
        (/${sol.n_species}, ${sol.n_reactions}/))

    ${real_type}, parameter :: stoich_matrix_rev(num_species, num_reactions) = reshape(&
        (/ ${str_np(nu_rev)} /), &
        (/${sol.n_species}, ${sol.n_reactions}/))

    character(len=12), parameter :: species_names(*) = &
        (/ ${", ".join('"'+'{0: <12}'.format(s)+'"' for s in sol.species_names)} /)

    character(len=4), parameter :: element_names(*) = &
        (/ ${", ".join('"'+'{0: <4}'.format(e)+'"' for e in sol.element_names)} /)

    ${real_type}, allocatable, dimension(:) :: concentrations
    ${real_type}, allocatable, dimension(:) :: k_fwd
    !$acc declare create(concentrations, k_fwd)

contains

    subroutine initialize_thermochem

        allocate(concentrations(num_species))
        allocate(k_fwd(num_reactions))
        !$acc enter data create(concentrations(num_species), &
        !$acc    k_fwd(num_reactions))

        concentrations = 0.d0
        !$acc update device(concentrations)

    end subroutine initialize_thermochem

    subroutine get_species_name(sp_index, sp_name)

        integer, intent(in) :: sp_index
        character(len=*), intent(out) :: sp_name
        
        sp_name = species_names(sp_index)

    end subroutine get_species_name

    subroutine get_species_index(sp_name, sp_index)

        character(len=*), intent(in) :: sp_name
        integer, intent(out) :: sp_index

        integer :: idx

        sp_index = 0
        loop:do idx = 1, num_species
            if(trim(adjustl(sp_name)) .eq. trim(species_names(idx))) then
                sp_index = idx
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
                el_index = idx
                exit loop
            end if
        end do loop

    end subroutine get_element_index

    subroutine get_density(num_x, num_y, num_z, pressure, temperature, mass_fractions, density)

        integer, intent(in) :: num_x
        integer, intent(in) :: num_y
        integer, intent(in) :: num_z
        ${real_type}, intent(in) :: pressure
        ${real_type}, intent(in), dimension(num_x, num_y, num_y) :: temperature
        ${real_type}, intent(in), dimension(num_species, num_x, num_y, num_z) :: mass_fractions
        ${real_type}, intent(out), dimension(num_x, num_y, num_z) :: density

        integer :: i, j, k, s
        ${real_type} :: mix_mol_weight

        !$acc parallel loop collapse(3) vector_length(32)
        do k = 1, num_z
            do j = 1, num_y
                do i = 1, num_x
                    mix_mol_weight = 0.d0
                    !$acc loop vector
                    do s = 1, num_species
                        mix_mol_weight = mix_mol_weight + &
                            inv_weights(s) * mass_fractions(s, i, j, k)
                    end do                    
                    density(i, j, k) = pressure / (gas_constant * &
                        mix_mol_weight * temperature(i, j, k))
                end do
            end do
        end do

    end subroutine get_density

    subroutine get_fwd_rate_coefficients(num_x, num_y, num_z, density, temperature, mass_fractions, k_fwd)

        integer, intent(in) :: num_x
        integer, intent(in) :: num_y
        integer, intent(in) :: num_z
        ${real_type}, intent(in), dimension(num_x, num_y, num_y) :: density
        ${real_type}, intent(in), dimension(num_x, num_y, num_y) :: temperature
        ${real_type}, intent(in), dimension(num_species, num_x, num_y, num_z) :: mass_fractions
        ${real_type}, intent(in), dimension(num_reactions, num_x, num_y, num_z) :: k_fwd

        integer :: i, j, k, s, r
        ${real_type} :: inv_temp, log_temp

        !$acc parallel loop collapse(3) vector_length(32)
        do k = 1, num_z
            do j = 1, num_y
                do i = 1, num_x
                    inv_temp = 1.d0 / temperature(i, j, k)
                    log_temp = log(temperature(i, j, k))
                    !$acc loop vector
                    do r = 1, num_reactions
                        k_fwd(r, i, j, k) = exp(arrhenius_params(r, 1) + &
                            arrhenius_params(r, 2)*log(temperature(i, j, k)) + &
                            arrhenius_params(r, 3)/temperature(i, j, k))
                    end do
                    ! [ECG] Need to add concentration dependence
                end do
            end do
        end do

    end subroutine get_fwd_rate_coefficients

    subroutine get_fwd_rates_of_progress(num_x, num_y, num_z, density, temperature, mass_fractions, r_fwd)

        integer, intent(in) :: num_x
        integer, intent(in) :: num_y
        integer, intent(in) :: num_z
        ${real_type}, intent(in), dimension(num_x, num_y, num_y) :: density
        ${real_type}, intent(in), dimension(num_x, num_y, num_y) :: temperature
        ${real_type}, intent(in), dimension(num_species, num_x, num_y, num_z) :: mass_fractions        
        ${real_type}, intent(out), dimension(num_reactions, num_x, num_y, num_z) :: r_fwd

        integer :: i, j, k, s, r
        ${real_type} :: inv_temp, log_temp

        !$acc parallel loop collapse(3) vector_length(32)
        do k = 1, num_z
            do j = 1, num_y
                do i = 1, num_x
                    inv_temp = 1.d0 / temperature(i, j, k)
                    log_temp = log(temperature(i, j, k))
                    !$acc loop vector
                    do s = 1, num_species
                        concentrations(s) = density(i, j, k) * mass_fractions(s, i, j, k) * inv_weights(s)
                    end do
                    !$acc end loop
                    !$acc loop vector
                    do r = 1, num_reactions
                        r_fwd(r, i, j, k) = exp(arrhenius_params(r, 1) + &
                            arrhenius_params(r, 2)*log_temp + &
                            arrhenius_params(r, 3)*inv_temp)
                        !$acc loop seq
                        do s = 1, num_species
                            r_fwd(r, i, j, k) = r_fwd(r, i, j, k) * concentrations(s)**stoich_matrix_fwd(s, r)
                        end do
                    end do
                    !$acc end loop
                end do
            end do
        end do

    end subroutine get_fwd_rates_of_progress

end module
""")

# }}}


def gen_thermochem_code(sol: ct.Solution, real_type="real(kind(1.d0))",
        module_name="thermochem") -> str:
    """For the mechanism given by *sol*, return Python source code for a class conforming
    to a module containing a class called ``Thermochemistry`` adhering to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
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

        nu_fwd=np.array([sol.reactant_stoich_coeff(s, r)
                         for s, r in product(range(sol.n_species),
                                             range(sol.n_reactions))]),

        nu_rev=np.array([sol.product_stoich_coeff(s, r)
                         for s, r in product(range(sol.n_species),
                                             range(sol.n_reactions))]),
        
        elem_matrix=np.array([sol.n_atoms(i, j)
                              for i, j in product(range(sol.n_species),
                                                  range(sol.n_elements))]),

        arrhenius_coefficients=np.array([[np.log(r.rate.pre_exponential_factor),
                                          r.rate.temperature_exponent,
                                          r.rate.activation_energy/ct.gas_constant]
                                         if not isinstance(r, ct.FalloffReaction)
                                         else [1, 0, 0] for r in sol.reactions()]).ravel(),

        falloff_reactions=list(compress(sol.reactions(),
                                        [isinstance(r, ct.FalloffReaction)
                                         for r in sol.reactions()])),
        non_falloff_reactions=list(compress(sol.reactions(),
                                            [not isinstance(r, ct.FalloffReaction)
                                             for r in sol.reactions()])),
        three_body_reactions=list(compress(sol.reactions(),
                                           [isinstance(r, ct.ThreeBodyReaction)
                                            for r in sol.reactions()])),
    ))


# vim: foldmethod=marker
