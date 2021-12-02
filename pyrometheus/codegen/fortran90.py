"""
Fortran code generation
----------------------

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
from itertools import compress


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
            index_str = "int(%s)" % self.rec(expr.index, PREC_NONE)

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

# }}}


# {{{ module template

module_tpl = Template("""
module ${module_name}
    implicit none

    integer, parameter :: num_species = ${sol.n_species}
    ${real_type}, parameter :: gas_constant = ${float_to_fortran(ct.gas_constant)}
    ${real_type}, parameter :: mol_weights(*) = &
        (/ ${str_np(sol.molecular_weights)} /)
    ${real_type}, parameter :: inv_weights(*) = &
        (/ ${str_np(1/sol.molecular_weights)} /)

contains
    function get_specific_gas_constant(mass_fractions)
        ${real_type} get_specific_gas_constant
        ${real_type} mass_fractions(num_species)

        get_specific_gas_constant = gas_constant * ( &
                %for i in range(sol.n_species):
                    + inv_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )
    end
end module
""")

# }}}


def gen_thermochem_code(sol: ct.Solution, real_type="real*8",
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
