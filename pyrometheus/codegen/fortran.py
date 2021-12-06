"""
Fortran code generation
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


file_extension = "f90"


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

    constexpr static ScalarT mol_weights[] = {${str_np(sol.molecular_weights)}};
    constexpr static ScalarT inv_weights[] = {${str_np(1/sol.molecular_weights)}};
    constexpr static ScalarT gas_constant = ${repr(ct.gas_constant)};

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

    static ScalarT get_density(ScalarT p, ScalarT temperature,
            ContainerT const &mass_fractions)
    {
        ScalarT mmw = get_mix_molecular_weight(mass_fractions);
        ScalarT rt = gas_constant * temperature;
        return p * mmw / rt;
    }

    static ContainerT get_species_specific_heats_r(ScalarT temperature)
    {
        ContainerT result = {
            % for sp in sol.species():
            ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))},
            % endfor
            };
        return result;
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
