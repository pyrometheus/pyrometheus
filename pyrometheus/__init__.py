"""
.. automodule:: pyrometheus.chem_expr
.. automodule:: pyrometheus.codegen.python
.. automodule:: pyrometheus.codegen.cpp
.. automodule:: pyrometheus.codegen.cpp_bandit
.. automodule:: pyrometheus.codegen.fortran
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

import typing
import warnings

from .bandit.general_thermochem import BaseMechanism
from .bandit.impl.cantera import CanteraMechanism
from .codegen import CodeGenerator, CodeGenerationOptions  # noqa: F401
from .codegen.python import PythonCodeGenerator
from .codegen.python_bandit import PythonBanditCodeGenerator
from .codegen.cpp import CppCodeGenerator
from .codegen.cpp_bandit import CppBanditCodeGenerator
from .codegen.fortran import FortranCodeGenerator
from .codegen.fortran_bandit import FortranBanditCodeGenerator


def get_code_generators() -> typing.Dict[str, CodeGenerator]:
    """:returns: Every code generator, keyed by name. The language names
        lead to the Bandit generators; the 1.x generators, deprecated,
        answer to ``<language>-legacy``.
    """
    return {
        generator.get_name(): generator
        for generator in (
            PythonBanditCodeGenerator,
            CppBanditCodeGenerator,
            FortranBanditCodeGenerator,
            PythonCodeGenerator,
            CppCodeGenerator,
            FortranCodeGenerator,
        )
    }


# {{{ the default generator, Bandit's Python one, at module level

def _warn_legacy_solution():
    warnings.warn(
        "Generating code from a cantera.Solution uses the deprecated 1.x "
        "generator; pass CanteraMechanism(solution) to use Bandit's.",
        DeprecationWarning, stacklevel=3,
    )


def generate(name, mechanism, opts=None) -> str:
    """:returns: Python source for *mechanism*, a
        :class:`~pyrometheus.bandit.general_thermochem.BaseMechanism`. A
        :class:`cantera.Solution` still works, through the deprecated 1.x
        generator.
    """
    if isinstance(mechanism, BaseMechanism):
        return PythonBanditCodeGenerator.generate(name, mechanism, opts)
    _warn_legacy_solution()
    return PythonCodeGenerator.generate(name, mechanism, opts)


def get_thermochem_class(mechanism):
    """:returns: The thermochemistry class generated for *mechanism*, as
        :func:`generate` describes.
    """
    if isinstance(mechanism, BaseMechanism):
        return PythonBanditCodeGenerator.get_thermochem_class(mechanism)
    _warn_legacy_solution()
    return PythonCodeGenerator.get_thermochem_class(mechanism)


get_name = PythonBanditCodeGenerator.get_name
compile_class = PythonBanditCodeGenerator.compile_class
supports_overloading = PythonBanditCodeGenerator.supports_overloading

# }}}
