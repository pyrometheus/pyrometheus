"""
.. automodule:: pyrometheus.chem_expr
.. automodule:: pyrometheus.codegen.python
.. automodule:: pyrometheus.codegen.cpp
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

from .codegen import CodeGenerator, CodeGenerationOptions  # noqa: F401
from .codegen.python import PythonCodeGenerator
from .codegen.cpp import CppCodeGenerator
from .codegen.fortran import FortranCodeGenerator


def get_code_generators() -> typing.Dict[str, CodeGenerator]:
    return {
        PythonCodeGenerator.get_name(): PythonCodeGenerator,
        CppCodeGenerator.get_name(): CppCodeGenerator,
        FortranCodeGenerator.get_name(): FortranCodeGenerator,
    }


# Python is the default code generator, expose its CodeGenerator's static methods
# globally in this module.
for name, method in PythonCodeGenerator.__dict__.items():
    if isinstance(method, staticmethod):
        globals()[name] = method
