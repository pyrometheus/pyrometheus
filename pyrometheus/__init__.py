"""
.. automodule:: pyrometheus.chem_expr
.. automodule:: pyrometheus.codegen.python
.. automodule:: pyrometheus.codegen.cpp
.. automodule:: pyrometheus.codegen.fortran90
.. automodule:: pyrometheus.codegen.fortranacc
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


import pyrometheus.codegen.python  # noqa: F401
import pyrometheus.codegen.cpp  # noqa: F401
import pyrometheus.codegen.fortran90  # noqa: F401
import pyrometheus.codegen.fortranacc  # noqa: F401
import pyrometheus.codegen.python as _py


# {{{ handle deprecations

def gen_thermochem_code(*args, **kwargs):
    from warnings import warn
    warn("get_thermochem_code should be imported from pyrometheus.codegen.python "
            "now. This alias in the root will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return _py.gen_thermochem_code(*args, **kwargs)


def get_thermochem_class(*args, **kwargs):
    from warnings import warn
    warn("get_thermochem_class should be imported from pyrometheus.codegen.python "
            "now. This alias in the root will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return _py.get_thermochem_class(*args, **kwargs)


def cti_to_mech_file(*args, **kwargs):
    from warnings import warn
    warn("cti_to_mech_file should be imported from pyrometheus.codegen.python "
            "now. This alias in the root will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return _py.cti_to_mech_file(*args, **kwargs)

# }}}

# vim: foldmethod=marker
