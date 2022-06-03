""":mod:`pyrometheus.mechanisms`: Utilities for managing thermochemistry mechanisms.

.. autofunction:: get_mechanisms_pkgname
.. autofunction:: get_mechanism_config_file_name
.. autofunction:: get_mechanism_config
.. autofunction:: import_mechdata
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
import sys
if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources  # pylint: disable=import-error
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


def get_mechanisms_pkgname() -> str:
    """Get a qualified package name for the location of mechanism data."""
    return "pyrometheus.mechanisms"


def get_mechanism_config_file_name(mechanism_name: str) -> str:
    """Form the mechanism config file name for a mechanism."""
    return f"{mechanism_name}.yaml"


def import_mechdata():
    """Import the mechanism data as a mechanism data resource.

    Returns
    -------
    :class:`importlib.abc.Traversable`
        Object of type :class:`importlib.abc.Traversable` representing the container
        (think directory) of the thermochemistry mechanism data (think yaml files).
    """
    return importlib_resources.files(get_mechanisms_pkgname())


def get_mechanism_config(mechanism_name: str) -> str:
    """Get the contents of a mechanism config file."""
    mech_data = import_mechdata()
    mech_file = mech_data / get_mechanism_config_file_name(mechanism_name)
    return mech_file.read_text()
