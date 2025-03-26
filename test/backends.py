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

import cantera as ct
import importlib
import pathlib
import typing

import pytest

from abc import abstractmethod

from pyrometheus.codegen import CodeGenerator
from pyrometheus.codegen.python import PythonCodeGenerator
from pyrometheus.codegen.cpp import CppCodeGenerator
from pyrometheus.codegen.fortran import FortranCodeGenerator

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jnp = None
else:
    jax.config.update("jax_enable_x64", True)


class Backend(CodeGenerator):
    """Abstract class exposing methods to interact with Pyrometheus' different
    implementations ("backends")."""

    def __init__(self, pyro):
        self.pyro = pyro

    @staticmethod
    @abstractmethod
    def extract_interface(module, mechname: str, user_np):
        """Given a loaded Python module of the generated code, returns an
        instantiated interface object for the Pyrometheus class contained
        therewithin.

        Parameters
        ----------
        module : module
            The loaded Python module containing the generated code.
        mechname : str
            The name of the mechanism.
        user_np : module
            The numpy-like module to use for array operations within the
            generated code.
        """
        return module


class PythonBackend(Backend, PythonCodeGenerator):
    @staticmethod
    def _gen_jax_wrapper(base_cls):
        class JaxPyroWrapper(base_cls):
            def _pyro_make_array(self, res_list):
                """
                This works around (e.g.) numpy.exp not working with
                object arrays of :mod:`numpy` scalars. It defaults to
                making object arrays, however if an array consists of
                all scalars, it makes a "plain old"
                :class:`numpy.ndarray`. See this
                ``numpy bug <https://github.com/numpy/numpy/issues/18004>`__
                for more context.

                """

                from numbers import Number
                # Needed to play nicely with Jax, which frequently creates
                # arrays of shape () when handed numbers
                all_numbers = all(
                    isinstance(e, Number)
                    or (isinstance(e, self.pyro_np.ndarray) and e.shape == ())
                    for e in res_list)

                if all_numbers:
                    return self.pyro_np.array(
                        res_list, dtype=self.pyro_np.float64
                    )

                result = self.pyro_np.empty_like(
                    self.pyro_np.array(res_list),
                )

                # 'result[:] = res_list' may look tempting, however:
                # https://github.com/numpy/numpy/issues/16564
                for idx in range(len(res_list)):
                    result = result.at[idx].set(res_list[idx])

                return result

            def _pyro_norm(self, argument, normord):
                """
                This works around numpy.linalg norm not working
                with scalars.  If the argument is a regular ole
                number, it uses :func:`numpy.abs`, otherwise it uses
                ``pyro_np.linalg.norm``.
                """
                # Wrap norm for scalars
                from numbers import Number
                if isinstance(argument, Number):
                    return self.pyro_np.abs(argument)
                # Needed to play nicely with Jax, which frequently creates
                # arrays of shape () when handed numbers
                if (isinstance(argument, self.pyro_np.ndarray)
                        and argument.shape == ()):
                    return self.pyro_np.abs(argument)
                return self.pyro_np.linalg.norm(argument, normord)

        return JaxPyroWrapper

    @staticmethod
    def extract_interface(module, mechname: str, user_np):
        base_cls = module.Thermochemistry

        if user_np == jnp:
            base_cls = PythonBackend._gen_jax_wrapper(base_cls)

        return base_cls(user_np)


class CppBackend(Backend, CppCodeGenerator):
    pass


class FortranBackend(Backend, FortranCodeGenerator):
    @staticmethod
    def extract_interface(module, mechname: str, user_np):
        return getattr(module, f"libpyro_fortran_{mechname}")


BACKENDS: typing.Dict[str, Backend] = {
    CppBackend.get_name(): CppBackend,
    FortranBackend.get_name(): FortranBackend,
    PythonBackend.get_name(): PythonBackend
}


class Pyrometheus:
    """Represents an instance of a Pyrometheus implementation ("Backend") which
    can be interacted with as if it were the original Python implementation."""

    def __init__(self, mechname: str, backend: Backend, user_np):
        """Initializes an instance of Pyrometheus for a given backend
        and mechanism, loading the previously compiled generated code
        and extracting the interface from it.

        Parameters
        ----------
        mechname : str
            The name of the mechanism.
        backend : Backend
            The backend to use.
        user_np : module
            The numpy-like module to use for array operations within the
            generated code.

        """
        self.backend = backend
        self.pyro = self.backend.extract_interface(
            importlib.import_module(
                f"libpyro_{self.backend.get_name()}_{mechname}"
            ),
            mechname,
            user_np
        )

    def __getattr__(self, name: str):
        """Intercepts attribute accesses and method calls to this Pyrometheus
        interface to forward them to the underlying backend."""
        if getattr(self.backend, name, None) is not None:
            return getattr(self.backend, name)

        return getattr(self.pyro, name)


def pyro_init(mechname: str, user_np,
              request: pytest.FixtureRequest) -> (ct.Solution, Pyrometheus):
    """Initializes an instance of Pyrometheus for the given mechanism using the
    implementation of Pyrometheus specified by the backend option passed to
    pytest as a command line argument.

    Parameters
    ----------
    mechname : str
        The name of the mechanism to get thermochemistry code for.
    user_np : module
        A numpy-like module to use for array operations within the
        generated code.
    request : pytest.FixtureRequest
        The pytest request fixture.

    Returns
    -------
    ct.Solution
        A Cantera Solution object containing the mechanism.
    Pyrometheus
        A Pyrometheus instance for the mechanism and backend.
    """
    mech_dir = pathlib.Path(__file__).parent / "mechs"
    sol = ct.Solution(mech_dir / f"{mechname}.yaml", "gas")
    backend = BACKENDS[request.config.getoption("backend").lower()]

    return sol, Pyrometheus(mechname, backend, user_np)
