__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
Copyright (C) 2021 Esteban Cisneros
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
import numpy as np
import importlib
import pathlib
import typing

import pyrometheus

from collections.abc import Iterable
from abc import abstractmethod

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jnp = None
else:
    jax.config.update("jax_enable_x64", True)


class Backend:
    def __init__(self, pyro):
        self.pyro = pyro

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @abstractmethod
    def fix_arg(self, x):
        return x

    @abstractmethod
    def fix_result(self, result):
        return result

    @staticmethod
    @abstractmethod
    def generate_code(sol: ct.Solution, name: str):
        pass

    @staticmethod
    @abstractmethod
    def extract_interface(module, mechname: str, user_np):
        return module


class PythonBackend(Backend):
    def get_name():
        return "python"

    @staticmethod
    def _gen_jax_wrapper(base_cls):
        class JaxPyroWrapper(base_cls):
            def _pyro_make_array(self, res_list):
                """This works around (e.g.) numpy.exp not working with object
                arrays of :mod:`numpy` scalars. It defaults to making object arrays,
                however if an array consists of all scalars, it makes a "plain old"
                :class:`numpy.ndarray`.

                See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
                for more context.
                """

                from numbers import Number
                # Needed to play nicely with Jax, which frequently creates
                # arrays of shape () when handed numbers
                all_numbers = all(
                    isinstance(e, Number)
                    or (isinstance(e, self.usr_np.ndarray) and e.shape == ())
                    for e in res_list)

                if all_numbers:
                    return self.usr_np.array(res_list, dtype=self.usr_np.float64)

                result = self.usr_np.empty_like(res_list, dtype=object,
                                                shape=(len(res_list),))

                # 'result[:] = res_list' may look tempting, however:
                # https://github.com/numpy/numpy/issues/16564
                for idx in range(len(res_list)):
                    result[idx] = res_list[idx]

                return result

            def _pyro_norm(self, argument, normord):
                """This works around numpy.linalg norm not working with scalars.

                If the argument is a regular ole number, it uses :func:`numpy.abs`,
                otherwise it uses ``usr_np.linalg.norm``.
                """
                # Wrap norm for scalars
                from numbers import Number
                if isinstance(argument, Number):
                    return self.usr_np.abs(argument)
                # Needed to play nicely with Jax, which frequently creates
                # arrays of shape () when handed numbers
                if (isinstance(argument, self.usr_np.ndarray)
                        and argument.shape == ()):
                    return self.usr_np.abs(argument)
                return self.usr_np.linalg.norm(argument, normord)

        return JaxPyroWrapper

    @staticmethod
    def generate_code(sol: ct.Solution, name: str):
        return pyrometheus.codegen.python.gen_thermochem_code(sol=sol)

    @staticmethod
    def extract_interface(module, mechname: str, user_np):
        base_cls = module.Thermochemistry

        if user_np == jnp:
            base_cls = PythonBackend._gen_jax_wrapper(base_cls)

        return base_cls(user_np)


class CppBackend(Backend):
    def get_name():
        return "cpp"

    def fix_arg(self, x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return self.pyro.VectorDouble(x)
        return super().fix_arg(x)

    def fix_result(self, result):
        if isinstance(result, self.pyro.VectorDouble):
            return np.array(result)
        return super().fix_result(result)

    @staticmethod
    def generate_code(sol: ct.Solution, name: str):
        return pyrometheus.codegen.cpp.gen_thermochem_code(sol=sol, namespace=name)


class FortranBackend(Backend):
    def get_name():
        return "fortran90"

    @staticmethod
    def extract_interface(module, mechname: str, user_np):
        return getattr(module, f"libpyro_fortran90_{mechname}")

    @staticmethod
    def generate_code(sol: ct.Solution, name: str):
        return pyrometheus.codegen.fortran90.gen_thermochem_code(
            sol=sol, module_name=name
        )


BACKENDS: typing.Dict[str, Backend] = {
    CppBackend.get_name(): CppBackend,
    FortranBackend.get_name(): FortranBackend,
    PythonBackend.get_name(): PythonBackend
}


class Pyrometheus:
    def __init__(self, mechname: str, backend: Backend, user_np):
        self.backend = backend
        self.pyro = self.backend.extract_interface(
            importlib.import_module(f"libpyro_{self.backend.get_name()}_{mechname}"),
            mechname,
            user_np
        )

    def __getattr__(self, name: str):
        attrib = getattr(self.pyro, name)
        if not callable(attrib):
            return attrib

        def wrapper(*args, **kwargs):
            assert (len(kwargs) == 0)

            backend_instance = self.backend(self.pyro)

            return backend_instance.fix_result(
                attrib(*[backend_instance.fix_arg(x) for x in args])
            )

        return wrapper


def pyro_init(mechname: str, user_np, request) -> (ct.Solution, Pyrometheus):
    mech_dir = pathlib.Path(__file__).parent / "mechs"
    sol = ct.Solution(mech_dir / f"{mechname}.yaml", "gas")
    backend = BACKENDS[request.config.getoption("backend").lower()]

    return sol, Pyrometheus(mechname, backend, user_np)
