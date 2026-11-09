"""Helpers shared by the parity suites that compare generated code with
the library a :class:`~pyrometheus.bandit.general_thermochem.BaseMechanism`
was read from.
"""

import numpy as np


def make_generated_gas(mech, pyro_np=np):
    """:returns: An instance of the Python class generated for *mech*,
    able to evaluate batched states.
    """
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
    base_class = PythonBanditCodeGenerator.get_thermochem_class(mech)

    class StackedGas(base_class):
        # The default object-array assembly cannot hold batched states.
        def _pyro_make_array(self, res_list):
            return self.pyro_np.stack([
                self.pyro_np.asarray(entry, dtype=self.pyro_np.float64)
                for entry in res_list
            ])

    return StackedGas(pyro_np)


def temperature_subscripts(expressions):
    """:returns: Every index that *expressions* subscript ``temperature``
    with.
    """
    from pymbolic.mapper import WalkMapper
    import pymbolic.primitives as prim

    class Collector(WalkMapper):
        def __init__(self):
            self.indices = set()

        def visit(self, expr, *args, **kwargs):
            if (isinstance(expr, prim.Subscript)
                    and expr.aggregate == prim.Variable("temperature")):
                self.indices.add(expr.index)
            return True

    collector = Collector()
    for expr in expressions:
        collector(expr)
    return collector.indices


def mechanism_expressions(mech):
    """:returns: The rate, thermo and mass-action expressions of *mech*,
    which between them hold every temperature it reads.
    """
    return [rate.expr for rate in mech.rate_coeffs] + [
        poly.expr
        for thermo in mech.species_nasa_thermo_polynomials
        for poly in (thermo.cp_poly, thermo.enthalpy_poly, thermo.gibbs_poly)
    ] + list(mech.mass_action_rates)


# {{{ compiled languages behind the generated Python's interface

languages = ["python", "cpp", "fortran"]

# Fortran spells one function differently from Python and C++.
_fortran_names = {
    "get_mixture_internal_energy_mass": "get_mixture_energy_mass",
}


class NativeGas:
    """Calls the functions of a compiled module (C++ through pybind11 or
    Fortran through f2py) with the generated Python's names and
    signatures, so that one test drives every language.

    Constants come from the generated functions themselves, since f2py
    cannot reach Fortran ``parameter`` arrays.
    """

    def __init__(self, module, num_species, renames=None):
        self._module = module
        self._renames = renames or {}
        self._num_species = num_species

    def _function(self, name):
        return getattr(self._module, self._renames.get(name, name))

    def __getattr__(self, name):
        if not name.startswith("get_"):
            raise AttributeError(name)
        function = self._function(name)

        def call(*args):
            result = function(*[
                np.ascontiguousarray(arg, dtype=np.float64)
                if isinstance(arg, np.ndarray) else arg
                for arg in args
            ])
            return np.asarray(result) if isinstance(result, (list, tuple)) \
                else result
        return call

    @property
    def molecular_weights(self):
        # One kg/m^3 of species k holds 1/W_k kmol/m^3.
        identity = np.eye(self._num_species)
        return np.array([
            1.0 / np.asarray(self.get_concentrations(1.0, row))[k]
            for k, row in enumerate(identity)
        ])

    @property
    def gas_constant(self):
        # p = rho R T / W for a pure species at unit density and
        # temperature.
        mass_fractions = np.eye(self._num_species)[0]
        return float(
            self.get_pressure(1.0, 1.0, mass_fractions)
            * self.molecular_weights[0]
        )

    @property
    def species_names(self):
        return [
            self.get_species_name(species_index)
            for species_index in range(self._num_species)
        ]


class FortranGas(NativeGas):
    def get_species_name(self, species_index):
        name = self._function("get_species_name")(species_index)
        if isinstance(name, bytes):
            name = name.decode()
        return str(name).strip()

    def get_species_index(self, species_name):
        return int(self._function("get_species_index")(species_name))


def make_language_gas(mech, language, tmp_path, module_name):
    """:returns: An object with the generated Python's interface that
    runs the code generated for *mech* in *language*.
    """
    if language == "python":
        return make_generated_gas(mech)
    if language == "cpp":
        from cpp_build import compile_cpp_module
        from pyrometheus.codegen.cpp_bandit import CppBanditCodeGenerator
        header = CppBanditCodeGenerator.generate(module_name, mech)
        module = compile_cpp_module(tmp_path, header, module_name,
                                    f"{module_name}_cpp")
        return NativeGas(module, mech.num_species)
    if language == "fortran":
        from fortran_build import compile_fortran_module
        from pyrometheus.codegen.fortran_bandit import (
            FortranBanditCodeGenerator,
        )
        # The extension module and the Fortran module it wraps share
        # one name.
        fortran_name = f"{module_name}_f"
        source = FortranBanditCodeGenerator.generate(fortran_name, mech)
        module = compile_fortran_module(tmp_path, source, fortran_name)
        return FortranGas(getattr(module, fortran_name),
                          mech.num_species, _fortran_names)
    raise ValueError(f"unknown language '{language}'")


def language_available(language):
    """:returns: A reason to skip *language*, or *None*."""
    if language == "cpp":
        from cpp_build import toolchain_available, toolchain_reason
        return None if toolchain_available else toolchain_reason
    if language == "fortran":
        from fortran_build import toolchain_available, toolchain_reason
        return None if toolchain_available else toolchain_reason
    return None

# }}}
