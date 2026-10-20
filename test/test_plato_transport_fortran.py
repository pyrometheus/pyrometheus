"""Parity of the generated Fortran transport subroutines against Plato.

The generated module is compiled with numpy.f2py and its transport
subroutines are compared with Plato's own transport evaluations at
heavy temperatures on Plato's collision-integral table nodes, where its
interpolated tables equal the fits bandit generates.

Requires PLATO_DB, an importable ``plato`` package, gfortran, meson and
ninja.
"""

import os

import numpy as np
import pytest

from fortran_build import (
    compile_fortran_module,
    stub_out_broken_vt_subroutine,
    toolchain_available,
    toolchain_reason,
)

try:
    import plato  # noqa: F401
    from pyrometheus.bandit.impl.plato import PlatoMechanism
    from pyrometheus.codegen.fortran_bandit import FortranBanditCodeGenerator
except ImportError:
    PlatoMechanism = None

pytestmark = [
    pytest.mark.skipif(
        PlatoMechanism is None or os.environ.get("PLATO_DB") is None,
        reason="PLATO_DB not set or plato package not importable",
    ),
    pytest.mark.skipif(not toolchain_available, reason=toolchain_reason),
]

# (Th, Tv), with Th on Plato's table nodes (200 K + k 50 K).
states = [(300.0, 300.0), (1000.0, 1000.0), (5050.0, 3000.0), (15000.0, 8000.0)]
mole_fraction_sets = {
    "air": {"N2": 0.79, "O2": 0.21},
    "dissociated": {"N": 0.3, "O": 0.2, "N2": 0.3, "NO": 0.05, "O2": 0.15},
    "atomic": {"N": 0.7, "O": 0.25, "N2": 0.02, "NO": 0.01, "O2": 0.02},
}


class CompiledCase:
    def __init__(self, mech, module):
        self.mech = mech
        self.module = module

    @property
    def oracle(self):
        return self.mech.namespace.thermochem

    def temperature(self, state):
        if self.mech.num_temp == 1:
            return state[0]
        return np.array(state[:self.mech.num_temp])

    def mole_fractions(self, name):
        mole_fractions = np.zeros(self.mech.num_species)
        for species_name, value in mole_fraction_sets[name].items():
            mole_fractions[self.mech.species_index(species_name)] = value
        return mole_fractions / mole_fractions.sum()

    def mass_fractions(self, name):
        weighted = self.mole_fractions(name) * self.mech.molecular_weights
        return weighted / weighted.sum()

    def number_density(self, temperature, mole_fractions, pressure=1e5):
        return self.oracle.number_density(
            pressure, np.atleast_1d(temperature), mole_fractions
        )


@pytest.fixture(scope="module", params=["empty", "TTv"])
def case(request, tmp_path_factory):
    mech = PlatoMechanism("air5", "air5", request.param, os.environ["PLATO_DB"])
    fortran_module_name = f"air5_{request.param.lower()}_transport"
    source = FortranBanditCodeGenerator.generate(fortran_module_name, mech)
    if mech.num_temp > 1:
        source = stub_out_broken_vt_subroutine(
            source, mech.num_species, mech.num_temp
        )
    module = compile_fortran_module(
        tmp_path_factory.mktemp(fortran_module_name), source,
        f"libpyro_{fortran_module_name}",
    )
    yield CompiledCase(mech, getattr(module, fortran_module_name))
    mech.finalize()


@pytest.mark.parametrize("state", states)
def test_fortran_binary_diffusivities_match_plato(case, state):
    temperature = case.temperature(state)
    mole_fractions = case.mole_fractions("dissociated")
    pressure = 1e5
    expected = case.oracle.binary_diffusion_coefficients(
        case.number_density(temperature, mole_fractions, pressure),
        np.atleast_1d(temperature), mole_fractions
    )
    actual = case.module.get_species_binary_mass_diffusivities(temperature)
    np.testing.assert_allclose(actual / pressure, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("state", states)
def test_fortran_species_viscosities_match_pure_gases_in_plato(case, state):
    temperature = case.temperature(state)
    viscosities = case.module.get_species_viscosities(temperature)
    for species_index in range(case.mech.num_species):
        pure = np.zeros(case.mech.num_species)
        pure[species_index] = 1.0
        expected = case.oracle.viscosity(
            case.number_density(temperature, pure),
            np.atleast_1d(temperature), pure
        )
        assert viscosities[species_index] == pytest.approx(
            expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("composition", mole_fraction_sets)
@pytest.mark.parametrize("state", states)
def test_fortran_mixture_viscosity_matches_plato(case, state, composition):
    temperature = case.temperature(state)
    mole_fractions = case.mole_fractions(composition)
    expected = case.oracle.viscosity(
        case.number_density(temperature, mole_fractions),
        np.atleast_1d(temperature), mole_fractions
    )
    actual = case.module.get_mixture_viscosity(
        temperature, case.mass_fractions(composition))
    assert actual == pytest.approx(expected, rel=1e-11, abs=0)


@pytest.mark.parametrize("composition", mole_fraction_sets)
@pytest.mark.parametrize("state", states)
def test_fortran_thermal_conductivities_match_plato(case, state, composition):
    temperature = case.temperature(state)
    mole_fractions = case.mole_fractions(composition)
    expected = case.oracle.thermal_conductivities(
        case.number_density(temperature, mole_fractions),
        np.atleast_1d(temperature), mole_fractions
    )
    actual = case.module.get_mixture_thermal_conductivities(
        temperature, case.mass_fractions(composition))
    np.testing.assert_allclose(
        np.atleast_1d(actual), expected, rtol=1e-11, atol=0)


@pytest.mark.parametrize("composition", mole_fraction_sets)
@pytest.mark.parametrize("state", states)
def test_fortran_effective_diffusivities_match_plato(case, state, composition):
    temperature = case.temperature(state)
    mole_fractions = case.mole_fractions(composition)
    pressure = 1e5
    expected = case.oracle.effective_diffusion_coefficients(
        case.number_density(temperature, mole_fractions, pressure),
        np.atleast_1d(temperature), mole_fractions
    )
    actual = case.module.get_species_effective_diffusivities(
        pressure, temperature, case.mass_fractions(composition))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)
