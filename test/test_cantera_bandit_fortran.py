"""Parity tests for the Fortran that the Bandit Cantera backend
generates. The source is compiled with f2py and executed, so these check
the rendered template rather than the symbolic expressions behind it.

Skipped automatically when gfortran, meson or ninja is unavailable.
"""

import pathlib

import cantera as ct
import numpy as np
import pytest

from fortran_build import (
    compile_fortran_module, toolchain_available, toolchain_reason
)
from pyrometheus.bandit.impl.cantera import CanteraMechanism
from pyrometheus.codegen.fortran_bandit import FortranBanditCodeGenerator


pytestmark = pytest.mark.skipif(
    not toolchain_available, reason=toolchain_reason
)

mech_dir = pathlib.Path(__file__).parent / "mechs"

# uiuc exercises fractional orders, sandiego three-body and Troe
# falloff, uconn32 adds Lindemann.
fortran_mechanisms = ["uiuc", "sandiego", "uconn32"]


@pytest.fixture(scope="module")
def fortran_gas(request, tmp_path_factory):
    mechname = request.param
    path = str(mech_dir / f"{mechname}.yaml")
    mech = CanteraMechanism(path)
    module_name = f"libpyro_fortran_bandit_{mechname}"
    source = FortranBanditCodeGenerator.generate(module_name, mech)
    module = compile_fortran_module(
        tmp_path_factory.mktemp(mechname), source, module_name
    )
    return ct.Solution(path), getattr(module, module_name)


def state(sol, pressure_atm=5.0):
    sol.TPY = (
        1400.0, pressure_atm * ct.one_atm,
        np.full(sol.n_species, 1 / sol.n_species)
    )
    return sol.density_mass, sol.T, sol.Y


@pytest.mark.parametrize(
    "fortran_gas", fortran_mechanisms, indirect=True
)
def test_fortran_production_rates_match_cantera(fortran_gas):
    sol, gas = fortran_gas
    density, temperature, mass_fractions = state(sol)
    actual = gas.get_net_production_rates(
        density, temperature, mass_fractions
    )
    np.testing.assert_allclose(
        actual, sol.net_production_rates, rtol=1e-9,
        atol=1e-9 * np.abs(sol.net_production_rates).max()
    )


@pytest.mark.parametrize(
    "fortran_gas", fortran_mechanisms, indirect=True
)
def test_fortran_state_conversions_match_cantera(fortran_gas):
    sol, gas = fortran_gas
    density, temperature, mass_fractions = state(sol)
    assert gas.get_density(
        sol.P, temperature, mass_fractions
    ) == pytest.approx(density, rel=1e-12)
    assert gas.get_pressure(
        density, temperature, mass_fractions
    ) == pytest.approx(sol.P, rel=1e-12)


@pytest.mark.parametrize(
    "fortran_gas", fortran_mechanisms, indirect=True
)
@pytest.mark.parametrize("do_energy", [False, True])
def test_fortran_temperature_inversion(fortran_gas, do_energy):
    sol, gas = fortran_gas
    _, temperature, mass_fractions = state(sol)
    target = sol.int_energy_mass if do_energy else sol.enthalpy_mass
    assert gas.get_temperature(
        target, 1000.0, mass_fractions, do_energy
    ) == pytest.approx(temperature, rel=1e-8)


@pytest.mark.parametrize(
    "fortran_gas", fortran_mechanisms, indirect=True
)
def test_fortran_equilibrium_constants_match_cantera(fortran_gas):
    sol, gas = fortran_gas
    _, temperature, _ = state(sol)
    reversible = [
        reaction_index for reaction_index in range(sol.n_reactions)
        if sol.reaction(reaction_index).reversible
    ]
    actual = gas.get_equilibrium_constants(temperature)[reversible]
    np.testing.assert_allclose(
        actual, -np.log(sol.equilibrium_constants[reversible]), rtol=1e-11
    )
