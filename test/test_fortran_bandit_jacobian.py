"""Parity test for the Fortran-codegen analytic species-production-rate
Jacobian (get_net_production_rates_jacobian, opt-in via
CodeGenerationOptions.compute_jacobian). Unlike the Python backend --
which differentiates the generated code with AD instead -- Fortran has
no good AD story, so this is where the analytic-Jacobian capability
(BaseMechanism.make_species_production_rate_jacobian /
chem_expr/jacobian.py) is actually meant to be used.

This generates real Fortran source, compiles it into an importable
extension module with numpy.f2py (same approach as
examples/fortran/species_source.py), and checks
get_net_production_rates_jacobian against a finite difference of
get_net_production_rates through the compiled module -- an actual
compiled-and-executed parity check, not just a symbolic one.

Requires PLATO_DB / PLATO_LIB, gfortran, and f2py's meson/ninja build
dependencies (``pip install meson ninja``). Skipped automatically when
any of these is unavailable.
"""

import os
import shutil

import numpy as np
import pytest

from fortran_build import compile_fortran_module

plato_available = (
    os.environ.get("PLATO_DB") is not None
    and os.environ.get("PLATO_LIB") is not None
)

try:
    from pyrometheus.bandit.impl.plato import PlatoMechanism
    from pyrometheus.codegen import CodeGenerationOptions
    from pyrometheus.codegen.fortran_bandit import (
        FortranBanditCodeGenerator as fortran,
    )
except ImportError:
    plato_available = False

fortran_toolchain_available = all(
    shutil.which(tool) is not None
    for tool in ("gfortran", "meson", "ninja")
)

requires_fortran_toolchain = pytest.mark.skipif(
    not (plato_available and fortran_toolchain_available),
    reason="PLATO_DB / PLATO_LIB / plato package / gfortran / meson / ninja "
           "not all available",
)

DB = os.environ.get("PLATO_DB", "")
MIXTURE = "air5"
REACTION = "air5"
TRANSFER = "TTv"
MODULE_NAME = "libpyro_fortran_jacobian_test"

# get_vt_energy_transfer_source is broken in the generated Fortran today
# (references an undeclared local `pressure`) -- a pre-existing,
# separately-flagged issue unrelated to the Jacobian capability this
# test targets (the Python template gained a pressure computation
# earlier; the Fortran template's copy of that subroutine was never
# updated to match). Stub its body out so the rest of the module --
# including get_net_production_rates_jacobian -- can actually be built
# and run.
_VT_SOURCE_MARKER = "subroutine get_vt_energy_transfer_source(&"


def _stub_out_broken_vt_subroutine(source, num_species, num_temp):
    lines = source.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if _VT_SOURCE_MARKER in ln)
    end = next(
        i for i, ln in enumerate(lines)
        if i > start and "end subroutine get_vt_energy_transfer_source" in ln
    )
    stub = [
        "    subroutine get_vt_energy_transfer_source(&\n",
        "        & density, mass_fractions, temperature, omega_vt)\n",
        "        real(dp), intent(in) :: density\n",
        f"        real(dp), intent(in), dimension({num_species}) :: "
        "mass_fractions\n",
        f"        real(dp), intent(in), dimension({num_temp}) :: temperature\n",
        "        real(dp), intent(out) :: omega_vt\n",
        "        omega_vt = 0.d0\n",
        "    end subroutine get_vt_energy_transfer_source\n",
    ]
    return "".join(lines[:start] + stub + lines[end + 1:])


@pytest.fixture(scope="module")
def jac_fort(tmp_path_factory):
    if not (plato_available and fortran_toolchain_available):
        pytest.skip("PLATO_DB / PLATO_LIB / plato package / gfortran / "
                    "meson / ninja not all available")

    mech = PlatoMechanism(MIXTURE, REACTION, TRANSFER, DB)
    try:
        opts = CodeGenerationOptions(compute_jacobian=True)
        fortran_module_name = "air5_ttv_jacobian_test"
        source = fortran.generate(fortran_module_name, mech, opts)
        source = _stub_out_broken_vt_subroutine(
            source, mech.num_species, mech.num_temp
        )
        tmp_path = tmp_path_factory.mktemp("fortran_jacobian")
        module = compile_fortran_module(tmp_path, source, MODULE_NAME)
    finally:
        mech.finalize()

    return getattr(module, fortran_module_name)


@requires_fortran_toolchain
def test_fortran_net_production_rates_jacobian_matches_finite_difference(
        jac_fort):
    density = 0.05
    temperature = np.array([8000.0, 4000.0])
    mass_fractions = np.array([0.05, 0.05, 0.4, 0.1, 0.4])

    jacobian = jac_fort.get_net_production_rates_jacobian(
        density, temperature, mass_fractions
    )
    assert jacobian.shape == (5, 1 + 2 + 5)

    def w_dot(density, temperature, mass_fractions):
        return jac_fort.get_net_production_rates(
            density, temperature, mass_fractions
        )

    rtol, atol = 2e-3, 1e-6
    relative_step = 1.0e-6

    # d/d(density)
    step = relative_step * density
    fd_density = (
        w_dot(density + step, temperature, mass_fractions)
        - w_dot(density - step, temperature, mass_fractions)
    ) / (2 * step)
    np.testing.assert_allclose(jacobian[:, 0], fd_density, rtol=rtol, atol=atol)

    # d/d(temperature[t])
    for t in range(2):
        temp_plus = temperature.copy()
        temp_minus = temperature.copy()
        step = relative_step * temperature[t]
        temp_plus[t] += step
        temp_minus[t] -= step
        fd_temp = (
            w_dot(density, temp_plus, mass_fractions)
            - w_dot(density, temp_minus, mass_fractions)
        ) / (2 * step)
        np.testing.assert_allclose(
            jacobian[:, 1 + t], fd_temp, rtol=rtol, atol=atol
        )

    # d/d(mass_fractions[k])
    step = relative_step
    for k in range(5):
        y_plus = mass_fractions.copy()
        y_minus = mass_fractions.copy()
        y_plus[k] += step
        y_minus[k] -= step
        fd_y = (
            w_dot(density, temperature, y_plus)
            - w_dot(density, temperature, y_minus)
        ) / (2 * step)
        np.testing.assert_allclose(
            jacobian[:, 3 + k], fd_y, rtol=rtol, atol=atol
        )
