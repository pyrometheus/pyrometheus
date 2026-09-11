"""Validate FlameletSolver._newton_loop_lax_steady (Stage 4 of
/u/csnrsgr2/.claude/plans/splendid-puzzling-sunset.md) against
_newton_loop_py, the trusted Python-loop reference, on the same inputs.

vmap requires the per-point steady Newton solve to trace under one jit,
which means the iteration count can no longer be a Python `for` loop --
_newton_loop_lax_steady is the bounded lax.while_loop replacement, with
explicit divergence detection matching _newton_loop_py's semantics
(converge, diverge, or exhaust maxiter). This must match the Python
version's final state, iteration count, and converged/diverged
classification before any batching work is built on top of it.

Run with: pytest test_flamelets_newton_loop.py --backend python
"""
import cantera as ct
import jax.numpy as jnp
import numpy as np
import pytest
from pyrometheus.codegen.python import PythonCodeGenerator as pyro
from pyrometheus.flamelets.domain import Domain, DomainConfig
from pyrometheus.flamelets.make_pyro import make_pyro_object
from pyrometheus.flamelets.solver import FlameletSolver
from pyrometheus.flamelets.state import FlameletState, _state_to_array
from pyrometheus.flamelets.utils import bell_profile, stoichiometric_mixture_fraction

MECH = "/u/csnrsgr2/packages/pyrometheus/test/mechs/sandiego.yaml"
FUEL = "H2"
TEMP_OX, TEMP_FU, PRESSURE = 500.0, 300.0, ct.one_atm
NUM_X = 101
NEWTON_MAXITER = 30
NEWTON_TOL = 1e-8


@pytest.fixture(scope="module")
def rig():
    """Minimal FlameletSolver setup, self-contained (no dependency on the
    sibling flamelet-adjoints repo's synthetic_data.py)."""
    sol = ct.Solution(MECH)
    pyro_gas = make_pyro_object(pyro.get_thermochem_class(sol), jnp)

    sol.TPX = TEMP_OX, PRESSURE, "O2:0.21, N2:0.79"
    y_ox = jnp.array(sol.Y)
    h_ox = float(sol.enthalpy_mass)

    sol.TPX = TEMP_FU, PRESSURE, f"{FUEL}:0.5, N2:0.5"
    y_fu = jnp.array(sol.Y)
    h_fu = float(sol.enthalpy_mass)

    z_st = float(stoichiometric_mixture_fraction(sol, y_ox, y_fu))

    domain = Domain(DomainConfig(num_x=NUM_X, x_l=0, x_r=1))
    solver = FlameletSolver(domain, pyro_gas, (y_ox, y_fu))
    z = jnp.array(domain.x)

    # Linear-mixing initial guess (deliberately crude -- not equilibrium --
    # so the outer Newton loop actually has iterating to do).
    z_np = np.asarray(z)
    linear_h = h_ox + (h_fu - h_ox) * z_np
    linear_y = (
        np.asarray(y_ox)[:, None]
        + (np.asarray(y_fu) - np.asarray(y_ox))[:, None] * z_np[None, :]
    )
    state_guess = FlameletState(
        enthalpy=jnp.array(linear_h), mass_fractions=jnp.array(linear_y)
    )
    temp_guess = jnp.array(TEMP_OX + (TEMP_FU - TEMP_OX) * z_np)

    def diss_profile(chi_st):
        boundary_val = 2 * domain.jac[0] ** 2
        profile = chi_st * bell_profile(z) / bell_profile(z_st)
        profile = profile.at[0].set(boundary_val)
        profile = profile.at[-1].set(boundary_val)
        return profile

    def newton_fn(state, diss_rate, viscous_diss, temp_guess, pressure, h_ox, h_fu):
        return _state_to_array(
            solver.gov_eqns.rhs(
                state, diss_rate, viscous_diss, temp_guess, pressure, h_ox, h_fu
            )
        )

    def newton_jac(state, diss_rate, viscous_diss, temp_guess, pressure, h_ox, h_fu):
        return solver.gov_eqns.jac(
            state, diss_rate, viscous_diss, temp_guess, pressure
        )

    return dict(
        solver=solver, state_guess=state_guess, temp_guess=temp_guess,
        pressure=PRESSURE, h_ox=h_ox, h_fu=h_fu, diss_profile=diss_profile,
        newton_fn=newton_fn, newton_jac=newton_jac,
    )


def _run_both(rig, chi, state_guess=None, temp_guess=None):
    solver = rig["solver"]
    diss_rate = rig["diss_profile"](chi)
    visc = jnp.zeros_like(diss_rate)
    sg = state_guess if state_guess is not None else rig["state_guess"]
    tg = temp_guess if temp_guess is not None else rig["temp_guess"]
    args = (diss_rate, visc, tg, rig["pressure"], rig["h_ox"], rig["h_fu"])

    state_py, it_py, delta_py, success_py = solver._newton_loop_py(
        False, sg, NEWTON_MAXITER, NEWTON_TOL, *args
    )
    state_lax, it_lax, delta_lax, success_lax = solver._newton_loop_lax_steady(
        rig["newton_fn"], rig["newton_jac"], sg, NEWTON_MAXITER, NEWTON_TOL, *args
    )
    return (
        (state_py, int(it_py), float(delta_py), bool(success_py)),
        (state_lax, int(it_lax), float(delta_lax), bool(success_lax)),
    )


@pytest.mark.parametrize("chi", [100.0, 500.0, 1000.0, 2000.0, 4000.0])
def test_matches_python_loop_across_chi(rig, chi):
    """Nominal-to-hard chi sweep (matches synthetic_data.py's calibrated
    range, up to the near-extinction case) -- both loops should reach the
    same answer via the same path regardless of difficulty."""
    (s_py, it_py, d_py, ok_py), (s_lax, it_lax, d_lax, ok_lax) = _run_both(rig, chi)

    assert ok_py == ok_lax, f"chi={chi}: success mismatch py={ok_py} lax={ok_lax}"
    assert it_py == it_lax, f"chi={chi}: iteration count mismatch {it_py} vs {it_lax}"
    np.testing.assert_allclose(
        np.asarray(s_py.enthalpy), np.asarray(s_lax.enthalpy),
        rtol=1e-10, atol=1e-8,
        err_msg=f"chi={chi}: enthalpy field mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(s_py.mass_fractions), np.asarray(s_lax.mass_fractions),
        rtol=1e-10, atol=1e-10,
        err_msg=f"chi={chi}: mass fraction field mismatch",
    )


def test_matches_python_loop_deep_past_extinction(rig):
    """chi=20000 is deep past this mechanism's extinction turning point
    (see plan Progress log: extinguishes between chi=4000 and chi=5000).
    The forward solve should settle to the frozen-mixing state either way
    -- confirm both loops agree on that outcome, including whether they
    classify it as converged or not."""
    (s_py, it_py, d_py, ok_py), (s_lax, it_lax, d_lax, ok_lax) = _run_both(
        rig, 20000.0
    )
    assert ok_py == ok_lax
    assert it_py == it_lax
    np.testing.assert_allclose(
        np.asarray(s_py.enthalpy), np.asarray(s_lax.enthalpy),
        rtol=1e-9, atol=1e-6,
    )


def test_matches_python_loop_bad_initial_guess(rig):
    """A deliberately bad state_guess (uniform, unphysical enthalpy/mass
    fractions far from any solution) -- targets the OUTER Newton loop's
    divergence detection specifically, the highest-risk part of this
    rewrite per the plan's Stage 4 contingency."""
    num_species = rig["solver"].gov_eqns.pyro_gas.num_species
    bad_state = FlameletState(
        enthalpy=jnp.full(NUM_X, 5.0e7),
        mass_fractions=jnp.full((num_species, NUM_X), 1.0 / num_species),
    )
    bad_temp_guess = jnp.full(NUM_X, 50000.0)
    (s_py, it_py, d_py, ok_py), (s_lax, it_lax, d_lax, ok_lax) = _run_both(
        rig, 1000.0, state_guess=bad_state, temp_guess=bad_temp_guess
    )
    assert ok_py == ok_lax, (
        f"bad initial guess: success mismatch py={ok_py} lax={ok_lax}, "
        f"it_py={it_py} it_lax={it_lax}"
    )
    assert it_py == it_lax


def test_lax_variant_never_exceeds_maxiter(rig):
    """Bounded-loop safety property: the returned iteration count must
    never exceed NEWTON_MAXITER - 1 (Python-loop convention), regardless
    of input -- this is the actual hang-safety guarantee Stage 4 exists
    to provide."""
    for chi in [100.0, 4000.0, 20000.0]:
        _, (_, it_lax, _, _) = _run_both(rig, chi)
        assert 0 <= it_lax <= NEWTON_MAXITER - 1
