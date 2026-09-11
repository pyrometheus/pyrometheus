"""Unit tests for the bounded Newton temperature solves in
pyrometheus.flamelets.make_pyro (get_temperature, get_temperature_from_enthalpy).

Stage 2 of /u/csnrsgr2/.claude/plans/splendid-puzzling-sunset.md: these two
functions used to be unbounded jax.lax.while_loop Newton solves -- a
sibling package (pyroflow) has a saved KeyboardInterrupt traceback from
exactly this loop hanging. The adversarial tests below are the regression
guard against that failure mode recurring: every case must return within a
hard wall-clock timeout, with a finite, physically-bounded result and a
correctly-set "hit cap" diagnostic flag.

Run with: pytest test_flamelets_temperature.py --backend python
(--backend is required by this directory's conftest.py, but unused here.)
"""
import signal
from contextlib import contextmanager

import cantera as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyrometheus.codegen.python import PythonCodeGenerator as pyro
from pyrometheus.flamelets.make_pyro import (
    _TEMPERATURE_MAX_K,
    _TEMPERATURE_MIN_K,
    make_pyro_object,
)

MECH = "/u/csnrsgr2/packages/pyrometheus/test/mechs/sandiego.yaml"


class HardTimeout(Exception):
    pass


@contextmanager
def hard_timeout(seconds):
    """Backstop wall-clock timeout independent of the internal iteration
    cap -- a bug in the cap itself is exactly what these tests check for,
    so don't rely solely on it to bound runtime."""

    def _handler(signum, frame):
        raise HardTimeout(f"exceeded {seconds}s hard timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture(scope="module")
def pyro_gas():
    sol = ct.Solution(MECH)
    return make_pyro_object(pyro.get_thermochem_class(sol), jnp)


@pytest.fixture(scope="module")
def num_species(pyro_gas):
    return len(pyro_gas.molecular_weights)


def _uniform_y(num_species):
    return jnp.array(np.full(num_species, 1.0 / num_species))


# -- Nominal correctness: forward-then-invert round trip ------------------
# get_temperature_from_enthalpy solves h(T, Y) = enthalpy for fixed Y; the
# most direct correctness check is picking a T0, computing enthalpy
# forward via the same model, then confirming the inversion recovers T0.

@pytest.mark.parametrize("t0", [300.0, 500.0, 800.0, 1200.0, 2000.0])
def test_nominal_round_trip(pyro_gas, num_species, t0):
    y = _uniform_y(num_species)
    h = pyro_gas.get_mixture_enthalpy_mass(jnp.array(t0), y)
    t_init = jnp.array(t0 * 0.8)  # imperfect but reasonable initial guess

    recovered, hit_cap = jax.jit(
        lambda h, y, t: pyro_gas.get_temperature_from_enthalpy(
            h, y, t, _with_diagnostics=True
        )
    )(h, y, t_init)

    assert not bool(hit_cap)
    assert abs(float(recovered) - t0) / t0 < 1e-8


@pytest.mark.parametrize("t0", [300.0, 500.0, 800.0, 1200.0, 2000.0])
def test_nominal_round_trip_energy(pyro_gas, num_species, t0):
    y = _uniform_y(num_species)
    e = pyro_gas.get_mixture_internal_energy_mass(jnp.array(t0), y)
    t_init = jnp.array(t0 * 0.8)

    recovered, hit_cap = jax.jit(
        lambda e, y, t: pyro_gas.get_temperature(
            e, t, y, _with_diagnostics=True
        )
    )(e, y, t_init)

    assert not bool(hit_cap)
    assert abs(float(recovered) - t0) / t0 < 1e-8


# -- Adversarial regression: must never hang, must fail safe --------------
# Each case is a concrete input shape that previously could drive the
# unbounded loop into non-convergence (extreme initial guess, an
# enthalpy/energy target unreachable by any physical composition at any
# temperature, or degenerate mass fractions). The mandatory invariant for
# *every* case is termination with a finite, in-band result -- that's the
# actual hang-safety property. Some of these (bad-but-reachable initial
# guesses) may legitimately still Newton-converge, since h(T)/e(T) is smooth
# and monotonic for this mechanism in a wide range; only cases with a
# genuinely unreachable target are additionally required to trip hit_cap,
# as a check that the cap mechanism itself actually fires when it should.

def _adversarial_cases(pyro_gas, num_species):
    y = _uniform_y(num_species)
    h_nominal = float(pyro_gas.get_mixture_enthalpy_mass(jnp.array(800.0), y))
    single_species_y = jnp.array(np.eye(num_species)[0])
    # (name, h, y, t_init, must_hit_cap)
    return [
        ("extreme_init_low", h_nominal, y, 1.0, False),
        ("extreme_init_high", h_nominal, y, 1.0e5, False),
        ("unreachable_h_huge", 1.0e12, y, 800.0, True),
        ("unreachable_h_very_negative", -1.0e12, y, 800.0, True),
        ("degenerate_single_species", h_nominal, single_species_y, 800.0, False),
        ("extreme_init_and_unreachable_h", 1.0e10, y, 1.0e6, True),
    ]


@pytest.fixture(scope="module")
def adversarial_cases(pyro_gas, num_species):
    return _adversarial_cases(pyro_gas, num_species)


@pytest.mark.parametrize("case_index", range(6))
def test_adversarial_never_hangs(pyro_gas, adversarial_cases, case_index):
    name, h, y, t_init, must_hit_cap = adversarial_cases[case_index]
    fn = jax.jit(
        lambda h, y, t: pyro_gas.get_temperature_from_enthalpy(
            h, y, t, _with_diagnostics=True
        )
    )
    with hard_timeout(30):
        temperature, hit_cap = fn(jnp.asarray(h), y, jnp.asarray(t_init))
        temperature = float(temperature)
        hit_cap = bool(hit_cap)
    assert np.isfinite(temperature), f"{name}: non-finite result"
    assert _TEMPERATURE_MIN_K <= temperature <= _TEMPERATURE_MAX_K, (
        f"{name}: {temperature} outside physical band "
        f"[{_TEMPERATURE_MIN_K}, {_TEMPERATURE_MAX_K}]"
    )
    if must_hit_cap:
        assert hit_cap, f"{name}: expected the cap/fail-safe to fire"


def test_adversarial_never_hangs_energy_form(pyro_gas, num_species):
    y = _uniform_y(num_species)
    fn = jax.jit(
        lambda e, y, t: pyro_gas.get_temperature(
            e, t, y, _with_diagnostics=True
        )
    )
    for name, e, t_init, must_hit_cap in [
        ("extreme_init_low", 1.0e5, 1.0, False),
        ("unreachable_e_huge", 1.0e12, 800.0, True),
        ("unreachable_e_very_negative", -1.0e12, 800.0, True),
    ]:
        with hard_timeout(30):
            temperature, hit_cap = fn(jnp.asarray(e), y, jnp.asarray(t_init))
            temperature = float(temperature)
            hit_cap = bool(hit_cap)
        assert np.isfinite(temperature), f"{name}: non-finite result"
        assert _TEMPERATURE_MIN_K <= temperature <= _TEMPERATURE_MAX_K
        if must_hit_cap:
            assert hit_cap, f"{name}: expected hit_cap=True"


def test_adversarial_under_disable_jit(pyro_gas, num_species):
    """Same adversarial cases without jit, to separate tracing bugs from
    algorithm bugs if the jitted variant above ever fails."""
    cases = _adversarial_cases(pyro_gas, num_species)
    with jax.disable_jit():
        for name, h, y, t_init, must_hit_cap in cases:
            with hard_timeout(30):
                temperature, hit_cap = pyro_gas.get_temperature_from_enthalpy(
                    jnp.asarray(h), y, jnp.asarray(t_init),
                    _with_diagnostics=True,
                )
            temperature = float(temperature)
            assert np.isfinite(temperature), f"{name}: non-finite"
            assert _TEMPERATURE_MIN_K <= temperature <= _TEMPERATURE_MAX_K, (
                f"{name}: {temperature} outside physical band"
            )
            if must_hit_cap:
                assert bool(hit_cap), f"{name}: expected hit_cap=True"


def test_default_signature_unchanged(pyro_gas, num_species):
    """The default call (no _with_diagnostics) must still return a plain
    array, matching every existing call site in equations.py/solver.py and
    in flamelet-adjoints/scripts/fit_flame.py."""
    y = _uniform_y(num_species)
    h = pyro_gas.get_mixture_enthalpy_mass(jnp.array(800.0), y)
    result = pyro_gas.get_temperature_from_enthalpy(h, y, jnp.array(750.0))
    assert isinstance(result, jax.Array)

    e = pyro_gas.get_mixture_internal_energy_mass(jnp.array(800.0), y)
    result = pyro_gas.get_temperature(e, jnp.array(750.0), y)
    assert isinstance(result, jax.Array)
