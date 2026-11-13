"""Validate CompressibleEOS.ensure_consistency_traceable (Stage 5, second
half) against ensure_consistency, the trusted Python-loop reference.

ensure_consistency's outer Gauss-Newton loop is a Python `for` loop with
an early `break` on convergence, calling evaluate_flamelet -> solve()
(Newton-then-BDF-fallback) each iteration -- none of that is traceable.
ensure_consistency_traceable replaces it with a bounded jax.lax.scan
(exactly config["eos"]["maxiter"] steps, converged-lanes frozen via
masking) over evaluate_flamelet_traceable -> _newton_loop_lax_steady
(Stage 4's bounded Newton loop, no BDF fallback). This must match the
Python version's final state/params/convergence classification before
any vmap batching is built on top of it.

Run with: pytest test_flamelets_eos_consistency.py --backend python
"""
from pathlib import Path
import cantera as ct
import jax.numpy as jnp
import numpy as np
import pytest
from pyrometheus.codegen.python import PythonCodeGenerator
from pyrometheus.flamelets.domain import Domain, DomainConfig
from pyrometheus.flamelets.make_pyro import make_pyro_object
from pyrometheus.flamelets.solver import FlameletSolver
from pyrometheus.flamelets.state import FlameletState
from pyrometheus.flamelets.thermodynamic_consistency import CompressibleEOS
from pyrometheus.flamelets.utils import bell_profile, stoichiometric_mixture_fraction

MECH = str(Path(__file__).parent / "mechs" / "sandiego.yaml")
FUEL = "H2"
TEMP_OX, TEMP_FU, PRESSURE = 500.0, 300.0, ct.one_atm
NUM_X = 101


def _config():
    return {
        "verbosity": False,
        "max_attempts": 10,
        "bdf": {
            "maxsteps": 10, "time_step": 1e-5,
            "newton": {"maxiter": 20, "tol": 1e-9},
        },
        "newton": {"maxiter": 30, "tol": 1e-8},
        "eos": {
            "maxiter": 10, "tol": 1e-8, "update_size": 0.4,
            "update_method": "gauss_newton",
        },
    }


@pytest.fixture(scope="module")
def rig():
    sol = ct.Solution(MECH)
    pyro_gas = make_pyro_object(PythonCodeGenerator.get_thermochem_class(sol), jnp)

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
    z_np = np.asarray(z)

    def diss_profile(chi_st):
        bval = 2 * domain.jac[0] ** 2
        p = chi_st * bell_profile(z) / bell_profile(z_st)
        return p.at[0].set(bval).at[-1].set(bval)

    # Baseline warm start: staged solve at chi=1000, the fixed physical
    # boundary conditions -- mirrors synthetic_data.py's FlameletModel /
    # fit_flame.py's get_baseline_guess pattern.
    config = _config()
    linear_h = h_ox + (h_fu - h_ox) * z_np
    linear_y = (
        np.asarray(y_ox)[:, None]
        + (np.asarray(y_fu) - np.asarray(y_ox))[:, None] * z_np[None, :]
    )
    equil_state = FlameletState(
        enthalpy=jnp.array(linear_h), mass_fractions=jnp.array(linear_y)
    )
    equil_temp = jnp.array(TEMP_OX + (TEMP_FU - TEMP_OX) * z_np)
    zero_visc = jnp.zeros(NUM_X)

    state_base, temp_base = solver.solve(
        False, config["newton"]["maxiter"], config["newton"]["tol"],
        config["bdf"]["newton"]["maxiter"], config["bdf"]["newton"]["tol"],
        config["bdf"]["time_step"], config["bdf"]["maxsteps"],
        False, config["max_attempts"],
        diss_profile(1000.0), zero_visc, equil_temp, PRESSURE, h_ox, h_fu,
        equil_state,
    )

    compressible_eos = CompressibleEOS(config, solver)
    return dict(
        solver=solver, compressible_eos=compressible_eos,
        state_base=state_base, temp_base=temp_base,
        pressure=PRESSURE, h_ox=h_ox, h_fu=h_fu,
        diss_profile=diss_profile, z=z, z_st=z_st,
    )


def _target_density_energy(rig, chi, z_index):
    """Forward-solve at the given chi (fixed baseline boundary
    conditions, per the plan's Stage 1 correction) and extract
    (density, energy) at one grid point -- the (density_sim, energy_sim)
    ensure_consistency is asked to match."""
    solver = rig["solver"]
    diss = rig["diss_profile"](chi)
    zero_visc = jnp.zeros(NUM_X)
    state, temp = solver.solve(
        False, 30, 1e-8, 20, 1e-9, 1e-5, 10, False, 10,
        diss, zero_visc, rig["temp_base"], rig["pressure"],
        rig["h_ox"], rig["h_fu"], rig["state_base"],
    )
    pyro_gas = solver.gov_eqns.pyro_gas
    density = pyro_gas.get_density(rig["pressure"], temp, state.mass_fractions)
    energy = pyro_gas.get_mixture_internal_energy_mass(temp, state.mass_fractions)
    return float(density[z_index]), float(energy[z_index]), diss


@pytest.mark.parametrize("chi,z_index", [
    (100.0, 30), (100.0, 70), (2000.0, 50), (4000.0, 26), (4000.0, 74),
])
def test_matches_python_ensure_consistency(rig, chi, z_index):
    ceos = rig["compressible_eos"]
    density_sim, energy_sim, diss = _target_density_energy(rig, chi, z_index)
    mix_frac_pdf = jnp.zeros(NUM_X).at[z_index].set(1)
    zero_visc = jnp.zeros(NUM_X)
    thermo_params = jnp.array([rig["pressure"], rig["h_ox"], rig["h_fu"]])

    (state_py, temp_py, params_py, it_py, delta_py, res_py, hist_py,
     _) = ceos.ensure_consistency(
        density_sim, energy_sim, thermo_params, mix_frac_pdf,
        diss, zero_visc, rig["temp_base"], rig["state_base"],
    )
    (state_lax, temp_lax, params_lax, it_lax, delta_lax, res_hist_lax,
     newton_ok_hist, converged_lax) = ceos.ensure_consistency_traceable(
        density_sim, energy_sim, thermo_params, mix_frac_pdf,
        diss, zero_visc, rig["temp_base"], rig["state_base"],
    )

    assert int(it_py) == int(it_lax), (
        f"chi={chi} z_index={z_index}: it mismatch py={int(it_py)} "
        f"lax={int(it_lax)}"
    )
    np.testing.assert_allclose(
        np.asarray(params_py), np.asarray(params_lax),
        rtol=1e-8, atol=1e-6,
        err_msg=f"chi={chi} z_index={z_index}: recovered params mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(state_py.enthalpy), np.asarray(state_lax.enthalpy),
        rtol=1e-8, atol=1e-6,
    )


def test_traceable_always_bounded(rig):
    """Safety property: the traceable path always runs exactly
    config['eos']['maxiter'] internal scan steps (unlike the Python
    version's early break) -- confirm it terminates and 'it' stays in
    range regardless of input, across the same chi/z-index sweep."""
    ceos = rig["compressible_eos"]
    thermo_params = jnp.array([rig["pressure"], rig["h_ox"], rig["h_fu"]])
    zero_visc = jnp.zeros(NUM_X)
    maxiter = ceos.config["eos"]["maxiter"]
    for chi, z_index in [(100.0, 30), (4000.0, 26)]:
        density_sim, energy_sim, diss = _target_density_energy(rig, chi, z_index)
        mix_frac_pdf = jnp.zeros(NUM_X).at[z_index].set(1)
        _, _, _, it, _, _, _, _ = ceos.ensure_consistency_traceable(
            density_sim, energy_sim, thermo_params, mix_frac_pdf,
            diss, zero_visc, rig["temp_base"], rig["state_base"],
        )
        assert 0 <= int(it) <= maxiter - 1


def test_batch_matches_sequential(rig):
    """ensure_consistency_batch (jax.vmap over ensure_consistency_traceable)
    across a batch mixing nominal and hard-tier (near-extinction) points
    at different mixture fractions must match per-point sequential calls
    exactly -- the Stage 5 exit criterion for the full EOS-consistency
    pipeline, not just the inner Newton solve."""
    ceos = rig["compressible_eos"]
    thermo_params = jnp.array([rig["pressure"], rig["h_ox"], rig["h_fu"]])
    zero_visc = jnp.zeros(NUM_X)

    cases = [(100.0, 30), (100.0, 70), (4000.0, 26), (4000.0, 74)]
    batch = len(cases)

    density_list, energy_list, diss_list, pdf_list = [], [], [], []
    for chi, z_index in cases:
        d, e, diss = _target_density_energy(rig, chi, z_index)
        density_list.append(d)
        energy_list.append(e)
        diss_list.append(diss)
        pdf_list.append(jnp.zeros(NUM_X).at[z_index].set(1))

    density_batch = jnp.array(density_list)
    energy_batch = jnp.array(energy_list)
    diss_batch = jnp.stack(diss_list)
    pdf_batch = jnp.stack(pdf_list)
    visc_batch = jnp.tile(zero_visc[None], (batch, 1))
    temp_batch = jnp.tile(rig["temp_base"][None], (batch, 1))
    state_batch = FlameletState(
        enthalpy=jnp.tile(rig["state_base"].enthalpy[None], (batch, 1)),
        mass_fractions=jnp.tile(
            rig["state_base"].mass_fractions[None], (batch, 1, 1)
        ),
    )

    (state_v, temp_v, params_v, it_v, delta_v, res_hist_v,
     newton_ok_v, converged_v) = ceos.ensure_consistency_batch(
        density_batch, energy_batch, thermo_params, pdf_batch,
        diss_batch, visc_batch, temp_batch, state_batch,
    )

    for i, (chi, z_index) in enumerate(cases):
        (state_seq, temp_seq, params_seq, it_seq, _, _, _,
         converged_seq) = ceos.ensure_consistency_traceable(
            density_batch[i], energy_batch[i], thermo_params, pdf_batch[i],
            diss_batch[i], visc_batch[i], temp_batch[i], rig["state_base"],
        )
        assert int(it_v[i]) == int(it_seq), (
            f"chi={chi} z_index={z_index}: it mismatch"
        )
        assert bool(converged_v[i]) == bool(converged_seq)
        np.testing.assert_allclose(
            np.asarray(params_v[i]), np.asarray(params_seq),
            rtol=1e-8, atol=1e-6,
            err_msg=f"chi={chi} z_index={z_index}: batched params mismatch",
        )
        np.testing.assert_allclose(
            np.asarray(state_v.enthalpy[i]), np.asarray(state_seq.enthalpy),
            rtol=1e-8, atol=1e-6,
        )


@pytest.mark.parametrize("chi,z_index", [(100.0, 30), (2000.0, 50)])
def test_picard_traceable_matches_python(rig, chi, z_index):
    """The traceable Picard path must reproduce ensure_consistency's
    Python-loop Picard iterates, and its history must start at the
    initial params."""
    config = _config()
    config["eos"]["update_method"] = "picard"
    ceos = CompressibleEOS(config, rig["solver"])
    density_sim, energy_sim, diss = _target_density_energy(rig, chi, z_index)
    mix_frac_pdf = jnp.zeros(NUM_X).at[z_index].set(1)
    zero_visc = jnp.zeros(NUM_X)
    thermo_params = jnp.array([rig["pressure"], rig["h_ox"], rig["h_fu"]])

    (_, _, params_py, it_py, _, res_py, hist_py, _) = ceos.ensure_consistency(
        density_sim, energy_sim, thermo_params, mix_frac_pdf,
        diss, zero_visc, rig["temp_base"], rig["state_base"],
    )
    (_, _, params_lax, it_lax, _, res_lax, _, _,
     history) = ceos.ensure_consistency_traceable_history(
        density_sim, energy_sim, thermo_params, mix_frac_pdf,
        diss, zero_visc, rig["temp_base"], rig["state_base"],
    )

    assert int(it_py) == int(it_lax)
    np.testing.assert_allclose(
        np.asarray(params_py), np.asarray(params_lax), rtol=1e-8, atol=1e-6,
    )
    n = int(it_lax) + 1
    np.testing.assert_allclose(
        np.asarray(res_py[:n]), np.asarray(res_lax[:n]), rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(hist_py[:n]),
        np.column_stack((history["enthalpy"], history["temperature"]))[:n],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(history["params"][0]), np.asarray(thermo_params),
    )


# -- Adjoint gradients vs. central finite differences ---------------------
# eos_gradient is evaluated at the converged baseline with the converged
# temperature as the guess (the Gauss--Newton loop's iteration-0 call),
# which used to drop all R*T sensitivities (zero-iteration while_loop in
# the temperature inversion). The PDFs exercise the boundary trace: a
# delta next to each wall (the old one-sided trace was 50% off there), an
# interior delta, a broad interior PDF, and weights *on* the Dirichlet
# nodes (direct boundary dependence).

def _pdf_cases(z):
    z = np.asarray(z)
    broad = (z * (1.0 - z)) ** 2
    broad[[0, -1]] = 0.0
    trapezoid = np.full(z.size, 1.0 / (z.size - 1))
    trapezoid[[0, -1]] *= 0.5
    return {
        "delta_ox_wall": np.eye(z.size)[1],
        "delta_fu_wall": np.eye(z.size)[-2],
        "delta_interior": np.eye(z.size)[40],
        "broad_interior": broad / broad.sum(),
        "with_boundary_weights": trapezoid,
    }


@pytest.fixture(scope="module")
def filtered_fd(rig):
    solver = rig["solver"]
    pyro_gas = solver.gov_eqns.pyro_gas
    diss = rig["diss_profile"](1000.0)
    zero_visc = jnp.zeros(NUM_X)

    def fields(params):
        pressure, h_ox, h_fu = params
        state, temp = solver.solve(
            False, 30, 1e-9, 20, 1e-9, 1e-5, 10, False, 10,
            diss, zero_visc, rig["temp_base"], pressure, h_ox, h_fu,
            rig["state_base"],
        )
        y = state.mass_fractions
        return (np.asarray(pyro_gas.get_density(pressure, temp, y)),
                np.asarray(pyro_gas.get_mixture_internal_energy_mass(temp, y)))

    p, h_ox, h_fu = rig["pressure"], rig["h_ox"], rig["h_fu"]
    dp, dh = 1e-4 * p, 1e-4 * abs(h_ox)
    perturbed = {
        "p": [fields((p + s * dp, h_ox, h_fu)) for s in (1, -1)],
        "h": [fields((p, h_ox + s * dh, h_fu + s * dh)) for s in (1, -1)],
    }
    return perturbed, dict(p=dp, h=dh), diss


@pytest.mark.parametrize("pdf_name", [
    "delta_ox_wall", "delta_fu_wall", "delta_interior", "broad_interior",
    "with_boundary_weights",
])
def test_eos_gradient_matches_finite_differences(rig, filtered_fd, pdf_name):
    perturbed, steps, diss = filtered_fd
    w = _pdf_cases(rig["z"])[pdf_name]
    (g_rho, g_e), _ = rig["compressible_eos"].eos_gradient(
        rig["state_base"], jnp.array(w), diss, jnp.zeros(NUM_X),
        rig["temp_base"], rig["pressure"],
    )
    g_adj = np.array([np.asarray(g_rho), np.asarray(g_e)])  # [qoi, param]
    g_fd = np.empty((2, 2))
    for j, key in enumerate(("p", "h")):
        (rho_p, e_p), (rho_m, e_m) = perturbed[key]
        g_fd[0, j] = w @ (rho_p - rho_m) / (2 * steps[key])
        g_fd[1, j] = w @ (e_p - e_m) / (2 * steps[key])
    rel = np.linalg.norm(g_adj - g_fd, axis=1) / np.linalg.norm(g_fd, axis=1)
    assert np.all(rel < 1e-6), (pdf_name, rel, g_adj, g_fd)


# -- Picard fixed point for a non-delta PDF -------------------------------
# The Picard pressure update must share its fixed point with the
# reconstructed density rho_hat = p <1/(R T)>. The former update
# p <- rho_sim <R T> converged to p * <R T><1/(R T)> > p for a PDF spanning
# the flame, stalling the residual at a variance-dependent plateau.

@pytest.mark.parametrize(("method", "picard_gradient"), [
    ("picard", False), ("picard", True), ("gauss_newton", False)])
def test_recovers_target_parameters_broad_pdf(rig, method, picard_gradient):
    import copy
    solver = rig["solver"]
    pyro_gas = solver.gov_eqns.pyro_gas
    diss = rig["diss_profile"](1000.0)
    zero_visc = jnp.zeros(NUM_X)
    w = jnp.array(_pdf_cases(rig["z"])["broad_interior"])

    p_t = 1.5 * rig["pressure"]
    dh = 0.05 * abs(rig["h_ox"])
    params_t = np.array([p_t, rig["h_ox"] + dh, rig["h_fu"] + dh])
    state_t, temp_t = solver.solve(
        False, 30, 1e-9, 20, 1e-9, 1e-5, 10, False, 10,
        diss, zero_visc, rig["temp_base"], *params_t, rig["state_base"],
    )
    y = state_t.mass_fractions
    density_sim = jnp.sum(w * pyro_gas.get_density(p_t, temp_t, y))
    energy_sim = jnp.sum(
        w * pyro_gas.get_mixture_internal_energy_mass(temp_t, y)
    )

    config = copy.deepcopy(_config())
    config["eos"].update(update_method=method, update_size=1.0,
                         maxiter=40, tol=1e-12,
                         picard_gradient=picard_gradient)
    ceos = CompressibleEOS(config, solver)
    params0 = jnp.array([rig["pressure"], rig["h_ox"], rig["h_fu"]])
    (_, _, params, it, _, res, _, _) = ceos.ensure_consistency_traceable(
        density_sim, energy_sim, params0, w, diss, zero_visc,
        rig["temp_base"], rig["state_base"],
    )
    np.testing.assert_allclose(np.asarray(params), params_t, rtol=1e-6)
