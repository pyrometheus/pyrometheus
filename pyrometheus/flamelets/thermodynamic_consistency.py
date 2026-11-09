"""Compressible-EOS consistency layer on top of the flamelet solver.

In a compressible LES/RANS code, the filtered density and energy of a
non-premixed reacting flow are typically tabulated as moments of the
flamelet state against a presumed mixture-fraction PDF.  This module
recovers the flamelet boundary parameters ``(pressure, h_ox, h_fu)``
that make those filtered quantities consistent with a target
``(density_sim, energy_sim)`` provided by the host solver.

The :class:`CompressibleEOS` class drives the search with either a
Gauss--Newton update (using density- *and* energy-gradient information
from adjoint solves) or a Picard update (using enthalpy gradients
only).  Sensitivities are computed by solving the adjoint of the
flamelet operator returned by
:meth:`FlameletEquations.adjoint_operator`.
"""

import time
import jax
import numpy as np
import jax.numpy as jnp
from typing import Dict
from functools import partial
from pyrometheus.flamelets.solver import FlameletSolver
from pyrometheus.flamelets.state import (
    FlameletState, _state_to_array, _array_to_state
)
from pyrometheus.flamelets.linear_solver import block_thomas


def trapezoidal_rule(integrand):
    """Composite trapezoidal-rule sum on a unit-spacing grid.

    The integrand is assumed to be sampled at the endpoints and at
    every interior node; the spacing is implicit and absorbed into
    the calling convention of the gradient routines.
    """
    return (
        0.5 * (integrand[0] + integrand[-1])
        + jnp.sum(integrand[1:-1])
    )


class CompressibleEOS:
    """Driver that enforces compressible-EOS consistency on a flamelet table.

    Parameters
    ----------
    config : dict
        Nested configuration dictionary supplying tolerances,
        iteration counts and step sizes for the inner Newton, BDF and
        outer EOS-consistency loops; see :meth:`ensure_consistency`
        for the schema used.
    forward_solver : FlameletSolver
        Pre-built flamelet solver supplying the residual, adjoint
        operator and compressible-EOS helpers.

    Attributes
    ----------
    config : dict
        Mutable copy of the user-provided configuration.
    fwd_solver : FlameletSolver
        The wrapped flamelet solver.
    unit_h : jnp.ndarray
        ``(num_vars, num_x)`` selector for the enthalpy component of
        the state, used to build the right-hand side of the adjoint
        problem.
    """

    def __init__(self,
                 config: Dict,
                 forward_solver: FlameletSolver,):
        self.config = config
        self.fwd_solver = forward_solver

        num_v = self.fwd_solver.gov_eqns.pyro_gas.num_species + 1
        num_x = self.fwd_solver.gov_eqns.laplacian.domain.num_x
        self.unit_h = jnp.zeros((num_v, num_x))
        self.unit_h = self.unit_h.at[0].set(jnp.ones(num_x))

    def update_config_option(self, option_path: str, option_val):
        """Set ``config[path][to][option] = option_val``.

        ``option_path`` uses ``"/"`` as a separator.  Intermediate
        levels that do not yet exist are created as empty dictionaries.

        Raises
        ------
        TypeError
            If an intermediate path segment exists but is not a dict.
        """

        keys = option_path.split("/")

        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            if not isinstance(current[k], dict):
                raise TypeError(f"Path segment {k} is not a dict")
            current = current[k]

        current[keys[-1]] = option_val

    @partial(jax.jit, static_argnums=0)
    def enthalpy_gradient(self,
                          state: FlameletState,
                          mixture_fraction_pdf: jnp.ndarray,
                          diss_rate: jnp.ndarray,
                          viscous_diss: jnp.ndarray,
                          temp_guess: jnp.ndarray,
                          pressure: jnp.float64):
        """Adjoint-based enthalpy gradient w.r.t. h_ox, h_fu

        Solves the adjoint equation with right-hand side
        ``-unit_h * mixture_fraction_pdf`` and contracts the boundary
        traces of the adjoint state into the gradient of
        ``<h>`` (the PDF average of ``h``) with respect to the
        oxidizer- and fuel-side boundary enthalpies.  The two
        contributions are combined into a single scalar gradient.

        Returns
        -------
        tuple
            ``(rt, adjoint_state_h, h_gradient)`` where ``rt`` is the
            point-wise ``R T``, ``adjoint_state_h`` is the adjoint
            field, and ``h_gradient`` is the combined boundary-enthalpy
            sensitivity.
        """

        # Get state as array
        state_as_array = _state_to_array(state)
        # Set up the problem
        rt = self.fwd_solver.gov_eqns.compressible_eos_rt(
            state_as_array,
            temp_guess,
            pressure
        )

        adj_op_lower, adj_op_central, adj_op_upper = (
            self.fwd_solver.gov_eqns.adjoint_operator(
                state,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure
            )
        )

        # Adjoint solve for enthalpy
        adj_rhs = -self.unit_h * mixture_fraction_pdf
        adjoint_h_as_array = block_thomas(
            adj_op_lower,
            adj_op_central,
            adj_op_upper,
            adj_rhs.T
        ).T
        adjoint_state_h = _array_to_state(adjoint_h_as_array)

        h_gradient_ox = (
            1.5 * adjoint_state_h.enthalpy[0]
            + 2 * adjoint_state_h.enthalpy[1]
            - 0.5 * adjoint_state_h.enthalpy[2]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        h_gradient_fu = (
            1.5 * adjoint_state_h.enthalpy[-1]
            + 2 * adjoint_state_h.enthalpy[-2]
            - 0.5 * adjoint_state_h.enthalpy[-3]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        h_gradient = h_gradient_ox + h_gradient_fu
        return rt, adjoint_state_h, h_gradient

    @partial(jax.jit, static_argnums=0)
    def eos_gradient(self,
                     state: FlameletState,
                     mixture_fraction_pdf: jnp.ndarray,
                     diss_rate: jnp.ndarray,
                     viscous_diss: jnp.ndarray,
                     temp_guess: jnp.ndarray,
                     pressure: jnp.float64):
        """Adjoint gradient of PDF-averaged density and internal energy.

        Solves two adjoint problems sharing the same operator: one
        whose right-hand side is the density sensitivity
        :math:`(p / (R T)^2)\\, \\nabla_{\\phi} (R T)`, and one whose
        right-hand side carries both the direct sensitivity of the
        internal energy to ``h`` and the indirect contribution through
        ``R T``.  The boundary traces of each adjoint state are
        combined with the source-term sensitivity with respect to
        pressure (also vmap/jacfwd'd in
        :class:`FlameletEquations`) and integrated over the
        mixture-fraction PDF with :func:`trapezoidal_rule` to produce
        the final ``(p, h)`` gradients of the filtered density and
        energy.

        Returns
        -------
        tuple
            ``((density_gradient, energy_gradient),
            (adjoint_state_d, adjoint_state_e))``.  Each gradient is a
            length-2 array ``(d/dp, d/dh_boundary)`` (with the latter
            applied symmetrically to both boundary enthalpies in the
            outer iteration).
        """
        # Get state as array
        state_as_array = _state_to_array(state)
        # Set up the problem
        rt = self.fwd_solver.gov_eqns.compressible_eos_rt(
            state_as_array,
            temp_guess,
            pressure
        )
        rt_jacobian = self.fwd_solver.gov_eqns.compressible_eos_rt_jacobian(
            state_as_array,
            temp_guess,
            pressure
        )

        adj_op_lower, adj_op_central, adj_op_upper = (
            self.fwd_solver.gov_eqns.adjoint_operator(
                state,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure
            )
        )

        ds_dp = self.fwd_solver.gov_eqns.source_gradient_wrt_pressure(
            state_as_array,
            viscous_diss,
            temp_guess,
            pressure
        ) / (0.5 * diss_rate[:, None])

        # Density
        adj_rhs = (pressure / rt**2) * rt_jacobian.T * mixture_fraction_pdf
        adjoint_d_as_array = block_thomas(
            adj_op_lower,
            adj_op_central,
            adj_op_upper,
            adj_rhs.T
        ).T
        adjoint_state_d = _array_to_state(adjoint_d_as_array)
        integrand = (1/rt) * mixture_fraction_pdf + jnp.einsum(
            "ij,ji->i",
            ds_dp,
            adjoint_d_as_array,
        )

        # Notes on the following:
        # Adjoint state already includes a dZ factor implicitly
        # So it is not included in the quadrature, and has to be
        # divided out _twice_ in the derivative
        density_gradient_p = trapezoidal_rule(integrand)
        density_gradient_ox = (
            1.5 * adjoint_state_d.enthalpy[0]
            + 2 * adjoint_state_d.enthalpy[1]
            - 0.5 * adjoint_state_d.enthalpy[2]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        density_gradient_fu = (
            1.5 * adjoint_state_d.enthalpy[-1]
            + 2 * adjoint_state_d.enthalpy[-2]
            - 0.5 * adjoint_state_d.enthalpy[-3]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        density_gradient = jnp.stack((
            density_gradient_p,
            density_gradient_ox + density_gradient_fu
        ))

        # Internal energy
        adj_rhs = -(
            self.unit_h
            - rt_jacobian.T
        ) * mixture_fraction_pdf
        adjoint_e_as_array = block_thomas(
            adj_op_lower,
            adj_op_central,
            adj_op_upper,
            adj_rhs.T
        ).T
        adjoint_state_e = _array_to_state(adjoint_e_as_array)

        integrand = jnp.einsum(
            "ij,ji->i",
            ds_dp,
            adjoint_e_as_array
        )
        energy_gradient_p = trapezoidal_rule(integrand)
        energy_gradient_ox = (
            1.5 * adjoint_state_e.enthalpy[0]
            + 2 * adjoint_state_e.enthalpy[1]
            - 0.5 * adjoint_state_e.enthalpy[2]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        energy_gradient_fu = (
            1.5 * adjoint_state_e.enthalpy[-1]
            + 2 * adjoint_state_e.enthalpy[-2]
            - 0.5 * adjoint_state_e.enthalpy[-3]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2
        energy_gradient = jnp.stack((
            energy_gradient_p,
            energy_gradient_ox + energy_gradient_fu
        ))

        return (
            (density_gradient, energy_gradient),
            (adjoint_state_d, adjoint_state_e)
        )

    @partial(jax.jit, static_argnums=0)
    def eos_gradient_batch(self,
                           state: FlameletState,
                           mixture_fraction_pdf: jnp.ndarray,
                           diss_rate: jnp.ndarray,
                           viscous_diss: jnp.ndarray,
                           temp_guess: jnp.ndarray,
                           pressure: jnp.ndarray):
        """Batched twin of :meth:`eos_gradient` via ``jax.vmap``. Used as
        the post-hoc gradient evaluation at each point's final converged
        ``(state, pressure)`` from :meth:`ensure_consistency_batch` --
        that method itself does not return a final-state gradient (only
        the per-iteration gradients used internally to drive the
        Gauss-Newton search, at each iteration's pre-update params).

        Every argument is batched along its leading axis (matching
        ``ensure_consistency_batch``'s per-point outputs), including
        ``pressure`` (unlike the other batched entry points in this
        module, where boundary conditions are shared/closed-over --
        here each point has its own *recovered* pressure from the EOS
        search, so it must be batched too).

        Returns
        -------
        tuple
            ``((density_gradient, energy_gradient), (adjoint_state_d,
            adjoint_state_e))``, each batched along its leading axis --
            same shape as :meth:`eos_gradient`'s single-point return.
        """
        return jax.vmap(
            self.eos_gradient,
            in_axes=(0, 0, 0, 0, 0, 0),
        )(
            state, mixture_fraction_pdf, diss_rate, viscous_diss,
            temp_guess, pressure,
        )

    def evaluate_flamelet(self,
                          params: jnp.ndarray,
                          mixture_fraction_pdf,
                          diss_rate: jnp.ndarray,
                          viscous_diss: jnp.ndarray,
                          temp_guess: jnp.ndarray,
                          state_guess: FlameletState):
        """Solve flamelet to compute ``(rho, e)`` and their gradients.

        Used as the inner kernel of the Gauss--Newton update: it
        produces the simulated density and energy along with their
        adjoint-based gradients with respect to ``(p, h)``.

        Parameters
        ----------
        params : jnp.ndarray
            Length-3 array ``(pressure, h_ox, h_fu)``.
        mixture_fraction_pdf : jnp.ndarray
            Presumed mixture-fraction PDF (or PDF * dZ) used as the
            integration weight.
        diss_rate, viscous_diss, temp_guess
            See :meth:`FlameletEquations.rhs`.
        state_guess : FlameletState
            Initial guess for the inner flamelet solve.

        Returns
        -------
        tuple
            ``(state, temperature, density, energy, density_gradient,
            energy_gradient)``.
        """
        pressure, h_ox, h_fu = params
        t_solve = time.time()
        state, temp = self.fwd_solver.solve(
            self.config["verbosity"],
            self.config["newton"]["maxiter"],
            self.config["newton"]["tol"],
            self.config["bdf"]["newton"]["maxiter"],
            self.config["bdf"]["newton"]["tol"],
            self.config["bdf"]["time_step"],
            self.config["bdf"]["maxsteps"],
            True,
            self.config["max_attempts"],
            diss_rate,
            viscous_diss,
            temp_guess,
            pressure,
            h_ox,
            h_fu,
            state_guess
        )
        state.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"solve time: {(time.time() - t_solve):.4e} s")

        t_adj = time.time()
        (density_gradient, energy_gradient), (adj_d, adj_e) = (
            self.eos_gradient(
                state,
                mixture_fraction_pdf,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
            )
        )
        adj_d.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"adjoint time: {(time.time() - t_adj):.4e} s")

        rt = self.fwd_solver.gov_eqns.compressible_eos_rt(
            _state_to_array(state),
            temp,
            pressure
        )
        density = jnp.sum(
            (pressure / rt) * mixture_fraction_pdf
        )
        energy = jnp.sum(
            (state.enthalpy - rt) * mixture_fraction_pdf
        )
        return (
            state,
            temp,
            density,
            energy,
            density_gradient,
            energy_gradient
        )

    def _gauss_newton_update(self,
                             density_sim: jnp.float64,
                             energy_sim: jnp.float64,
                             params: jnp.ndarray,
                             mixture_fraction_pdf: jnp.ndarray,
                             diss_rate: jnp.ndarray,
                             viscous_diss: jnp.ndarray,
                             temp_guess: jnp.ndarray,
                             state_guess: FlameletState):
        """Single Gauss--Newton update for ``(pressure, h_boundary)``.

        Uses :meth:`evaluate_flamelet` to gather the
        density/energy residuals and their 2x2 Jacobian, then solves
        the normal equations.  The single boundary-enthalpy update
        is mirrored to both ``h_ox`` and ``h_fu`` to preserve a fixed
        ``h_ox - h_fu``.

        Returns
        -------
        tuple
            ``(state, temperature, update, residual)`` where
            ``update`` is the length-3 search direction to be applied
            to ``params`` and ``residual`` is the length-2 mismatch
            vector.
        """

        state, temp, density, energy, d_grad, e_grad = self.evaluate_flamelet(
            params,
            mixture_fraction_pdf,
            diss_rate,
            viscous_diss,
            temp_guess,
            state_guess
        )
        grad_matrix = jnp.stack((
            d_grad,
            e_grad
        ))
        residual = jnp.stack((
            density - density_sim,
            energy - energy_sim
        ))

        v = jnp.linalg.solve(
            grad_matrix.T @ grad_matrix,
            -grad_matrix.T @ residual
        )
        update = jnp.array([v[0], v[1], v[1]])
        return state, temp, update, residual

    def _steady_newton_fn(self, state, diss_rate, viscous_diss, temp_guess,
                          pressure, h_ox, h_fu):
        return _state_to_array(
            self.fwd_solver.gov_eqns.rhs(
                state, diss_rate, viscous_diss, temp_guess,
                pressure, h_ox, h_fu
            )
        )

    def _steady_newton_jac(self, state, diss_rate, viscous_diss, temp_guess,
                           pressure, h_ox, h_fu):
        return self.fwd_solver.gov_eqns.jac(
            state, diss_rate, viscous_diss, temp_guess, pressure
        )

    def evaluate_flamelet_traceable(self,
                                    params: jnp.ndarray,
                                    mixture_fraction_pdf,
                                    diss_rate: jnp.ndarray,
                                    viscous_diss: jnp.ndarray,
                                    temp_guess: jnp.ndarray,
                                    state_guess: FlameletState):
        """Traceable (bounded, no BDF fallback) twin of
        :meth:`evaluate_flamelet`, for use inside a ``jax.lax.scan``/
        ``jax.vmap`` batch.

        :meth:`evaluate_flamelet` calls :meth:`FlameletSolver.solve`,
        whose Python-level Newton-then-BDF-fallback orchestration can't
        be traced (data-dependent early return on a JAX array's runtime
        value). This uses :meth:`FlameletSolver._newton_loop_lax_steady`
        (the bounded, divergence-aware lax loop) instead -- the "vmap the
        pure-Newton (fast) path" strategy from the plan's Stage 4
        contingency tier 2. Points whose Newton attempt fails (returned
        ``success=False``) are NOT retried via BDF here; the caller is
        responsible for gathering those and re-solving via the existing
        :meth:`evaluate_flamelet`/:meth:`FlameletSolver.solve` path.

        Returns
        -------
        tuple
            ``(state, temperature, density, energy, density_gradient,
            energy_gradient, newton_success)`` -- same as
            :meth:`evaluate_flamelet` plus the Newton success flag.
        """
        pressure, h_ox, h_fu = params
        state, it, delta, newton_success = self.fwd_solver._newton_loop_lax_steady(
            self._steady_newton_fn,
            self._steady_newton_jac,
            state_guess,
            self.config["newton"]["maxiter"],
            self.config["newton"]["tol"],
            diss_rate,
            viscous_diss,
            temp_guess,
            pressure,
            h_ox,
            h_fu,
        )
        temp = self.fwd_solver.gov_eqns.pyro_gas.get_temperature_from_enthalpy(
            state.enthalpy, state.mass_fractions, temp_guess
        )

        (density_gradient, energy_gradient), (adj_d, adj_e) = (
            self.eos_gradient(
                state,
                mixture_fraction_pdf,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
            )
        )

        rt = self.fwd_solver.gov_eqns.compressible_eos_rt(
            _state_to_array(state),
            temp,
            pressure
        )
        density = jnp.sum(
            (pressure / rt) * mixture_fraction_pdf
        )
        energy = jnp.sum(
            (state.enthalpy - rt) * mixture_fraction_pdf
        )
        return (
            state,
            temp,
            density,
            energy,
            density_gradient,
            energy_gradient,
            newton_success,
        )

    def _gauss_newton_update_traceable(self,
                                       density_sim: jnp.float64,
                                       energy_sim: jnp.float64,
                                       params: jnp.ndarray,
                                       mixture_fraction_pdf: jnp.ndarray,
                                       diss_rate: jnp.ndarray,
                                       viscous_diss: jnp.ndarray,
                                       temp_guess: jnp.ndarray,
                                       state_guess: FlameletState):
        """Traceable twin of :meth:`_gauss_newton_update`, built on
        :meth:`evaluate_flamelet_traceable`. See that method's docstring
        for what's different (no BDF fallback).

        Returns
        -------
        tuple
            ``(state, temperature, update, residual, newton_success)``.
        """
        (state, temp, density, energy, d_grad, e_grad,
         newton_success) = self.evaluate_flamelet_traceable(
            params,
            mixture_fraction_pdf,
            diss_rate,
            viscous_diss,
            temp_guess,
            state_guess
        )
        grad_matrix = jnp.stack((
            d_grad,
            e_grad
        ))
        residual = jnp.stack((
            density - density_sim,
            energy - energy_sim
        ))

        v = jnp.linalg.solve(
            grad_matrix.T @ grad_matrix,
            -grad_matrix.T @ residual
        )
        update = jnp.array([v[0], v[1], v[1]])
        return state, temp, update, residual, newton_success

    def _picard_update(self,
                       density_sim: jnp.float64,
                       energy_sim: jnp.float64,
                       params: jnp.ndarray,
                       mixture_fraction_pdf: jnp.ndarray,
                       diss_rate: jnp.ndarray,
                       viscous_diss: jnp.ndarray,
                       temp_guess: jnp.ndarray,
                       state_guess: FlameletState):
        """Single Picard-style update for ``(pressure, h_boundary)``.

        Decouples the two updates: ``pressure`` is updated by the
        ideal-gas relation ``rho * <R T> = p`` evaluated under the
        PDF, and the boundary enthalpy is updated using only the
        cheaper enthalpy-only adjoint gradient from
        :meth:`enthalpy_gradient`.  Cheaper than
        :meth:`_gauss_newton_update` but converges more slowly.

        Returns
        -------
        tuple
            ``(state, temperature, update, residual)``.
        """

        pressure, h_ox, h_fu = params
        t_solve = time.time()
        state, temp = self.fwd_solver.solve(
            self.config["verbosity"],
            self.config["newton"]["maxiter"],
            self.config["newton"]["tol"],
            self.config["bdf"]["newton"]["maxiter"],
            self.config["bdf"]["newton"]["tol"],
            self.config["bdf"]["time_step"],
            self.config["bdf"]["maxsteps"],
            True,
            self.config["max_attempts"],
            diss_rate,
            viscous_diss,
            temp_guess,
            pressure,
            h_ox,
            h_fu,
            state_guess
        )
        state.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"solve time: {(time.time() - t_solve):.4e} s")

        rt, _, dh_ox = self.enthalpy_gradient(
            state,
            mixture_fraction_pdf,
            diss_rate,
            viscous_diss,
            temp_guess,
            pressure
        )
        density = jnp.sum(
            (pressure / rt) * mixture_fraction_pdf
        )
        energy = jnp.sum(
            (state.enthalpy - rt) * mixture_fraction_pdf
        )
        dp = density_sim * jnp.sum(
            rt * mixture_fraction_pdf
        ) - pressure
        dh = (energy_sim - energy) / dh_ox

        update = jnp.array([
            dp,
            dh,
            dh  # Assumes fixed \Delta h = h_ox - h_fu
        ])
        residual = jnp.array([
            density_sim - density,
            energy_sim - energy
        ])
        return state, temp, update, residual

    def warmup(self,
               params: jnp.ndarray,
               mixture_fraction_pdf: jnp.ndarray,
               diss_rate: jnp.ndarray,
               viscous_diss: jnp.ndarray,
               temp_wmp: jnp.ndarray,
               state_wmp: FlameletState):
        """Trigger JIT compilation of the forward and adjoint solves.

        Runs one full :meth:`FlameletSolver.solve` followed by one
        :meth:`enthalpy_gradient` and one :meth:`eos_gradient` so that
        the JIT cost is paid up front rather than during the first
        consistency iteration.
        """
        pressure, h_ox, h_fu = params

        if self.config["verbosity"]:
            print("Compressible EOS: warming up forward solver")

        t_wmp = time.time()
        s_wmp, _ = self.fwd_solver.solve(
            self.config["verbosity"],
            self.config["newton"]["maxiter"],
            self.config["newton"]["tol"],
            self.config["bdf"]["newton"]["maxiter"],
            self.config["bdf"]["newton"]["tol"],
            self.config["bdf"]["time_step"],
            self.config["bdf"]["maxsteps"],
            True,
            self.config["max_attempts"],
            diss_rate,
            viscous_diss,
            temp_wmp,
            pressure,
            h_ox,
            h_fu,
            state_wmp
        )
        s_wmp.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"EOS: warmup time: {(time.time() - t_wmp):.4e} s")

        if self.config["verbosity"]:
            print("Compressible EOS: warming up h-only adjoint solver")

        t_wmp = time.time()
        _, adj_wmp, _ = self.enthalpy_gradient(
            state_wmp,
            mixture_fraction_pdf,
            diss_rate,
            viscous_diss,
            temp_wmp,
            pressure,
        )
        adj_wmp.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"EOS: warmup time: {(time.time()-t_wmp):.4e} s")
            print("EOS: warming up full adjoint solver")

        t_wmp = time.time()
        _, (adj_wmp, _) = self.eos_gradient(
            state_wmp,
            mixture_fraction_pdf,
            diss_rate,
            viscous_diss,
            temp_wmp,
            pressure,
        )
        adj_wmp.enthalpy.block_until_ready()
        if self.config["verbosity"]:
            print(f"EOS: warmup time: {(time.time()-t_wmp):.4e} s")

        return

    def ensure_consistency(self,
                           density_sim: jnp.float64,
                           energy_sim: jnp.float64,
                           params: jnp.ndarray,
                           mixture_fraction_pdf: jnp.ndarray,
                           diss_rate: jnp.ndarray,
                           viscous_diss: jnp.ndarray,
                           temp_guess: jnp.ndarray,
                           state_guess: FlameletState):
        """Iterate ``(p, h_ox, h_fu)`` until filtered EOS matches target.

        Repeatedly calls :meth:`_gauss_newton_update` or
        :meth:`_picard_update` (selected by
        ``config["eos"]["update_method"]``), applying a damped update
        ``params + a * v`` with ``a = config["eos"]["update_size"]``.
        Stops when ``|v|`` falls below ``config["eos"]["tol"]`` or
        when ``config["eos"]["maxiter"]`` iterations have been
        performed.

        Parameters
        ----------
        density_sim, energy_sim : float
            Target filtered density and internal energy provided by
            the host compressible solver.
        params : jnp.ndarray
            Initial ``(pressure, h_ox, h_fu)``.
        mixture_fraction_pdf, diss_rate, viscous_diss, temp_guess,
        state_guess
            See :meth:`evaluate_flamelet`.

        Returns
        -------
        tuple
            ``(state, temperature, params, it, delta, residual,
            history, success)``.  ``residual`` is the per-iteration
            squared residual history, ``history`` is the per-iteration
            ``(<h>, <T>)`` history under the PDF, and ``success`` is
            always ``False`` (the trailing flag is reserved for
            future use).

        Raises
        ------
        ValueError
            If ``config["eos"]["update_method"]`` is not one of
            ``"gauss_newton"`` or ``"picard"``.
        """
        state = state_guess
        temp = temp_guess

        a = self.config["eos"]["update_size"]

        update_method = self.config["eos"]["update_method"]
        if update_method == "gauss_newton":
            update_fn = self._gauss_newton_update
            if self.config["verbosity"]:
                print("EOS: using Gauss-Newton update")
        elif update_method == "picard":
            update_fn = self._picard_update
            if self.config["verbosity"]:
                print("EOS: using Picard update")
        else:
            raise ValueError(f"Available {update_method} not implemented")

        residual = np.full(self.config["eos"]["maxiter"], np.nan)
        history = np.full(
            (self.config["eos"]["maxiter"], 2), np.nan
        )
        for it in range(self.config["eos"]["maxiter"]):
            t_iter = time.time()
            state, temp, v, res = update_fn(
                density_sim,
                energy_sim,
                params,
                mixture_fraction_pdf,
                diss_rate,
                viscous_diss,
                temp,
                state
            )
            v.block_until_ready()
            cost_val = jnp.linalg.norm(res)**2
            delta = jnp.linalg.norm(v)
            if self.config["verbosity"]:
                print(f"EOS iteration {it}: "
                      f"time: {(time.time()-t_iter):.4e} s"
                      f", residual = {cost_val:.4e}"
                      f", |v| = {delta:.4e}"
                      ", new params = [{:s}]".format(
                          ", ".join([
                              f"{a:.4e}" for a in params + a * v
                          ])
                      ))
            params = params + a * v
            residual[it] = cost_val
            history[it] = np.array([
                jnp.sum(state.enthalpy * mixture_fraction_pdf),
                jnp.sum(temp * mixture_fraction_pdf)
            ])
            if delta < self.config["eos"]["tol"]:
                if self.config["verbosity"]:
                    print(f"EOS converged at iteration {it}")

                break

        return state, temp, params, it, delta, residual, history, False

    def ensure_consistency_traceable(self,
                                     density_sim: jnp.float64,
                                     energy_sim: jnp.float64,
                                     params: jnp.ndarray,
                                     mixture_fraction_pdf: jnp.ndarray,
                                     diss_rate: jnp.ndarray,
                                     viscous_diss: jnp.ndarray,
                                     temp_guess: jnp.ndarray,
                                     state_guess: FlameletState):
        """Bounded, traceable/batchable twin of :meth:`ensure_consistency`.

        Runs a ``jax.lax.while_loop`` that stops as soon as ``|v| < tol``
        (or ``config["eos"]["maxiter"]`` iterations have been performed,
        whichever comes first) -- rather than a Python ``for`` loop with
        an early ``break`` (which can't be traced), and rather than an
        earlier version of this method that used a fixed-``maxiter``
        ``jax.lax.scan`` with converged-lane masking: that scan always
        paid for the full ``maxiter`` forward+adjoint passes even when
        every point converged on iteration 0 (confirmed empirically --
        see the plan's Stage 5 GPU-scaling notes -- a ~10x waste on a
        batch that happened to all converge immediately). Under
        ``jax.vmap``, JAX's own batching rule for ``lax.while_loop``
        already freezes a lane's carry once *that lane's* predicate goes
        false, while the whole batch keeps running until the *slowest*
        lane's predicate does -- so a mixed batch still pays for its
        hardest point's iteration count, but no longer pays ``maxiter``
        for every point regardless of difficulty. No manual masking is
        needed here (unlike the scan version): since the loop only calls
        ``body_fn`` while not yet converged, every call's update is
        unconditionally the right one to apply.

        Only ``update_method == "gauss_newton"`` is supported (uses
        :meth:`_gauss_newton_update_traceable`, built on
        :meth:`evaluate_flamelet_traceable` -- no BDF fallback, see that
        method's docstring). ``picard`` is not yet ported to this
        traceable path.

        Returns
        -------
        tuple
            ``(state, temp, params, it, delta, residual_hist,
            newton_success_hist, converged)`` -- ``it`` is the iteration
            index at which ``converged`` first became ``True`` (or
            ``maxiter - 1`` if it never did), matching
            :meth:`ensure_consistency`'s convention.
            ``residual_hist``/``newton_success_hist`` are length-
            ``maxiter`` per-iteration histories, ``nan``/``False``-filled
            past the iteration where the loop actually stopped (a
            correctness improvement over the old scan version, which
            filled those slots with repeated recomputations at the
            frozen, already-converged params instead of leaving them
            empty -- no caller currently reads these histories, but the
            new values are the more honest ones).
        """
        if self.config["eos"]["update_method"] != "gauss_newton":
            raise ValueError(
                "ensure_consistency_traceable only supports "
                "update_method='gauss_newton'"
            )

        a = self.config["eos"]["update_size"]
        tol = self.config["eos"]["tol"]
        maxiter = self.config["eos"]["maxiter"]

        def cond_fn(carry):
            _, _, _, it, _, converged, _, _ = carry
            return jnp.logical_and(jnp.logical_not(converged), it < maxiter)

        def body_fn(carry):
            state, temp, params, it, _, _, residual_hist, newton_success_hist = carry
            (state_new, temp_new, update, residual,
             newton_success) = self._gauss_newton_update_traceable(
                density_sim, energy_sim, params, mixture_fraction_pdf,
                diss_rate, viscous_diss, temp, state
            )
            new_delta = jnp.linalg.norm(update)
            new_converged = new_delta < tol
            new_params = params + a * update
            cost_val = jnp.linalg.norm(residual) ** 2

            return (
                state_new, temp_new, new_params, it + 1, new_delta,
                new_converged,
                residual_hist.at[it].set(cost_val),
                newton_success_hist.at[it].set(newton_success),
            )

        carry_init = (
            state_guess, temp_guess, params,
            jnp.array(0, dtype=jnp.int32), jnp.array(jnp.inf),
            jnp.array(False),
            jnp.full(maxiter, jnp.nan),
            jnp.zeros(maxiter, dtype=bool),
        )
        (state, temp, params, it, delta, converged, residual_hist,
         newton_success_hist) = jax.lax.while_loop(
            cond_fn, body_fn, carry_init
        )

        return (
            state, temp, params, it - 1, delta,
            residual_hist, newton_success_hist, converged,
        )

    @partial(jax.jit, static_argnums=0)
    def ensure_consistency_batch(self,
                                 density_sim: jnp.ndarray,
                                 energy_sim: jnp.ndarray,
                                 params: jnp.ndarray,
                                 mixture_fraction_pdf: jnp.ndarray,
                                 diss_rate: jnp.ndarray,
                                 viscous_diss: jnp.ndarray,
                                 temp_guess: jnp.ndarray,
                                 state_guess: FlameletState):
        """Batched twin of :meth:`ensure_consistency_traceable` via
        ``jax.vmap``, for many points at once sharing the same initial
        ``params`` guess -- the fixed physical baseline boundary
        conditions, the same for every point (see the plan's Stage 1
        correction / ``FlameletModel.baseline_params`` in
        ``synthetic_data.py``). Only what varies per point
        (``density_sim``/``energy_sim``/``mixture_fraction_pdf``/
        ``diss_rate``/``viscous_diss``/``temp_guess``/``state_guess``) is
        batched; ``self`` and ``params`` stay closed-over/shared, the
        same sharing pattern already used by
        :meth:`FlameletSolver.flamelet_newton_step`'s ``static_argnums``.

        Returns
        -------
        tuple
            Same fields as :meth:`ensure_consistency_traceable`, each
            batched along its leading axis.
        """
        return jax.vmap(
            self.ensure_consistency_traceable,
            in_axes=(0, 0, None, 0, 0, 0, 0, 0),
        )(
            density_sim, energy_sim, params, mixture_fraction_pdf,
            diss_rate, viscous_diss, temp_guess, state_guess,
        )
