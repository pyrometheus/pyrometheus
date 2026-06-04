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
    return (
        0.5 * (integrand[0] + integrand[-1]) +
        jnp.sum(integrand[1:-1])
    )


class CompressibleEOS:

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

        ds_dp = self.fwd_solver.gov_eqns.source_gradient_wrt_pressure(
            state_as_array,
            viscous_diss,
            temp_guess,
            pressure
        ) / (0.5 * diss_rate[:, None])

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
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2  # Replace with a stencil?
        energy_gradient_fu = (
            1.5 * adjoint_state_e.enthalpy[-1]
            + 2 * adjoint_state_e.enthalpy[-2]
            - 0.5 * adjoint_state_e.enthalpy[-3]
        ) / self.fwd_solver.gov_eqns.laplacian.domain.dx ** 2  # Replace with a stencil?
        energy_gradient = jnp.stack((
            energy_gradient_p,
            energy_gradient_ox + energy_gradient_fu
        ))

        return (
            (density_gradient, energy_gradient),
            (adjoint_state_d, adjoint_state_e)
        )

    def evaluate_flamelet(self,
                          params: jnp.ndarray,
                          mixture_fraction_pdf,
                          diss_rate: jnp.ndarray,
                          viscous_diss: jnp.ndarray,
                          temp_guess: jnp.ndarray,
                          state_guess: FlameletState):

        pressure, h_ox, h_fu = params
        t_solve = time.time()
        state, temp = self.fwd_solver.solve(
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

    def _picard_update(self,
                       density_sim: jnp.float64,
                       energy_sim: jnp.float64,
                       params: jnp.ndarray,
                       mixture_fraction_pdf: jnp.ndarray,
                       diss_rate: jnp.ndarray,
                       viscous_diss: jnp.ndarray,
                       temp_guess: jnp.ndarray,
                       state_guess: FlameletState):

        pressure, h_ox, h_fu = params
        t_solve = time.time()
        state, temp = self.fwd_solver.solve(
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

        pressure, h_ox, h_fu = params

        print("Compressible EOS: warming up forward solver")
        t_wmp = time.time()
        s_wmp, _ = self.fwd_solver.solve(
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
        print(f"Compressible EOS: warmup time: {(time.time() - t_wmp):.4e} s")

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
        print(f"Compressible EOS: warmup time: {(time.time() - t_wmp):.4e} s")

        print("Compressible EOS: warming up full adjoint solver")
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
        print(f"Compressible EOS: warmup time: {(time.time() - t_wmp):.4e} s")
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

        state = state_guess
        temp = temp_guess

        a = self.config["eos"]["update_size"]

        update_method = self.config["eos"]["update_method"]
        if update_method == "gauss_newton":
            update_fn = self._gauss_newton_update
            print("Compressible EOS: using Gauss-Newton update")
        elif update_method == "picard":
            update_fn = self._picard_update
            print("Compressible EOS: using Picard update")
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
            print(f"Compressible EOS iteration {it}: "
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
                print(f"Compressible EOS converged at iteration {it}")
                break

        return state, temp, params, it, delta, residual, history, False
