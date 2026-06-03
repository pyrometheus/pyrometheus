import time
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
from pyrometheus.flamelets.state import (
    FlameletState,
    _state_to_array,
    _array_to_state
)
from pyrometheus.flamelets.equations import FlameletEquations
from pyrometheus.flamelets.onedim_laplacian import Laplacian
from pyrometheus.flamelets.linear_solver import block_thomas


class FlameletSolver:

    def __init__(self, domain, pyro_gas, mass_frac_bcs):
        y_ox, y_fu = mass_frac_bcs
        self.domain = domain
        self.gov_eqns = FlameletEquations(
            pyro_gas,
            Laplacian(domain),
            y_ox, y_fu
        )
        self.num_v = self.gov_eqns.pyro_gas.num_species + 1

    def _newton_update(self,
                       fn: Callable,
                       jac: Callable,
                       state: FlameletState,
                       *args,):
        rhs = fn(
            state,
            *args
        )
        jac_lower, jac_central, jac_upper = jac(
            state,
            *args
        )
        return _array_to_state(
            block_thomas(
                jac_lower,
                jac_central,
                jac_upper,
                -rhs.T
            ).T
        )

    def _newton_loop_py(self,
                        state: FlameletState,
                        maxiter: int,
                        tol: jnp.float64,
                        *args,):
        err_prev = jnp.inf
        for it in range(maxiter):
            t_iter = time.time()
            v = self.flamelet_newton_step(
                state,
                *args
            )
            v.enthalpy.block_until_ready()
            print(f'Newton iteration {it}: time: {(time.time()-t_iter):.4e} s'
                  f', |v| = {jnp.linalg.norm(_state_to_array(v)):.4e}')
            state += v
            delta = jnp.linalg.norm(
                _state_to_array(v)
            )
            if delta < tol:
                return state, it, delta, True

            if err_prev < delta:
                return state, it, delta, False

            err_prev = delta

        return state, it, delta, False

    def _newton_loop_lax(self,
                         fn: Callable,
                         jac: Callable,
                         state: FlameletState,
                         maxiter: int,
                         tol: jnp.float64,
                         *args,):

        def cond_fn(carry):
            _, it, delta = carry
            return jnp.logical_and(
                delta > tol,
                it < maxiter
            )

        def body_fn(carry):
            state, it, _ = carry
            v = self._newton_update(
                fn,
                jac,
                state,
                *args
            )
            return tuple((
                state + v,
                it + 1,
                jnp.linalg.norm(
                    _state_to_array(v)
                )
            ))

        carry_init = (
            state,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(jnp.inf)
        )
        return jax.lax.while_loop(
            cond_fn,
            body_fn,
            carry_init
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def flamelet_time_step(self,
                           state: FlameletState,
                           newton_maxiter: int,
                           newton_tol: jnp.float64,
                           state_prev: FlameletState,
                           time_step: jnp.float64,
                           diss_rate: jnp.ndarray,
                           viscous_diss: jnp.ndarray,
                           temp_guess: jnp.ndarray,
                           pressure: jnp.float64,
                           h_ox: jnp.float64,
                           h_fu: jnp.float64,):

        def _crank_nicolson_fn(state: FlameletState,
                               state_prev: FlameletState,
                               time_step: jnp.float64,
                               diss_rate: jnp.ndarray,
                               viscous_diss: jnp.ndarray,
                               temp_guess: jnp.ndarray,
                               pressure: jnp.float64,
                               h_ox: jnp.float64,
                               h_fu: jnp.float64,):
            flamelet_rhs = self.gov_eqns.rhs(
                state,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
                h_ox,
                h_fu
            )
            flamelet_rhs_prev = self.gov_eqns.rhs(
                state_prev,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
                h_ox,
                h_fu
            )
            return _state_to_array(
                state - state_prev - 0.5 * time_step * (
                    flamelet_rhs + flamelet_rhs_prev
                )
            )

        def _crank_nicolson_jac(state: FlameletState,
                                state_prev: FlameletState,
                                time_step: jnp.float64,
                                diss_rate: jnp.ndarray,
                                viscous_diss: jnp.ndarray,
                                temp_guess: jnp.ndarray,
                                pressure: jnp.float64,
                                h_ox: jnp.float64,
                                h_fu: jnp.float64,):
            flamelet_jac = self.gov_eqns.jac(
                state,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure
            )
            return (
                -0.5 * time_step * flamelet_jac[0],
                jnp.eye(self.num_v) - 0.5 * time_step * flamelet_jac[1],
                -0.5 * time_step * flamelet_jac[2]
            )

        return self._newton_loop_lax(
                _crank_nicolson_fn,
                _crank_nicolson_jac,
                state,
                newton_maxiter,
                newton_tol,
                state_prev,
                time_step,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
                h_ox,
                h_fu
            )

    @partial(jax.jit, static_argnums=0)
    def flamelet_newton_step(self,
                             state_guess: FlameletState,
                             diss_rate: jnp.ndarray,
                             viscous_diss: jnp.ndarray,
                             temp_guess: jnp.ndarray,
                             pressure: jnp.float64,
                             h_ox: jnp.float64,
                             h_fu: jnp.float64,):

        def _newton_fn(state,
                       diss_rate: jnp.ndarray,
                       viscous_diss: jnp.ndarray,
                       temp_guess: jnp.ndarray,
                       pressure: jnp.float64,
                       h_ox: jnp.float64,
                       h_fu: jnp.float64,):
            return _state_to_array(
                self.gov_eqns.rhs(
                    state,
                    diss_rate,
                    viscous_diss,
                    temp_guess,
                    pressure,
                    h_ox,
                    h_fu,
                )
            )

        def _newton_jac(state,
                        diss_rate: jnp.ndarray,
                        viscous_diss: jnp.ndarray,
                        temp_guess: jnp.ndarray,
                        pressure: jnp.float64,
                        h_ox: jnp.float64,
                        h_fu: jnp.float64,):
            return self.gov_eqns.jac(
                state,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure
            )

        return self._newton_update(
            _newton_fn,
            _newton_jac,
            state_guess,
            diss_rate,
            viscous_diss,
            temp_guess,
            pressure,
            h_ox,
            h_fu
        )

    def flamelet_time_march(self,
                            newton_maxiter: int,
                            newton_tol: jnp.float64,
                            maxsteps: int,
                            time_step: jnp.float64,
                            diss_rate: jnp.ndarray,
                            viscous_diss: jnp.ndarray,
                            temp_guess: jnp.ndarray,
                            pressure: jnp.float64,
                            h_ox: jnp.float64,
                            h_fu: jnp.float64,
                            state_guess: FlameletState,):

        state = state_guess
        temp = temp_guess
        for step in range(maxsteps):
            state_prev = state
            t_start = time.time()
            state, newton_it, newton_err = self.flamelet_time_step(
                state,
                newton_maxiter,
                newton_tol,
                state_prev,
                time_step,
                diss_rate,
                viscous_diss,
                temp,
                pressure,
                h_ox,
                h_fu
            )
            state.enthalpy.block_until_ready()
            t_step = time.time() - t_start
            temp = self.gov_eqns.pyro_gas.get_temperature_from_enthalpy(
                state.enthalpy,
                state.mass_fractions,
                temp_guess
            )
            print(f'BDF time step {step}: '
                  f'time: {t_step:.4e} s, '
                  f'T_max = {temp.max():.3f} K ')
        print(f'BDF Time march: T_max = {temp.max():.3f} K')
        return state, temp

    def warmup(self, warmup_fn_name: str, *args):
        if warmup_fn_name == 'flamelet_newton_step':
            return self.flamelet_newton_step(
                *args
            )
        elif warmup_fn_name == 'flamelet_time_step':
            return self.flamelet_time_step(
                *args
            )
        else:
            raise ValueError(f'Unknown warmup function name {warmup_fn_name}')

    def solve(self,
              newton_maxiter: int,
              newton_tol: jnp.float64,
              bdf_newton_maxiter: int,
              bdf_newton_tol: jnp.float64,
              bdf_time_step: jnp.float64,
              bdf_maxsteps: int,
              try_newton: bool,
              max_attempts: int,
              diss_rate: jnp.ndarray,
              viscous_diss: jnp.ndarray,
              temp_guess: jnp.ndarray,
              pressure: jnp.float64,
              h_ox: jnp.float64,
              h_fu: jnp.float64,
              state_guess: FlameletState,):

        if try_newton:
            state, newton_it, newton_err, success = self._newton_loop_py(
                state_guess,
                newton_maxiter,
                newton_tol,
                diss_rate,
                viscous_diss,
                temp_guess,
                pressure,
                h_ox,
                h_fu
            )
            if success:
                print(f'Converged after {newton_it} Newton iterations with '
                      f'error {newton_err:.4e}')
                temp = self.gov_eqns.pyro_gas.get_temperature_from_enthalpy(
                    state.enthalpy,
                    state.mass_fractions,
                    temp_guess
                )
                return state, temp

        state = state_guess
        temp = temp_guess
        for i in range(max_attempts):
            print(f'Attempt {i} of BDF time marching')
            state, temp = self.flamelet_time_march(
                bdf_newton_maxiter,
                bdf_newton_tol,
                bdf_maxsteps,
                bdf_time_step,
                diss_rate,
                viscous_diss,
                temp,
                pressure,
                h_ox,
                h_fu,
                state
            )
            state_t, newton_it, newton_err, success = self._newton_loop_py(
                state,
                newton_maxiter,
                newton_tol,
                diss_rate,
                viscous_diss,
                temp,
                pressure,
                h_ox,
                h_fu
            )
            temp_t = self.gov_eqns.pyro_gas.get_temperature_from_enthalpy(
                state.enthalpy,
                state.mass_fractions,
                temp_guess
            )
            if success:
                print(f'Converged after {i+1} BDF attempts with '
                      f'error {newton_err:.4e}')
                return state_t, temp_t

        print('Flamelet solution did not converge')
        return state, temp
