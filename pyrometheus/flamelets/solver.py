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
    """Top-level driver that solves the steady flamelet equations.

    The solver assembles a :class:`FlameletEquations` on top of a
    :class:`Laplacian` and provides three layers of interface:

    1. :meth:`flamelet_newton_step` -- one Newton step on the steady
       residual, JIT-compiled with respect to ``self``.
    2. :meth:`flamelet_time_step` -- one implicit (Crank--Nicolson)
       time step plus Newton solve, used as a globalisation strategy.
    3. :meth:`solve` -- a Python-level driver that first attempts a
       pure Newton iteration, and falls back to a BDF-style time
       march followed by Newton, restarting up to ``max_attempts``
       times if Newton still fails to converge.

    Linear sub-problems are solved with :func:`block_thomas`.

    Parameters
    ----------
    domain : Domain
        Mixture-fraction grid.
    pyro_gas : object
        Pyrometheus thermochemistry object.
    mass_frac_bcs : tuple of jnp.ndarray
        Pair ``(y_ox, y_fu)`` with the oxidizer- and fuel-side species
        mass fractions.

    Attributes
    ----------
    domain : Domain
        Mixture-fraction grid.
    gov_eqns : FlameletEquations
        Governing-equation assembly.
    num_v : int
        Number of unknowns per grid point (``1 + num_species``).
    """

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
        """Single Newton update ``v = -J(state)^{-1} R(state)``.

        Parameters
        ----------
        fn : Callable
            Residual function ``state, *args -> FlameletState``.
        jac : Callable
            Jacobian-block function returning
            ``(lower, central, upper)`` ready for
            :func:`block_thomas`.
        state : FlameletState
            Current iterate.

        Returns
        -------
        FlameletState
            Newton update ``v``.
        """
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
                        verbosity: bool,
                        state: FlameletState,
                        maxiter: int,
                        tol: jnp.float64,
                        *args,):
        """Plain-Python Newton iteration for the steady residual.

        Wraps :meth:`flamelet_newton_step`, prints iteration timing,
        and stops on either convergence (``|v| < tol``) or divergence
        (``|v|`` larger than the previous iterate).

        Returns
        -------
        tuple
            ``(state, it, delta, success)`` where ``state`` is the
            final iterate, ``it`` is the iteration count, ``delta`` is
            the last update norm and ``success`` is ``True`` iff the
            iteration converged within ``maxiter`` steps.
        """
        err_prev = jnp.inf
        for it in range(maxiter):
            t_iter = time.time()
            v = self.flamelet_newton_step(
                state,
                *args
            )
            v.enthalpy.block_until_ready()
            if verbosity:
                print(f"Newton iter {it}: time: {(time.time()-t_iter):.4e} s"
                      f", |v| = {jnp.linalg.norm(_state_to_array(v)):.4e}")

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
        """JAX-traced Newton loop using ``jax.lax.while_loop``.

        Iterates :meth:`_newton_update` until either ``|v| < tol`` or
        ``maxiter`` iterations have been performed.  The whole loop is
        traceable so callers may JIT it.

        Returns
        -------
        tuple
            ``(state, it, delta)`` from the final ``while_loop``
            carry.
        """

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
        r"""One Crank--Nicolson time step solved with Newton's method.

        Builds the Crank--Nicolson residual and Jacobian for

        .. math::

            \\phi^{n+1} - \\phi^{n}
            - \\tfrac{1}{2}\\, \\Delta t\\, \\big(R(\\phi^{n+1})
            + R(\\phi^{n})\\big)
            = 0,

        and drives it to convergence with
        :meth:`_newton_loop_lax`.  Used as a globalisation step when
        a pure steady Newton iteration fails.

        Parameters
        ----------
        state : FlameletState
            Newton initial guess (typically ``state_prev``).
        newton_maxiter : int
            Maximum number of inner Newton iterations.
        newton_tol : float
            Newton convergence tolerance on ``|v|``.
        state_prev : FlameletState
            State at the previous time step.
        time_step : float
            Time-step size :math:`\\Delta t`.
        diss_rate, viscous_diss, temp_guess, pressure, h_ox, h_fu
            See :meth:`FlameletEquations.rhs`.

        Returns
        -------
        tuple
            ``(state, it, delta)`` from the underlying
            :meth:`_newton_loop_lax`.
        """

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
        """One JIT-compiled Newton step on the steady flamelet residual.

        Forms the residual via :meth:`FlameletEquations.rhs` and the
        block-tridiagonal Jacobian via :meth:`FlameletEquations.jac`,
        and returns the Newton update from a single
        :meth:`_newton_update` call.
        """

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
                            verbosity: bool,
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
        """Advance the flamelet through ``maxsteps`` implicit time steps.

        Performs ``maxsteps`` Crank--Nicolson steps via
        :meth:`flamelet_time_step`, updating the temperature guess
        after each step.  Prints the per-step timing and the maximum
        temperature.

        Returns
        -------
        tuple
            ``(state, temperature)`` at the end of the time march.
        """
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
            if verbosity:
                print(f"BDF time step {step}: "
                      f"time: {t_step:.4e} s, "
                      f"T_max = {temp.max():.3f} K ")

        if verbosity:
            print(f"BDF Time march: T_max = {temp.max():.3f} K")

        return state, temp

    def warmup(self, warmup_fn_name: str, *args):
        """Trigger JIT compilation of one of the solver's traced methods.

        Calling this once at start-up with representative arguments
        keeps the first user-visible solve from paying the JIT cost.

        Parameters
        ----------
        warmup_fn_name : {"flamelet_newton_step", "flamelet_time_step"}
            Name of the method to warm up.
        *args
            Arguments forwarded to the chosen method.

        Raises
        ------
        ValueError
            If ``warmup_fn_name`` is not one of the supported names.
        """
        if warmup_fn_name == "flamelet_newton_step":
            return self.flamelet_newton_step(
                *args
            )
        elif warmup_fn_name == "flamelet_time_step":
            return self.flamelet_time_step(
                *args
            )
        else:
            raise ValueError(f"Unknown warmup function name {warmup_fn_name}")

    def solve(self,
              verbosity: bool,
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
        """Solve the steady flamelet equations with a hybrid strategy.

        First (optionally) attempts a pure Newton iteration on the
        steady residual.  If that fails -- or if ``try_newton`` is
        ``False`` -- the solver falls back to up to ``max_attempts``
        sequences of ``bdf_maxsteps`` Crank--Nicolson steps followed
        by a tight Newton solve.  As soon as Newton converges, the
        corresponding ``(state, temperature)`` pair is returned.

        Parameters
        ----------
        newton_maxiter, newton_tol
            Iteration cap and tolerance for the steady Newton solves.
        bdf_newton_maxiter, bdf_newton_tol
            Iteration cap and tolerance for the Newton solves inside
            each implicit time step.
        bdf_time_step, bdf_maxsteps
            Time-step size and number of time steps per BDF attempt.
        try_newton : bool
            If ``True``, try a direct steady Newton iteration before
            resorting to time marching.
        max_attempts : int
            Maximum number of BDF + Newton restart cycles.
        diss_rate, viscous_diss, temp_guess, pressure, h_ox, h_fu
            See :meth:`FlameletEquations.rhs`.
        state_guess : FlameletState
            Initial guess for the flamelet state.

        Returns
        -------
        tuple
            ``(state, temperature)``.  If no attempt converges, the
            last available pair is returned and a failure message is
            printed.
        """
        if try_newton:
            state, newton_it, newton_err, success = self._newton_loop_py(
                verbosity,
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
                if verbosity:
                    print(f"Converged after {newton_it} Newton iterations with"
                          f" error {newton_err:.4e}")

                temp = self.gov_eqns.pyro_gas.get_temperature_from_enthalpy(
                    state.enthalpy,
                    state.mass_fractions,
                    temp_guess
                )
                return state, temp

        state = state_guess
        temp = temp_guess
        for i in range(max_attempts):
            if verbosity:
                print(f"Attempt {i} of BDF time marching")

            state, temp = self.flamelet_time_march(
                verbosity,
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
                verbosity,
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
                state_t.enthalpy,
                state_t.mass_fractions,
                temp_guess
            )
            if success:
                if verbosity:
                    print(f"Converged after {i+1} BDF attempts with "
                          f"error {newton_err:.4e}")

                return state_t, temp_t

        if verbosity:
            print("Flamelet solution did not converge")

        return state, temp
