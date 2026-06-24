import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd
from reactors import Reactor
from profiling import SerialTimer


def create_time_windows(initial_time, final_time, num_snapshots):
    from numpy import linspace, stack
    snap_times = linspace(
        initial_time, final_time,
        num_snapshots
    )
    return snap_times, stack((
        snap_times[:-1], snap_times[1:]
    )).T


class TimeIntegrator:

    def __init__(self, rxr: Reactor):
        self.rxr = rxr
        self.post_step = None
        self.timer = SerialTimer()

    def configure(self, config: dict, **kwargs):
        if 'post_step' in kwargs:
            self.post_step = kwargs.get('post_step')

    def rhs(self, state):
        raise NotImplementedError

    def step(self, state, *params):
        raise NotImplementedError

    def time_march(self, initial_time, final_time, step_size,
                   initial_state, *params,):

        win_size = final_time - initial_time
        num_steps = int(
            round(win_size / step_size)
        )
        state = initial_state
        self.step_size = win_size / num_steps

        for i in range(num_steps):
            my_t = self.timer.start()
            state = self.step(state, *params)
            self.timer.record('time_integ::rhs', self.timer.stop(my_t))

            if self.post_step:
                my_t = self.timer.start()
                state = self.post_step(state)
                self.timer.record(
                    'time_integ::post_step',
                    self.timer.stop(my_t)
                )

        return state


class RungeKutta(TimeIntegrator):

    def step(self, state, *params):
        k_1 = self.rxr.rhs(state)
        k_2 = self.rxr.rhs(state + 0.5 * self.step_size * k_1)
        k_3 = self.rxr.rhs(state + 0.5 * self.step_size * k_2)
        k_4 = self.rxr.rhs(state + self.step_size * k_3)
        return state + self.step_size * (k_1 + 2 * (k_2 + k_3) + k_4) / 6


class CrankNicolson(TimeIntegrator):

    def configure(self, config: dict):
        self.num_it = config['crank_nicolson']['newton_it']
        self.tol = config['crank_nicolson']['newton_tol']
        self.rhs_jac = jit(jacfwd(self.rxr.rhs))

    @partial(jit, static_argnums=(0,))
    def my_rhs(self, state, state_prev):
        return state - state_prev - 0.5 * self.step_size * (
            self.rxr.rhs(state) + self.rxr.rhs(state_prev)
        )

    @partial(jit, static_argnums=(0,))
    def my_jac(self, state, state_prev):
        return jnp.eye(state.shape[0]) - 0.5 * self.step_size * (
            self.rxr.jac(state) + self.rxr.jac(state_prev)
        )

    def newton(self, fun, jac, y, *args):

        for it in range(self.num_it):
            dy = jnp.linalg.solve(
                jac(y, *args), fun(y, *args)
            )
            y -= dy
            err = jnp.linalg.norm(dy)
            if err < self.tol:
                return y

    def step(self,  state, *params):
        state_prev = state
        return self.newton(
            self.my_rhs, self.my_jac, state, state_prev,
        )
