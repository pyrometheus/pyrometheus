import time
import jax
import jax.numpy as jnp
import numpy as np

from pyrometheus.bandit.impl.plato import PlatoMechanism
from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator as pyro

from typing import Callable
from matplotlib import pyplot as plt


def make_mechanism(lib_name, pyro_np, hardcode_params=True):
    return PlatoMechanism(
        mixture='air5',
        reaction_set='air5',
        transfer='TTv',
        plato_db_path=os.environ.get('PLATO_DB'),
        pyro_np=pyro_np,
        hardcode_params=hardcode_params
    )


def make_pyro_object(pyro_cls, pyro_np):

    if pyro_np == np:
        class PyroNumPy(pyro_cls):
            def _pyro_make_array(self, res_list):
                return np.stack(res_list)

        return PyroNumPy(pyro_np)

    elif pyro_np == jnp:
        class PyroJAX(pyro_cls):

            def _pyro_make_array(self, res_list):
                array = pyro_np.empty_like(
                    pyro_np.array(res_list)
                )
                for idx in range(len(res_list)):
                    array = array.at[idx].set(res_list[idx])

                return array

        return PyroJAX(pyro_np)

    else:
        raise ValueError(f'This example does not support {pyro_np}')


def _newton_loop(fn: Callable,
                 jac: Callable,
                 state: jnp.ndarray,
                 state_prev: jnp.ndarray,
                 step_size: jnp.float64):

    tol = 1e-8
    max_iter = 40

    def cond_fn(carry):
        _, it, delta = carry
        return jnp.logical_and(
            delta > tol,
            it < max_iter
        )

    def body_fn(carry):
        state, it, _ = carry
        v = jnp.linalg.solve(
            jac(state, state_prev, step_size),
            -fn(state, state_prev, step_size)
        )
        return tuple((state + v, it + 1, jnp.linalg.norm(v)))

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


@jax.jit
def one_step(state, state_prev, step_size):

    def rhs(state):
        density = jnp.sum(state)
        mass_fractions = state / density
        w_dot = pyro_gas.get_net_production_rates(
            density, temperature, mass_fractions
        )
        return w_dot * pyro_gas.molecular_weights

    jac = jax.jacfwd(rhs)

    def crank_nicolson_fn(state, state_prev, step_size):
        return (
            state - state_prev
            - 0.5 * step_size *
            (rhs(state) + rhs(state_prev))
        )

    def crank_nicolson_jac(state, state_prev, step_size):
        return jnp.eye(pyro_gas.num_species) - 0.5 * step_size * jac(state)

    return _newton_loop(
        crank_nicolson_fn,
        crank_nicolson_jac,
        state,
        state_prev,
        step_size
    )


def time_march(num_steps: int,
               step_size: jnp.float64,
               initial_state: jnp.ndarray,):

    sol = np.empty((num_steps + 1, pyro_gas.num_species))
    sol[0] = initial_state.copy()

    state = initial_state
    for step in range(num_steps):
        state_prev = state
        _t_step = time.time()
        state, newton_it, newton_err = one_step(
            state, state_prev, step_size
        )
        state.block_until_ready()
        _t_step = time.time() - _t_step
        print(f'Step {step}: cost {_t_step:.4e} s')
        sol[step + 1] = state.copy()
    return sol


if __name__ == '__main__':

    lib_name = 'plato'
    mech = make_mechanism(lib_name, np, hardcode_params=True)
    pyro_cls = pyro.get_thermochem_class(mech)
    pyro_gas = make_pyro_object(pyro_cls, jnp)

    # {{{ Initial Condition

    cold_temp = 500
    bath_temp = 1e4
    temperature = jnp.array([
        bath_temp, cold_temp
    ])
    pressure = 1e3
    mole_fractions = jnp.zeros(pyro_gas.num_species)
    mole_fractions = mole_fractions.at[mech.species_index('O2')].set(0.21)
    mole_fractions = mole_fractions.at[mech.species_index('N2')].set(0.79)
    mass_fractions = pyro_gas.molecular_weights * mole_fractions / jnp.sum(
        pyro_gas.molecular_weights * mole_fractions
    )

    mix_molecular_weight = 1 / jnp.sum(
        mass_fractions / pyro_gas.molecular_weights
    )
    density = pressure * mix_molecular_weight / (
        pyro_gas.gas_constant * cold_temp
    )
    densities = density * mass_fractions

    # }}}

    # {{{ Solve

    num_steps = 200000
    step_size = 1e-5
    sol_s = time_march(
        num_steps, step_size, densities,
    )

    # }}}

    # {{{ Plot
    colors = ['k',
              'orangered',
              'mediumseagreen',
              'royalblue',
              'mediumpurple',]

    ref_sol = np.loadtxt(
        'plato_solution_isothermal.dat',
        skiprows=7
    )

    sol_t = step_size * np.arange(0, num_steps + 1, 1)
    sol_d = jnp.sum(sol_s, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.spines[['top', 'right']].set_visible(False)
    for i in range(pyro_gas.num_species):
        ax.loglog(
            sol_t[1:], sol_s[1:, i] / sol_d[1:],
            color=colors[i],
            linewidth=2,
            label=mech.species_name(i)
        )
        ax.loglog(
            ref_sol[1::20, 0],
            ref_sol[1::20, 1 + i],
            linestyle='None',
            marker='o',
            color=colors[i],
            mec=colors[i],
            mfc='w',
            markersize=5,
        )

    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Mass Fractions', fontsize=16)
    ax.legend(frameon=False, labelcolor='linecolor',
              bbox_to_anchor=(0.5, 1.15),
              loc='upper center',
              ncol=pyro_gas.num_species, fontsize=12)
    plt.savefig('./output_isothermal.png', bbox_inches='tight',)
    plt.close()

    # }}}
    exit()
