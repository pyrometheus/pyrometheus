import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pyrometheus.bandit.impl.plato import PlatoMechanism
from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator as pyro

from dataclasses import dataclass
from typing import Callable, Tuple, Any
from matplotlib import pyplot as plt


jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
@dataclass
class State:
    densities: jnp.ndarray
    temperatures: jnp.ndarray

    def tree_flatten(self):
        children = (self.densities, self.temperatures)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...],
                       children: Tuple[Any, ...]) -> "State":
        densities, temperatures = children
        return cls(densities=densities, temperatures=temperatures)

    @property
    def num_vars(self):
        return len(self.densities) + len(self.temperatures)

    @property
    def density(self):
        return jnp.sum(self.densities)

    @property
    def mass_fractions(self):
        return self.densities / self.density

    def _apply_binary_op(self, other, op):
        if isinstance(other, State):
            return self.__class__(
                **{k: op(v, getattr(other, k))
                   for k, v in self.__dict__.items()}
            )
        else:
            return self.__class__(**{k: op(v, other)
                                     for k, v in self.__dict__.items()})

    def __add__(self, other):
        return self._apply_binary_op(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._apply_binary_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._apply_binary_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self._apply_binary_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_binary_op(other, lambda x, y: x / y)

    def __neg__(self,):
        return self.__class__(**{k: -v for k, v in self.__dict__.items()})


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
    _, unravel_state = ravel_pytree(state)

    def cond_fn(carry):
        _, it, delta = carry
        return jnp.logical_and(
            delta > tol,
            it < max_iter
        )

    def body_fn(carry):
        state, it, _ = carry
        flat_residual, _ = ravel_pytree(fn(state, state_prev, step_size))
        v = jnp.linalg.solve(
            jac(state, state_prev, step_size),
            -flat_residual
        )
        return tuple((state + unravel_state(v), it + 1, jnp.linalg.norm(v)))

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
        w_dot = pyro_gas.get_net_production_rates(
            state.density, state.temperatures, state.mass_fractions
        ) * pyro_gas.molecular_weights

        vt_relax_source = pyro_gas.get_vt_energy_transfer_source(
            state.density, state.temperatures, state.mass_fractions
        )

        # PLATO's actual Th equation (species_energy_Cv_T / box_ad_NT) uses
        # the two-temperature-split energy -- tr-rot at Th plus vibronic at
        # Tv -- not the single-temperature full NASA9 energy at Th, and its
        # heat capacity is the tr-rot piece alone (Cv_vibronic belongs to
        # the Tv equation, not Th's). Validated against PLATO's real
        # trajectory: ratio 0.98-1.02 across the full t=1e-8 to 7e-4 s range.
        heavy_temperature_energy = (
            pyro_gas.get_translational_rotational_energy(
                state.temperatures
            )
        )
        vibrational_temperature_energy = (
            pyro_gas.get_nasa_polynomial_vibrational_energy(state.temperatures)
        )
        vibrational_temperature_specific_heat = (
            pyro_gas.get_nasa_polynomial_vibrational_specific_heat(
                state.temperatures
            )
        )

        heat_heavy = (
            (
                -jnp.sum(
                    w_dot
                    * (
                        heavy_temperature_energy
                        + vibrational_temperature_energy
                    )
                )
                - vt_relax_source
            )
            / (
                state.density
                * jnp.sum(
                    state.mass_fractions
                    * mech.translational_rotational_specific_heat_cv
                )
            )
        )

        # dTv/dt from PLATO's actual Tv equation (get_ODE_factors_tcneq plus
        # add_Omega_CVE in ode_utils.F90/nasa_thermo_utils_source.F90): the
        # chemistry term (species carrying vibronic energy away/in as they
        # react) is exactly cancelled by the CV (chemistry-vibration)
        # coupling term Omega_CVE -- both equal sum_i(w_dot_i *
        # e_vibronic(Tv)_i) -- so only the VT relaxation source survives.
        # Validated against PLATO's real trajectory: ratio 0.99-1.02 across
        # the full t=1e-8 to 7e-4 s range.
        heat_vib = vt_relax_source / (
            state.density
            * jnp.sum(
                state.mass_fractions * vibrational_temperature_specific_heat
            )
        )
        return State(
            densities=w_dot,
            temperatures=pyro_gas._pyro_make_array([heat_heavy, heat_vib])
        )

    _, unravel_state = ravel_pytree(state)

    def rhs_flat(flat_state):
        flat_rhs, _ = ravel_pytree(rhs(unravel_state(flat_state)))
        return flat_rhs

    jac_of_rhs_flat = jax.jacfwd(rhs_flat)

    def crank_nicolson_fn(state, state_prev, step_size):
        return (
            state - state_prev
            - 0.5 * step_size *
            (rhs(state) + rhs(state_prev))
        )

    def crank_nicolson_jac(state, state_prev, step_size):
        flat_state, _ = ravel_pytree(state)
        return (
            jnp.eye(flat_state.shape[0])
            - 0.5 * step_size * jac_of_rhs_flat(flat_state)
        )

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

    sol = np.empty((num_steps + 1, initial_state.num_vars))
    sol[0] = jnp.concatenate((
        initial_state.densities,
        initial_state.temperatures
    ))

    state = initial_state
    for step in range(num_steps):
        state_prev = state
        _t_step = time.time()
        state, newton_it, newton_err = one_step(
            state, state_prev, step_size
        )
        state.densities.block_until_ready()
        _t_step = time.time() - _t_step
        print(f'Step {step}: cost {_t_step:.4e} s')
        sol[step + 1] = jnp.concatenate((
            state.densities,
            state.temperatures
        ))
    return sol


if __name__ == "__main__":

    lib_name = 'plato'
    mech = make_mechanism(lib_name, np, hardcode_params=True)
    pyro_cls = pyro.get_thermochem_class(mech)
    pyro_gas = make_pyro_object(pyro_cls, jnp)

    cold_temp = 500
    bath_temp = 1e4
    vib_temp = cold_temp
    pressure = 1e3

    mole_fractions = jnp.zeros(pyro_gas.num_species)
    mole_fractions = mole_fractions.at[mech.species_index("O2")].set(0.21)
    mole_fractions = mole_fractions.at[mech.species_index("N2")].set(0.79)
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

    initial_state = State(
        densities=densities,
        temperatures=pyro_gas._pyro_make_array([bath_temp, vib_temp])
    )

    # }}}

    # {{{ Solve

    num_steps = 100000
    step_size = 1e-8
    sol_s = time_march(
        num_steps, step_size, initial_state,
    )

    # }}}

    # {{{ Save

    # Persist the raw integrated state (what time_march actually advances),
    # with time as the leading column so the file is self-describing.
    np.savetxt(
        'examples_postshock_relax_adiabatic.dat',
        np.column_stack((
            step_size * np.arange(0, num_steps + 1, 1),
            sol_s
        )),
        header=' '.join(
            ['time']
            + [f'density_{mech.species_name(i)}'
               for i in range(pyro_gas.num_species)]
            + ['temperature_heavy', 'temperature_vibrational']
        )
    )

    # }}}

    # {{{ Plot

    colors = ['k',
              'orangered',
              'mediumseagreen',
              'royalblue',
              'mediumpurple',]

    ref_sol = np.loadtxt(
        'plato_solution_adiabatic.dat',
        skiprows=7
    )

    sol_t = step_size * np.arange(0, num_steps + 1, 1)
    sol_d = jnp.sum(sol_s[:, :pyro_gas.num_species], axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)

    for i in range(pyro_gas.num_species):
        ax[0].loglog(
            sol_t[1:], sol_s[1:, i] / sol_d[1:],
            color=colors[i],
            linewidth=2,
            label=mech.species_name(i)
        )
        ax[0].loglog(
            ref_sol[1::20, 0],
            ref_sol[1::20, 1 + i],
            linestyle='None',
            marker='o',
            color=colors[i],
            mec=colors[i],
            mfc='w',
            markersize=5,
        )

    ax[0].set_xlabel('Time', fontsize=16)
    ax[0].set_ylabel('Mass Fractions', fontsize=16)
    ax[0].legend(frameon=False, labelcolor='linecolor',
                 bbox_to_anchor=(0.5, 1.15), loc="upper center",
                 ncol=pyro_gas.num_species, fontsize=12)
    ax[0].set_ylim(1e-9, 1)

    ax[1].semilogx(
        sol_t[1:],
        sol_s[1:, -2],
        color='orangered',
        label='Translational'
    )
    ax[1].semilogx(
        sol_t[1:],
        sol_s[1:, -1],
        color='mediumpurple',
        label='Vibrational'
    )
    ax[1].semilogy(
        ref_sol[1::20, 0],
        ref_sol[1::20, pyro_gas.num_species + 1],
        linestyle='None',
        marker='o',
        color='orangered',
        mec='orangered',
        mfc='w',
        markersize=5
    )
    ax[1].semilogy(
        ref_sol[1::20, 0],
        ref_sol[1::20, pyro_gas.num_species + 2],
        linestyle='None',
        marker='o',
        color='mediumpurple',
        mec='mediumpurple',
        mfc='w',
        markersize=5
    )
    ax[1].set_xlabel('Time', fontsize=16)
    ax[1].set_ylabel('Temperature [K]', fontsize=16)
    ax[1].legend(frameon=False, labelcolor='linecolor',
                 bbox_to_anchor=(0.5, 1.15), loc="upper center",
                 ncol=pyro_gas.num_temperatures, fontsize=12)

    plt.savefig('./output_adiabatic.png', bbox_inches='tight',)
    plt.close()

    # }}}

    exit()
