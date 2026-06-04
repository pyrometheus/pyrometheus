import jax
import jax.numpy as jnp
from typing import Tuple
from pyrometheus.flamelets.state import (
    FlameletState, _state_to_array, _array_to_state
)


# Assuming the Laplacian has Dirichlet stencils at the boundaries,
# then we need to set the dissipation rate to 2*dx^2 at i = 0, N-1.
# The reason for this choice is the form of the equations that
# this class implements: 0.5 * chi * Lap(state)/dx**2 + S = 0
# where Lap = D1 * D1, with D1 the stencil matrix of first-order
# finite difference approximation.
# if self.diss_rate[0]:
#     self.diss_rate = self.diss_rate.at[0].set(
#         2 * self.laplacian.domain.jac[0]**2
#     )
# if self.diss_rate[-1]:
#     self.diss_rate = self.diss_rate.at[-1].set(
#         2 * self.laplacian.domain.jac[0]**2
#     )
class FlameletEquations:

    def __init__(self,
                 pyro_gas,
                 laplacian,
                 y_ox,
                 y_fu,):
        self.pyro_gas = pyro_gas
        self.laplacian = laplacian
        self.mol_wts = self.pyro_gas.molecular_weights  # [:, None]
        self.laplacian.assemble_block_form(self.pyro_gas.num_species + 1)

        # Create boundary conditions
        self.mass_frac_bc = jnp.hstack((
            y_ox.reshape(-1, 1),
            jnp.zeros((
                self.pyro_gas.num_species,
                self.laplacian.domain.num_x-2
            )),
            y_fu.reshape(-1, 1)
        ))

        self.interior_mask = FlameletState(
            enthalpy=jnp.hstack((
                jnp.zeros(1),
                jnp.ones(self.laplacian.domain.num_x - 2),
                jnp.zeros(1)
            )),
            mass_fractions=jnp.hstack((
                jnp.zeros((self.pyro_gas.num_species, 1)),
                jnp.ones((
                    self.pyro_gas.num_species,
                    self.laplacian.domain.num_x - 2
                )),
                jnp.zeros((self.pyro_gas.num_species, 1))
            ))
        )

        # Source terms Jacobian and gradient wrt pressure
        self.vmap_jacobian = jax.vmap(
            jax.jacfwd(
                self.local_source_terms, argnums=0
            ),
            in_axes=(1, 0, 0, None)
        )
        self.source_gradient_wrt_pressure = jax.vmap(
            jax.jacfwd(
                self.local_source_terms, argnums=-1,
            ),
            in_axes=(1, 0, 0, None)
        )

        # Compressible EOS routines for thermodynamic consistency
        self.compressible_eos_rt = jax.vmap(
            self.local_compressible_eos_rt,
            in_axes=(1, 0, None)
        )
        self.compressible_eos_rt_jacobian = jax.vmap(
            jax.jacfwd(
                self.local_compressible_eos_rt,
                argnums=0
            ),
            in_axes=(1, 0, None)
        )

    def set_dissipation_rate(self, diss_rate: jnp.ndarray):
        self.diss_rate = diss_rate

    def set_viscous_dissipation(self, viscous_diss: jnp.ndarray):
        self.viscous_diss = viscous_diss

    def rhs(self,
            state: FlameletState,
            diss_rate: jnp.ndarray,
            viscous_diss: jnp.ndarray,
            temp_guess: jnp.ndarray,
            pressure: jnp.float64,
            h_ox: jnp.float64,
            h_fu: jnp.float64,):
        return (
            0.5 * diss_rate * self.laplacian(state)
            + self.source_terms(
                state,
                viscous_diss,
                temp_guess,
                pressure,
            ) * self.interior_mask
            - self.boundary_conditions(
                h_ox,
                h_fu
            )
        )

    def jac(self,
            state: FlameletState,
            diss_rate: jnp.ndarray,
            viscous_diss: jnp.ndarray,
            temp_guess: jnp.ndarray,
            pressure: jnp.float64,) -> Tuple[jnp.ndarray, ...]:

        num_v = self.pyro_gas.num_species + 1
        zeros_nv = jnp.zeros((1, num_v, num_v))
        source_jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss,
            temp_guess,
            pressure
        )
        source_blocks = jnp.concatenate((
            zeros_nv, source_jacobian[1:-1], zeros_nv
        ))
        central_blocks = (
            jnp.einsum(
                "i,ijk->ijk",
                0.5 * diss_rate,
                self.laplacian.central_blocks
            ) +
            source_blocks
        )
        lower_blocks = jnp.einsum(
            "i,ijk->ijk",
            0.5 * diss_rate[1:],
            self.laplacian.lower_blocks
        )
        upper_blocks = jnp.einsum(
            "i,ijk->ijk",
            0.5 * diss_rate[:-1],
            self.laplacian.upper_blocks
        )
        return (lower_blocks, central_blocks, upper_blocks)

    def equation_of_state(self,
                          state: FlameletState,
                          temp_guess: jnp.ndarray,
                          pressure: jnp.float64,):
        temperature = self.pyro_gas.get_temperature_from_enthalpy(
            state.enthalpy,
            state.mass_fractions,
            temp_guess
        )
        density = self.pyro_gas.get_density(
            pressure, temperature, state.mass_fractions
        )
        return density, temperature

    def source_terms(self,
                     state: FlameletState,
                     viscous_diss: jnp.ndarray,
                     temp_guess: jnp.ndarray,
                     pressure: jnp.float64,):
        density, temperature = self.equation_of_state(
            state,
            temp_guess,
            pressure,
        )
        w_dot = self.pyro_gas.get_net_production_rates(
            density,
            temperature,
            state.mass_fractions
        )
        m_dot = (
            jnp.expand_dims(self.mol_wts, w_dot.ndim-1)
            * w_dot
        ).squeeze()

        return FlameletState(
            enthalpy=viscous_diss / density,
            mass_fractions=m_dot / density
        )

    def local_source_terms(self,
                           local_state_as_array: jnp.ndarray,
                           viscous_diss: jnp.ndarray,
                           temp_guess: jnp.ndarray,
                           pressure: jnp.float64) -> jnp.ndarray:
        local_state = FlameletState(
            enthalpy=local_state_as_array[0],
            mass_fractions=local_state_as_array[1:]
        )
        s_dot = self.source_terms(
            state=local_state,
            viscous_diss=viscous_diss,
            temp_guess=temp_guess,
            pressure=pressure
        )
        return jnp.concatenate(
            (jnp.atleast_1d(s_dot.enthalpy),
             s_dot.mass_fractions.squeeze()),
            axis=0
        )

    def boundary_conditions(self,
                            h_ox: jnp.float64,
                            h_fu: jnp.float64):

        return FlameletState(
            enthalpy=jnp.hstack((
                h_ox,
                jnp.zeros(self.laplacian.domain.num_x-2),
                h_fu
            )),
            mass_fractions=self.mass_frac_bc
        )

    def jacobian_action(self,
                        v: FlameletState,
                        state: FlameletState,
                        diss_rate: jnp.ndarray,
                        viscous_diss: jnp.ndarray,
                        temp_guess: jnp.ndarray,
                        pressure: jnp.float64,):
        jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss, temp_guess, pressure
        )
        jvp = jnp.einsum(
            "ijk,ki->ji",
            jacobian,  # (nx, nv, nv)
            _state_to_array(v)  # (nv, nx)
        )  # (nv, nx)
        return (
            0.5 * diss_rate * self.laplacian(v)
            + _array_to_state(jvp)
        )

    def adjoint_operator(self,
                         state: FlameletState,
                         diss_rate: jnp.ndarray,
                         viscous_diss: jnp.ndarray,
                         temp_guess: jnp.ndarray,
                         pressure: jnp.float64,) -> Tuple[jnp.ndarray, ...]:
        num_v = self.pyro_gas.num_species + 1
        zeros_nv = jnp.zeros((1, num_v, num_v))
        source_jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss,
            temp_guess,
            pressure
        )
        source_blocks = jnp.concatenate((
            zeros_nv,
            jnp.transpose(
                2 * source_jacobian[1:-1] / diss_rate[1:-1, None, None],
                (0, 2, 1)
            ),
            zeros_nv
        ))
        # Central blocks: need to remove 1/dx**2 factor from the BCs
        # This factor is stored in the first and last entry of the dissipation rate
        central_blocks = (
            jnp.concatenate((
                jnp.expand_dims(
                    0.5 * diss_rate[0] * self.laplacian.central_blocks[0],
                    axis=0
                ),
                self.laplacian.central_blocks[1:-1],
                jnp.expand_dims(
                    0.5 * diss_rate[-1] * self.laplacian.central_blocks[-1],
                    axis=0
                )
            ))
            + source_blocks
        )
        lower_blocks = self.laplacian.lower_blocks
        upper_blocks = self.laplacian.upper_blocks
        return (lower_blocks, central_blocks, upper_blocks)

    def local_compressible_eos_rt(self,
                                  local_state_as_array: jnp.ndarray,
                                  local_temp_guess: jnp.float64,
                                  pressure: jnp.float64):
        w_mix = self.pyro_gas.get_mixture_molecular_weight(
            local_state_as_array[1:]
        )
        temperature = self.pyro_gas.get_temperature_from_enthalpy(
            local_state_as_array[0], local_state_as_array[1:], local_temp_guess
        )
        return self.pyro_gas.gas_constant * temperature / w_mix
