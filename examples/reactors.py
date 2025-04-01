import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic
)
from dataclasses import field, dataclass
from domain import Operators
from characteristics import outflow_nscbc


# {{{ AoS wrapper for autoignition simulations

@with_container_arithmetic(
    bcast_obj_array=False,
    bcast_container_types=(np.ndarray, jnp.ndarray),
    rel_comparison=False,
    eq_comparison=False,
)
@dataclass_array_container
@dataclass(frozen=True, eq=False)
class HomogeneousState:
    density: np.float64
    internal_energy: np.float64
    mass_fractions: np.ndarray

# }}}


# {{{ SoA wrapper for flow simulations

@with_container_arithmetic(
    bcast_obj_array=False,
    bcast_container_types=(np.ndarray,),
    rel_comparison=False,
    eq_comparison=False,
)
@dataclass_array_container
@dataclass(frozen=True, eq=False)
class FlameState:
    momentum: np.ndarray
    total_energy: np.ndarray
    densities: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=object)
    )

    @property
    def velocity(self,):
        return self.momentum

    @property
    def pressure(self,):
        return self.total_energy

    @property
    def mass_fractions(self,):
        return self.densities

# }}}


# {{{ Helper functions for boundary states

def get_boundary_state(b_idx, state: FlameState):
    return FlameState(
        state.momentum[b_idx],
        state.total_energy[b_idx],
        state.densities[:, b_idx]
    )


def set_boundary_state(b_idx, b_state: np.ndarray, state: FlameState,):
    state.momentum[b_idx] = b_state[0]
    state.total_energy[b_idx] = b_state[1]
    state.densities[:, b_idx] = b_state[2:]

# }}}


class Reactor:

    def __init__(self, pyro_gas):
        self.pyro_gas = pyro_gas

    def rhs(self, state):
        raise NotImplementedError

    @property
    def num_species(self):
        return self.pyro_gas.num_species


class HomogeneousReactor(Reactor):

    def set_density_and_energy(self, pressure, temperature,
                               mass_fractions):
        self.density = self.pyro_gas.get_density(
            pressure, temperature, mass_fractions
        )
        self.energy = self.pyro_gas.get_mixture_internal_energy_mass(
            temperature, mass_fractions
        )
        self.temp_guess = temperature

    def get_temperature(self, mass_fractions):
        return self.pyro_gas.get_temperature(
            self.energy, self.temp_guess,
            mass_fractions,
            do_energy=True
        )

    def rhs(self, mass_fractions,):
        temperature = self.get_temperature(mass_fractions)
        return (
            self.pyro_gas.molecular_weights *
            self.pyro_gas.get_net_production_rates(
                self.density, temperature,
                mass_fractions
            ) / self.density
        )

    jac = jacfwd(rhs, argnums=1)


class Flame(Reactor):

    def __init__(self,
                 pyro_gas,
                 op: Operators,
                 transport_model):
        self.pyro_gas = pyro_gas
        self.op = op
        if transport_model == 'le':
            self.species_mass_flux = self.species_mass_flux_consle
        elif transport_model == 'mixavg':
            self.species_mass_flux = self.species_mass_flux_mixavg
        else:
            self.species_mass_flux = self.species_mass_flux_consle

    def configure_buffer_zones(self,
                               buffer_support,
                               buffer_rate,
                               buffer_target: FlameState,
                               domain_support):
        self.buffer_support = buffer_support
        self.buffer_rate = buffer_rate
        self.buffer_target = buffer_target
        self.buffer_strength = (buffer_support * buffer_rate).ravel()
        self.domain_support = domain_support

    def set_temperature_guess(self, temp_guess):
        self.temp_guess = temp_guess

    def filter_state(self, cons_vars):
        return FlameState(
            self.op.filt(cons_vars.momentum),
            self.op.filt(cons_vars.total_energy),
            self.pyro_gas._pyro_make_array([
                self.op.filt(d) for d in cons_vars.densities
            ])
        )

    def rhs(self, cons_vars: FlameState):
        prim_vars, density, temperature = self.equation_of_state(
            cons_vars
        )
        d_finv_dx = self.inviscid_flux_divergence(
            cons_vars, prim_vars, density, temperature
        )
        d_fvis_dx = self.viscous_flux_divergence(
            cons_vars, prim_vars, density, temperature
        )
        omega = self.chemical_source_term(
            prim_vars, density, temperature
        )
        return omega - d_fvis_dx - d_finv_dx

    def equation_of_state(self, cons_vars: FlameState) -> FlameState:
        """
        Evaluate primitive variables from conserved variables.
        """
        density = self.pyro_gas.usr_np.sum(
            cons_vars.densities, axis=0
        )
        mass_fractions = cons_vars.densities / density
        velocity = cons_vars.momentum / density
        energy = (
            cons_vars.total_energy / density -
            0.5 * velocity ** 2
        )

        temperature = self.pyro_gas.get_temperature(
            energy, self.temp_guess, mass_fractions,
            do_energy=True
        )
        pressure = self.pyro_gas.get_pressure(
            density, temperature, mass_fractions
        )
        return FlameState(
            velocity, pressure, mass_fractions
        ), density, temperature

    def chemical_source_term(self,
                             prim_vars: FlameState,
                             density, temperature):
        return FlameState(
            self.pyro_gas.usr_np.zeros_like(prim_vars.momentum),
            self.pyro_gas.usr_np.zeros_like(prim_vars.total_energy),
            (
                self.pyro_gas.molecular_weights *
                self.pyro_gas.get_net_production_rates(
                    density, temperature,
                    prim_vars.mass_fractions
                )
            )
        )

    def inviscid_flux_divergence(self,
                                 cons_vars: FlameState,
                                 prim_vars: FlameState,
                                 density, temperature):
        """
        Compute the divergence of the inviscid flux.
        """
        # Flux divergence
        df_dx = FlameState(
            self.op.d_dx(
                (cons_vars.momentum**2 / density) + prim_vars.pressure
            ),
            self.op.d_dx(
                prim_vars.velocity * (
                    cons_vars.total_energy + prim_vars.pressure
                )
            ),
            self.pyro_gas._pyro_make_array([
                self.op.d_dx(prim_vars.velocity * d)
                for d in cons_vars.densities
            ])
        )

        # Boundary conditions
        idx = 0
        normal = -1
        df_dx_l = outflow_nscbc(
            get_boundary_state(idx, df_dx),
            get_boundary_state(idx, cons_vars),
            get_boundary_state(idx, prim_vars),
            density[idx], temperature[idx],
            normal, self.pyro_gas,
        )
        set_boundary_state(idx, df_dx_l, df_dx)

        idx = -1
        normal = 1
        df_dx_r = outflow_nscbc(
            get_boundary_state(idx, df_dx),
            get_boundary_state(idx, cons_vars),
            get_boundary_state(idx, prim_vars),
            density[idx], temperature[idx],
            normal, self.pyro_gas,
        )
        set_boundary_state(idx, df_dx_r, df_dx)

        return df_dx

    def viscous_flux_divergence(self,
                                cons_vars: FlameState,
                                prim_vars: FlameState,
                                density, temperature):
        """
        Compute the divergence of the viscous flux.
        """
        # Velocity & temperature gradients
        du_dx = self.op.d_dx(prim_vars.velocity)
        dt_dx = self.op.d_dx(temperature)

        # Transport properties
        visc = self.pyro_gas.get_mixture_viscosity_mixavg(
            temperature, prim_vars.mass_fractions
        )
        cond = self.pyro_gas.get_mixture_thermal_conductivity_mixavg(
            temperature, prim_vars.mass_fractions
        )

        mass_flux, heat_flux_m = self.species_mass_flux(
            prim_vars, density, temperature, cond
        )

        # Other fluxes
        viscous_stress = (4/3) * visc * du_dx
        viscous_diss = prim_vars.velocity * viscous_stress
        heat_flux = -heat_flux_m + cond * dt_dx + viscous_diss
        # Return flux divergence
        return FlameState(
            self.op.d_dx(-viscous_stress),
            (
                self.op.d_dx(-heat_flux)
            ),
            self.pyro_gas._pyro_make_array([
                self.op.d_dx(d) for d in mass_flux
            ])
        )

    def species_mass_flux_consle(self,
                                 prim_vars,
                                 density,
                                 temperature,
                                 cond):
        cp_mix = self.pyro_gas.get_mixture_specific_heat_cp_mass(
            temperature, prim_vars.mass_fractions
        )
        diff = cond / (density * cp_mix)
        return np.array([
            -density * diff * self.op.d_dx(y)
            for y in prim_vars.mass_fractions
        ]), np.zeros_like(density)

    def species_mass_flux_mixavg(self,
                                 prim_vars,
                                 density,
                                 temperature,
                                 cond):
        # Species enthalpies
        enthalpies = (
            self.pyro_gas.inv_molecular_weights *
            self.pyro_gas.get_species_enthalpies_rt(
                temperature
            )
        ) * self.pyro_gas.gas_constant * temperature

        # Species mole fractions
        mol_weight = self.pyro_gas.get_mix_molecular_weight(
            prim_vars.mass_fractions
        )
        mole_frac = self.pyro_gas.get_mole_fractions(
            mol_weight, prim_vars.mass_fractions
        )

        # Diffusivities
        diff = self.pyro_gas.get_species_mass_diffusivities_mixavg(
            prim_vars.pressure, temperature, prim_vars.mass_fractions
        )

        # Fluxes
        mixavg_flux = (
            -(density * self.pyro_gas.molecular_weights / mol_weight) *
            np.array([
                d * self.op.d_dx(x)
                for d, x in zip(diff, mole_frac)
            ], dtype=np.float64)
        )
        corr_flux = -prim_vars.mass_fractions * self.pyro_gas.usr_np.sum(
            mixavg_flux, axis=0
        )
        mass_flux = mixavg_flux + corr_flux

        heat_flux = self.op.usr_np.sum(
            enthalpies * mass_flux,
            axis=0
        )
        return mass_flux, heat_flux

    def buffer_zone(self, state):
        return FlameState(
            self.buffer_strength * (
                state.momentum - self.buffer_target.momentum
            ),
            self.buffer_strength * (
                state.total_energy - self.buffer_target.total_energy
            ),
            self.buffer_strength * (
                state.densities - self.buffer_target.densities
            )
        )
