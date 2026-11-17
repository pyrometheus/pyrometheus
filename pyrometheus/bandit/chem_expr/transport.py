import numpy as np
import pymbolic.primitives as p
from dataclasses import dataclass, field, InitVar
from typing import List


# {{{ Constants

_boltzmann = 1.380649e-23
_avogadro = 6.02214076e23
_elementary_charge = 1.602176634e-19
_universal_gas_constant = _boltzmann * _avogadro
_square_angstrom = 1e-20

exp = p.Variable("exp")
log = p.Variable("log")
sqrt = p.Variable("sqrt")

# Chapman-Enskog first-approximation prefactors, with molar masses in
# kg/mol: mu = viscosity_factor sqrt(T M) / Q22 and
# 1/(n D) = diffusivity_factor Q11 sqrt(mu_ij) / sqrt(T).
viscosity_factor = 5 / 16 * np.sqrt(np.pi * _boltzmann / _avogadro)
diffusivity_factor = (16 / 3) / np.sqrt(2 * np.pi * _universal_gas_constant)
conductivity_factor = 15 / 4

# }}}


# {{{ Collision integrals

@dataclass
class LaricchiutaParameters:
    """Phenomenological-potential fit of Laricchiuta et al. (2007) for one
    pair of collision partners.

    .. attribute:: well_depth

        Potential well depth phi0 [eV].

    .. attribute:: sigma_squared

        Squared reference diameter [Angstrom^2].

    .. attribute:: coefficients

        Fit coefficients of shape ``(4, 7)``, one row per integral
        sigma^2 Omega(l,s) with (l,s) = (1,1), (1,2), (1,3), (2,2).
    """
    well_depth: float
    sigma_squared: float
    coefficients: np.ndarray


def laricchiuta_log_omega_expr(coefficients, log_reduced_temperature):
    """:returns: The logarithm of the reduced collision integral
        Omega(l,s)* from Laricchiuta et al. (2007), Eq. (15).
    """
    a = coefficients
    x = log_reduced_temperature
    rising = exp((x - a[2]) / a[3])
    falling = exp((a[2] - x) / a[3])
    upper = exp((x - a[5]) / a[6])
    lower = exp((a[5] - x) / a[6])
    return (
        (a[0] + a[1] * x) * (rising / (rising + falling))
        + a[4] * upper / (upper + lower)
    )


@dataclass
class PairCollisionIntegrals:
    """Collision integrals Q(l,s) = pi sigma^2 Omega(l,s)* [m^2] and
    their ratios for one pair of collision partners, as expressions in
    *temperature*.
    """
    parameters: InitVar[LaricchiutaParameters]
    temperature: InitVar[p.ExpressionNode]
    q_11: p.ExpressionNode = field(init=False)
    q_22: p.ExpressionNode = field(init=False)
    a_star: p.ExpressionNode = field(init=False)
    b_star: p.ExpressionNode = field(init=False)
    c_star: p.ExpressionNode = field(init=False)

    def __post_init__(self, parameters, temperature):
        log_reduced_temperature = log(
            (_boltzmann * temperature / _elementary_charge)
            / parameters.well_depth
        )
        omega_11, omega_12, omega_13, omega_22 = (
            parameters.sigma_squared
            * exp(laricchiuta_log_omega_expr(row, log_reduced_temperature))
            for row in parameters.coefficients
        )
        to_area = np.pi * _square_angstrom
        self.q_11 = to_area * omega_11
        self.q_22 = to_area * omega_22
        self.a_star = omega_22 / omega_11
        self.b_star = (5 * omega_12 - 4 * omega_13) / omega_11
        self.c_star = omega_12 / omega_11

# }}}


# {{{ Mixture transport

def inverse_number_density_diffusivity_expr(
        q_11, reduced_molar_mass, temperature) -> p.ExpressionNode:
    """:returns: 1/(n D_ij) [s/m^2 m^3] of a heavy pair in the
        Chapman-Enskog first approximation, with *reduced_molar_mass* in
        kg/mol.
    """
    return (
        diffusivity_factor * q_11 * np.sqrt(reduced_molar_mass)
        / sqrt(temperature)
    )


@dataclass
class MixtureTransport:
    """First-order Chapman-Enskog transport of a heavy-particle mixture,
    following the symmetric transport systems of Devoto and of Ferziger
    and Kaper.

    The matrices are expressions in ``mole_fractions[i]`` and the heavy
    temperature; the viscosity and translational conductivity follow
    from solving ``matrix x = mole_fractions`` and forming
    ``sum(mole_fractions * x)``.

    .. attribute:: inverse_number_density_diffusivities

        1/(n D_ij), shape ``(N, N)``.

    .. attribute:: species_viscosities
    .. attribute:: viscosity_matrix
    .. attribute:: conductivity_matrix
    .. attribute:: internal_conductivities

        Internal (Eucken) conductivity, one entry per heat capacity row of
        *internal_specific_heats*.

    .. attribute:: number_density_effective_diffusivities

        n D_i of each species in the mixture, from
        D_i = (1 - X_i) / (sum_j X_j / D_ij - X_i / D_ii).
    """
    molar_masses: InitVar[np.ndarray]
    pair_collision_integrals: InitVar[List[List[PairCollisionIntegrals]]]
    temperature: InitVar[p.ExpressionNode]
    internal_specific_heats: InitVar[List[List[p.ExpressionNode]]]
    inverse_number_density_diffusivities: list = field(init=False)
    species_viscosities: list = field(init=False)
    viscosity_matrix: list = field(init=False)
    conductivity_matrix: list = field(init=False)
    internal_conductivities: list = field(init=False)
    number_density_effective_diffusivities: list = field(init=False)

    def __post_init__(self, molar_masses, pair_collision_integrals,
                      temperature, internal_specific_heats):
        num_species = len(molar_masses)
        mole_fractions = p.Variable("mole_fractions")
        species = range(num_species)

        self.inverse_number_density_diffusivities = [
            [
                inverse_number_density_diffusivity_expr(
                    pair_collision_integrals[i][j].q_11,
                    molar_masses[i] * molar_masses[j]
                    / (molar_masses[i] + molar_masses[j]),
                    temperature,
                )
                for j in species
            ]
            for i in species
        ]
        self.species_viscosities = [
            viscosity_factor * sqrt(temperature) * np.sqrt(molar_masses[i])
            / pair_collision_integrals[i][i].q_22
            for i in species
        ]
        species_conductivities = [
            conductivity_factor * self.species_viscosities[i]
            * _universal_gas_constant / molar_masses[i]
            for i in species
        ]

        viscosity_matrix = [[0] * num_species for _ in species]
        conductivity_matrix = [[0] * num_species for _ in species]
        for i in species:
            mole_fraction_squared = mole_fractions[i] * mole_fractions[i]
            viscosity_matrix[i][i] = (
                mole_fraction_squared / self.species_viscosities[i]
            )
            conductivity_matrix[i][i] = (
                mole_fraction_squared / species_conductivities[i]
            )
        for i in species:
            for j in range(i + 1, num_species):
                integrals = pair_collision_integrals[i][j]
                mass_sum = molar_masses[i] + molar_masses[j]
                mass_share_i = molar_masses[i] / mass_sum
                mass_share_j = molar_masses[j] / mass_sum
                weight = (
                    mole_fractions[i] * mole_fractions[j]
                    * self.inverse_number_density_diffusivities[i][j]
                )

                viscosity_weight = 2 * _avogadro * weight / mass_sum
                scaled_a_star = 0.6 * integrals.a_star
                viscosity_matrix[i][j] = viscosity_weight * (-1 + scaled_a_star)
                viscosity_matrix[j][i] = viscosity_matrix[i][j]
                viscosity_matrix[i][i] += viscosity_weight * (
                    1 + (molar_masses[j] / molar_masses[i]) * scaled_a_star
                )
                viscosity_matrix[j][j] += viscosity_weight * (
                    1 + (molar_masses[i] / molar_masses[j]) * scaled_a_star
                )

                conductivity_weight = weight / (25 * _boltzmann)
                shares_product = mass_share_i * mass_share_j
                conductivity_matrix[i][j] = -conductivity_weight * shares_product * (
                    55 - 16 * integrals.a_star - 12 * integrals.b_star
                )
                conductivity_matrix[j][i] = conductivity_matrix[i][j]
                cross = 16 * integrals.a_star * shares_product
                remainder = 25 - 12 * integrals.b_star
                conductivity_matrix[i][i] += conductivity_weight * (
                    30 * mass_share_i * mass_share_i + cross
                    + remainder * mass_share_j * mass_share_j
                )
                conductivity_matrix[j][j] += conductivity_weight * (
                    30 * mass_share_j * mass_share_j + cross
                    + remainder * mass_share_i * mass_share_i
                )
        self.viscosity_matrix = viscosity_matrix
        self.conductivity_matrix = conductivity_matrix

        # Generalized Eucken: lambda_int = sum_i M_i X_i Cv_int,i
        # / (N_A sum_j X_j / (n D_ij)), heavy particles only.
        diffusion_weights = [
            _avogadro * sum(
                mole_fractions[j]
                * self.inverse_number_density_diffusivities[i][j]
                for j in species
            )
            for i in species
        ]
        self.number_density_effective_diffusivities = [
            (1 - mole_fractions[i]) / (
                sum(
                    mole_fractions[j]
                    * self.inverse_number_density_diffusivities[i][j]
                    for j in species
                )
                - mole_fractions[i]
                * self.inverse_number_density_diffusivities[i][i]
            )
            for i in species
        ]
        self.internal_conductivities = [
            sum(
                molar_masses[i] * mole_fractions[i] * heat_capacities[i]
                / diffusion_weights[i]
                for i in species
            )
            for heat_capacities in internal_specific_heats
        ]

# }}}
