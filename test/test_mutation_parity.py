"""Parity of Pyrometheus-generated code against Mutation++'s own
evaluations.

Every expected value comes out of a :class:`mutationpp.Mixture`;
nothing is recomputed from the parameters bandit reads. Mutation++
works in mol and keeps the third body out of its rate coefficients,
while bandit works in kmol and folds it in, so the oracles convert
those two conventions and nothing else.

The tests are written once and run for every case in :data:`cases`.
What a case lacks (a vibrational temperature) skips; what bandit gets
wrong for it is listed in :data:`known_gaps` and runs as a strict
xfail, so that fixing a gap fails the suite until its entry goes.

Requires the MPP_DATA_DIRECTORY environment variable and an importable
``mutationpp``.
"""

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from bandit_parity import (
    language_available, languages, make_generated_gas, make_language_gas,
    mechanism_expressions, temperature_subscripts,
)

try:
    import mutationpp  # noqa: F401
except ImportError:
    mutation_available = False
else:
    mutation_available = os.environ.get("MPP_DATA_DIRECTORY") is not None

# Imported unconditionally once mutationpp is available: a failure to
# import the adapter is then a defect, not a reason to skip.
if mutation_available:
    from pyrometheus.bandit.impl.mutation import MutationMechanism

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jnp = None
else:
    jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    not mutation_available,
    reason="mutationpp not importable or MPP_DATA_DIRECTORY not set",
)

_kmol_per_mol = 1e3


# {{{ cases and known gaps

@dataclass(frozen=True)
class MutationSpec:
    mixture: str
    state_model: str
    thermo_database: str

    @property
    def label(self):
        return f"{self.mixture}-{self.state_model}-{self.thermo_database}"


# air_11 adds ions and electrons: associative and electron-impact
# ionization, charge exchange and electron-impact dissociation.
cases = [
    MutationSpec(mixture, state_model, thermo_database)
    for mixture in ["air_5", "air_11"]
    for state_model in ["ChemNonEq1T", "ChemNonEqTTv"]
    for thermo_database in ["NASA-9", "NASA-7"]
]


@dataclass(frozen=True)
class KnownGap:
    """A strict xfail for the tests of *block* whose case and parameters
    satisfy *applies*(case, params)."""
    applies: Callable[["MutationCase", dict], bool]
    block: str
    reason: str


known_gaps = []


def block(*names):
    """Tag a test with the physics blocks :data:`known_gaps` refers to."""
    def decorate(test):
        test.physics_blocks = set(names)
        return test
    return decorate

# }}}


# {{{ fixtures and oracles

@dataclass
class MutationCase:
    spec: MutationSpec
    language: str
    mech: "MutationMechanism"
    gas: object

    @property
    def oracle(self):
        return self.mech.namespace.mix

    @property
    def num_temp(self):
        return self.mech.num_temp

    @property
    def num_species(self):
        return self.mech.num_species

    def temperature(self, state):
        """Return the temperature argument for a (Th, Tv) *state*,
        truncated to the temperatures this case carries.
        """
        if self.num_temp == 1:
            return state[0]
        return np.array(state[:self.num_temp])

    def set_state(self, density, state, mass_fractions):
        """Put the oracle at partial densities and temperatures."""
        self.oracle.setState(
            list(density * np.asarray(mass_fractions)),
            list(state[:self.num_temp]), 1
        )

    @property
    def is_ionized(self):
        return "e-" in self.mech.species_names

    @property
    def specific_gas_constants(self):
        return self.mech.namespace.gas_constant / self.mech.molecular_weights


# Every test runs on the generated Python, C++ and Fortran, each checked
# against Mutation++ itself rather than against one another.
@pytest.fixture(
    scope="module",
    params=[(spec, language) for spec in cases for language in languages],
    ids=lambda param: f"{param[0].label}-{param[1]}",
)
def case(request, tmp_path_factory):
    spec, language = request.param
    reason = language_available(language)
    if reason is not None:
        pytest.skip(reason)
    mech = MutationMechanism(
        spec.mixture, state_model=spec.state_model,
        thermo_database=spec.thermo_database,
    )
    module_name = "mutation_" + "_".join(
        part.lower().replace("-", "") for part in
        (spec.mixture, spec.state_model, spec.thermo_database)
    )
    gas = make_language_gas(
        mech, language, tmp_path_factory.mktemp(module_name), module_name
    )
    return MutationCase(spec=spec, language=language, mech=mech, gas=gas)


@pytest.fixture(autouse=True)
def _mark_known_gaps(request, case):
    blocks = getattr(request.function, "physics_blocks", set())
    callspec = getattr(request.node, "callspec", None)
    params = dict(callspec.params) if callspec else {}
    for gap in known_gaps:
        if gap.block in blocks and gap.applies(case, params):
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=gap.reason)
            )


def require(case, multi_temperature=None, language=None):
    if language is not None and case.language != language:
        pytest.skip(f"{language} only")
    if multi_temperature is True and case.num_temp == 1:
        pytest.skip("needs a vibrational temperature")
    if multi_temperature is False and case.num_temp > 1:
        pytest.skip("single-temperature API")

# }}}


# {{{ states

# (Th, Tv): thermal equilibrium, vibrationally frozen (Tv < Th,
# post-shock) and vibrationally hot (Tv > Th, expansion).
equilibrium_states = [(1500.0,) * 2, (6500.0,) * 2, (12000.0,) * 2]
nonequilibrium_states = [
    (9000.0, 3000.0), (15000.0, 7000.0), (4000.0, 8000.0),
]
states = equilibrium_states + nonequilibrium_states

# Either side of the 1000 K and 6000 K breaks, never on one: there
# Mutation++ takes the upper fit and bandit the lower, and the fits
# only meet to ~1e-8.
thermo_temperatures = [300.0, 999.0, 1001.0, 3000.0, 5999.0, 6001.0, 15000.0]
densities = [1e-3, 5e-2]

# Mole-fraction shares of each species class; the 1e-12 floor keeps
# every species in every reaction.
compositions = {
    "molecular": {"molecule": 0.95, "atom": 0.05},
    "dissociated": {"molecule": 0.4, "atom": 0.6},
    "atomic": {"molecule": 0.05, "atom": 0.95},
    "ionized": {"molecule": 0.3, "atom": 0.6, "ion": 0.05},
}
composition_names = list(compositions)


def species_class(name):
    if name == "e-":
        return "electron"
    if name.endswith("+"):
        return "ion"
    return "molecule" if len(name) > 1 else "atom"


def mass_fractions_for(case, composition):
    if composition == "ionized" and not case.is_ionized:
        pytest.skip("no ions in this mixture")
    classes = [species_class(name) for name in case.mech.species_names]
    mole_fractions = np.full(case.num_species, 1e-12)
    for kind, share in compositions[composition].items():
        members = [k for k, c in enumerate(classes) if c == kind]
        mole_fractions[members] += share / len(members)
    if case.is_ionized:
        # Quasi-neutral: one electron per singly charged ion.
        ions = [k for k, c in enumerate(classes) if c == "ion"]
        mole_fractions[classes.index("electron")] = mole_fractions[ions].sum()
    weighted = mole_fractions * case.mech.molecular_weights
    return weighted / weighted.sum()

# }}}


# {{{ species metadata and constants

@block("constants")
def test_molecular_weights_match_mutation(case):
    np.testing.assert_allclose(
        case.gas.molecular_weights,
        _kmol_per_mol * np.array(case.oracle.speciesMw()), rtol=1e-15
    )


@block("constants")
def test_gas_constant_matches_mutation(case):
    assert case.gas.gas_constant == _kmol_per_mol * case.oracle.RU()


def mutation_species_names(case):
    return [
        case.oracle.speciesName(species_index)
        for species_index in range(case.num_species)
    ]


@block("constants")
def test_species_names_follow_mutation_order(case):
    if case.language == "fortran":
        pytest.skip("f2py cannot return Fortran's assumed-length names")
    assert case.gas.species_names == mutation_species_names(case)


@block("constants")
def test_species_indices_follow_mutation_order(case):
    for species_index, name in enumerate(mutation_species_names(case)):
        assert case.gas.get_species_index(name) == species_index


@block("constants")
@pytest.mark.parametrize("composition", composition_names)
def test_equation_of_state_matches_mutation(case, composition):
    mass_fractions = mass_fractions_for(case, composition)
    density, state = 2e-2, (5000.0, 5000.0)
    case.set_state(density, state, mass_fractions)
    assert case.gas.get_pressure(
        density, case.temperature(state), mass_fractions
    ) == pytest.approx(case.oracle.P(), rel=1e-14)
    assert case.gas.get_density(
        case.oracle.P(), case.temperature(state), mass_fractions
    ) == pytest.approx(density, rel=1e-14)


@block("constants")
@pytest.mark.parametrize("state", nonequilibrium_states)
def test_equation_of_state_out_of_equilibrium_matches_mutation(case, state):
    # Electrons carry their partial pressure at the electron temperature.
    require(case, multi_temperature=True)
    composition = "ionized" if case.is_ionized else "dissociated"
    mass_fractions = mass_fractions_for(case, composition)
    case.set_state(2e-2, state, mass_fractions)
    temperature = case.temperature(state)
    assert case.gas.get_pressure(
        2e-2, temperature, mass_fractions
    ) == pytest.approx(case.oracle.P(), rel=1e-14)
    assert case.gas.get_density(
        case.oracle.P(), temperature, mass_fractions
    ) == pytest.approx(2e-2, rel=1e-14)


@block("constants")
def test_temperature_subscripts_are_valid_array_indices(case):
    # C++ and Fortran index temperature as an array: Python's
    # temperature[-1] reads out of bounds there. The expressions are the
    # same for every language.
    require(case, language="python")
    assert temperature_subscripts(
        mechanism_expressions(case.mech)
    ) <= set(range(case.num_temp))

# }}}


# {{{ thermodynamics

def uniform_mass_fractions(case):
    return np.full(case.num_species, 1 / case.num_species)


def uniform_temperature(case, temperature):
    return case.temperature((temperature,) * 2)


@block("thermo")
@pytest.mark.parametrize("temperature", thermo_temperatures)
def test_species_specific_heats_match_mutation(case, temperature):
    gas = case.gas
    np.testing.assert_allclose(
        gas.get_species_specific_heats_cp_r(
            uniform_temperature(case, temperature)
        ),
        case.oracle.speciesCpOverR(temperature), rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", thermo_temperatures)
def test_species_enthalpies_match_mutation(case, temperature):
    case.set_state(1e-2, (temperature,) * 2, uniform_mass_fractions(case))
    np.testing.assert_allclose(
        case.gas.get_species_enthalpies_rt(
            uniform_temperature(case, temperature)
        ),
        case.oracle.speciesHOverRT(), rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", thermo_temperatures)
def test_species_internal_energies_match_mutation(case, temperature):
    case.set_state(1e-2, (temperature,) * 2, uniform_mass_fractions(case))
    # Mutation++ lists one block of species per energy equation, the
    # totals first.
    np.testing.assert_allclose(
        case.gas.get_species_internal_energies_rt(
            uniform_temperature(case, temperature)
        ),
        np.array(case.oracle.getEnergiesMass())[:case.num_species]
        / case.specific_gas_constants / temperature,
        rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", thermo_temperatures)
def test_species_gibbs_energies_match_mutation(case, temperature):
    np.testing.assert_allclose(
        case.gas.get_species_gibbs_rt(uniform_temperature(case, temperature)),
        np.array(case.oracle.getSTGibbsMass(temperature))
        / case.specific_gas_constants / temperature,
        rtol=1e-12, atol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", [800.0, 3000.0, 6500.0, 12000.0])
@pytest.mark.parametrize("composition", composition_names)
def test_mixture_energies_match_mutation(case, temperature, composition):
    gas = case.gas
    mass_fractions = mass_fractions_for(case, composition)
    case.set_state(1e-2, (temperature,) * 2, mass_fractions)
    temperature_arg = uniform_temperature(case, temperature)
    assert gas.get_mixture_enthalpy_mass(
        temperature_arg, mass_fractions
    ) == pytest.approx(case.oracle.mixtureHMass(), rel=1e-12)
    assert gas.get_mixture_internal_energy_mass(
        temperature_arg, mass_fractions
    ) == pytest.approx(case.oracle.mixtureEnergyMass(), rel=1e-12)


@block("thermo")
@pytest.mark.parametrize("temperature", [800.0, 3000.0, 6500.0, 12000.0])
@pytest.mark.parametrize("composition", composition_names)
def test_mixture_specific_heats_match_mutation(case, temperature, composition):
    # With two temperatures Mutation++'s frozen specific heats are the
    # heavy-particle part alone, which bandit does not generate.
    require(case, multi_temperature=False)
    gas = case.gas
    mass_fractions = mass_fractions_for(case, composition)
    case.set_state(1e-2, (temperature,), mass_fractions)
    assert gas.get_mixture_specific_heat_cp_mass(
        temperature, mass_fractions
    ) == pytest.approx(case.oracle.mixtureFrozenCpMass(), rel=1e-12)
    assert gas.get_mixture_specific_heat_cv_mass(
        temperature, mass_fractions
    ) == pytest.approx(case.oracle.mixtureFrozenCvMass(), rel=1e-12)


@block("thermo")
@pytest.mark.parametrize("temperature", [800.0, 3000.0, 6500.0, 12000.0])
@pytest.mark.parametrize("do_energy", [False, True])
def test_temperature_inversion_recovers_mutation_state(
        case, temperature, do_energy):
    require(case, multi_temperature=False)
    # NASA-7 fits stop at 6000 K; past it their enthalpy turns over
    # (cp < 0 by 12000 K) and Newton settles on the stationary point.
    fitted_range = min(
        case.mech.species_thermo_params(species_index).t_bounds[-1]
        for species_index in range(case.num_species)
    )
    if temperature > fitted_range:
        pytest.skip("enthalpy is not invertible past the fitted range")
    mass_fractions = mass_fractions_for(case, "dissociated")
    case.set_state(1e-2, (temperature,), mass_fractions)
    target = (
        case.oracle.mixtureEnergyMass() if do_energy
        else case.oracle.mixtureHMass()
    )
    assert case.gas.get_temperature(
        target, 1000.0, mass_fractions, do_energy
    ) == pytest.approx(temperature, rel=1e-9)

# }}}


# {{{ rate coefficients

def third_body_concentrations(case, concentrations):
    """:returns: The third-body concentration of every reaction, in
    kmol/m^3, or one for reactions without a third body.
    """
    mech = case.mech
    return np.array([
        sum(
            mech.third_body_efficiencies(reaction_index).get(
                species_index, 1.0
            ) * concentrations[species_index]
            for species_index in range(case.num_species)
        ) if mech.is_third_body(reaction_index) else 1.0
        for reaction_index in range(mech.num_reactions)
    ])


def molar_unit_factors(case, side):
    """:returns: The factor converting each reaction's *side* ("reactants"
    or "products") rate coefficient from mol to kmol. Mutation++ leaves
    the third body out of both species lists but not out of the order.
    """
    mech = case.mech
    return np.array([
        _kmol_per_mol ** (
            len(getattr(mech.reaction(reaction_index), side))
            + mech.is_third_body(reaction_index) - 1
        )
        for reaction_index in range(mech.num_reactions)
    ])


def kinetics_state(case, state, composition, density=5e-2):
    mass_fractions = mass_fractions_for(case, composition)
    case.set_state(density, state, mass_fractions)
    concentrations = case.gas.get_concentrations(density, mass_fractions)
    return mass_fractions, concentrations


@block("kinetics")
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_forward_rate_coefficients_match_mutation(case, state, composition):
    _, concentrations = kinetics_state(case, state, composition)
    expected = (
        np.array(case.oracle.forwardRateCoefficients())
        * molar_unit_factors(case, "reactants")
        * third_body_concentrations(case, concentrations)
    )
    np.testing.assert_allclose(
        case.gas.get_fwd_rate_coefficients(
            case.temperature(state), concentrations
        ),
        expected, rtol=1e-12
    )


def generated_reverse_rate_coefficients(case, temperature, concentrations):
    """:returns: The reverse rate coefficient of every reaction, without
    its third body, read off the generated rates of progress. With a
    reaction's reactants removed only its reverse term remains, so
    nothing cancels. A reactant that is also a product, like the
    electron in electron-impact reactions, has to stay.
    """
    mech = case.mech
    coefficients = []
    for reaction_index in range(mech.num_reactions):
        isolated = concentrations.copy()
        isolated[[
            species_index for species_index in mech.reactants(reaction_index)
            if species_index not in mech.products(reaction_index)
        ]] = 0.0
        _, product_coeffs = mech.stoichiometric_coefficients(reaction_index)
        product_term = np.prod([
            isolated[species_index] ** coefficient
            for species_index, coefficient
            in zip(mech.products(reaction_index), product_coeffs)
        ])
        rate = case.gas.get_net_rates_of_progress(temperature, isolated)
        coefficients.append(
            -rate[reaction_index] / product_term
            / third_body_concentrations(case, isolated)[reaction_index]
        )
    return np.array(coefficients)


@block("kinetics")
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_reverse_rate_coefficients_match_mutation(case, state, composition):
    _, concentrations = kinetics_state(case, state, composition)
    expected = (
        np.array(case.oracle.backwardRateCoefficients())
        * molar_unit_factors(case, "products")
    )
    # exp(ln Keq) turns the absolute rounding of ln Keq into relative
    # error, which charge exchange, with |dG/RT| in the tens, pushes
    # past 1e-12.
    np.testing.assert_allclose(
        generated_reverse_rate_coefficients(
            case, case.temperature(state), concentrations
        ),
        expected, rtol=1e-11
    )


@block("kinetics")
def test_dissociation_reverses_at_the_heavy_temperature(case):
    # Mutation++'s recombination depends on Th alone, so changing Tv
    # must leave every reverse rate coefficient it evaluates at Th where
    # it was. Reactions that reverse on the electron temperature follow
    # Tv, which stands in for it in the two-temperature model.
    require(case, multi_temperature=True)
    _, concentrations = kinetics_state(case, (9000.0, 9000.0), "dissociated")
    heavy = [
        reaction_index for reaction_index in range(case.mech.num_reactions)
        if case.oracle.reaction(reaction_index).rev_rate_coeff_temperature
        == "translational"
    ]

    def reverse_coefficients(vibrational):
        return generated_reverse_rate_coefficients(
            case, np.array([9000.0, vibrational]), concentrations
        )[heavy]

    np.testing.assert_allclose(
        reverse_coefficients(3000.0), reverse_coefficients(9000.0),
        rtol=1e-13
    )

# }}}


# {{{ rates of progress and species production rates

@block("kinetics")
@pytest.mark.parametrize("density", densities)
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_net_rates_of_progress_match_mutation(
        case, state, composition, density):
    _, concentrations = kinetics_state(case, state, composition, density)
    expected = np.array(case.oracle.netRatesOfProgress()) / _kmol_per_mol
    actual = case.gas.get_net_rates_of_progress(
        case.temperature(state), concentrations
    )
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@block("kinetics")
@pytest.mark.parametrize("density", densities)
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_net_production_rates_match_mutation(
        case, state, composition, density):
    mass_fractions, _ = kinetics_state(case, state, composition, density)
    expected = np.array(case.oracle.netProductionRates())
    actual = case.gas.get_net_production_rates(
        density, case.temperature(state), mass_fractions
    ) * case.gas.molecular_weights
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@block("kinetics")
@pytest.mark.parametrize("state", states)
def test_net_production_rates_conserve_mass(case, state):
    mass_fractions = mass_fractions_for(case, "dissociated")
    rates = case.gas.get_net_production_rates(
        5e-2, case.temperature(state), mass_fractions
    ) * case.gas.molecular_weights
    assert abs(rates.sum()) < 1e-12 * np.abs(rates).max()

# }}}


# {{{ two-temperature energy exchange

@block("energy_exchange")
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_vt_energy_transfer_matches_mutation(case, state, composition):
    # Mutation++'s OmegaVT: Landau-Teller with Millikan-White and Park's
    # high-temperature correction, harmonic-oscillator energies.
    require(case, multi_temperature=True)
    mass_fractions = mass_fractions_for(case, composition)
    case.set_state(5e-2, state, mass_fractions)
    expected = case.oracle.omega_vt()
    actual = case.gas.get_vt_energy_transfer_source(
        5e-2, case.temperature(state), mass_fractions
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@block("energy_exchange")
@pytest.mark.parametrize("temperature", thermo_temperatures)
def test_energy_split_adds_up_to_mutation_species_energies(case, temperature):
    # At one temperature the translational-rotational and vibronic parts
    # must rebuild Mutation++'s full species energy.
    require(case, multi_temperature=True)
    case.set_state(1e-2, (temperature,) * 2, uniform_mass_fractions(case))
    temperature_arg = uniform_temperature(case, temperature)
    actual = (
        case.gas.get_translational_rotational_energy(temperature_arg)
        + case.gas.get_nasa_polynomial_vibrational_energy(temperature_arg)
    )
    expected = np.array(case.oracle.getEnergiesMass())[:case.num_species]
    np.testing.assert_allclose(
        actual, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


@block("energy_exchange")
@pytest.mark.parametrize("vibrational", [800.0, 3000.0, 7000.0, 15000.0])
def test_vibronic_specific_heat_is_the_energy_slope(case, vibrational):
    require(case, multi_temperature=True)
    step = 1e-4 * vibrational

    def energy(tv):
        return np.asarray(case.gas.get_nasa_polynomial_vibrational_energy(
            np.array([9000.0, tv])
        ))

    slope = (energy(vibrational + step) - energy(vibrational - step)) / (
        2 * step
    )
    np.testing.assert_allclose(
        case.gas.get_nasa_polynomial_vibrational_specific_heat(
            np.array([9000.0, vibrational])
        ),
        slope, rtol=1e-6, atol=1e-6 * np.abs(slope).max()
    )

# }}}


# {{{ array semantics

def batched_states(case):
    rng = np.random.default_rng(seed=0)
    num_states = 16
    temperature = rng.uniform(1000.0, 15000.0, (case.num_temp, num_states))
    if case.num_temp == 1:
        temperature = temperature[0]
    mass_fractions = rng.dirichlet(np.ones(case.num_species), num_states).T
    density = rng.uniform(1e-3, 1e-1, num_states)
    return density, temperature, mass_fractions


def mutation_pointwise(case, density, temperature, mass_fractions):
    temperature = np.atleast_2d(temperature)
    rates = []
    for i in range(density.size):
        case.set_state(density[i], temperature[:, i], mass_fractions[:, i])
        rates.append(np.array(case.oracle.netProductionRates()))
    return np.stack(rates, axis=-1)


def assert_batched_production_rates(case, production):
    density, temperature, mass_fractions = batched_states(case)
    expected = mutation_pointwise(case, density, temperature, mass_fractions)
    np.testing.assert_allclose(
        np.asarray(production(density, temperature, mass_fractions))
        * case.gas.molecular_weights[:, None],
        expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@block("kinetics")
def test_batched_states_match_mutation_pointwise(case):
    # Generated C++ and Fortran evaluate one state per call.
    require(case, language="python")
    assert_batched_production_rates(case, case.gas.get_net_production_rates)


@block("kinetics")
@pytest.mark.skipif(jnp is None, reason="jax is not installed")
def test_jax_generated_code_matches_mutation(case):
    require(case, language="python")
    gas = make_generated_gas(case.mech, jnp)
    production = jax.jit(gas.get_net_production_rates)
    assert_batched_production_rates(
        case,
        lambda density, temperature, mass_fractions: production(
            jnp.asarray(density), jnp.asarray(temperature),
            jnp.asarray(mass_fractions)
        )
    )

# }}}
