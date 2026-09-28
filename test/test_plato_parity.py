"""Parity of Pyrometheus-generated code against Plato's own evaluations.

Every expected value comes out of the Plato library (``get_source``,
``get_species_*``, ``get_ptau_vt``); nothing is recomputed from the
parameters bandit reads. Plato does not expose per-reaction rate
coefficients, so they are recovered from ``get_source`` at sparse
compositions where only one reaction's reactants are present.

Requires PLATO_DB and an importable ``plato`` package.
"""

import ctypes
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pytest

try:
    import plato  # noqa: F401
    from pyrometheus.bandit.impl.plato import PlatoMechanism
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
except ImportError:
    PlatoMechanism = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jnp = None
else:
    jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    PlatoMechanism is None or os.environ.get("PLATO_DB") is None,
    reason="PLATO_DB not set or plato package not importable",
)


# {{{ states

# (Th, Tv) pairs: thermal equilibrium, vibrationally frozen (Tv < Th,
# post-shock) and vibrationally hot (Tv > Th, expansion). Every value
# sits above Plato's rate-evaluation floor of 350 K and off the NASA-9
# interval breaks (see test_nasa_interval_breaks_follow_plato).
two_temperature_states = [
    (1500.0, 1500.0), (6500.0, 6500.0), (12000.0, 12000.0),
    (9000.0, 3000.0), (15000.0, 7000.0), (4000.0, 8000.0),
]
single_temperatures = [800.0, 3000.0, 6500.0, 12000.0]

# Crosses both NASA-9 interval breaks (1000 K, 6000 K) from either side.
nasa_temperatures = [300.0, 999.0, 1001.0, 3000.0, 5999.0, 6001.0, 18000.0]

# Mass fractions ordered as Plato's air5: N, O, N2, NO, O2.
compositions = {
    "cold_air": [1e-12, 1e-12, 0.767, 1e-12, 0.233],
    "partially_dissociated": [0.1, 0.1, 0.5, 0.1, 0.2],
    "nitric_oxide_rich": [0.02, 0.08, 0.4, 0.4, 0.1],
    "atomic": [0.7, 0.25, 0.02, 0.02, 0.01],
}
densities = [1e-3, 5e-2]

# }}}


# {{{ fixtures and oracles

def make_generated_gas(mech, pyro_np=np):
    base_class = PythonBanditCodeGenerator.get_thermochem_class(mech)

    class StackedGas(base_class):
        # The default object-array assembly cannot hold batched states.
        def _pyro_make_array(self, res_list):
            return self.pyro_np.stack([
                self.pyro_np.asarray(entry, dtype=self.pyro_np.float64)
                for entry in res_list
            ])

    return StackedGas(pyro_np)


@dataclass
class PlatoCase:
    mech: "PlatoMechanism"
    gas: object

    @property
    def oracle(self):
        return self.mech.namespace.thermochem

    def temperature(self, heavy, vibrational=None):
        if self.mech.num_temp == 1:
            return heavy
        return np.array([heavy, heavy if vibrational is None else vibrational])


def open_case(transfer):
    mech = PlatoMechanism("air5", "air5", transfer, os.environ["PLATO_DB"])
    return PlatoCase(mech=mech, gas=make_generated_gas(mech))


# Plato holds one live instance per process, so switching mixtures
# closes the open one first.
_open_cases: Dict[str, PlatoCase] = {}


def plato_case(transfer):
    if transfer not in _open_cases:
        close_cases()
        _open_cases[transfer] = open_case(transfer)
    return _open_cases[transfer]


def close_cases():
    for case in _open_cases.values():
        case.mech.finalize()
    _open_cases.clear()


@pytest.fixture(scope="module", autouse=True)
def _close_plato_after_module():
    yield
    close_cases()


@pytest.fixture
def ttv():
    return plato_case("TTv")


def plato_source_terms(oracle, partial_densities, temperature):
    """Return Plato's full source vector: species mass production rates
    [kg/(m^3 s)] followed by one energy source per temperature [W/m^3].
    ``Thermochemistry.get_production_rates`` discards the energy part.
    """
    from plato import _lib
    partial_densities = np.ascontiguousarray(partial_densities, dtype=float)
    temperature = np.ascontiguousarray(np.atleast_1d(temperature), dtype=float)
    omega = np.zeros(oracle.n_species + oracle.n_temp)
    _lib._lib.plato_get_source(
        ctypes.byref(ctypes.c_uint(0)),
        partial_densities.ctypes.data_as(_lib._P_double),
        temperature.ctypes.data_as(_lib._P_double),
        omega.ctypes.data_as(_lib._P_double),
    )
    return omega

# }}}


# {{{ species metadata and constants

def test_molecular_weights_match_plato(ttv):
    np.testing.assert_allclose(
        ttv.gas.molecular_weights, 1e3 * ttv.oracle.molar_masses, rtol=1e-15
    )


def test_specific_gas_constants_match_plato(ttv):
    np.testing.assert_allclose(
        ttv.gas.gas_constant / ttv.gas.molecular_weights,
        ttv.oracle.gas_constants, rtol=1e-14
    )


def test_energies_of_formation_match_plato(ttv):
    # Plato splits energy with its tabulated heat of formation, not the
    # NASA-9 fit evaluated at 298.15 K; the two differ by ~200 J/kg.
    np.testing.assert_allclose(
        ttv.mech.standard_energy_of_formation,
        ttv.oracle.formation_energies, rtol=1e-14, atol=1e-8
    )


def test_species_names_match_plato(ttv):
    assert ttv.gas.species_names == ttv.oracle.species_names

# }}}


# {{{ NASA-9 thermodynamics

@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_species_enthalpies_match_plato(ttv, temperature):
    gas = ttv.gas
    enthalpy = (
        gas.get_species_enthalpies_rt(ttv.temperature(temperature))
        * gas.gas_constant * temperature / gas.molecular_weights
    )
    np.testing.assert_allclose(
        enthalpy, ttv.oracle.get_species_enthalpy(temperature), rtol=1e-12
    )


@pytest.mark.parametrize("temperature", nasa_temperatures)
@pytest.mark.parametrize("pressure", [1e3, 1e5, 1e7])
def test_species_entropies_match_plato(ttv, temperature, pressure):
    mech = ttv.mech
    context = {
        "temperature": ttv.temperature(temperature),
        "log": np.log, "exp": np.exp,
    }
    from pymbolic import evaluate
    entropy_r = np.array([
        evaluate(thermo.entropy_poly.expr, context)
        for thermo in mech.species_nasa_thermo_polynomials
    ])
    specific_gas_constant = mech.namespace.gas_constant / mech.molecular_weights
    entropy = specific_gas_constant * (
        entropy_r - np.log(pressure / mech.namespace.ref_pressure)
    )
    np.testing.assert_allclose(
        entropy, ttv.oracle.get_species_entropy(temperature, pressure),
        rtol=1e-12
    )


@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_translational_rotational_specific_heats_match_plato(ttv, temperature):
    # Plato's species cp is the frozen (translational-rotational) part;
    # the vibronic part belongs to the Tv equation.
    np.testing.assert_allclose(
        ttv.mech.translational_rotational_specific_heat_cp,
        ttv.oracle.get_species_cp(temperature), rtol=1e-12
    )


@pytest.mark.xfail(strict=True, reason=(
    "Plato's interval search puts T equal to a break in the upper "
    "interval; bandit's nasa_conditional uses T > bound and picks the "
    "lower one. The NASA-9 fits jump by ~4e-8 there."))
@pytest.mark.parametrize("temperature", [1000.0, 6000.0])
def test_nasa_interval_breaks_follow_plato(ttv, temperature):
    gas = ttv.gas
    enthalpy = (
        gas.get_species_enthalpies_rt(ttv.temperature(temperature))
        * gas.gas_constant * temperature / gas.molecular_weights
    )
    np.testing.assert_allclose(
        enthalpy, ttv.oracle.get_species_enthalpy(temperature), rtol=1e-12
    )


@pytest.mark.parametrize("heavy, vibrational", two_temperature_states)
def test_split_species_energies_match_plato(ttv, heavy, vibrational):
    temperature = ttv.temperature(heavy, vibrational)
    energy = (
        ttv.gas.get_translational_rotational_energy(temperature)
        + ttv.gas.get_nasa_polynomial_vibrational_energy(temperature)
    )
    np.testing.assert_allclose(
        energy, ttv.oracle.get_species_energy(temperature), rtol=1e-12
    )


@pytest.mark.parametrize("vibrational", [800.0, 3000.0, 7000.0, 15000.0])
def test_vibrational_specific_heats_match_plato_energy_slope(ttv, vibrational):
    step = 1e-4 * vibrational

    def plato_energy(tv):
        return ttv.oracle.get_species_energy(np.array([5000.0, tv]))

    slope = (
        plato_energy(vibrational + step) - plato_energy(vibrational - step)
    ) / (2 * step)
    np.testing.assert_allclose(
        ttv.gas.get_nasa_polynomial_vibrational_specific_heat(
            ttv.temperature(5000.0, vibrational)
        ),
        slope, rtol=1e-6, atol=1e-9 * np.abs(slope).max()
    )

# }}}


# {{{ rate coefficients recovered from isolated reactions

@dataclass
class Isolation:
    """A composition that switches on one direction of one reaction.

    Plato's rate of progress for the target reaction is read from the
    production rate of *signature_species*, after removing the
    contribution of any other reaction the composition also switches on.
    """
    reaction_index: int
    direction: str
    concentrations: np.ndarray
    signature_species: int
    contaminants: List[int]

    @property
    def label(self):
        return f"{self.direction}-{self.reaction_index}"


def net_stoichiometry(mech):
    nu = np.zeros((mech.num_species, mech.num_reactions))
    for reaction_index, reaction in enumerate(mech.reactions):
        for species_index in reaction["reactant_species"]:
            nu[species_index, reaction_index] -= 1
        for species_index in reaction["product_species"]:
            nu[species_index, reaction_index] += 1
    return nu


def active_reactions(mech, present):
    """Return the reactions that proceed forward or backward when only
    *present* species have nonzero concentration.
    """
    active = set()
    for reaction_index, reaction in enumerate(mech.reactions):
        partners = set(reaction["partners"])
        has_partner = not partners or bool(partners & present)
        for side in ("reactant_species", "product_species"):
            if set(reaction[side]) <= present and has_partner:
                active.add(reaction_index)
    return active


def make_isolation(mech, reaction_index, direction):
    reaction = mech.reactions[reaction_index]
    # Recombination is third order, so it needs a dense gas to stand
    # out against the bimolecular reactions it cannot avoid switching on.
    is_recombination = direction == "reverse" and reaction["partners"]
    concentration = 1.0 if is_recombination else 1e-3
    side = "reactant_species" if direction == "forward" else "product_species"
    present = set(reaction[side])
    opposite = set(reaction[
        "product_species" if direction == "forward" else "reactant_species"
    ])
    partners = reaction["partners"]
    if partners and not present & set(partners):
        # A partner from the opposite side would switch on the other
        # direction as well.
        present.add(min(partners, key=lambda s: s in opposite))

    nu = net_stoichiometry(mech)
    others = active_reactions(mech, present) - {reaction_index}
    candidates = [
        species_index for species_index in range(mech.num_species)
        if nu[species_index, reaction_index] != 0
    ]
    signature = min(
        candidates,
        key=lambda s: sum(nu[s, other] != 0 for other in others)
    )
    concentrations = np.zeros(mech.num_species)
    concentrations[sorted(present)] = concentration
    return Isolation(
        reaction_index=reaction_index,
        direction=direction,
        concentrations=concentrations,
        signature_species=signature,
        contaminants=sorted(o for o in others if nu[signature, o] != 0),
    )


def mass_action_factor(mech, isolation):
    reaction = mech.reactions[isolation.reaction_index]
    side = (
        "reactant_species" if isolation.direction == "forward"
        else "product_species"
    )
    concentrations = isolation.concentrations
    factor = np.prod([concentrations[s] for s in reaction[side]])
    if reaction["partners"]:
        factor *= sum(concentrations[s] for s in reaction["partners"])
    return factor


def recover_plato_rate_coefficient(case, isolation, temperature):
    """Return Plato's rate coefficient for the isolated reaction and
    direction, and how many times larger the subtracted contamination
    is than the target reaction's own contribution.
    """
    mech, gas = case.mech, case.gas
    nu = net_stoichiometry(mech)
    signature = isolation.signature_species
    omega = plato_source_terms(
        case.oracle,
        isolation.concentrations * gas.molecular_weights,
        temperature,
    )
    signature_rate = omega[signature] / gas.molecular_weights[signature]
    # Other reactions touching the signature species are taken from the
    # generated code; each is validated by its own isolation.
    generated_progress = gas.get_net_rates_of_progress(
        temperature, isolation.concentrations
    )
    contamination = sum(
        nu[signature, other] * generated_progress[other]
        for other in isolation.contaminants
    )
    progress = (
        (signature_rate - contamination)
        / nu[signature, isolation.reaction_index]
    )
    sign = 1 if isolation.direction == "forward" else -1
    rate_coefficient = sign * progress / mass_action_factor(mech, isolation)
    return rate_coefficient, (
        abs(contamination) / abs(signature_rate - contamination)
    )


def generated_rate_coefficient(case, isolation, temperature):
    progress = case.gas.get_net_rates_of_progress(
        temperature, isolation.concentrations
    )[isolation.reaction_index]
    sign = 1 if isolation.direction == "forward" else -1
    return sign * progress / mass_action_factor(case.mech, isolation)


def isolations_for(mech):
    return [
        make_isolation(mech, reaction_index, direction)
        for reaction_index in range(mech.num_reactions)
        for direction in ("forward", "reverse")
    ]


# The subtraction amplifies any mismatch in the contaminating reactions
# by this factor at most, which keeps recovery well inside rtol=1e-10.
max_contamination_ratio = 100.0

# Atom-partner dissociation drowns under the exchange reactions at low
# temperature (by 1e12 at 1500 K), so a state may leave this many
# directions unrecovered; test_every_reaction_direction_is_recoverable
# still checks all of them at 9000 K.
max_unrecovered_per_state = 2


def test_every_reaction_direction_is_recoverable(ttv):
    temperature = ttv.temperature(9000.0, 9000.0)
    recoverable = [
        isolation.label for isolation in isolations_for(ttv.mech)
        if recover_plato_rate_coefficient(ttv, isolation, temperature)[1]
        < max_contamination_ratio
    ]
    assert len(recoverable) == 2 * ttv.mech.num_reactions, recoverable


def well_conditioned_recoveries(case, temperature, direction):
    """Yield each isolation of *direction* whose recovery is trusted at
    *temperature*, with Plato's rate coefficient.
    """
    for isolation in isolations_for(case.mech):
        if isolation.direction != direction:
            continue
        expected, amplification = recover_plato_rate_coefficient(
            case, isolation, temperature
        )
        if amplification < max_contamination_ratio:
            yield isolation, expected


@pytest.mark.parametrize("heavy, vibrational", two_temperature_states)
def test_forward_rate_coefficients_match_plato(ttv, heavy, vibrational):
    temperature = ttv.temperature(heavy, vibrational)
    k_fwd = ttv.gas.get_fwd_rate_coefficients(temperature, None)
    checked = 0
    for isolation, expected in well_conditioned_recoveries(
            ttv, temperature, "forward"):
        assert k_fwd[isolation.reaction_index] == pytest.approx(
            expected, rel=1e-10
        ), isolation.label
        checked += 1
    assert checked >= ttv.mech.num_reactions - max_unrecovered_per_state


@pytest.mark.parametrize("heavy, vibrational", two_temperature_states)
def test_reverse_rate_coefficients_match_plato(ttv, heavy, vibrational):
    temperature = ttv.temperature(heavy, vibrational)
    checked = 0
    for isolation, expected in well_conditioned_recoveries(
            ttv, temperature, "reverse"):
        assert generated_rate_coefficient(
            ttv, isolation, temperature
        ) == pytest.approx(expected, rel=1e-10), isolation.label
        checked += 1
    assert checked >= ttv.mech.num_reactions - max_unrecovered_per_state


def test_dissociation_is_driven_by_geometric_temperature(ttv):
    # kf(Th, Tv) = kf(sqrt(Th Tv)): swapping Th and Tv leaves it alone.
    k_fwd = ttv.gas.get_fwd_rate_coefficients
    dissociation = [
        r for r, reaction in enumerate(ttv.mech.reactions)
        if reaction["type"] == "dh_cat_mt"
    ]
    np.testing.assert_allclose(
        k_fwd(ttv.temperature(9000.0, 4000.0), None)[dissociation],
        k_fwd(ttv.temperature(4000.0, 9000.0), None)[dissociation],
        rtol=1e-13
    )

# }}}


# {{{ species production rates

production_states = list(itertools.product(
    two_temperature_states, compositions.items(), densities
))


@pytest.mark.parametrize(
    "temperatures, composition, density", production_states,
    ids=[f"{t[0]:g}-{t[1]:g}-{c[0]}-{d:g}" for t, c, d in production_states]
)
def test_net_production_rates_match_plato(
        ttv, temperatures, composition, density):
    temperature = ttv.temperature(*temperatures)
    mass_fractions = np.array(composition[1])
    expected = plato_source_terms(
        ttv.oracle, density * mass_fractions, temperature
    )[:ttv.mech.num_species]
    actual = ttv.gas.get_net_production_rates(
        density, temperature, mass_fractions
    ) * ttv.gas.molecular_weights
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


def test_net_production_rates_conserve_mass(ttv):
    mass_fractions = np.array(compositions["partially_dissociated"])
    rates = ttv.gas.get_net_production_rates(
        5e-2, ttv.temperature(9000.0, 3000.0), mass_fractions
    ) * ttv.gas.molecular_weights
    assert abs(rates.sum()) < 1e-12 * np.abs(rates).max()

# }}}


# {{{ energy exchange

# Plato raises Th to -0.3333333333 rather than -1/3 in Millikan-White,
# which moves p tau by up to ~1e-8 at low temperature.
relaxation_time_rtol = 2e-8


@pytest.mark.parametrize("heavy", [500.0, 2000.0, 8000.0, 20000.0, 30000.0])
def test_pressure_relaxation_times_match_plato(ttv, heavy):
    np.testing.assert_allclose(
        ttv.gas.get_pressure_relaxation_times(ttv.temperature(heavy, 1000.0)),
        ttv.oracle.get_ptau_vt(heavy), rtol=relaxation_time_rtol
    )


def plato_vibrational_source(case, density, temperature, mass_fractions):
    """Split Plato's Tv-equation source into its Landau-Teller (VT) part
    and its chemistry-vibration (CV) part, sum_s omega_s e_vib,s(Tv).
    """
    omega = plato_source_terms(
        case.oracle, density * mass_fractions, temperature
    )
    num_species = case.mech.num_species
    chemistry_vibration = np.sum(
        omega[:num_species]
        * case.gas.get_nasa_polynomial_vibrational_energy(temperature)
    )
    return omega[num_species + 1], chemistry_vibration


energy_states = list(itertools.product(
    [s for s in two_temperature_states if s[0] != s[1]],
    compositions.items(),
))


@pytest.mark.parametrize(
    "temperatures, composition", energy_states,
    ids=[f"{t[0]:g}-{t[1]:g}-{c[0]}" for t, c in energy_states]
)
def test_vt_energy_transfer_matches_plato(ttv, temperatures, composition):
    temperature = ttv.temperature(*temperatures)
    mass_fractions = np.array(composition[1])
    total, chemistry_vibration = plato_vibrational_source(
        ttv, 5e-2, temperature, mass_fractions
    )
    actual = ttv.gas.get_vt_energy_transfer_source(
        5e-2, temperature, mass_fractions
    )
    assert actual == pytest.approx(
        total - chemistry_vibration,
        rel=relaxation_time_rtol, abs=1e-10 * abs(chemistry_vibration)
    )


@pytest.mark.parametrize(
    "temperatures, composition", energy_states,
    ids=[f"{t[0]:g}-{t[1]:g}-{c[0]}" for t, c in energy_states]
)
def test_vibrational_energy_source_matches_plato(
        ttv, temperatures, composition):
    temperature = ttv.temperature(*temperatures)
    mass_fractions = np.array(composition[1])
    gas = ttv.gas
    total, _ = plato_vibrational_source(ttv, 5e-2, temperature, mass_fractions)
    mass_production = gas.get_net_production_rates(
        5e-2, temperature, mass_fractions
    ) * gas.molecular_weights
    actual = gas.get_vt_energy_transfer_source(
        5e-2, temperature, mass_fractions
    ) + np.sum(
        mass_production
        * gas.get_nasa_polynomial_vibrational_energy(temperature)
    )
    assert actual == pytest.approx(total, rel=relaxation_time_rtol)


def test_vt_energy_transfer_vanishes_at_thermal_equilibrium(ttv):
    mass_fractions = np.array(compositions["partially_dissociated"])
    temperature = ttv.temperature(7000.0, 7000.0)
    total, chemistry_vibration = plato_vibrational_source(
        ttv, 5e-2, temperature, mass_fractions
    )
    assert ttv.gas.get_vt_energy_transfer_source(
        5e-2, temperature, mass_fractions
    ) == 0.0
    assert total == pytest.approx(chemistry_vibration, rel=1e-12)

# }}}


# {{{ array semantics

def batched_states():
    rng = np.random.default_rng(seed=0)
    num_states = 16
    heavy = rng.uniform(1000.0, 15000.0, num_states)
    vibrational = rng.uniform(1000.0, 15000.0, num_states)
    mass_fractions = rng.dirichlet(np.ones(5), num_states).T
    density = rng.uniform(1e-3, 1e-1, num_states)
    return density, np.stack([heavy, vibrational]), mass_fractions


def plato_pointwise(case, density, temperature, mass_fractions):
    return np.stack([
        plato_source_terms(
            case.oracle,
            density[i] * mass_fractions[:, i],
            temperature[:, i],
        )
        for i in range(density.size)
    ], axis=-1)


def test_batched_states_match_plato_pointwise(ttv):
    density, temperature, mass_fractions = batched_states()
    expected = plato_pointwise(ttv, density, temperature, mass_fractions)
    gas = ttv.gas
    production = gas.get_net_production_rates(
        density, temperature, mass_fractions
    ) * gas.molecular_weights[:, None]
    np.testing.assert_allclose(
        production, expected[:ttv.mech.num_species],
        rtol=1e-10, atol=1e-10 * np.abs(expected[:5]).max()
    )
    vibrational_source = gas.get_vt_energy_transfer_source(
        density, temperature, mass_fractions
    ) + np.sum(
        production * gas.get_nasa_polynomial_vibrational_energy(temperature),
        axis=0
    )
    np.testing.assert_allclose(
        vibrational_source, expected[-1], rtol=relaxation_time_rtol
    )


@pytest.mark.skipif(jnp is None, reason="jax is not installed")
def test_jax_generated_code_matches_plato(ttv):
    density, temperature, mass_fractions = batched_states()
    expected = plato_pointwise(ttv, density, temperature, mass_fractions)
    gas = make_generated_gas(ttv.mech, jnp)
    production = np.asarray(jax.jit(gas.get_net_production_rates)(
        jnp.asarray(density), jnp.asarray(temperature),
        jnp.asarray(mass_fractions)
    )) * ttv.gas.molecular_weights[:, None]
    np.testing.assert_allclose(
        production, expected[:ttv.mech.num_species],
        rtol=1e-10, atol=1e-10 * np.abs(expected[:5]).max()
    )
    vt = np.asarray(jax.jit(gas.get_vt_energy_transfer_source)(
        jnp.asarray(density), jnp.asarray(temperature),
        jnp.asarray(mass_fractions)
    ))
    assert np.all(np.isfinite(vt))

# }}}


# {{{ single-temperature mixtures

@pytest.fixture
def single_temperature():
    return plato_case("empty")


@pytest.mark.parametrize("temperature", single_temperatures)
@pytest.mark.parametrize("composition", compositions.values(),
                         ids=compositions.keys())
def test_single_temperature_production_rates_match_plato(
        single_temperature, temperature, composition):
    case = single_temperature
    mass_fractions = np.array(composition)
    expected = plato_source_terms(
        case.oracle, 5e-2 * mass_fractions, temperature
    )[:case.mech.num_species]
    actual = case.gas.get_net_production_rates(
        5e-2, temperature, mass_fractions
    ) * case.gas.molecular_weights
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_single_temperature_enthalpies_match_plato(
        single_temperature, temperature):
    gas = single_temperature.gas
    enthalpy = (
        gas.get_species_enthalpies_rt(temperature)
        * gas.gas_constant * temperature / gas.molecular_weights
    )
    np.testing.assert_allclose(
        enthalpy,
        single_temperature.oracle.get_species_enthalpy(temperature),
        rtol=1e-12
    )

# }}}
