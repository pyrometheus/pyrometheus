"""Parity of Pyrometheus-generated code against Plato's own evaluations.

Every expected value comes out of the Plato library (``get_source``,
``get_species_*``, ``get_ptau_vt``); nothing is recomputed from the
parameters bandit reads. Plato does not expose per-reaction rate
coefficients, so they are recovered from ``get_source`` at sparse
compositions where only one reaction's reactants are present.

The tests are written once and run for every case in :data:`cases`, as
the release suite does for Cantera mechanisms. What a case lacks (a
vibrational temperature, VT transfer) skips; what bandit gets wrong for
it is listed in :data:`known_gaps` and runs as a strict xfail.

Requires PLATO_DB and an importable ``plato`` package.
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Callable, List

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


# {{{ cases and known gaps

@dataclass(frozen=True)
class PlatoSpec:
    mixture: str
    reaction_set: str
    transfer: str

    @property
    def label(self):
        return f"{self.reaction_set}-{self.transfer}"

    @property
    def is_ionized(self):
        return self.mixture.startswith("air11")


# Every NASA-9 mixture in the Plato database whose kinetics Plato's NASA
# build can evaluate. air11_dunn_kang is left out: its heavy-impact
# ionization (Ih) rates end the process in compute_rate_coeff.
cases = [
    PlatoSpec(mixture, reaction_set, transfer)
    for mixture, reaction_set in [
        ("air5", "air5"),
        ("air11", "air11_park93"),
    ]
    for transfer in ["empty", "TTv", "TTvTe"]
]


@dataclass(frozen=True)
class KnownGap:
    """A strict xfail for the tests of *block* whose case and parameters
    satisfy *applies*(spec, params)."""
    applies: Callable[[PlatoSpec, dict], bool]
    block: str
    reason: str


def _ionized(spec, params):
    return spec.is_ionized


def _electron_temperature(spec, state):
    return state[{"TTv": 1, "TTvTe": 2}[spec.transfer]]


def _ionized_multi_temperature(spec, params):
    return spec.is_ionized and spec.transfer != "empty"


def _electrons_out_of_equilibrium(spec, params):
    state = params["state"]
    return (_ionized_multi_temperature(spec, params)
            and _electron_temperature(spec, state) != state[0])


def _electron_energy_exchange(spec, params):
    if not _ionized_multi_temperature(spec, params):
        return False
    # Batched states draw random compositions, electrons included.
    composition = params.get("composition", "ionized")
    return spec.transfer == "TTvTe" or composition == "ionized"


known_gaps = [
    KnownGap(_ionized, "transport", (
        "transport is ported for neutral mixtures only: ion-neutral charge "
        "exchange, electron-neutral fits and screened Coulomb pairs are "
        "not generated")),
    KnownGap(_ionized, "kinetics", (
        "bandit drops the electron from electron-impact rates and "
        "stoichiometry (N + e -> N+ + 2e is built as N -> N+), from the "
        "products of associative ionization, and evaluates their backward "
        "rates at Th instead of Te")),
    KnownGap(_ionized_multi_temperature, "electron_heat_capacity", (
        "the electron's translational heat capacity is booked to Th; "
        "Plato books it to the electron temperature")),
    KnownGap(_electrons_out_of_equilibrium, "energy_split", (
        "the electron's translational energy is evaluated at Th instead "
        "of the electron temperature")),
    KnownGap(_electron_energy_exchange, "energy_exchange", (
        "Plato's Tv source carries TE and CEL electron terms bandit does "
        "not generate, and with TTvTe it books electronic excitation to "
        "Te where bandit's NASA split lumps it into Tv")),
]


def block(*names):
    """Tag a test with the physics blocks :data:`known_gaps` refers to."""
    def decorate(test):
        test.physics_blocks = set(names)
        return test
    return decorate

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
    spec: PlatoSpec
    mech: "PlatoMechanism"
    gas: object

    @property
    def oracle(self):
        return self.mech.namespace.thermochem

    @property
    def num_temp(self):
        return self.mech.num_temp

    def temperature(self, state):
        """Return the temperature argument for a (Th, Tv, Te) *state*,
        truncated to the temperatures this case carries.
        """
        if self.num_temp == 1:
            return state[0]
        return np.array(state[:self.num_temp])

    def species_of_class(self, kind):
        return [
            species_index for species_index, name
            in enumerate(self.oracle.species_names)
            if species_class(self, species_index) == kind
        ]


def species_class(case, species_index):
    name = case.oracle.species_names[species_index]
    if name == "em":
        return "electron"
    if name.endswith("p"):
        return "ion"
    # theta_vib expects a 1-based component index.
    if len(case.oracle.theta_vib(species_index + 1)):
        return "molecule"
    return "atom"


# Plato holds one live instance per process; pytest closes a
# module-scoped parametrized fixture before opening its next parameter.
@pytest.fixture(scope="module", params=cases, ids=lambda spec: spec.label)
def case(request):
    spec = request.param
    mech = PlatoMechanism(
        spec.mixture, spec.reaction_set, spec.transfer,
        os.environ["PLATO_DB"]
    )
    yield PlatoCase(spec=spec, mech=mech, gas=make_generated_gas(mech))
    mech.finalize()


@pytest.fixture(autouse=True)
def _mark_known_gaps(request, case):
    blocks = getattr(request.function, "physics_blocks", set())
    params = dict(getattr(request.node, "callspec", None).params)
    for gap in known_gaps:
        if gap.block in blocks and gap.applies(case.spec, params):
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=gap.reason)
            )


def require(case, multi_temperature=None, vt=False, plato_thermo=False):
    # Plato's NASA-9 species properties take at most two temperatures;
    # asking with more kills the process with exit status 0
    # (nasa_thermo%get_h_unit_mass), so the check must come first.
    if plato_thermo and case.num_temp > 2:
        pytest.skip("Plato's NASA-9 species properties take <= 2 temperatures")
    if multi_temperature is True and case.num_temp == 1:
        pytest.skip("needs a vibrational temperature")
    if multi_temperature is False and case.num_temp > 1:
        pytest.skip("single-temperature API")
    if vt and not case.oracle.vt_active:
        pytest.skip("no VT transfer in this case")


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


# {{{ states

# (Th, Tv, Te): thermal equilibrium, vibrationally frozen (Tv < Th,
# post-shock) and vibrationally hot (Tv > Th, expansion). Every value
# sits above Plato's rate-evaluation floor of 350 K.
equilibrium_states = [
    (1500.0,) * 3, (6500.0,) * 3, (12000.0,) * 3,
]
nonequilibrium_states = [
    (9000.0, 3000.0, 12000.0),
    (15000.0, 7000.0, 9000.0),
    (4000.0, 8000.0, 5000.0),
]
states = equilibrium_states + nonequilibrium_states

# Crosses both NASA-9 interval breaks (1000 K, 6000 K) from either side.
nasa_temperatures = [300.0, 999.0, 1001.0, 3000.0, 5999.0, 6001.0, 18000.0]
densities = [1e-3, 5e-2]
composition_names = ["molecular", "dissociated", "atomic", "ionized"]


def mole_shares(case, composition):
    """Split mole fractions between species classes; zero shares keep a
    1e-12 floor so every species participates.
    """
    shares = {
        "molecular": {"molecule": 0.95, "atom": 0.05},
        "dissociated": {"molecule": 0.4, "atom": 0.6},
        "atomic": {"molecule": 0.05, "atom": 0.95},
        "ionized": {"molecule": 0.3, "atom": 0.68, "ion": 0.01},
    }[composition]
    mole_fractions = np.full(case.mech.num_species, 1e-12)
    for kind, share in shares.items():
        members = case.species_of_class(kind)
        mole_fractions[members] += share / len(members)
    electrons = case.species_of_class("electron")
    if electrons:
        # Quasi-neutral: one electron per singly charged ion.
        ions = case.species_of_class("ion")
        mole_fractions[electrons] = mole_fractions[ions].sum()
    return mole_fractions / mole_fractions.sum()


def mass_fractions_for(case, composition):
    if composition == "ionized" and not case.spec.is_ionized:
        pytest.skip("no ions in this mixture")
    weighted = mole_shares(case, composition) * case.gas.molecular_weights
    return weighted / weighted.sum()

# }}}


# {{{ species metadata and constants

@block("constants")
def test_molecular_weights_match_plato(case):
    np.testing.assert_allclose(
        case.gas.molecular_weights, 1e3 * case.oracle.molar_masses,
        rtol=1e-15
    )


@block("constants")
def test_specific_gas_constants_match_plato(case):
    np.testing.assert_allclose(
        case.gas.gas_constant / case.gas.molecular_weights,
        case.oracle.gas_constants, rtol=1e-14
    )


@block("constants")
def test_energies_of_formation_match_plato(case):
    # Plato splits energy with its tabulated heat of formation, not the
    # NASA-9 fit evaluated at 298.15 K; the two differ by ~200 J/kg.
    np.testing.assert_allclose(
        case.mech.standard_energy_of_formation,
        case.oracle.formation_energies, rtol=1e-14, atol=1e-8
    )


@block("constants")
def test_species_indices_follow_plato_order(case):
    assert case.gas.species_names == case.oracle.species_names
    for species_index, species_name in enumerate(case.oracle.species_names):
        assert case.gas.get_species_index(species_name) == species_index


@block("constants")
def test_equation_of_state_matches_plato_gas_constants(case):
    mass_fractions = mass_fractions_for(case, "dissociated")
    density, temperature = 2e-2, 5000.0
    pressure = density * temperature * (
        mass_fractions @ case.oracle.gas_constants
    )
    assert case.gas.get_pressure(
        density, temperature, mass_fractions
    ) == pytest.approx(pressure, rel=1e-14)
    assert case.gas.get_density(
        pressure, temperature, mass_fractions
    ) == pytest.approx(density, rel=1e-14)

# }}}


# {{{ NASA-9 thermodynamics

def uniform_temperature(case, temperature):
    return case.temperature((temperature,) * 3)


@block("thermo")
@pytest.mark.parametrize("temperature", nasa_temperatures + [1000.0, 6000.0])
def test_species_enthalpies_match_plato(case, temperature):
    require(case, plato_thermo=True)
    # 1000 K and 6000 K sit exactly on the breaks, where the fits jump by
    # ~4e-8: Plato evaluates the upper interval there.
    gas = case.gas
    enthalpy = (
        gas.get_species_enthalpies_rt(uniform_temperature(case, temperature))
        * gas.gas_constant * temperature / gas.molecular_weights
    )
    np.testing.assert_allclose(
        enthalpy, case.oracle.get_species_enthalpy(temperature), rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", nasa_temperatures)
@pytest.mark.parametrize("pressure", [1e3, 1e5, 1e7])
def test_species_entropies_match_plato(case, temperature, pressure):
    require(case, plato_thermo=True)
    from pymbolic import evaluate
    mech = case.mech
    context = {
        "temperature": uniform_temperature(case, temperature),
        "log": np.log, "exp": np.exp,
    }
    entropy_r = np.array([
        evaluate(thermo.entropy_poly.expr, context)
        for thermo in mech.species_nasa_thermo_polynomials
    ])
    specific_gas_constant = mech.namespace.gas_constant / mech.molecular_weights
    entropy = specific_gas_constant * (
        entropy_r - np.log(pressure / mech.namespace.ref_pressure)
    )
    np.testing.assert_allclose(
        entropy, case.oracle.get_species_entropy(temperature, pressure),
        rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", [300.0, 3000.0, 9000.0, 18000.0])
def test_species_specific_heats_match_plato_enthalpy_slope(case, temperature):
    require(case, plato_thermo=True)
    gas = case.gas
    step = 1e-4 * temperature
    slope = (
        case.oracle.get_species_enthalpy(temperature + step)
        - case.oracle.get_species_enthalpy(temperature - step)
    ) / (2 * step)
    np.testing.assert_allclose(
        gas.get_species_specific_heats_cp_r(
            uniform_temperature(case, temperature)
        ) * gas.gas_constant / gas.molecular_weights,
        slope, rtol=1e-7
    )


@block("thermo")
@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_species_specific_heats_match_plato(case, temperature):
    # With one temperature Plato's species cp is the full NASA-9 cp.
    require(case, multi_temperature=False)
    gas = case.gas
    np.testing.assert_allclose(
        gas.get_species_specific_heats_cp_r(temperature)
        * gas.gas_constant / gas.molecular_weights,
        case.oracle.get_species_cp(temperature), rtol=1e-12
    )


@block("electron_heat_capacity")
@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_translational_rotational_specific_heats_match_plato(
        case, temperature):
    # With more temperatures Plato's species cp is the frozen part that
    # the heavy-particle temperature carries.
    require(case, multi_temperature=True, plato_thermo=True)
    np.testing.assert_allclose(
        case.mech.translational_rotational_specific_heat_cp,
        case.oracle.get_species_cp(temperature), rtol=1e-12
    )


@block("thermo")
@pytest.mark.parametrize("temperature", nasa_temperatures)
def test_species_internal_energies_match_plato(case, temperature):
    require(case, multi_temperature=False)
    gas = case.gas
    np.testing.assert_allclose(
        gas.get_species_internal_energies_rt(temperature)
        * gas.gas_constant * temperature / gas.molecular_weights,
        case.oracle.get_species_energy(np.array([temperature])),
        rtol=1e-12
    )


def plato_mixture_properties(oracle, temperature, mass_fractions):
    cp = oracle.get_species_cp(temperature)
    return {
        "cp": mass_fractions @ cp,
        "cv": mass_fractions @ (cp - oracle.gas_constants),
        "enthalpy": mass_fractions @ oracle.get_species_enthalpy(temperature),
        "energy": mass_fractions @ oracle.get_species_energy(
            np.array([temperature])
        ),
    }


@block("thermo")
@pytest.mark.parametrize("temperature", [800.0, 3000.0, 6500.0, 12000.0])
@pytest.mark.parametrize("composition", composition_names)
def test_mixture_properties_match_plato(case, temperature, composition):
    require(case, multi_temperature=False)
    gas = case.gas
    mass_fractions = mass_fractions_for(case, composition)
    expected = plato_mixture_properties(
        case.oracle, temperature, mass_fractions
    )
    for method, key in [
        (gas.get_mixture_specific_heat_cp_mass, "cp"),
        (gas.get_mixture_specific_heat_cv_mass, "cv"),
        (gas.get_mixture_enthalpy_mass, "enthalpy"),
        (gas.get_mixture_internal_energy_mass, "energy"),
    ]:
        assert method(temperature, mass_fractions) == pytest.approx(
            expected[key], rel=1e-12
        ), key


@block("thermo")
@pytest.mark.parametrize("temperature", [800.0, 3000.0, 6500.0, 12000.0])
@pytest.mark.parametrize("do_energy", [False, True])
def test_temperature_inversion_recovers_plato_state(
        case, temperature, do_energy):
    require(case, multi_temperature=False)
    mass_fractions = mass_fractions_for(case, "dissociated")
    expected = plato_mixture_properties(
        case.oracle, temperature, mass_fractions
    )
    target = expected["energy" if do_energy else "enthalpy"]
    assert case.gas.get_temperature(
        target, 1000.0, mass_fractions, do_energy
    ) == pytest.approx(temperature, rel=1e-9)

# }}}


# {{{ two-temperature energy split

@block("energy_split")
@pytest.mark.parametrize("state", states)
def test_split_species_energies_match_plato(case, state):
    require(case, multi_temperature=True, plato_thermo=True)
    temperature = case.temperature(state)
    energy = (
        case.gas.get_translational_rotational_energy(temperature)
        + case.gas.get_nasa_polynomial_vibrational_energy(temperature)
    )
    np.testing.assert_allclose(
        energy, case.oracle.get_species_energy(temperature), rtol=1e-12
    )


@block("electron_heat_capacity")
@pytest.mark.parametrize("vibrational", [800.0, 3000.0, 7000.0, 15000.0])
def test_vibrational_specific_heats_match_plato_energy_slope(
        case, vibrational):
    require(case, multi_temperature=True, plato_thermo=True)
    step = 1e-4 * vibrational

    def plato_energy(tv):
        return case.oracle.get_species_energy(
            case.temperature((5000.0, tv, tv))
        )

    slope = (
        plato_energy(vibrational + step) - plato_energy(vibrational - step)
    ) / (2 * step)
    np.testing.assert_allclose(
        case.gas.get_nasa_polynomial_vibrational_specific_heat(
            case.temperature((5000.0, vibrational, vibrational))
        ),
        slope, rtol=1e-6, atol=1e-9 * np.abs(slope).max()
    )

# }}}


# {{{ rate coefficients recovered from isolated reactions

# Plato's electron-impact reactions carry the electron as a catalyst that
# its species getters leave out (kinetics_source.F90, De and Ie).
electron_impact_types = ("de", "ie")


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


def electron_indices(mech):
    return [mech.species_index("em")] if "em" in mech.species_names else []


def catalysts(mech, reaction):
    if reaction["type"] in electron_impact_types:
        return set(electron_indices(mech))
    return set()


def active_reactions(mech, present):
    """Return the reactions that proceed forward or backward when only
    *present* species have nonzero concentration.
    """
    active = set()
    for reaction_index, reaction in enumerate(mech.reactions):
        partners = set(reaction["partners"])
        has_partner = not partners or bool(partners & present)
        needs = catalysts(mech, reaction)
        for side in ("reactant_species", "product_species"):
            if (set(reaction[side]) | needs) <= present and has_partner:
                active.add(reaction_index)
    return active


def make_isolation(mech, reaction_index, direction):
    reaction = mech.reactions[reaction_index]
    # Recombination is third order, so it needs a dense gas to stand
    # out against the bimolecular reactions it cannot avoid switching on.
    is_recombination = direction == "reverse" and reaction["partners"]
    concentration = 1.0 if is_recombination else 1e-3
    side = "reactant_species" if direction == "forward" else "product_species"
    present = set(reaction[side]) | catalysts(mech, reaction)
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
    for catalyst in catalysts(mech, reaction):
        factor *= concentrations[catalyst]
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
    signal = abs(signature_rate - contamination)
    return rate_coefficient, (
        abs(contamination) / signal if signal else np.inf
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
# temperature (by 1e12 at 1500 K), so a state may leave this fraction of
# directions unrecovered; test_every_reaction_direction_is_recoverable
# still requires nearly all of them at 9000 K.
max_unrecovered_fraction = 0.25


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


@block("isolation")
def test_every_reaction_direction_is_recoverable(case):
    temperature = case.temperature((9000.0,) * 3)
    unrecovered = [
        isolation.label for isolation in isolations_for(case.mech)
        if recover_plato_rate_coefficient(case, isolation, temperature)[1]
        >= max_contamination_ratio
    ]
    assert len(unrecovered) <= 0.1 * 2 * case.mech.num_reactions, unrecovered


def assert_recovered_rate_coefficients(case, state, direction, generated):
    temperature = case.temperature(state)
    checked = 0
    for isolation, expected in well_conditioned_recoveries(
            case, temperature, direction):
        assert generated(isolation, temperature) == pytest.approx(
            expected, rel=1e-10
        ), isolation.label
        checked += 1
    assert checked >= (1 - max_unrecovered_fraction) * case.mech.num_reactions


@block("kinetics")
@pytest.mark.parametrize("state", states)
def test_forward_rate_coefficients_match_plato(case, state):
    def generated(isolation, temperature):
        return case.gas.get_fwd_rate_coefficients(
            temperature, isolation.concentrations
        )[isolation.reaction_index]

    assert_recovered_rate_coefficients(case, state, "forward", generated)


@block("kinetics")
@pytest.mark.parametrize("state", states)
def test_reverse_rate_coefficients_match_plato(case, state):
    def generated(isolation, temperature):
        return generated_rate_coefficient(case, isolation, temperature)

    assert_recovered_rate_coefficients(case, state, "reverse", generated)


@block("dissociation_temperature")
def test_dissociation_is_driven_by_geometric_temperature(case):
    # kf(Th, Tv) = kf(sqrt(Th Tv)): swapping Th and Tv leaves it alone.
    require(case, multi_temperature=True)
    k_fwd = case.gas.get_fwd_rate_coefficients
    dissociation = [
        r for r, reaction in enumerate(case.mech.reactions)
        if reaction["type"] == "dh_cat_mt"
    ]
    np.testing.assert_allclose(
        k_fwd(case.temperature((9000.0, 4000.0, 9000.0)), None)[dissociation],
        k_fwd(case.temperature((4000.0, 9000.0, 9000.0)), None)[dissociation],
        rtol=1e-13
    )

# }}}


# {{{ species production rates

@block("kinetics")
@pytest.mark.parametrize("density", densities)
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", states)
def test_net_production_rates_match_plato(case, state, composition, density):
    temperature = case.temperature(state)
    mass_fractions = mass_fractions_for(case, composition)
    expected = plato_source_terms(
        case.oracle, density * mass_fractions, temperature
    )[:case.mech.num_species]
    actual = case.gas.get_net_production_rates(
        density, temperature, mass_fractions
    ) * case.gas.molecular_weights
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@block("kinetics")
def test_net_production_rates_conserve_mass(case):
    mass_fractions = mass_fractions_for(case, "dissociated")
    rates = case.gas.get_net_production_rates(
        5e-2, case.temperature(nonequilibrium_states[0]), mass_fractions
    ) * case.gas.molecular_weights
    assert abs(rates.sum()) < 1e-12 * np.abs(rates).max()

# }}}


# {{{ energy exchange

# Plato raises Th to -0.3333333333 rather than -1/3 in Millikan-White,
# which moves p tau by up to ~1e-8 at low temperature.
relaxation_time_rtol = 2e-8


@block("relaxation_times")
@pytest.mark.parametrize("heavy", [500.0, 2000.0, 8000.0, 20000.0, 30000.0])
def test_pressure_relaxation_times_match_plato(case, heavy):
    require(case, multi_temperature=True, vt=True)
    np.testing.assert_allclose(
        case.gas.get_pressure_relaxation_times(
            case.temperature((heavy, 1000.0, 1000.0))
        ),
        case.oracle.get_ptau_vt(heavy), rtol=relaxation_time_rtol
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


@block("energy_exchange")
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", nonequilibrium_states)
def test_vt_energy_transfer_matches_plato(case, state, composition):
    require(case, multi_temperature=True, vt=True)
    temperature = case.temperature(state)
    mass_fractions = mass_fractions_for(case, composition)
    total, chemistry_vibration = plato_vibrational_source(
        case, 5e-2, temperature, mass_fractions
    )
    actual = case.gas.get_vt_energy_transfer_source(
        5e-2, temperature, mass_fractions
    )
    assert actual == pytest.approx(
        total - chemistry_vibration,
        rel=relaxation_time_rtol, abs=1e-10 * abs(chemistry_vibration)
    )


@block("kinetics", "energy_exchange")
@pytest.mark.parametrize("composition", composition_names)
@pytest.mark.parametrize("state", nonequilibrium_states)
def test_vibrational_energy_source_matches_plato(case, state, composition):
    require(case, multi_temperature=True, vt=True)
    temperature = case.temperature(state)
    mass_fractions = mass_fractions_for(case, composition)
    gas = case.gas
    total, _ = plato_vibrational_source(case, 5e-2, temperature, mass_fractions)
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


@block("relaxation_times")
def test_vt_energy_transfer_vanishes_at_thermal_equilibrium(case):
    require(case, multi_temperature=True, vt=True)
    mass_fractions = mass_fractions_for(case, "dissociated")
    temperature = case.temperature((7000.0,) * 3)
    total, chemistry_vibration = plato_vibrational_source(
        case, 5e-2, temperature, mass_fractions
    )
    assert case.gas.get_vt_energy_transfer_source(
        5e-2, temperature, mass_fractions
    ) == 0.0
    assert total == pytest.approx(chemistry_vibration, rel=1e-12)

# }}}


# {{{ array semantics

def batched_states(case):
    rng = np.random.default_rng(seed=0)
    num_states = 16
    temperature = rng.uniform(1000.0, 15000.0, (case.num_temp, num_states))
    if case.num_temp == 1:
        temperature = temperature[0]
    mass_fractions = rng.dirichlet(
        np.ones(case.mech.num_species), num_states
    ).T
    density = rng.uniform(1e-3, 1e-1, num_states)
    return density, temperature, mass_fractions


def plato_pointwise(case, density, temperature, mass_fractions):
    temperature = np.atleast_2d(temperature)
    return np.stack([
        plato_source_terms(
            case.oracle,
            density[i] * mass_fractions[:, i],
            temperature[:, i],
        )
        for i in range(density.size)
    ], axis=-1)


@block("kinetics")
def test_batched_states_match_plato_pointwise(case):
    density, temperature, mass_fractions = batched_states(case)
    num_species = case.mech.num_species
    expected = plato_pointwise(case, density, temperature, mass_fractions)
    production = case.gas.get_net_production_rates(
        density, temperature, mass_fractions
    ) * case.gas.molecular_weights[:, None]
    np.testing.assert_allclose(
        production, expected[:num_species],
        rtol=1e-10, atol=1e-10 * np.abs(expected[:num_species]).max()
    )


@block("kinetics", "energy_exchange")
def test_batched_vibrational_source_matches_plato_pointwise(case):
    require(case, multi_temperature=True, vt=True)
    density, temperature, mass_fractions = batched_states(case)
    num_species = case.mech.num_species
    expected = plato_pointwise(case, density, temperature, mass_fractions)
    gas = case.gas
    production = gas.get_net_production_rates(
        density, temperature, mass_fractions
    ) * gas.molecular_weights[:, None]
    vibrational_source = gas.get_vt_energy_transfer_source(
        density, temperature, mass_fractions
    ) + np.sum(
        production * gas.get_nasa_polynomial_vibrational_energy(temperature),
        axis=0
    )
    np.testing.assert_allclose(
        vibrational_source, expected[num_species + 1],
        rtol=relaxation_time_rtol
    )


@block("kinetics")
@pytest.mark.skipif(jnp is None, reason="jax is not installed")
def test_jax_generated_code_matches_plato(case):
    density, temperature, mass_fractions = batched_states(case)
    num_species = case.mech.num_species
    expected = plato_pointwise(case, density, temperature, mass_fractions)
    gas = make_generated_gas(case.mech, jnp)
    production = np.asarray(jax.jit(gas.get_net_production_rates)(
        jnp.asarray(density), jnp.asarray(temperature),
        jnp.asarray(mass_fractions)
    )) * case.gas.molecular_weights[:, None]
    np.testing.assert_allclose(
        production, expected[:num_species],
        rtol=1e-10, atol=1e-10 * np.abs(expected[:num_species]).max()
    )

# }}}


# {{{ transport

# Th sits on PLATO's collision-integral table nodes (200 K + k 50 K),
# where its linear interpolation equals the fits bandit generates.
transport_states = [
    (300.0, 300.0, 300.0),
    (1000.0, 1000.0, 1000.0),
    (5050.0, 3000.0, 3000.0),
    (15000.0, 8000.0, 8000.0),
    (40000.0, 20000.0, 20000.0),
]
off_node_heavy_temperatures = [325.0, 1025.0, 5025.0, 15025.0]
transport_compositions = ["molecular", "dissociated", "atomic"]


def heavy_pairs(case):
    num_species = case.mech.num_species
    return [(i, j) for i in range(num_species) for j in range(i, num_species)]


def interpolation_bound(case, heavy_temperature):
    """:returns: the largest relative difference PLATO's linear table
    interpolation makes from the fits midway between nodes."""
    _, _, step = case.oracle.collision_integral_table_range()
    return 0.12 * (step / heavy_temperature) ** 2


def plato_number_density(case, temperature, mole_fractions, pressure=1e5):
    return case.oracle.number_density(
        pressure, np.atleast_1d(temperature), mole_fractions
    )


def mole_fractions_of(case, mass_fractions):
    weighted = mass_fractions / case.gas.molecular_weights
    return weighted / weighted.sum()


def evaluate_pair_integrals(case, i, j, heavy_temperature):
    from pymbolic import evaluate
    integrals = case.mech.pair_collision_integrals[i][j]
    context = {
        "temperature": case.temperature((heavy_temperature,) * 3),
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    }
    return {
        name: evaluate(getattr(integrals, name), context)
        for name in ("q_11", "q_22", "b_star", "c_star")
    }


@block("transport")
@pytest.mark.parametrize("heavy", [state[0] for state in transport_states])
def test_collision_integrals_match_plato_at_table_nodes(case, heavy):
    require(case, plato_thermo=True)
    for i, j in heavy_pairs(case):
        expected = case.oracle.collision_integrals(i + 1, j + 1, heavy)
        actual = evaluate_pair_integrals(case, i, j, heavy)
        for name, value in expected.items():
            assert actual[name] == pytest.approx(value, rel=1e-12, abs=0), (
                i, j, name)


@block("transport")
@pytest.mark.parametrize("heavy", off_node_heavy_temperatures)
def test_collision_integrals_between_table_nodes(case, heavy):
    require(case, plato_thermo=True)
    bound = interpolation_bound(case, heavy)
    for i, j in heavy_pairs(case):
        expected = case.oracle.collision_integrals(i + 1, j + 1, heavy)
        actual = evaluate_pair_integrals(case, i, j, heavy)
        assert actual["q_11"] == pytest.approx(
            expected["q_11"], rel=bound, abs=0)


@block("transport")
@pytest.mark.parametrize("state", transport_states)
def test_binary_diffusivities_match_plato(case, state):
    require(case, plato_thermo=True)
    temperature = case.temperature(state)
    mole_fractions = mole_shares(case, "dissociated")
    pressure = 1e5
    expected = case.oracle.binary_diffusion_coefficients(
        plato_number_density(case, temperature, mole_fractions, pressure),
        np.atleast_1d(temperature), mole_fractions
    )
    # The release convention: generated code returns D_ij times pressure.
    actual = np.array(
        case.gas.get_species_binary_mass_diffusivities(temperature)
    ) / pressure
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@block("transport")
@pytest.mark.parametrize("state", transport_states)
def test_species_viscosities_match_pure_gases_in_plato(case, state):
    require(case, plato_thermo=True)
    temperature = case.temperature(state)
    viscosities = case.gas.get_species_viscosities(temperature)
    for species_index in range(case.mech.num_species):
        pure = np.zeros(case.mech.num_species)
        pure[species_index] = 1.0
        expected = case.oracle.viscosity(
            plato_number_density(case, temperature, pure),
            np.atleast_1d(temperature), pure
        )
        # PLATO's 1e-14 mole-fraction floor leaves traces of every other
        # species in a "pure" gas.
        assert viscosities[species_index] == pytest.approx(
            expected, rel=1e-10, abs=0)


@block("transport")
@pytest.mark.parametrize("composition", transport_compositions)
@pytest.mark.parametrize("state", transport_states)
def test_mixture_viscosity_matches_plato(case, state, composition):
    require(case, plato_thermo=True)
    temperature = case.temperature(state)
    mass_fractions = mass_fractions_for(case, composition)
    mole_fractions = mole_fractions_of(case, mass_fractions)
    expected = case.oracle.viscosity(
        plato_number_density(case, temperature, mole_fractions),
        np.atleast_1d(temperature), mole_fractions
    )
    assert case.gas.get_mixture_viscosity(
        temperature, mass_fractions
    ) == pytest.approx(expected, rel=1e-11, abs=0)


@block("transport")
@pytest.mark.parametrize("composition", transport_compositions)
@pytest.mark.parametrize("state", transport_states)
def test_mixture_thermal_conductivities_match_plato(case, state, composition):
    require(case, plato_thermo=True)
    temperature = case.temperature(state)
    mass_fractions = mass_fractions_for(case, composition)
    mole_fractions = mole_fractions_of(case, mass_fractions)
    expected = case.oracle.thermal_conductivities(
        plato_number_density(case, temperature, mole_fractions),
        np.atleast_1d(temperature), mole_fractions
    )
    actual = np.atleast_1d(case.gas.get_mixture_thermal_conductivities(
        temperature, mass_fractions
    ))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)


@block("transport")
@pytest.mark.parametrize("heavy", off_node_heavy_temperatures)
def test_mixture_transport_between_table_nodes(case, heavy):
    require(case, plato_thermo=True)
    temperature = case.temperature((heavy, 0.8 * heavy, 0.8 * heavy))
    mass_fractions = mass_fractions_for(case, "dissociated")
    mole_fractions = mole_fractions_of(case, mass_fractions)
    number_density = plato_number_density(case, temperature, mole_fractions)
    bound = interpolation_bound(case, heavy)
    assert case.gas.get_mixture_viscosity(
        temperature, mass_fractions
    ) == pytest.approx(case.oracle.viscosity(
        number_density, np.atleast_1d(temperature), mole_fractions
    ), rel=bound, abs=0)
    np.testing.assert_allclose(
        np.atleast_1d(case.gas.get_mixture_thermal_conductivities(
            temperature, mass_fractions)),
        case.oracle.thermal_conductivities(
            number_density, np.atleast_1d(temperature), mole_fractions),
        rtol=bound, atol=0
    )


def batched_transport_states(case):
    rng = np.random.default_rng(seed=1)
    num_states = 8
    _, _, step = case.oracle.collision_integral_table_range()
    heavy = 200.0 + step * rng.integers(10, 600, num_states)
    temperature = np.stack([heavy, rng.uniform(0.5, 1.0, num_states) * heavy])
    temperature = temperature[:case.num_temp]
    if case.num_temp == 1:
        temperature = temperature[0]
    mass_fractions = rng.dirichlet(np.ones(case.mech.num_species), num_states).T
    return temperature, mass_fractions


def plato_transport_pointwise(case, temperature, mass_fractions):
    temperature = np.atleast_2d(temperature)
    viscosity, conductivities = [], []
    for k in range(mass_fractions.shape[1]):
        mole_fractions = mole_fractions_of(case, mass_fractions[:, k])
        number_density = plato_number_density(
            case, temperature[:, k], mole_fractions)
        viscosity.append(case.oracle.viscosity(
            number_density, temperature[:, k], mole_fractions))
        conductivities.append(case.oracle.thermal_conductivities(
            number_density, temperature[:, k], mole_fractions))
    return np.array(viscosity), np.stack(conductivities, axis=-1)


@block("transport")
def test_batched_transport_matches_plato_pointwise(case):
    require(case, plato_thermo=True)
    temperature, mass_fractions = batched_transport_states(case)
    viscosity, conductivities = plato_transport_pointwise(
        case, temperature, mass_fractions)
    np.testing.assert_allclose(
        case.gas.get_mixture_viscosity(temperature, mass_fractions),
        viscosity, rtol=1e-11, atol=0)
    np.testing.assert_allclose(
        np.reshape(case.gas.get_mixture_thermal_conductivities(
            temperature, mass_fractions), conductivities.shape),
        conductivities, rtol=1e-11, atol=0)


@block("transport")
@pytest.mark.skipif(jnp is None, reason="jax is not installed")
def test_jax_transport_matches_plato(case):
    require(case, plato_thermo=True)
    temperature, mass_fractions = batched_transport_states(case)
    viscosity, _ = plato_transport_pointwise(case, temperature, mass_fractions)
    gas = make_generated_gas(case.mech, jnp)
    actual = jax.jit(gas.get_mixture_viscosity)(
        jnp.asarray(temperature), jnp.asarray(mass_fractions))
    np.testing.assert_allclose(np.asarray(actual), viscosity, rtol=1e-11, atol=0)

# }}}
