"""Parity of the C++ that the Bandit layer generates, for every library
it reads mechanisms from.

Each case's header is compiled with pybind11 and checked two ways:

* against the Python that bandit generates from the same mechanism, for
  every function the two share. The Python is what the library parity
  suites validate, so agreement to round-off carries their coverage
  (and their known gaps) over to C++;
* against the library itself, where the Python is known to agree.

Cases whose library is missing (Plato needs PLATO_DB, Mutation++ needs
MPP_DATA_DIRECTORY) are skipped, as is everything without a C++
compiler and pybind11.
"""

import inspect
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import cantera as ct
import numpy as np
import pytest

from cpp_build import compile_cpp_module, toolchain_available, toolchain_reason
from pyrometheus.codegen import CodeGenerationOptions
from pyrometheus.codegen.cpp_bandit import CppBanditCodeGenerator
from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator

try:
    import plato  # noqa: F401
except ImportError:
    plato_available = False
else:
    plato_available = os.environ.get("PLATO_DB") is not None

try:
    import mutationpp  # noqa: F401
except ImportError:
    mutation_available = False
else:
    mutation_available = os.environ.get("MPP_DATA_DIRECTORY") is not None

pytestmark = pytest.mark.skipif(not toolchain_available, reason=toolchain_reason)

mech_dir = pathlib.Path(__file__).parent / "mechs"


# {{{ cases

@dataclass(frozen=True)
class CppSpec:
    library: str
    label: str
    arguments: tuple
    jacobian: bool = False

    @property
    def available(self):
        return {
            "cantera": True,
            "plato": plato_available,
            "mutation": mutation_available,
        }[self.library]

    def make_mechanism(self):
        if self.library == "cantera":
            from pyrometheus.bandit.impl.cantera import CanteraMechanism
            return CanteraMechanism(str(mech_dir / f"{self.arguments[0]}.yaml"))
        if self.library == "plato":
            from pyrometheus.bandit.impl.plato import PlatoMechanism
            return PlatoMechanism(*self.arguments, os.environ["PLATO_DB"])
        from pyrometheus.bandit.impl.mutation import MutationMechanism
        mixture, state_model = self.arguments
        return MutationMechanism(
            mixture, state_model=state_model, thermo_database="NASA-9"
        )


# uiuc exercises fractional orders, sandiego three-body and Troe
# falloff, uconn32 adds Lindemann. Plato and Mutation++ cover one, two
# and three temperatures, VT transfer, transport and an ionized mixture.
cases = [
    CppSpec("cantera", "uiuc", ("uiuc",)),
    CppSpec("cantera", "sandiego", ("sandiego",), jacobian=True),
    CppSpec("cantera", "uconn32", ("uconn32",)),
    CppSpec("plato", "air5-empty", ("air5", "air5", "empty")),
    CppSpec("plato", "air5-TTv", ("air5", "air5", "TTv"), jacobian=True),
    CppSpec("plato", "air11-empty", ("air11", "air11_park93", "empty")),
    CppSpec("plato", "air11-TTv", ("air11", "air11_park93", "TTv")),
    CppSpec("plato", "air11-TTvTe", ("air11", "air11_park93", "TTvTe")),
    CppSpec("mutation", "air_5-1T", ("air_5", "ChemNonEq1T")),
    CppSpec("mutation", "air_5-TTv", ("air_5", "ChemNonEqTTv"), jacobian=True),
    CppSpec("mutation", "air_11-1T", ("air_11", "ChemNonEq1T")),
    CppSpec("mutation", "air_11-TTv", ("air_11", "ChemNonEqTTv"), jacobian=True),
]


@dataclass
class CppCase:
    spec: CppSpec
    mech: object
    cpp: object
    gas: object
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    @property
    def num_temp(self):
        return self.mech.num_temp

    def temperature(self, state):
        """:returns: The temperature argument for a (Th, Tv, Te) *state*,
            truncated to the temperatures this case carries.
        """
        if self.num_temp == 1:
            return state[0]
        return np.array(state[:self.num_temp])

    def mass_fractions(self, seed=0):
        """:returns: A composition in which every species takes part.
            Electrons, when present, balance the ions' charge: a random
            share would make the gas mostly electrons by count and drive
            rates to where finite differences lose every digit.
        """
        rng = np.random.default_rng(seed)
        mass_fractions = rng.uniform(0.05, 1.0, self.mech.num_species)
        names = self.mech.species_names
        electrons = [k for k, name in enumerate(names) if name in ("e-", "em")]
        if electrons:
            weights = np.asarray(self.mech.molecular_weights)
            ions = [k for k, name in enumerate(names)
                    if name.endswith("+") or (name.endswith("p")
                                              and name != "em")]
            mass_fractions[electrons[0]] = weights[electrons[0]] * sum(
                mass_fractions[k] / weights[k] for k in ions
            )
        return mass_fractions / mass_fractions.sum()


def make_python_gas(mech):
    base_class = PythonBanditCodeGenerator.get_thermochem_class(mech)

    class StackedGas(base_class):
        # The default object-array assembly cannot hold the matrices the
        # transport solves take.
        def _pyro_make_array(self, res_list):
            return self.pyro_np.stack([
                self.pyro_np.asarray(entry, dtype=self.pyro_np.float64)
                for entry in res_list
            ])

    return StackedGas(np)


# Plato holds one live instance per process; pytest closes a
# module-scoped parametrized fixture before opening its next parameter.
@pytest.fixture(scope="module", params=cases, ids=lambda spec: spec.label)
def case(request, tmp_path_factory):
    spec = request.param
    if not spec.available:
        pytest.skip(f"{spec.library} is not available")
    mech = spec.make_mechanism()
    struct_name = spec.label.replace("-", "_").lower()
    header = CppBanditCodeGenerator.generate(
        struct_name, mech, CodeGenerationOptions(compute_jacobian=spec.jacobian)
    )
    cpp = compile_cpp_module(
        tmp_path_factory.mktemp(struct_name), header, struct_name,
        f"libpyro_cpp_{struct_name}"
    )
    yield CppCase(spec=spec, mech=mech, cpp=cpp, gas=make_python_gas(mech))
    if spec.library == "plato":
        mech.finalize()


def require(case, library=None, num_temp=None, ionized=None):
    if library is not None and case.spec.library != library:
        pytest.skip(f"checks {library} only")
    if num_temp is not None and case.num_temp != num_temp:
        pytest.skip(f"needs {num_temp} temperature(s)")
    if ionized is not None and ("em" in case.mech.species_names) != ionized:
        pytest.skip("ionized mixtures only" if ionized else "neutral mixtures only")

# }}}


# {{{ states

# (Th, Tv, Te): thermal equilibrium, vibrationally frozen (post-shock)
# and vibrationally hot (expansion).
states = [
    (1400.0, 1400.0, 1400.0),
    (2500.0, 2500.0, 2500.0),
    (9000.0, 3000.0, 12000.0),
    (4000.0, 8000.0, 5000.0),
]


def states_for(case):
    # Cantera's NASA-7 fits stop at a few thousand kelvin.
    if case.spec.library == "cantera":
        return states[:2]
    return states

# }}}


# {{{ C++ against the generated Python

def test_cpp_bandit_generator_is_registered(tmp_path):
    from pyrometheus import get_code_generators
    generator = get_code_generators()["cpp-bandit"]
    assert generator is CppBanditCodeGenerator
    header = generator.generate(
        "uiuc", generator.load_mechanism(str(mech_dir / "uiuc.yaml")))
    assert "struct uiuc" in header


# Every function the two backends share, by family. Arguments are
# filled from the Python signature.
shared_functions = [
    "get_mixture_molecular_weight", "get_mole_fractions",
    "get_concentrations", "get_density", "get_pressure",
    "get_species_specific_heats_cp_r", "get_species_specific_heats_cv_r",
    "get_species_enthalpies_rt", "get_species_internal_energies_rt",
    "get_species_gibbs_rt", "get_mixture_specific_heat_cp_mass",
    "get_mixture_specific_heat_cv_mass", "get_mixture_enthalpy_mass",
    "get_mixture_internal_energy_mass", "get_equilibrium_constants",
    "get_fwd_rate_coefficients", "get_net_rates_of_progress",
    "get_net_production_rates",
    "get_species_vibrational_energies",
    "get_species_vibrational_specific_heats",
    "get_pressure_relaxation_times", "get_vt_energy_transfer_source",
    "get_translational_rotational_energy",
    "get_nasa_polynomial_vibrational_energy",
    "get_nasa_polynomial_vibrational_specific_heat",
    "get_species_binary_mass_diffusivities",
    "get_species_effective_diffusivities", "get_species_viscosities",
    "get_mixture_viscosity", "get_mixture_thermal_conductivities",
]

# The NASA vibrational split subtracts the translational-rotational
# energy from the full NASA energy. For the electron both are ~1e11 J/kg
# and the remainder ~0, so round-off is judged against the terms.
cancellation_rtol = {
    "get_nasa_polynomial_vibrational_energy": 1e-10,
    "get_nasa_polynomial_vibrational_specific_heat": 1e-10,
}


def call_arguments(case, function, state, mass_fractions):
    density = 5e-2
    values = {
        "density": density,
        "temperature": case.temperature(state),
        "mass_fractions": mass_fractions,
        "pressure": 1e5,
        "concentrations": case.gas.get_concentrations(density, mass_fractions),
    }
    # Equation-of-state conversions take the heavy temperature alone.
    if function in ("get_density", "get_pressure"):
        values["temperature"] = state[0]
    names = inspect.signature(getattr(case.gas, function)).parameters
    return [values[name] for name in names]


@pytest.mark.parametrize("state", states, ids=lambda s: "-".join(map(str, map(int, s))))
@pytest.mark.parametrize("function", shared_functions)
def test_cpp_matches_generated_python(case, function, state):
    if state not in states_for(case):
        pytest.skip("outside this library's fit range")
    if not hasattr(case.gas, function):
        assert not hasattr(case.cpp, function)
        pytest.skip(f"{case.spec.label} does not generate {function}")
    arguments = call_arguments(case, function, state, case.mass_fractions())
    expected = np.asarray(getattr(case.gas, function)(*arguments), dtype=float)
    actual = np.asarray(getattr(case.cpp, function)(*arguments))
    assert actual.shape == expected.shape
    rtol = cancellation_rtol.get(function, 1e-12)
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * np.abs(expected).max()
    )


def test_cpp_generates_what_python_generates(case):
    cpp_functions = {name for name in dir(case.cpp) if name.startswith("get_")}
    python_functions = {
        name for name in dir(case.gas)
        if name.startswith("get_") and name != "get_species_index"
    }
    assert python_functions <= cpp_functions


def test_cpp_constants_match_generated_python(case):
    assert case.cpp.num_species == case.gas.num_species
    assert case.cpp.num_reactions == case.gas.num_reactions
    assert case.cpp.num_temperatures == case.gas.num_temperatures
    assert case.cpp.gas_constant == case.gas.gas_constant
    assert case.cpp.one_atm == case.gas.one_atm
    np.testing.assert_array_equal(case.cpp.molecular_weights, case.gas.molecular_weights)


def test_cpp_species_names_round_trip(case):
    for species_index, species_name in enumerate(case.gas.species_names):
        assert case.cpp.get_species_name(species_index) == species_name
        assert case.cpp.get_species_index(species_name) == species_index
    assert case.cpp.get_species_index("not-a-species") == -1


@pytest.mark.parametrize("do_energy", [False, True])
def test_cpp_temperature_inversion_round_trips(case, do_energy):
    require(case, num_temp=1)
    mass_fractions = case.mass_fractions()
    temperature = 1800.0
    energy = (
        case.cpp.get_mixture_internal_energy_mass if do_energy
        else case.cpp.get_mixture_enthalpy_mass
    )(temperature, mass_fractions)
    assert case.cpp.get_temperature(
        energy, 1200.0, mass_fractions, do_energy
    ) == pytest.approx(temperature, rel=1e-9)


def test_cpp_jacobian_matches_finite_difference(case):
    if not case.spec.jacobian:
        pytest.skip("the Jacobian is generated for a subset of cases")
    state = states_for(case)[-1]
    density = 5e-2
    temperature = np.atleast_1d(case.temperature(state)).astype(float)
    mass_fractions = case.mass_fractions()
    variables = np.concatenate([[density], temperature, mass_fractions])
    num_temp = case.num_temp

    def production_rates(variables):
        temperature = variables[1:1 + num_temp]
        return np.asarray(case.cpp.get_net_production_rates(
            variables[0], temperature if num_temp > 1 else temperature[0],
            variables[1 + num_temp:]
        ))

    jacobian = np.asarray(case.cpp.get_net_production_rates_jacobian(
        density, case.temperature(state), mass_fractions
    ))
    assert jacobian.shape == (case.mech.num_species, variables.size)
    for column, value in enumerate(variables):
        step = 1e-6 * value
        forward, backward = variables.copy(), variables.copy()
        forward[column] += step
        backward[column] -= step
        finite_difference = (
            production_rates(forward) - production_rates(backward)
        ) / (2 * step)
        np.testing.assert_allclose(
            jacobian[:, column], finite_difference, rtol=1e-5,
            atol=1e-5 * np.abs(finite_difference).max() + 1e-300,
            err_msg=f"column {column}"
        )

# }}}


# {{{ C++ against Cantera

def cantera_solution(case, temperature, mass_fractions, pressure=5 * ct.one_atm):
    solution = ct.Solution(str(mech_dir / f"{case.spec.arguments[0]}.yaml"))
    solution.TPY = temperature, pressure, mass_fractions
    return solution


@pytest.mark.parametrize("temperature", [300.0, 1000.0, 1400.0, 2500.0])
def test_cpp_species_thermo_matches_cantera(case, temperature):
    require(case, library="cantera")
    # Cantera's standard_gibbs_RT moves with pressure as ln(p / p0).
    solution = cantera_solution(
        case, temperature, case.mass_fractions(), pressure=ct.one_atm)
    for function, expected in [
        ("get_species_specific_heats_cp_r", solution.standard_cp_R),
        ("get_species_enthalpies_rt", solution.standard_enthalpies_RT),
        ("get_species_gibbs_rt", solution.standard_gibbs_RT),
    ]:
        np.testing.assert_allclose(
            getattr(case.cpp, function)(temperature), expected,
            rtol=1e-12, atol=1e-12, err_msg=function
        )


@pytest.mark.parametrize("temperature", [1400.0, 2500.0])
def test_cpp_mixture_thermo_matches_cantera(case, temperature):
    require(case, library="cantera")
    mass_fractions = case.mass_fractions()
    solution = cantera_solution(case, temperature, mass_fractions)
    for function, expected in [
        ("get_mixture_specific_heat_cp_mass", solution.cp_mass),
        ("get_mixture_specific_heat_cv_mass", solution.cv_mass),
        ("get_mixture_enthalpy_mass", solution.enthalpy_mass),
        ("get_mixture_internal_energy_mass", solution.int_energy_mass),
    ]:
        assert getattr(case.cpp, function)(temperature, mass_fractions) == \
            pytest.approx(expected, rel=1e-11), function
    assert case.cpp.get_density(solution.P, temperature, mass_fractions) == \
        pytest.approx(solution.density_mass, rel=1e-12)
    assert case.cpp.get_pressure(solution.density_mass, temperature, mass_fractions) == \
        pytest.approx(solution.P, rel=1e-12)


@pytest.mark.parametrize("temperature", [1400.0, 2500.0])
def test_cpp_kinetics_matches_cantera(case, temperature):
    require(case, library="cantera")
    mass_fractions = case.mass_fractions()
    solution = cantera_solution(case, temperature, mass_fractions)
    reversible = [
        reaction_index for reaction_index in range(solution.n_reactions)
        if solution.reaction(reaction_index).reversible
    ]
    np.testing.assert_allclose(
        np.asarray(case.cpp.get_equilibrium_constants(temperature))[reversible],
        -np.log(solution.equilibrium_constants[reversible]), rtol=1e-11
    )
    concentrations = case.cpp.get_concentrations(solution.density_mass, mass_fractions)
    expected = solution.net_rates_of_progress
    np.testing.assert_allclose(
        case.cpp.get_net_rates_of_progress(temperature, concentrations), expected,
        rtol=1e-9, atol=1e-9 * np.abs(expected).max()
    )
    expected = solution.net_production_rates
    np.testing.assert_allclose(
        case.cpp.get_net_production_rates(
            solution.density_mass, temperature, mass_fractions),
        expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max()
    )

# }}}


# {{{ C++ against Plato

# Th on Plato's collision-integral table nodes (200 K + k 50 K), where
# its interpolated tables equal the fits bandit generates.
transport_states = [
    (300.0, 300.0, 300.0), (5050.0, 3000.0, 3000.0), (15000.0, 8000.0, 8000.0),
]


def plato_oracle(case):
    return case.mech.namespace.thermochem


def plato_number_density(case, temperature, mole_fractions, pressure):
    return plato_oracle(case).number_density(
        pressure, np.atleast_1d(temperature), mole_fractions
    )


@pytest.mark.parametrize("state", transport_states)
def test_cpp_transport_matches_plato(case, state):
    require(case, library="plato")
    if not case.mech.has_transport():
        pytest.skip("no transport generated for this mixture")
    oracle = plato_oracle(case)
    temperature = case.temperature(state)
    plato_temperature = np.atleast_1d(temperature)
    mass_fractions = case.mass_fractions()
    mole_fractions = np.asarray(case.cpp.get_mole_fractions(mass_fractions))
    pressure = 1e5
    number_density = plato_number_density(case, temperature, mole_fractions, pressure)

    np.testing.assert_allclose(
        np.asarray(case.cpp.get_species_binary_mass_diffusivities(temperature)) / pressure,
        oracle.binary_diffusion_coefficients(
            number_density, plato_temperature, mole_fractions),
        rtol=1e-12, atol=0
    )
    assert case.cpp.get_mixture_viscosity(temperature, mass_fractions) == pytest.approx(
        oracle.viscosity(number_density, plato_temperature, mole_fractions),
        rel=1e-11, abs=0)
    np.testing.assert_allclose(
        np.atleast_1d(case.cpp.get_mixture_thermal_conductivities(
            temperature, mass_fractions)),
        oracle.thermal_conductivities(number_density, plato_temperature, mole_fractions),
        rtol=1e-11, atol=0
    )
    np.testing.assert_allclose(
        case.cpp.get_species_effective_diffusivities(pressure, temperature, mass_fractions),
        oracle.effective_diffusion_coefficients(
            number_density, plato_temperature, mole_fractions),
        rtol=1e-11, atol=0
    )


@pytest.mark.parametrize("state", states, ids=lambda s: "-".join(map(str, map(int, s))))
def test_cpp_production_rates_match_plato(case, state):
    # Ionized kinetics is a known gap of the bandit port itself; see
    # test_plato_parity.known_gaps.
    require(case, library="plato", ionized=False)
    density = 5e-2
    temperature = case.temperature(state)
    mass_fractions = case.mass_fractions()
    expected = plato_oracle(case).get_production_rates(
        density * mass_fractions, np.atleast_1d(temperature)
    )
    actual = np.asarray(case.cpp.get_net_production_rates(
        density, temperature, mass_fractions)) * case.cpp.molecular_weights
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max()
    )


@pytest.mark.parametrize("heavy", [500.0, 8000.0, 20000.0])
def test_cpp_relaxation_times_match_plato(case, heavy):
    require(case, library="plato")
    if case.num_temp == 1 or not plato_oracle(case).vt_active:
        pytest.skip("no VT transfer in this case")
    # Plato raises Th to -0.3333333333 rather than -1/3 in
    # Millikan-White, which moves p tau by up to ~1e-8.
    np.testing.assert_allclose(
        case.cpp.get_pressure_relaxation_times(case.temperature((heavy, 1000.0, 1000.0))),
        plato_oracle(case).get_ptau_vt(heavy), rtol=2e-8
    )

# }}}


# {{{ C++ against Mutation++

def mutation_state(case, temperature, densities):
    """Put Mutation++ at a state and return the matching concentrations
    in kmol/m^3.
    """
    temperatures = list(np.atleast_1d(temperature))
    case.mech.namespace.mix.setState(list(densities), temperatures, 1)
    return np.asarray(densities) / case.mech.molecular_weights


@pytest.mark.parametrize("state", states[2:], ids=lambda s: "-".join(map(str, map(int, s))))
def test_cpp_forward_rate_coefficients_match_mutation(case, state):
    require(case, library="mutation")
    temperature = case.temperature(state)
    densities = 5e-2 * case.mass_fractions()
    concentrations = mutation_state(case, temperature, densities)
    mech = case.mech
    # Mutation++ works in mol and leaves the third body out of the rate
    # constant; bandit works in kmol and folds it in.
    expected = []
    for reaction_index in range(mech.num_reactions):
        third_body = 1.0
        if mech.is_third_body(reaction_index):
            efficiencies = mech.third_body_efficiencies(reaction_index)
            third_body = sum(
                efficiencies.get(species_index, 1.0) * concentrations[species_index]
                for species_index in range(mech.num_species)
            )
        expected.append(
            mech.namespace.mix.forwardRateCoefficients()[reaction_index]
            * 1.0e3 ** (mech.reaction(reaction_index).order - 1) * third_body
        )
    np.testing.assert_allclose(
        case.cpp.get_fwd_rate_coefficients(temperature, concentrations),
        expected, rtol=1e-12
    )


@pytest.mark.parametrize("temperature", [800.0, 4000.0, 12000.0])
def test_cpp_species_thermo_matches_mutation(case, temperature):
    require(case, library="mutation")
    mix = case.mech.namespace.mix
    mutation_state(case, (temperature,) * case.num_temp, [1e-3] * case.mech.num_species)
    np.testing.assert_allclose(
        case.cpp.get_species_specific_heats_cp_r(case.temperature((temperature,) * 3)),
        mix.speciesCpOverR(temperature), rtol=1e-10
    )
    np.testing.assert_allclose(
        case.cpp.get_species_enthalpies_rt(case.temperature((temperature,) * 3)),
        mix.speciesHOverRT(), rtol=1e-10
    )

# }}}
