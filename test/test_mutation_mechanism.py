"""Tests for :class:`pyrometheus.bandit.impl.mutation.MutationMechanism`:
mechanism metadata, stoichiometry and participation sets.

Requires the MPP_DATA_DIRECTORY environment variable and an importable
mutationpp. Skipped automatically when either is absent.
"""

import os

import numpy as np
import pytest

try:
    import mutationpp  # noqa: F401
except ImportError:
    mutation_available = False
else:
    mutation_available = os.environ.get("MPP_DATA_DIRECTORY") is not None

pytestmark = pytest.mark.skipif(
    not mutation_available,
    reason="mutationpp not importable or MPP_DATA_DIRECTORY not set",
)

# Imported unconditionally: once mutationpp itself is available, a
# failure to import the adapter is a defect, not a reason to skip.
if mutation_available:
    from pyrometheus.bandit.impl.mutation import MutationMechanism

MIXTURE = "air_5"
STATE_MODEL = "ChemNonEqTTv"


@pytest.fixture(scope="module")
def mech():
    return MutationMechanism(MIXTURE, state_model=STATE_MODEL)


# --- Species metadata ---

def test_num_species(mech):
    assert mech.num_species == 5


def test_species_names(mech):
    assert mech.species_names == ["N", "O", "NO", "N2", "O2"]


def test_num_temperatures_follows_the_state_model(mech):
    assert mech.num_temp == 2


def test_molecular_weights_are_in_kilograms_per_kilomole(mech):
    # Mutation++ reports kg/mol; bandit expressions use kg/kmol.
    assert mech.molecular_weights.shape == (mech.num_species,)
    np.testing.assert_allclose(
        mech.molecular_weights,
        [14.0067, 15.9994, 30.0061, 28.0134, 31.9988],
        rtol=1e-4,
    )


def test_species_index_round_trips(mech):
    for species_index in range(mech.num_species):
        assert mech.species_index(
            mech.species_name(species_index)
        ) == species_index


# --- Reaction metadata ---

def test_num_reactions(mech):
    assert mech.num_reactions == 5


def test_every_reaction_is_reversible(mech):
    for reaction_index in range(mech.num_reactions):
        assert mech.is_reversible(reaction_index)


def test_third_body_reactions_come_first(mech):
    # air5_Park lists three dissociations, then two exchanges.
    assert [
        mech.is_third_body(reaction_index)
        for reaction_index in range(mech.num_reactions)
    ] == [True, True, True, False, False]


def test_third_body_efficiencies_are_keyed_by_species_index(mech):
    efficiencies = mech.third_body_efficiencies(0)
    assert efficiencies
    assert set(efficiencies) <= set(range(mech.num_species))
    assert all(efficiency > 0.0 for efficiency in efficiencies.values())


def test_reactions_without_a_third_body_have_no_efficiencies(mech):
    assert mech.third_body_efficiencies(3) == {}


# --- Stoichiometry ---

def test_dissociation_stoichiometry(mech):
    # N2+M=2N+M: one reactant, one product raised to the second power.
    nitrogen = mech.species_index("N")
    dinitrogen = mech.species_index("N2")
    assert mech.reactants(0) == [dinitrogen]
    assert mech.products(0) == [nitrogen]
    assert mech.stoichiometric_coefficients(0) == ([1], [2])


def test_exchange_stoichiometry(mech):
    # N2+O=NO+N. Mutation++ reports participants in ascending species
    # index order rather than the order the formula writes them.
    assert mech.reactants(3) == sorted(
        [mech.species_index("N2"), mech.species_index("O")]
    )
    assert mech.products(3) == sorted(
        [mech.species_index("NO"), mech.species_index("N")]
    )
    assert mech.stoichiometric_coefficients(3) == ([1, 1], [1, 1])


def test_participants_are_reported_in_ascending_index_order(mech):
    for reaction_index in range(mech.num_reactions):
        assert mech.reactants(reaction_index) == sorted(
            mech.reactants(reaction_index)
        )
        assert mech.products(reaction_index) == sorted(
            mech.products(reaction_index)
        )


def test_repeated_species_are_collapsed_into_a_coefficient(mech):
    for reaction_index in range(mech.num_reactions):
        reactants = mech.reactants(reaction_index)
        products = mech.products(reaction_index)
        assert len(reactants) == len(set(reactants))
        assert len(products) == len(set(products))


def test_stoichiometry_conserves_every_element(mech):
    element_counts = {
        "N": {"N": 1}, "O": {"O": 1}, "NO": {"N": 1, "O": 1},
        "N2": {"N": 2}, "O2": {"O": 2},
    }

    def atoms(indices, coefficients):
        total = {"N": 0, "O": 0}
        for species_index, coefficient in zip(indices, coefficients):
            for element, count in element_counts[
                    mech.species_name(species_index)].items():
                total[element] += coefficient * count
        return total

    for reaction_index in range(mech.num_reactions):
        reactant_coeffs, product_coeffs = mech.stoichiometric_coefficients(
            reaction_index
        )
        assert atoms(mech.reactants(reaction_index), reactant_coeffs) == \
            atoms(mech.products(reaction_index), product_coeffs)


def test_species_indices_are_in_bounds(mech):
    for reaction_index in range(mech.num_reactions):
        for species_index in (mech.reactants(reaction_index)
                              + mech.products(reaction_index)):
            assert 0 <= species_index < mech.num_species


# --- Participation sets ---

def test_participation_set_by_name_and_index_agree(mech):
    for species_index in range(mech.num_species):
        assert mech.participation_set(species_index) == \
            mech.participation_set(mech.species_name(species_index))


def test_participation_set_is_consistent_with_stoichiometry(mech):
    for species_index in range(mech.num_species):
        forward, reverse = mech.participation_set(species_index)
        for reaction_index in forward:
            assert species_index in mech.reactants(reaction_index)
        for reaction_index in reverse:
            assert species_index in mech.products(reaction_index)


def test_production_balance_returns_matching_coefficients(mech):
    for species_index in range(mech.num_species):
        (forward, reverse), (forward_coeffs, reverse_coeffs) = (
            mech.production_balance(species_index)
        )
        assert len(forward) == len(forward_coeffs)
        assert len(reverse) == len(reverse_coeffs)
        assert all(coefficient > 0 for coefficient in forward_coeffs)
        assert all(coefficient > 0 for coefficient in reverse_coeffs)


def test_nitrogen_atom_is_produced_by_dissociation(mech):
    forward, reverse = mech.participation_set("N")
    # N2+M=2N+M produces two N.
    assert 0 in reverse
    (_, _), (_, reverse_coeffs) = mech.production_balance(
        mech.species_index("N")
    )
    assert 2 in reverse_coeffs


# --- Rate coefficients ---

def evaluation_context(mech, temperatures, concentrations):
    return {
        "temperature": np.asarray(temperatures),
        "concentrations": np.asarray(concentrations),
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    }


def mixture_state(mech, temperatures, densities):
    """Put Mutation++ at a state and return the matching bandit-side
    concentrations in kmol/m^3.
    """
    mech.namespace.mix.setState(list(densities), list(temperatures), 1)
    return np.asarray(densities) / mech.molecular_weights


def third_body_concentration(mech, reaction_index, concentrations):
    if not mech.is_third_body(reaction_index):
        return 1.0
    efficiencies = mech.third_body_efficiencies(reaction_index)
    return sum(
        efficiencies.get(species_index, 1.0) * concentrations[species_index]
        for species_index in range(mech.num_species)
    )


@pytest.mark.parametrize("temperatures", [
    [9000.0, 5000.0], [10000.0, 4000.0], [6000.0, 6000.0],
])
def test_forward_rate_coefficients_match_mutation(mech, temperatures):
    from pymbolic import evaluate

    densities = [1e-3, 2e-3, 1.5e-3, 3e-3, 2.5e-3]
    concentrations = mixture_state(mech, temperatures, densities)
    context = evaluation_context(mech, temperatures, concentrations)

    # Mutation++ works in mol and leaves the third body out of the rate
    # constant; bandit works in kmol and folds it in.
    expected = [
        mech.namespace.mix.forwardRateCoefficients()[reaction_index]
        * 1.0e3 ** (mech.reaction(reaction_index).order - 1)
        * third_body_concentration(mech, reaction_index, concentrations)
        for reaction_index in range(mech.num_reactions)
    ]
    actual = [
        evaluate(mech.rate_coeffs[reaction_index].expr, context)
        for reaction_index in range(mech.num_reactions)
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_dissociation_rates_depend_on_both_temperatures(mech):
    # Forward dissociation runs on sqrt(T*Tv): holding T fixed and
    # changing Tv alone must move the coefficient.
    from pymbolic import evaluate

    densities = [1e-3] * 5
    dissociation = mech.rate_coeffs[0].expr
    exchange = mech.rate_coeffs[3].expr
    values = [
        (evaluate(dissociation, evaluation_context(
            mech, temperatures, mixture_state(mech, temperatures, densities))),
         evaluate(exchange, evaluation_context(
             mech, temperatures, mixture_state(mech, temperatures, densities))))
        for temperatures in ([9000.0, 3000.0], [9000.0, 7000.0])
    ]
    assert values[0][0] != values[1][0]
    # Exchange runs on the heavy temperature alone, so it must not.
    assert values[0][1] == pytest.approx(values[1][1])


def test_rate_coefficients_are_built_for_every_reaction(mech):
    assert len(mech.rate_coeffs) == mech.num_reactions
    assert len(mech.mass_action_rates) == mech.num_reactions
    assert len(mech.species_prod_rates) == mech.num_species


def test_ionized_mixtures_are_rejected():
    with pytest.raises(NotImplementedError, match="electron"):
        MutationMechanism("air_11", state_model=STATE_MODEL)
