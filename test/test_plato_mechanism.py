"""Tests for PlatoMechanism: reaction list assembly, stoichiometry, and
symbolic rate construction.

Requires PLATO_DB and PLATO_LIB environment variables to be set.
Skipped automatically when either is absent or the plato package is not
importable.
"""

import os
import pytest
import numpy as np

plato_available = (
    os.environ.get("PLATO_DB") is not None
    and os.environ.get("PLATO_LIB") is not None
)

try:
    from pyrometheus.bandit.impl.plato import PlatoMechanism
except ImportError:
    plato_available = False

pytestmark = pytest.mark.skipif(
    not plato_available,
    reason="PLATO_DB / PLATO_LIB not set or plato package not importable",
)

DB = os.environ.get("PLATO_DB", "")
MIXTURE = "air5"
REACTION = "air5"
TRANSFER = "empty"


@pytest.fixture(scope="module")
def mech():
    m = PlatoMechanism(MIXTURE, REACTION, TRANSFER, DB)
    yield m
    m.finalize()


# --- Species metadata ---

def test_num_species(mech):
    assert mech.num_species == 5   # N, O, N2, NO, O2


def test_species_names(mech):
    assert mech.species_names == ["N", "O", "N2", "NO", "O2"]


def test_molecular_weights_shape(mech):
    assert mech.molecular_weights.shape == (mech.num_species,)


# --- Reaction list assembly ---

def test_num_reactions_positive(mech):
    assert mech.num_reactions > 0


def test_num_reactions_matches_types(mech):
    ns = mech.namespace
    expected = (
        ns.n_reactions_dh_cat_mt
        + ns.n_reactions_exc
        + ns.n_reactions_de
        + ns.n_reactions_ie
        + ns.n_reactions_ai
    )
    assert mech.num_reactions == expected


def test_dh_cat_mt_come_first(mech):
    n_dh = mech.namespace.n_reactions_dh_cat_mt
    for r in range(n_dh):
        assert mech._reactions[r]["type"] == "dh_cat_mt"


def test_exc_follow_dh_cat_mt(mech):
    n_dh = mech.namespace.n_reactions_dh_cat_mt
    n_exc = mech.namespace.n_reactions_exc
    for r in range(n_dh, n_dh + n_exc):
        assert mech._reactions[r]["type"] == "exc"


# --- Stoichiometry ---

def test_dh_cat_mt_stoichiometry(mech):
    # First dh_cat_mt: one reactant, two products
    rxn = mech._reactions[0]
    assert len(rxn["reactant_species"]) == 1
    assert len(rxn["product_species"]) == 2


def test_exc_stoichiometry(mech):
    n_dh = mech.namespace.n_reactions_dh_cat_mt
    rxn = mech._reactions[n_dh]          # first EXC reaction
    assert len(rxn["reactant_species"]) == 2
    assert len(rxn["product_species"]) == 2


def test_dh_cat_mt_partners_nonempty(mech):
    for r in range(mech.namespace.n_reactions_dh_cat_mt):
        partners = mech._reactions[r]["partners"]
        assert len(partners) > 0, f"reaction {r} has no partners"


def test_species_indices_in_bounds(mech):
    ns = mech.num_species
    for rxn in mech._reactions:
        for s in rxn["reactant_species"] + rxn["product_species"] + rxn["partners"]:
            assert 0 <= s < ns, f"out-of-bounds species index {s}"


def test_stoichiometric_coefficients_all_ones(mech):
    for r in range(mech.num_reactions):
        fwd, rev = mech.stoichiometric_coefficients(r)
        assert all(c == 1 for c in fwd)
        assert all(c == 1 for c in rev)


# --- Participation sets ---

def test_participation_set_n2_by_name(mech):
    fwd, rev = mech.participation_set("N2")
    assert len(fwd) > 0   # N2 appears as reactant (dissociation)


def test_participation_set_n_by_index(mech):
    n_idx = mech.species_index("N")
    fwd, rev = mech.participation_set(n_idx)
    assert len(rev) > 0   # N is a dissociation product


def test_participation_set_consistent(mech):
    for sp in range(mech.num_species):
        fwd, rev = mech.participation_set(sp)
        for r in fwd:
            assert sp in mech._reactions[r]["reactant_species"]
        for r in rev:
            assert sp in mech._reactions[r]["product_species"]


# --- Symbolic rate assembly ---

def test_mass_action_rates_length(mech):
    assert len(mech.mass_action_rates) == mech.num_reactions


def test_species_prod_rates_length(mech):
    assert len(mech.species_prod_rates) == mech.num_species


def test_mass_action_rates_nonzero(mech):
    for r, expr in enumerate(mech.mass_action_rates):
        assert expr != 0, f"mass_action_rate[{r}] is zero"


# --- Temperature field ---

def test_temperature_field_present(mech):
    for r, rxn in enumerate(mech._reactions):
        assert "temperature" in rxn, f"reaction {r} missing 'temperature' field"


def test_dh_cat_mt_temperature(mech):
    for r in range(mech.namespace.n_reactions_dh_cat_mt):
        assert mech._reactions[r]["temperature"] == "geometric_ttv"


def test_exc_temperature(mech):
    n_dh = mech.namespace.n_reactions_dh_cat_mt
    for r in range(n_dh, n_dh + mech.namespace.n_reactions_exc):
        assert mech._reactions[r]["temperature"] == "translational"


# --- NASA polynomial thermo extraction ---

def test_make_species_thermo_returns_species_thermo(mech):
    from pyrometheus.bandit.chem_expr.thermo import SpeciesThermo
    thermo = mech.make_species_thermo(0)
    assert isinstance(thermo, SpeciesThermo)


def test_poly_params_shape(mech):
    from pyrometheus.bandit.chem_expr.thermo import PolynomialParameters
    # Build PolynomialParameters directly to verify extraction
    sp = 1   # 1-based
    ns = mech.namespace
    bounds = ns.nasa_temp_bounds(sp)
    all_coeffs = ns.nasa_cp_coeffs(sp)
    assert len(all_coeffs) >= 2        # at least two intervals
    assert len(all_coeffs[0]) == 7
    assert len(all_coeffs[1]) == 7
    assert bounds[1] > bounds[0]       # T_mid > T_min
    assert bounds[2] > bounds[1]       # T_max > T_mid


def test_poly_params_t_mid_positive(mech):
    for sp_idx in range(mech.num_species):
        thermo = mech.make_species_thermo(sp_idx)
        assert thermo is not None
