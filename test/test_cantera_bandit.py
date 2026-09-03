"""Tests for :class:`pyrometheus.bandit.impl.cantera.CanteraMechanism`.

Expressions are checked by evaluating them against the Cantera solution
they were built from, so nothing here needs code generation.
"""

import pathlib

import cantera as ct
import numpy as np
import pytest
from pymbolic import evaluate

from pyrometheus.bandit.impl.cantera import CanteraMechanism


mech_dir = pathlib.Path(__file__).parent / "mechs"

# Mechanisms whose reactions are all plain Arrhenius, so their rates can
# be compared against Cantera before three-body and falloff land.
elementary_mechanisms = ["uiuc", "bfer"]
all_mechanisms = elementary_mechanisms + ["sandiego", "hong", "uconn32"]


def make_mechanism(mechname):
    path = str(mech_dir / f"{mechname}.yaml")
    return ct.Solution(path), CanteraMechanism(path)


def evaluation_context(sol):
    return {
        "concentrations": sol.concentrations,
        "k_fwd": sol.forward_rate_constants,
        # The generated equilibrium constant is used as
        # r_fwd - exp(log_k_eq) * r_rev, so it is the reciprocal of
        # Cantera's concentration equilibrium constant.
        "log_k_eq": -np.log(sol.equilibrium_constants),
        "exp": np.exp,
        "log": np.log,
    }


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_stoichiometric_coefficients_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    for reaction_index in range(mech.num_reactions):
        reactant_coefficients, product_coefficients = (
            mech.stoichiometric_coefficients(reaction_index)
        )
        assert reactant_coefficients == [
            sol.reactant_stoich_coeff(species_index, reaction_index)
            for species_index in mech.reactants(reaction_index)
        ]
        assert product_coefficients == [
            sol.product_stoich_coeff(species_index, reaction_index)
            for species_index in mech.products(reaction_index)
        ]


def test_fractional_stoichiometry_is_preserved():
    _, mech = make_mechanism("uiuc")
    # H2 + 0.5 O2 <=> H2O -- truncating 0.5 to 0 drops O2 entirely.
    reactant_coefficients, _ = mech.stoichiometric_coefficients(1)
    assert 0.5 in reactant_coefficients


def test_declared_reaction_orders_are_preserved():
    _, mech = make_mechanism("uiuc")
    # C2H4 + O2 => 2 CO + 2 H2, with declared orders 0.5 and 0.65.
    assert mech.reaction_orders(0) == [0.5, 0.65]


def test_reaction_orders_default_to_stoichiometry():
    _, mech = make_mechanism("sandiego")
    for reaction_index in range(mech.num_reactions):
        assert (mech.reaction_orders(reaction_index)
                == mech.stoichiometric_coefficients(reaction_index)[0])


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_integral_coefficients_stay_integers(mechname):
    _, mech = make_mechanism(mechname)
    for reaction_index in range(mech.num_reactions):
        for coefficients in mech.stoichiometric_coefficients(reaction_index):
            for coefficient in coefficients:
                if float(coefficient).is_integer():
                    assert isinstance(coefficient, int)


@pytest.mark.parametrize("mechname", elementary_mechanisms)
def test_net_rates_of_progress_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    sol.TPY = 1200.0, ct.one_atm, np.full(sol.n_species, 1 / sol.n_species)
    context = evaluation_context(sol)
    rates = np.array([
        evaluate(expr, context) for expr in mech.mass_action_rates
    ])
    np.testing.assert_allclose(
        rates, sol.net_rates_of_progress, rtol=1e-10
    )
