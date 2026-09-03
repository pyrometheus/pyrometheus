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


def evaluate_polynomials(polynomials, temperature):
    context = {"temperature": temperature, "log": np.log, "exp": np.exp}
    return np.array([
        evaluate(polynomial.expr, context) for polynomial in polynomials
    ])


@pytest.mark.parametrize("mechname", all_mechanisms)
@pytest.mark.parametrize("temperature", [300.0, 800.0, 1200.0, 2500.0])
def test_species_nasa_thermo_matches_cantera(mechname, temperature):
    sol, mech = make_mechanism(mechname)
    sol.TP = temperature, ct.one_atm
    species_thermo = [
        mech.make_species_nasa_thermo(species_index)
        for species_index in range(mech.num_species)
    ]
    for polynomials, expected in [
        ([t.cp_poly for t in species_thermo], sol.standard_cp_R),
        ([t.enthalpy_poly for t in species_thermo], sol.standard_enthalpies_RT),
        ([t.entropy_poly for t in species_thermo], sol.standard_entropies_R),
        ([t.gibbs_poly for t in species_thermo], sol.standard_gibbs_RT),
    ]:
        np.testing.assert_allclose(
            evaluate_polynomials(polynomials, temperature),
            expected, rtol=1e-12
        )


def test_unsupported_thermo_model_raises():
    _, mech = make_mechanism("uiuc")
    species = mech.species(0)
    species.thermo = ct.ConstantCp(
        species.thermo.min_temp, species.thermo.max_temp,
        species.thermo.reference_pressure, [300.0, 0.0, 0.0, 4.0e4]
    )
    with pytest.raises(NotImplementedError, match="ConstantCp"):
        mech.make_species_nasa_thermo(0)


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_thermo_is_built_on_construction(mechname):
    _, mech = make_mechanism(mechname)
    assert len(mech.species_nasa_thermo_polynomials) == mech.num_species
    assert len(mech.equil_constants) == mech.num_reactions


@pytest.mark.parametrize("mechname", all_mechanisms)
@pytest.mark.parametrize("temperature", [800.0, 1500.0, 2500.0])
def test_equilibrium_constants_match_cantera(mechname, temperature):
    sol, mech = make_mechanism(mechname)
    sol.TP = temperature, ct.one_atm
    context = {
        "temperature": temperature,
        "gibbs_rt": sol.standard_gibbs_RT,
        "log": np.log,
        "exp": np.exp,
    }
    reversible = [
        reaction_index for reaction_index in range(mech.num_reactions)
        if mech.is_reversible(reaction_index)
    ]
    actual = np.array([
        evaluate(mech.equil_constants[reaction_index], context)
        for reaction_index in reversible
    ])
    np.testing.assert_allclose(
        actual, -np.log(sol.equilibrium_constants[reversible]), rtol=1e-11
    )


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
