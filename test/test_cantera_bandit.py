"""Tests for :class:`pyrometheus.bandit.impl.cantera.CanteraMechanism`.

Expressions are checked by evaluating them against the Cantera solution
they were built from, so nothing here needs code generation.
"""

import pathlib

import cantera as ct
import numpy as np
import pymbolic.primitives as p
import pytest
from pymbolic import evaluate, substitute

from pyrometheus.bandit.chem_expr.kinetics import third_body_concentration_expr
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


def test_third_body_concentration_applies_default_efficiency():
    expr = third_body_concentration_expr(3, {1: 2.5}, 0.5)
    value = evaluate(expr, {"concentrations": np.array([2.0, 4.0, 8.0])})
    assert value == pytest.approx(0.5 * 2.0 + 2.5 * 4.0 + 0.5 * 8.0)


def test_third_body_concentration_omits_unit_default_efficiency():
    expr = third_body_concentration_expr(3, {1: 2.5})
    assert "1.0*" not in str(expr)
    value = evaluate(expr, {"concentrations": np.array([2.0, 4.0, 8.0])})
    assert value == pytest.approx(2.0 + 2.5 * 4.0 + 8.0)


@pytest.mark.parametrize("mechname", ["sandiego", "hong", "uconn32"])
def test_three_body_rate_coefficients_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    sol.TPY = 1400.0, ct.one_atm, np.full(sol.n_species, 1 / sol.n_species)
    context = {
        "temperature": sol.T,
        "concentrations": sol.concentrations,
        "exp": np.exp,
        "log": np.log,
    }
    three_body = [
        reaction_index for reaction_index in range(mech.num_reactions)
        if sol.reaction(reaction_index).reaction_type
        == "three-body-Arrhenius"
    ]
    assert three_body
    # Cantera 3 carries the third-body factor in the rate of progress
    # rather than the rate constant; bandit folds it into the constant.
    expected = []
    for reaction_index in three_body:
        third_body = sol.reaction(reaction_index).third_body
        expected.append(sol.forward_rate_constants[reaction_index] * sum(
            third_body.efficiencies.get(
                species_name, third_body.default_efficiency
            ) * concentration
            for species_name, concentration
            in zip(sol.species_names, sol.concentrations)
        ))
    actual = [
        evaluate(mech.rate_coeffs[reaction_index].expr, context)
        for reaction_index in three_body
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


# Placeholders that stand in for a staged intermediate array. None may
# survive into the composed graph the Jacobian is differentiated from.
staged_placeholder_names = {
    "concentrations", "k_fwd", "log_k_eq", "gibbs_rt", "r_net",
    "k_high", "k_low", "reduced_pressure", "falloff_center",
    "falloff_factor", "falloff_function", "falloff_rate_coefficients",
}
composed_graph_names = {
    "density", "temperature", "mass_fractions", "exp", "log", "sqrt",
}


def expression_dependencies(expr):
    from pymbolic.mapper.dependency import DependencyMapper
    if not isinstance(expr, p.ExpressionNode):
        return set()
    return {
        variable.name for variable in DependencyMapper(
            include_subscripts=False, include_calls=False
        )(expr)
    }


def evaluate_composed(composed, density, temperature, mass_fractions):
    context = {
        "density": density,
        "temperature": temperature,
        "mass_fractions": mass_fractions,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    }
    return np.array([
        0.0 if not isinstance(expr, p.ExpressionNode)
        else evaluate(expr, context)
        for expr in composed
    ])


@pytest.mark.parametrize("mechname", ["sandiego", "hong", "uconn32"])
def test_falloff_rate_coefficients_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    sol.TPY = 1400.0, 5 * ct.one_atm, np.full(sol.n_species, 1 / sol.n_species)
    context = {
        "temperature": sol.T,
        "concentrations": sol.concentrations,
        "exp": np.exp,
        "log": np.log,
    }
    falloff_subst = mech._falloff_substitution_map()
    indices = mech.falloff_reaction_indices()
    assert indices
    actual = [
        evaluate(
            substitute(mech.rate_coeffs[reaction_index].expr, falloff_subst),
            context
        )
        for reaction_index in indices
    ]
    np.testing.assert_allclose(
        actual, sol.forward_rate_constants[indices], rtol=1e-11
    )


@pytest.mark.parametrize("mechname", all_mechanisms)
@pytest.mark.parametrize("pressure_atm", [0.1, 1.0, 50.0])
def test_composed_production_rates_match_cantera(mechname, pressure_atm):
    sol, mech = make_mechanism(mechname)
    sol.TPY = (
        1400.0, pressure_atm * ct.one_atm,
        np.full(sol.n_species, 1 / sol.n_species)
    )
    actual = evaluate_composed(
        mech._compose_species_production_rate_graph(),
        sol.density_mass, sol.T, sol.Y
    )
    np.testing.assert_allclose(
        actual, sol.net_production_rates, rtol=1e-9,
        atol=1e-9 * np.abs(sol.net_production_rates).max()
    )


@pytest.mark.parametrize("mechname", ["sandiego", "uconn32"])
def test_composed_graph_resolves_every_staged_placeholder(mechname):
    _, mech = make_mechanism(mechname)
    assert mech.has_falloff_reactions()
    for expr in mech._compose_species_production_rate_graph():
        names = expression_dependencies(expr)
        assert not names & staged_placeholder_names
        assert names <= composed_graph_names


def test_jacobian_resolves_every_staged_placeholder():
    _, mech = make_mechanism("sandiego")
    mech.make_species_production_rate_jacobian()
    for row in mech.species_production_rate_jacobian_exprs:
        for entry in row:
            assert not (
                expression_dependencies(entry) & staged_placeholder_names
            )


def test_jacobian_matches_finite_difference():
    sol, mech = make_mechanism("sandiego")
    sol.TPY = 1400.0, 5 * ct.one_atm, np.full(sol.n_species, 1 / sol.n_species)
    density, temperature, mass_fractions = sol.density_mass, sol.T, sol.Y

    mech.make_species_production_rate_jacobian()
    composed = mech._compose_species_production_rate_graph()
    jacobian = np.array([
        evaluate_composed(row, density, temperature, mass_fractions)
        for row in mech.species_production_rate_jacobian_exprs
    ])
    assert jacobian.shape == (mech.num_species, 2 + mech.num_species)

    def production_rates(density, temperature, mass_fractions):
        return evaluate_composed(
            composed, density, temperature, mass_fractions
        )

    def central_difference(perturb, step):
        return (production_rates(*perturb(step))
                - production_rates(*perturb(-step))) / (2 * step)

    columns = [
        central_difference(
            lambda h: (density + h, temperature, mass_fractions),
            1e-6 * density
        ),
        central_difference(
            lambda h: (density, temperature + h, mass_fractions),
            1e-4 * temperature
        ),
    ]
    for species_index in range(mech.num_species):
        def perturb(step, species_index=species_index):
            perturbed = mass_fractions.copy()
            perturbed[species_index] += step
            return density, temperature, perturbed
        columns.append(central_difference(perturb, 1e-7))

    scale = np.abs(jacobian).max()
    for column_index, expected in enumerate(columns):
        np.testing.assert_allclose(
            jacobian[:, column_index], expected,
            rtol=2e-3, atol=1e-6 * scale
        )


def make_generated_gas(mech):
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
    return PythonBanditCodeGenerator.get_thermochem_class(mech)(np)


def equilibrated_state(sol, pressure_atm=5.0):
    sol.TPY = (
        1400.0, pressure_atm * ct.one_atm,
        np.full(sol.n_species, 1 / sol.n_species)
    )
    return sol.density_mass, sol.T, sol.Y


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_generated_state_conversions_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    gas = make_generated_gas(mech)
    density, temperature, mass_fractions = equilibrated_state(sol)
    assert gas.get_density(
        sol.P, temperature, mass_fractions) == pytest.approx(density, rel=1e-12)
    assert gas.get_pressure(
        density, temperature, mass_fractions) == pytest.approx(sol.P, rel=1e-12)


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_generated_mixture_properties_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    gas = make_generated_gas(mech)
    _, temperature, mass_fractions = equilibrated_state(sol)
    for method, expected in [
        (gas.get_mixture_specific_heat_cp_mass, sol.cp_mass),
        (gas.get_mixture_specific_heat_cv_mass, sol.cv_mass),
        (gas.get_mixture_enthalpy_mass, sol.enthalpy_mass),
        (gas.get_mixture_internal_energy_mass, sol.int_energy_mass),
    ]:
        assert method(temperature, mass_fractions) == pytest.approx(
            expected, rel=1e-11
        )


@pytest.mark.parametrize("mechname", all_mechanisms)
@pytest.mark.parametrize("do_energy", [False, True])
def test_generated_temperature_inversion(mechname, do_energy):
    sol, mech = make_mechanism(mechname)
    gas = make_generated_gas(mech)
    _, temperature, mass_fractions = equilibrated_state(sol)
    target = sol.int_energy_mass if do_energy else sol.enthalpy_mass
    assert gas.get_temperature(
        target, 1000.0, mass_fractions, do_energy
    ) == pytest.approx(temperature, rel=1e-8)


@pytest.mark.parametrize("mechname", all_mechanisms)
def test_generated_production_rates_match_cantera(mechname):
    sol, mech = make_mechanism(mechname)
    gas = make_generated_gas(mech)
    density, temperature, mass_fractions = equilibrated_state(sol)
    actual = gas.get_net_production_rates(
        density, temperature, mass_fractions
    )
    np.testing.assert_allclose(
        actual, sol.net_production_rates, rtol=1e-9,
        atol=1e-9 * np.abs(sol.net_production_rates).max()
    )


def test_generated_species_lookup():
    sol, mech = make_mechanism("sandiego")
    gas = make_generated_gas(mech)
    assert gas.species_names == list(sol.species_names)
    for species_index, species_name in enumerate(sol.species_names):
        assert gas.get_species_index(species_name) == species_index


@pytest.mark.parametrize("slug", ["python-bandit", "fortran-bandit"])
def test_bandit_generators_are_registered(slug):
    from pyrometheus import get_code_generators
    from pyrometheus.bandit.general_thermochem import BaseMechanism
    generator = get_code_generators()[slug]
    mech = generator.load_mechanism(str(mech_dir / "sandiego.yaml"))
    assert isinstance(mech, BaseMechanism)
    assert generator.generate("Thermochemistry", mech)


def test_release_generators_still_load_cantera_solutions():
    from pyrometheus import get_code_generators
    generator = get_code_generators()["python"]
    assert isinstance(
        generator.load_mechanism(str(mech_dir / "uiuc.yaml"), "gas"),
        ct.Solution
    )


def render_sources(mechname, mech):
    from pyrometheus.codegen.fortran_bandit import FortranBanditCodeGenerator
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
    return (
        PythonBanditCodeGenerator.generate("Thermochemistry", mech),
        FortranBanditCodeGenerator.generate(mechname, mech),
    )


@pytest.mark.parametrize("mechname", elementary_mechanisms)
def test_falloff_free_mechanisms_render_no_falloff_block(mechname):
    _, mech = make_mechanism(mechname)
    assert not mech.has_falloff_reactions()
    for source in render_sources(mechname, mech):
        assert "get_falloff_rates" not in source


@pytest.mark.parametrize("mechname", ["sandiego", "uconn32"])
def test_falloff_mechanisms_render_falloff_block(mechname):
    _, mech = make_mechanism(mechname)
    for source in render_sources(mechname, mech):
        assert "get_falloff_rates" in source


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
