__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cantera as ct
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jnp = None
    numpy_list = [np]
else:
    jax.config.update("jax_enable_x64", True)
    numpy_list = [np, jnp]

from backends import PythonBackend, pyro_init, BACKENDS


@pytest.mark.parametrize("mechname", [
    "uiuc", "sandiego", "uconn32", "hong"
])
@pytest.mark.parametrize("user_np", numpy_list)
def test_get_rate_coefficients(mechname: str, user_np,
request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes the rate coefficients matching Cantera
    for given temperature and composition"""
    sol, ptk = pyro_init(mechname, user_np, request)
    three_body_reactions = [(i, r) for i, r in enumerate(sol.reactions())
                            if r.reaction_type == "three-body-Arrhenius"]
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:
        # Set new temperature in Cantera
        sol.TPY = t, ct.one_atm, (1/sol.n_species) * np.ones(
            (sol.n_species,)
        )
        # Concentrations
        y = sol.Y
        rho = sol.density
        c = ptk.get_concentrations(rho, y)
        # Get rate coefficients and compare
        k_ct = sol.forward_rate_constants
        k_pm = ptk.get_fwd_rate_coefficients(t, c)
        # It seems like Cantera 3.0 has bumped the third-body efficiency
        # factor from the rate coefficient to the rate of progress
        for i, r in three_body_reactions:
            eff = np.sum([c[sol.species_index(sp)]
                          * r.third_body.efficiencies[sp]
                          for sp in r.third_body.efficiencies])
            eff += np.sum([c[sp]
                           * r.third_body.default_efficiency
                           for sp in range(sol.n_species)
                           if sol.species_name(sp) not in
                           r.third_body.efficiencies])
            k_ct[i] *= eff
        print(k_ct)
        print()
        print(k_pm)
        print()
        print(np.abs((k_ct-k_pm)/k_ct))
        print(np.linalg.norm((k_ct-k_pm)/k_ct, np.inf))
        assert np.linalg.norm((k_ct-k_pm)/k_ct, np.inf) < 1e-13


@pytest.mark.parametrize("mechname", [
    "uiuc", "sandiego", "uconn32", "gri30", "hong"
])
@pytest.mark.parametrize("user_np", numpy_list)
def test_get_pressure(mechname: str, user_np, request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted pressure for given density,
    temperature, and mass fractions
    """
    # Create Cantera and pyrometheus objects
    sol, ptk = pyro_init(mechname, user_np, request)

    # Temperature, equivalence ratio, oxidizer ratio, stoichiometry ratio
    t = 300.0
    phi = 2.0
    alpha = 0.21
    nu = 0.5
    # Species mass fractions
    i_fu = ptk.get_species_index("H2")
    i_ox = ptk.get_species_index("O2")
    i_di = ptk.get_species_index("N2")
    x = np.zeros(ptk.num_species)
    x[i_fu] = (alpha * phi) / (nu + alpha * phi)
    x[i_ox] = nu * x[i_fu] / phi
    x[i_di] = (1.0 - alpha) * x[i_ox] / alpha
    # Get equilibrium composition
    sol.TPX = t, ct.one_atm, x
    sol.equilibrate("UV")
    t, rho, y = sol.TDY
    p_ct = sol.P
    # Compute pressure with pyrometheus and compare to Cantera
    p_pm = ptk.get_pressure(rho, t, y)
    assert abs(p_ct - p_pm) / p_ct < 1.0e-12


@pytest.mark.parametrize("mechname", [
    "uiuc", "sandiego", "uconn32", "gri30", "hong"
])
@pytest.mark.parametrize("user_np", numpy_list)
def test_get_thermo_properties(mechname: str, user_np,
request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_r, h_rt, and k_eq
    correctly by comparing against Cantera"""
    # Create Cantera and pyrometheus objects
    sol, ptk = pyro_init(mechname, user_np, request)

    # Loop over temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:
        # Set state in cantera for comparison
        sol.TP = t, ct.one_atm
        # Get properties from pyrometheus and compare to Cantera
        cp_pm = ptk.get_species_specific_heats_r(t)
        cp_err = np.linalg.norm(cp_pm - sol.standard_cp_R, np.inf)
        print(f"cp_pm = {cp_pm}")
        print(f"cnt_cp = {sol.standard_cp_R}")
        assert cp_err < 1.0e-13
        s_pm = ptk.get_species_entropies_r(t)
        s_err = np.linalg.norm(s_pm - sol.standard_entropies_R, np.inf)
        print(f"s_pm = {s_pm}")
        print(f"cnt_s = {sol.standard_entropies_R}")
        assert s_err < 1.0e-13
        h_pm = ptk.get_species_enthalpies_rt(t)
        h_err = np.linalg.norm(h_pm - sol.standard_enthalpies_RT, np.inf)
        print(f"h_pm = {h_pm}")
        print(f"cnt_h = {sol.standard_enthalpies_RT}")
        assert h_err < 1.0e-13
        keq_pm1 = ptk.get_equilibrium_constants(t)
        print(f"keq1 = {keq_pm1}")
        keq_pm = 1.0 / np.exp(ptk.get_equilibrium_constants(t))
        keq_ct = sol.equilibrium_constants
        print(f"keq_pm = {keq_pm}")
        print(f"keq_cnt = {keq_ct}")
        print(f"temperature = {t}")
        # exclude meaningless check on equilibrium constants for irreversible
        # reaction
        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:
                keq_err = np.abs((keq_pm[i] - keq_ct[i]) / keq_ct[i])
                print(f"i = {i}, keq_err = {keq_err}")
                assert keq_err < 1.0e-13
        # keq_pm_test = keq_pm[1:]
        # keq_ct_test = keq_ct[1:]
        # keq_err = np.linalg.norm((keq_pm_test - keq_ct_test) / keq_ct_test, np.inf)
        # assert keq_err < 1.0e-13


@pytest.mark.parametrize("mechname", [
    "uiuc", "sandiego", "gri30", "hong"
])
@pytest.mark.parametrize("user_np", numpy_list)
def test_get_temperature(mechname: str, user_np, request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted temperature for given internal energy
    and mass fractions"""
    # Create Cantera and pyrometheus objects
    sol, ptk = pyro_init(mechname, user_np, request)

    tol = 1.0e-10
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    # First test individual species
    y = np.zeros(ptk.num_species)
    for sp in range(ptk.num_species):
        y[sp] = 1.0
        for t in temp:
            sol.TPY = t, ct.one_atm, y
            e = sol.int_energy_mass
            t_guess = 0.9 * t
            t_pm = ptk.get_temperature(e, t_guess, y, True)
            assert np.abs(t - t_pm) < tol
        y[sp] = 0.0
    # Now test a mixture with fully-populated composition
    # All mass fractions set to the same value for now,
    # though a more representative test would be ignition composition
    y = np.ones(ptk.num_species) / ptk.num_species
    for t in temp:
        sol.TPY = t, ct.one_atm, y
        e = sol.int_energy_mass
        t_guess = 0.9 * t
        t_pm = ptk.get_temperature(e, t_guess, y, True)
        assert np.abs(t - t_pm) < tol


@pytest.mark.parametrize(
    "mechname, fuel, stoich_ratio, dt",
    [("uiuc", "C2H4", 3.0, 1e-7),
     ("sandiego", "H2", 0.5, 1e-6),
     ("hong", "H2", 0.5, 1e-6)]
)
@pytest.mark.parametrize("user_np", numpy_list)
def test_kinetics(mechname: str, fuel: str, stoich_ratio: float, dt: float,
user_np, request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted rates of progress for given
    temperature and composition"""

    sol, ptk = pyro_init(mechname, user_np, request)

    # Homogeneous reactor to get test data
    init_temperature = 1500.0
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")
    x = np.zeros(ptk.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Init Cantera reactor
    sol.TPX = init_temperature, ct.one_atm, x
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])
    time = 0.0
    for _ in range(100):
        time += dt
        sim.advance(time)
        # Cantera kinetics
        r_ct = reactor.kinetics.net_rates_of_progress
        omega_ct = reactor.kinetics.net_production_rates
        # Get state from Cantera
        temp = reactor.T
        rho = reactor.density
        y = np.where(reactor.Y > 0, reactor.Y, 0)
        # Prometheus kinetics
        c = ptk.get_concentrations(rho, y)
        r_pm = ptk.get_net_rates_of_progress(temp, c)
        omega_pm = ptk.get_net_production_rates(rho, temp, y)
        err_r = np.linalg.norm(r_ct-r_pm, np.inf)
        err_omega = np.linalg.norm(omega_ct - omega_pm, np.inf)
        # Print
        print("T = ", reactor.T)
        print("y_ct", reactor.Y)
        print("y = ", y)
        print("omega_ct = ", omega_ct)
        print("omega_pm = ", omega_pm)
        print("err_omega = ", err_omega)
        print("err_r = ", err_r)
        print()
        # Compare
        assert err_r < 1.0e-10
        assert err_omega < 1.0e-10


@pytest.mark.skipif(jnp is None, reason="JAX not installed")
def test_autodiff_accuracy(request: pytest.FixtureRequest):
    backend = BACKENDS[request.config.getoption("backend").lower()]
    if backend != PythonBackend:
        pytest.skip("JAX only supported with Python backend")

    sol, ptk = pyro_init("sandiego", jnp, request)

    # mass ratios
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5
    # indices
    i_fu = ptk.get_species_index("H2")
    i_ox = ptk.get_species_index("O2")
    i_di = ptk.get_species_index("N2")
    # mole fractions
    x = np.zeros(ptk.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # mass fractions
    y = x * ptk.wts / sum(x*ptk.wts)
    # energy
    temperature = 1500
    enthalpy = ptk.get_mixture_enthalpy_mass(temperature, y)

    # get equilibrium temperature
    sol.TPX = temperature, ct.one_atm, x
    y = sol.Y

    mass_fractions = jnp.array(y)

    guess_temp = 1400

    def chemical_source_term(mass_fractions):
        temperature = ptk.get_temperature(enthalpy, guess_temp, mass_fractions)
        density = ptk.get_density(ptk.one_atm, temperature, mass_fractions)
        return ptk.get_net_production_rates(density, temperature, mass_fractions)

    from jax import jacfwd
    chemical_jacobian = jacfwd(chemical_source_term)

    def jacobian_fd_approx(mass_fractions, delta_y):

        # Second-order (central) difference
        return jnp.array([
            (chemical_source_term(mass_fractions+delta_y*v)
                    - chemical_source_term(mass_fractions-delta_y*v))/(2*delta_y)
            for v in jnp.eye(len(mass_fractions))
        ]).T

    j = chemical_jacobian(mass_fractions)

    deltas = np.array([1e-5, 1e-6, 1e-7])
    err = np.zeros(len(deltas))
    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()
    for i, delta_y in enumerate(deltas):
        j_fd = jacobian_fd_approx(mass_fractions, delta_y)
        # Lapack norm (Anderson)
        err[i] = np.linalg.norm(j-j_fd, "fro")/np.linalg.norm(j, "fro")
        eocrec.add_data_point(delta_y, err[i])

    print("------------------------------------------------------")
    print("expected order: 2")
    print("------------------------------------------------------")
    print(eocrec.pretty_print())
    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 1.95


@pytest.mark.parametrize(
    "mechname, fuel, stoich_ratio",
    [("gri30", "CH4", 2),
     ("uconn32", "C2H4", 3),
     ("sandiego", "H2", 0.5),
     ("hong", "H2", 0.5)]
)
@pytest.mark.parametrize("user_np", numpy_list)
def test_falloff_kinetics(mechname: str, fuel: str, stoich_ratio: float, user_np,
request: pytest.FixtureRequest):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted falloff rate coefficients"""

    sol, ptk = pyro_init(mechname, user_np, request)

    # Homogeneous reactor to get test data
    init_temperature = 1500
    equiv_ratio = 1
    ox_di_ratio = 0.21

    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    x = np.zeros(ptk.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio

    # Init Cantera reactor
    sol.TPX = init_temperature, ct.one_atm, x
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])

    # Falloff reactions
    if not all((isinstance(r, ct.Reaction) for r in sol.reactions())):
        i_falloff = [i for i, r in enumerate(sol.reactions())
                     if isinstance(r, ct.FalloffReaction)]
    else:
        i_falloff = [i for i, r in enumerate(sol.reactions())
                     if r.reaction_type.startswith("falloff")]

    dt = 1e-6
    time = 0
    for _ in range(100):
        time += dt
        sim.advance(time)

        # Cantera kinetics
        k_ct = reactor.kinetics.forward_rate_constants

        # Get state from Cantera
        density = reactor.density
        temperature = reactor.T
        mass_fractions = np.where(reactor.Y > 0, reactor.Y, 0)

        # Prometheus kinetics
        concentrations = ptk.get_concentrations(density, mass_fractions)
        k_pm = np.array(ptk.get_fwd_rate_coefficients(temperature, concentrations))
        err = np.linalg.norm(
            np.where(
                k_ct[i_falloff],
                (k_ct[i_falloff] - k_pm[i_falloff])/k_ct[i_falloff],
                0
            ),
            np.inf
        )

        # Print
        print("T = ", reactor.T)
        print("k_ct = ", k_ct[i_falloff])
        print("k_pm = ", k_pm[i_falloff])
        print("err = ", err)

        # Compare
        assert err < 4e-14
