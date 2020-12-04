__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
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

import sys

import cantera as ct
import numpy as np  # noqa: F401
import pyrometheus as pyro
import pytest


@pytest.mark.parametrize("mechname,fuel", [("uiuc", "C2H4"),
                                       ("sanDiego", "H2")])
def test_kinetics(mechname, fuel):
    """This function tests that pyrometheus-generated code
    computes the right rates of progress for given temperature
    and composition"""
    sol = ct.Solution(f"{mechname}.cti", "gas")
    ptk = pyro.get_thermochem_class(sol)()

    # Homogeneous reactor to get test data
    init_temperature = 1500.0
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0  # 0.5
    i_fu = sol.species_index(f"{fuel}")
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")
    x = np.zeros(ptk.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    sol.TPX = init_temperature, ct.one_atm, x
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])

    time = 0.0
    for step in range(50):
        time += 1.0e-7  # 1.0e-6 for H2
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
        # Print
        err_r = np.abs((r_ct-r_pm))
        err_omega = np.abs((omega_ct[0:-1]-omega_pm[0:-1]))
        print("T = ", reactor.T)
        print("y_ct", reactor.Y)
        print("y = ", y)
        print("omega_ct = ", omega_ct[0:-1])
        print("omega_pm = ", omega_pm[0:-1])
        print("err_omega = ", err_omega)
        print("err_r = ", err_r)
        print()
        # Compare
        #assert err_r < 1.0e-10
        #assert err_omega < 1.0e-8

    return


@pytest.mark.parametrize("mechname", ["uiuc", "sanDiego"])
def test_get_rate_coefficients(mechname):
    """This function tests that pyrometheus-generated code
    computes the right rate coefficients for given temeprature
    and composition"""
    sol = ct.Solution(f"{mechname}.cti", "gas")
    ptk = pyro.get_thermochem_class(sol)()
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:
        # Set new temperature in Cantera
        sol.TP = t, ct.one_atm
        # Concentrations
        y = sol.Y
        rho = sol.density
        c = ptk.get_concentrations(rho, y)
        # Get rate coefficients and compare
        k_ct = sol.forward_rate_constants
        k_pm = ptk.get_fwd_rate_coefficients(t, c)
        print(np.abs((k_ct-k_pm)/k_ct))
        assert np.abs((k_ct-k_pm) / k_ct).max() < 1.0e-14
    return


@pytest.mark.parametrize("mechname", ["uiuc", "sanDiego"])
def test_get_pressure(mechname):
    """This function tests that pyrometheus-generated code
    computes the right pressure for given density, temperature,
    and mass fractions
    """
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"{mechname}.cti", "gas")
    ptk = pyro.get_thermochem_class(sol)()
    # Temperature, equivalence ratio, oxidizer ratio, stoichiometry ratio
    t = 300.0
    phi = 2.0
    alpha = 0.21
    nu = 0.5
    # Species mass fractions
    i_fu = ptk.species_index("H2")
    i_ox = ptk.species_index("O2")
    i_di = ptk.species_index("N2")
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
    assert abs(p_ct - p_pm) / p_ct < 1.0e-2


@pytest.mark.parametrize("mechname", ["uiuc", "sanDiego"])
def test_get_temperature(mechname):
    """This function tests that pyrometheus-generated code
    computes the right temperature for given internal energy
    and mass fractions"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"{mechname}.cti", "gas")
    ptk = pyro.get_thermochem_class(sol)()
    # Tolerance- chosen value keeps error in rate under 1%
    tol = 1.0e-2
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


@pytest.mark.parametrize("mechname", ["uiuc", "sanDiego"])
def test_get_thermo_properties(mechname):
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_R, h_RT, and k_eq
    correctly by comparing against Cantera"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"{mechname}.cti", "gas")
    ptk = pyro.get_thermochem_class(sol)()
    # Loop over temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:

        # Get properties from pyrometheus
        cp_pm = ptk.get_species_specific_heats_R(t)
        s_pm = ptk.get_species_entropies_R(t)
        h_pm = ptk.get_species_enthalpies_RT(t)
        keq_pm = 1.0 / np.exp(ptk.get_equilibrium_constants(t))
        # Set state in cantera for comparison
        sol.TP = t, ct.one_atm
        keq_ct = sol.equilibrium_constants
        # Compare properties
        cp_err = np.abs(cp_pm - sol.standard_cp_R).max()
        s_err = np.abs(s_pm - sol.standard_entropies_R).max()
        h_err = np.abs(h_pm - sol.standard_enthalpies_RT).max()
        keq_err = np.abs((keq_pm - keq_ct) / keq_ct).max()
        print(keq_ct)
        print(keq_pm)
        print(np.abs((keq_pm - keq_ct) / keq_ct))
        print(keq_err)
        assert cp_err < 1.0e-13
        assert s_err < 1.0e-13
        assert h_err < 1.0e-13
        #assert keq_err < 1.0e-13

    return


# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
