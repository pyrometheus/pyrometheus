__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
Copyright (C) 2021 Esteban Cisneros
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
import os


def generate_pyro_cpp():
    """This function generates C++ code and compiled
    a corresponding Python module."""

    mechanisms = ["uiuc", "uconn32", "sandiego"]

    for mech in mechanisms:
        # Generate    
        sol = ct.Solution(f"mechs/{mech}.cti", "gas")
        with open(f"include/{mech}.h", "w") as mech_file:
            code = pyro.codegen.cpp.gen_thermochem_code(sol, mech)
            print(code, file=mech_file)

    # Now compile
    os.system("c++ -std=c++11 -Xlinker -undefined -Xlinker dynamic_lookup -fPIC -shared -I. wrapper.cpp -I ~/Packages/miniconda3/lib/python3.8/site-packages/pybind11/include -I ~/Packages/miniconda3/include/python3.8  -L ~/Packages/miniconda3/lib/ -o pyro_cpp.so")
    
    return


@pytest.mark.parametrize("mechname, fuel",
                         [("uiuc", "C2H4"),
                          ("uconn32", "C2H4"),
                          ("sandiego", "H2")])
def test_get_rate_coefficients(mechname, fuel):
    """This function tests that pyrometheus-generated code
    computes the rate coefficients matching Cantera
    for given temperature and composition"""

    sol = ct.Solution(f"mechs/{mechname}.cti", "gas")

    # Import Pyro cpp module
    import pyro_cpp

    if mechname == "uiuc":
        ptk = pyro_cpp.uiuc
    elif mechname == "sandiego":
        ptk = pyro_cpp.sandiego
    elif mechname == "uconn32":
        ptk = pyro_cpp.uconn32

    y = np.zeros(sol.n_species)
    y[sol.species_index(fuel)] = 1
    
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:
        # Set new temperature in Cantera
        sol.TPY = t, ct.one_atm, y
        # Concentrations
        y = pyro_cpp.VectorDouble(sol.Y)
        rho = sol.density
        c = ptk.get_concentrations(rho, y)
        # Get rate coefficients and compare
        k_ct = sol.forward_rate_constants
        k_pm = ptk.get_fwd_rate_coefficients(t, c)
        print(k_ct)
        print(np.abs((k_ct-k_pm)/k_ct))
        assert np.linalg.norm((k_ct-k_pm)/k_ct, np.inf) < 1e-13
    return


@pytest.mark.parametrize("mechname", ["uconn32", "sandiego"])
def test_get_thermo_properties(mechname):
    """
    This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_r, h_rt, and k_eq
    correctly by comparing against Cantera"""
    sol = ct.Solution(f"mechs/{mechname}.cti", "gas")

    # Import Pyro cpp module
    import pyro_cpp

    if mechname == "uiuc":
        ptk = pyro_cpp.uiuc
    elif mechname == "sandiego":
        ptk = pyro_cpp.sandiego
    elif mechname == "uconn32":
        ptk = pyro_cpp.uconn32

    temp = np.linspace(500, 3000, 10)
    for t in temp:

        sol.TP = t, ct.one_atm

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

        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:
                keq_err = np.abs((keq_pm[i] - keq_ct[i]) / keq_ct[i])
                print(f"i = {i}, keq_err = {keq_err}")
                assert keq_err < 1.0e-13
    
    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego"])
def test_get_temperature(mechname):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted temperature for given internal energy
    and mass fractions"""
    # Create Cantera object
    sol = ct.Solution(f"mechs/{mechname}.cti", "gas")

    # Import Pyro cpp module
    import pyro_cpp

    if mechname == "uiuc":
        ptk = pyro_cpp.uiuc
    elif mechname == "sandiego":
        ptk = pyro_cpp.sandiego
    elif mechname == "uconn32":
        ptk = pyro_cpp.uconn32

    # Newton tolerance
    tol = 1.0e-10
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    # First test individual species
    y = np.zeros(sol.n_species)
    for sp in range(sol.n_species):
        y[sp] = 1.0
        for t in temp:
            sol.TPY = t, ct.one_atm, y
            h = sol.enthalpy_mass
            t_guess = 0.9 * t
            t_pm = ptk.get_temperature(h, t_guess, pyro_cpp.VectorDouble(y))
            print(t, t_pm)
            assert np.abs(t - t_pm) < tol
        y[sp] = 0.0

    # Now test a mixture with fully-populated composition
    # All mass fractions set to the same value for now,
    # though a more representative test would be ignition composition
    y = np.ones(sol.n_species) / sol.n_species
    for t in temp:
        sol.TPY = t, ct.one_atm, y
        h = sol.enthalpy_mass
        t_guess = 0.9 * t
        t_pm = ptk.get_temperature(h, t_guess, pyro_cpp.VectorDouble(y))
        print(t, t_pm)
        assert np.abs(t - t_pm) < tol


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
                         [("uiuc", "C2H4", 3.0, 1e-7),
                          ("sandiego", "H2", 0.5, 1e-6)])
def test_kinetics(mechname, fuel, stoich_ratio, dt):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted rates of progress for given
    temperature and composition"""
    sol = ct.Solution(f"mechs/{mechname}.cti", "gas")

    # Import Pyro cpp module
    import pyro_cpp

    if mechname == "uiuc":
        ptk = pyro_cpp.uiuc
    elif mechname == "sandiego":
        ptk = pyro_cpp.sandiego
    elif mechname == "uconn32":
        ptk = pyro_cpp.uconn32
    
    # Homogeneous reactor to get test data
    init_temperature = 1500.0
    equiv_ratio = 1.0
    ox_di_ratio = 0.21

    i_fu = sol.species_index("H2")
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    x = np.zeros(sol.n_species)
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
        c = ptk.get_concentrations(rho, pyro_cpp.VectorDouble(y))
        r_pm = ptk.get_net_rates_of_progress(temp, pyro_cpp.VectorDouble(c))
        omega_pm = ptk.get_net_production_rates(rho, temp,
                                                pyro_cpp.VectorDouble(y))
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

    return


@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
                         [("uconn32", "C2H4", 3),
                          ("sandiego", "H2", 0.5)])
def test_falloff_kinetics(mechname, fuel, stoich_ratio):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted falloff rate coefficients"""
    sol = ct.Solution(f"mechs/{mechname}.cti", "gas")

    # Import Pyro cpp module
    import pyro_cpp

    if mechname == "uiuc":
        ptk = pyro_cpp.uiuc
    elif mechname == "sandiego":
        ptk = pyro_cpp.sandiego
    elif mechname == "uconn32":
        ptk = pyro_cpp.uconn32

    # Homogeneous reactor to get test data
    init_temperature = 1500
    equiv_ratio = 1
    ox_di_ratio = 0.21
    stoich_ratio = 0.5

    i_fu = sol.species_index("H2")
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    x = np.zeros(sol.n_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio

    # Init Cantera reactor
    sol.TPX = init_temperature, ct.one_atm, x
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])

    # Falloff reactions
    i_falloff = [i for i, r in enumerate(sol.reactions())
            if isinstance(r, ct.FalloffReaction)]

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
        concentrations = ptk.get_concentrations(density,
                                                pyro_cpp.VectorDouble(
                                                    mass_fractions))
        k_pm = ptk.get_fwd_rate_coefficients(temperature, concentrations)
        err = np.linalg.norm((k_ct[i_falloff] -
                              np.array(k_pm)[i_falloff])/k_ct[i_falloff],
                             np.inf)

        # Print
        print("T = ", reactor.T)
        print("k_ct = ", k_ct[i_falloff])
        print("k_pm = ", np.array(k_pm)[i_falloff])
        print("err = ", err)

        # Compare
        assert err < 4e-14

    return
    

# run single tests using
# $ python test_codegen.py 'test_thermo_properties()'
if __name__ == "__main__":

    generate_pyro_cpp()
    
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
