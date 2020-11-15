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


def test_get_pressure():
    """This function tests that pyrometheus-generate code
    computes the right pressure for given density, temperature,
    and mass fractions
    """
    # Create Cantera and pyrometheus objects
    sol = ct.Solution("sanDiego.cti", "gas")
    ptk = pyro.gen_python_code(sol)()
    print(ptk.species_indices)
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


def test_get_temperature():
    """This function tests that pyrometheus-generated code
    computes the right temperature for given internal energy
    and mass fractions"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution("sanDiego.cti", "gas")
    ptk = pyro.gen_python_code(sol)()
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


def test_get_thermo_properties():
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_R, h_RT, and k_eq
    correctly by comparing against Cantera"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution("sanDiego.cti", "gas")
    ptk = pyro.gen_python_code(sol)()
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
        cp_err = (cp_pm - sol.standard_cp_R).max()
        s_err = (s_pm - sol.standard_entropies_R).max()
        h_err = (h_pm - sol.standard_enthalpies_RT).max()
        keq_err = ((keq_pm - keq_ct) / keq_ct).max()
        assert cp_err < 1.0e-14
        assert s_err < 1.0e-14
        assert h_err < 1.0e-14
        assert keq_err < 1.0e-13

    return


# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
