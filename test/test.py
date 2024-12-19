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

from backends import pyro_init


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
[("uiuc", "C2H4", 1.0, 1e-7),
 ("sandiego", "H2", 0.5, 1e-7),
 ("uconn32", "C2H4", 3, 1e-7)])
@pytest.mark.parametrize("user_np", numpy_list)
def test_transport(mechname: str, fuel: str, stoich_ratio: float, dt: float, user_np,
request: pytest.FixtureRequest):
    """This function tests multiple aspects of pyro transport
    1. Transport properties of individual species
    2. Transport properties of species mixtures
    """

    sol, ptk = pyro_init(mechname, user_np, request)

    i_di = sol.species_index("N2")
    i_ox = sol.species_index("O2")
    i_fu = sol.species_index(fuel)

    """Test on pointwise quantities
    """
    num_temp = 31
    temp = np.linspace(300, 3000, num_temp)
    pres = ct.one_atm

    # Individual species viscosities and conductivities
    for t in temp:
        # Cantera state and mass diffusivities
        sol.TP = t, pres
        ct_diff = sol.binary_diff_coeffs
        # Pyro transport
        pyro_visc = ptk.get_species_viscosities(t)
        pyro_cond = ptk.get_species_thermal_conductivities(t)
        pyro_diff = ptk.get_species_binary_mass_diffusivities(t)
        # Loop over species
        for sp_idx, sp_name in enumerate(sol.species_names):
            sol.Y = sp_name + ":1"
            # Errors
            err_visc = user_np.abs(pyro_visc[sp_idx] - sol.viscosity)
            err_cond = user_np.abs(
                pyro_cond[sp_idx] - sol.thermal_conductivity
            )
            err_diff = user_np.abs(
                pyro_diff[sp_idx][sp_idx]/pres - ct_diff[sp_idx, sp_idx]
            )
            assert err_visc < 1e-12
            assert err_cond < 1e-12
            assert err_diff < 1e-12

    # Now test for mixtures from a 0D reactor
    time = 0

    init_temp = 1200
    equiv_ratio = 1
    ox_di_ratio = 0.21

    x = np.zeros(ptk.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio

    sol.TPX = init_temp, pres, x
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])

    for _ in range(100):
        time += dt
        sim.advance(time)
        sol.TPY = reactor.T, pres, reactor.Y

        pyro_visc = ptk.get_mixture_viscosity_mixavg(sol.T, sol.Y)
        pyro_cond = ptk.get_mixture_thermal_conductivity_mixavg(sol.T, sol.Y)
        pyro_diff = ptk.get_species_mass_diffusivities_mixavg(
            sol.P, sol.T, sol.Y
        )
        err_visc = user_np.abs(pyro_visc - sol.viscosity)
        err_cond = user_np.abs(pyro_cond - sol.thermal_conductivity)
        err_diff = user_np.linalg.norm(pyro_diff - sol.mix_diff_coeffs)

        assert err_visc < 1e-12
        assert err_cond < 1e-12
        assert err_diff < 1e-12

    """Test on object, multi-dim arrays that represent 1D grids.
    """
    t_mix = 300

    num_points = 51
    z = user_np.linspace(0, 1, num_points)

    sol.X = fuel + ":0.5, N2:0.5"
    y_fu = sol.Y

    sol.X = "O2:0.21, N2:0.79"
    y_ox = sol.Y

    y = (y_ox + (y_fu - y_ox)*z[:, None]).T

    temp = t_mix * user_np.ones(num_points)

    if ptk.supports_overloading():
        pyro_diff_cold = ptk.get_species_mass_diffusivities_mixavg(
            pres, temp, y
        )
    else:
        pyro_diff_cold = np.zeros([sol.n_species, num_points])
        for i in range(num_points):
            pyro_diff_cold[:, i] = ptk.get_species_mass_diffusivities_mixavg(
                pres, temp[i], y[:, i])

    ct_diff_cold = np.zeros([sol.n_species, num_points])
    ct_diff_equil = np.zeros([sol.n_species, num_points])

    temp_equil = np.zeros(num_points)
    y_equil = np.zeros([sol.n_species, num_points])

    for i in range(num_points):
        mf = np.array([y[s][i] for s in range(sol.n_species)])
        sol.TPY = t_mix, pres, mf
        ct_diff_cold[:, i] = sol.mix_diff_coeffs

        sol.equilibrate("HP")
        temp_equil[i] = sol.T
        y_equil[:, i] = sol.Y
        ct_diff_equil[:, i] = sol.mix_diff_coeffs

    if ptk.supports_overloading():
        pyro_diff_equil = ptk.get_species_mass_diffusivities_mixavg(
            pres, temp_equil, y_equil)
    else:
        pyro_diff_equil = np.zeros([sol.n_species, num_points])
        for i in range(num_points):
            pyro_diff_equil[:, i] = ptk.get_species_mass_diffusivities_mixavg(
                pres, temp_equil[i], y_equil[:, i])

    for i in range(sol.n_species):
        err_cold = user_np.linalg.norm(
            ct_diff_cold[i] - pyro_diff_cold[i])

        err_equil = user_np.linalg.norm(
            ct_diff_equil[i] - pyro_diff_equil[i], np.inf)

        # print(f"Species: {s}\t... Norm(c): {err_cold}\t ... "
        #       f"Norm(e): {err_equil}")
        assert err_cold < 1e-11 and err_equil < 1e-11

    """Test on object, multi-dim arrays that represent 2D grids.
    """
    z_1, z_2 = np.meshgrid(z, z)
    y = ((y_ox + (y_fu - y_ox)*z_2[:, :, None]).T)

    # Get pyro values
    temp = t_mix * user_np.ones([num_points, num_points])
    if ptk.supports_overloading():
        pyro_diff_cold = ptk.get_species_mass_diffusivities_mixavg(pres, temp, y)
    else:
        pyro_diff_cold = np.zeros([sol.n_species, num_points, num_points])
        for i in range(num_points):
            for j in range(num_points):
                pyro_diff_cold[:, i, j] = ptk.get_species_mass_diffusivities_mixavg(
                    pres, temp[i, j], y[:, i, j])

    # Equilibrium values (from 1D test)
    temp_equil = np.tile(temp_equil, (num_points, 1))
    y_equil_twodim = np.zeros([ptk.num_species, num_points, num_points])
    for i_sp in range(ptk.num_species):
        y_equil_twodim[i_sp] = np.tile(y_equil[i_sp], (num_points, 1))

    y_equil = (y_equil_twodim)

    # Now a clunky loop for Cantera
    from itertools import product

    ct_diff_cold = np.zeros([sol.n_species, num_points, num_points])
    ct_diff_equil = np.zeros_like(ct_diff_cold)

    for i, j in product(range(num_points), range(num_points)):
        mf = np.array([y[s][i, j] for s in range(sol.n_species)])
        sol.TPY = t_mix, pres, mf
        ct_diff_cold[:, i, j] = sol.mix_diff_coeffs

        mf = np.array([y_equil[s][i, j] for s in range(sol.n_species)])
        sol.TPY = temp_equil[i, j], pres, mf
        ct_diff_equil[:, i, j] = sol.mix_diff_coeffs

    if ptk.supports_overloading():
        pyro_diff_equil = ptk.get_species_mass_diffusivities_mixavg(
            pres, temp_equil, y_equil)
    else:
        pyro_diff_equil = np.zeros([sol.n_species, num_points, num_points])
        for i in range(num_points):
            for j in range(num_points):
                pyro_diff_equil[:, i, j] = ptk.get_species_mass_diffusivities_mixavg(
                    pres, temp_equil[i, j], y_equil[:, i, j])

    # Compare
    for i in range(sol.n_species):
        err_cold = user_np.linalg.norm(
            ct_diff_cold[i] - pyro_diff_cold[i], "fro")

        err_equil = user_np.linalg.norm(
            ct_diff_equil[i] - pyro_diff_equil[i], "fro")

        assert err_cold < 1e-12 and err_equil < 1e-12

    """Now test on profiles that have single-species states
    (Y_i = 1 and Y_j = 0 for j != i)
    """
    t_mix = 300

    num_points = 51
    z = user_np.linspace(0.35, 0.65, num_points)

    y_fu = 0.5 * (1 + user_np.tanh(50 * (z - 0.5)))
    y_ox = 1 - y_fu

    y = np.zeros([ptk.num_species, num_points])
    y[i_fu] = y_fu
    y[i_ox] = y_ox

    temp = t_mix * user_np.ones(num_points)

    if ptk.supports_overloading():
        pyro_diff = ptk.get_species_mass_diffusivities_mixavg(
            ct.one_atm, temp, y)
    else:
        pyro_diff = np.zeros([sol.n_species, num_points])
        for i in range(num_points):
            pyro_diff[:, i] = ptk.get_species_mass_diffusivities_mixavg(
                ct.one_atm, temp[i], y[:, i])

    ct_diff = np.zeros([sol.n_species, num_points])

    temp_equil = np.zeros(num_points)
    y_equil = np.zeros([sol.n_species, num_points])

    for i in range(num_points):
        mf = np.array([y[s][i] for s in range(sol.n_species)])
        sol.TPY = t_mix, ct.one_atm, mf
        ct_diff[:, i] = sol.mix_diff_coeffs

    for i in range(sol.n_species):
        err = user_np.linalg.norm(
            ct_diff[i] - pyro_diff[i])

        assert err < 1e-10
