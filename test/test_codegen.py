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

try:
    import jax
except ImportError:
    numpy_list = [np]
    jnp = None
else:
    import jax.numpy as jnp  # noqa: F401
    jax.config.update("jax_enable_x64", 1)
    numpy_list = [np, jnp]


def make_jax_pyro_class(pyro_gas_base_cls, usr_np):
    if usr_np != jnp:
        return pyro_gas_base_cls(usr_np)

    class PyroJaxNumpy(pyro_gas_base_cls):

        def _pyro_make_array(self, res_list):
            """This works around (e.g.) numpy.exp not working with object
            arrays of :mod:`numpy` scalars. It defaults to making object arrays,
            however if an array consists of all scalars, it makes a "plain old"
            :class:`numpy.ndarray`.
            See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
            for more context.
            """

            from numbers import Number
            # Needed to play nicely with Jax, which frequently creates
            # arrays of shape () when handed numbers
            all_numbers = all(
                isinstance(e, Number)
                or (isinstance(e, self.usr_np.ndarray) and e.shape == ())
                for e in res_list)

            if all_numbers:
                return self.usr_np.array(res_list, dtype=self.usr_np.float64)

            result = self.usr_np.empty_like(res_list, dtype=object,
                                            shape=(len(res_list),))

            # 'result[:] = res_list' may look tempting, however:
            # https://github.com/numpy/numpy/issues/16564
            for idx in range(len(res_list)):
                result[idx] = res_list[idx]

            return result

        def _pyro_norm(self, argument, normord):
            """This works around numpy.linalg norm not working with scalars.

            If the argument is a regular ole number, it uses :func:`numpy.abs`,
            otherwise it uses ``usr_np.linalg.norm``.
            """
            # Wrap norm for scalars
            from numbers import Number
            if isinstance(argument, Number):
                return self.usr_np.abs(argument)
            # Needed to play nicely with Jax, which frequently creates
            # arrays of shape () when handed numbers
            if isinstance(argument, self.usr_np.ndarray) and argument.shape == ():
                return self.usr_np.abs(argument)
            return self.usr_np.linalg.norm(argument, normord)

    return PyroJaxNumpy(usr_np=usr_np)


# Write out all the mechanisms for inspection
@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("lang_module", [
    pyro.codegen.python,
    ])
def test_generate_mechfile(lang_module, mechname):
    """This "test" produces the mechanism codes."""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    with open(f"mechs/{mechname}.{lang_module.file_extension}", "w") as mech_file:
        code = lang_module.gen_thermochem_code(sol)
        print(code, file=mech_file)


@pytest.mark.parametrize("mechname", ["uiuc.yaml", "sandiego.yaml",
                                      "uiuc.cti"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_rate_coefficients(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the rate coefficients matching Cantera
    for given temperature and composition"""
    sol = ct.Solution(f"mechs/{mechname}", "gas")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, usr_np)

    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:
        # Set new temperature in Cantera
        sol.TP = t, ct.one_atm
        # Concentrations
        y = sol.Y
        rho = sol.density
        c = pyro_gas.get_concentrations(rho, y)
        # Get rate coefficients and compare
        k_ct = sol.forward_rate_constants
        k_pm = pyro_gas.get_fwd_rate_coefficients(t, c)
        print(k_ct)
        print(np.abs((k_ct-k_pm)/k_ct))
        assert np.linalg.norm((k_ct-k_pm)/k_ct, np.inf) < 1e-14
    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_pressure(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted pressure for given density,
    temperature, and mass fractions
    """
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, usr_np)

    # Temperature, equivalence ratio, oxidizer ratio, stoichiometry ratio
    t = 300.0
    phi = 2.0
    alpha = 0.21
    nu = 0.5

    # Species mass fractions
    i_fu = pyro_gas.species_index("H2")
    i_ox = pyro_gas.species_index("O2")
    i_di = pyro_gas.species_index("N2")
    x = np.zeros(pyro_gas.num_species)
    x[i_fu] = (alpha * phi) / (nu + alpha * phi)
    x[i_ox] = nu * x[i_fu] / phi
    x[i_di] = (1.0 - alpha) * x[i_ox] / alpha

    # Get equilibrium composition
    sol.TPX = t, ct.one_atm, x
    sol.equilibrate("UV")
    t, rho, y = sol.TDY
    p_ct = sol.P

    # Compute pressure with pyrometheus and compare to Cantera
    p_pm = pyro_gas.get_pressure(rho, t, y)
    assert abs(p_ct - p_pm) / p_ct < 1.0e-12


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_thermo_properties(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_r, h_rt, and k_eq
    correctly by comparing against Cantera"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, usr_np)

    # Loop over temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    for t in temp:

        # Set state in cantera for comparison
        sol.TP = t, ct.one_atm

        # Get properties from pyrometheus and compare to Cantera
        cp_pm = pyro_gas.get_species_specific_heats_r(t)
        cp_err = np.linalg.norm(cp_pm - sol.standard_cp_R, np.inf)
        print(f"cp_pm = {cp_pm}")
        print(f"cnt_cp = {sol.standard_cp_R}")
        assert cp_err < 1.0e-13

        s_pm = pyro_gas.get_species_entropies_r(t)
        s_err = np.linalg.norm(s_pm - sol.standard_entropies_R, np.inf)
        print(f"s_pm = {s_pm}")
        print(f"cnt_s = {sol.standard_entropies_R}")
        assert s_err < 1.0e-13

        h_pm = pyro_gas.get_species_enthalpies_rt(t)
        h_err = np.linalg.norm(h_pm - sol.standard_enthalpies_RT, np.inf)
        print(f"h_pm = {h_pm}")
        print(f"cnt_h = {sol.standard_enthalpies_RT}")
        assert h_err < 1.0e-13

        keq_pm1 = pyro_gas.get_equilibrium_constants(t)
        print(f"keq1 = {keq_pm1}")
        keq_pm = 1.0 / np.exp(pyro_gas.get_equilibrium_constants(t))
        keq_ct = sol.equilibrium_constants

        print(f"keq_pm = {keq_pm}")
        print(f"keq_cnt = {keq_ct}")
        print(f"temperature = {t}")
        # xclude meaningless check on equilibrium constants for irreversible reaction
        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:
                keq_err = np.abs((keq_pm[i] - keq_ct[i]) / keq_ct[i])
                print(f"keq_err = {keq_err}")
                assert keq_err < 1.0e-13
        # keq_pm_test = keq_pm[1:]
        # keq_ct_test = keq_ct[1:]
        # keq_err = np.linalg.norm((keq_pm_test - keq_ct_test) / keq_ct_test, np.inf)
        # assert keq_err < 1.0e-13

    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "gri30"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_temperature(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted temperature for given internal energy
    and mass fractions"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, usr_np)
    tol = 1.0e-10
    # Test temperatures
    temp = np.linspace(500.0, 3000.0, 10)
    # First test individual species
    y = np.zeros(pyro_gas.num_species)
    for sp in range(pyro_gas.num_species):
        y[sp] = 1.0
        for t in temp:
            sol.TPY = t, ct.one_atm, y
            e = sol.int_energy_mass
            t_guess = 0.9 * t
            t_pm = pyro_gas.get_temperature(e, t_guess, y, True)
            assert np.abs(t - t_pm) < tol
        y[sp] = 0.0

    # Now test a mixture with fully-populated composition
    # All mass fractions set to the same value for now,
    # though a more representative test would be ignition composition
    y = np.ones(pyro_gas.num_species) / pyro_gas.num_species
    for t in temp:
        sol.TPY = t, ct.one_atm, y
        e = sol.int_energy_mass
        t_guess = 0.9 * t
        t_pm = pyro_gas.get_temperature(e, t_guess, y, True)
        assert np.abs(t - t_pm) < tol


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
                         [("uiuc", "C2H4", 3.0, 1e-7),
                          ("sandiego", "H2", 0.5, 1e-6)])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_kinetics(mechname, fuel, stoich_ratio, dt, usr_np):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted rates of progress for given
    temperature and composition"""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, usr_np)

    # Homogeneous reactor to get test data
    init_temperature = 1500.0
    equiv_ratio = 1.0
    ox_di_ratio = 0.21

    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    x = np.zeros(pyro_gas.num_species)
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
        c = pyro_gas.get_concentrations(rho, y)
        r_pm = pyro_gas.get_net_rates_of_progress(temp, c)
        omega_pm = pyro_gas.get_net_production_rates(rho, temp, y)
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


def test_autodiff_accuracy():
    pytest.importorskip("jax")
    assert jnp is not None

    sol = ct.Solution("mechs/sandiego.yaml", "gas")
    pyro_gas_base_cls = pyro.codegen.python.get_thermochem_class(sol)

    pyro_gas = make_jax_pyro_class(pyro_gas_base_cls, jnp)

    # mass ratios
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5
    # indices
    i_fu = pyro_gas.species_index("H2")
    i_ox = pyro_gas.species_index("O2")
    i_di = pyro_gas.species_index("N2")
    # mole fractions
    x = np.zeros(pyro_gas.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # mass fractions
    y = x * pyro_gas.wts / sum(x*pyro_gas.wts)
    # energy
    temperature = 1500
    enthalpy = pyro_gas.get_mixture_enthalpy_mass(temperature, y)

    # get equilibrium temperature
    sol.TPX = temperature, ct.one_atm, x
    y = sol.Y

    mass_fractions = jnp.array(y)

    guess_temp = 1400

    def chemical_source_term(mass_fractions):
        temperature = pyro_gas.get_temperature(enthalpy, guess_temp, mass_fractions)
        density = pyro_gas.get_density(pyro_gas.one_atm, temperature, mass_fractions)
        return pyro_gas.get_net_production_rates(
            density, temperature, mass_fractions)

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


@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
                         [("gri30", "CH4", 2),
                          ("uconn32", "C2H4", 3),
                          ("sandiego", "H2", 0.5)])
def test_falloff_kinetics(mechname, fuel, stoich_ratio):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted falloff rate coefficients"""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_gas = pyro.codegen.python.get_thermochem_class(sol)()

    # Homogeneous reactor to get test data
    init_temperature = 1500
    equiv_ratio = 1
    ox_di_ratio = 0.21

    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    x = np.zeros(pyro_gas.num_species)
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
        concentrations = pyro_gas.get_concentrations(density, mass_fractions)
        k_pm = pyro_gas.get_fwd_rate_coefficients(temperature, concentrations)
        err = np.linalg.norm((k_ct[i_falloff] - k_pm[i_falloff])/k_ct[i_falloff],
                np.inf)

        # Print
        print("T = ", reactor.T)
        print("k_ct = ", k_ct[i_falloff])
        print("k_pm = ", k_pm[i_falloff])
        print("err = ", err)

        # Compare
        assert err < 4e-14

    return


@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
                         [("gri30", "CH4", 2),
                          ("uconn32", "C2H4", 3),
                          ("sandiego", "H2", 0.5)])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_external_rate_params(mechname, fuel, stoich_ratio, usr_np):
    """
    This function tests that pyrometheus-generated code
    computes Cantera-predicted rates of progress"""

    # Cantera
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")

    # Need pre-exponentials from Cantera
    a = usr_np.array([])
    b = usr_np.array([])
    t_a = usr_np.array([])

    for r in sol.reactions():
        if not isinstance(r, ct.FalloffReaction):
            a = usr_np.append(a, r.rate.pre_exponential_factor)
            b = usr_np.append(b, r.rate.temperature_exponent)
            t_a = usr_np.append(t_a,
                                r.rate.activation_energy / ct.gas_constant)
        else:
            # Falloff coefficients are hard-coded
            a = usr_np.append(a, 0)
            b = usr_np.append(b, 0)
            t_a = usr_np.append(t_a, 0)

    rate_params = (a, b, t_a)

    # Pyro code
    pyro_class = pyro.codegen.python.get_thermochem_class(
        sol, fixed_coeffs=False)
    pyro_gas = make_jax_pyro_class(pyro_class, usr_np)

    # Set state and compute rates
    temp = 1500
    fu = fuel + ":1"
    ox = "O2:0.21, N2:0.79"

    sol.TP = temp, ct.one_atm
    sol.set_equivalence_ratio(phi=1, fuel=fu, oxidizer=ox)

    rho = sol.density
    y = sol.Y

    k_ct = sol.forward_rate_constants
    r_ct = sol.net_rates_of_progress
    omega_ct = sol.net_production_rates

    c = pyro_gas.get_concentrations(rho, y)
    k = pyro_gas.get_fwd_rate_coefficients(temp, c, rate_params)
    r = pyro_gas.get_net_rates_of_progress(temp, c, rate_params)
    omega = pyro_gas.get_net_production_rates(rho, temp, y, rate_params)

    assert (np.linalg.norm((k - k_ct), np.inf)
            / np.linalg.norm(k_ct, np.inf)) < 1e-14
    assert np.linalg.norm(r - r_ct, np.inf) < 1e-10
    assert np.linalg.norm(omega - omega_ct, np.inf) < 1e-10

    return


@pytest.mark.parametrize("usr_np", numpy_list)
def test_pyro_on_grids(usr_np):
    """This function tests that pyrometheus-generated code
    computes Cantera-predicted production rates on grids"""
    sol = ct.Solution("mechs/sandiego.yaml", "gas")
    pyro_gas = pyro.codegen.python.get_thermochem_class(sol)()

    num_points = 101
    grid = usr_np.linspace(0, 1, num_points)

    pressure = ct.one_atm
    temp_ox = 1200
    temp_fu = 500

    sol.TPX = temp_ox, pressure, "O2:0.2, O:0.01, N2:0.79"
    y_ox = sol.Y

    sol.TPX = temp_fu, pressure, "H2:0.49, H:0.01, N2:0.5"
    y_fu = sol.Y

    mass_frac = (y_ox + (y_fu - y_ox)*grid[:, None]).T
    temperature = temp_ox + (temp_fu - temp_ox)*grid
    density = pyro_gas.get_density(pressure, temperature, mass_frac)
    omega = pyro_gas.get_net_production_rates(density, temperature, mass_frac)

    omega_ct = usr_np.array([])
    for i in range(num_points):
        sol.TPY = temperature[i], ct.one_atm, mass_frac[:, i]
        omega_ct = usr_np.vstack((omega_ct, sol.net_production_rates)) \
            if omega_ct.size else sol.net_production_rates

    for sp in range(pyro_gas.num_species):
        err = np.linalg.norm(omega[sp][:] - omega_ct[:, sp])
        assert err < 1e-10
    return


# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
