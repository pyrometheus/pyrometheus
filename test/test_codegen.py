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


def make_jax_pyro_class(ptk_base_cls, usr_np):
    if usr_np != jnp:
        return ptk_base_cls(usr_np)

    class PyroJaxNumpy(ptk_base_cls):

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
@pytest.mark.parametrize("lang_module", [pyro.codegen.python])
def test_generate_mechfile(lang_module, mechname):
    """This "test" produces the mechanism codes."""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    with open(f"mechs/{mechname}.{lang_module.file_extension}", "w") as mech_file:
        code = lang_module.gen_thermochem_code(sol)
        print(code, file=mech_file)


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_pressure(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted pressure for given density,
    temperature, and mass fractions
    """
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    ptk = make_jax_pyro_class(ptk_base_cls, usr_np)

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
    assert abs(p_ct - p_pm) / p_ct < 1.0e-12


@pytest.mark.parametrize("mechname, fuel",
                         [("uiuc", "C2H4"),
                          ("sandiego", "H2"),
                          ("uconn32", "C2H2"),
                          ("gri30", "CH4")])
@pytest.mark.parametrize("reactor_type",
                         ["IdealGasReactor", "IdealGasConstPressureReactor"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_thermo_properties(mechname, fuel, reactor_type, usr_np):
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_r, h_rt, g_rt, and k_eq
    correctly by comparing against Cantera"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    ptk = make_jax_pyro_class(ptk_base_cls, usr_np)

    def error(x):
        return np.linalg.norm(x, np.inf)

    # Loop over temperatures
    time = 0.0
    dt = 1e-6

    gas_const = ptk.gas_constant

    oxidizer = {"O2": 1.0, "N2": 3.76}
    sol.set_equivalence_ratio(phi=1.0, fuel=fuel + ":1", oxidizer=oxidizer)

    sol.TP = 1200.0, ct.one_atm

    # constant density, variable pressure
    if reactor_type == "IdealGasReactor":
        reactor = ct.IdealGasReactor(sol)

    # constant pressure, variable density
    if reactor_type == "IdealGasConstPressureReactor":
        reactor = ct.IdealGasConstPressureReactor(sol)

    sim = ct.ReactorNet([reactor])
    for _ in range(100):
        time += dt
        sim.advance(time)

        temperature = sol.T
        pressure = sol.P
        Y = sol.Y  # noqa

        # Get properties from pyrometheus and compare to Cantera
        print(temperature, pressure, Y)

        # species heat capacity
        spec_cp = ptk.get_species_specific_heats_r(temperature)
        # print(f"cp_pm = {spec_cp}")
        # print(f"cnt_cp = {sol.standard_cp_R}")
        assert error(spec_cp - sol.standard_cp_R) < 1.0e-13

        # species entropy
        spec_entropy = ptk.get_species_entropies_r(pressure, temperature)
        # print(f"s_pm = {spec_entropy}")
        # print(f"cnt_s = {sol.standard_entropies_R}")
        assert error(spec_entropy - sol.standard_entropies_R) < 1.0e-13

        # species enthalpy
        spec_enthalpy = ptk.get_species_enthalpies_rt(temperature)
        # print(f"h_pm = {spec_enthalpy}")
        # print(f"cnt_h = {sol.standard_enthalpies_RT}")
        assert error(spec_enthalpy - sol.standard_enthalpies_RT) < 1.0e-13

        # species Gibbs energy
        spec_gibbs = ptk.get_species_gibbs_rt(pressure, temperature)
        # print(f"g_pm = {spec_gibbs}")
        # print(f"cnt_h = {sol.standard_gibbs_RT}")
        assert error(spec_gibbs - sol.standard_gibbs_RT) < 1.0e-13

        # mixture entropy mole
        s_mix_mole = ptk.get_mixture_entropy_mole(pressure, temperature, Y)
        assert (s_mix_mole - sol.entropy_mole) < 5.0e-6  # round-off error
        assert (s_mix_mole - sol.entropy_mole)/sol.entropy_mole < 1.0e-12

        # mixture entropy mass
        s_mix_mass = ptk.get_mixture_entropy_mass(pressure, temperature, Y)
        assert (s_mix_mass - sol.entropy_mass) < 5.0e-6  # round-off error
        assert (s_mix_mass - sol.entropy_mass)/sol.entropy_mass < 1.0e-12

        # delta enthalpy
        nu = (sol.product_stoich_coeffs() - sol.reactant_stoich_coeffs())
        delta_h = nu.T@ptk.get_species_enthalpies_rt(temperature)
        assert error(sol.delta_enthalpy/(gas_const*temperature) - delta_h) < 1e-13

        # delta entropy
        # zero or negative mole fractions values are troublesome due to the log
        mmw = ptk.get_mix_molecular_weight(Y)
        mole_fracs = ptk.get_mole_fractions(mmw, Y)
        X = usr_np.where(usr_np.less(mole_fracs, 1e-15), 1e-15, mole_fracs)  # noqa
        delta_s = nu.T@(ptk.get_species_entropies_r(pressure, temperature)
                        - usr_np.log(X))  # see CHEMKIN manual for more details
        # exclude meaningless check on entropy for irreversible reaction
        for i, reaction in enumerate(sol.reactions()):
            # if reaction.reversible: # FIXME three-body reactions are misbehaving...
            if isinstance(reaction, ct.Arrhenius):
                print(sol.delta_entropy[i]/gas_const, delta_s[i], reaction)
                assert (sol.delta_entropy[i]/gas_const - delta_s[i]) < 1e-13

    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "gri30"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_temperature(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted temperature for given internal energy
    and mass fractions"""
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    ptk = make_jax_pyro_class(ptk_base_cls, usr_np)
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


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt, tol",
                         [("uiuc", "C2H4", 3.0, 1e-6, 1.0e-11),
                          ("sandiego", "H2", 0.5, 1e-6, 5.0e-11)])
@pytest.mark.parametrize("reactor_type",
                         ["IdealGasReactor", "IdealGasConstPressureReactor"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_kinetics(mechname, fuel, stoich_ratio, dt, tol, reactor_type, usr_np):
    """This function tests that pyrometheus-generated code computes the
    Cantera-predicted rates of progress for given temperature and composition"""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk_base_cls = pyro.codegen.python.get_thermochem_class(sol)
    ptk = make_jax_pyro_class(ptk_base_cls, usr_np)

    # Init Cantera reactor
    oxidizer = {"O2": 1.0, "N2": 3.76}
    sol.set_equivalence_ratio(phi=stoich_ratio,
                              fuel=fuel + ":1", oxidizer=oxidizer)
    sol.TP = 1200.0, ct.one_atm

    # constant density, variable pressure
    if reactor_type == "IdealGasReactor":
        reactor = ct.IdealGasReactor(sol)

    # constant pressure, variable density
    if reactor_type == "IdealGasConstPressureReactor":
        reactor = ct.IdealGasConstPressureReactor(sol)

    sim = ct.ReactorNet([reactor])

    def error(x):
        return np.linalg.norm(x, np.inf)

    time = 0.0
    for _ in range(100):
        time += dt
        sim.advance(time)

        # Get state from Cantera
        temp = sol.T
        rho = sol.density
        pressure = sol.P
        y = np.where(reactor.Y > 0, reactor.Y, 0)

        print(temp, rho, pressure, y)

        c = ptk.get_concentrations(rho, y)

        # forward
        kfd_pm = ptk.get_fwd_rate_coefficients(temp, c)
        kfw_ct = sol.forward_rate_constants
        for i, _ in enumerate(sol.reactions()):
            assert np.abs((kfd_pm[i] - kfw_ct[i]) / kfw_ct[i]) < 1.0e-13

        # equilibrium
        keq_pm = usr_np.exp(-1.*ptk.get_equilibrium_constants(pressure, temp))
        keq_ct = sol.equilibrium_constants
        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:  # skip irreversible reactions
                assert np.abs((keq_pm[i] - keq_ct[i]) / keq_ct[i]) < 1.0e-13

        # reverse rates
        krv_pm = ptk.get_rev_rate_coefficients(pressure, temp, c)
        krv_ct = sol.reverse_rate_constants
        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:  # skip irreversible reactions
                assert np.abs((krv_pm[i] - krv_ct[i]) / krv_ct[i]) < 1.0e-13

        # reaction progress
        rates_pm = ptk.get_net_rates_of_progress(pressure, temp, c)
        rates_ct = sol.net_rates_of_progress
        for i, _ in enumerate(sol.reactions()):
            print(rates_pm[i], rates_ct[i], rates_pm[i] - rates_ct[i])
            assert np.abs((rates_pm[i] - rates_ct[i])) < tol

        # species production/destruction
        omega_pm = ptk.get_net_production_rates(rho, temp, y)
        omega_ct = sol.net_production_rates
        for i in range(sol.n_species):
            print(omega_pm[i], omega_ct[i], omega_pm[i] - omega_ct[i])
            assert np.abs((omega_pm[i] - omega_ct[i])) < tol

    return


def test_autodiff_accuracy():
    pytest.importorskip("jax")
    assert jnp is not None

    sol = ct.Solution("mechs/sandiego.yaml", "gas")
    ptk_base_cls = pyro.codegen.python.get_thermochem_class(sol)

    ptk = make_jax_pyro_class(ptk_base_cls, jnp)

    # mass ratios
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5
    # indices
    i_fu = ptk.species_index("H2")
    i_ox = ptk.species_index("O2")
    i_di = ptk.species_index("N2")
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


# FIXME this test does not exercise any pressure-dependence in the reactions
@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
                         [("gri30", "CH4", 2.0),
                          ("uconn32", "C2H4", 3.0),
                          ("sandiego", "H2", 0.5)])
def test_falloff_kinetics(mechname, fuel, stoich_ratio):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted falloff rate coefficients"""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk = pyro.codegen.python.get_thermochem_class(sol)()

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
        concentrations = ptk.get_concentrations(density, mass_fractions)
        k_pm = ptk.get_fwd_rate_coefficients(temperature, concentrations)
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


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
                         [("uiuc", "C2H4", 1.0, 1e-7)])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_transport_properties(mechname, fuel, stoich_ratio, dt, usr_np):
    """This function tests that pyrometheus-generated code computes transport
    properties (viscosity, thermal conductivity and species mass diffusivity)
    correctly by comparing against Cantera"""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    ptk_base_cls = pyro.get_thermochem_class(sol)
    ptk = make_jax_pyro_class(ptk_base_cls, usr_np)

    # Loop over temperatures
    ntemp = 22
    temp = np.linspace(300.0, 2400.0, ntemp)

    for t in temp:
        sol.TP = t, ct.one_atm
        mu_pm = ptk.get_species_viscosities(t)
        kappa_pm = ptk.get_species_thermal_conductivities(t)
        dii_ct = usr_np.diag(sol.binary_diff_coeffs)
        # Loop over each individual species by making a single-species mixture
        for i, name in enumerate(sol.species_names):
            sol.Y = name + ":1"
            y = sol.Y
            dii_pm = ptk.get_species_mass_diffusivities_mixavg(
                 pressure=ct.one_atm, temperature=t, mass_fractions=y)
            # Viscosity error
            mu_err = np.abs(mu_pm[i] - sol.viscosity)
            assert mu_err < 1.0e-12

            # Conductivity
            kappa_err = np.abs(kappa_pm[i] - sol.thermal_conductivity)
            assert kappa_err < 1.0e-12

            # Self mass diffusivity
            dii_err = np.abs(dii_pm[i] - dii_ct[i])
            assert dii_err < 1.0e-12

    # Now test mixture rules, with a reactor to get sensible mass fractions
    init_temperature = 1200.0

    air = "O2:1.0,N2:3.76"
    sol.set_equivalence_ratio(phi=stoich_ratio, fuel=fuel+":1", oxidizer=air)

    sol.TP = init_temperature, ct.one_atm
    reactor = ct.IdealGasConstPressureReactor(sol)
    sim = ct.ReactorNet([reactor])

    time = 0.0

    for _ in range(100):
        time += dt
        sim.advance(time)

        # Cantera transport
        sol.TPY = reactor.T, ct.one_atm, reactor.Y

        mu_ct = sol.viscosity
        kappa_ct = sol.thermal_conductivity
        diff_ct = sol.mix_diff_coeffs

        # Get state from Cantera
        temp = reactor.T
        y = np.where(reactor.Y > 0, reactor.Y, 0)

        # Pyrometheus transport
        mu = ptk.get_mixture_viscosity_mixavg(temp, y)
        kappa = ptk.get_mixture_thermal_conductivity_mixavg(temp, y)
        diff = ptk.get_species_mass_diffusivities_mixavg(ct.one_atm, temp, y)

        err_mu = np.abs(mu - mu_ct)
        assert err_mu < 1.0e-13

        err_kappa = np.abs(kappa - kappa_ct)
        assert err_kappa < 1.0e-13

        for i in range(sol.n_species):
            err_diff = np.abs(diff[i] - diff_ct[i])
            assert err_diff < 1.0e-12

    return


# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
