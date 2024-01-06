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

            # Jax does not support object arrays
            # result = self.usr_np.empty_like(res_list, dtype=object,
            #                                 shape=(len(res_list),))
            result = self._pyro_zeros_like(self.usr_np.array(res_list))

            # 'result[:] = res_list' may look tempting, however:
            # https://github.com/numpy/numpy/issues/16564
            for idx in range(len(res_list)):
                result = result.at[idx].set(res_list[idx])

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
# @pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32"])
@pytest.mark.parametrize("lang_module", [pyro.codegen.python])
def test_generate_mechfile(lang_module, mechname):
    """This "test" produces the mechanism codes."""
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    with open(f"mechs/{mechname}.{lang_module.file_extension}", "w") as mech_file:
        code = lang_module.gen_thermochem_code(sol)
        print(code, file=mech_file)


# @pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32"])
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


# @pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "gri30"])
@pytest.mark.parametrize("mechname", ["uiuc", "sandiego"])
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


#@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
#                         [("gri30", "CH4", 2),
#                          ("uconn32", "C2H4", 3),
#                          ("sandiego", "H2", 0.5)])
@pytest.mark.parametrize("mechname, fuel, stoich_ratio",
                         [("uconn32", "C2H4", 3),
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


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
                         [("sandiego", "H2", 0.5, 1e-6),
                          ("uconn32", "C2H4", 3, 1e-7),
                          ])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_transport(mechname, fuel, stoich_ratio, dt, usr_np):

    """This function tests multiple aspects of pyro transport
    1. Transport properties of individual species
    2. Transport properties of species mixtures
    Tests are pointwise compositions and over object arrays that
    represent grids.
    """

    sol = ct.Solution(f"mechs/{mechname}.yaml")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_class, usr_np)

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
        pyro_visc = pyro_gas.get_species_viscosities(t)
        pyro_cond = pyro_gas.get_species_thermal_conductivities(t)
        pyro_diff = pyro_gas.get_species_binary_mass_diffusivities(t)
        # Loop over species
        for sp_idx, sp_name in enumerate(sol.species_names):
            sol.Y = sp_name + ":1"
            y = sol.Y
            # Errors
            err_visc = usr_np.abs(pyro_visc[sp_idx] - sol.viscosity)
            err_cond = usr_np.abs(pyro_cond[sp_idx] - sol.thermal_conductivity)
            err_diff = usr_np.abs(pyro_diff[sp_idx][sp_idx]/pres
                                  - ct_diff[sp_idx, sp_idx])
            # print(f"Species: {sp_name}\t... visc: {err_visc}\t ... "
            #       f"cond: {err_cond}\t ... diff: {err_diff}")
            assert err_visc < 1e-12
            assert err_cond < 1e-12
            assert err_diff < 1e-12

    # Now test for mixtures from a 0D reactor
    time = 0

    init_temp = 1200
    equiv_ratio = 1
    ox_di_ratio = 0.21

    x = np.zeros(pyro_gas.num_species)
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

        pyro_visc = pyro_gas.get_mixture_viscosity_mixavg(sol.T, sol.Y)
        pyro_cond = pyro_gas.get_mixture_thermal_conductivity_mixavg(sol.T,
                                                                     sol.Y)
        pyro_diff = pyro_gas.get_species_mass_diffusivities_mixavg(sol.P,
                                                                   sol.T,
                                                                   sol.Y)
        err_visc = usr_np.abs(pyro_visc - sol.viscosity)
        err_cond = usr_np.abs(pyro_cond - sol.thermal_conductivity)
        err_diff = usr_np.linalg.norm(pyro_diff - sol.mix_diff_coeffs)

        assert err_visc < 1e-12
        assert err_cond < 1e-12
        assert err_diff < 1e-12

    """Test on object, multi-dim arrays that represent 1D grids.
    """
    t_mix = 300

    num_points = 51
    z = usr_np.linspace(0, 1, num_points)

    sol.X = fuel + ":0.5, N2:0.5"
    y_fu = sol.Y

    sol.X = "O2:0.21, N2:0.79"
    y_ox = sol.Y

    y = (y_ox + (y_fu - y_ox)*z[:, None]).T

    temp = t_mix * usr_np.ones(num_points)
    pyro_diff_cold = pyro_gas.get_species_mass_diffusivities_mixavg(
        pres, temp, y)

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

    ct_diff_cold = pyro_gas._pyro_make_array(ct_diff_cold)
    ct_diff_equil = pyro_gas._pyro_make_array(ct_diff_equil)
    y_equil = pyro_gas._pyro_make_array(y_equil)

    pyro_diff_equil = pyro_gas.get_species_mass_diffusivities_mixavg(
        pres, temp_equil, y_equil)

    for i in range(sol.n_species):
        err_cold = usr_np.linalg.norm(
            ct_diff_cold[i] - pyro_diff_cold[i])

        err_equil = usr_np.linalg.norm(
            ct_diff_equil[i] - pyro_diff_equil[i], np.inf)

        # print(f"Species: {s}\t... Norm(c): {err_cold}\t ... "
        #       f"Norm(e): {err_equil}")
        assert err_cold < 1e-11 and err_equil < 1e-11

    """Test on object, multi-dim arrays that represent 2D grids.
    """
    z_1, z_2 = np.meshgrid(z, z)
    y = pyro_gas._pyro_make_array(
        (y_ox + (y_fu - y_ox)*z_2[:, :, None]).T)

    # Get pyro values
    temp = t_mix * usr_np.ones([num_points, num_points])
    pyro_diff_cold = pyro_gas.get_species_mass_diffusivities_mixavg(
        pres, temp, y)

    # Equilibrium values (from 1D test)
    temp_equil = np.tile(temp_equil, (num_points, 1))
    y_equil_twodim = np.zeros([pyro_gas.num_species, num_points, num_points])
    for i_sp in range(pyro_gas.num_species):
        y_equil_twodim[i_sp] = np.tile(y_equil[i_sp], (num_points, 1))

    y_equil = pyro_gas._pyro_make_array(y_equil_twodim)

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

    ct_diff_cold = pyro_gas._pyro_make_array(ct_diff_cold)
    ct_diff_equil = pyro_gas._pyro_make_array(ct_diff_equil)

    pyro_diff_equil = pyro_gas.get_species_mass_diffusivities_mixavg(
        pres, temp_equil, y_equil)

    # Compare
    for i in range(sol.n_species):
        err_cold = usr_np.linalg.norm(
            ct_diff_cold[i] - pyro_diff_cold[i], "fro")

        err_equil = usr_np.linalg.norm(
            ct_diff_equil[i] - pyro_diff_equil[i], "fro")

        assert err_cold < 1e-12 and err_equil < 1e-12

    """Now test on profiles that have single-species states
    (Y_i = 1 and Y_j = 0 for j != i)
    """
    t_mix = 300

    num_points = 51
    z = usr_np.linspace(0.35, 0.65, num_points)

    y_fu = 0.5 * (1 + usr_np.tanh(50 * (z - 0.5)))
    y_ox = 1 - y_fu

    y = np.zeros([pyro_gas.num_species, num_points])
    y[i_fu] = y_fu
    y[i_ox] = y_ox
    y = pyro_gas._pyro_make_array(y)
    # y = pyro_gas._pyro_make_array(jnp.zeros([
    #     pyro_gas.num_species, num_points]))

    temp = t_mix * usr_np.ones(num_points)
    pyro_diff = pyro_gas.get_species_mass_diffusivities_mixavg(
        ct.one_atm, temp, y)

    ct_diff = np.zeros([sol.n_species, num_points])

    temp_equil = np.zeros(num_points)
    y_equil = np.zeros([sol.n_species, num_points])

    for i in range(num_points):
        mf = np.array([y[s][i] for s in range(sol.n_species)])
        sol.TPY = t_mix, ct.one_atm, mf
        ct_diff[:, i] = sol.mix_diff_coeffs

    ct_diff = pyro_gas._pyro_make_array(ct_diff)

    for i in range(sol.n_species):
        err = usr_np.linalg.norm(
            ct_diff[i] - pyro_diff[i])

        assert err < 1e-10


# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
