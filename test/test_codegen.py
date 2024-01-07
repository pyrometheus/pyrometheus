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

import pytato

import jax
jax.config.update("jax_enable_x64", 1)

from arraycontext import NumpyArrayContext, EagerJAXArrayContext
from arraycontext import pytest_generate_tests_for_array_contexts

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    ["numpy"]
)

numpy_list = []


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
                                      "uiuc.yaml"])
@pytest.mark.parametrize("usr_np", numpy_list)
def test_get_rate_coefficients(mechname, usr_np):
    """This function tests that pyrometheus-generated code
    computes the rate coefficients matching Cantera
    for given temperature and composition"""

    actx = actx_factory()

    sol = ct.Solution(f"mechs/{mechname}", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

    three_body_reactions = [(i, r) for i, r in enumerate(sol.reactions())
                            if r.reaction_type == "three-body-Arrhenius"]

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

        # It seems like Cantera 3.0 has bumped the third-body efficiency
        # factor from the rate coefficient to the rate of progress
        for i, r in three_body_reactions:
            eff = np.sum([c[sol.species_index(sp)]
                          * r.third_body.efficiencies[sp]
                          for sp in r.third_body.efficiencies])
            eff += np.sum([c[sp] for sp in range(sol.n_species)
                           if sol.species_name(sp) not in
                           r.third_body.efficiencies])
            k_ct[i] *= eff

        print(k_ct)
        print()
        print(k_pm)
        print()
        print(np.abs((k_ct-k_pm)/k_ct))
        assert np.linalg.norm((k_ct-k_pm)/k_ct, np.inf) < 1e-14
    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
def test_get_pressure(actx_factory, mechname):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted pressure for given density,
    temperature, and mass fractions
    """

    actx = actx_factory()

    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

    # Temperature, equivalence ratio, oxidizer ratio, stoichiometry ratio
    t = 300.0
    phi = 2.0
    alpha = 0.21
    nu = 0.5

    # Species mass fractions
    import numpy as np
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
    p_pm = pyro_gas.get_pressure(rho, t, actx.from_numpy(y))
    assert abs(p_ct - p_pm) / p_ct < 1.0e-12


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "uconn32", "gri30"])
def test_get_thermo_properties(actx_factory, mechname):
    """This function tests that pyrometheus-generated code
    computes thermodynamic properties c_p, s_r, h_rt, and k_eq
    correctly by comparing against Cantera"""

    actx = actx_factory()
    
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

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
        # exclude meaningless check on equilibrium constants for
        # irreversible reaction
        for i, reaction in enumerate(sol.reactions()):
            if reaction.reversible:
                keq_err = np.abs((keq_pm[i] - keq_ct[i]) / keq_ct[i])
                print(f"keq_err = {keq_err}")
                assert keq_err < 1.0e-13

    return


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "gri30"])
def test_get_temperature(actx_factory, mechname):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted temperature for given internal energy
    and mass fractions"""

    actx = actx_factory()
    
    # Create Cantera and pyrometheus objects
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)
    
    tol = 1.0e-10
    # Test temperatures
    import numpy as np
    temp = np.linspace(500.0, 3000.0, 10)
    # First test individual species
    y = np.zeros(pyro_gas.num_species)
    for sp in range(pyro_gas.num_species):
        y[sp] = 1.0
        for t in temp:
            sol.TPY = t, ct.one_atm, y
            e = sol.int_energy_mass
            t_guess = 0.9 * t
            t_pm = t
            t_pm = pyro_gas.get_temperature(e, t_guess,
                                           actx.from_numpy(y), True)
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
        t_pm = pyro_gas.get_temperature(e, t_guess,
                                        actx.from_numpy(y), True)
        assert np.abs(t - t_pm) < tol


@pytest.mark.parametrize("mechname, fuel, stoich_ratio, dt",
                         [("uiuc", "C2H4", 3.0, 1e-7),
                          ("sandiego", "H2", 0.5, 1e-6)])
def test_kinetics(actx_factory, mechname, fuel, stoich_ratio, dt):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted rates of progress for given
    temperature and composition"""

    actx = actx_factory()
    
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

    # Homogeneous reactor to get test data
    init_temperature = 1500.0
    equiv_ratio = 1.0
    ox_di_ratio = 0.21

    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    import numpy as np
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
        c = pyro_gas.get_concentrations(rho, actx.from_numpy(y))
        r_pm = pyro_gas.get_net_rates_of_progress(temp, c)
        omega_pm = pyro_gas.get_net_production_rates(rho, temp, actx.from_numpy(y))
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

    actx = EagerJAXArrayContext()

    sol = ct.Solution("mechs/sandiego.yaml", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

    # mass ratios
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5
    # indices
    i_fu = pyro_gas.species_index("H2")
    i_ox = pyro_gas.species_index("O2")
    i_di = pyro_gas.species_index("N2")

    # mole fractions
    x = actx.zeros(pyro_gas.num_species, dtype="float64")
    x = x.at[i_fu].set(
        (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    )
    x = x.at[i_ox].set(stoich_ratio*x[i_fu]/equiv_ratio)
    x = x.at[i_di].set((1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio)
    # mass fractions
    y = x * pyro_gas.wts / sum(x*pyro_gas.wts)
    # energy
    temperature = 1500
    # enthalpy = pyro_gas.get_mixture_enthalpy_mass(temperature, y)

    # get equilibrium temperature
    sol.TPX = temperature, ct.one_atm, actx.to_numpy(x)
    y = sol.Y
    mass_fractions = actx.from_numpy(y)

    # guess_temp = 1400

    def chemical_source_term(state):
        density = pyro_gas.get_density(pyro_gas.one_atm, state[0], state[1:])
        return pyro_gas.get_net_production_rates(density, state[0], state[1:])

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

    import numpy as np
    deltas = np.array([1e-5, 1e-6, 1e-7])
    err = np.zeros(len(deltas))
    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()
    for i, delta_y in enumerate(deltas):
        j_fd = jacobian_fd_approx(mass_fractions, delta_y)
        # Lapack norm (Anderson)
        err[i] = np.linalg.norm(j-j_fd, "fro")/np.linalg.norm(
            actx.to_numpy(j), "fro")
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
def test_falloff_kinetics(actx_factory, mechname, fuel, stoich_ratio):
    """This function tests that pyrometheus-generated code
    computes the Cantera-predicted falloff rate coefficients"""

    actx = actx_factory()
    
    sol = ct.Solution(f"mechs/{mechname}.yaml", "gas")
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = pyro_class(actx)

    # Homogeneous reactor to get test data
    init_temperature = 1500
    equiv_ratio = 1
    ox_di_ratio = 0.21

    i_fu = sol.species_index(fuel)
    i_ox = sol.species_index("O2")
    i_di = sol.species_index("N2")

    import numpy as np
    x = np.zeros(pyro_gas.num_species)
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
        concentrations = pyro_gas.get_concentrations(density,
                                                     actx.from_numpy(mass_fractions))
        k_pm = actx.to_numpy(
            pyro_gas.get_fwd_rate_coefficients(temperature, concentrations)
        )

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


def test_on_grids(actx_factory):

    def convert_mole_to_mass_fractions(pyro_gas, x):
        return pyro_gas.actx.array(x * pyro_gas.wts / sum(x*pyro_gas.wts))

    actx = actx_factory()
    sol = ct.Solution("mechs/sandiego.yaml")
    pyro_code = pyro.codegen.python.gen_thermochem_code(sol)
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    
    # Pyro realizations with different NumPys
    pyro_gas = pyro_class(actx)

    # Grid and profiles
    num_species = pyro_gas.num_species
    num_points = 101
    mixture_fraction = actx.from_numpy(np.linspace(0, 1, num_points))

    one_atm = pyro_gas.one_atm
    temp_ox = 500
    temp_fu = 300

    x_ox = np.zeros(num_species)
    x_ox[pyro_gas.species_index("O2")] = 0.21
    x_ox[pyro_gas.species_index("N2")] = 0.79
    y_ox = x_ox * pyro_gas.wts / sum(x_ox * pyro_gas.wts)    
    
    x_fu = np.zeros(num_species)
    x_fu[pyro_gas.species_index("H2")] = 0.5
    x_fu[pyro_gas.species_index("N2")] = 0.5
    y_fu = x_fu * pyro_gas.wts / sum(x_fu * pyro_gas.wts)

    mass_fractions = (y_ox + (y_fu - y_ox)*mixture_fraction[:, None]).T
    temperature = temp_ox + (temp_fu - temp_ox)*mixture_fraction

    # Jax: Compute
    density = pyro_gas.get_density(one_atm, temperature, mass_fractions)
    omega = pyro_gas.get_net_production_rates(density, temperature,
                                              mass_fractions)

    # Build computational graph
    # temperature = pytato.make_placeholder("temperature", shape=(num_points,),
    #                                       dtype="float64")
    # mass_fractions = pytato.make_placeholder("mass_frac", shape=(num_species, num_points,),
    #                                          dtype="float64")

    # density = gas_jax_lazy.get_density(one_atm, temperature, mass_fractions)
    # omega = gas_jax_lazy.get_net_production_rates(density, temperature, mass_fractions)

    # GRAPH_DOT = "graph.dot"
    # GRAPH_SVG = "graph.svg"
    # dot_code = pytato.get_dot_graph(omega)
    # with open(GRAPH_DOT, "w") as outf:
    #     outf.write(dot_code)
        
    # import shutil
    # import subprocess
    # dot_path = shutil.which("dot")
    # subprocess.run([dot_path, "-Tsvg", GRAPH_DOT, "-o", GRAPH_SVG], check=True)
    return
    
# run single tests using
# $ python test_codegen.py 'test_sandiego()'
if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
