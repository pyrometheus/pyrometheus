import cantera as ct
import pyrometheus as pyro
from pyro_jax import *
from functools import partial
from jax import jit
from jax import jacfwd
from itertools import product
from scipy.special import erfcinv as erfc_inv


def convert_mole_to_mass_fractions(x: jnp.array, wts: jnp.array):
    return jnp.array(x * wts / sum(x*wts), dtype=jnp.float64)


def stoichiometric_mixture_fraction(wts_elem, elem_matrix, y_ox, y_fu):
    # Bilger weights    
    bilger_weights = np.array([0.5/wts_elem[0], -1/wts_elem[1], 0])    
    # Coupling functions
    beta_ox = bilger_weights.dot(elem_matrix.dot(y_ox))
    beta_fu = bilger_weights.dot(elem_matrix.dot(y_fu))
    return -beta_ox/(beta_fu - beta_ox)


def dissipation_rate_profile(z): 
    return jnp.exp(-2*erfc_inv(2*z)**2)


def newton(fun, jac, num_vars, num_points, q, *args, num_it=10, tol=1e-6, tag='', verbosity=0):
    for it in range(num_it):        
        dq = jnp.reshape(jnp.linalg.solve(
            jnp.reshape(jac(q, *args), 
                        (num_vars*num_points, num_vars*num_points)), 
            fun(q, *args).ravel()), (num_vars, num_points))
        q -= dq
        if verbosity:
            print(tag + "Newton: it = %i, err = %.4e" % (it, jnp.linalg.norm(dq)))
        if jnp.linalg.norm(dq) < tol:
            return q


def crank_nicolson(fun, jac, num_vars, num_points, q, dt, num_steps, newton_it=10, newton_tol=1e-6, verbosity=0):
    
    step = 0
    time = 0
    # history = jnp.array([time, guess_temp, y[pyro_gas.species_index('H')]])
    
    print("Chirometheus: Crank-Nicolson: t = %.4e [s], T_max = %.3f [K]" % (time, q[0].max()))
    
    while step < num_steps:
        q_prev = q
        q = newton(fun, jac, num_vars, num_points, q, q_prev, dt, 
                   num_it=newton_it, tol=newton_tol, tag="Chirometheus: Crank-Nicolson:", 
                   verbosity=verbosity)
        step += 1
        time += dt
        print("Chirhometues: Crank-Nicolson: t = %.4e [s], T_max = %.3f [K]" % (time, q[0].max()))
    return q        
        

def steady_flamelet():

    print("Chirometheus: solve for steady flamelets")
    
    # Cantera solution object (and elemental weights, matrix)
    sol = ct.Solution("~/Packages/pyrometheus/test/mechs/sandiego.yaml")
    
    wts_s = jnp.array(sol.molecular_weights)
    wts_e = jnp.array([wts_s[sol.species_index('H')], wts_s[sol.species_index('O')],
                       wts_s[sol.species_index('N2')]/2])
    elem_matrix = jnp.zeros([sol.n_elements, sol.n_species])
    for e, s in product(range(sol.n_elements), range(sol.n_species)):
        elem_matrix = elem_matrix.at[e, s].set(sol.n_atoms(s, e)*wts_e[e]/wts_s[s])

    # Pyrometheus-generated code
    pyro_code = pyro.codegen.python.gen_thermochem_code(sol)
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_class, jnp)

    """This next block sets the pressure and boundary conditions for
    the flamelet calculation."""
    num_vars = pyro_gas.num_species + 1
    pressure = pyro_gas.one_atm

    temp_ox = 500
    temp_fu = 300
    
    x_ox = jnp.zeros(pyro_gas.num_species)
    x_ox = x_ox.at[pyro_gas.species_index('O2')].set(0.21)
    x_ox = x_ox.at[pyro_gas.species_index('N2')].set(0.79)
    y_ox = convert_mole_to_mass_fractions(x_ox, pyro_gas.wts)
    h_ox = pyro_gas.get_mixture_enthalpy_mass(temp_ox, y_ox)
    
    
    x_fu = jnp.zeros(pyro_gas.num_species)
    x_fu = x_fu.at[pyro_gas.species_index('H2')].set(0.5)
    x_fu = x_fu.at[pyro_gas.species_index('N2')].set(0.5)
    y_fu = convert_mole_to_mass_fractions(x_fu, pyro_gas.wts)
    h_fu = pyro_gas.get_mixture_enthalpy_mass(temp_fu, y_fu)
    
    z_st = stoichiometric_mixture_fraction(wts_e, elem_matrix, y_ox, y_fu)
    print("Chirometheus: stoich mixture fraction: Z_st = %.4f" % z_st)
    print("Chirometheus: pressure: p = %.4e" % pressure)
    print("Chirometheus: air stream temperature: T_ox = %.4f [K]" % temp_ox)
    print("Chirometheus: fuel stream temperature: T_fu = %.4f [K]" % temp_fu)

    # Grid and dissipation rate (chi) profile
    num_points = 101

    mixture_fraction = jnp.linspace(0, 1, num_points)
    dz = 1/(num_points-1)

    chi_st = 1000
    diss_rate = chi_st * dissipation_rate_profile(mixture_fraction)/dissipation_rate_profile(z_st)
    print("Chirometheus: stoich dissipation rate: chi_st = %.4f [1/s]" % chi_st)

    # Initial profiles (equilibrium)
    mass_fractions = (y_ox + (y_fu - y_ox)*mixture_fraction[:, None]).T
    temperature = temp_ox + (temp_fu - temp_ox)*mixture_fraction
    
    for i in range(0, num_points):
        y = y_ox + (y_fu - y_ox)*mixture_fraction[i]
        h = h_ox + (h_fu - h_ox)*mixture_fraction[i]
        sol.HPY = h, pressure, y
        sol.equilibrate('HP')
        mass_fractions = mass_fractions.at[:, i].set(jnp.array(sol.Y))
        temperature = temperature.at[i].set(sol.T)

    state = jnp.vstack((temperature, mass_fractions))

    """This next block implements functions that are to be JIT-compiled.
    These include finite differences, the steady-flamelet RHS, and
    Crank-Nicolson with """
    @jit
    def first_deriv(f):
        return jnp.hstack((
            ((f[:, 1] - f[:, 0])/dz)[:, None], 
            0.5*(f[:, 2:] - f[:, :-2])/dz, 
            ((f[:, -1] - f[:, -2])/dz)[:, None]
        ))

    @jit
    def second_deriv(f):    
        return jnp.hstack((
            ((f[:, 2] - 2*f[:, 1] + f[:, 0])/(dz**2))[:, None], 
            (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2])/(dz**2), 
            ((f[:, -1] - 2*f[:, -2] + f[:, -3])/(dz**2))[:, None]
        ))

    laplacian = jit(jacfwd(second_deriv))

    @jit
    def get_density(state):
        return pyro_gas.get_density(pressure, state[0], state[1:])
    
    @jit
    def get_internal_energy(state):
        return pyro_gas.get_mixture_internal_energy_mass(state[0], state[1:])
    
    @jit
    def get_mixture_specific_heat_cp_mass(state):
        return pyro_gas.get_mixture_specific_heat_cp_mass(state[0], state[1:])
    
    @jit
    def chemical_source_term(state):
        density = get_density(state)
        return pyro_gas.wts[:, None] * pyro_gas.get_net_production_rates(density, state[0], state[1:])
    
    chemical_jacobian = jit(jacfwd(chemical_source_term))
    
    @jit
    def heat_release(state):
        cp_mix = get_mixture_specific_heat_cp_mass(state)
        h_species = pyro_gas.iwts[:, None] * pyro_gas.get_species_enthalpies_rt(state[0]) * \
            pyro_gas.gas_constant * state[0]
        m_dot = chemical_source_term(state)
        return -jnp.sum(m_dot * h_species, axis=0) / cp_mix
    
    @jit
    def flux(state):
        # Mixture thermochemistry
        density = get_density(state)
        cp_mix = get_mixture_specific_heat_cp_mass(state)
        # Species thermochemistry
        h_species = pyro_gas.iwts[:, None] * pyro_gas.get_species_enthalpies_rt(state[0]) * \
            pyro_gas.gas_constant * temperature
        cp_species = pyro_gas.iwts[:, None] * pyro_gas.get_species_specific_heats_r(temperature) * \
            pyro_gas.gas_constant
        # Flux
        return 0.5*(diss_rate*density/cp_mix) * first_deriv(state[0][None, :]) * (
            first_deriv(cp_mix[None, :]) + 
            jnp.sum(first_deriv(state[1:]) * cp_species, axis=0))
    
    @jit
    def steady_flamelet_rhs(state):
        density = get_density(state)
        m_dot = chemical_source_term(state)
        rhs = 0.5 * diss_rate * density * second_deriv(state) + jnp.vstack(
            (heat_release(state), m_dot)) + jnp.vstack((flux(state), jnp.zeros_like(m_dot)))
        return jnp.hstack((
            (state[:, 0] - jnp.hstack((temp_ox, y_ox))).reshape((num_vars, 1)), 
            rhs[:, 1:-1]/density[1:-1],
            (state[:, -1] - jnp.hstack((temp_fu, y_fu))).reshape((num_vars, 1))
        ))
    
    steady_flamelet_jac = jit(jacfwd(steady_flamelet_rhs))
    
    @jit
    def crank_nicolson_rhs(q, q_prev, dt): 
        return q - q_prev - 0.5 * dt * jnp.hstack(
            (jnp.zeros((num_vars, 1)), 
             (steady_flamelet_rhs(q) + 
              steady_flamelet_rhs(q_prev))[:, 1:-1], 
             jnp.zeros((num_vars, 1))))
    
    crank_nicolson_jac = jit(jacfwd(crank_nicolson_rhs))

    """
    Now solve for steady flamelets by combining pseudo-time stepping and
    Newton iterations.
    """
    # Solver config
    config = {}
    config['num_iter'] = 20
    config['crank_nicolson'] = {}
    config['crank_nicolson']['dt'] = 1e-5
    config['crank_nicolson']['num_steps'] = 5
    config['crank_nicolson']['newton_it'] = 20
    config['crank_nicolson']['newton_tol'] = 1e-6
    config['crank_nicolson']['verbosity'] = 0
    config['newton'] = {}
    config['newton']['num_it'] = 5
    config['newton']['tol'] = 1e-6

    # Solve for steady flamelet
    working_state = state

    for it in range(config['num_iter']):
        print("Chirometheus: i = %i, t = %.4e" % 
              (it, 5.0 * it * config['crank_nicolson']['dt']))
        # Advance first
        working_state = crank_nicolson(crank_nicolson_rhs, crank_nicolson_jac,
                                    num_vars, num_points, working_state,
                                    config['crank_nicolson']['dt'], 
                                    config['crank_nicolson']['num_steps'],
                                    config['crank_nicolson']['newton_it'],
                                    config['crank_nicolson']['newton_tol'],
                                    config['crank_nicolson']['verbosity'])
        # Try Newton
        newton_state = newton(steady_flamelet_rhs, steady_flamelet_jac, 
                            num_vars, num_points, working_state, 
                            num_it = config['newton']['num_it'], 
                            tol=config['newton']['tol'])
        if newton_state is None:
            print("Chirometheus: Newton failed")
        else:
            print("Chirometheus: Newton worked: T_max = %.4f [K]" % newton_state[0].max())
            return newton_state
        
    return


if __name__ == "__main__":

    steady_flamelet()
    exit
