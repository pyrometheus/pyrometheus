import cantera as ct
import pyrometheus as pyro
from pyro_jax import *
from functools import partial
from jax import jit
from jax import jacfwd


def newton(fun, jac, y, *args, num_it=10, tol=1e-6):
    """Newton method"""
    for it in range(num_it):    
        dy = jnp.linalg.solve(jac(y, *args), fun(y, *args))
        y -= dy
        if jnp.linalg.norm(dy) < tol:
            return y


def run_autoignition():

    # Cantera solution object
    sol = ct.Solution('../test/mechs/uconn32.yaml')

    # Pyrometheus-generated code
    pyro_class = pyro.codegen.python.get_thermochem_class(sol)
    pyro_gas = make_jax_pyro_class(pyro_class, jnp)

    # Thermodynamic conditions
    pressure = pyro_gas.one_atm
    temperature = 1250

    # mass ratios
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3
    # indices
    i_fu = pyro_gas.species_index("C2H4")
    i_ox = pyro_gas.species_index("O2")
    i_di = pyro_gas.species_index("N2")
    # mole fractions
    x = np.zeros(pyro_gas.num_species)
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # mass fractions
    mass_fractions = jnp.array(x * pyro_gas.wts / sum(x*pyro_gas.wts),
                               dtype=jnp.float64)
    # internal energy
    energy = pyro_gas.get_mixture_internal_energy_mass(temperature, mass_fractions)
    # density
    density = pyro_gas.get_density(pressure, temperature, mass_fractions)

    # JIT-compiled functions
    @jit
    def get_temperature(mass_fractions):
        return pyro_gas.get_temperature_energy(mass_fractions, energy, guess_temp)

    @jit
    def get_pressure(mass_fractions, temperature):
        return pyro_gas.get_pressure(density, temperature, mass_fractions)

    @jit
    def chemical_source_term(mass_fractions):
        temperature = pyro_gas.get_temperature_energy(mass_fractions, energy,
                                                   guess_temp)
        return pyro_gas.wts * pyro_gas.get_net_production_rates(density, temperature,
                                                          mass_fractions) / density

    chemical_jacobian = jit(jacfwd(chemical_source_term))

    crank_nicolson_rhs = lambda y, y_prev, dt : y - y_prev - 0.5 * dt * (
        chemical_source_term(y) + chemical_source_term(y_prev))
        
    crank_nicolson_jac = jit(jacfwd(crank_nicolson_rhs))

    # Crank-Nicolson time-stepping
    y = mass_fractions
    guess_temp = temperature
    dt = 1e-7
    time = 0
    final_time = 1e-3

    history = jnp.array([time, guess_temp,
                         y[pyro_gas.species_index('C2H4')],
                         y[pyro_gas.species_index('CO2')],
                         y[pyro_gas.species_index('H')],
                         y[pyro_gas.species_index('CH3')]])
    
    print('t = %.4e [s], T = %.f [K]' % (time, guess_temp))        
    
    while time < final_time:
        y_prev = y
        y = newton(crank_nicolson_rhs, crank_nicolson_jac, y, y_prev, dt)
        guess_temp = get_temperature(y)
        time += dt
        if not int(time/dt)%10:
            history = np.vstack((history, [time, guess_temp,
                                           y[pyro_gas.species_index('C2H4')],
                                           y[pyro_gas.species_index('CO2')],
                                           y[pyro_gas.species_index('H')],
                                           y[pyro_gas.species_index('CH3')]]))
        if not int(time/dt)%100:
            print('t = %.4e [s], T = %.3f [K]' % (time, guess_temp))
    
    return    


if __name__ == "__main__":

    run_autoignition()
    exit
