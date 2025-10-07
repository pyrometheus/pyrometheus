import sys
import yaml
import h5py
import numpy as np
import cantera as ct
from pyrometheus.codegen.python import PythonCodeGenerator as pyro
from dataclasses import dataclass
from make_pyro import make_pyro_object
from reactors import Flame, FlameState
from domain import Mesh, Operators
from time_integ import (
    RungeKutta, create_time_windows
)


@dataclass
class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def form_file_name(config, step):
    if step:
        from numpy import floor, log10
        num_zeros = floor(log10(step)).astype(int)
        step_str = (8 - num_zeros) * '0' + f'{step}'
    else:
        step_str = 9*'0'

    dir_id = config['output_dir']
    return f'output/flame_{dir_id}/flame_{step_str}.h5'


def restart(config):

    # Open h5 database
    db = h5py.File(form_file_name(config, config['restart_step']), 'r')

    # Mesh
    x = db['domain/mesh/x'][:]
    mesh = Mesh(x=x)
    op = Operators(mesh, pyro_gas.pyro_np)
    reactor = Flame(pyro_gas, op, config['transport_model'])

    # Guess temperature
    temp_guess = db['prim_vars/temperature'][:]
    reactor.set_temperature_guess(temp_guess)

    # Initial state
    cons_vars = FlameState(
        db['cons_vars/momentum'][:],
        db['cons_vars/total_energy'][:],
        db['cons_vars/densities'][:, :]
    )

    # Time
    t_offset = db['time/time_and_step'][0]
    return t_offset, config['restart_step'], cons_vars, mesh, reactor


def initialize(config):

    # Grid
    import numpy as np
    x = np.linspace(0, config['x_max'], config['num_x'])

    # Mixture states
    fuel = config['fuel']
    pres = ct.one_atm
    temp_c = 300
    temp_h = 1600

    sol.TP = temp_c, pres
    sol.set_equivalence_ratio(
        config['equiv_ratio'],
        f'{fuel}:1',
        'O2:0.21, N2:0.79'
    )
    mass_frac_c = sol.Y

    sol.TPY = temp_h, ct.one_atm, mass_frac_c
    sol.equilibrate('TP')
    mass_frac_h = sol.Y

    # Mollifier
    blend_factor = 1e2 / x.max()
    mollifier = 0.5 * (
        1 - np.tanh(blend_factor * (
            x - 0.5 * config['x_max']
        ))
    )

    temp = temp_c + (temp_h - temp_c) * mollifier
    mass_frac = np.array([
        y_c + (y_h - y_c) * mollifier
        for y_c, y_h in zip(mass_frac_c, mass_frac_h)
    ])
    density = pyro_gas.get_density(
        pres * np.ones_like(temp), temp, mass_frac
    )
    energy = pyro_gas.get_mixture_internal_energy_mass(
        temp, mass_frac
    )

    # Create objects
    mesh = Mesh(x=x)
    op = Operators(mesh, pyro_gas.pyro_np)
    reactor = Flame(pyro_gas, op, config['transport_model'])
    reactor.set_temperature_guess(temp)

    cons_vars = FlameState(
        momentum=np.zeros_like(temp),
        total_energy=density * energy,
        densities=np.array([
            density * y for y in mass_frac
        ])
    )

    return 0, 0, cons_vars, mesh, reactor


def write_to_file(config,
                  step,
                  time,
                  step_size,
                  reactor,
                  cons_vars: FlameState,
                  prim_vars: FlameState,
                  temperature):

    # Write to file
    file_name = form_file_name(config, step)
    db = h5py.File(file_name, 'w')
    # Time
    g = db.create_group('time')
    g.create_dataset('time_and_step', data=[time, step_size])
    # Domain
    g = db.create_group('domain')
    g_mesh = g.create_group('mesh')
    g_mesh.create_dataset('x', data=reactor.op.mesh.x)

    # Conserved variables
    g = db.create_group('cons_vars')
    g.create_dataset('momentum', data=cons_vars.momentum)
    g.create_dataset('total_energy', data=cons_vars.total_energy)
    g.create_dataset('densities', data=cons_vars.densities)
    # Primitive variables
    g = db.create_group('prim_vars')
    g.create_dataset('velocity', data=prim_vars.velocity)
    g.create_dataset('pressure', data=prim_vars.pressure)
    g.create_dataset('mass_fractions', data=prim_vars.mass_fractions)
    g.create_dataset('temperature', data=temperature)
    db.close()
    return


def time_integration():

    # Initialize state, mesh, and reactor
    if 'restart_step' in config:
        restart_step = config['restart_step']
        print(f'{bcolor.OKGREEN} Initializing from restart file '
              f'{restart_step} {bcolor.ENDC}')
        t_offset, init_step, cons_vars, mesh, reactor = restart(config)
    else:
        t_offset, init_step, cons_vars, mesh, reactor = initialize(config)
        prim_vars, dens, temp = reactor.equation_of_state(cons_vars)
        write_to_file(
            config, init_step, t_offset, config['step_size'],
            reactor, cons_vars, prim_vars, temp
        )

    print(f'mesh size: {mesh.num_x}, array shape: {cons_vars.momentum.shape}')

    # Time integrator
    step_size = config['step_size']
    num_snapshots = config['num_snapshots']
    time_integ = RungeKutta(reactor,)

    assert time_integ.rxr.species_mass_flux == reactor.species_mass_flux_mixavg
    
    if config['filter']:
        time_integ.configure(config, post_step=reactor.filter_state)

    snap_times, time_windows = create_time_windows(
        config['initial_time'], config['final_time'],
        num_snapshots
    )

    my_t = time_integ.timer.start()
    for i, (ti, tf) in enumerate(time_windows):
        cons_vars = time_integ.time_march(
            ti, tf,
            step_size,
            cons_vars,
        )
        prim_vars, dens, temp = reactor.equation_of_state(cons_vars)
        time_integ.rxr.set_temperature_guess(temp)
        print_step = i + init_step + 1
        print(
            f'Snapshot: {print_step}, Time: {(t_offset + tf):.4e}, '
            f'{bcolor.WARNING}Temperature: {temp.max():.4f} {bcolor.ENDC}'
        )
        if not i % config['write_freq']:
            write_step = i + init_step + 1
            print(f'{bcolor.OKGREEN}>-->--> Writing snapshot {write_step} '
                  f'to file{bcolor.ENDC}')
            write_to_file(
                config, write_step, t_offset + tf, step_size,
                reactor, cons_vars, prim_vars, temp
            )

    time_integ.timer.record('outer_time_loop', time_integ.timer.stop(my_t))
    time_integ.timer.report(total='outer_time_loop')

    time_integ.rxr.timer.record('outer_time_loop', time_integ.rxr.timer.stop(my_t))
    time_integ.rxr.timer.report(total='outer_time_loop')

    return


def run_flame(input_file):

    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)

        time_integration('explicit', config)

    return


if __name__ == '__main__':

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'input/flame_demo.yaml'

    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)
        mech = config['mech']
        sol = ct.Solution(f'mech/{mech}.yaml')
        pyro_cls = pyro.get_thermochem_class(sol)
        pyro_gas = make_pyro_object(pyro_cls, np)
        
        def make_array(res_list):
            return np.stack(res_list)

        pyro_gas._pyro_make_array = make_array
        pyro_gas.molecular_weights = pyro_gas.molecular_weights.reshape(
            -1, 1
        )
        pyro_gas.inv_molecular_weights = pyro_gas.inv_molecular_weights.reshape(
            -1, 1
        )

        time_integration()
