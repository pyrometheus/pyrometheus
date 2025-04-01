import sys
import yaml
import h5py
import cantera as ct
import pyrometheus as pyro
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


def form_file_name(step):
    if step:
        from numpy import floor, log10
        num_zeros = floor(log10(step)).astype(int)
        step_str = (8 - num_zeros) * '0' + f'{step}'
    else:
        step_str = 9*'0'

    return 'output/flame/flame_' + step_str + '.h5'


def initialize(config, pyro_gas):

    # Open h5 database
    db = h5py.File(form_file_name(config['init_step']), 'r')

    # Mesh
    x = db['domain/mesh/x'][:]
    mesh = Mesh(x=x)
    op = Operators(mesh, pyro_gas.usr_np)
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
    return t_offset, cons_vars, mesh, reactor


def write_to_file(step,
                  time,
                  step_size,
                  reactor,
                  cons_vars: FlameState,
                  prim_vars: FlameState,
                  temperature):

    # Write to file
    file_name = form_file_name(step)
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


def time_integration(scheme: int, config: dict, pyro_gas):

    # Initialize state, mesh, and reactor
    t_offset, cons_vars, mesh, reactor = initialize(config, pyro_gas)
    print(f'mesh size: {mesh.num_x}, array shape: {cons_vars.momentum.shape}')

    # Time integrator
    step_size = config['step_size']
    num_snapshots = config['num_snapshots']
    time_integ = RungeKutta(reactor,)

    time_integ.configure(config, post_step=reactor.filter_state)

    snap_times, time_windows = create_time_windows(
        config['initial_time'], config['final_time'],
        num_snapshots
    )
    for i, (ti, tf) in enumerate(time_windows):
        cons_vars = time_integ.time_march(
            ti, tf,
            step_size,
            cons_vars,
        )
        prim_vars, dens, temp = reactor.equation_of_state(cons_vars)

        print_step = i + config['init_step']
        print(
            f'Snapshot: {print_step}, Time: {(t_offset + tf):.4e}, '
            f'{bcolor.WARNING}Temperature: {temp.max():.4f} {bcolor.ENDC}'
        )

        if i and not i % config['write_freq']:
            write_step = i + config['init_step']
            print(f'{bcolor.OKGREEN}>-->--> Writing snapshot {write_step} '
                  f'to file{bcolor.ENDC}')
            write_to_file(
                write_step, t_offset + tf, step_size,
                reactor, cons_vars, prim_vars, temp
            )

    return


def run_flame(input_file, pyro_cls, sol):

    import numpy as np
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

    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)
        time_integration('explicit', config, pyro_gas)

    return


def run_pyro():

    sol = ct.Solution('sandiego.yaml')
    pyro_cls = pyro.codegen.python.get_thermochem_class(sol)

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'input/flame_demo.yaml'

    run_flame(input_file, pyro_cls, sol)


if __name__ == '__main__':
    run_pyro()
