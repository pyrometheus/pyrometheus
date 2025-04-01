import sys
import os
import yaml
import h5py
import numpy as np
import cantera as ct
import pyrometheus as pyro
from flop_counter import FLOPCounter
from make_pyro import make_pyro_object
from make_mixture_arrays import from_equiv_ratio, from_ignition_file
from reactors import HomogeneousReactor
from time_integ import (
    RungeKutta, CrankNicolson, create_time_windows)


def time_integration(scheme: int, config: dict, reactor: HomogeneousReactor):

    if scheme == 'implicit':
        step_size = 10 * config['step_size']
        num_snapshots = config['num_snapshots'] // 10
        time_integ = CrankNicolson(reactor)
    else:
        step_size = config['step_size']
        num_snapshots = config['num_snapshots']
        time_integ = RungeKutta(reactor)

    time_integ.configure(config)

    temperature = config['mixture']['temperature']
    mass_fractions = from_equiv_ratio(
        config['mixture']['equiv_ratio'], reactor.pyro_gas,
        reactor.pyro_gas.usr_np
    )
    time_integ.rxr.set_density_and_energy(
        reactor.pyro_gas.one_atm, temperature, mass_fractions
    )

    from numpy import empty, append
    state = empty(
        (num_snapshots,) + (reactor.num_species + 1,),
    )
    state[0] = append(mass_fractions, temperature)

    snap_times, time_windows = create_time_windows(
        config['initial_time'], config['final_time'],
        num_snapshots
    )
    for i, (ti, tf) in enumerate(time_windows):
        mass_fractions = time_integ.time_march(
            ti, tf,
            step_size,
            mass_fractions,
        )
        temperature = reactor.get_temperature(mass_fractions)
        state[i + 1] = append(mass_fractions, temperature)
        print(
            f'Time: {tf:.4e},'
            f'Temperature: {temperature:.4f}'
        )

    mode = 'a' if os.path.isfile(config['output_file_name']) else 'w'
    db = h5py.File(config['output_file_name'], mode)
    g = db.create_group(scheme)
    g.create_dataset('time', data=snap_times)
    g.create_dataset('state', data=state)
    db.close()
    return


def quick_example_calculation(pyro_gas, sol):
    """
    Demonstrate how to call Pyro routines to compute molar
    production rates and, for JAX NumPy, their derivatives.
    """
    # Set the state of the mixture
    print('Quick example calculation')
    print('Start with simple arrays')
    temperature = 1200
    sol.TP = temperature, pyro_gas.one_atm
    sol.set_equivalence_ratio(
        phi=1, fuel='H2:1',
        oxidizer='O2:0.21, N2:0.79'
    )
    mass_fractions = pyro_gas._pyro_make_array(sol.Y)

    # Compute the density and molar net production rates
    density = pyro_gas.get_density(
        pyro_gas.one_atm, temperature, mass_fractions
    )
    omega = pyro_gas.get_net_production_rates(
        density, temperature, mass_fractions
    )

    # Compare against Cantera
    assert pyro_gas.usr_np.linalg.norm(
        omega - sol.net_production_rates
    ) < 1e-14

    print('Example with simple arrays successful... '
          'Now showcase data on grids')

    # Set the mixture state
    num_x = 256
    num_y = 256
    mass_fractions = pyro_gas._pyro_make_array([
        pyro_gas.usr_np.tile(y, (num_y, num_x)) for y in sol.Y
    ])
    temperature = 1200 * pyro_gas.usr_np.ones((num_y, num_x))

    # Compute the density and molar net production rates
    density = pyro_gas.get_density(
        pyro_gas.one_atm, temperature, mass_fractions
    )
    omega = pyro_gas.get_net_production_rates(
        density, temperature, mass_fractions
    )
    print(omega.shape)

    for i, w in enumerate(omega):
        print(f'Species {i}: {w.shape}')
        assert pyro_gas.usr_np.linalg.norm(
            w - sol.net_production_rates[i]
        ) < 1e-14


def ad_with_jax(pyro_gas, sol):

    temp_guess = 1200
    density, energy, mass_fractions = from_ignition_file(pyro_gas)

    def chemical_source_term(mass_fractions):
        temperature = pyro_gas.get_temperature(
            energy, temp_guess, mass_fractions
        )
        return (
            pyro_gas.molecular_weights *
            pyro_gas.get_net_production_rates(
                density, temperature, mass_fractions
            )
        )

    from jax import jacfwd
    jacobian = jacfwd(chemical_source_term)
    j_ad = jacobian(mass_fractions)
    w_ad = chemical_source_term(mass_fractions)

    delta = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

    for d in delta:
        j_fd = pyro_gas.usr_np.array([
            (
                chemical_source_term(mass_fractions + d * v) - w_ad
            ) / d
            for v in pyro_gas.usr_np.eye(pyro_gas.num_species)
        ]).T
        rel_error = (
            pyro_gas.usr_np.linalg.norm(j_ad - j_fd, 'fro') /
            pyro_gas.usr_np.linalg.norm(j_ad, 'fro')
        )
        print(f' Perturbation: {d:.2e}, Error: {rel_error:.4e}')

    return


def run_with_pytato(input_file, pyro_cls, sol, run_eager=True):

    import pytato as pt
    pyro_gas = make_pyro_object(pyro_cls, pt)

    inner_length = 16
    num_x = 64 * inner_length
    num_y = 64 * inner_length
    print(f'Running with num_x = {num_x}, num_y = {num_y}')
    temperature = pt.make_placeholder(
        name='temperature', shape=(num_x, num_y),
        dtype='float64'
    )
    mass_fractions = pt.make_placeholder(
        name='mass_fractions', shape=(pyro_gas.num_species, num_x, num_y),
        dtype='float64'
    )

    # Trace the computational graph
    density = pyro_gas.get_density(
        pyro_gas.one_atm, temperature, mass_fractions
    )
    omega = pyro_gas.get_net_production_rates(
        density, temperature, mass_fractions
    )

    pyro_graph = pt.make_dict_of_named_arrays({
        f'omega{i}': om for i, om in enumerate(omega)
    })
    pyro_kernel = pt.generate_loopy(pyro_graph).program

    # Count flops
    flop_counter = FLOPCounter()
    flop_counter(pyro_graph)
    print(f'Roofline: FLOP Count: {flop_counter.num_flops}, '
          f'log count: {flop_counter.num_log}, '
          f'exp count: {flop_counter.num_exp}')
    print(f'Roofline: Memory access: {flop_counter.mem_access}')

    # Execute in OpenCL with numerical data
    import pyopencl as cl
    from pyopencl.array import to_device
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    pyro_exec = pyro_kernel.executor(queue)
    num_rounds = 4

    num_flops = flop_counter.total_flops(
        ctx.devices[0].name, num_x * num_y
    )

    temp = 1200 * np.ones((num_x, num_y))
    mass_frac = np.zeros((pyro_gas.num_species, num_x, num_y))

    for i, y in enumerate(from_equiv_ratio(1, pyro_gas, np)):
        mass_frac[i] = y * np.ones((num_x, num_y))

    temp_dev = to_device(
        queue, temp
    )
    mass_frac_dev = to_device(
        queue, mass_frac
    )

    from time import time
    if run_eager:
        queue.finish()
        t_start = time()
        for _ in range(num_rounds):
            _ = pyro_exec(
                queue, temperature=temp_dev, mass_fractions=mass_frac_dev
            )
        queue.finish()
        t_elapsed = time() - t_start
        t_ppt = 1e9 * t_elapsed / num_x / num_y / num_rounds
        rate_a = num_rounds * num_flops / t_elapsed / 1e9
        print(f'time: {(t_elapsed / num_rounds):.4e} s')
        print(f'time per point: {t_ppt:.4e} ns')
        print(f'rate: {rate_a:.4e} G ops/s')

    # Now parallelize
    from loopy import split_iname
    for i_sp in range(pyro_gas.num_species):
        for i_ax in [0, 1]:
            iname = f'omega{i_sp}_dim{i_ax}'
            pyro_kernel = split_iname(
                pyro_kernel, iname, inner_length, outer_tag=f'g.{i_ax}',
                inner_tag=f'l.{i_ax}'
            )

    pyro_exec = pyro_kernel.executor(queue)
    _ = pyro_exec(queue, temperature=temp_dev, mass_fractions=mass_frac_dev)
    # Run the SIMD program
    queue.finish()
    t_start = time()
    for _ in range(num_rounds):
        _ = pyro_exec(
            queue, temperature=temp_dev, mass_fractions=mass_frac_dev
        )
    queue.finish()
    t_elapsed = time() - t_start
    t_ppt = 1e9 * t_elapsed / num_x / num_y / num_rounds
    rate_b = num_rounds * num_flops / t_elapsed / 1e9
    print(f'time: {(t_elapsed / num_rounds):.4e} s')
    print(f'time per point: {t_ppt:.4e} ns')
    print(f'rate: {rate_b:.4e} G ops/s')

    # print(f'speed-up: {(rate_b / rate_a):.4e}')
    return


def run_with_jax_numpy(input_file, pyro_cls, sol):

    import jax.numpy as jnp
    pyro_gas = make_pyro_object(pyro_cls, jnp)

    # Quick example of a calculation
    quick_example_calculation(pyro_gas, sol)
    ad_with_jax(pyro_gas, sol)
    exit()

    # Implicit time integration
    reactor = HomogeneousReactor(pyro_gas)
    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)
        time_integration('implicit', config, reactor)


def run_with_numpy(input_file, pyro_cls, sol):

    import numpy as np
    pyro_gas = make_pyro_object(pyro_cls, np)

    # Quick example of a calculation
    quick_example_calculation(pyro_gas, sol)

    # Explicit time integration
    reactor = HomogeneousReactor(pyro_gas)

    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)
        time_integration('explicit', config, reactor)

    return


def run_pyro():

    sol = ct.Solution('sandiego.yaml')
    pyro_cls = pyro.codegen.python.get_thermochem_class(sol)

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'input/autoign_demo.yaml'

    # run_with_numpy(input_file, pyro_cls, sol)
    run_with_jax_numpy(input_file, pyro_cls, sol)
    # run_with_pytato(input_file, pyro_cls, sol, run_eager=False)


if __name__ == '__main__':
    run_pyro()
