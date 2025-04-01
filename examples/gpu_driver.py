import numpy as np
import cantera as ct
import pyrometheus as pyro
import loopy as lp
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from make_pyro import make_pyro_object
from make_mixture_arrays import from_equiv_ratio


def hack_pow_code_str(code_str):
    tmp = code_str.splitlines()
    tmp[2] = '__device__ ' + tmp[2]
    return '\n'.join(tmp)


def get_local_state(pyro_gas):
    temp_real = 1200
    mass_frac = from_equiv_ratio(1, pyro_gas, np)
    density = pyro_gas.get_density(
        pyro_gas.one_atm, temp_real, mass_frac
    )
    return temp_real, density, mass_frac


def measure_exec_time(grid, block, cuda_prg,
                      density_dev, temp_dev,
                      mass_frac_dev, omega_dev):

    num_rounds = 1000

    s_evt = drv.Event()
    e_evt = drv.Event()

    s_evt.record()
    for r in range(num_rounds):
        cuda_prg(
            density_dev, temp_dev, mass_frac_dev, *omega_dev,
            grid=grid, block=block,
        )

    e_evt.record()
    e_evt.synchronize()
    return 1e-3 * s_evt.time_till(e_evt)/num_rounds


def execute_kernel(pyro_gas, inner_length, num_x, num_y, kernel, knl_tag,):

    print(f'Running kernel with tag {knl_tag}')
    kernel = kernel.copy(target=lp.CudaTarget())
    code_str = lp.generate_code_v2(kernel).device_code()

    # Check for devices
    num_dev = drv.Device.count()
    if not num_dev:
        print('Found no devices, exiting gracefully.')
        exit()
    else:
        print(f'Found {num_dev} devices')

    print(f'Device name: {drv.Device(0).name()}')

    # Compile cuda code
    prg = SourceModule(
        hack_pow_code_str(code_str),
        options=['-maxrregcount=32']
    ).get_function('chemical_source_terms')

    t, d, mf = get_local_state(pyro_gas)
    dens = d * np.ones((num_x, num_y))
    temp = t * np.ones((num_x, num_y))
    mass_frac = np.zeros((pyro_gas.num_species, num_x, num_y))
    omega = [
        np.zeros_like(temp)
        for i in range(pyro_gas.num_species)
    ]

    for i, y in enumerate(mf):
        mass_frac[i] = y * np.ones((num_x, num_y))

    dens_dev = gpuarray.to_gpu(dens)
    temp_dev = gpuarray.to_gpu(temp)
    mass_frac_dev = gpuarray.to_gpu(mass_frac)
    omega_dev = [gpuarray.to_gpu(w) for w in omega]

    # Call one time
    grid = (
        (num_x + inner_length - 1)//inner_length,
        (num_y + inner_length - 1)//inner_length,
        1
    )
    block = (inner_length, inner_length, 1)
    print(f'grid = {grid}, block = {block}')
    prg(
        dens_dev, temp_dev, mass_frac_dev, *omega_dev,
        grid=grid, block=block,
    )

    # Now measure execution time
    t = measure_exec_time(
        grid, block, prg,
        dens_dev, temp_dev, mass_frac_dev,
        omega_dev
    )

    t_per_point = 1e9 * t / (num_x * num_y)
    print(10*'-->-->-->')
    print(f'-->-->--> kernel tag: {knl_tag}')
    print(f'-->-->--> total call time: {(1e6 * t):.6e} us')
    print(f'-->-->--> time per point: {t_per_point:.6e} ns')
    print(10*'-->-->-->', '\n')

    return


def run(pyro_gas,):

    inner_length = 32
    num_x = 32 * inner_length
    num_y = 32 * inner_length

    print('Hello from the GPU driver')
    print(f'Problem size: ({num_x}, {num_y})')

    density = pt.make_placeholder(
        name='density', shape=(num_x, num_y),
        dtype='float64'
    )
    temperature = pt.make_placeholder(
        name='temperature', shape=(num_x, num_y),
        dtype='float64'
    )
    mass_fractions = pt.make_placeholder(
        name='mass_fractions',
        shape=(pyro_gas.num_species, num_x, num_y),
        dtype='float64'
    )

    # Trace the computational graph
    def chemical_source_term(density, temperature, mass_fractions):
        omega = pyro_gas.get_net_production_rates(
            density, temperature, mass_fractions
        )
        return pyro_gas._pyro_make_array([
            w * om for w, om in zip(
                pyro_gas.molecular_weights,
                omega
            )
        ])

    omega = chemical_source_term(
        density, temperature, mass_fractions
    )
    pyro_graph = pt.make_dict_of_named_arrays({
        f'omega{i}': w for i, w in enumerate(omega)
    })

    # Loopy kernels
    pyro_kernel_orig = pt.generate_loopy(
        pyro_graph,
        function_name='chemical_source_terms',
    ).program

    # Fuse loops
    all_inames = pyro_kernel_orig['chemical_source_terms'].inames
    inames_rnm_i = [n for n in all_inames if n[-1] == '0']
    inames_rnm_j = [n for n in all_inames if n[-1] == '1']

    pyro_kernel_fuse = pyro_kernel_orig
    pyro_kernel_fuse = lp.rename_inames(
        pyro_kernel_fuse, inames_rnm_i, 'i'
    )
    pyro_kernel_fuse = lp.rename_inames(
        pyro_kernel_fuse, inames_rnm_j, 'j'
    )

    # Split inames
    pyro_kernel_split = pyro_kernel_fuse
    pyro_kernel_split = lp.split_iname(
        pyro_kernel_split, 'i', inner_length,
        outer_tag='g.0', inner_tag='l.0'
    )
    pyro_kernel_split = lp.split_iname(
        pyro_kernel_split, 'j', inner_length,
        outer_tag='g.1', inner_tag='l.1'
    )

    # Add prefetch
    pyro_kernel_fetch = pyro_kernel_split
    pyro_kernel_fetch = lp.add_prefetch(
        pyro_kernel_fetch, 'density',
    )
    pyro_kernel_fetch = lp.add_prefetch(
        pyro_kernel_fetch, 'temperature',
    )
    pyro_kernel_fetch = lp.add_prefetch(
        pyro_kernel_fetch, 'mass_fractions'
    )
    pyro_kernel_fetch = lp.add_inames_for_unused_hw_axes(
        pyro_kernel_fetch, 'id:*fetch*'
    )

    # Run kernel
    execute_kernel(
        pyro_gas, inner_length, num_x, num_y,
        pyro_kernel_split, 'split'
    )
    execute_kernel(
        pyro_gas, inner_length, num_x, num_y,
        pyro_kernel_fetch, 'fetch'
    )
    return


if __name__ == '__main__':
    sol = ct.Solution('sandiego.yaml')
    pyro_cls = pyro.codegen.python.get_thermochem_class(sol)

    import pytato as pt
    pyro_gas = make_pyro_object(pyro_cls, pt)
    run(pyro_gas)
