"""Pyrometheus example using PyOpenCL."""

import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
# from pytools.obj_array import make_obj_array

from meshmode.array_context import (
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)

import cantera


def main(ctx_factory=cl.create_some_context,
         cti_filename=None, actx_class=PyOpenCLArrayContext):
    """Run the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    cantera_soln = cantera.Solution(cti_filename, phase_id="gas")

    import pyrometheus
    pyro_class = pyrometheus.get_thermochem_class(cantera_soln)
    pyro = pyro_class(actx)

    i_fu = cantera_soln.species_index("C2H4")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")

    temperature_0 = 1000
    temperature_1 = 300
    delta_t = temperature_1 - temperature_0

    y_0 = np.zeros(cantera_soln.n_species)
    y_0[i_ox] = .21
    y_0[i_di] = .79

    y_1 = np.zeros(cantera_soln.n_species)
    y_1[i_fu] = 1
    delta_y = y_1 - y_0

    num_zeta = 1000
    d_zeta = 1.0 / (num_zeta - 1)

    temperature = actx.from_numpy(np.array([temperature_0 + i*d_zeta*delta_t
                                            for i in range(num_zeta)]))
    print(f"{temperature.shape=}")

    mf = np.array([np.array([y_0[i] + j*d_zeta*delta_y[i] for j in range(num_zeta)])
                   for i in range(cantera_soln.n_species)])

    mass_fractions = actx.from_numpy(mf)

    pressure = actx.from_numpy(np.array([cantera.one_atm for _ in range(num_zeta)]))

    get_density = actx.compile(pyro.get_density)
    get_rates = actx.compile(pyro.get_net_production_rates)

    density = get_density(pressure, temperature, mass_fractions)

    rates = get_rates(density, temperature, mass_fractions)

    # print it out to make sure it actually gets computed
    print(f"{rates=}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pyrometheus PyOpenCL Example")
    parser.add_argument("-l", "--lazy", action="store_true",
                        help="switch to a lazy computation mode")
    parser.add_argument("-i", "--cti_file",
                        help="full path to mechanism CTI file")
    args = parser.parse_args()

    if args.lazy:
        actx_class = PytatoPyOpenCLArrayContext
    else:
        actx_class = PyOpenCLArrayContext

    cti_filename = "./mech.cti"
    if args.cti_file:
        cti_filename = args.cti_file

    main(cti_filename=cti_filename, actx_class=actx_class)
