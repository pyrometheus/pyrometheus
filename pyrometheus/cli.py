#!/usr/bin/env python3

import argparse
from importlib.metadata import version as _get_version

import cantera as ct

from . import get_code_generators


def generate_surface_source(generator, args):
    """Surface source for *args*, with the backend's own way of naming the gas side.

    Each backend refers to the separately generated gas-phase code differently --
    a Fortran module to `use`, a C++ header to include, a Python class passed in --
    so only the two that name it in their source take the extra argument.
    """
    interface = ct.Interface(args.mech, args.phase)

    if generator.get_name() == "fortran":
        return generator.generate_surface(args.name, interface,
                                          gas_module_name=args.gas_name)
    if generator.get_name() == "cpp":
        return generator.generate_surface(args.name, interface,
                                          gas_header_name=f"{args.gas_name}.hpp")
    return generator.generate_surface(args.name, interface)


def main():
    generators = get_code_generators()

    parser = argparse.ArgumentParser(
        prog="pyrometheus",
        description="Code generation for combustion thermochemistry"
        "based on Cantera.",
    )
    parser.add_argument("--version",
                        action="version",
                        version=f"%(prog)s {_get_version('pyrometheus')}")
    parser.add_argument("-l", "--lang", "--language",
                        help="Language to generate code for.",
                        choices=generators.keys(), required=True)
    parser.add_argument("-m", "--mech", "--mechanism",
                        help="Path to the mechanism file.", required=True)
    parser.add_argument("-o", "--output",
                        help="Path to the output file.", required=True)
    parser.add_argument("-n", "--name", "--namespace",
                        help="Namespace to use for the generated code.",
                        required=True)
    parser.add_argument("-p", "--phase",
                        help="Phase name to use for the generated code.")
    parser.add_argument("-s", "--surface",
                        action="store_true",
                        help="Generate heterogeneous (surface) kinetics for an "
                             "interface phase, rather than gas-phase "
                             "thermochemistry. The adjacent phases are read "
                             "from the mechanism.")
    parser.add_argument("--gas-name",
                        default="thermochem",
                        help="With --surface: name of the separately generated "
                             "gas-phase module or header the surface code "
                             "refers to (default: thermochem).")

    args = parser.parse_args()

    generator = generators[args.lang]

    if args.surface:
        # Adjacent phases are not passed: Cantera resolves them from the
        # mechanism's own adjacent-phases list, and supplying a partial list
        # overrides that rather than adding to it.
        source = generate_surface_source(generator, args)
    else:
        solution = ct.Solution(args.mech, args.phase)
        # A phase that couples others is an interface; generating gas-phase code
        # from it silently produces something that is not what was asked for.
        if solution.n_total_species > solution.n_species:
            parser.error(
                f"phase '{args.phase}' couples other phases, so it is an "
                "interface. Pass --surface to generate its heterogeneous "
                "kinetics.")
        source = generator.generate(args.name, solution)

    with open(args.output, "w") as f:
        f.write(source)
