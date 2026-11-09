#!/usr/bin/env python3

import argparse
import warnings
from importlib.metadata import version as _get_version

from . import get_code_generators
from .bandit import library_options, load_mechanism


def main(argv=None):
    generators = get_code_generators()

    parser = argparse.ArgumentParser(
        prog="pyrometheus",
        description="Code generation for thermochemistry, from mechanisms "
        "read by Cantera, Mutation++ or Plato.",
    )
    parser.add_argument("--version",
                        action="version",
                        version=f"%(prog)s {_get_version('pyrometheus')}")
    parser.add_argument("-l", "--lang", "--language",
                        help="Language to generate code for. The "
                        "'-legacy' generators are the deprecated 1.x ones, "
                        "which read Cantera mechanisms only.",
                        choices=generators.keys(), required=True)
    parser.add_argument("--library", choices=list(library_options),
                        default="cantera",
                        help="Library that reads the mechanism.")
    parser.add_argument("-m", "--mech", "--mechanism",
                        help="Cantera mechanism file, or the Mutation++ or "
                        "Plato mixture name.", required=True)
    parser.add_argument("-o", "--output",
                        help="Path to the output file.", required=True)
    parser.add_argument("-n", "--name", "--namespace",
                        help="Namespace to use for the generated code.",
                        required=True)
    parser.add_argument("-p", "--phase",
                        help="Cantera: phase to read from the mechanism file.")
    parser.add_argument("--state-model",
                        help="Mutation++: state model, such as ChemNonEq1T "
                        "or ChemNonEqTTv.")
    parser.add_argument("--thermo-database",
                        help="Mutation++: thermodynamic database, such as "
                        "NASA-9 or NASA-7.")
    parser.add_argument("--reaction-set", help="Plato: reaction set.")
    parser.add_argument("--transfer",
                        help="Plato: energy transfer model, such as TTv.")
    parser.add_argument("--plato-db-path",
                        help="Plato: database path (default: $PLATO_DB).")

    args = parser.parse_args(argv)

    generator = generators[args.lang]
    if args.lang.endswith("-legacy"):
        if args.library != "cantera":
            parser.error(f"{args.lang} reads Cantera mechanisms only")
        warnings.warn(
            f"{args.lang} is the deprecated 1.x generator; use "
            f"{args.lang[:-len('-legacy')]} instead.",
            FutureWarning, stacklevel=2,
        )
        mechanism = generator.load_mechanism(args.mech, args.phase)
    else:
        given = {
            option: getattr(args, option)
            for options in library_options.values() for option in options
            if getattr(args, option) is not None
        }
        foreign = set(given) - set(library_options[args.library])
        if foreign:
            parser.error(
                f"{args.library} mechanisms take no "
                + ", ".join("--" + option.replace("_", "-")
                            for option in sorted(foreign))
            )
        mechanism = load_mechanism(args.library, args.mech, **given)

    try:
        source = generator.generate(args.name, mechanism)
    finally:
        # Plato allows one live instance per process.
        if hasattr(mechanism, "finalize"):
            mechanism.finalize()

    with open(args.output, "w") as f:
        f.write(source)
