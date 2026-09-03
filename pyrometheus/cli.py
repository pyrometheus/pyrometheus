#!/usr/bin/env python3

import argparse
from importlib.metadata import version as distribution_version

from . import get_code_generators


def main():
    generators = get_code_generators()

    parser = argparse.ArgumentParser(
        prog="pyrometheus",
        description="Code generation for combustion thermochemistry"
        "based on Cantera.",
    )
    version = distribution_version("pyrometheus")
    parser.add_argument("--version",
                        action="version", version=f"%(prog)s {version}")
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

    args = parser.parse_args()

    generator = generators[args.lang]
    source = generator.generate(
        args.name, generator.load_mechanism(args.mech, args.phase)
    )

    with open(args.output, "w") as f:
        f.write(source)
