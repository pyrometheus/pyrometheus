#!/usr/bin/env python
# CMake uses this script to generate the thermochemistry code for every backend
# and mechanism.

import cantera as ct
import argparse

from backends import BACKENDS


parser = argparse.ArgumentParser()
parser.add_argument("--mech",    type=str, required=True)
parser.add_argument("--backend", type=str, choices=BACKENDS.keys(), required=True)
parser.add_argument("--name",    type=str, required=True)
parser.add_argument("--output",  type=str, required=True)
args = parser.parse_args()

content = BACKENDS[args.backend].generate_code(
    ct.Solution(args.mech, "gas"), args.name)
with open(args.output, "w") as f:
    f.write(content)
