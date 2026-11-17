# This file is a Pytest configuration file.

import plato_cassette
from backends import BACKENDS

# Before any test imports plato: record it, or stand in for it.
plato_cassette.install()


def pytest_addoption(parser):
    parser.addoption("--backend", type=str,
        choices=BACKENDS.keys(), required=True,
        help="Backend (Pyrometheus implementation) to use for code generation")
