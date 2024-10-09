from backends import BACKENDS


def pytest_addoption(parser):
    parser.addoption("--backend", type=str,
        choices=BACKENDS.keys(), required=True,
        help="Backend to use for code generation")
