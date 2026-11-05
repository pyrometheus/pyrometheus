"""Build generated C++ into an importable extension module with
pybind11, so that generated headers can be compiled and run from the
test suite rather than only inspected symbolically.
"""

import importlib
import re
import shutil
import subprocess
import sys
import sysconfig

try:
    import pybind11
except ImportError:
    pybind11 = None


compiler = shutil.which("c++")

toolchain_available = compiler is not None and pybind11 is not None

toolchain_reason = "a C++ compiler and an importable pybind11 are required"

_static_function = re.compile(r"^\s*static\s+[\w:<>, ]+?\b(get_\w+)\s*\(", re.M)

_constants = [
    "num_species", "num_reactions", "num_temperatures",
    "one_atm", "gas_constant", "molecular_weights",
]


def generated_functions(header):
    """:returns: The names of the ``get_*`` functions in *header*, in
    order of appearance.
    """
    return list(dict.fromkeys(_static_function.findall(header)))


def binding_source(header_name, struct_name, module_name, functions):
    """:returns: A pybind11 module exposing every function and constant
    of ``pyro::<struct_name><double>``.
    """
    definitions = "\n".join(
        f'    m.def("{function}", &mech_t::{function});'
        for function in functions
    )
    constants = "\n".join(
        f'    m.attr("{constant}") = pybind11::cast(mech_t::{constant});'
        for constant in _constants
    )
    return f"""
#include "{header_name}"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using mech_t = pyro::{struct_name}<double>;

PYBIND11_MODULE({module_name}, m)
{{
{definitions}
{constants}
}}
"""


def compile_cpp_module(tmp_path, header, struct_name, module_name,
                       optimization="-O1"):
    """Compile *header*, which defines ``pyro::<struct_name>``, into an
    extension module named *module_name* under *tmp_path* and import it.

    :returns: The imported extension module, with one function per
        generated ``get_*`` function.
    """
    header_name = f"{struct_name}.hpp"
    (tmp_path / header_name).write_text(header)
    binding_path = tmp_path / f"{module_name}.cpp"
    binding_path.write_text(binding_source(
        header_name, struct_name, module_name, generated_functions(header)
    ))

    output = tmp_path / f"{module_name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    command = [
        compiler, "-std=c++17", optimization, "-shared", "-fPIC",
        "-Wall", "-Wextra",
        f"-I{tmp_path}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        str(binding_path), "-o", str(output),
    ]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C++ compilation failed:\n{result.stderr[-5000:]}")

    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(tmp_path))
