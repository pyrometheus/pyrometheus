"""Build generated Fortran into an importable extension module with
numpy.f2py, so that generated code can be compiled and run from the test
suite rather than only inspected symbolically.
"""

import importlib
import os
import shutil
import subprocess
import sys


toolchain_available = all(
    shutil.which(tool) is not None
    for tool in ("gfortran", "meson", "ninja")
)

toolchain_reason = "gfortran, meson and ninja are not all available"


def compile_fortran_module(tmp_path, source, module_name):
    """Compile *source* into an extension module named *module_name*
    under *tmp_path* and import it.

    :returns: The imported extension module. The generated Fortran
        module itself is an attribute of it, named after the module name
        passed to the code generator.
    """
    source_path = tmp_path / f"{module_name}.f90"
    f2cmap_path = tmp_path / ".f2py_f2cmap"
    source_path.write_text(source)
    f2cmap_path.write_text("dict(real=dict(sp='float', dp='double'))\n")

    env = dict(os.environ)
    env["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    )

    subprocess.run(
        [
            sys.executable, "-m", "numpy.f2py", "-c",
            f"{module_name}.f90", "-m", module_name,
            "--f2cmap", ".f2py_f2cmap",
            "--f90flags=-cpp -DPYROMETHEUS_CALLER_INDEXING=1",
        ],
        cwd=tmp_path, env=env, check=True,
        capture_output=True, text=True,
    )

    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(tmp_path))


# get_vt_energy_transfer_source is broken in the generated Fortran today
# (references an undeclared local `pressure`) -- a pre-existing,
# separately-flagged issue unrelated to the Jacobian capability this
# test targets (the Python template gained a pressure computation
# earlier; the Fortran template's copy of that subroutine was never
# updated to match). Stub its body out so the rest of the module --
# including get_net_production_rates_jacobian -- can actually be built
# and run.
_VT_SOURCE_MARKER = "subroutine get_vt_energy_transfer_source(&"


def stub_out_broken_vt_subroutine(source, num_species, num_temp):
    lines = source.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if _VT_SOURCE_MARKER in ln)
    end = next(
        i for i, ln in enumerate(lines)
        if i > start and "end subroutine get_vt_energy_transfer_source" in ln
    )
    stub = [
        "    subroutine get_vt_energy_transfer_source(&\n",
        "        & density, mass_fractions, temperature, omega_vt)\n",
        "        real(dp), intent(in) :: density\n",
        f"        real(dp), intent(in), dimension({num_species}) :: "
        "mass_fractions\n",
        f"        real(dp), intent(in), dimension({num_temp}) :: temperature\n",
        "        real(dp), intent(out) :: omega_vt\n",
        "        omega_vt = 0.d0\n",
        "    end subroutine get_vt_energy_transfer_source\n",
    ]
    return "".join(lines[:start] + stub + lines[end + 1:])
