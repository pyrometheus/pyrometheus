"""Generate, compile, and exercise Bandit's Fortran backend for the PLATO
air5/TTv mechanism: species production rates and the Landau-Teller VT
energy-transfer source term.

Requires a Fortran compiler (e.g. gfortran) and numpy's f2py build
dependencies (``pip install meson ninja``) to compile the generated source
into an importable extension module.
"""
import importlib
import os
import subprocess
import sys

import numpy as np

from pyrometheus.bandit.impl.plato import PlatoMechanism
from pyrometheus.codegen.fortran_bandit import FortranBanditCodeGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_NAME = "libpyro_fortran_air5"
SOURCE_PATH = os.path.join(HERE, f"{MODULE_NAME}.f90")
F2CMAP_PATH = os.path.join(HERE, ".f2py_f2cmap")


def make_mechanism():
    return PlatoMechanism(
        mixture='air5',
        reaction_set='air5',
        transfer='TTv',
        plato_db_path='/Users/ecisneros/Packages/plato-database/',
        hardcode_params=True
    )


def generate_source(mech):
    source = FortranBanditCodeGenerator.generate(MODULE_NAME, mech)
    with open(SOURCE_PATH, "w") as f:
        f.write(source)
    with open(F2CMAP_PATH, "w") as f:
        f.write("dict(real=dict(sp='float', dp='double'))\n")


def compile_module():
    """Compile the generated Fortran source into an importable extension
    module with numpy.f2py. PYROMETHEUS_CALLER_INDEXING=1 makes
    get_species_name/get_species_index use 0-based indices, matching the
    Python caller's convention; -cpp forces preprocessing of the
    #ifdef/#define directives regardless of source-file extension.
    """
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    subprocess.run(
        [
            sys.executable, "-m", "numpy.f2py", "-c",
            f"{MODULE_NAME}.f90", "-m", MODULE_NAME,
            "--f2cmap", ".f2py_f2cmap",
            "--f90flags=-cpp -DPYROMETHEUS_CALLER_INDEXING=1",
        ],
        cwd=HERE, env=env, check=True,
    )


def import_module():
    sys.path.insert(0, HERE)
    module = importlib.import_module(MODULE_NAME)
    # f2py nests everything under a submodule sharing the Fortran module's
    # name, since the generated code declares "module libpyro_fortran_air5".
    return getattr(module, MODULE_NAME)


if __name__ == "__main__":
    mech = make_mechanism()
    generate_source(mech)
    compile_module()
    fort = import_module()
    mech.finalize()

    print(f"Compiled {MODULE_NAME}: "
          f"{fort.num_species} species, {fort.num_reactions} reactions, "
          f"{fort.num_temperatures} temperatures")

    # Representative post-shock state: hot translational temperature,
    # vibrational modes still cold, mostly-dissociated air.
    heavy_temperature = 10000.0
    vibrational_temperature = 300.0
    temperature = np.array([heavy_temperature, vibrational_temperature])
    density = 1.0e-3
    mass_fractions = np.array([0.0, 0.0, 0.767, 0.0, 0.233])  # N2, O2 only

    species_source = fort.get_net_production_rates(
        density, temperature, mass_fractions
    )
    vt_energy_source = fort.get_vt_energy_transfer_source(
        density, mass_fractions, temperature
    )

    print("\nSpecies production rates omega_i [kg/(m^3 s)]:")
    for name, rate in zip(
            [s.decode().strip() for s in fort.species_names], species_source):
        print(f"  {name:>4s}: {rate: .6e}")

    print(f"\nVT energy transfer source Omega_VT [W/m^3]: {vt_energy_source:.6e}")
    print("(positive: energy flows from the translational to the "
          "vibrational mode, since T_h > T_v)")
