"""Bandit is Pyrometheus' default code generator: the plain language
names, the module-level functions and the command line all lead to it,
while the 1.x generators stay reachable, deprecated, under ``-legacy``
names and for callers that pass a Cantera ``Solution``.
"""

import os
import pathlib
import warnings

import cantera as ct
import pytest

import pyrometheus
from bandit_parity import mutation_loads
from pyrometheus.bandit import load_mechanism
from pyrometheus.bandit.impl.cantera import CanteraMechanism
from pyrometheus.cli import main
from pyrometheus.codegen.cpp import CppCodeGenerator
from pyrometheus.codegen.cpp_bandit import CppBanditCodeGenerator
from pyrometheus.codegen.fortran import FortranCodeGenerator
from pyrometheus.codegen.fortran_bandit import FortranBanditCodeGenerator
from pyrometheus.codegen.python import PythonCodeGenerator
from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator

mech_path = str(pathlib.Path(__file__).parent / "mechs" / "sandiego.yaml")

bandit_generators = {
    "python": PythonBanditCodeGenerator,
    "cpp": CppBanditCodeGenerator,
    "fortran": FortranBanditCodeGenerator,
}
legacy_generators = {
    "python-legacy": PythonCodeGenerator,
    "cpp-legacy": CppCodeGenerator,
    "fortran-legacy": FortranCodeGenerator,
}


# {{{ generator registry

@pytest.mark.parametrize("language", bandit_generators)
def test_plain_language_names_are_bandit(language):
    generator = pyrometheus.get_code_generators()[language]
    assert generator is bandit_generators[language]
    assert generator.get_name() == language


@pytest.mark.parametrize("language", legacy_generators)
def test_legacy_generators_keep_a_name(language):
    generator = pyrometheus.get_code_generators()[language]
    assert generator is legacy_generators[language]
    assert generator.get_name() == language

# }}}


# {{{ module-level functions

def test_module_level_generation_is_bandit():
    mech = CanteraMechanism(mech_path)
    assert pyrometheus.generate("pyro", mech) == \
        PythonBanditCodeGenerator.generate("pyro", mech)
    gas = pyrometheus.get_thermochem_class(mech)(None)
    assert gas.species_names == mech.species_names


def test_solutions_take_the_deprecated_legacy_path():
    sol = ct.Solution(mech_path)
    with pytest.warns(DeprecationWarning, match="CanteraMechanism"):
        source = pyrometheus.generate("pyro", sol)
    assert source == PythonCodeGenerator.generate("pyro", sol)
    with pytest.warns(DeprecationWarning, match="CanteraMechanism"):
        pyrometheus.get_thermochem_class(sol)


def test_legacy_classes_do_not_warn_by_themselves():
    # pyrometheus.flamelets is built on them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        PythonCodeGenerator.generate("pyro", ct.Solution(mech_path))

# }}}


# {{{ mechanisms

def test_cantera_mechanisms_accept_a_solution():
    sol = ct.Solution(mech_path)
    assert CanteraMechanism(sol).species_names == sol.species_names


def test_cantera_mechanisms_take_a_phase():
    mech = CanteraMechanism(mech_path, phase="gas")
    assert mech.species_names == ct.Solution(mech_path, "gas").species_names


def test_load_mechanism_reads_cantera():
    mech = load_mechanism("cantera", mech_path, phase="gas")
    assert isinstance(mech, CanteraMechanism)


@pytest.mark.skipif(not mutation_loads(), reason="Mutation++ unavailable")
def test_load_mechanism_reads_mutation():
    from pyrometheus.bandit.impl.mutation import MutationMechanism
    mech = load_mechanism("mutation", "air_5", state_model="ChemNonEq1T")
    assert isinstance(mech, MutationMechanism)
    assert mech.num_temp == 1


def test_load_mechanism_rejects_options_of_other_libraries():
    with pytest.raises(TypeError, match="transfer"):
        load_mechanism("cantera", mech_path, transfer="TTv")


def test_load_mechanism_rejects_unknown_libraries():
    with pytest.raises(ValueError, match="unknown library"):
        load_mechanism("chemkin", mech_path)

# }}}


# {{{ command line

def run_cli(tmp_path, *arguments):
    output = tmp_path / "generated"
    main(["--name", "pyro", "--output", str(output), *arguments])
    return output.read_text()


@pytest.mark.parametrize("language", bandit_generators)
def test_cli_generates_bandit_code_by_default(tmp_path, language):
    source = run_cli(tmp_path, "--language", language, "--mechanism",
                     mech_path, "--phase", "gas")
    assert source == bandit_generators[language].generate(
        "pyro", CanteraMechanism(mech_path, phase="gas")
    )


def test_cli_legacy_generation_is_deprecated(tmp_path):
    with pytest.warns(FutureWarning, match="python-legacy"):
        source = run_cli(tmp_path, "--language", "python-legacy",
                         "--mechanism", mech_path, "--phase", "gas")
    assert source == PythonCodeGenerator.generate(
        "pyro", ct.Solution(mech_path, "gas")
    )


def test_cli_legacy_generators_read_cantera_only(tmp_path):
    with pytest.raises(SystemExit):
        run_cli(tmp_path, "--language", "python-legacy",
                "--library", "mutation", "--mechanism", "air_5")


@pytest.mark.skipif(not mutation_loads(), reason="Mutation++ unavailable")
def test_cli_reads_mutation(tmp_path):
    from pyrometheus.bandit.impl.mutation import MutationMechanism
    source = run_cli(tmp_path, "--language", "python", "--library",
                     "mutation", "--mechanism", "air_5", "--state-model",
                     "ChemNonEqTTv", "--thermo-database", "NASA-7")
    assert source == PythonBanditCodeGenerator.generate(
        "pyro", MutationMechanism("air_5", state_model="ChemNonEqTTv",
                                  thermo_database="NASA-7")
    )


@pytest.mark.skipif(os.environ.get("PLATO_DB") is None,
                    reason="PLATO_DB not set")
def test_cli_reads_plato(tmp_path):
    pytest.importorskip("plato")
    source = run_cli(tmp_path, "--language", "python", "--library", "plato",
                     "--mechanism", "air5", "--reaction-set", "air5",
                     "--transfer", "TTv")
    assert "class Thermochemistry" in source

# }}}
