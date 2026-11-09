"""The recorded Plato answers stand in for Plato faithfully: they encode
what Plato returns without loss, find a call across last-bit argument
differences but not across real ones, and, where live Plato is
available, generate the very code that Plato itself does.
"""

import gzip
import json
import os

import numpy as np
import pytest

import plato_cassette
from plato_cassette import (
    ReplayingThermochemistry, _call_key, _decode, _encode, cassette_dir,
)


def test_encoding_round_trips():
    value = {
        "array": np.arange(6.0).reshape(2, 3),
        "integers": np.array([1, 2], dtype=np.int32),
        "pair": (1.5, 2.5),
        "names": ["N2", "O2"],
        "scalar": np.float64(0.1),
    }
    decoded = _decode(json.loads(json.dumps(_encode(value))))
    assert decoded.keys() == value.keys()
    np.testing.assert_array_equal(decoded["array"], value["array"])
    assert decoded["integers"].dtype == np.int32
    assert decoded["pair"] == (1.5, 2.5)
    assert decoded["names"] == ["N2", "O2"]
    assert decoded["scalar"] == 0.1


@pytest.fixture
def recording(tmp_path):
    arguments = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "case.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({
            "attributes": {"n_species": 3},
            "calls": {
                _call_key("viscosity", (arguments,), {}): _encode(4.0),
            },
        }, f)
    return ReplayingThermochemistry(path), arguments


def test_replay_answers_exact_and_last_bit_arguments(recording):
    oracle, arguments = recording
    assert oracle.n_species == 3
    assert oracle.viscosity(arguments) == 4.0
    assert oracle.viscosity(arguments * (1 + 4e-16)) == 4.0


def test_replay_refuses_different_arguments(recording):
    oracle, arguments = recording
    with pytest.raises(KeyError, match="not recorded"):
        oracle.viscosity(arguments * (1 + 1e-9))
    with pytest.raises(AttributeError, match="not recorded"):
        oracle.thermal_conductivities


def recorded_cases():
    return sorted(
        tuple(path.name[:-len(".json.gz")].rsplit("-", 2))
        for path in cassette_dir.glob("*.json.gz")
    )


@pytest.mark.skipif(not plato_cassette._real_plato_available(),
                    reason="needs live Plato to compare against")
@pytest.mark.parametrize("case", recorded_cases(), ids="-".join)
def test_recordings_generate_what_plato_generates(case):
    from pyrometheus.bandit.impl.plato import Plato, PlatoMechanism
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
    mixture, reaction_set, transfer = case

    live = PlatoMechanism(mixture, reaction_set, transfer,
                          os.environ["PLATO_DB"])
    try:
        expected = PythonBanditCodeGenerator.generate("pyro", live)
    finally:
        live.finalize()

    namespace = Plato.__new__(Plato)
    namespace.thermochem = ReplayingThermochemistry(
        plato_cassette.cassette_path(mixture, reaction_set, transfer))
    replayed = _build_from(PlatoMechanism, namespace)
    assert PythonBanditCodeGenerator.generate("pyro", replayed) == expected


def _build_from(mechanism_class, namespace):
    """Build a mechanism as its constructor does, around *namespace*."""
    import unittest.mock
    with unittest.mock.patch(
            "pyrometheus.bandit.impl.plato.Plato", lambda *args: namespace):
        return mechanism_class("unused", "unused", "unused", None)
