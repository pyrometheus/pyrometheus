"""Record Plato's answers once, and replay them wherever Plato is absent.

Plato is not public, but Bandit's Plato adapter and its parity suites
only ever talk to one object, ``plato.Thermochemistry``. With
:data:`record_variable` set and Plato available, every attribute it is
asked for and every call made on it is saved, one file per mixture,
reaction set and transfer model, under :data:`cassette_dir`. Without
Plato, :func:`install` puts a stand-in ``plato`` module in its place
that answers from those files, so the mechanism is built and checked
exactly as it would be against Plato.

A call is looked up by its exact arguments first, then by the closest
recorded call whose numeric arguments agree to :data:`argument_rtol`,
so that last-bit differences in how a platform computes a test input do
not miss a recording.
"""

import atexit
import ctypes
import gzip
import json
import os
import pathlib
import sys
import types

import numpy as np

cassette_dir = pathlib.Path(__file__).parent / "data" / "plato"
record_variable = "PYROMETHEUS_RECORD_PLATO"
argument_rtol = 1e-12


# {{{ encoding

def _encode(value):
    if isinstance(value, np.ndarray):
        return {"__array__": _encode(value.tolist()), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    return value


def _decode(value):
    if isinstance(value, dict):
        if "__array__" in value:
            return np.array(_decode(value["__array__"]), dtype=value["dtype"])
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        if "__dict__" in value:
            return {_decode(k): _decode(v) for k, v in value["__dict__"]}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _call_key(name, args, kwargs):
    return json.dumps(
        [name, _encode(list(args)), _encode(dict(sorted(kwargs.items())))]
    )


def cassette_path(mixture, reaction, transfer):
    return cassette_dir / f"{mixture}-{reaction}-{transfer}.json.gz"


def _load(path):
    if not path.exists():
        return {"attributes": {}, "calls": {}}
    with gzip.open(path, "rt") as f:
        return json.load(f)

# }}}


# {{{ recording

def _lib_source_terms(thermochem, partial_densities, temperature):
    """:returns: Plato's full source vector, which only its C library
    exposes: species mass production rates [kg/(m^3 s)], then one
    energy source per temperature [W/m^3].
    """
    from plato import _lib
    partial_densities = np.ascontiguousarray(partial_densities, dtype=float)
    temperature = np.ascontiguousarray(np.atleast_1d(temperature), dtype=float)
    omega = np.zeros(thermochem.n_species + thermochem.n_temp)
    _lib._lib.plato_get_source(
        ctypes.byref(ctypes.c_uint(0)),
        partial_densities.ctypes.data_as(_lib._P_double),
        temperature.ctypes.data_as(_lib._P_double),
        omega.ctypes.data_as(_lib._P_double),
    )
    return omega


class RecordingThermochemistry:
    """Wraps a live ``plato.Thermochemistry`` and saves what it answers."""

    def __init__(self, thermochem, path):
        self._thermochem = thermochem
        self._path = path
        self._cassette = _load(path)
        atexit.register(self._save)

    def __getattr__(self, name):
        value = getattr(self._thermochem, name)
        if not callable(value):
            self._cassette["attributes"][name] = _encode(value)
            return value

        def call(*args, **kwargs):
            result = value(*args, **kwargs)
            self._cassette["calls"][_call_key(name, args, kwargs)] = \
                _encode(result)
            return result
        return call

    def get_source(self, partial_densities, temperature):
        result = _lib_source_terms(
            self._thermochem, partial_densities, temperature)
        key = _call_key("get_source", (partial_densities, temperature), {})
        self._cassette["calls"][key] = _encode(result)
        return result

    def _save(self):
        if not self._cassette["calls"] and not self._cassette["attributes"]:
            return
        # Several instances record the same case over a session.
        merged = _load(self._path)
        for part in ("attributes", "calls"):
            merged[part].update(self._cassette[part])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self._path, "wt") as f:
            json.dump(merged, f)

    def finalize(self):
        self._save()
        self._thermochem.finalize()

# }}}


# {{{ replaying

def _close(recorded, given):
    if isinstance(recorded, np.ndarray) or isinstance(given, np.ndarray):
        recorded, given = np.asarray(recorded), np.asarray(given)
        return (recorded.shape == given.shape
                and np.allclose(recorded, given, rtol=argument_rtol, atol=0))
    if isinstance(recorded, (list, tuple)) and isinstance(given, (list, tuple)):
        return len(recorded) == len(given) and all(
            _close(r, g) for r, g in zip(recorded, given))
    if isinstance(recorded, float) or isinstance(given, float):
        return bool(np.isclose(recorded, given, rtol=argument_rtol, atol=0))
    return recorded == given


class ReplayingThermochemistry:
    """Answers as the recorded ``plato.Thermochemistry`` did."""

    def __init__(self, path):
        if not path.exists():
            raise FileNotFoundError(
                f"no Plato recording at {path}; record it with Plato "
                f"available and {record_variable}=1")
        cassette = _load(path)
        self._attributes = cassette["attributes"]
        self._calls = {}
        for key, result in cassette["calls"].items():
            name, args, kwargs = json.loads(key)
            self._calls.setdefault(name, []).append(
                (key, _decode(args), _decode(kwargs), result))
        self._exact = cassette["calls"]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._attributes:
            return _decode(self._attributes[name])
        if name in self._calls:
            return lambda *args, **kwargs: self._replay(name, args, kwargs)
        raise AttributeError(f"Plato's '{name}' was not recorded")

    def get_source(self, partial_densities, temperature):
        return self._replay("get_source", (partial_densities, temperature), {})

    def _replay(self, name, args, kwargs):
        key = _call_key(name, args, kwargs)
        if key in self._exact:
            return _decode(self._exact[key])
        args = _decode(_encode(list(args)))
        kwargs = _decode(_encode(dict(sorted(kwargs.items()))))
        for _, recorded_args, recorded_kwargs, result in self._calls[name]:
            if _close(recorded_args, args) and _close(
                    sorted(recorded_kwargs.items()), sorted(kwargs.items())):
                return _decode(result)
        raise KeyError(
            f"Plato's {name}{tuple(args)} was not recorded; record again "
            f"with Plato available and {record_variable}=1")

    def finalize(self):
        pass

# }}}


# {{{ installation

def _real_plato_available():
    if os.environ.get("PLATO_DB") is None:
        return False
    try:
        import plato  # noqa: F401
    except ImportError:
        return False
    return not getattr(plato, "replaying", False)


def install():
    """Wrap Plato for recording when asked to, or stand in for it when it
    is absent and recordings exist. Call before anything imports
    ``plato``.
    """
    if _real_plato_available():
        if os.environ.get(record_variable):
            import plato
            live = plato.Thermochemistry

            def recording(solver, mixture, reaction, transfer, **kwargs):
                return RecordingThermochemistry(
                    live(solver, mixture, reaction, transfer, **kwargs),
                    cassette_path(mixture, reaction, transfer))
            plato.Thermochemistry = recording
        return
    if not any(cassette_dir.glob("*.json.gz")):
        return
    stand_in = types.ModuleType("plato")
    stand_in.replaying = True

    def replaying(solver, mixture, reaction, transfer, **kwargs):
        return ReplayingThermochemistry(
            cassette_path(mixture, reaction, transfer))
    stand_in.Thermochemistry = replaying
    sys.modules["plato"] = stand_in


def plato_available():
    """:returns: Whether Plato, live or recorded, can answer."""
    try:
        import plato
    except ImportError:
        return False
    return getattr(plato, "replaying", False) or _real_plato_available()


def plato_db_path():
    """:returns: Plato's database, which a recording does not need."""
    return os.environ.get("PLATO_DB")


def plato_source_terms(oracle, partial_densities, temperature):
    """:returns: Plato's full source vector: species mass production
    rates [kg/(m^3 s)] followed by one energy source per temperature
    [W/m^3]. ``Thermochemistry.get_production_rates`` discards the
    energy part.
    """
    if isinstance(oracle, (RecordingThermochemistry,
                           ReplayingThermochemistry)):
        return oracle.get_source(partial_densities, temperature)
    return _lib_source_terms(oracle, partial_densities, temperature)

# }}}
