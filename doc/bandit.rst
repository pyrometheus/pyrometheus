Mechanism Libraries
===================

Pyrometheus 2 reads mechanisms through *Bandit*, a layer that turns the
data of a thermochemistry library into symbolic expressions and renders
them as Python, C++ or Fortran. Three libraries are supported:

.. list-table::
   :header-rows: 1

   * - Library
     - Mechanism
     - Temperatures
     - Ions and electrons
     - Energy exchange
     - Transport
   * - `Cantera <https://cantera.org>`__
     - YAML file
     - one
     - no
     - --
     - no
   * - `Mutation++ <https://github.com/mutationpp/Mutationpp>`__
     - mixture name
     - one or two (``ChemNonEq1T``, ``ChemNonEqTTv``)
     - yes
     - Landau--Teller VT, NASA energy split
     - no
   * - Plato (experimental)
     - mixture and reaction set
     - one, two or three (``empty``, ``TTv``, ``TTvTe``)
     - partial
     - Landau--Teller VT, NASA energy split
     - neutral mixtures

The code generated for each library is checked against that library
itself, in every language, by the test suite.

Plato is not publicly available. ``PlatoMechanism`` is experimental
and needs access to Plato and its database. Its test suites replay
recorded Plato answers, so they run without it.

With Plato's ionized mixtures (``air11``), the generated kinetics leave
the electron out of electron-impact reactions and evaluate their reverse
rates at the heavy temperature; electron energy terms and ionized
transport are not generated. For ions and electrons, use Mutation++.

Building a mechanism
--------------------

Each library has its own mechanism class, and
:func:`pyrometheus.bandit.load_mechanism` builds any of them by name:

.. code-block:: python

    from pyrometheus.bandit import load_mechanism
    from pyrometheus.bandit.impl.cantera import CanteraMechanism
    from pyrometheus.bandit.impl.mutation import MutationMechanism

    sandiego = CanteraMechanism("sandiego.yaml", phase="gas")
    air = MutationMechanism("air_11", state_model="ChemNonEqTTv",
                            thermo_database="NASA-9")
    same_air = load_mechanism("mutation", "air_11",
                              state_model="ChemNonEqTTv")

``CanteraMechanism`` also accepts a :class:`cantera.Solution`.
Mutation++ must include the Python bindings added in
`mutationpp/Mutationpp#341 <https://github.com/mutationpp/Mutationpp/pull/341>`__.

.. autofunction:: pyrometheus.bandit.load_mechanism

Generating code
---------------

From Python, generate source for a mechanism, or a ready-to-use Python
class:

.. code-block:: python

    import numpy as np
    import pyrometheus as pyro
    from pyrometheus.codegen.cpp_bandit import CppBanditCodeGenerator

    gas = pyro.get_thermochem_class(air)(np)
    header = CppBanditCodeGenerator.generate("air", air)

From the command line, ``--library`` selects how the mechanism is read::

    python -m pyrometheus --language python --mechanism sandiego.yaml \
        --phase gas --name sandiego --output sandiego.py
    python -m pyrometheus --language cpp --library mutation \
        --mechanism air_11 --state-model ChemNonEqTTv \
        --name air --output air.hpp

Generated C++ is a header-only, C++17 ``pyro::<name><ScalarT>`` struct
of static functions on ``std::array``. Generated Fortran is a module
whose functions run inside loops, offloadable with directives.

Multi-temperature mechanisms
----------------------------

Wherever a function takes ``temperature``, a mechanism with several
temperatures takes all of them as an array: the heavy-particle
temperature first, then the vibrational one and, for Plato's ``TTvTe``,
the electron one. This includes ``get_pressure`` and
``get_density``, where the electrons' partial pressure follows the
electron temperature.

Two-temperature mechanisms also generate the energy-exchange terms an
adiabatic reactor needs: ``get_vt_energy_transfer_source``,
``get_pressure_relaxation_times``, and the NASA energy split into
``get_translational_rotational_energy`` at the heavy temperature and
``get_nasa_polynomial_vibrational_energy`` (with its specific heat) at
the vibrational one. ``examples/postshock_relax`` integrates isothermal
and adiabatic post-shock relaxation with them.

Coming from Pyrometheus 1
-------------------------

The ``python``, ``cpp`` and ``fortran`` generators are now Bandit's. The
1.x generators remain, deprecated, as ``python-legacy``, ``cpp-legacy``
and ``fortran-legacy``, and read Cantera mechanisms only.
``pyrometheus.get_thermochem_class`` and
``pyrometheus.generate`` still accept a :class:`cantera.Solution`,
through the 1.x generator and with a :class:`DeprecationWarning`; pass
``CanteraMechanism(solution)`` to use Bandit instead.
