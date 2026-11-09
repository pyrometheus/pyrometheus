Getting Started
===============

Install
-------

You can obtain Pyrometheus from the Python Package Index (PyPI) by running::

    python3 -m pip install pyrometheus

Alternatively, Pyrometheus is available through `Spack
<https://spack.io>`__::

    spack install py-pyrometheus

This installs Pyrometheus together with its runtime dependencies
(Cantera, Mako, pymbolic). The package was added to spack-packages in
`#4696 <https://github.com/spack/spack-packages/pull/4696>`__ and
tracks ``pyrometheus`` releases on PyPI.

To read Mutation++ mechanisms, also install Mutation++ with the Python
bindings of `mutationpp/Mutationpp#341
<https://github.com/mutationpp/Mutationpp/pull/341>`__::

    python3 -m pip install "mutationpp @ git+https://github.com/ecisneros8/Mutationpp@nanobind-bindings"

Test
----

You can install the required Python dependencies using pip::

    python3 -m pip install .[test] jax

The suites compare generated Python, C++ and Fortran with Cantera,
Mutation++ and Plato. C++ needs a compiler and pybind11, Fortran needs
gfortran, meson and ninja on the ``PATH``, and the Mutation++ suites
need Mutation++; whatever is missing is skipped. Run them with::

    python3 -m pytest test --backend python

Plato is not public, so its suites replay answers recorded under
``test/data/plato``. With Plato and ``PLATO_DB`` available, they run
against Plato itself; set ``PYROMETHUS_RECORD_PLATO=1`` to record the
answers again.

The deprecated 1.x generators keep their own suite, built with CMake::

    cmake -S test -B build
    cmake --build build
    ctest --test-dir build

Building the Documentation
--------------------------

You need to install Sphinx (a Python documentation tool) and two third-party
packages to build the docs::

    python3 -m pip install .[docs]

Then simply run::

    cd doc
    make html

and browse the docs starting from ``doc/_build/html/index.html``.
