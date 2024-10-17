Getting Started
===============

Install
-------

You can obtain Pyrometheus from the Python Package Index (PyPI) by running::

    python3 -m pip install pyrometheus

Test
----

You can install the required Python dependencies using pip::

    python3 -m pip install .[test]

Then, install CMake (a build system generator) before using it to run the tests::

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
