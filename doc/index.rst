Welcome to pyrometheus's documentation!
=======================================

.. When you update this description, consider also updating the one in the README.

Pyrometheus is a code generator for chemical mechanisms, based on :ref:`cantera <cantera:sec-cython-documentation>`.

Here’s an example to give you an impression:

.. code-block:: python

   import numpy as np
   import pyrometheus as pyro

   sol = ct.Solution("mech.yaml")
   ptk = pyro.get_thermochem_class(sol)()

Table of Contents
-----------------

.. toctree::
    :maxdepth: 2

    start
    model
    codegen
    misc
    🚀 Github <https://github.com/pyrometheus/pyrometheus>
    💾 Download Releases <https://pypi.org/project/pyrometheus>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
