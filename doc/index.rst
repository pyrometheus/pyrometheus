Welcome to pyrometheus's documentation!
=======================================

.. When you update this description, consider also updating the one in the README.

Pyrometheus generates Python, C++ and Fortran code for the
thermochemistry of reacting flows, from mechanisms read by `Cantera
<https://cantera.org>`__, `Mutation++
<https://github.com/mutationpp/Mutationpp>`__ or Plato.

Here’s an example to give you an impression:

.. code-block:: python

   import numpy as np
   import pyrometheus as pyro
   from pyrometheus.bandit.impl.cantera import CanteraMechanism

   mech = CanteraMechanism("mech.yaml")
   gas = pyro.get_thermochem_class(mech)(np)

Table of Contents
-----------------

.. toctree::
    :maxdepth: 2

    start
    model
    bandit
    codegen
    misc
    🚀 Github <https://github.com/pyrometheus/pyrometheus>
    💾 Download Releases <https://pypi.org/project/pyrometheus>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
