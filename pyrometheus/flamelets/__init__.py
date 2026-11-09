"""Steady, one-dimensional flamelet solver built on top of pyrometheus.

This subpackage assembles and solves the non-premixed flamelet equations in
mixture-fraction space,

.. math::

    \\tfrac{1}{2}\\, \\chi(Z)\\, \\partial_{ZZ} \\phi(Z) + S(\\phi) = 0,

where the state :math:`\\phi` collects the mixture enthalpy and the species
mass fractions, :math:`\\chi(Z)` is the scalar dissipation rate, and
:math:`S(\\phi)` is the chemical source term obtained from a
pyrometheus-generated thermochemistry object.  Dirichlet boundary
conditions impose the oxidizer (:math:`Z = 0`) and fuel (:math:`Z = 1`)
stream compositions and enthalpies.

The package exposes:

- :class:`DomainConfig`, :class:`Domain` -- discretization of the mixture
  fraction interval.
- :class:`FlameletEquations` -- residual, Jacobian, source-term and
  adjoint-operator assembly.
- :class:`FlameletSolver` -- Newton / Crank--Nicolson time-marching driver
  using a block-Thomas linear solve.
- :class:`FlameletState`, :func:`_state_to_array`, :func:`_array_to_state`
  -- the dataclass-style container used to pass enthalpy and mass-fraction
  fields through the solver.
- :func:`make_pyro_object` -- factory that wraps a pyrometheus-generated
  class with the array-library-specific helpers required by the solver.
"""

from .domain import DomainConfig, Domain
from .equations import FlameletEquations
from .make_pyro import make_pyro_object
from .solver import FlameletSolver
from .state import FlameletState, _array_to_state, _state_to_array


__all__ = [
    "DomainConfig",
    "Domain",
    "FlameletEquations",
    "FlameletSolver",
    "FlameletState",
    "make_pyro_object",
    "_state_to_array",
    "_array_to_state",
]
