import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Any
from pyrometheus.flamelets.make_pyro import detect_array_library


# {{{ Base class

@dataclass
class StateContainer:
    """Abstract base for dataclass-style multi-field state containers.

    A ``StateContainer`` bundles one or more named array fields (the
    dataclass attributes of the concrete subclass) and exposes them with
    a uniform algebraic interface: element-wise addition, subtraction,
    multiplication, division and negation are implemented in terms of
    field-wise operations through :meth:`_apply_binary_op`.  Slice and
    indexing operations are forwarded to each field while preserving a
    leading vector dimension, so that subclasses such as
    :class:`FlameletState` may hold both scalar (``enthalpy``) and
    vector-valued (``mass_fractions``) fields with a single API.

    Concrete subclasses must define their fields as dataclass
    attributes, override :attr:`spatial_dimension`, and may override
    :meth:`zeros_like` for performance.
    """

    @property
    def spatial_dimension(
        self,
    ):
        """Number of spatial axes of the contained fields."""
        raise NotImplementedError("Subclasses must implement this method.")

    def clone(
        self,
    ):
        """Return a deep copy of the container with each field copied."""
        return self.__class__(
            **{k: v.copy() for k, v in self.__dict__.items()}
        )

    def zeros(self, shape):
        """Return a container of the same type whose fields have a new spatial shape.

        Each field is replaced by a zero array whose leading
        (non-spatial) dimensions are preserved and whose trailing
        spatial dimensions are replaced by ``shape``.  The array
        library (``numpy`` or ``jax.numpy``) is inferred from the
        existing fields.

        Parameters
        ----------
        shape : tuple of int
            New spatial shape to allocate for every field.
        """
        # Auto-detect array library from first array in container
        first_array = next(iter(self.__dict__.values()))
        pyro_np = detect_array_library(first_array)

        # Determine number of spatial dimensions to preserve leading dimensions
        num_spatial_dims = len(self.spatial_shape)

        return self.__class__(
            **{
                k: pyro_np.zeros(
                    v.shape[:-num_spatial_dims] + shape, dtype=v.dtype,
                )
                for k, v in self.__dict__.items()
            }
        )

    def zeros_like(self):
        """Return a zero-valued container with the same fields, shapes and dtypes."""
        # Auto-detect array library from first array in container
        first_array = next(iter(self.__dict__.values()))
        pyro_np = detect_array_library(first_array)
        return self.__class__(
            **{
                k: pyro_np.zeros_like(v, dtype=v.dtype,)
                for k, v in self.__dict__.items()
            }
        )

    def _apply_binary_op(self, other, op):
        """Apply a binary operator field-wise to ``self`` and ``other``.

        If ``other`` is another :class:`StateContainer`, ``op`` is
        applied between matching fields; otherwise it is broadcast
        against every field of ``self``.
        """
        if isinstance(other, StateContainer):
            return self.__class__(
                **{k: op(
                    v, getattr(other, k)
                ) for k, v in self.__dict__.items()}
            )
        else:
            return self.__class__(
                **{k: op(v, other) for k, v in self.__dict__.items()}
            )

    def __add__(self, other):
        return self._apply_binary_op(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._apply_binary_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._apply_binary_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self._apply_binary_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_binary_op(other, lambda x, y: x / y)

    def __neg__(
        self,
    ):
        return self.__class__(**{k: -v for k, v in self.__dict__.items()})

    def __getitem__(self, idx):
        """Return a new container with each field sliced over the spatial axes.

        A leading ``slice(None)`` is prepended to ``idx`` so that any
        vector dimension (such as the species axis of
        ``mass_fractions``) is preserved while only the spatial axes
        are indexed.
        """
        if not isinstance(idx, tuple):
            idx = (idx,)
        vector_field_idx = (slice(None),) + idx
        return self.__class__(
            **{k: getattr(
                self, k
            )[vector_field_idx] for k, v in self.__dict__.items()}
        )

    def __setitem__(self, idx, other):
        """Assign the matching slice of every field from ``other``.

        As with :meth:`__getitem__`, a leading ``slice(None)`` is
        prepended so that the vector dimension of each field is
        preserved and only the spatial axes are written.
        """
        if not isinstance(idx, tuple):
            idx = (idx,)
        vector_field_idx = (slice(None),) + idx
        return self.__class__(
            **{
                k: getattr(
                    self, k
                ).__setitem__(vector_field_idx, getattr(other, k))
                for k, v in self.__dict__.items()
            }
        )

    def interior_view(self, num_h: int):
        """Return a container restricted to the interior of the spatial domain.

        Drops the first and last ``num_h`` entries along the (single)
        spatial axis of every field, exposing the strictly interior
        nodes (away from the Dirichlet boundaries).

        Parameters
        ----------
        num_h : int
            Number of halo / boundary nodes to strip from each end.
        """
        view = (slice(num_h, -num_h),)
        vector_field_view = (slice(None),) + view
        return self.__class__(
            **{k: getattr(
                self, k
            )[vector_field_view] for k, v in self.__dict__.items()}
        )

# }}}


# {{{ Flamelet state

@jax.tree_util.register_pytree_node_class
@dataclass
class FlameletState(StateContainer):
    """State of a flamelet: mixture enthalpy and species mass fractions.

    The state bundles the two unknowns of the steady flamelet equations
    on a one-dimensional mixture-fraction grid:

    - ``enthalpy``: array of shape ``(num_x,)`` holding the mixture
      enthalpy at every grid point.
    - ``mass_fractions``: array of shape ``(num_species, num_x)``
      holding the species mass fractions at every grid point.

    The class is registered as a JAX pytree node, so instances may be
    threaded through :func:`jax.jit`, :func:`jax.vmap` and
    :func:`jax.jacfwd` transparently.  Arithmetic is inherited from
    :class:`StateContainer` and operates field-wise.
    """

    enthalpy: jnp.ndarray
    mass_fractions: jnp.ndarray

    def tree_flatten(self):
        """Flatten the state to a pytree leaf tuple for JAX."""
        children = (self.enthalpy, self.mass_fractions)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...],
                       children: Tuple[Any, ...]) -> "FlameletState":
        """Rebuild a :class:`FlameletState` from JAX pytree children."""
        enthalpy, mass_fractions = children
        return cls(enthalpy=enthalpy, mass_fractions=mass_fractions)

    @property
    def num_vars(self):
        """Total number of unknowns per grid point (``1 + num_species``)."""
        return 1 + len(self.mass_fractions)

    @property
    def spatial_dimension(self,):
        """Number of spatial axes (always ``1`` for a flamelet)."""
        return len(self.enthalpy.shape)

    @property
    def spatial_shape(self,):
        """Spatial shape of the state, inherited from ``enthalpy``."""
        return self.enthalpy.shape

    def zeros_like(self):
        """Zero-valued :class:`FlameletState` with the same shapes and dtype."""
        return FlameletState(
            enthalpy=jnp.zeros_like(self.enthalpy),
            mass_fractions=jnp.zeros_like(self.mass_fractions)
        )

    def __getitem__(self, idx):
        """Slice the state spatially while preserving the species axis.

        ``idx`` selects along the spatial (grid) axis of ``enthalpy``
        and along the trailing (grid) axis of ``mass_fractions``; the
        leading species axis of ``mass_fractions`` is kept intact.
        """
        if not isinstance(idx, tuple):
            scalar_field_idx = (idx,)
        else:
            scalar_field_idx = idx

        vector_field_idx = (slice(None),) + scalar_field_idx
        return FlameletState(
            self.enthalpy[scalar_field_idx],
            self.mass_fractions[vector_field_idx]
        )

    def __setitem__(self, idx):
        """Not implemented: :class:`FlameletState` is functionally updated."""
        raise NotImplementedError

    def set_to_other(self, idx, other):
        """Functionally assign a slice of ``self`` from ``other``.

        Parameters
        ----------
        idx : int, slice or tuple
            Index applied to the spatial axis of ``enthalpy`` and to
            the trailing axis of ``mass_fractions``.
        other : FlameletState
            State whose values are copied into the selected slice.

        Notes
        -----
        Because the underlying arrays are JAX arrays, ``.at[...].set``
        produces a new array; this method has no observable side
        effect on ``self``.
        """
        if not isinstance(idx, tuple):
            scalar_field_idx = (idx,)
        else:
            scalar_field_idx = idx

        vector_field_idx = (slice(None),) + scalar_field_idx
        self.enthalpy.at[scalar_field_idx].set(other.enthalpy)
        self.mass_fractions.at[vector_field_idx].set(other.mass_fractions)


def _state_to_array(state: FlameletState) -> jnp.ndarray:
    """Stack a :class:`FlameletState` into an array of shape ``(num_vars, num_x)``.

    Row ``0`` is the enthalpy; rows ``1:`` are the species mass
    fractions.  This is the array layout consumed by the per-grid-point
    vectorized routines in :mod:`equations`.
    """
    s, _ = state.tree_flatten()
    return jnp.vstack(s)


def _array_to_state(array: jnp.ndarray) -> FlameletState:
    """Inverse of :func:`_state_to_array`.

    Interprets ``array[0]`` as the enthalpy field and ``array[1:]`` as
    the species mass-fraction matrix.
    """
    return FlameletState.tree_unflatten(
        (), (array[0], array[1:])
    )

# }}}
