import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Any
from pyrometheus.flamelets.make_pyro import detect_array_library


# {{{ Base class

@dataclass
class StateContainer:

    @property
    def spatial_dimension(
        self,
    ):
        raise NotImplementedError("Subclasses must implement this method.")

    def clone(
        self,
    ):
        return self.__class__(
            **{k: v.copy() for k, v in self.__dict__.items()}
        )

    def zeros(self, shape):
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
        if not isinstance(idx, tuple):
            idx = (idx,)
        vector_field_idx = (slice(None),) + idx
        return self.__class__(
            **{k: getattr(
                self, k
            )[vector_field_idx] for k, v in self.__dict__.items()}
        )

    def __setitem__(self, idx, other):
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
    enthalpy: jnp.ndarray
    mass_fractions: jnp.ndarray

    def tree_flatten(self):
        children = (self.enthalpy, self.mass_fractions)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...],
                       children: Tuple[Any, ...]) -> "FlameletState":
        enthalpy, mass_fractions = children
        return cls(enthalpy=enthalpy, mass_fractions=mass_fractions)

    @property
    def num_vars(self):
        return 1 + len(self.mass_fractions)

    @property
    def spatial_dimension(self,):
        return len(self.enthalpy.shape)

    @property
    def spatial_shape(self,):
        return self.enthalpy.shape

    def zeros_like(self):
        return FlameletState(
            enthalpy=jnp.zeros_like(self.enthalpy),
            mass_fractions=jnp.zeros_like(self.mass_fractions)
        )

    def __getitem__(self, idx):
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
        raise NotImplementedError

    def set_to_other(self, idx, other):
        if not isinstance(idx, tuple):
            scalar_field_idx = (idx,)
        else:
            scalar_field_idx = idx

        vector_field_idx = (slice(None),) + scalar_field_idx
        self.enthalpy.at[scalar_field_idx].set(other.enthalpy)
        self.mass_fractions.at[vector_field_idx].set(other.mass_fractions)


def _state_to_array(state: FlameletState) -> jnp.ndarray:
    s, _ = state.tree_flatten()
    return jnp.vstack(s)


def _array_to_state(array: jnp.ndarray) -> FlameletState:
    return FlameletState.tree_unflatten(
        (), (array[0], array[1:])
    )

# }}}
