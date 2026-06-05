import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
from pyrometheus.flamelets.state import StateContainer


@dataclass
class Stencil:
    """Finite-difference stencil along a single axis of a state container.

    A stencil is the pair of arrays ``(coefficients, indices)``
    representing the linear combination

    .. math::

        (S f)_i = \\sum_k c_k\\, f_{i + j_k},

    together with the range of nodes ``i`` at which the stencil is
    actually applied.  Two stencil flavours are supported through the
    ``tag`` field:

    - ``"interior"`` stencils are symmetric stencils applied to a
      contiguous interior strip of nodes; ``offset_start`` and
      ``offset_end`` may further shrink that strip away from the
      boundary.
    - ``"near-boundary"`` stencils are one-point stencils anchored at
      ``location`` (``0`` for the left boundary, ``-1`` for the right
      boundary) used to encode Dirichlet rows of the operator matrix.

    The post-initialization step computes two derived attributes used
    when the stencil is applied:

    - ``ranges`` : list of ``(start, end)`` slices, one per coefficient,
      identifying the input shift associated with each coefficient.
    - ``output_slice`` : the slice along the working axis at which the
      stencil writes its contribution.

    Attributes
    ----------
    coefficients : list of float
        Stencil weights :math:`c_k`.
    indices : list of int
        Index offsets :math:`j_k` (relative to ``i``) at which each
        coefficient samples the input.
    tag : {"interior", "near-boundary"}
        Stencil flavour controlling how ``ranges`` and
        ``output_slice`` are computed.
    location : int, optional
        For near-boundary stencils, the absolute index of the boundary
        node (``0`` or ``-1``).  Required when ``tag == "near-boundary"``.
    offset_start, offset_end : int, optional
        For interior stencils, additional shrinkage of the output
        slice on the left/right.
    dtype : numpy.dtype, optional
        Floating-point dtype of the stencil coefficients.
    """

    coefficients: List[np.float64]
    indices: List[np.int32]
    tag: Literal["interior", "near-boundary"]
    location: Optional[int] = None
    offset_start: Optional[int] = 0
    offset_end: Optional[int] = 0
    dtype: np.dtype = np.float64

    def __post_init__(self):

        if self.tag == "interior":
            r = (len(self.indices) - 1) // 2
            js = self.offset_start
            je = self.offset_end
            self.ranges = [
                (
                    r + i + js, -r + i + je
                ) if r != (i + je) else (r + i + js, None)
                for i in self.indices
            ]
            self.output_slice = slice(r + js, -r + je)
        elif self.tag == "near-boundary":
            if self.location is None:
                raise ValueError(
                    "location must be provided for near-boundary stencils."
                )
            self.ranges = [
                (
                    (i + self.location, i + self.location + 1)
                    if i + self.location + 1
                    else (i + self.location, None)
                )
                for i in self.indices
            ]
            my_end = self.location + 1 if self.location != -1 else None
            self.output_slice = slice(self.location, my_end)
        else:
            raise ValueError(f"Unknown stencil tag: {self.tag}")
        return


@dataclass
class BoundaryStencil(Stencil):
    """:class:`Stencil` augmented with a ghost-point index.

    Attributes
    ----------
    ghost_point_index : int
        Index of the ghost cell associated with this boundary stencil.
    """

    ghost_point_index: int = 0


@dataclass
class PeriodicBoundaryStencil(BoundaryStencil):
    """Periodic variant of :class:`BoundaryStencil`.

    Attributes
    ----------
    partner_index : int
        Index of the periodic partner that maps the ghost point back
        into the interior.
    """

    partner_index: int = 0


def output_view(
    loc_or_range: Union[int, slice], axis: int, spatial_dimension: int
) -> Tuple[Union[int, slice], ...]:
    """Build a slicing tuple that selects a range along ``axis``.

    All axes other than ``axis`` get a ``slice(None)``; ``axis`` is
    populated with ``loc_or_range``, which may be either an integer
    location or a :class:`slice`.

    Parameters
    ----------
    loc_or_range : int or slice
        Index (or slice) of the entries to select along ``axis``.
    axis : int
        Spatial axis being indexed.
    spatial_dimension : int
        Total number of spatial axes of the target container.

    Returns
    -------
    tuple
        Tuple suitable for use as a NumPy/JAX advanced index.
    """
    view = [slice(None)] * spatial_dimension
    view[axis] = loc_or_range
    return tuple(view)


def shifted_view(
    s: StateContainer, my_start: int, my_end: int, axis: int
) -> StateContainer:
    """Return ``s[:, ..., my_start:my_end, ...]`` along ``axis``.

    Used internally by :func:`apply_stencil` to gather the shifted
    input contributions associated with each stencil coefficient.
    """
    my_slice = [slice(None)] * s.spatial_dimension
    my_slice[axis] = slice(my_start, my_end)
    return s[tuple(my_slice)]


def apply_stencil(
    s: StateContainer, stencil: Stencil, axis: int, output_shape: Tuple[int]
) -> StateContainer:
    """Apply a :class:`Stencil` to a :class:`StateContainer` along ``axis``.

    Sums the contributions ``c_k * s[..., j_k_start:j_k_end, ...]`` for
    each coefficient and index range stored in ``stencil``.  The
    resulting container has spatial shape ``output_shape``.

    Parameters
    ----------
    s : StateContainer
        Input state to which the stencil is applied.
    stencil : Stencil
        Stencil whose coefficients and shifted ranges describe the
        linear combination.
    axis : int
        Spatial axis along which the stencil acts.
    output_shape : tuple of int
        Spatial shape of the output container.

    Returns
    -------
    StateContainer
        New container of the same type as ``s`` holding the stencil
        output on ``output_shape``.
    """
    return sum(
        (
            coeff * shifted_view(s, idx_start, idx_end, axis=axis)
            for coeff, (idx_start, idx_end) in zip(
                    stencil.coefficients, stencil.ranges
            )
        ),
        start=s.zeros(output_shape),
    )
