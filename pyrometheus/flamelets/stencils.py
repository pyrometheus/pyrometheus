import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
from pyrometheus.flamelets.state import StateContainer


@dataclass
class Stencil:
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
    ghost_point_index: int = 0


@dataclass
class PeriodicBoundaryStencil(BoundaryStencil):
    partner_index: int = 0


def output_view(
    loc_or_range: Union[int, slice], axis: int, spatial_dimension: int
) -> Tuple[Union[int, slice], ...]:
    """
    Create a view for the output slice based on the dimension and axis.
    """
    view = [slice(None)] * spatial_dimension
    view[axis] = loc_or_range
    return tuple(view)


def shifted_view(
    s: StateContainer, my_start: int, my_end: int, axis: int
) -> StateContainer:
    my_slice = [slice(None)] * s.spatial_dimension
    my_slice[axis] = slice(my_start, my_end)
    return s[tuple(my_slice)]


def apply_stencil(
    s: StateContainer, stencil: Stencil, axis: int, output_shape: Tuple[int]
) -> StateContainer:
    return sum(
        (
            coeff * shifted_view(s, idx_start, idx_end, axis=axis)
            for coeff, (idx_start, idx_end) in zip(
                    stencil.coefficients, stencil.ranges
            )
        ),
        start=s.zeros(output_shape),
    )
