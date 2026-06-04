import jax.numpy as jnp
from typing import List, Tuple
from pyrometheus.flamelets.stencils import Stencil, output_view, apply_stencil
from pyrometheus.flamelets.domain import Domain
from pyrometheus.flamelets.state import StateContainer, FlameletState


# {{{ Base operator class

class Operator:

    def __init__(self, domain: Domain = None, stencils: List[Stencil] = None):
        self.domain = domain
        if stencils:
            self.stencils = stencils
        else:
            raise ValueError("Stencils must be provided for the operator.")

    def apply_stencils_along_axis(self,
                                  s: StateContainer,
                                  axis: int) -> StateContainer:
        out = s.zeros_like()
        for _, stencil in enumerate(self.stencils):
            view = output_view(stencil.output_slice, axis, s.spatial_dimension)
            out[view] = apply_stencil(s, stencil, axis, s[view].spatial_shape)
        return out

    def apply_operator(self,
                       s: Tuple[StateContainer]) -> Tuple[StateContainer]:
        """
        Apply the operator to the array `f`.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, s: StateContainer) -> StateContainer:
        """
        Apply the operator to the state container `s`.
        """
        return self.apply_operator(s)

# }}}


class Laplacian(Operator):

    def __init__(self, domain: Domain):
        self.stencils = [
            Stencil(
                coefficients=[1],
                indices=[0],
                tag="near-boundary",
                location=0,
                dtype=jnp.float64
            ),
            Stencil(
                coefficients=[1, -2, 1],
                indices=[-1, 0, 1],
                tag="interior",
                offset_start=0,
                offset_end=0,
                dtype=jnp.float64,
            ),
            Stencil(
                coefficients=[1],
                indices=[0],
                tag="near-boundary",
                location=-1,
                dtype=jnp.float64
            )
        ]
        # Assemble the operator's matrix form.
        # Note: this assumes Dirichlet boundary conditions
        super().__init__(domain, stencils=self.stencils)

    def assemble_block_form(self, num_vars):

        num_x = self.domain.num_x
        dx_sqr = self.domain.jac[0] ** 2
        eye_nv = jnp.eye(num_vars)
        zeros_nv = jnp.zeros((num_vars, num_vars))

        bc_block = eye_nv / dx_sqr
        self.central_blocks = jnp.stack(
            (bc_block,)
            + tuple(-2 * eye_nv / dx_sqr for _ in range(num_x - 2))
            + (bc_block,)
        )
        self.lower_blocks = jnp.stack(
            tuple(eye_nv / dx_sqr for _ in range(num_x - 2))
            + (zeros_nv,)
        )
        self.upper_blocks = jnp.stack(
            (zeros_nv,)
            + tuple(eye_nv / dx_sqr for _ in range(num_x - 2))
        )

    def apply_operator(self, s: FlameletState):
        res_list = []
        for stencil in self.stencils:
            view = output_view(stencil.output_slice, 0, s.spatial_dimension)
            res_list += [
                apply_stencil(s, stencil, 0, s[view].spatial_shape)
                / self.domain.jac[0]**2
            ]
        return FlameletState(
            **{f: jnp.hstack(
                [getattr(res, f) for res in res_list],
            ) for f in s.__dict__.keys()}
        )
