import jax.numpy as jnp
from typing import List, Tuple
from pyrometheus.flamelets.stencils import Stencil, output_view, apply_stencil
from pyrometheus.flamelets.domain import Domain
from pyrometheus.flamelets.state import StateContainer, FlameletState


# {{{ Base operator class

class Operator:
    """Base class for finite-difference operators on a :class:`Domain`.

    An ``Operator`` is a collection of :class:`Stencil` objects whose
    output slices tile the spatial domain.  Subclasses encode a specific
    differential operator (e.g. the Laplacian) by providing the right
    stencils and overriding :meth:`apply_operator` to scale the stencil
    outputs by the appropriate powers of the grid Jacobian.

    Parameters
    ----------
    domain : Domain
        Grid over which the operator is defined.
    stencils : list of Stencil
        Stencils that, taken together, define the action of the
        operator on every node of the domain.

    Raises
    ------
    ValueError
        If ``stencils`` is empty or ``None``.
    """

    def __init__(self, domain: Domain = None, stencils: List[Stencil] = None):
        self.domain = domain
        if stencils:
            self.stencils = stencils
        else:
            raise ValueError("Stencils must be provided for the operator.")

    def apply_stencils_along_axis(self,
                                  s: StateContainer,
                                  axis: int) -> StateContainer:
        """Apply every stencil to ``s`` along ``axis`` and tile the results.

        Each stencil writes into the slice given by its ``output_slice``
        attribute; together the stencils are expected to cover the
        entire domain along ``axis``.

        Parameters
        ----------
        s : StateContainer
            Input state.
        axis : int
            Spatial axis along which the stencils act.

        Returns
        -------
        StateContainer
            Output of the operator with the same shape as ``s``.
        """
        out = s.zeros_like()
        for _, stencil in enumerate(self.stencils):
            view = output_view(stencil.output_slice, axis, s.spatial_dimension)
            out[view] = apply_stencil(s, stencil, axis, s[view].spatial_shape)
        return out

    def apply_operator(self,
                       s: Tuple[StateContainer]) -> Tuple[StateContainer]:
        """Apply the operator to ``s``.

        Must be overridden by subclasses to combine the stencil outputs
        with the appropriate geometric scaling (e.g. ``1 / dx**2`` for
        a second derivative).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, s: StateContainer) -> StateContainer:
        """Shorthand for :meth:`apply_operator`."""
        return self.apply_operator(s)

# }}}


class Laplacian(Operator):
    """One-dimensional Laplacian with Dirichlet boundary rows.

    The interior of the operator is the standard second-order
    three-point central difference

    .. math::

        (\\Delta f)_i = (f_{i-1} - 2 f_i + f_{i+1}) / \\Delta x^2.

    The boundary rows act as identity rows (``f_0`` and ``f_{N-1}``
    are taken unchanged) so that combining this operator with a
    Dirichlet right-hand side produces a system whose solution
    satisfies the prescribed boundary values.

    Parameters
    ----------
    domain : Domain
        Grid on which the Laplacian is defined.

    Attributes
    ----------
    stencils : list of Stencil
        Two single-point near-boundary stencils flanking a
        three-point interior stencil.
    central_blocks, lower_blocks, upper_blocks : jnp.ndarray
        Set by :meth:`assemble_block_form`.  Block-tridiagonal
        representation of the operator with one ``num_vars * num_vars``
        block per grid point.
    """

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
        """Build the block-tridiagonal matrix representation of the operator.

        Allocates ``central_blocks``, ``lower_blocks`` and
        ``upper_blocks``: arrays of ``num_x``, ``num_x - 1`` and
        ``num_x - 1`` blocks (respectively) of shape
        ``(num_vars, num_vars)``.  The interior rows hold the standard
        three-point central-difference Laplacian (multiplied by the
        identity in variable space), while the first and last central
        blocks act as identity Dirichlet rows scaled by ``1 / dx**2``.

        Parameters
        ----------
        num_vars : int
            Number of unknowns per grid point (``1 + num_species``
            for a flamelet).
        """

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
        """Apply the Laplacian to a :class:`FlameletState`.

        Each stencil is evaluated on its output slice, divided by
        ``dx ** 2`` and then concatenated into a single field-wise
        result.  The first and last entries of every field are the
        unaltered boundary values (identity rows), and the remaining
        entries form the discrete second derivative of ``s``.

        Parameters
        ----------
        s : FlameletState
            Input state on the full grid.

        Returns
        -------
        FlameletState
            ``Lap(s)`` evaluated on the full grid.
        """
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
