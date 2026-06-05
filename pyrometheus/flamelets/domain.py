import numpy as np
from dataclasses import dataclass


@dataclass
class DomainConfig:
    """Configuration for a one-dimensional mixture-fraction domain.

    Attributes
    ----------
    num_x : int
        Number of grid points used to discretize the mixture-fraction
        interval, including the two boundary nodes.
    x_l, x_r : float
        Left (oxidizer) and right (fuel) endpoints of the
        mixture-fraction domain.  In the standard flamelet formulation
        these are 0 and 1, respectively.
    dtype : numpy.dtype, optional
        Floating-point precision used for the grid arrays.  Defaults to
        ``numpy.float64``.
    pyro_np : module
        Array library used to allocate the grid.  Defaults to ``numpy``;
        set to ``jax.numpy`` to produce a JAX-traceable domain.
    num_dim : int, optional
        Number of spatial dimensions of the domain.  Only ``num_dim=1``
        is currently supported by the flamelet solver.
    """

    num_x: int
    x_l: float
    x_r: float
    dtype: np.dtype = np.float64
    pyro_np = np
    num_dim: int = 1


class Domain:
    """Uniform one-dimensional mixture-fraction grid.

    Builds the discrete grid ``x`` over ``[x_l, x_r]`` together with
    the constant grid spacing ``dx`` and its associated Jacobian
    ``jac = [dx]``.  Instances are consumed by :class:`Laplacian` and
    :class:`FlameletEquations` to assemble the finite-difference
    representations of the flamelet operators.

    Parameters
    ----------
    domain_config : DomainConfig
        Discretization parameters describing the mixture-fraction
        interval.

    Attributes
    ----------
    dtype : numpy.dtype
        Floating-point precision used for the grid arrays.
    num_dim : int
        Number of spatial dimensions (always ``1`` in this package).
    num_x : int
        Number of grid points along the mixture-fraction axis.
    pyro_np : module
        Array library used to allocate the grid (``numpy`` or
        ``jax.numpy``).
    dx : float
        Constant grid spacing in mixture fraction.
    x : ndarray
        Grid coordinates of shape ``(num_x,)`` spanning ``[x_l, x_r]``.
    jac : ndarray
        One-element array ``[dx]`` containing the geometric Jacobian of
        the mapping from index to mixture fraction.  Used by stencil
        and Laplacian assembly routines.
    """

    def __init__(self, domain_config: DomainConfig):
        self.dtype = domain_config.dtype
        self.num_dim = domain_config.num_dim
        self.num_x = domain_config.num_x
        self.pyro_np = domain_config.pyro_np

        self.dx = (domain_config.x_r - domain_config.x_l) / (self.num_x - 1)
        self.x = self.pyro_np.linspace(
            domain_config.x_l,
            domain_config.x_r,
            self.num_x,
            endpoint=True,
        )
        self.jac = self.pyro_np.array([self.dx])
