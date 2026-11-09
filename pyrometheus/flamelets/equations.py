"""Assembly of the residual, Jacobian and adjoint of the flamelet equations."""

import jax
import jax.numpy as jnp
from typing import Tuple
from pyrometheus.flamelets.state import (
    FlameletState, _state_to_array, _array_to_state
)


# Assuming the Laplacian has Dirichlet stencils at the boundaries,
# then we need to set the dissipation rate to 2*dx^2 at i = 0, N-1.
# The reason for this choice is the form of the equations that
# this class implements: 0.5 * chi * Lap(state)/dx**2 + S = 0
# where Lap = D1 * D1, with D1 the stencil matrix of first-order
# finite difference approximation.
class FlameletEquations:
    """Discrete residual and Jacobian of the steady flamelet equations.

    The class encodes

    .. math::

        \\tfrac{1}{2}\\, \\chi(Z)\\, \\partial_{ZZ}\\, \\phi
        + S(\\phi) - \\phi_{\\rm BC} = 0,

    where :math:`\\phi = (h, Y_1, \\ldots, Y_{n_s})` collects the
    mixture enthalpy and species mass fractions, :math:`\\chi(Z)` is
    the scalar dissipation rate and :math:`S(\\phi)` is the chemical
    source term.  Dirichlet boundary conditions are enforced by
    requiring identity rows of the Laplacian at the first and last
    grid points and by replacing the dissipation rate at those points
    with ``2 * dx**2`` (which exactly cancels the ``1 / dx**2`` baked
    into :meth:`Laplacian.apply_operator`).

    Besides the residual :meth:`rhs` and the block-tridiagonal
    Jacobian :meth:`jac`, the class also exposes:

    - :meth:`jacobian_action` -- matrix-free Jacobian--vector product,
      used by Krylov solvers;
    - :meth:`adjoint_operator` -- the (transposed) operator required
      by the EOS-consistency adjoint solves in
      :mod:`thermodynamic_consistency`;
    - :meth:`local_compressible_eos_rt` and the associated vmapped /
      jacfwd derivatives -- needed by the compressible EOS gradient
      computations.

    Parameters
    ----------
    pyro_gas : object
        Pyrometheus thermochemistry object (typically produced by
        :func:`make_pyro_object`).
    laplacian : Laplacian
        Laplacian operator on the mixture-fraction grid.  Its
        :meth:`assemble_block_form` is invoked during construction.
    y_ox, y_fu : jnp.ndarray
        Oxidizer- and fuel-side species mass-fraction vectors of
        length ``num_species``.

    Attributes
    ----------
    pyro_gas : object
        Pyrometheus thermochemistry object.
    laplacian : Laplacian
        Laplacian used to discretise the diffusion term.
    mol_wts : jnp.ndarray
        Species molecular weights, cached for source-term assembly.
    mass_frac_bc : jnp.ndarray
        Boundary-condition matrix of shape ``(num_species, num_x)``
        whose first and last columns hold ``y_ox`` and ``y_fu``,
        respectively, and whose interior is zero.
    interior_mask : FlameletState
        Mask that is one on interior nodes and zero at the boundary
        nodes; used to zero source terms at the boundaries.
    vmap_jacobian : callable
        ``jax.vmap`` of ``jax.jacfwd`` of :meth:`local_source_terms`
        with respect to the state vector at every grid point.
    source_gradient_wrt_pressure : callable
        ``jax.vmap`` of ``jax.jacfwd`` of :meth:`local_source_terms`
        with respect to the pressure.
    compressible_eos_rt, compressible_eos_rt_jacobian : callable
        ``jax.vmap`` (and ``jax.jacfwd``) of
        :meth:`local_compressible_eos_rt`, used by the compressible
        EOS-consistency machinery.
    """

    def __init__(self,
                 pyro_gas,
                 laplacian,
                 y_ox,
                 y_fu,):
        self.pyro_gas = pyro_gas
        self.laplacian = laplacian
        self.mol_wts = self.pyro_gas.molecular_weights
        self.laplacian.assemble_block_form(self.pyro_gas.num_species + 1)

        # Create boundary conditions
        self.mass_frac_bc = jnp.hstack((
            y_ox.reshape(-1, 1),
            jnp.zeros((
                self.pyro_gas.num_species,
                self.laplacian.domain.num_x-2
            )),
            y_fu.reshape(-1, 1)
        ))

        self.interior_mask = FlameletState(
            enthalpy=jnp.hstack((
                jnp.zeros(1),
                jnp.ones(self.laplacian.domain.num_x - 2),
                jnp.zeros(1)
            )),
            mass_fractions=jnp.hstack((
                jnp.zeros((self.pyro_gas.num_species, 1)),
                jnp.ones((
                    self.pyro_gas.num_species,
                    self.laplacian.domain.num_x - 2
                )),
                jnp.zeros((self.pyro_gas.num_species, 1))
            ))
        )

        # Source terms Jacobian and gradient wrt pressure
        self.vmap_jacobian = jax.vmap(
            jax.jacfwd(
                self.local_source_terms, argnums=0
            ),
            in_axes=(1, 0, 0, None)
        )
        self.source_gradient_wrt_pressure = jax.vmap(
            jax.jacfwd(
                self.local_source_terms, argnums=-1,
            ),
            in_axes=(1, 0, 0, None)
        )

        # Compressible EOS routines for thermodynamic consistency
        self.compressible_eos_rt = jax.vmap(
            self.local_compressible_eos_rt,
            in_axes=(1, 0, None)
        )
        self.compressible_eos_rt_jacobian = jax.vmap(
            jax.jacfwd(
                self.local_compressible_eos_rt,
                argnums=0
            ),
            in_axes=(1, 0, None)
        )

    def _enforce_dirichlet_bcs(self,
                               diss_rate: jnp.ndarray):
        """Overwrite the dissipation rate at the boundaries with ``2 * dx**2``.

        Because the Laplacian's boundary rows are identity rows scaled
        by ``1 / dx**2``, multiplying them by ``0.5 * (2 * dx**2)``
        recovers a clean identity row, which when balanced against
        :meth:`boundary_conditions` enforces the desired Dirichlet
        values.
        """
        diss_rate = diss_rate.at[0].set(
            2 * self.laplacian.domain.jac[0]**2
        )
        diss_rate = diss_rate.at[-1].set(
            2 * self.laplacian.domain.jac[0]**2
        )

    def rhs(self,
            state: FlameletState,
            diss_rate: jnp.ndarray,
            viscous_diss: jnp.ndarray,
            temp_guess: jnp.ndarray,
            pressure: jnp.float64,
            h_ox: jnp.float64,
            h_fu: jnp.float64,):
        """Evaluate the steady flamelet residual.

        Returns the field

        .. math::

            R(\\phi) = \\tfrac{1}{2}\\, \\chi\\, \\Delta \\phi
            + S(\\phi)\\, m_{\\rm int} - \\phi_{\\rm BC},

        where :math:`m_{\\rm int}` is :attr:`interior_mask` and
        :math:`\\phi_{\\rm BC}` is constructed by
        :meth:`boundary_conditions` from ``h_ox``, ``h_fu`` and the
        stored ``mass_frac_bc``.  The dissipation rate at the
        boundary nodes is patched by :meth:`_enforce_dirichlet_bcs`.

        Parameters
        ----------
        state : FlameletState
            Current iterate of ``(h, Y)``.
        diss_rate : jnp.ndarray
            Scalar dissipation-rate field, length ``num_x``.
        viscous_diss : jnp.ndarray
            Viscous dissipation contribution to the enthalpy source,
            length ``num_x``.
        temp_guess : jnp.ndarray
            Initial guess for the implicit temperature solve.
        pressure : float
            Thermodynamic pressure.
        h_ox, h_fu : float
            Enthalpy values imposed at the oxidizer and fuel
            boundaries.

        Returns
        -------
        FlameletState
            Field-wise residual whose zero defines the steady-state
            flamelet solution.
        """
        self._enforce_dirichlet_bcs(diss_rate)
        return (
            0.5 * diss_rate * self.laplacian(state)
            + self.source_terms(
                state,
                viscous_diss,
                temp_guess,
                pressure,
            ) * self.interior_mask
            - self.boundary_conditions(
                h_ox,
                h_fu
            )
        )

    def jac(self,
            state: FlameletState,
            diss_rate: jnp.ndarray,
            viscous_diss: jnp.ndarray,
            temp_guess: jnp.ndarray,
            pressure: jnp.float64,) -> Tuple[jnp.ndarray, ...]:
        """Assemble the block-tridiagonal Jacobian of :meth:`rhs`.

        Computes the per-grid-point Jacobian of the source term with
        :attr:`vmap_jacobian`, zeroes its first and last entries
        (which lie on Dirichlet rows) and adds it to the appropriately
        scaled Laplacian block tridiagonal.

        Parameters
        ----------
        state, diss_rate, viscous_diss, temp_guess, pressure
            See :meth:`rhs`.

        Returns
        -------
        tuple of jnp.ndarray
            ``(lower_blocks, central_blocks, upper_blocks)`` ready to
            be passed to :func:`block_thomas`.
        """
        self._enforce_dirichlet_bcs(diss_rate)
        num_v = self.pyro_gas.num_species + 1
        zeros_nv = jnp.zeros((1, num_v, num_v))
        source_jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss,
            temp_guess,
            pressure
        )
        source_blocks = jnp.concatenate((
            zeros_nv, source_jacobian[1:-1], zeros_nv
        ))
        central_blocks = (
            jnp.einsum(
                "i,ijk->ijk",
                0.5 * diss_rate,
                self.laplacian.central_blocks
            )
            + source_blocks
        )
        lower_blocks = jnp.einsum(
            "i,ijk->ijk",
            0.5 * diss_rate[1:],
            self.laplacian.lower_blocks
        )
        upper_blocks = jnp.einsum(
            "i,ijk->ijk",
            0.5 * diss_rate[:-1],
            self.laplacian.upper_blocks
        )
        return (lower_blocks, central_blocks, upper_blocks)

    def equation_of_state(self,
                          state: FlameletState,
                          temp_guess: jnp.ndarray,
                          pressure: jnp.float64,):
        """Return ``(density, temperature)`` from the current state.

        Recovers the temperature implicitly from the enthalpy via the
        pyrometheus helper ``get_temperature_from_enthalpy`` and then
        uses the ideal-gas equation of state to compute density.

        Parameters
        ----------
        state : FlameletState
            Current ``(h, Y)``.
        temp_guess : jnp.ndarray
            Initial guess for the implicit temperature solve.
        pressure : float
            Thermodynamic pressure.
        """
        temperature = self.pyro_gas.get_temperature_from_enthalpy(
            state.enthalpy,
            state.mass_fractions,
            temp_guess
        )
        density = self.pyro_gas.get_density(
            pressure, temperature, state.mass_fractions
        )
        return density, temperature

    def source_terms(self,
                     state: FlameletState,
                     viscous_diss: jnp.ndarray,
                     temp_guess: jnp.ndarray,
                     pressure: jnp.float64,):
        """Return the field-wise chemical source term of the flamelet equations.

        The enthalpy equation is forced by ``viscous_diss / density``;
        the species equations are forced by
        ``mol_wts * omega_dot / density``, where ``omega_dot`` is the
        molar net production rate evaluated by ``pyro_gas``.

        Parameters
        ----------
        state : FlameletState
            Current ``(h, Y)``.
        viscous_diss : jnp.ndarray
            Viscous dissipation forcing for the enthalpy equation.
        temp_guess : jnp.ndarray
            Initial guess for the implicit temperature solve.
        pressure : float
            Thermodynamic pressure.

        Returns
        -------
        FlameletState
            Source term :math:`S(\\phi)` of the flamelet equations.
        """
        density, temperature = self.equation_of_state(
            state,
            temp_guess,
            pressure,
        )
        w_dot = self.pyro_gas.get_net_production_rates(
            density,
            temperature,
            state.mass_fractions
        )
        m_dot = (
            jnp.expand_dims(self.mol_wts, w_dot.ndim-1)
            * w_dot
        ).squeeze()

        return FlameletState(
            enthalpy=viscous_diss / density,
            mass_fractions=m_dot / density
        )

    def local_source_terms(self,
                           local_state_as_array: jnp.ndarray,
                           viscous_diss: jnp.ndarray,
                           temp_guess: jnp.ndarray,
                           pressure: jnp.float64) -> jnp.ndarray:
        """Single-grid-point version of :meth:`source_terms`.

        Operates on the flat state array ``[h, Y_1, ..., Y_ns]`` at a
        single grid point and returns the corresponding source vector,
        as a flat array.  Used as the building block for the
        autodiffed, vmapped Jacobians stored as
        :attr:`vmap_jacobian` and
        :attr:`source_gradient_wrt_pressure`.
        """
        local_state = FlameletState(
            enthalpy=local_state_as_array[0],
            mass_fractions=local_state_as_array[1:]
        )
        s_dot = self.source_terms(
            state=local_state,
            viscous_diss=viscous_diss,
            temp_guess=temp_guess,
            pressure=pressure
        )
        return jnp.concatenate(
            (jnp.atleast_1d(s_dot.enthalpy),
             s_dot.mass_fractions.squeeze()),
            axis=0
        )

    def boundary_conditions(self,
                            h_ox: jnp.float64,
                            h_fu: jnp.float64):
        """Build the Dirichlet boundary-condition state used by :meth:`rhs`.

        The interior of the state is zero; the first and last
        enthalpy entries are ``h_ox`` and ``h_fu``, and the first and
        last species columns are ``y_ox`` and ``y_fu`` (taken from
        the cached :attr:`mass_frac_bc`).
        """

        return FlameletState(
            enthalpy=jnp.hstack((
                h_ox,
                jnp.zeros(self.laplacian.domain.num_x-2),
                h_fu
            )),
            mass_fractions=self.mass_frac_bc
        )

    def jacobian_action(self,
                        v: FlameletState,
                        state: FlameletState,
                        diss_rate: jnp.ndarray,
                        viscous_diss: jnp.ndarray,
                        temp_guess: jnp.ndarray,
                        pressure: jnp.float64,):
        """Matrix-free product of the flamelet Jacobian with a direction ``v``.

        Returns

        .. math::

            J(\\phi)\\, v = \\tfrac{1}{2}\\, \\chi\\, \\Delta v
            + \\frac{\\partial S}{\\partial \\phi}(\\phi)\\, v,

        evaluated point-wise.

        Parameters
        ----------
        v : FlameletState
            Direction in state space.
        state, diss_rate, viscous_diss, temp_guess, pressure
            See :meth:`rhs`.
        """
        self._enforce_dirichlet_bcs(diss_rate)
        jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss, temp_guess, pressure
        )
        jvp = jnp.einsum(
            "ijk,ki->ji",
            jacobian,  # (nx, nv, nv)
            _state_to_array(v)  # (nv, nx)
        )  # (nv, nx)
        return (
            0.5 * diss_rate * self.laplacian(v)
            + _array_to_state(jvp)
        )

    def adjoint_operator(self,
                         state: FlameletState,
                         diss_rate: jnp.ndarray,
                         viscous_diss: jnp.ndarray,
                         temp_guess: jnp.ndarray,
                         pressure: jnp.float64,) -> Tuple[jnp.ndarray, ...]:
        """Assemble the block-tridiagonal adjoint operator.

        Builds the transpose of the linearised steady operator,
        normalised so that the interior source-Jacobian contribution
        appears as ``2 * J_S^T / chi`` (dividing out the ``0.5 * chi``
        factor from :meth:`jac`) and the boundary rows are returned
        as plain identity rows in mixture-fraction units (the
        ``1 / dx**2`` factor encoded in the Laplacian boundary blocks
        is removed by multiplying by ``0.5 * chi`` at the boundaries,
        whose dissipation rates carry the ``2 * dx**2`` factor set by
        :meth:`_enforce_dirichlet_bcs`).

        Parameters
        ----------
        state, diss_rate, viscous_diss, temp_guess, pressure
            See :meth:`rhs`.

        Returns
        -------
        tuple of jnp.ndarray
            ``(lower_blocks, central_blocks, upper_blocks)`` of the
            adjoint operator, ready to be passed to
            :func:`block_thomas`.
        """
        self._enforce_dirichlet_bcs(diss_rate)
        num_v = self.pyro_gas.num_species + 1
        zeros_nv = jnp.zeros((1, num_v, num_v))
        source_jacobian = self.vmap_jacobian(
            _state_to_array(state),
            viscous_diss,
            temp_guess,
            pressure
        )
        source_blocks = jnp.concatenate((
            zeros_nv,
            jnp.transpose(
                2 * source_jacobian[1:-1] / diss_rate[1:-1, None, None],
                (0, 2, 1)
            ),
            zeros_nv
        ))
        # Central blocks: need to remove 1/dx**2 factor from the BCs
        # This factor is stored in the first and last entry of the dissipation rate
        central_blocks = (
            jnp.concatenate((
                jnp.expand_dims(
                    0.5 * diss_rate[0] * self.laplacian.central_blocks[0],
                    axis=0
                ),
                self.laplacian.central_blocks[1:-1],
                jnp.expand_dims(
                    0.5 * diss_rate[-1] * self.laplacian.central_blocks[-1],
                    axis=0
                )
            ))
            + source_blocks
        )
        lower_blocks = self.laplacian.lower_blocks
        upper_blocks = self.laplacian.upper_blocks
        return (lower_blocks, central_blocks, upper_blocks)

    def local_compressible_eos_rt(self,
                                  local_state_as_array: jnp.ndarray,
                                  local_temp_guess: jnp.float64,
                                  pressure: jnp.float64):
        """Per-grid-point evaluation of ``R T`` with ``R = R_u / W_mix``.

        Recovers the temperature implicitly from the local enthalpy
        and mass fractions and combines it with the mixture-specific
        gas constant.  Used by the compressible-EOS consistency
        machinery in :mod:`thermodynamic_consistency`.
        """
        w_mix = self.pyro_gas.get_mixture_molecular_weight(
            local_state_as_array[1:]
        )
        temperature = self.pyro_gas.get_temperature_from_enthalpy(
            local_state_as_array[0], local_state_as_array[1:], local_temp_guess
        )
        return self.pyro_gas.gas_constant * temperature / w_mix
