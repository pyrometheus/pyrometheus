"""Miscellaneous helpers used to set up a flamelet problem."""

import numpy as np
import jax.numpy as jnp
from itertools import product
from scipy.special import erfcinv


def bell_profile(z):
    """Smoluchowski-style bell profile for the scalar dissipation rate.

    Returns the analytic profile

    .. math::

        \\chi(Z) \\propto \\exp\\!\\big(-2\\, \\mathrm{erfcinv}(2 Z)^2\\big),

    commonly used as the mixture-fraction-dependent shape of
    :math:`\\chi(Z)` for a counterflow flamelet.

    Parameters
    ----------
    z : array_like
        Mixture-fraction values in ``(0, 1)``.
    """
    return jnp.exp(-2 * erfcinv(2 * z) ** 2)


def stoichiometric_mixture_fraction(sol, y_ox, y_fu):
    """Bilger stoichiometric mixture fraction for a fuel/oxidizer pair.

    Computes the conventional Bilger coupling function from the
    oxidizer- and fuel-stream mass fractions ``y_ox`` and ``y_fu`` and
    extracts the value at stoichiometry.  Element weights
    :math:`2/W_C`, :math:`1/(2 W_H)` and :math:`-1/W_O` are used for
    carbon, hydrogen and oxygen, respectively; all other elements
    receive a zero weight.

    Parameters
    ----------
    sol : Cantera-like Solution
        Object that exposes ``element_names``, ``molecular_weights``,
        ``n_species``, ``n_elements`` and ``n_atoms(species,
        element)``.
    y_ox, y_fu : ndarray
        Oxidizer- and fuel-side mass-fraction vectors.

    Returns
    -------
    float
        Stoichiometric mixture fraction
        :math:`Z_{\\mathrm{st}} = -\\beta_{\\mathrm{ox}} /
        (\\beta_{\\mathrm{fu}} - \\beta_{\\mathrm{ox}})`.
    """

    _bilger_dict = {
        "C": 2,
        "H": 0.5,
        "O": -1
    }

    wts_e = {
        "C": 12.011,
        "H": 1.00794,
        "O": 15.999,
        "Ar": 39.948,
        "N": 28.0133
    }

    wts_s = sol.molecular_weights
    bilger_weights = np.zeros(sol.n_elements)
    for i, e in enumerate(sol.element_names):
        if e in _bilger_dict:
            bilger_weights[i] = _bilger_dict[e] / wts_e[e]
        else:
            bilger_weights[i] = 0

    ns = sol.n_species
    ne = sol.n_elements
    elem_matrix = np.zeros((ne, ns))
    for e, s in product(range(ne), range(ns)):
        e_name = sol.element_names[e]
        elem_matrix[e, s] = sol.n_atoms(s, e) * (
            wts_e[e_name] / wts_s[s]
        )

    beta_ox = bilger_weights.dot(
        elem_matrix.dot(y_ox)
    )
    beta_fu = bilger_weights.dot(
        elem_matrix.dot(y_fu)
    )
    return -beta_ox / (beta_fu - beta_ox)
