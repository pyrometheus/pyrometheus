import numpy as np
import jax.numpy as jnp
from itertools import product
from scipy.special import erfcinv


def bell_profile(z):
    return jnp.exp(-2 * erfcinv(2 * z) ** 2)


def stoichiometric_mixture_fraction(sol, y_ox, y_fu):

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
