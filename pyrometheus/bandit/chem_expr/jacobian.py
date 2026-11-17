"""Generic symbolic differentiation for Bandit chemistry expression
graphs. Nothing here knows about PLATO, Cantera, MFC, or any specific
mechanism -- it only differentiates whatever pymbolic expression tree
it's handed. Mechanism-specific graph assembly (deciding *what* to
differentiate) lives in :mod:`pyrometheus.bandit.general_thermochem`.
"""

import pymbolic.primitives as p
from pymbolic import differentiate
from pymbolic.mapper.differentiator import map_math_functions_by_name


def _bandit_function_derivative_map(i, func, pars, allowed_nonsmoothness="none"):
    """pymbolic's stock differentiator recognizes transcendental calls
    written the ``math.exp(...)`` way (a :class:`~pymbolic.primitives.Lookup`
    on a ``math`` variable). This codebase instead calls them as bare
    ``Variable("exp")(...)``/``Variable("log")(...)``/``Variable("sqrt")(...)``
    (see ``chem_expr/kinetics.py``, ``impl/plato.py``'s geometric-mean
    temperature) -- rendered later as ``self.pyro_np.exp`` etc. by each
    codegen backend's ``map_call``. Recognize that calling convention
    here and fall back to the stock recognizer for anything else.
    """
    if func == p.Variable("exp") and len(pars) == 1:
        return p.Variable("exp")(*pars)
    elif func == p.Variable("log") and len(pars) == 1:
        return p.quotient(1, pars[0])
    elif func == p.Variable("sqrt") and len(pars) == 1:
        return p.quotient(1, 2 * p.Variable("sqrt")(*pars))
    else:
        return map_math_functions_by_name(i, func, pars, allowed_nonsmoothness)


def jacobian_row(expr, wrt_vars):
    """Return ``d(expr)/d(wrt_vars[k])`` for each *k*, via generic
    pymbolic differentiation.

    :arg expr: A :class:`pymbolic.primitives.ExpressionNode`.
    :arg wrt_vars: A list of leaf variables (or indexed/subscripted
        variables) appearing in *expr*.
    :returns: A list the same length as *wrt_vars*. Entries where
        *expr* doesn't structurally depend on ``wrt_vars[k]`` come back
        as the literal ``int`` ``0`` -- pymbolic's differentiation
        mapper returns 0 for any subtree not containing the target
        variable, which is what gives sparsity for free, with no
        special-casing needed here.

    NASA-polynomial thermo expressions branch (:class:`~pymbolic.primitives.If`)
    on which temperature interval a fit applies to, so this differentiates
    each branch and keeps the same condition (``allowed_nonsmoothness=
    "discontinuous"``) rather than refusing to differentiate through it --
    the branches are fit to agree in value and slope at the interval
    boundary, so this doesn't introduce a real discontinuity in practice.
    """
    return [
        differentiate(
            expr, var,
            func_mapper=_bandit_function_derivative_map,
            allowed_nonsmoothness="discontinuous",
        )
        for var in wrt_vars
    ]
