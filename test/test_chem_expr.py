"""Behavior of Bandit's symbolic expression builders that does not depend
on a thermochemistry library.
"""

import pytest

from pyrometheus.bandit.chem_expr.thermo import nasa_poly_gibbs_expr


def test_gibbs_expressions_reject_what_is_not_a_polynomial():
    with pytest.raises(TypeError, match="str"):
        nasa_poly_gibbs_expr("not a polynomial")
