"""Helpers shared by the parity suites that compare generated code with
the library a :class:`~pyrometheus.bandit.general_thermochem.BaseMechanism`
was read from.
"""

import numpy as np


def make_generated_gas(mech, pyro_np=np):
    """:returns: An instance of the Python class generated for *mech*,
    able to evaluate batched states.
    """
    from pyrometheus.codegen.python_bandit import PythonBanditCodeGenerator
    base_class = PythonBanditCodeGenerator.get_thermochem_class(mech)

    class StackedGas(base_class):
        # The default object-array assembly cannot hold batched states.
        def _pyro_make_array(self, res_list):
            return self.pyro_np.stack([
                self.pyro_np.asarray(entry, dtype=self.pyro_np.float64)
                for entry in res_list
            ])

    return StackedGas(pyro_np)


def temperature_subscripts(expressions):
    """:returns: Every index that *expressions* subscript ``temperature``
    with.
    """
    from pymbolic.mapper import WalkMapper
    import pymbolic.primitives as prim

    class Collector(WalkMapper):
        def __init__(self):
            self.indices = set()

        def visit(self, expr, *args, **kwargs):
            if (isinstance(expr, prim.Subscript)
                    and expr.aggregate == prim.Variable("temperature")):
                self.indices.add(expr.index)
            return True

    collector = Collector()
    for expr in expressions:
        collector(expr)
    return collector.indices


def mechanism_expressions(mech):
    """:returns: The rate, thermo and mass-action expressions of *mech*,
    which between them hold every temperature it reads.
    """
    return [rate.expr for rate in mech.rate_coeffs] + [
        poly.expr
        for thermo in mech.species_nasa_thermo_polynomials
        for poly in (thermo.cp_poly, thermo.enthalpy_poly, thermo.gibbs_poly)
    ] + list(mech.mass_action_rates)
