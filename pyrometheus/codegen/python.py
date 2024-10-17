"""
Python code generation
----------------------

.. autofunction:: gen_thermochem_code
.. autofunction:: get_thermochem_class

"""


__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from numbers import Number

import pymbolic.primitives as p
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
import cantera as ct
import numpy as np  # noqa: F401

from mako.template import Template
import pyrometheus.chem_expr


file_extension = "py"


# {{{ code generation helpers


class CodeGenerationMapper(StringifyMapper):
    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    OP_NAMES = {
            ">=": "greater_equal",
            ">": "greater",
            "==": "equal",
            "!=": "not_equal",
            "<=": "less_equal",
            "<": "less",
            }

    def map_comparison(self, expr, enclosing_prec, *args, **kwargs):
        return (f"self.usr_np.{self.OP_NAMES[expr.operator]}"
            f"({self.rec(expr.left, PREC_NONE, *args, **kwargs)}, "
            f"{self.rec(expr.right, PREC_NONE, *args, **kwargs)})")

    def map_if(self, expr, enclosing_prec, *args, **kwargs):
        return "self.usr_np.where(%s, %s, %s)" % (
            self.rec(expr.condition, PREC_NONE, *args, **kwargs),
            self.rec(expr.then, PREC_NONE, *args, **kwargs),
            self.rec(expr.else_, PREC_NONE, *args, **kwargs),
        )

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "self.usr_np.%s(%s)",
            self.rec(expr.function, PREC_CALL, *args, **kwargs),
            self.join_rec(", ", expr.parameters, PREC_NONE, *args, **kwargs),
        )


def str_np_inner(ary):
    if isinstance(ary, Number):
        return repr(ary)
    elif ary.shape:
        return "[%s]" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return "np.array(%s)" % str_np_inner(ary)


# }}}


# {{{ main code template

code_tpl = Template(
    """\"""
.. autoclass:: Thermochemistry
\"""


from warnings import warn
import numpy as np


class Thermochemistry:
    \"""
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    \"""

    def __init__(self, usr_np=np):
        \"""Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        \"""

        self.usr_np = usr_np
        self.model_name = ${repr(sol.source)}
        self.num_elements = ${sol.n_elements}
        self.num_species = ${sol.n_species}
        self.num_reactions = ${sol.n_reactions}
        self.num_falloff = ${
            sum(1 if isinstance(r, ct.FalloffReaction) else 0
            for r in sol.reactions())}

        self.one_atm = ${ct.one_atm}
        self.gas_constant = ${ct.gas_constant}
        self.big_number = 1.0e300

        self.species_names = ${sol.species_names}
        self.species_indices = ${
            dict([[sol.species_name(i), i]
                for i in range(sol.n_species)])}

        self.molecular_weights = ${str_np(sol.molecular_weights)}
        self.inv_molecular_weights = 1/self.molecular_weights

    @property
    def wts(self):
        warn("Thermochemistry.wts is deprecated and will go away in 2024. "
             "Use molecular_weights instead.", DeprecationWarning, stacklevel=2)

        return self.molecular_weights

    @property
    def iwts(self):
        warn("Thermochemistry.iwts is deprecated and will go away in 2024. "
             "Use inv_molecular_weights instead.", DeprecationWarning, stacklevel=2)

        return self.inv_molecular_weights

    def _pyro_zeros_like(self, argument):
        # FIXME: This is imperfect, as a NaN will stay a NaN.
        return 0 * argument

    def _pyro_make_array(self, res_list):
        \"""This works around (e.g.) numpy.exp not working with object
        arrays of numpy scalars. It defaults to making object arrays, however
        if an array consists of all scalars, it makes a "plain old"
        :class:`numpy.ndarray`.

        See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
        for more context.
        \"""

        from numbers import Number
        all_numbers = all(isinstance(e, Number) for e in res_list)

        dtype = np.float64 if all_numbers else object
        result = np.empty((len(res_list),), dtype=dtype)

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for idx in range(len(res_list)):
            result[idx] = res_list[idx]

        return result

    def _pyro_norm(self, argument, normord):
        \"""This works around numpy.linalg norm not working with scalars.

        If the argument is a regular ole number, it uses :func:`numpy.abs`,
        otherwise it uses ``usr_np.linalg.norm``.
        \"""
        # Wrap norm for scalars

        from numbers import Number

        if isinstance(argument, Number):
            return np.abs(argument)
        return self.usr_np.linalg.norm(argument, normord)

    def get_species_name(self, species_index):
        return self.species_name[species_index]

    def get_species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * (
                %for i in range(sol.n_species):
                    + self.inv_molecular_weights[${i}]*mass_fractions[${i}]
                %endfor
                )

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
                %for i in range(sol.n_species):
                    + self.inv_molecular_weights[${i}]*mass_fractions[${i}]
                %endfor
                )

    def get_concentrations(self, rho, mass_fractions):
        return self._pyro_make_array([
            %for i in range(sol.n_species):
            self.inv_molecular_weights[${i}] * rho * mass_fractions[${i}],
            %endfor
        ])

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([
            mass_fractions[i]
            * spec_property[i]
            * self.inv_molecular_weights[i]
            for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_specific_heats_r_derivative(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
            ${cgm(ce.poly_deriv_to_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_enthalpies_rt_derivative(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
            ${cgm(ce.poly_deriv_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_entropies_rt_derivative(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
            ${cgm(ce.poly_deriv_to_entropy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_specific_heats_r(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
            ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
            ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_entropies_r(self, temperature):
        return self._pyro_make_array([
            % for sp in sol.species():
                ${cgm(ce.poly_to_entropy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_gibbs_rt(self, temperature):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
            %for i, react in enumerate(sol.reactions()):
                %if react.reversible:
                    ${cgm(ce.equilibrium_constants_expr(
                        sol, i, Variable("g0_rt")))},
                %else:
                    -0.17364695002734*temperature,
                %endif
            %endfor
                ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
        iter_temp = t_guess * ones

        for _ in range(num_iter):
            iter_rhs = enthalpy_or_energy - he_fun(iter_temp, y)
            iter_deriv = -pv_fun(iter_temp, y)
            dt = -iter_rhs / iter_deriv
            iter_temp += dt
            if self._pyro_norm(dt, np.inf) < tol:
                return iter_temp

        raise RuntimeError("Temperature iteration failed to converge")

    %if falloff_reactions:
    def get_falloff_rates(self, temperature, concentrations, k_fwd):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_high = self._pyro_make_array([
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.high_rate, Variable("temperature")))},
        %endfor
                ])

        k_low = self._pyro_make_array([
        %for _, react in falloff_reactions:
            ${cgm(ce.rate_coefficient_expr(
                react.rate.low_rate, Variable("temperature")))},
        %endfor
                ])

        reduced_pressure = self._pyro_make_array([
        %for i, (_, react) in enumerate(falloff_reactions):
            (${cgm(ce.third_body_efficiencies_expr(
                sol, react, Variable("concentrations")))})*k_low[${i}]/k_high[${i}],
        %endfor
                            ])

        falloff_center = self._pyro_make_array([
        %for _, react in falloff_reactions:
            ${cgm(ce.troe_falloff_center_expr(react, Variable("temperature")))},
        %endfor
                        ])

        falloff_factor = self._pyro_make_array([
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.troe_falloff_factor_expr(react, i,
            Variable("reduced_pressure"), Variable("falloff_center")))},
        %endfor
        ])

        falloff_function = self._pyro_make_array([
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.falloff_function_expr(
                react, i,
                Variable("falloff_factor"),
                Variable("falloff_center")))},
        %endfor
                            ])*reduced_pressure/(1+reduced_pressure)

        %for j, (i, react) in enumerate(falloff_reactions):
        k_fwd[${i}] = k_high[${j}]*falloff_function[${j}]*ones
        %endfor
        return

    %endif
    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
        %for react in sol.reactions():
        %if react.equation in [r.equation for _, r in falloff_reactions]:
            0*temperature,
        %else:
            ${cgm(ce.rate_coefficient_expr(react.rate,
                                        Variable("temperature")))} * ones,
        %endif
        %endfor
                ]
        %if falloff_reactions:
        self.get_falloff_rates(temperature, concentrations, k_fwd)
        %endif

        %for i, react in three_body_reactions:
        k_fwd[${i}] *= (${cgm(ce.third_body_efficiencies_expr(
            sol, react, Variable("concentrations")))})
        %endfor
        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                %for i in range(sol.n_reactions):
                    ${cgm(ce.rate_of_progress_expr(sol, i,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("log_k_eq")))},
                %endfor
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
            %for sp in sol.species():
                ${cgm(ce.production_rate_expr(
                    sol, sp.name, Variable("r_net")))} * ones,
            %endfor
               ])""", strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution) -> str:
    """For the mechanism given by *sol*, return Python source code for a class
    conforming to a module containing a class called ``Thermochemistry``
    adhering to the :class:`~pyrometheus.thermochem_example.Thermochemistry`
    interface.
    """

    falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                   if r.reaction_type.startswith("falloff")]
    three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                      if r.reaction_type == "three-body-Arrhenius"]

    return code_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        ce=pyrometheus.chem_expr,

        falloff_reactions=falloff_rxn,
        three_body_reactions=three_body_rxn,
    )


def compile_class(code_str, class_name="Thermochemistry"):
    exec_dict = {}
    exec(compile(code_str, "<generated code>", "exec"), exec_dict)
    exec_dict["_MODULE_SOURCE_CODE"] = code_str

    return exec_dict[class_name]


def get_thermochem_class(sol: ct.Solution):
    """For the mechanism given by *sol*, return a class conforming to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
    return compile_class(gen_thermochem_code(sol))


def cti_to_mech_file(cti_file_name, mech_file_name):
    """Write python file for mechanism specified by CTI file."""
    with open(mech_file_name, "w") as outf:
        code = gen_thermochem_code(ct.Solution(cti_file_name, "gas"))
        print(code, file=outf)

# vim: foldmethod=marker
