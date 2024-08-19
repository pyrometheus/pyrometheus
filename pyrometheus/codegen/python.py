"""
Python code generation
----------------------

.. autofunction:: gen_thermochem_code
.. autofunction:: get_thermochem_class

"""


__copyright__ = """
Copyright (C) 2020 Esteban Cisneros
Copyright (C) 2020 Andreas Kloeckner
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
import torch


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
    .. automethod:: get_specific_heat_ratio
    .. automethod:: get_sound_speed
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_enthalpies_deriv
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: get_temperature_RPY
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

        self.molecular_weights = self._pyro_make_tensor(${str_np_inner(sol.molecular_weights)})
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

    def _pyro_make_tensor(self, res_list):
        if self.usr_np is torch:
            if torch.is_tensor(res_list[0]):
                for i,val in enumerate(res_list):
                    res_list[i] = val.squeeze()
                return torch.stack(res_list, 0)
            else:
                return self.usr_np.tensor(res_list, dtype=torch.float64)
        else:
            return self.usr_np.array(res_list)

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

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
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
        return self._pyro_make_tensor([
                %for i in range(sol.n_species):
                    self.iwts[${i}] * mass_fractions[${i}] * rho,
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

    def get_specific_heat_ratio(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        inv_mix_mol_weight = sum([
            mass_fractions[i]
            * self.inv_molecular_weights[i]
            for i in range(self.num_species)])
        return cpmix / (cpmix - inv_mix_mol_weight)

    def get_sound_speed(self, temperature, mass_fractions):
        gamma = self.get_specific_heat_ratio(temperature, mass_fractions)
        mix_gas_constant = self.get_specific_gas_constant(mass_fractions)
        return self.usr_np.sqrt(gamma * mix_gas_constant * temperature)

    def get_species_specific_heats_r(self, temperature):
        return self._pyro_make_tensor([
            % for sp in sol.species():
            ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_tensor([
            % for sp in sol.species():
            ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
                ])
    
    def get_species_enthalpies_deriv(self, temperature, h_rt=None):
        \"""The derivative of the NASA polynomial for enthalpy is not the 
        temperature derivative of enthalpy because the NASA polynomials return
        h_k(T)/RT. So, if we let x_k = h_k(T)/RT, the derivative is dx_k/dT.
        Because h_k(T) = x_k*R*T, we can get the temperature derivative of
        h_k(T) by applying the chain and product rules:
        
            dh_k(T)/dT = d(x_k*R*T)/dT = R*(x_k + T*dx_k/dT),
            
        where x_k is computed using `get_species_enthalpies_rt` and dx_k/dT
        is the result of differentiating the NASA polynomial.        
        \"""
        h_rt_T_deriv = self._pyro_make_tensor([
            % for sp in sol.species():
            ${cgm(ce.poly_to_enthalpy_deriv_expr(sp.thermo, "temperature"))},
            % endfor
                ])
            
        # Makes use of already computed h_rt if available.
        if h_rt is None:
            h_rt = self.get_species_enthalpies_rt(temperature)
        
        return self.gas_constant*(h_rt + temperature*h_rt_T_deriv)

    def get_species_entropies_r(self, temperature):
        return self._pyro_make_tensor([
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
        return self._pyro_make_tensor([
            %for i, react in enumerate(sol.reactions()):
                %if react.reversible:
                    ${cgm(ce.equilibrium_constants_expr(
                        sol, i, Variable("g0_rt")))},
                %else:
                    -0.17364695002734*temperature,
                %endif
            %endfor
                ])

    def get_temperature_RPY(self, rho, p, Y):
        return p / (rho * self.get_specific_gas_constant(Y))

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
        t_i = t_guess * ones

        for _ in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i = t_i + dt
            if self._pyro_norm(dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    %if falloff_reactions:
    def get_falloff_rates(self, temperature, concentrations, k_fwd):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_high = self._pyro_make_array([
        %for _, react in falloff_reactions:
            %if 'uses_legacy' in dir(react) and react.uses_legacy:
            ${cgm(ce.rate_coefficient_expr(
                react.high_rate, Variable("temperature")))},
            %else:
            ${cgm(ce.ArrheniusMapper().fixed_coeffs(
                ce.ArrheniusExpression(Variable("a"),
                Variable("b"), Variable("t_act"),
                Variable("temperature")),
                react.rate.high_rate))} * ones,
            %endif
        %endfor
                ])

        k_low = self._pyro_make_array([
        %for _, react in falloff_reactions:
            %if 'uses_legacy' in dir(react) and react.uses_legacy:
            ${cgm(ce.rate_coefficient_expr(
                react.low_rate, Variable("temperature")))},
            %else:
            ${cgm(ce.ArrheniusMapper().fixed_coeffs(
                ce.ArrheniusExpression(Variable("a"),
                Variable("b"), Variable("t_act"),
                Variable("temperature")),
                react.rate.low_rate))} * ones,
            %endif
        %endfor
                ])

        reduced_pressure = self._pyro_make_tensor([
        %for i, (_, react) in enumerate(falloff_reactions):
            (${cgm(ce.third_body_efficiencies_expr(
                sol, react, Variable("concentrations")))})*k_low[${i}]/k_high[${i}],
        %endfor
                            ])

        falloff_center = self._pyro_make_tensor([
        %for _, react in falloff_reactions:
            ${cgm(ce.troe_falloff_expr(react, Variable("temperature")))} * ones,
        %endfor
                        ])

        falloff_function = self._pyro_make_tensor([
        %for i, (_, react) in enumerate(falloff_reactions):
            ${cgm(ce.falloff_function_expr(
                react, i, Variable("temperature"), Variable("reduced_pressure"),
                Variable("falloff_center")))} * ones,
        %endfor
                            ])*reduced_pressure/(1+reduced_pressure)

        %for j, (i, react) in enumerate(falloff_reactions):
        k_fwd[${i}] = k_high[${j}]*falloff_function[${j}]*ones
        %endfor
        return

    %endif
    %if fixed_coeffs:
    def get_fwd_rate_coefficients(self, temperature, concentrations):
    %else:
    def get_fwd_rate_coefficients(self, temperature, concentrations, rate_params):
        a = rate_params[0]
        b = rate_params[1]
        t_act = rate_params[2]
    %endif
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
        %for i, react in enumerate(sol.reactions()):
        %if react.equation in [r.equation for _, r in falloff_reactions]:
            0*temperature,
        %else:
            %if fixed_coeffs:
                ${cgm(ce.ArrheniusMapper().fixed_coeffs(
                    ce.ArrheniusExpression(Variable("a"),
                    Variable("b"), Variable("t_act"),
                    Variable("temperature")),
                    sol.reaction(i).rate))} * ones,
            %else:
                ${cgm(ce.ArrheniusMapper()(
                    ce.ArrheniusExpression(Variable("a"),
                    Variable("b"), Variable("t_act"),
                    Variable("temperature")), i))} * ones,
            %endif
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
        return k_fwd

    %if fixed_coeffs:
    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
    %else:
    def get_net_rates_of_progress(self,temperature, concentrations, rate_params):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations,
                                               rate_params)
    %endif
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_tensor([
                %for i in range(sol.n_reactions):
                    ${cgm(ce.rate_of_progress_expr(sol, i,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("log_k_eq")))},
                %endfor
               ])

    %if fixed_coeffs:
    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
    %else:
    def get_net_production_rates(self, rho, temperature, mass_fractions,
                                 rate_params):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c, rate_params)
    %endif
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_tensor([
            %for sp in sol.species():
                ${cgm(ce.production_rate_expr(
                    sol, sp.name, Variable("r_net")))} * ones,
            %endfor
               ])

    ## TBD
    ##def get_Euler_split_flux_Jac_2D_TTv(self, r, Ys, u, v, a, ev, T, up, nx, ny, Q, direction):
    
    """, strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution, fixed_coeffs=True) -> str:
    """For the mechanism given by *sol*, return Python source code for
    a class conforming to a module containing a class called ``Thermochemistry``
    adhering to the :class:`~pyrometheus.thermochem_example.Thermochemistry`
    interface.
    """
    if not all((isinstance(r, ct.Reaction) for r in sol.reactions())):
        # Cantera version < 3.0
        from warnings import warn
        warn("Specific reaction types (e.g., ct.FalloffReaction) are "
             "deprecated in Cantera 3, where all in a 'ct.Solution' "
             "object are of generic 'ct.Reaction' type. "
             "Identify reactions using 'ct.Reaction.reaction_type' insteady")
        falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                       if isinstance(r, ct.FalloffReaction)]
        three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                          if isinstance(r, ct.ThreeBodyReaction)]
    else:
        # Cantera version == 3.0
        falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                       if r.reaction_type.startswith("falloff")]
        three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                          if r.reaction_type == "three-body-Arrhenius"]

    return code_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        str_np_inner=str_np_inner,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        fixed_coeffs=fixed_coeffs,
        ce=pyrometheus.chem_expr,

        falloff_reactions=falloff_rxn,
        three_body_reactions=three_body_rxn,
    )


def compile_class(code_str, class_name="Thermochemistry"):
    exec_dict = {}
    exec(compile(code_str, "<generated code>", "exec"), exec_dict)
    exec_dict["_MODULE_SOURCE_CODE"] = code_str

    return exec_dict[class_name]


def get_thermochem_class(sol: ct.Solution, fixed_coeffs=True):
    """For the mechanism given by *sol*, return a class conforming to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
    return compile_class(
        gen_thermochem_code(sol, fixed_coeffs=fixed_coeffs))


def cti_to_mech_file(cti_file_name, mech_file_name):
    """Write python file for mechanism specified by CTI file."""
    with open(mech_file_name, "w") as outf:
        code = gen_thermochem_code(ct.Solution(cti_file_name, "gas"))
        print(code, file=outf)

# vim: foldmethod=marker
