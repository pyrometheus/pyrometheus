from numbers import Number

import pymbolic.primitives as p
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
import cantera as ct
import numpy as np  # noqa: F401

from mako.template import Template
from pyrometheus.bandit.general_thermochem import BaseMechanism

from itertools import product

from . import CodeGenerator, CodeGenerationOptions


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
        return (f"self.pyro_np.{self.OP_NAMES[expr.operator]}"
                f"({self.rec(expr.left, PREC_NONE, *args, **kwargs)}, "
                f"{self.rec(expr.right, PREC_NONE, *args, **kwargs)})")

    def map_if(self, expr, enclosing_prec, *args, **kwargs):
        return "self.pyro_np.where(%s, %s, %s)" % (
            self.rec(expr.condition, PREC_NONE, *args, **kwargs),
            self.rec(expr.then, PREC_NONE, *args, **kwargs),
            self.rec(expr.else_, PREC_NONE, *args, **kwargs),
        )

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "self.pyro_np.%s(%s)",
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

# {{{ Template

code_tpl = Template(
    """
import numpy as np


class Thermochemistry:
    
    def __init__(self, pyro_np):
        self.pyro_np = pyro_np
        self.num_temperatures = ${bandit_mech.num_temp}
        self.num_species = ${bandit_mech.num_species}
        self.num_reactions = ${bandit_mech.num_reactions}
        self.one_atm = ${bandit_mech.namespace.one_atm}
        self.gas_constant = ${bandit_mech.namespace.gas_constant}
        self.molecular_weights = ${str_np(bandit_mech.molecular_weights)}
        self.inv_molecular_weights = 1/self.molecular_weights

    def _pyro_zeros_like(self, argument):
        return 0 * argument

    def _pyro_ones_like(self, argument):
        return 0 * argument + 1.0

    def _pyro_make_array(self, res_list):
        raise NotImplementedError

    def get_concentrations(self, density, mass_fractions):
        return self._pyro_make_array([
            %for i in range(bandit_mech.num_species):
            density * mass_fractions[${i}] / self.molecular_weights[${i}],
            %endfor
        ])

    def get_species_gibbs_rt(self, temperature):
        return self._pyro_make_array([
            %for sp_thermo in bandit_mech.species_thermo_polynomials:
            ${cgm(sp_thermo.gibbs_poly.expr)},
            %endfor
        ])

    def get_equilibrium_constants(self, temperature):
        gibbs_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
            %for i, expr in enumerate(bandit_mech.equil_constants):
            %if bandit_mech.is_reversible(i):
            ${cgm(expr)},
            %else:
            -0.17364695002734*temperature,
            %endif
            %endfor
        ])

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_ones_like(temperature)
        return self._pyro_make_array([
            %for rate_coeff in bandit_mech.rate_coeffs:
            ${cgm(rate_coeff.expr)},
            %endfor
        ])

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
            %for expr in bandit_mech.mass_action_rates:
            ${cgm(expr)},
            %endfor
        ])

    def get_net_production_rates(self, density, temperature, mass_fractions):
        concentrations = self.get_concentrations(density, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, concentrations)        
        ones = self._pyro_ones_like(r_net[0])
        return self._pyro_make_array([
            %for expr in bandit_mech.species_prod_rates:
            ${cgm(expr)} * ones,
            %endfor
        ])
""", strict_undefined=True)

# }}}

# {{{

class PythonBanditCodeGenerator(CodeGenerator):
    @staticmethod
    def get_name() -> str:
        return "python"

    @staticmethod
    def supports_overloading() -> bool:
        return True

    @staticmethod
    def generate(name: str,
                 bandit_mech: BaseMechanism,
                 opts: CodeGenerationOptions = None) -> str:
        if opts is None:
            opts = CodeGenerationOptions()

        if opts.directive_offload is not None:
            raise TypeError(
                "OpenMP/ACC directive based offloading is not supported for "
                "Python code generation"
            )

        return code_tpl.render(
            bandit_mech=bandit_mech,
            str_np=str_np,
            cgm=CodeGenerationMapper(),
            Variable=p.Variable,
        )

    @staticmethod
    def compile_class(name: str, source: str):
        """Compile the generated Python source code into a class definition"""
        exec_dict = {}
        exec(compile(source, "<generated code>", "exec"), exec_dict)
        exec_dict["_MODULE_SOURCE_CODE"] = source

        return exec_dict[name]

    @staticmethod
    def get_thermochem_class(bandit_mech: BaseMechanism):
        """For the mechanism given by *bandit_mech*, return an instance of
        Pyrometheus' generated Python class for thermochemistry.
        """
        name = "Thermochemistry"
        return PythonBanditCodeGenerator.compile_class(
            name=name,
            source=PythonBanditCodeGenerator.generate(
                name,
                bandit_mech
            )
        )

# }}}
