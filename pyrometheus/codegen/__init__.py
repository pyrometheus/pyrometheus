from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
import cantera as ct
from pyrometheus.bandit.general_thermochem import BaseMechanism


@dataclass(frozen=True)
class CodeGenerationOptions:
    scalar_type: Optional[str] = None
    directive_offload: Optional[str] = None
    # Opt-in: generate get_net_production_rates_jacobian by composing the
    # full chemistry graph and differentiating it symbolically. Off by
    # default -- see BaseMechanism.make_species_production_rate_jacobian
    # for the cost tradeoff.
    compute_jacobian: bool = False


class CodeGenerator:
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Returns the name (slug) of the code generator."""
        pass

    @staticmethod
    def load_mechanism(mech_path: str, phase: Optional[str] = None):
        """Returns the mechanism object this generator's generate()
        expects, read from the mechanism file at *mech_path*. Generators
        built on the Bandit interface override this to return a
        :class:`BaseMechanism` instead.
        """
        return ct.Solution(mech_path, phase)

    @staticmethod
    @abstractmethod
    def supports_overloading() -> bool:
        """Returns whether the code generator supports operator overloading."""
        pass

    @staticmethod
    @abstractmethod
    def generate(name: str,
                 bandit_mech: BaseMechanism,
                 opts: CodeGenerationOptions = None) -> str:
        """Invokes Pyrometheus to generate the thermochemistry code for this
        generator and mechanism contained in the passed Cantera Solution object.

        Parameters
        ----------
        name : str
            A module, class, or namespace name for the generated code.
        base_mech : BaseMechanism
            The object containing the mechanism to generate
            thermochemistry code for.
        opts : CodeGenerationOptions
            Options to pass to the code generator.
        """
        pass
