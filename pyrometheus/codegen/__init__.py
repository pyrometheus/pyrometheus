from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from pyrometheus.bandit.general_thermochem import BaseMechanism


@dataclass(frozen=True)
class CodeGenerationOptions:
    scalar_type: Optional[str] = None
    directive_offload: Optional[str] = None


class CodeGenerator:
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Returns the name (slug) of the code generator."""
        pass

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
