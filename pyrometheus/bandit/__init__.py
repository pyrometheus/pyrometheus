from pyrometheus.bandit.impl.cantera import CanteraMechanism
from pyrometheus.bandit.impl.plato import PlatoMechanism


# The options each library's mechanism takes besides its name.
library_options = {
    "cantera": {"phase"},
    "mutation": {"state_model", "thermo_database"},
    "plato": {"reaction_set", "transfer", "plato_db_path"},
}


def load_mechanism(library: str, mechanism: str, **options):
    """:returns: The ``BaseMechanism`` that *library* reads for
        *mechanism*: a mechanism file for Cantera, a mixture name for
        Mutation++ and Plato.

    :arg options: ``phase`` for Cantera; ``state_model`` and
        ``thermo_database`` for Mutation++; ``reaction_set``,
        ``transfer`` and ``plato_db_path`` for Plato.
    """
    if library not in library_options:
        raise ValueError(f"unknown library '{library}'")
    unknown = set(options) - library_options[library]
    if unknown:
        raise TypeError(
            f"{library} mechanisms take no {', '.join(sorted(unknown))}"
        )
    if library == "cantera":
        return CanteraMechanism(mechanism, **options)
    if library == "mutation":
        from pyrometheus.bandit.impl.mutation import MutationMechanism
        return MutationMechanism(mechanism, **options)
    import os
    options.setdefault("plato_db_path", os.environ.get("PLATO_DB"))
    return PlatoMechanism(mixture=mechanism, **options)
