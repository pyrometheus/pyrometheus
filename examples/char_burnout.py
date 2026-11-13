"""Burnout of a char particle.

A global char mechanism: five irreversible gasification channels on solid
carbon, one site species, and three phases.

    surface   C-site                       total_surface_offset
    gas       CO CO2 H2 H2O O2 N2 ...      total_gas_offset
    bulk      C(gr)                        total_bulk_offset

A pure condensed species enters a rate law through its activity, which is one,
not its molar density. The mechanism sets a zero reaction order on C(gr), so
the rates are first order in the attacking gas species.

What this does:

1. Checks the generated production rates against Cantera.
2. Splits the carbon consumption by channel.
3. Burns a particle: shrinks it from 100 um and reports the burnout time.

Run with::

    python char_burnout.py
"""

import os

import cantera as ct
import numpy as np

from pyrometheus.codegen.python import PythonCodeGenerator as pyro

MECH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "mech", "carbon_surface.yaml")
PHASE = "carbon_surface"

# Hot combustion products with oxygen left, as in a furnace burnout zone.
COMPOSITION = "O2:0.15, CO2:0.05, H2O:0.03, N2:0.77"
PRESSURE = ct.one_atm

INITIAL_DIAMETER = 100.0e-6


def make_objects():
    interface = ct.Interface(MECH, PHASE)
    gas = interface.adjacent["gas"]
    bulk = interface.adjacent["graphite"]

    gas_pyro = pyro.get_thermochem_class(gas)()
    surface = pyro.get_surface_thermochem_class(interface)(gas_pyro)

    return interface, gas, bulk, gas_pyro, surface


def total_concentrations(surface, gas_conc):
    """Activity concentrations, one entry per phase block.

    The single site species is fully covered, so the surface block is the site
    density. The bulk entry is an activity of one.
    """
    coverages = np.ones(surface.num_species)

    concentrations = np.zeros(surface.num_total_species)
    concentrations[surface.total_surface_offset] = \
        surface.get_site_concentrations(coverages)[0]
    concentrations[surface.total_gas_offset:
                   surface.total_gas_offset + surface.gas.num_species] = gas_conc
    concentrations[surface.total_bulk_offset] = 1.0

    return concentrations, coverages


def gas_state(gas, gas_pyro, temperature):
    gas.TPX = temperature, PRESSURE, COMPOSITION
    density = gas_pyro.get_density(PRESSURE, temperature, gas.Y)

    return gas_pyro.get_concentrations(density, gas.Y)


def verify_against_cantera(interface, gas, gas_pyro, surface, temperatures):
    print("Production rates against Cantera")
    print(f"  {'T [K]':>7}{'C(gr) [kmol/m^2/s]':>22}{'largest rel. diff':>20}")

    for temperature in temperatures:
        gas_conc = gas_state(gas, gas_pyro, temperature)
        concentrations, coverages = total_concentrations(surface, gas_conc)

        mine = np.asarray(surface.get_surface_net_production_rates(
            temperature, concentrations, coverages), dtype=np.float64)
        interface.TP = temperature, PRESSURE
        theirs = interface.net_production_rates

        difference = np.max(np.abs(mine - theirs)) / np.max(np.abs(theirs))
        print(f"  {temperature:>7.0f}{mine[surface.total_bulk_offset]:>22.6e}"
              f"{difference:>20.2e}")
    print()


def channel_breakdown(interface, gas, gas_pyro, surface, temperatures):
    """Share of carbon consumption by oxidizer.

    Rates of progress from the generated routine; labels and carbon
    stoichiometry from Cantera.
    """
    reactions = interface.reactions()
    oxidizers = [next(s for s in r.reactants if s != "C(gr)") for r in reactions]
    per_event = np.array([r.reactants["C(gr)"] for r in reactions])

    for reaction in reactions:
        print(f"  {reaction.equation}")
    print()
    print("Share of carbon consumption by channel")
    print("  " + f"{'T [K]':>7}" + "".join(f"{o:>9}" for o in oxidizers))

    for temperature in temperatures:
        gas_conc = gas_state(gas, gas_pyro, temperature)
        concentrations, coverages = total_concentrations(surface, gas_conc)

        rates = np.asarray(surface.get_surface_rates_of_progress(
            temperature, concentrations, coverages), dtype=np.float64)
        carbon = rates * per_event
        share = 100.0 * carbon / carbon.sum()

        print(f"  {temperature:>7.0f}" + "".join(f"{s:>8.2f}%" for s in share))
    print()


def report_burnout(surface, gas, gas_pyro, bulk, temperatures):
    """Shrink a particle until it is gone.

    Quasi-steady and isothermal, with the gas composition held fixed. The
    surface recedes at

        d(D)/dt = 2 * sdot_C * W_C / rho_C

    with sdot_C negative. Nothing on the right depends on D, so the diameter
    falls linearly. No diffusion limit is applied, so this is the
    kinetics-limited burnout time, a lower bound on the real one.
    """
    print(f"Burnout of a {1.0e6 * INITIAL_DIAMETER:.0f} um particle "
          f"(kinetics-limited)")
    print(f"  {'T [K]':>7}{'regression [um/s]':>20}{'burnout time [s]':>20}")

    for temperature in temperatures:
        gas_conc = gas_state(gas, gas_pyro, temperature)
        concentrations, coverages = total_concentrations(surface, gas_conc)
        rates = surface.get_surface_net_production_rates(
            temperature, concentrations, coverages)

        # Carbon consumed per unit area, kmol/m^2/s. Negative.
        carbon = float(rates[surface.total_bulk_offset])
        regression = 2.0 * carbon * bulk.molecular_weights[0] / bulk.density

        print(f"  {temperature:>7.0f}{-1.0e6 * regression:>20.3e}"
              f"{INITIAL_DIAMETER / -regression:>20.3e}")


def main():
    interface, gas, bulk, gas_pyro, surface = make_objects()

    print(f"{os.path.basename(MECH)}: {surface.num_species} site species, "
          f"{surface.num_reactions} reactions")
    print(f"adjacent gas: {surface.gas.num_species} species; "
          f"bulk {bulk.species_names[0]} at {bulk.density:.0f} kg/m^3\n")

    temperatures = np.array([1200.0, 1400.0, 1600.0, 1800.0, 2000.0])
    verify_against_cantera(interface, gas, gas_pyro, surface, temperatures)
    channel_breakdown(interface, gas, gas_pyro, surface, temperatures)
    report_burnout(surface, gas, gas_pyro, bulk, temperatures)


if __name__ == "__main__":
    main()
