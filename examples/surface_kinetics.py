"""Catalytic methane oxidation on platinum.

Generates the gas-phase and surface classes for Cantera's ptcombust.yaml
(Deutschmann et al.), solves for the steady coverages of a Pt surface in a
fixed CH4/air mixture, compares them with Cantera, and sweeps temperature
through light-off.

The mechanism has 24 reactions: 19 interface-Arrhenius, 5 sticking, 2 with
coverage dependence.

    python surface_kinetics.py
"""

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from pyrometheus.codegen.python import PythonCodeGenerator as pyro

MECH = "ptcombust.yaml"
PHASE = "Pt_surf"

# Lean methane/air.
COMPOSITION = "CH4:0.095, O2:0.21, N2:0.79"
PRESSURE = ct.one_atm


def make_objects():
    """Generate and instantiate the gas and surface classes.

    The surface class takes the gas object, which it uses for the adjacent
    phase's enthalpies.
    """
    interface = ct.Interface(MECH, PHASE)
    gas = interface.adjacent["gas"]

    gas_pyro = pyro.get_thermochem_class(gas)()
    surface = pyro.get_surface_thermochem_class(interface)(gas_pyro)

    return interface, gas, gas_pyro, surface


def gas_concentrations(gas_pyro, temperature, mass_fractions):
    density = gas_pyro.get_density(PRESSURE, temperature, mass_fractions)
    return gas_pyro.get_concentrations(density, mass_fractions)


def total_concentrations(surface, gas_conc, coverages):
    """Activity concentrations in the interface's kinetics order.

    The phase offsets come from the generated class; the order is set by the
    mechanism.
    """
    concentrations = np.zeros(surface.num_total_species)
    surface_block = slice(surface.total_surface_offset,
                          surface.total_surface_offset + surface.num_species)
    gas_block = slice(surface.total_gas_offset,
                      surface.total_gas_offset + surface.gas.num_species)

    concentrations[surface_block] = surface.get_site_concentrations(coverages)
    concentrations[gas_block] = gas_conc

    return concentrations


def steady_coverages(surface, temperature, gas_conc, guess=None):
    """Coverages at which the surface species stop changing.

    Integrate the transient, then polish with Newton. Newton on its own
    converges to a nearby root with small negative coverages, which a
    coverage-dependent rate cannot raise to a power.

    The production rates conserve sites, so the system is singular; the
    free-site equation is replaced by the site-conservation constraint.
    """
    n_surface = surface.num_species

    # get_site_concentrations is linear in the coverages, so evaluating it at
    # one gives the constant factor site_density/size relating the two.
    per_coverage = np.asarray(
        surface.get_site_concentrations(np.ones(n_surface)), dtype=np.float64)

    def coverage_rates(coverages):
        rates = surface.get_surface_net_production_rates(
            temperature, total_concentrations(surface, gas_conc, coverages),
            coverages)
        return np.asarray(rates[:n_surface], dtype=np.float64) / per_coverage

    def residual(coverages):
        out = coverage_rates(coverages)
        out[0] = coverages.sum() - 1.0
        return out

    if guess is None:
        guess = np.full(n_surface, 1.0 / n_surface)

    # Stiff, so BDF.
    transient = solve_ivp(lambda _, coverages: coverage_rates(coverages),
                          (0.0, 1.0e-3), guess, method="BDF",
                          rtol=1.0e-8, atol=1.0e-16)

    # hybr reports failure once the residual stops improving, so check the
    # residual instead of solution.success.
    coverages = root(residual, transient.y[:, -1],
                     method="hybr", tol=1.0e-14).x

    # The rates grow by six orders of magnitude over the sweep, so judge the
    # residual against the rates of progress it is a difference of.
    rates_of_progress = np.asarray(surface.get_surface_rates_of_progress(
        temperature, total_concentrations(surface, gas_conc, coverages),
        coverages), dtype=np.float64)
    scale = np.max(np.abs(rates_of_progress)) / np.min(per_coverage)

    if np.max(np.abs(residual(coverages))) > 1.0e-9 * max(scale, 1.0):
        raise RuntimeError(
            f"steady coverages did not converge at {temperature:.0f} K")

    return coverages


def report_steady_state(interface, gas, gas_pyro, surface, temperature):
    """Solve for the coverages and compare with Cantera."""
    gas.TPX = temperature, PRESSURE, COMPOSITION
    coverages = steady_coverages(
        surface, temperature, gas_concentrations(gas_pyro, temperature, gas.Y))

    interface.TP = temperature, PRESSURE
    interface.coverages = np.full(surface.num_species, 1.0 / surface.num_species)
    interface.advance_coverages_to_steady_state()

    print(f"Steady coverages at {temperature:.0f} K")
    print(f"  {'species':<10}{'pyrometheus':>16}{'cantera':>16}{'difference':>14}")
    for name, mine, theirs in zip(surface.species_names,
                                  coverages, interface.coverages):
        print(f"  {name:<10}{mine:>16.8e}{theirs:>16.8e}"
              f"{abs(mine - theirs):>14.2e}")
    print(f"  largest difference: "
          f"{np.max(np.abs(coverages - interface.coverages)):.2e}\n")


def sweep_temperature(gas, gas_pyro, surface, temperatures):
    """Sweep temperature through light-off.

    Each temperature seeds the next, which keeps the transients short.
    """
    print("Light-off sweep")
    print(f"  {'T [K]':>7}{'PT(S)':>10}{'O(S)':>10}{'OH(S)':>12}{'CO(S)':>12}"
          f"{'CH4 [kmol/m^2/s]':>20}{'q [W/m^2]':>14}")

    methane = surface.gas.species_indices["CH4"]
    coverages = None

    for temperature in temperatures:
        gas.TPX = temperature, PRESSURE, COMPOSITION
        gas_conc = gas_concentrations(gas_pyro, temperature, gas.Y)

        coverages = steady_coverages(surface, temperature, gas_conc, coverages)
        concentrations = total_concentrations(surface, gas_conc, coverages)

        rates = surface.get_surface_net_production_rates(
            temperature, concentrations, coverages)
        heat_flux = surface.get_surface_net_heat_release_rate(
            temperature, concentrations, coverages)
        named = dict(zip(surface.species_names, coverages))

        print(f"  {temperature:>7.0f}{named['PT(S)']:>10.4f}{named['O(S)']:>10.4f}"
              f"{named['OH(S)']:>12.3e}{named['CO(S)']:>12.3e}"
              f"{rates[surface.total_gas_offset + methane]:>20.4e}"
              f"{heat_flux:>14.4e}")


def main():
    interface, gas, gas_pyro, surface = make_objects()

    print(f"{MECH}: {surface.num_species} surface species, "
          f"{surface.num_reactions} reactions, "
          f"site density {surface.site_density:.4e} kmol/m^2")
    print(f"adjacent gas: {surface.gas.num_species} species\n")

    report_steady_state(interface, gas, gas_pyro, surface, 800.0)
    sweep_temperature(gas, gas_pyro, surface, np.arange(600.0, 1501.0, 100.0))


if __name__ == "__main__":
    main()
