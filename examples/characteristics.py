

def characteristic_decomposition(cons_vars, prim_vars,
                                 density, temperature,
                                 pyro_gas):

    # Some additional thermodynamics
    if len(pyro_gas.molecular_weights.shape) > 1:
        pyro_gas.molecular_weights = pyro_gas.molecular_weights.ravel()
        pyro_gas.inv_molecular_weights = pyro_gas.inv_molecular_weights.ravel()

    e_species = (
        pyro_gas.inv_molecular_weights.ravel() *
        (pyro_gas.get_species_enthalpies_rt(temperature) - 1) *
        pyro_gas.gas_constant * temperature
    )
    mol_weight = pyro_gas.get_mix_molecular_weight(
        prim_vars.mass_fractions
    )
    cp_mass = pyro_gas.get_mixture_specific_heat_cp_mass(
        temperature, prim_vars.mass_fractions
    )
    cv_mass = pyro_gas.get_mixture_specific_heat_cv_mass(
        temperature, prim_vars.mass_fractions
    )
    e_mass = pyro_gas.get_mixture_internal_energy_mass(
        temperature, prim_vars.mass_fractions
    )
    gamma = cp_mass / cv_mass

    sound_speed = pyro_gas.usr_np.sqrt(
        gamma * prim_vars.pressure / density
    )
    total_enthalpy = (
        (cons_vars.total_energy / density) +
        prim_vars.pressure / density
    )

    if len(pyro_gas.molecular_weights.shape) == 1:
        pyro_gas.molecular_weights = (
            pyro_gas.molecular_weights.reshape(-1, 1)
        )
        pyro_gas.inv_molecular_weights = (
            pyro_gas.inv_molecular_weights.reshape(-1, 1)
        )

    # Pressure derivatives
    dp_de = (
        density * pyro_gas.gas_constant / (cv_mass * mol_weight)
    )
    dp_ddi = pyro_gas.gas_constant * (
        temperature / (pyro_gas.molecular_weights.ravel()) +
        (e_mass - e_species) / (cv_mass * mol_weight)
    )

    #
    sign = pyro_gas.usr_np.array([-1, 1])
    num_variables = pyro_gas.num_species + 2

    wave_speeds = pyro_gas.usr_np.array([
        prim_vars.velocity - sound_speed, prim_vars.velocity,
        prim_vars.velocity + sound_speed
    ])

    # eigenvalues: u - a, u + a
    eigenvec_r = pyro_gas.usr_np.zeros((num_variables, num_variables))
    eigenvec_r[0, [0, -1]] = prim_vars.velocity + sign * sound_speed
    eigenvec_r[1, [0, -1]] = (
        total_enthalpy + sign * prim_vars.velocity * sound_speed
    )
    eigenvec_r[2:, 0] = prim_vars.mass_fractions
    eigenvec_r[2:, -1] = prim_vars.mass_fractions

    # eigenvalues: u
    eigenvec_r[0, 1:-1] = prim_vars.velocity
    eigenvec_r[1, 1:-1] = (
        (cons_vars.total_energy / density) -
        density * (dp_ddi / dp_de)
    )
    eigenvec_r[2:, 1:-1] = pyro_gas.usr_np.eye(pyro_gas.num_species)
    return wave_speeds, eigenvec_r


def outflow_nscbc(flux_div, cons_vars, prim_vars, density, temperature,
                  normal, pyro_gas,):

    ws, r = characteristic_decomposition(
        cons_vars, prim_vars, density, temperature, pyro_gas,
    )
    df_dx = pyro_gas.usr_np.hstack((
        flux_div.momentum,
        flux_div.total_energy,
        flux_div.densities
    ))
    l_vec = pyro_gas.usr_np.linalg.solve(r, df_dx)

    # Subsonic outflow
    if normal == 1:
        l_vec[0] = 0
    elif normal == -1:
        l_vec[-1] = 0
    else:
        pass

    return r @ l_vec
