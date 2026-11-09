def from_equiv_ratio(equiv_ratio, pyro_gas, usr_np):
    # Some constants
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5

    # Species indices
    i_fu = pyro_gas.get_species_index('H2')
    i_ox = pyro_gas.get_species_index('O2')
    i_di = pyro_gas.get_species_index('N2')

    # Determine mole fractions for given equiv_ratio
    from numpy import zeros
    x = zeros(pyro_gas.num_species)
    x[i_fu] = (
        ox_di_ratio * equiv_ratio
    ) / (
        stoich_ratio + ox_di_ratio * equiv_ratio
    )
    x[i_ox] = stoich_ratio*x[i_fu] / equiv_ratio
    x[i_di] = (1 - ox_di_ratio) * x[i_ox] / ox_di_ratio

    # Return mass fractions
    return usr_np.array(
        pyro_gas.molecular_weights * x / usr_np.sum(
            pyro_gas.molecular_weights * x
        )
    )


def from_ignition_file(pyro_gas):
    import h5py
    db = h5py.File('output/autoignition.h5', 'r')
    t = db['explicit/time'][:]
    s = db['explicit/state'][:, :]
    db.close()

    i = 375
    y = s[i, :-1]
    temp = s[i, -1]

    print(
        f'From ignition file: Time: {t[i]:.4e}, Temperature {temp:.2f}, '
        'Mass fractions = [{:s}]'.format(', '.join([
            '{:.2e}'.format(a) for a in y
        ]))
    )

    density = pyro_gas.get_density(pyro_gas.one_atm, temp, y)
    energy = pyro_gas.get_mixture_internal_energy_mass(temp, y)
    return density, energy, pyro_gas._pyro_make_array(y)
