
#ifndef PYROMETHEUS_CALLER_INDEXING
#define PYROMETHEUS_CALLER_INDEXING 0
#endif


#define GPU_ROUTINE(name) ! name


module libpyro_fortran_air5

    implicit none

    integer, parameter :: sp = selected_real_kind(6,37) ! Single precision
    integer, parameter :: dp = selected_real_kind(15,307) ! Double precision

    integer, parameter :: num_species = 5
    integer, parameter :: num_reactions = 8
    integer, parameter :: num_temperatures = 2
    real(dp), parameter :: one_atm = &
        101325.0d0
    real(dp), parameter :: gas_constant = &
        8314.462d0
    real(dp), parameter :: molecular_weights(5) = &
        (/ 14.00674d0, 15.9994d0, 28.01348d0, 30.006140000000002d0, 31.9988d0 /)
    real(dp), parameter :: inv_molecular_weights(5) = &
        (/ 0.07139420022075087d0, 0.06250234383789392d0, 0.03569710011037543d0,&
            0.03332651250710688d0, 0.03125117191894696d0 /)

    character(len=12), parameter :: species_names(5) = &
        (/ "N           " , "O           " , "N2          " , "NO          " , &
            "O2          " /)

contains

    subroutine get_species_name(sp_index, sp_name)

        integer, intent(in) :: sp_index
        character(len=*), intent(out) :: sp_name

        sp_name = species_names(sp_index + PYROMETHEUS_CALLER_INDEXING)

    end subroutine get_species_name

    subroutine get_species_index(sp_name, sp_index)

        character(len=*), intent(in) :: sp_name
        integer, intent(out) :: sp_index

        integer :: idx

        sp_index = 0
        loop:do idx = 1, num_species
            if(trim(adjustl(sp_name)) .eq. trim(species_names(idx))) then
                sp_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_species_index

    subroutine get_specific_gas_constant(mass_fractions, specific_gas_constant)

        GPU_ROUTINE(get_specific_gas_constant)

        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: specific_gas_constant

        specific_gas_constant = gas_constant * ( &
                    + inv_molecular_weights(1)*mass_fractions(1) &
                    + inv_molecular_weights(2)*mass_fractions(2) &
                    + inv_molecular_weights(3)*mass_fractions(3) &
                    + inv_molecular_weights(4)*mass_fractions(4) &
                    + inv_molecular_weights(5)*mass_fractions(5) &
                )

    end subroutine get_specific_gas_constant

    subroutine get_density(pressure, temperature, mass_fractions, density)

        GPU_ROUTINE(get_density)

        real(dp), intent(in) :: pressure
        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: density

        real(dp) :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        density = pressure * mix_mol_weight / (gas_constant * temperature)

    end subroutine get_density

    subroutine get_pressure(density, temperature, mass_fractions, pressure)

        GPU_ROUTINE(get_pressure)

        real(dp), intent(in) :: density
        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: pressure

        real(dp) :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        pressure = density * gas_constant * temperature / mix_mol_weight

    end subroutine get_pressure

    subroutine get_mixture_molecular_weight(mass_fractions, mix_mol_weight)

        GPU_ROUTINE(get_mixture_molecular_weight)

        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: mix_mol_weight

        mix_mol_weight = 1.0d0 / ( &
                    + inv_molecular_weights(1)*mass_fractions(1) &
                    + inv_molecular_weights(2)*mass_fractions(2) &
                    + inv_molecular_weights(3)*mass_fractions(3) &
                    + inv_molecular_weights(4)*mass_fractions(4) &
                    + inv_molecular_weights(5)*mass_fractions(5) &
                )

    end subroutine get_mixture_molecular_weight

    subroutine get_concentrations(density, mass_fractions, concentrations)

        GPU_ROUTINE(get_concentrations)

        real(dp), intent(in) :: density
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out), dimension(5) :: &
            concentrations

            concentrations(1) = density * &
                inv_molecular_weights(1) * mass_fractions(1)
            concentrations(2) = density * &
                inv_molecular_weights(2) * mass_fractions(2)
            concentrations(3) = density * &
                inv_molecular_weights(3) * mass_fractions(3)
            concentrations(4) = density * &
                inv_molecular_weights(4) * mass_fractions(4)
            concentrations(5) = density * &
                inv_molecular_weights(5) * mass_fractions(5)

    end subroutine get_concentrations

    subroutine get_mole_fractions(mix_mol_weight, mass_fractions,              &
        mole_fractions)

        GPU_ROUTINE(get_mole_fractions)

        real(dp), intent(in) :: mix_mol_weight
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out), dimension(5) :: &
            mole_fractions

            mole_fractions(1) = inv_molecular_weights(1) * &
                mass_fractions(1) * mix_mol_weight
            mole_fractions(2) = inv_molecular_weights(2) * &
                mass_fractions(2) * mix_mol_weight
            mole_fractions(3) = inv_molecular_weights(3) * &
                mass_fractions(3) * mix_mol_weight
            mole_fractions(4) = inv_molecular_weights(4) * &
                mass_fractions(4) * mix_mol_weight
            mole_fractions(5) = inv_molecular_weights(5) * &
                mass_fractions(5) * mix_mol_weight

    end subroutine get_mole_fractions

    subroutine get_mass_averaged_property(&
        & mass_fractions, spec_property, mix_property)

        GPU_ROUTINE(get_mass_averaged_property)

        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(in), dimension(5) :: &
            spec_property
        real(dp), intent(out) :: mix_property

        mix_property = ( &
                + inv_molecular_weights(1)*mass_fractions(1) &
                *spec_property(1) &
                + inv_molecular_weights(2)*mass_fractions(2) &
                *spec_property(2) &
                + inv_molecular_weights(3)*mass_fractions(3) &
                *spec_property(3) &
                + inv_molecular_weights(4)*mass_fractions(4) &
                *spec_property(4) &
                + inv_molecular_weights(5)*mass_fractions(5) &
                *spec_property(5) &
        )

    end subroutine get_mass_averaged_property

    subroutine get_mixture_specific_heat_cp_mass(temperature, mass_fractions,  &
        cp_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cp_mass)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: cp_mix

        real(dp), dimension(5) :: cp0_r

        call get_species_specific_heats_cp_r(temperature, cp0_r)
        call get_mass_averaged_property(mass_fractions, cp0_r, cp_mix)
        cp_mix = cp_mix * gas_constant

    end subroutine get_mixture_specific_heat_cp_mass

    subroutine get_mixture_specific_heat_cv_mass(temperature, mass_fractions,  &
        cv_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cv_mass)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: cv_mix

        real(dp), dimension(5) :: cv0_r

        call get_species_specific_heats_cv_r(temperature, cv0_r)
        call get_mass_averaged_property(mass_fractions, cv0_r, cv_mix)
        cv_mix = cv_mix * gas_constant

    end subroutine get_mixture_specific_heat_cv_mass

    subroutine get_mixture_enthalpy_mass(temperature, mass_fractions, h_mix)

        GPU_ROUTINE(get_mixture_enthalpy_mass)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: h_mix

        real(dp), dimension(5) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_mass_averaged_property(mass_fractions, h0_rt, h_mix)
        h_mix = h_mix * gas_constant * temperature(1)

    end subroutine get_mixture_enthalpy_mass

    subroutine get_mixture_energy_mass(temperature, mass_fractions, e_mix)

        GPU_ROUTINE(get_mixture_energy_mass)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out) :: e_mix

        real(dp), dimension(5) :: e0_rt

        call get_species_internal_energies_rt(temperature, e0_rt)
        call get_mass_averaged_property(mass_fractions, e0_rt, e_mix)
        e_mix = e_mix * gas_constant * temperature(1)

    end subroutine get_mixture_energy_mass

    subroutine get_species_specific_heats_cp_r(temperature, cp0_r)

        GPU_ROUTINE(get_species_specific_heats_cp_r)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: cp0_r

        cp0_r(1) = merge(547518105.0d0 / temperature(1)**2d0 + (-310757.498d0) &
            / temperature(1) + 69.1678274d0 + (-0.00684798813d0) *             &
            temperature(1) + 3.8275724d-07 * temperature(1)**2d0 +             &
            (-1.098367709d-11) * temperature(1)**3d0 + 1.277986024d-16 *       &
            temperature(1)**4d0, merge(88765.0138d0 / temperature(1)**2d0 +    &
            (-107.12315d0) / temperature(1) + 2.362188287d0 + 0.0002916720081d0&
            * temperature(1) + (-1.7295151d-07) * temperature(1)**2d0 +        &
            4.01265788d-11 * temperature(1)**3d0 + (-2.677227571d-15) *        &
            temperature(1)**4d0, 0.0d0 / temperature(1)**2d0 + 0.0d0 /         &
            temperature(1) + 2.5d0 + 0.0d0 * temperature(1) + 0.0d0 *          &
            temperature(1)**2d0 + 0.0d0 * temperature(1)**3d0 + 0.0d0 *        &
            temperature(1)**4d0, temperature(1) > 1000.0d0), temperature(1) >  &
            6000.0d0)
        cp0_r(2) = merge(177900426.4d0 / temperature(1)**2d0 + (-108232.8257d0)&
            / temperature(1) + 28.10778365d0 + (-0.002975232262d0) *           &
            temperature(1) + 1.854997534d-07 * temperature(1)**2d0 +           &
            (-5.79623154d-12) * temperature(1)**3d0 + 7.191720164d-17 *        &
            temperature(1)**4d0, merge(261902.0262d0 / temperature(1)**2d0 +   &
            (-729.872203d0) / temperature(1) + 3.31717727d0 +                  &
            (-0.000428133436d0) * temperature(1) + 1.036104594d-07 *           &
            temperature(1)**2d0 + (-9.43830433d-12) * temperature(1)**3d0 +    &
            2.725038297d-16 * temperature(1)**4d0, (-7953.6113d0) /            &
            temperature(1)**2d0 + 160.7177787d0 / temperature(1) +             &
            1.966226438d0 + 0.00101367031d0 * temperature(1) +                 &
            (-1.110415423d-06) * temperature(1)**2d0 + 6.5175075d-10 *         &
            temperature(1)**3d0 + (-1.584779251d-13) * temperature(1)**4d0,    &
            temperature(1) > 1000.0d0), temperature(1) > 6000.0d0)
        cp0_r(3) = merge(831013916.0d0 / temperature(1)**2d0 + (-642073.354d0) &
            / temperature(1) + 202.0264635d0 + (-0.03065092046d0) *            &
            temperature(1) + 2.486903333d-06 * temperature(1)**2d0 +           &
            (-9.70595411d-11) * temperature(1)**3d0 + 1.437538881d-15 *        &
            temperature(1)**4d0, merge(587712.406d0 / temperature(1)**2d0 +    &
            (-2239.249073d0) / temperature(1) + 6.06694922d0 +                 &
            (-0.00061396855d0) * temperature(1) + 1.491806679d-07 *            &
            temperature(1)**2d0 + (-1.923105485d-11) * temperature(1)**3d0 +   &
            1.061954386d-15 * temperature(1)**4d0, 22103.71497d0 /             &
            temperature(1)**2d0 + (-381.846182d0) / temperature(1) +           &
            6.08273836d0 + (-0.00853091441d0) * temperature(1) +               &
            1.384646189d-05 * temperature(1)**2d0 + (-9.62579362d-09) *        &
            temperature(1)**3d0 + 2.519705809d-12 * temperature(1)**4d0,       &
            temperature(1) > 1000.0d0), temperature(1) > 6000.0d0)
        cp0_r(4) = merge((-957530354.0d0) / temperature(1)**2d0 + 591243.448d0 &
            / temperature(1) + (-138.4566826d0) + 0.01694339403d0 *            &
            temperature(1) + (-1.007351096d-06) * temperature(1)**2d0 +        &
            2.912584076d-11 * temperature(1)**3d0 + (-3.29510935d-16) *        &
            temperature(1)**4d0, merge(223901.8716d0 / temperature(1)**2d0 +   &
            (-1289.651623d0) / temperature(1) + 5.43393603d0 +                 &
            (-0.00036560349d0) * temperature(1) + 9.88096645d-08 *             &
            temperature(1)**2d0 + (-1.416076856d-11) * temperature(1)**3d0 +   &
            9.38018462d-16 * temperature(1)**4d0, (-11439.16503d0) /           &
            temperature(1)**2d0 + 153.6467592d0 / temperature(1) + 3.43146873d0&
            + (-0.002668592368d0) * temperature(1) + 8.48139912d-06 *          &
            temperature(1)**2d0 + (-7.68511105d-09) * temperature(1)**3d0 +    &
            2.386797655d-12 * temperature(1)**4d0, temperature(1) > 1000.0d0), &
            temperature(1) > 6000.0d0)
        cp0_r(5) = merge(497529430.0d0 / temperature(1)**2d0 + (-286610.6874d0)&
            / temperature(1) + 66.9035225d0 + (-0.00616995902d0) *             &
            temperature(1) + 3.016396027d-07 * temperature(1)**2d0 +           &
            (-7.4214166d-12) * temperature(1)**3d0 + 7.27817577d-17 *          &
            temperature(1)**4d0, merge((-1037939.022d0) / temperature(1)**2d0 +&
            2344.830282d0 / temperature(1) + 1.819732036d0 + 0.001267847582d0 *&
            temperature(1) + (-2.188067988d-07) * temperature(1)**2d0 +        &
            2.053719572d-11 * temperature(1)**3d0 + (-8.19346705d-16) *        &
            temperature(1)**4d0, (-34255.6342d0) / temperature(1)**2d0 +       &
            484.700097d0 / temperature(1) + 1.119010961d0 + 0.00429388924d0 *  &
            temperature(1) + (-6.83630052d-07) * temperature(1)**2d0 +         &
            (-2.0233727d-09) * temperature(1)**3d0 + 1.039040018d-12 *         &
            temperature(1)**4d0, temperature(1) > 1000.0d0), temperature(1) >  &
            6000.0d0)

    end subroutine get_species_specific_heats_cp_r

    subroutine get_species_specific_heats_cv_r(temperature, cv0_r)

        GPU_ROUTINE(get_species_specific_heats_cv_r)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: cv0_r

        call get_species_specific_heats_cp_r(temperature, cv0_r)
            cv0_r(1) = cv0_r(1) - 1.d0
            cv0_r(2) = cv0_r(2) - 1.d0
            cv0_r(3) = cv0_r(3) - 1.d0
            cv0_r(4) = cv0_r(4) - 1.d0
            cv0_r(5) = cv0_r(5) - 1.d0

    end subroutine get_species_specific_heats_cv_r

    subroutine get_species_enthalpies_rt(temperature, h0_rt)

        GPU_ROUTINE(get_species_enthalpies_rt)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: h0_rt

        h0_rt(1) = merge((-547518105.0d0) / temperature(1)**2d0 +              &
            ((-310757.498d0) * log(temperature(1))) / temperature(1) +         &
            69.1678274d0 + (-0.003423994065d0) * temperature(1) +              &
            1.2758574666666667d-07 * temperature(1)**2d0 + (-2.7459192725d-12) &
            * temperature(1)**3d0 + 2.555972048d-17 * temperature(1)**4d0 +    &
            2550585.618d0 / temperature(1), merge((-88765.0138d0) /            &
            temperature(1)**2d0 + ((-107.12315d0) * log(temperature(1))) /     &
            temperature(1) + 2.362188287d0 + 0.00014583600405d0 *              &
            temperature(1) + (-5.7650503333333334d-08) * temperature(1)**2d0 + &
            1.00316447d-11 * temperature(1)**3d0 + (-5.354455142d-16) *        &
            temperature(1)**4d0 + 56973.5133d0 / temperature(1), -0.0d0 /      &
            temperature(1)**2d0 + (-0.0d0 * log(temperature(1))) /             &
            temperature(1) + 2.5d0 + -0.0d0 * temperature(1) + -0.0d0 *        &
            temperature(1)**2d0 + -0.0d0 * temperature(1)**3d0 + -0.0d0 *      &
            temperature(1)**4d0 + 56104.6378d0 / temperature(1), temperature(1)&
            > 1000.0d0), temperature(1) > 6000.0d0)
        h0_rt(2) = merge((-177900426.4d0) / temperature(1)**2d0 +              &
            ((-108232.8257d0) * log(temperature(1))) / temperature(1) +        &
            28.10778365d0 + (-0.001487616131d0) * temperature(1) +             &
            6.183325113333333d-08 * temperature(1)**2d0 + (-1.449057885d-12) * &
            temperature(1)**3d0 + 1.4383440328d-17 * temperature(1)**4d0 +     &
            889094.263d0 / temperature(1), merge((-261902.0262d0) /            &
            temperature(1)**2d0 + ((-729.872203d0) * log(temperature(1))) /    &
            temperature(1) + 3.31717727d0 + (-0.000214066718d0) *              &
            temperature(1) + 3.45368198d-08 * temperature(1)**2d0 +            &
            (-2.3595760825d-12) * temperature(1)**3d0 + 5.4500765939999995d-17 &
            * temperature(1)**4d0 + 33924.2806d0 / temperature(1), 7953.6113d0 &
            / temperature(1)**2d0 + (160.7177787d0 * log(temperature(1))) /    &
            temperature(1) + 1.966226438d0 + 0.000506835155d0 * temperature(1) &
            + (-3.7013847433333333d-07) * temperature(1)**2d0 + 1.629376875d-10&
            * temperature(1)**3d0 + (-3.169558502d-14) * temperature(1)**4d0 + &
            28403.62437d0 / temperature(1), temperature(1) > 1000.0d0),        &
            temperature(1) > 6000.0d0)
        h0_rt(3) = merge((-831013916.0d0) / temperature(1)**2d0 +              &
            ((-642073.354d0) * log(temperature(1))) / temperature(1) +         &
            202.0264635d0 + (-0.01532546023d0) * temperature(1) +              &
            8.289677776666666d-07 * temperature(1)**2d0 + (-2.4264885275d-11) *&
            temperature(1)**3d0 + 2.875077762d-16 * temperature(1)**4d0 +      &
            4938707.04d0 / temperature(1), merge((-587712.406d0) /             &
            temperature(1)**2d0 + ((-2239.249073d0) * log(temperature(1))) /   &
            temperature(1) + 6.06694922d0 + (-0.000306984275d0) *              &
            temperature(1) + 4.9726889300000005d-08 * temperature(1)**2d0 +    &
            (-4.8077637125d-12) * temperature(1)**3d0 + 2.123908772d-16 *      &
            temperature(1)**4d0 + 12832.10415d0 / temperature(1),              &
            (-22103.71497d0) / temperature(1)**2d0 + ((-381.846182d0) *        &
            log(temperature(1))) / temperature(1) + 6.08273836d0 +             &
            (-0.004265457205d0) * temperature(1) + 4.615487296666666d-06 *     &
            temperature(1)**2d0 + (-2.406448405d-09) * temperature(1)**3d0 +   &
            5.039411618000001d-13 * temperature(1)**4d0 + 710.846086d0 /       &
            temperature(1), temperature(1) > 1000.0d0), temperature(1) >       &
            6000.0d0)
        h0_rt(4) = merge(957530354.0d0 / temperature(1)**2d0 + (591243.448d0 * &
            log(temperature(1))) / temperature(1) + (-138.4566826d0) +         &
            0.008471697015d0 * temperature(1) + (-3.357836986666666d-07) *     &
            temperature(1)**2d0 + 7.28146019d-12 * temperature(1)**3d0 +       &
            (-6.5902187d-17) * temperature(1)**4d0 + (-4677501.24d0) /         &
            temperature(1), merge((-223901.8716d0) / temperature(1)**2d0 +     &
            ((-1289.651623d0) * log(temperature(1))) / temperature(1) +        &
            5.43393603d0 + (-0.000182801745d0) * temperature(1) +              &
            3.293655483333333d-08 * temperature(1)**2d0 + (-3.54019214d-12) *  &
            temperature(1)**3d0 + 1.8760369239999998d-16 * temperature(1)**4d0 &
            + 17503.17656d0 / temperature(1), 11439.16503d0 /                  &
            temperature(1)**2d0 + (153.6467592d0 * log(temperature(1))) /      &
            temperature(1) + 3.43146873d0 + (-0.001334296184d0) *              &
            temperature(1) + 2.82713304d-06 * temperature(1)**2d0 +            &
            (-1.9212777625d-09) * temperature(1)**3d0 + 4.77359531d-13 *       &
            temperature(1)**4d0 + 9098.21441d0 / temperature(1), temperature(1)&
            > 1000.0d0), temperature(1) > 6000.0d0)
        h0_rt(5) = merge((-497529430.0d0) / temperature(1)**2d0 +              &
            ((-286610.6874d0) * log(temperature(1))) / temperature(1) +        &
            66.9035225d0 + (-0.00308497951d0) * temperature(1) +               &
            1.0054653423333334d-07 * temperature(1)**2d0 + (-1.85535415d-12) * &
            temperature(1)**3d0 + 1.455635154d-17 * temperature(1)**4d0 +      &
            2293554.027d0 / temperature(1), merge(1037939.022d0 /              &
            temperature(1)**2d0 + (2344.830282d0 * log(temperature(1))) /      &
            temperature(1) + 1.819732036d0 + 0.000633923791d0 * temperature(1) &
            + (-7.29355996d-08) * temperature(1)**2d0 + 5.13429893d-12 *       &
            temperature(1)**3d0 + (-1.63869341d-16) * temperature(1)**4d0 +    &
            (-16890.10929d0) / temperature(1), 34255.6342d0 /                  &
            temperature(1)**2d0 + (484.700097d0 * log(temperature(1))) /       &
            temperature(1) + 1.119010961d0 + 0.00214694462d0 * temperature(1) +&
            (-2.27876684d-07) * temperature(1)**2d0 + (-5.05843175d-10) *      &
            temperature(1)**3d0 + 2.078080036d-13 * temperature(1)**4d0 +      &
            (-3391.45487d0) / temperature(1), temperature(1) > 1000.0d0),      &
            temperature(1) > 6000.0d0)

    end subroutine get_species_enthalpies_rt

    subroutine get_species_internal_energies_rt(temperature, e0_rt)

        GPU_ROUTINE(get_species_internal_energies_rt)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: e0_rt

        call get_species_enthalpies_rt(temperature, e0_rt)
            e0_rt(1) = e0_rt(1) - 1.d0
            e0_rt(2) = e0_rt(2) - 1.d0
            e0_rt(3) = e0_rt(3) - 1.d0
            e0_rt(4) = e0_rt(4) - 1.d0
            e0_rt(5) = e0_rt(5) - 1.d0

    end subroutine get_species_internal_energies_rt

    subroutine get_species_entropies_r(temperature, s0_r)

        GPU_ROUTINE(get_species_entropies_r)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: s0_r

        s0_r(1) = merge((-547518105.0d0) / (2d0 * temperature(1)**2d0) + (-1d0)&
            * (-310757.498d0) / temperature(1) + 69.1678274d0 *                &
            log(temperature(1)) + (-0.00684798813d0) * temperature(1) +        &
            1.9137862d-07 * temperature(1)**2d0 + (-3.6612256966666666d-12) *  &
            temperature(1)**3d0 + 3.19496506d-17 * temperature(1)**4d0 +       &
            (-584.876971d0), merge((-88765.0138d0) / (2d0 *                    &
            temperature(1)**2d0) + (-1d0) * (-107.12315d0) / temperature(1) +  &
            2.362188287d0 * log(temperature(1)) + 0.0002916720081d0 *          &
            temperature(1) + (-8.6475755d-08) * temperature(1)**2d0 +          &
            1.3375526266666666d-11 * temperature(1)**3d0 + (-6.6930689275d-16) &
            * temperature(1)**4d0 + 4.86523579d0, -0.0d0 / (2d0 *              &
            temperature(1)**2d0) + (-1d0) * -0.0d0 / temperature(1) + 2.5d0 *  &
            log(temperature(1)) + -0.0d0 * temperature(1) + -0.0d0 *           &
            temperature(1)**2d0 + -0.0d0 * temperature(1)**3d0 + -0.0d0 *      &
            temperature(1)**4d0 + 4.19390932d0, temperature(1) > 1000.0d0),    &
            temperature(1) > 6000.0d0)
        s0_r(2) = merge((-177900426.4d0) / (2d0 * temperature(1)**2d0) + (-1d0)&
            * (-108232.8257d0) / temperature(1) + 28.10778365d0 *              &
            log(temperature(1)) + (-0.002975232262d0) * temperature(1) +       &
            9.27498767d-08 * temperature(1)**2d0 + (-1.93207718d-12) *         &
            temperature(1)**3d0 + 1.797930041d-17 * temperature(1)**4d0 +      &
            (-218.1728151d0), merge((-261902.0262d0) / (2d0 *                  &
            temperature(1)**2d0) + (-1d0) * (-729.872203d0) / temperature(1) + &
            3.31717727d0 * log(temperature(1)) + (-0.000428133436d0) *         &
            temperature(1) + 5.18052297d-08 * temperature(1)**2d0 +            &
            (-3.146101443333333d-12) * temperature(1)**3d0 + 6.8125957425d-17 *&
            temperature(1)**4d0 + (-0.667958535d0), 7953.6113d0 / (2d0 *       &
            temperature(1)**2d0) + (-1d0) * 160.7177787d0 / temperature(1) +   &
            1.966226438d0 * log(temperature(1)) + 0.00101367031d0 *            &
            temperature(1) + (-5.552077115d-07) * temperature(1)**2d0 +        &
            2.1725025d-10 * temperature(1)**3d0 + (-3.9619481275d-14) *        &
            temperature(1)**4d0 + 8.40424182d0, temperature(1) > 1000.0d0),    &
            temperature(1) > 6000.0d0)
        s0_r(3) = merge((-831013916.0d0) / (2d0 * temperature(1)**2d0) + (-1d0)&
            * (-642073.354d0) / temperature(1) + 202.0264635d0 *               &
            log(temperature(1)) + (-0.03065092046d0) * temperature(1) +        &
            1.2434516665d-06 * temperature(1)**2d0 + (-3.235318036666667d-11) *&
            temperature(1)**3d0 + 3.5938472025d-16 * temperature(1)**4d0 +     &
            (-1672.099736d0), merge((-587712.406d0) / (2d0 *                   &
            temperature(1)**2d0) + (-1d0) * (-2239.249073d0) / temperature(1) +&
            6.06694922d0 * log(temperature(1)) + (-0.00061396855d0) *          &
            temperature(1) + 7.459033395d-08 * temperature(1)**2d0 +           &
            (-6.410351616666667d-12) * temperature(1)**3d0 + 2.654885965d-16 * &
            temperature(1)**4d0 + (-15.86639599d0), (-22103.71497d0) / (2d0 *  &
            temperature(1)**2d0) + (-1d0) * (-381.846182d0) / temperature(1) + &
            6.08273836d0 * log(temperature(1)) + (-0.00853091441d0) *          &
            temperature(1) + 6.923230945d-06 * temperature(1)**2d0 +           &
            (-3.2085978733333332d-09) * temperature(1)**3d0 + 6.2992645225d-13 &
            * temperature(1)**4d0 + (-10.76003316d0), temperature(1) >         &
            1000.0d0), temperature(1) > 6000.0d0)
        s0_r(4) = merge(957530354.0d0 / (2d0 * temperature(1)**2d0) + (-1d0) * &
            591243.448d0 / temperature(1) + (-138.4566826d0) *                 &
            log(temperature(1)) + 0.01694339403d0 * temperature(1) +           &
            (-5.03675548d-07) * temperature(1)**2d0 + 9.708613586666667d-12 *  &
            temperature(1)**3d0 + (-8.237773375d-17) * temperature(1)**4d0 +   &
            1242.081218d0, merge((-223901.8716d0) / (2d0 * temperature(1)**2d0)&
            + (-1d0) * (-1289.651623d0) / temperature(1) + 5.43393603d0 *      &
            log(temperature(1)) + (-0.00036560349d0) * temperature(1) +        &
            4.940483225d-08 * temperature(1)**2d0 + (-4.720256186666667d-12) * &
            temperature(1)**3d0 + 2.345046155d-16 * temperature(1)**4d0 +      &
            (-8.50166709d0), 11439.16503d0 / (2d0 * temperature(1)**2d0) +     &
            (-1d0) * 153.6467592d0 / temperature(1) + 3.43146873d0 *           &
            log(temperature(1)) + (-0.002668592368d0) * temperature(1) +       &
            4.24069956d-06 * temperature(1)**2d0 + (-2.5617036833333334d-09) * &
            temperature(1)**3d0 + 5.9669941375d-13 * temperature(1)**4d0 +     &
            6.72872749d0, temperature(1) > 1000.0d0), temperature(1) > 6000.0d0)
        s0_r(5) = merge((-497529430.0d0) / (2d0 * temperature(1)**2d0) + (-1d0)&
            * (-286610.6874d0) / temperature(1) + 66.9035225d0 *               &
            log(temperature(1)) + (-0.00616995902d0) * temperature(1) +        &
            1.5081980135d-07 * temperature(1)**2d0 + (-2.4738055333333335d-12) &
            * temperature(1)**3d0 + 1.8195439425d-17 * temperature(1)**4d0 +   &
            (-553.062161d0), merge(1037939.022d0 / (2d0 * temperature(1)**2d0) &
            + (-1d0) * 2344.830282d0 / temperature(1) + 1.819732036d0 *        &
            log(temperature(1)) + 0.001267847582d0 * temperature(1) +          &
            (-1.094033994d-07) * temperature(1)**2d0 + 6.845731906666667d-12 * &
            temperature(1)**3d0 + (-2.0483667625d-16) * temperature(1)**4d0 +  &
            17.38716506d0, 34255.6342d0 / (2d0 * temperature(1)**2d0) + (-1d0) &
            * 484.700097d0 / temperature(1) + 1.119010961d0 *                  &
            log(temperature(1)) + 0.00429388924d0 * temperature(1) +           &
            (-3.41815026d-07) * temperature(1)**2d0 + (-6.744575666666666d-10) &
            * temperature(1)**3d0 + 2.597600045d-13 * temperature(1)**4d0 +    &
            18.4969947d0, temperature(1) > 1000.0d0), temperature(1) > 6000.0d0)

    end subroutine get_species_entropies_r

    subroutine get_species_gibbs_rt(temperature, g0_rt)

        GPU_ROUTINE(get_species_gibbs_rt)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: g0_rt

        g0_rt(1) = merge((-547518105.0d0) / temperature(1)**2d0 +              &
            ((-310757.498d0) * log(temperature(1))) / temperature(1) +         &
            69.1678274d0 + (-0.003423994065d0) * temperature(1) +              &
            1.2758574666666667d-07 * temperature(1)**2d0 + (-2.7459192725d-12) &
            * temperature(1)**3d0 + 2.555972048d-17 * temperature(1)**4d0 +    &
            2550585.618d0 / temperature(1) + (-1d0) * ((-547518105.0d0) / (2d0 &
            * temperature(1)**2d0) + (-1d0) * (-310757.498d0) / temperature(1) &
            + 69.1678274d0 * log(temperature(1)) + (-0.00684798813d0) *        &
            temperature(1) + 1.9137862d-07 * temperature(1)**2d0 +             &
            (-3.6612256966666666d-12) * temperature(1)**3d0 + 3.19496506d-17 * &
            temperature(1)**4d0 + (-584.876971d0)), merge((-88765.0138d0) /    &
            temperature(1)**2d0 + ((-107.12315d0) * log(temperature(1))) /     &
            temperature(1) + 2.362188287d0 + 0.00014583600405d0 *              &
            temperature(1) + (-5.7650503333333334d-08) * temperature(1)**2d0 + &
            1.00316447d-11 * temperature(1)**3d0 + (-5.354455142d-16) *        &
            temperature(1)**4d0 + 56973.5133d0 / temperature(1) + (-1d0) *     &
            ((-88765.0138d0) / (2d0 * temperature(1)**2d0) + (-1d0) *          &
            (-107.12315d0) / temperature(1) + 2.362188287d0 *                  &
            log(temperature(1)) + 0.0002916720081d0 * temperature(1) +         &
            (-8.6475755d-08) * temperature(1)**2d0 + 1.3375526266666666d-11 *  &
            temperature(1)**3d0 + (-6.6930689275d-16) * temperature(1)**4d0 +  &
            4.86523579d0), -0.0d0 / temperature(1)**2d0 + (-0.0d0 *            &
            log(temperature(1))) / temperature(1) + 2.5d0 + -0.0d0 *           &
            temperature(1) + -0.0d0 * temperature(1)**2d0 + -0.0d0 *           &
            temperature(1)**3d0 + -0.0d0 * temperature(1)**4d0 + 56104.6378d0 /&
            temperature(1) + (-1d0) * (-0.0d0 / (2d0 * temperature(1)**2d0) +  &
            (-1d0) * -0.0d0 / temperature(1) + 2.5d0 * log(temperature(1)) +   &
            -0.0d0 * temperature(1) + -0.0d0 * temperature(1)**2d0 + -0.0d0 *  &
            temperature(1)**3d0 + -0.0d0 * temperature(1)**4d0 + 4.19390932d0),&
            temperature(1) > 1000.0d0), temperature(1) > 6000.0d0)
        g0_rt(2) = merge((-177900426.4d0) / temperature(1)**2d0 +              &
            ((-108232.8257d0) * log(temperature(1))) / temperature(1) +        &
            28.10778365d0 + (-0.001487616131d0) * temperature(1) +             &
            6.183325113333333d-08 * temperature(1)**2d0 + (-1.449057885d-12) * &
            temperature(1)**3d0 + 1.4383440328d-17 * temperature(1)**4d0 +     &
            889094.263d0 / temperature(1) + (-1d0) * ((-177900426.4d0) / (2d0 *&
            temperature(1)**2d0) + (-1d0) * (-108232.8257d0) / temperature(1) +&
            28.10778365d0 * log(temperature(1)) + (-0.002975232262d0) *        &
            temperature(1) + 9.27498767d-08 * temperature(1)**2d0 +            &
            (-1.93207718d-12) * temperature(1)**3d0 + 1.797930041d-17 *        &
            temperature(1)**4d0 + (-218.1728151d0)), merge((-261902.0262d0) /  &
            temperature(1)**2d0 + ((-729.872203d0) * log(temperature(1))) /    &
            temperature(1) + 3.31717727d0 + (-0.000214066718d0) *              &
            temperature(1) + 3.45368198d-08 * temperature(1)**2d0 +            &
            (-2.3595760825d-12) * temperature(1)**3d0 + 5.4500765939999995d-17 &
            * temperature(1)**4d0 + 33924.2806d0 / temperature(1) + (-1d0) *   &
            ((-261902.0262d0) / (2d0 * temperature(1)**2d0) + (-1d0) *         &
            (-729.872203d0) / temperature(1) + 3.31717727d0 *                  &
            log(temperature(1)) + (-0.000428133436d0) * temperature(1) +       &
            5.18052297d-08 * temperature(1)**2d0 + (-3.146101443333333d-12) *  &
            temperature(1)**3d0 + 6.8125957425d-17 * temperature(1)**4d0 +     &
            (-0.667958535d0)), 7953.6113d0 / temperature(1)**2d0 +             &
            (160.7177787d0 * log(temperature(1))) / temperature(1) +           &
            1.966226438d0 + 0.000506835155d0 * temperature(1) +                &
            (-3.7013847433333333d-07) * temperature(1)**2d0 + 1.629376875d-10 *&
            temperature(1)**3d0 + (-3.169558502d-14) * temperature(1)**4d0 +   &
            28403.62437d0 / temperature(1) + (-1d0) * (7953.6113d0 / (2d0 *    &
            temperature(1)**2d0) + (-1d0) * 160.7177787d0 / temperature(1) +   &
            1.966226438d0 * log(temperature(1)) + 0.00101367031d0 *            &
            temperature(1) + (-5.552077115d-07) * temperature(1)**2d0 +        &
            2.1725025d-10 * temperature(1)**3d0 + (-3.9619481275d-14) *        &
            temperature(1)**4d0 + 8.40424182d0), temperature(1) > 1000.0d0),   &
            temperature(1) > 6000.0d0)
        g0_rt(3) = merge((-831013916.0d0) / temperature(1)**2d0 +              &
            ((-642073.354d0) * log(temperature(1))) / temperature(1) +         &
            202.0264635d0 + (-0.01532546023d0) * temperature(1) +              &
            8.289677776666666d-07 * temperature(1)**2d0 + (-2.4264885275d-11) *&
            temperature(1)**3d0 + 2.875077762d-16 * temperature(1)**4d0 +      &
            4938707.04d0 / temperature(1) + (-1d0) * ((-831013916.0d0) / (2d0 *&
            temperature(1)**2d0) + (-1d0) * (-642073.354d0) / temperature(1) + &
            202.0264635d0 * log(temperature(1)) + (-0.03065092046d0) *         &
            temperature(1) + 1.2434516665d-06 * temperature(1)**2d0 +          &
            (-3.235318036666667d-11) * temperature(1)**3d0 + 3.5938472025d-16 *&
            temperature(1)**4d0 + (-1672.099736d0)), merge((-587712.406d0) /   &
            temperature(1)**2d0 + ((-2239.249073d0) * log(temperature(1))) /   &
            temperature(1) + 6.06694922d0 + (-0.000306984275d0) *              &
            temperature(1) + 4.9726889300000005d-08 * temperature(1)**2d0 +    &
            (-4.8077637125d-12) * temperature(1)**3d0 + 2.123908772d-16 *      &
            temperature(1)**4d0 + 12832.10415d0 / temperature(1) + (-1d0) *    &
            ((-587712.406d0) / (2d0 * temperature(1)**2d0) + (-1d0) *          &
            (-2239.249073d0) / temperature(1) + 6.06694922d0 *                 &
            log(temperature(1)) + (-0.00061396855d0) * temperature(1) +        &
            7.459033395d-08 * temperature(1)**2d0 + (-6.410351616666667d-12) * &
            temperature(1)**3d0 + 2.654885965d-16 * temperature(1)**4d0 +      &
            (-15.86639599d0)), (-22103.71497d0) / temperature(1)**2d0 +        &
            ((-381.846182d0) * log(temperature(1))) / temperature(1) +         &
            6.08273836d0 + (-0.004265457205d0) * temperature(1) +              &
            4.615487296666666d-06 * temperature(1)**2d0 + (-2.406448405d-09) * &
            temperature(1)**3d0 + 5.039411618000001d-13 * temperature(1)**4d0 +&
            710.846086d0 / temperature(1) + (-1d0) * ((-22103.71497d0) / (2d0 *&
            temperature(1)**2d0) + (-1d0) * (-381.846182d0) / temperature(1) + &
            6.08273836d0 * log(temperature(1)) + (-0.00853091441d0) *          &
            temperature(1) + 6.923230945d-06 * temperature(1)**2d0 +           &
            (-3.2085978733333332d-09) * temperature(1)**3d0 + 6.2992645225d-13 &
            * temperature(1)**4d0 + (-10.76003316d0)), temperature(1) >        &
            1000.0d0), temperature(1) > 6000.0d0)
        g0_rt(4) = merge(957530354.0d0 / temperature(1)**2d0 + (591243.448d0 * &
            log(temperature(1))) / temperature(1) + (-138.4566826d0) +         &
            0.008471697015d0 * temperature(1) + (-3.357836986666666d-07) *     &
            temperature(1)**2d0 + 7.28146019d-12 * temperature(1)**3d0 +       &
            (-6.5902187d-17) * temperature(1)**4d0 + (-4677501.24d0) /         &
            temperature(1) + (-1d0) * (957530354.0d0 / (2d0 *                  &
            temperature(1)**2d0) + (-1d0) * 591243.448d0 / temperature(1) +    &
            (-138.4566826d0) * log(temperature(1)) + 0.01694339403d0 *         &
            temperature(1) + (-5.03675548d-07) * temperature(1)**2d0 +         &
            9.708613586666667d-12 * temperature(1)**3d0 + (-8.237773375d-17) * &
            temperature(1)**4d0 + 1242.081218d0), merge((-223901.8716d0) /     &
            temperature(1)**2d0 + ((-1289.651623d0) * log(temperature(1))) /   &
            temperature(1) + 5.43393603d0 + (-0.000182801745d0) *              &
            temperature(1) + 3.293655483333333d-08 * temperature(1)**2d0 +     &
            (-3.54019214d-12) * temperature(1)**3d0 + 1.8760369239999998d-16 * &
            temperature(1)**4d0 + 17503.17656d0 / temperature(1) + (-1d0) *    &
            ((-223901.8716d0) / (2d0 * temperature(1)**2d0) + (-1d0) *         &
            (-1289.651623d0) / temperature(1) + 5.43393603d0 *                 &
            log(temperature(1)) + (-0.00036560349d0) * temperature(1) +        &
            4.940483225d-08 * temperature(1)**2d0 + (-4.720256186666667d-12) * &
            temperature(1)**3d0 + 2.345046155d-16 * temperature(1)**4d0 +      &
            (-8.50166709d0)), 11439.16503d0 / temperature(1)**2d0 +            &
            (153.6467592d0 * log(temperature(1))) / temperature(1) +           &
            3.43146873d0 + (-0.001334296184d0) * temperature(1) +              &
            2.82713304d-06 * temperature(1)**2d0 + (-1.9212777625d-09) *       &
            temperature(1)**3d0 + 4.77359531d-13 * temperature(1)**4d0 +       &
            9098.21441d0 / temperature(1) + (-1d0) * (11439.16503d0 / (2d0 *   &
            temperature(1)**2d0) + (-1d0) * 153.6467592d0 / temperature(1) +   &
            3.43146873d0 * log(temperature(1)) + (-0.002668592368d0) *         &
            temperature(1) + 4.24069956d-06 * temperature(1)**2d0 +            &
            (-2.5617036833333334d-09) * temperature(1)**3d0 + 5.9669941375d-13 &
            * temperature(1)**4d0 + 6.72872749d0), temperature(1) > 1000.0d0), &
            temperature(1) > 6000.0d0)
        g0_rt(5) = merge((-497529430.0d0) / temperature(1)**2d0 +              &
            ((-286610.6874d0) * log(temperature(1))) / temperature(1) +        &
            66.9035225d0 + (-0.00308497951d0) * temperature(1) +               &
            1.0054653423333334d-07 * temperature(1)**2d0 + (-1.85535415d-12) * &
            temperature(1)**3d0 + 1.455635154d-17 * temperature(1)**4d0 +      &
            2293554.027d0 / temperature(1) + (-1d0) * ((-497529430.0d0) / (2d0 &
            * temperature(1)**2d0) + (-1d0) * (-286610.6874d0) / temperature(1)&
            + 66.9035225d0 * log(temperature(1)) + (-0.00616995902d0) *        &
            temperature(1) + 1.5081980135d-07 * temperature(1)**2d0 +          &
            (-2.4738055333333335d-12) * temperature(1)**3d0 + 1.8195439425d-17 &
            * temperature(1)**4d0 + (-553.062161d0)), merge(1037939.022d0 /    &
            temperature(1)**2d0 + (2344.830282d0 * log(temperature(1))) /      &
            temperature(1) + 1.819732036d0 + 0.000633923791d0 * temperature(1) &
            + (-7.29355996d-08) * temperature(1)**2d0 + 5.13429893d-12 *       &
            temperature(1)**3d0 + (-1.63869341d-16) * temperature(1)**4d0 +    &
            (-16890.10929d0) / temperature(1) + (-1d0) * (1037939.022d0 / (2d0 &
            * temperature(1)**2d0) + (-1d0) * 2344.830282d0 / temperature(1) + &
            1.819732036d0 * log(temperature(1)) + 0.001267847582d0 *           &
            temperature(1) + (-1.094033994d-07) * temperature(1)**2d0 +        &
            6.845731906666667d-12 * temperature(1)**3d0 + (-2.0483667625d-16) *&
            temperature(1)**4d0 + 17.38716506d0), 34255.6342d0 /               &
            temperature(1)**2d0 + (484.700097d0 * log(temperature(1))) /       &
            temperature(1) + 1.119010961d0 + 0.00214694462d0 * temperature(1) +&
            (-2.27876684d-07) * temperature(1)**2d0 + (-5.05843175d-10) *      &
            temperature(1)**3d0 + 2.078080036d-13 * temperature(1)**4d0 +      &
            (-3391.45487d0) / temperature(1) + (-1d0) * (34255.6342d0 / (2d0 * &
            temperature(1)**2d0) + (-1d0) * 484.700097d0 / temperature(1) +    &
            1.119010961d0 * log(temperature(1)) + 0.00429388924d0 *            &
            temperature(1) + (-3.41815026d-07) * temperature(1)**2d0 +         &
            (-6.744575666666666d-10) * temperature(1)**3d0 + 2.597600045d-13 * &
            temperature(1)**4d0 + 18.4969947d0), temperature(1) > 1000.0d0),   &
            temperature(1) > 6000.0d0)

    end subroutine get_species_gibbs_rt

    subroutine get_equilibrium_constants(temperature, k_eq)

        GPU_ROUTINE(get_equilibrium_constants)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(8) :: k_eq

        real(dp), dimension(5) :: gibbs_rt

        call get_species_gibbs_rt(temperature, gibbs_rt)

        k_eq(1) = 0d0 + 1d0 * gibbs_rt(1) + 1d0 * gibbs_rt(1) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(3)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(2) = 0d0 + 1d0 * gibbs_rt(1) + 1d0 * gibbs_rt(1) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(3)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(3) = 0d0 + 1d0 * gibbs_rt(2) + 1d0 * gibbs_rt(2) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(5)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(4) = 0d0 + 1d0 * gibbs_rt(2) + 1d0 * gibbs_rt(2) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(5)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(5) = 0d0 + 1d0 * gibbs_rt(1) + 1d0 * gibbs_rt(2) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(4)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(6) = 0d0 + 1d0 * gibbs_rt(1) + 1d0 * gibbs_rt(2) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(4)) + (-1d0) * 1d0 * log(12.027236398458493d0 /     &
            temperature(1))
        k_eq(7) = 0d0 + 1d0 * gibbs_rt(5) + 1d0 * gibbs_rt(1) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(4) + 1d0 * gibbs_rt(2))
        k_eq(8) = 0d0 + 1d0 * gibbs_rt(4) + 1d0 * gibbs_rt(1) + (-1d0) * (0d0 +&
            1d0 * gibbs_rt(3) + 1d0 * gibbs_rt(2))

    end subroutine get_equilibrium_constants

    subroutine get_fwd_rate_coefficients(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_fwd_rate_coefficients)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            concentrations
        real(dp), intent(out), dimension(8) :: k_fwd

        k_fwd(1) = exp(44.84772905555498d0 + (-1.6d0) * log(sqrt(temperature(1)&
            * temperature(2))) + (-1d0) * 113200.0d0 / sqrt(temperature(1) *   &
            temperature(2)))
        k_fwd(2) = exp(43.392441822948136d0 + (-1.6d0) *                       &
            log(sqrt(temperature(1) * temperature(2))) + (-1d0) * 113200.0d0 / &
            sqrt(temperature(1) * temperature(2)))
        k_fwd(3) = exp(43.74911676688687d0 + (-1.5d0) * log(sqrt(temperature(1)&
            * temperature(2))) + (-1d0) * 59500.0d0 / sqrt(temperature(1) *    &
            temperature(2)))
        k_fwd(4) = exp(42.13967885445277d0 + (-1.5d0) * log(sqrt(temperature(1)&
            * temperature(2))) + (-1d0) * 59500.0d0 / sqrt(temperature(1) *    &
            temperature(2)))
        k_fwd(5) = exp(32.33150148172096d0 + 0.0d0 * log(sqrt(temperature(1) * &
            temperature(2))) + (-1d0) * 75500.0d0 / sqrt(temperature(1) *      &
            temperature(2)))
        k_fwd(6) = exp(29.240459028362647d0 + 0.0d0 * log(sqrt(temperature(1) *&
            temperature(2))) + (-1d0) * 75500.0d0 / sqrt(temperature(1) *      &
            temperature(2)))
        k_fwd(7) = exp(22.85149754279568d0 + 0.0d0 * log(temperature(1)) +     &
            (-1d0) * 19450.0d0 / temperature(1))
        k_fwd(8) = exp(34.092489292282266d0 + (-1.0d0) * log(temperature(1)) + &
            (-1d0) * 38400.0d0 / temperature(1))

    end subroutine get_fwd_rate_coefficients

    subroutine get_net_rates_of_progress(temperature, concentrations, r_net)

        GPU_ROUTINE(get_net_rates_of_progress)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            concentrations
        real(dp), intent(out), dimension(8) :: r_net

        real(dp), dimension(8) :: k_fwd
        real(dp), dimension(8) :: log_k_eq

        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        r_net(1) = (0d0 + concentrations(1) + concentrations(2)) * (k_fwd(1) * &
            concentrations(3) + (-1d0) * exp(log_k_eq(1)) * k_fwd(1) *         &
            concentrations(1) * concentrations(1))
        r_net(2) = (0d0 + concentrations(3) + concentrations(4) +              &
            concentrations(5)) * (k_fwd(2) * concentrations(3) + (-1d0) *      &
            exp(log_k_eq(2)) * k_fwd(2) * concentrations(1) * concentrations(1))
        r_net(3) = (0d0 + concentrations(1) + concentrations(2)) * (k_fwd(3) * &
            concentrations(5) + (-1d0) * exp(log_k_eq(3)) * k_fwd(3) *         &
            concentrations(2) * concentrations(2))
        r_net(4) = (0d0 + concentrations(3) + concentrations(4) +              &
            concentrations(5)) * (k_fwd(4) * concentrations(5) + (-1d0) *      &
            exp(log_k_eq(4)) * k_fwd(4) * concentrations(2) * concentrations(2))
        r_net(5) = (0d0 + concentrations(1) + concentrations(2) +              &
            concentrations(4)) * (k_fwd(5) * concentrations(4) + (-1d0) *      &
            exp(log_k_eq(5)) * k_fwd(5) * concentrations(1) * concentrations(2))
        r_net(6) = (0d0 + concentrations(3) + concentrations(5)) * (k_fwd(6) * &
            concentrations(4) + (-1d0) * exp(log_k_eq(6)) * k_fwd(6) *         &
            concentrations(1) * concentrations(2))
        r_net(7) = k_fwd(7) * (concentrations(4)**1d0 * concentrations(2)**1d0 &
            + (-1d0) * exp(log_k_eq(7)) * concentrations(5)**1d0 *             &
            concentrations(1)**1d0)
        r_net(8) = k_fwd(8) * (concentrations(3)**1d0 * concentrations(2)**1d0 &
            + (-1d0) * exp(log_k_eq(8)) * concentrations(4)**1d0 *             &
            concentrations(1)**1d0)

    end subroutine get_net_rates_of_progress

    subroutine get_net_production_rates(density, temperature, mass_fractions,  &
        omega)

        GPU_ROUTINE(get_net_production_rates)

        real(dp), intent(in) :: density
        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(out), dimension(5) :: omega

        real(dp), dimension(5) :: concentrations
        real(dp), dimension(8) :: r_net

        call get_concentrations(density, mass_fractions, concentrations)
        call get_net_rates_of_progress(temperature, concentrations, r_net)

        omega(1) = (0d0 + 2d0 * r_net(1) + 2d0 * r_net(2) + 1d0 * r_net(5) +   &
            1d0 * r_net(6) + 1d0 * r_net(7) + 1d0 * r_net(8) + 0d0) * (0d0 *   &
            r_net(1) + 1d0)
        omega(2) = (0d0 + 2d0 * r_net(3) + 2d0 * r_net(4) + 1d0 * r_net(5) +   &
            1d0 * r_net(6) + (-1d0) * (0d0 + 1d0 * r_net(7) + 1d0 * r_net(8))) &
            * (0d0 * r_net(1) + 1d0)
        omega(3) = (0d0 + (-1d0) * (0d0 + 1d0 * r_net(1) + 1d0 * r_net(2) + 1d0&
            * r_net(8))) * (0d0 * r_net(1) + 1d0)
        omega(4) = (0d0 + 1d0 * r_net(8) + (-1d0) * (0d0 + 1d0 * r_net(5) + 1d0&
            * r_net(6) + 1d0 * r_net(7))) * (0d0 * r_net(1) + 1d0)
        omega(5) = (0d0 + 1d0 * r_net(7) + (-1d0) * (0d0 + 1d0 * r_net(3) + 1d0&
            * r_net(4))) * (0d0 * r_net(1) + 1d0)

    end subroutine get_net_production_rates

    subroutine get_species_vibrational_energies(temperature, e_v)

        GPU_ROUTINE(get_species_vibrational_energies)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: e_v

        e_v(1) = 0.0d0
        e_v(2) = 0.0d0
        e_v(3) = 1007185.1406598167d0 / (exp(3393.4559799745266d0 /            &
            temperature(2)) + (-1d0))
        e_v(4) = 759128.3438235039d0 / (exp(2739.625409646012d0 /              &
            temperature(2)) + (-1d0))
        e_v(5) = 590738.4637314674d0 / (exp(2273.4991095335427d0 /             &
            temperature(2)) + (-1d0))

    end subroutine get_species_vibrational_energies

    subroutine get_species_vibrational_specific_heats(temperature, cv_v)

        GPU_ROUTINE(get_species_vibrational_specific_heats)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), dimension(5) :: cv_v

        cv_v(1) = 0.0d0
        cv_v(2) = 0.0d0
        cv_v(3) = (296.80218237791235d0 * exp(3393.4559799745266d0 /           &
            temperature(2)) * (3393.4559799745266d0 / temperature(2))**2d0) /  &
            (exp(3393.4559799745266d0 / temperature(2)) + (-1d0))**2d0
        cv_v(4) = (277.0920218328648d0 * exp(2739.625409646012d0 /             &
            temperature(2)) * (2739.625409646012d0 / temperature(2))**2d0) /   &
            (exp(2739.625409646012d0 / temperature(2)) + (-1d0))**2d0
        cv_v(5) = (259.83668137555156d0 * exp(2273.4991095335427d0 /           &
            temperature(2)) * (2273.4991095335427d0 / temperature(2))**2d0) /  &
            (exp(2273.4991095335427d0 / temperature(2)) + (-1d0))**2d0

    end subroutine get_species_vibrational_specific_heats

    subroutine get_pressure_relaxation_times(temperature, ptau_vt)

        GPU_ROUTINE(get_pressure_relaxation_times)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), &
            dimension(15) :: ptau_vt

        ptau_vt(1) = exp(180.0d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.0262d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09663243071902243d0
        ptau_vt(2) = exp(72.4d0 * (temperature(1)**(-0.3333333333333333d0) +   &
            (-0.015d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10091262087401684d0
        ptau_vt(3) = exp(221.0d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.029d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.11835007393322575d0
        ptau_vt(4) = exp(225.0d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.0293d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12036525963850686d0
        ptau_vt(5) = exp(229.0d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.0295d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12221663270785389d0
        ptau_vt(6) = exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +   &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09772005558182907d0
        ptau_vt(7) = exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +   &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10215316983308932d0
        ptau_vt(8) = exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +   &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12036525963850686d0
        ptau_vt(9) = exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +   &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12248701972045854d0
        ptau_vt(10) = exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12443964566488024d0
        ptau_vt(11) = exp(72.4d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.015d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09870299097693017d0
        ptau_vt(12) = exp(47.7d0 * (temperature(1)**(-0.3333333333333333d0) +  &
            (-0.059d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10327761938903639d0
        ptau_vt(13) = exp(134.0d0 * (temperature(1)**(-0.3333333333333333d0) + &
            (-0.0295d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12221663270785389d0
        ptau_vt(14) = exp(136.0d0 * (temperature(1)**(-0.3333333333333333d0) + &
            (-0.0298d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12443964566488024d0
        ptau_vt(15) = exp(138.0d0 * (temperature(1)**(-0.3333333333333333d0) + &
            (-0.03d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *   &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12648873467625488d0

    end subroutine get_pressure_relaxation_times

    subroutine get_vt_energy_transfer_source(&
        & density, mass_fractions, temperature, omega_vt)

        GPU_ROUTINE(get_vt_energy_transfer_source)

        real(dp), intent(in) :: density
        real(dp), intent(in), dimension(5) :: &
            mass_fractions
        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out) :: omega_vt

        omega_vt = 0.d0
        omega_vt = omega_vt + density * mass_fractions(3) *                    &
            (1007185.1406598167d0 / (exp(3393.4559799745266d0 / temperature(1))&
            + (-1d0)) + (-1d0) * 1007185.1406598167d0 /                        &
            (exp(3393.4559799745266d0 / temperature(2)) + (-1d0))) * (pressure &
            * (0d0 + (mass_fractions(1) / 14.00674d0) / (exp(180.0d0 *         &
            (temperature(1)**(-0.3333333333333333d0) + (-0.0262d0)) +          &
            (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *                &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09663243071902243d0) + (mass_fractions(2) / 15.9994d0) /         &
            (exp(72.4d0 * (temperature(1)**(-0.3333333333333333d0) +           &
            (-0.015d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10091262087401684d0) + (mass_fractions(3) / 28.01348d0) /        &
            (exp(221.0d0 * (temperature(1)**(-0.3333333333333333d0) +          &
            (-0.029d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.11835007393322575d0) + (mass_fractions(4) / 30.006140000000002d0)&
            / (exp(225.0d0 * (temperature(1)**(-0.3333333333333333d0) +        &
            (-0.0293d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12036525963850686d0) + (mass_fractions(5) / 31.9988d0) /         &
            (exp(229.0d0 * (temperature(1)**(-0.3333333333333333d0) +          &
            (-0.0295d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12221663270785389d0))) / (0d0 + mass_fractions(1) / 14.00674d0 + &
            mass_fractions(2) / 15.9994d0 + mass_fractions(3) / 28.01348d0 +   &
            mass_fractions(4) / 30.006140000000002d0 + mass_fractions(5) /     &
            31.9988d0)
        omega_vt = omega_vt + density * mass_fractions(4) *                    &
            (759128.3438235039d0 / (exp(2739.625409646012d0 / temperature(1)) +&
            (-1d0)) + (-1d0) * 759128.3438235039d0 / (exp(2739.625409646012d0 /&
            temperature(2)) + (-1d0))) * (pressure * (0d0 + (mass_fractions(1) &
            / 14.00674d0) / (exp(49.5d0 *                                      &
            (temperature(1)**(-0.3333333333333333d0) + (-0.042d0)) +           &
            (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *                &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09772005558182907d0) + (mass_fractions(2) / 15.9994d0) /         &
            (exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +           &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10215316983308932d0) + (mass_fractions(3) / 28.01348d0) /        &
            (exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +           &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12036525963850686d0) + (mass_fractions(4) / 30.006140000000002d0)&
            / (exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +         &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12248701972045854d0) + (mass_fractions(5) / 31.9988d0) /         &
            (exp(49.5d0 * (temperature(1)**(-0.3333333333333333d0) +           &
            (-0.042d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12443964566488024d0))) / (0d0 + mass_fractions(1) / 14.00674d0 + &
            mass_fractions(2) / 15.9994d0 + mass_fractions(3) / 28.01348d0 +   &
            mass_fractions(4) / 30.006140000000002d0 + mass_fractions(5) /     &
            31.9988d0)
        omega_vt = omega_vt + density * mass_fractions(5) *                    &
            (590738.4637314674d0 / (exp(2273.4991095335427d0 / temperature(1)) &
            + (-1d0)) + (-1d0) * 590738.4637314674d0 /                         &
            (exp(2273.4991095335427d0 / temperature(2)) + (-1d0))) * (pressure &
            * (0d0 + (mass_fractions(1) / 14.00674d0) / (exp(72.4d0 *          &
            (temperature(1)**(-0.3333333333333333d0) + (-0.015d0)) +           &
            (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *                &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.09870299097693017d0) + (mass_fractions(2) / 15.9994d0) /         &
            (exp(47.7d0 * (temperature(1)**(-0.3333333333333333d0) +           &
            (-0.059d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *  &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.10327761938903639d0) + (mass_fractions(3) / 28.01348d0) /        &
            (exp(134.0d0 * (temperature(1)**(-0.3333333333333333d0) +          &
            (-0.0295d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12221663270785389d0) + (mass_fractions(4) / 30.006140000000002d0)&
            / (exp(136.0d0 * (temperature(1)**(-0.3333333333333333d0) +        &
            (-0.0298d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 * &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12443964566488024d0) + (mass_fractions(5) / 31.9988d0) /         &
            (exp(138.0d0 * (temperature(1)**(-0.3333333333333333d0) +          &
            (-0.03d0)) + (-18.42d0)) * 101325.0d0 + (3.000517285412319d-24 *   &
            temperature(1)**0.5d0 * merge((temperature(1) / 50000.0d0)**2d0,   &
            0.16d0, temperature(1) <= 20000.0d0)) / 3d-21 *                    &
            0.12648873467625488d0))) / (0d0 + mass_fractions(1) / 14.00674d0 + &
            mass_fractions(2) / 15.9994d0 + mass_fractions(3) / 28.01348d0 +   &
            mass_fractions(4) / 30.006140000000002d0 + mass_fractions(5) /     &
            31.9988d0)

    end subroutine get_vt_energy_transfer_source

    subroutine get_translational_rotational_energy(temperature, tr_rot_energy)

        GPU_ROUTINE(get_translational_rotational_energy)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), &
            dimension(5) :: tr_rot_energy

        tr_rot_energy(1) = 890.406547133737d0 * (temperature(1) + (-298.15d0)) &
            + 33569432.59315255d0
        tr_rot_energy(2) = 779.5100441266547d0 * (temperature(1) + (-298.15d0))&
            + 15418991.166404353d0
        tr_rot_energy(3) = 742.0054559447808d0 * (temperature(1) + (-298.15d0))&
            + (-88491.57045770453d0)
        tr_rot_energy(4) = 692.730054582162d0 * (temperature(1) + (-298.15d0)) &
            + 2959121.9081894886d0
        tr_rot_energy(5) = 649.5917034388789d0 * (temperature(1) + (-298.15d0))&
            + (-77470.30695236099d0)

    end subroutine get_translational_rotational_energy

    subroutine get_nasa_polynomial_vibrational_energy(temperature, vibe_energy)

        GPU_ROUTINE(get_nasa_polynomial_vibrational_energy)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), &
            dimension(5) :: vibe_energy

        vibe_energy(1) = merge((-547518105.0d0) / temperature(2)**2d0 +        &
            ((-310757.498d0) * log(temperature(2))) / temperature(2) +         &
            69.1678274d0 + (-0.003423994065d0) * temperature(2) +              &
            1.2758574666666667d-07 * temperature(2)**2d0 + (-2.7459192725d-12) &
            * temperature(2)**3d0 + 2.555972048d-17 * temperature(2)**4d0 +    &
            2550585.618d0 / temperature(2), merge((-88765.0138d0) /            &
            temperature(2)**2d0 + ((-107.12315d0) * log(temperature(2))) /     &
            temperature(2) + 2.362188287d0 + 0.00014583600405d0 *              &
            temperature(2) + (-5.7650503333333334d-08) * temperature(2)**2d0 + &
            1.00316447d-11 * temperature(2)**3d0 + (-5.354455142d-16) *        &
            temperature(2)**4d0 + 56973.5133d0 / temperature(2), -0.0d0 /      &
            temperature(2)**2d0 + (-0.0d0 * log(temperature(2))) /             &
            temperature(2) + 2.5d0 + -0.0d0 * temperature(2) + -0.0d0 *        &
            temperature(2)**2d0 + -0.0d0 * temperature(2)**3d0 + -0.0d0 *      &
            temperature(2)**4d0 + 56104.6378d0 / temperature(2), temperature(2)&
            > 1000.0d0), temperature(2) > 6000.0d0) * 593.6043647558247d0 *    &
            temperature(2) + (-1d0) * 1484.0109118895616d0 * (temperature(2) + &
            (-298.15d0)) + (-33746415.7345045d0)
        vibe_energy(2) = merge((-177900426.4d0) / temperature(2)**2d0 +        &
            ((-108232.8257d0) * log(temperature(2))) / temperature(2) +        &
            28.10778365d0 + (-0.001487616131d0) * temperature(2) +             &
            6.183325113333333d-08 * temperature(2)**2d0 + (-1.449057885d-12) * &
            temperature(2)**3d0 + 1.4383440328d-17 * temperature(2)**4d0 +     &
            889094.263d0 / temperature(2), merge((-261902.0262d0) /            &
            temperature(2)**2d0 + ((-729.872203d0) * log(temperature(2))) /    &
            temperature(2) + 3.31717727d0 + (-0.000214066718d0) *              &
            temperature(2) + 3.45368198d-08 * temperature(2)**2d0 +            &
            (-2.3595760825d-12) * temperature(2)**3d0 + 5.4500765939999995d-17 &
            * temperature(2)**4d0 + 33924.2806d0 / temperature(2), 7953.6113d0 &
            / temperature(2)**2d0 + (160.7177787d0 * log(temperature(2))) /    &
            temperature(2) + 1.966226438d0 + 0.000506835155d0 * temperature(2) &
            + (-3.7013847433333333d-07) * temperature(2)**2d0 + 1.629376875d-10&
            * temperature(2)**3d0 + (-3.169558502d-14) * temperature(2)**4d0 + &
            28403.62437d0 / temperature(2), temperature(2) > 1000.0d0),        &
            temperature(2) > 6000.0d0) * 519.6733627511031d0 * temperature(2) +&
            (-1d0) * 1299.183406877758d0 * (temperature(2) + (-298.15d0)) +    &
            (-15573931.779508594d0)
        vibe_energy(3) = merge((-831013916.0d0) / temperature(2)**2d0 +        &
            ((-642073.354d0) * log(temperature(2))) / temperature(2) +         &
            202.0264635d0 + (-0.01532546023d0) * temperature(2) +              &
            8.289677776666666d-07 * temperature(2)**2d0 + (-2.4264885275d-11) *&
            temperature(2)**3d0 + 2.875077762d-16 * temperature(2)**4d0 +      &
            4938707.04d0 / temperature(2), merge((-587712.406d0) /             &
            temperature(2)**2d0 + ((-2239.249073d0) * log(temperature(2))) /   &
            temperature(2) + 6.06694922d0 + (-0.000306984275d0) *              &
            temperature(2) + 4.9726889300000005d-08 * temperature(2)**2d0 +    &
            (-4.8077637125d-12) * temperature(2)**3d0 + 2.123908772d-16 *      &
            temperature(2)**4d0 + 12832.10415d0 / temperature(2),              &
            (-22103.71497d0) / temperature(2)**2d0 + ((-381.846182d0) *        &
            log(temperature(2))) / temperature(2) + 6.08273836d0 +             &
            (-0.004265457205d0) * temperature(2) + 4.615487296666666d-06 *     &
            temperature(2)**2d0 + (-2.406448405d-09) * temperature(2)**3d0 +   &
            5.039411618000001d-13 * temperature(2)**4d0 + 710.846086d0 /       &
            temperature(2), temperature(2) > 1000.0d0), temperature(2) >       &
            6000.0d0) * 296.80218237791235d0 * temperature(2) + (-1d0) *       &
            1038.807638322693d0 * (temperature(2) + (-298.15d0)) +             &
            (-0.00021827002996040878d0)
        vibe_energy(4) = merge(957530354.0d0 / temperature(2)**2d0 +           &
            (591243.448d0 * log(temperature(2))) / temperature(2) +            &
            (-138.4566826d0) + 0.008471697015d0 * temperature(2) +             &
            (-3.357836986666666d-07) * temperature(2)**2d0 + 7.28146019d-12 *  &
            temperature(2)**3d0 + (-6.5902187d-17) * temperature(2)**4d0 +     &
            (-4677501.24d0) / temperature(2), merge((-223901.8716d0) /         &
            temperature(2)**2d0 + ((-1289.651623d0) * log(temperature(2))) /   &
            temperature(2) + 5.43393603d0 + (-0.000182801745d0) *              &
            temperature(2) + 3.293655483333333d-08 * temperature(2)**2d0 +     &
            (-3.54019214d-12) * temperature(2)**3d0 + 1.8760369239999998d-16 * &
            temperature(2)**4d0 + 17503.17656d0 / temperature(2), 11439.16503d0&
            / temperature(2)**2d0 + (153.6467592d0 * log(temperature(2))) /    &
            temperature(2) + 3.43146873d0 + (-0.001334296184d0) *              &
            temperature(2) + 2.82713304d-06 * temperature(2)**2d0 +            &
            (-1.9212777625d-09) * temperature(2)**3d0 + 4.77359531d-13 *       &
            temperature(2)**4d0 + 9098.21441d0 / temperature(2), temperature(2)&
            > 1000.0d0), temperature(2) > 6000.0d0) * 277.0920218328648d0 *    &
            temperature(2) + (-1d0) * 969.8220764150268d0 * (temperature(2) +  &
            (-298.15d0)) + (-3041736.8944989573d0)
        vibe_energy(5) = merge((-497529430.0d0) / temperature(2)**2d0 +        &
            ((-286610.6874d0) * log(temperature(2))) / temperature(2) +        &
            66.9035225d0 + (-0.00308497951d0) * temperature(2) +               &
            1.0054653423333334d-07 * temperature(2)**2d0 + (-1.85535415d-12) * &
            temperature(2)**3d0 + 1.455635154d-17 * temperature(2)**4d0 +      &
            2293554.027d0 / temperature(2), merge(1037939.022d0 /              &
            temperature(2)**2d0 + (2344.830282d0 * log(temperature(2))) /      &
            temperature(2) + 1.819732036d0 + 0.000633923791d0 * temperature(2) &
            + (-7.29355996d-08) * temperature(2)**2d0 + 5.13429893d-12 *       &
            temperature(2)**3d0 + (-1.63869341d-16) * temperature(2)**4d0 +    &
            (-16890.10929d0) / temperature(2), 34255.6342d0 /                  &
            temperature(2)**2d0 + (484.700097d0 * log(temperature(2))) /       &
            temperature(2) + 1.119010961d0 + 0.00214694462d0 * temperature(2) +&
            (-2.27876684d-07) * temperature(2)**2d0 + (-5.05843175d-10) *      &
            temperature(2)**3d0 + 2.078080036d-13 * temperature(2)**4d0 +      &
            (-3391.45487d0) / temperature(2), temperature(2) > 1000.0d0),      &
            temperature(2) > 6000.0d0) * 259.83668137555156d0 * temperature(2) &
            + (-1d0) * 909.4283848144305d0 * (temperature(2) + (-298.15d0)) +  &
            0.0004002403019474347d0

    end subroutine get_nasa_polynomial_vibrational_energy

    subroutine get_nasa_polynomial_vibrational_specific_heat(&
        & temperature, vibe_specific_heat)

        GPU_ROUTINE(get_nasa_polynomial_vibrational_specific_heat)

        real(dp), intent(in), dimension(2) :: temperature
        real(dp), intent(out), &
            dimension(5) :: vibe_specific_heat

        vibe_specific_heat(1) = merge(547518105.0d0 / temperature(2)**2d0 +    &
            (-310757.498d0) / temperature(2) + 69.1678274d0 +                  &
            (-0.00684798813d0) * temperature(2) + 3.8275724d-07 *              &
            temperature(2)**2d0 + (-1.098367709d-11) * temperature(2)**3d0 +   &
            1.277986024d-16 * temperature(2)**4d0, merge(88765.0138d0 /        &
            temperature(2)**2d0 + (-107.12315d0) / temperature(2) +            &
            2.362188287d0 + 0.0002916720081d0 * temperature(2) +               &
            (-1.7295151d-07) * temperature(2)**2d0 + 4.01265788d-11 *          &
            temperature(2)**3d0 + (-2.677227571d-15) * temperature(2)**4d0,    &
            0.0d0 / temperature(2)**2d0 + 0.0d0 / temperature(2) + 2.5d0 +     &
            0.0d0 * temperature(2) + 0.0d0 * temperature(2)**2d0 + 0.0d0 *     &
            temperature(2)**3d0 + 0.0d0 * temperature(2)**4d0, temperature(2) >&
            1000.0d0), temperature(2) > 6000.0d0) * 593.6043647558247d0 +      &
            (-1484.0109118895616d0)
        vibe_specific_heat(2) = merge(177900426.4d0 / temperature(2)**2d0 +    &
            (-108232.8257d0) / temperature(2) + 28.10778365d0 +                &
            (-0.002975232262d0) * temperature(2) + 1.854997534d-07 *           &
            temperature(2)**2d0 + (-5.79623154d-12) * temperature(2)**3d0 +    &
            7.191720164d-17 * temperature(2)**4d0, merge(261902.0262d0 /       &
            temperature(2)**2d0 + (-729.872203d0) / temperature(2) +           &
            3.31717727d0 + (-0.000428133436d0) * temperature(2) +              &
            1.036104594d-07 * temperature(2)**2d0 + (-9.43830433d-12) *        &
            temperature(2)**3d0 + 2.725038297d-16 * temperature(2)**4d0,       &
            (-7953.6113d0) / temperature(2)**2d0 + 160.7177787d0 /             &
            temperature(2) + 1.966226438d0 + 0.00101367031d0 * temperature(2) +&
            (-1.110415423d-06) * temperature(2)**2d0 + 6.5175075d-10 *         &
            temperature(2)**3d0 + (-1.584779251d-13) * temperature(2)**4d0,    &
            temperature(2) > 1000.0d0), temperature(2) > 6000.0d0) *           &
            519.6733627511031d0 + (-1299.183406877758d0)
        vibe_specific_heat(3) = merge(831013916.0d0 / temperature(2)**2d0 +    &
            (-642073.354d0) / temperature(2) + 202.0264635d0 +                 &
            (-0.03065092046d0) * temperature(2) + 2.486903333d-06 *            &
            temperature(2)**2d0 + (-9.70595411d-11) * temperature(2)**3d0 +    &
            1.437538881d-15 * temperature(2)**4d0, merge(587712.406d0 /        &
            temperature(2)**2d0 + (-2239.249073d0) / temperature(2) +          &
            6.06694922d0 + (-0.00061396855d0) * temperature(2) +               &
            1.491806679d-07 * temperature(2)**2d0 + (-1.923105485d-11) *       &
            temperature(2)**3d0 + 1.061954386d-15 * temperature(2)**4d0,       &
            22103.71497d0 / temperature(2)**2d0 + (-381.846182d0) /            &
            temperature(2) + 6.08273836d0 + (-0.00853091441d0) * temperature(2)&
            + 1.384646189d-05 * temperature(2)**2d0 + (-9.62579362d-09) *      &
            temperature(2)**3d0 + 2.519705809d-12 * temperature(2)**4d0,       &
            temperature(2) > 1000.0d0), temperature(2) > 6000.0d0) *           &
            296.80218237791235d0 + (-1038.807638322693d0)
        vibe_specific_heat(4) = merge((-957530354.0d0) / temperature(2)**2d0 + &
            591243.448d0 / temperature(2) + (-138.4566826d0) + 0.01694339403d0 &
            * temperature(2) + (-1.007351096d-06) * temperature(2)**2d0 +      &
            2.912584076d-11 * temperature(2)**3d0 + (-3.29510935d-16) *        &
            temperature(2)**4d0, merge(223901.8716d0 / temperature(2)**2d0 +   &
            (-1289.651623d0) / temperature(2) + 5.43393603d0 +                 &
            (-0.00036560349d0) * temperature(2) + 9.88096645d-08 *             &
            temperature(2)**2d0 + (-1.416076856d-11) * temperature(2)**3d0 +   &
            9.38018462d-16 * temperature(2)**4d0, (-11439.16503d0) /           &
            temperature(2)**2d0 + 153.6467592d0 / temperature(2) + 3.43146873d0&
            + (-0.002668592368d0) * temperature(2) + 8.48139912d-06 *          &
            temperature(2)**2d0 + (-7.68511105d-09) * temperature(2)**3d0 +    &
            2.386797655d-12 * temperature(2)**4d0, temperature(2) > 1000.0d0), &
            temperature(2) > 6000.0d0) * 277.0920218328648d0 +                 &
            (-969.8220764150268d0)
        vibe_specific_heat(5) = merge(497529430.0d0 / temperature(2)**2d0 +    &
            (-286610.6874d0) / temperature(2) + 66.9035225d0 +                 &
            (-0.00616995902d0) * temperature(2) + 3.016396027d-07 *            &
            temperature(2)**2d0 + (-7.4214166d-12) * temperature(2)**3d0 +     &
            7.27817577d-17 * temperature(2)**4d0, merge((-1037939.022d0) /     &
            temperature(2)**2d0 + 2344.830282d0 / temperature(2) +             &
            1.819732036d0 + 0.001267847582d0 * temperature(2) +                &
            (-2.188067988d-07) * temperature(2)**2d0 + 2.053719572d-11 *       &
            temperature(2)**3d0 + (-8.19346705d-16) * temperature(2)**4d0,     &
            (-34255.6342d0) / temperature(2)**2d0 + 484.700097d0 /             &
            temperature(2) + 1.119010961d0 + 0.00429388924d0 * temperature(2) +&
            (-6.83630052d-07) * temperature(2)**2d0 + (-2.0233727d-09) *       &
            temperature(2)**3d0 + 1.039040018d-12 * temperature(2)**4d0,       &
            temperature(2) > 1000.0d0), temperature(2) > 6000.0d0) *           &
            259.83668137555156d0 + (-909.4283848144305d0)

    end subroutine get_nasa_polynomial_vibrational_specific_heat

end module libpyro_fortran_air5
