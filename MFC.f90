
#ifndef PYROMETHEUS_CALLER_INDEXING
#define PYROMETHEUS_CALLER_INDEXING 0
#endif


                #define GPU_ROUTINE(name) ! name
            

module MFC

    implicit none

    integer, parameter :: sp = selected_real_kind(6,37) ! Single precision
    integer, parameter :: dp = selected_real_kind(15,307) ! Double precision

    integer, parameter :: num_elements = 4
    integer, parameter :: num_species = 10
    integer, parameter :: num_reactions = 31
    integer, parameter :: num_falloff = 5
    real(dp), parameter :: one_atm = 101325.0d0
    real(dp), parameter :: gas_constant = 8314.46261815324d0
    real(dp), parameter :: molecular_weights(10) = &
        (/ 2.016d0, 1.008d0, 15.999d0, 31.998d0, 17.007d0, 18.015d0, 33.006d0, &
            34.014d0, 28.014d0, 39.95d0 /)
    real(dp), parameter :: inv_molecular_weights(10) = &
        (/ 0.49603174603174605d0, 0.9920634920634921d0, 0.06250390649415588d0, &
            0.03125195324707794d0, 0.05879931792791203d0,                      &
            0.055509297807382736d0, 0.030297521662727988d0,                    &
            0.029399658963956014d0, 0.03569643749553795d0,                     &
            0.025031289111389236d0 /)

    ! GPU Create (molecular_weights, inv_molecular_weights)

    character(len=12), parameter :: species_names(10) = &
        (/ "H2          " , "H           " , "O           " , "O2          " , &
            "OH          " , "H2O         " , "HO2         " , "H2O2        " ,&
            "N2          " , "Ar          " /)

    character(len=4), parameter :: element_names(4) = &
        (/ "O   " , "H   " , "N   " , "Ar  " /)

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

    subroutine get_element_index(el_name, el_index)

        character(len=*), intent(in) :: el_name
        integer, intent(out) :: el_index

        integer :: idx

        el_index = 0
        loop:do idx = 1, num_elements
            if(trim(adjustl(el_name)) .eq. trim(element_names(idx))) then
                el_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_element_index

    subroutine get_specific_gas_constant(mass_fractions, specific_gas_constant)

        GPU_ROUTINE(get_specific_gas_constant)

        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: specific_gas_constant

        specific_gas_constant = gas_constant * ( &
                    + inv_molecular_weights(1)*mass_fractions(1) &
                    + inv_molecular_weights(2)*mass_fractions(2) &
                    + inv_molecular_weights(3)*mass_fractions(3) &
                    + inv_molecular_weights(4)*mass_fractions(4) &
                    + inv_molecular_weights(5)*mass_fractions(5) &
                    + inv_molecular_weights(6)*mass_fractions(6) &
                    + inv_molecular_weights(7)*mass_fractions(7) &
                    + inv_molecular_weights(8)*mass_fractions(8) &
                    + inv_molecular_weights(9)*mass_fractions(9) &
                    + inv_molecular_weights(10)*mass_fractions(10) &
                )

    end subroutine get_specific_gas_constant

    subroutine get_density(pressure, temperature, mass_fractions, density)

        GPU_ROUTINE(get_density)

        real(dp), intent(in) :: pressure
        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: density

        real(dp) :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        density = pressure * mix_mol_weight / (gas_constant * temperature)

    end subroutine get_density

    subroutine get_pressure(density, temperature, mass_fractions, pressure)

        GPU_ROUTINE(get_pressure)

        real(dp), intent(in) :: density
        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: pressure

        real(dp) :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        pressure = density * gas_constant * temperature / mix_mol_weight

    end subroutine get_pressure

    subroutine get_mixture_molecular_weight(mass_fractions, mix_mol_weight)

        GPU_ROUTINE(get_mixture_molecular_weight)

        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: mix_mol_weight

        mix_mol_weight = 1.0d0 / ( &
                    + inv_molecular_weights(1)*mass_fractions(1) &
                    + inv_molecular_weights(2)*mass_fractions(2) &
                    + inv_molecular_weights(3)*mass_fractions(3) &
                    + inv_molecular_weights(4)*mass_fractions(4) &
                    + inv_molecular_weights(5)*mass_fractions(5) &
                    + inv_molecular_weights(6)*mass_fractions(6) &
                    + inv_molecular_weights(7)*mass_fractions(7) &
                    + inv_molecular_weights(8)*mass_fractions(8) &
                    + inv_molecular_weights(9)*mass_fractions(9) &
                    + inv_molecular_weights(10)*mass_fractions(10) &
                )

    end subroutine get_mixture_molecular_weight

    subroutine get_concentrations(density, mass_fractions, concentrations)

        GPU_ROUTINE(get_concentrations)

        real(dp), intent(in) :: density
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out), dimension(10) :: concentrations

        concentrations = density * inv_molecular_weights * mass_fractions

    end subroutine get_concentrations

    subroutine get_mole_fractions(mix_mol_weight, mass_fractions,              &
        mole_fractions)

        GPU_ROUTINE(get_mole_fractions)

        real(dp), intent(in) :: mix_mol_weight
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out), dimension(10) :: mole_fractions

        mole_fractions = inv_molecular_weights * mass_fractions * mix_mol_weight

    end subroutine get_mole_fractions

    subroutine get_mass_averaged_property(&
        & mass_fractions, spec_property, mix_property)

        GPU_ROUTINE(get_mass_averaged_property)

        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(in), dimension(10) :: spec_property
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
                + inv_molecular_weights(6)*mass_fractions(6) &
                *spec_property(6) &
                + inv_molecular_weights(7)*mass_fractions(7) &
                *spec_property(7) &
                + inv_molecular_weights(8)*mass_fractions(8) &
                *spec_property(8) &
                + inv_molecular_weights(9)*mass_fractions(9) &
                *spec_property(9) &
                + inv_molecular_weights(10)*mass_fractions(10) &
                *spec_property(10) &
        )

    end subroutine get_mass_averaged_property

    subroutine get_mixture_specific_heat_cp_mass(temperature, mass_fractions,  &
        cp_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cp_mass)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: cp_mix

        real(dp), dimension(10) :: cp0_r

        call get_species_specific_heats_r(temperature, cp0_r)
        call get_mass_averaged_property(mass_fractions, cp0_r, cp_mix)
        cp_mix = cp_mix * gas_constant

    end subroutine get_mixture_specific_heat_cp_mass

    subroutine get_mixture_specific_heat_cv_mass(temperature, mass_fractions,  &
        cv_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cv_mass)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: cv_mix

        real(dp), dimension(10) :: cp0_r

        call get_species_specific_heats_r(temperature, cp0_r)
        cp0_r(:) = cp0_r(:) - 1.d0
        call get_mass_averaged_property(mass_fractions, cp0_r, cv_mix)
        cv_mix = cv_mix * gas_constant

    end subroutine get_mixture_specific_heat_cv_mass

    subroutine get_mixture_enthalpy_mass(temperature, mass_fractions, h_mix)

        GPU_ROUTINE(get_mixture_enthalpy_mass)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: h_mix

        real(dp), dimension(10) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_mass_averaged_property(mass_fractions, h0_rt, h_mix)
        h_mix = h_mix * gas_constant * temperature

    end subroutine get_mixture_enthalpy_mass

    subroutine get_mixture_energy_mass(temperature, mass_fractions, e_mix)

        GPU_ROUTINE(get_mixture_energy_mass)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: e_mix

        real(dp), dimension(10) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        h0_rt(:) = h0_rt - 1.d0
        call get_mass_averaged_property(mass_fractions, h0_rt, e_mix)
        e_mix = e_mix * gas_constant * temperature

    end subroutine get_mixture_energy_mass

    subroutine get_species_specific_heats_r(temperature, cp0_r)

        GPU_ROUTINE(get_species_specific_heats_r)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: cp0_r

        cp0_r(1) = merge(3.3372792d0 + (-4.94024731d-05) * temperature +       &
            4.99456778d-07 * temperature**2d0 + (-1.79566394d-10) *            &
            temperature**3d0 + 2.00255376d-14 * temperature**4d0, 2.34433112d0 &
            + 0.00798052075d0 * temperature + (-1.9478151d-05) *               &
            temperature**2d0 + 2.01572094d-08 * temperature**3d0 +             &
            (-7.37611761d-12) * temperature**4d0, temperature > 1000.0d0)
        cp0_r(2) = merge(2.50000001d0 + (-2.30842973d-11) * temperature +      &
            1.61561948d-14 * temperature**2d0 + (-4.73515235d-18) *            &
            temperature**3d0 + 4.98197357d-22 * temperature**4d0, 2.5d0 +      &
            7.05332819d-13 * temperature + (-1.99591964d-15) * temperature**2d0&
            + 2.30081632d-18 * temperature**3d0 + (-9.27732332d-22) *          &
            temperature**4d0, temperature > 1000.0d0)
        cp0_r(3) = merge(2.56942078d0 + (-8.59741137d-05) * temperature +      &
            4.19484589d-08 * temperature**2d0 + (-1.00177799d-11) *            &
            temperature**3d0 + 1.22833691d-15 * temperature**4d0, 3.1682671d0 +&
            (-0.00327931884d0) * temperature + 6.64306396d-06 *                &
            temperature**2d0 + (-6.12806624d-09) * temperature**3d0 +          &
            2.11265971d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(4) = merge(3.28253784d0 + 0.00148308754d0 * temperature +        &
            (-7.57966669d-07) * temperature**2d0 + 2.09470555d-10 *            &
            temperature**3d0 + (-2.16717794d-14) * temperature**4d0,           &
            3.78245636d0 + (-0.00299673416d0) * temperature + 9.84730201d-06 * &
            temperature**2d0 + (-9.68129509d-09) * temperature**3d0 +          &
            3.24372837d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(5) = merge(3.09288767d0 + 0.000548429716d0 * temperature +       &
            1.26505228d-07 * temperature**2d0 + (-8.79461556d-11) *            &
            temperature**3d0 + 1.17412376d-14 * temperature**4d0, 3.99201543d0 &
            + (-0.00240131752d0) * temperature + 4.61793841d-06 *              &
            temperature**2d0 + (-3.88113333d-09) * temperature**3d0 +          &
            1.3641147d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(6) = merge(3.03399249d0 + 0.00217691804d0 * temperature +        &
            (-1.64072518d-07) * temperature**2d0 + (-9.7041987d-11) *          &
            temperature**3d0 + 1.68200992d-14 * temperature**4d0, 4.19864056d0 &
            + (-0.0020364341d0) * temperature + 6.52040211d-06 *               &
            temperature**2d0 + (-5.48797062d-09) * temperature**3d0 +          &
            1.77197817d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(7) = merge(4.17228741d0 + 0.00188117627d0 * temperature +        &
            (-3.46277286d-07) * temperature**2d0 + 1.94657549d-11 *            &
            temperature**3d0 + 1.76256905d-16 * temperature**4d0, 4.30179807d0 &
            + (-0.00474912097d0) * temperature + 2.11582905d-05 *              &
            temperature**2d0 + (-2.42763914d-08) * temperature**3d0 +          &
            9.29225225d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(8) = merge(4.16500285d0 + 0.00490831694d0 * temperature +        &
            (-1.90139225d-06) * temperature**2d0 + 3.71185986d-10 *            &
            temperature**3d0 + (-2.87908305d-14) * temperature**4d0,           &
            4.27611269d0 + (-0.000542822417d0) * temperature + 1.67335701d-05 *&
            temperature**2d0 + (-2.15770813d-08) * temperature**3d0 +          &
            8.62454363d-12 * temperature**4d0, temperature > 1000.0d0)
        cp0_r(9) = merge(2.92664d0 + 0.0014879768d0 * temperature +            &
            (-5.68476d-07) * temperature**2d0 + 1.0097038d-10 *                &
            temperature**3d0 + (-6.753351d-15) * temperature**4d0, 3.298677d0 +&
            0.0014082404d0 * temperature + (-3.963222d-06) * temperature**2d0 +&
            5.641515d-09 * temperature**3d0 + (-2.444854d-12) *                &
            temperature**4d0, temperature > 1000.0d0)
        cp0_r(10) = merge(2.5d0 + 0.0d0 * temperature, 2.5d0 + 0.0d0 *         &
            temperature, temperature > 1000.0d0)

    end subroutine get_species_specific_heats_r

    subroutine get_species_enthalpies_rt(temperature, h0_rt)

        GPU_ROUTINE(get_species_enthalpies_rt)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: h0_rt

        h0_rt(1) = merge(3.3372792d0 + (-2.470123655d-05) * temperature +      &
            1.6648559266666665d-07 * temperature**2d0 + (-4.48915985d-11) *    &
            temperature**3d0 + 4.00510752d-15 * temperature**4d0 +             &
            (-950.158922d0) / temperature, 2.34433112d0 + 0.003990260375d0 *   &
            temperature + (-6.4927169999999995d-06) * temperature**2d0 +       &
            5.03930235d-09 * temperature**3d0 + (-1.4752235220000002d-12) *    &
            temperature**4d0 + (-917.935173d0) / temperature, temperature >    &
            1000.0d0)
        h0_rt(2) = merge(2.50000001d0 + (-1.154214865d-11) * temperature +     &
            5.385398266666667d-15 * temperature**2d0 + (-1.1837880875d-18) *   &
            temperature**3d0 + 9.96394714d-23 * temperature**4d0 + 25473.6599d0&
            / temperature, 2.5d0 + 3.526664095d-13 * temperature +             &
            (-6.653065466666667d-16) * temperature**2d0 + 5.7520408d-19 *      &
            temperature**3d0 + (-1.855464664d-22) * temperature**4d0 +         &
            25473.6599d0 / temperature, temperature > 1000.0d0)
        h0_rt(3) = merge(2.56942078d0 + (-4.298705685d-05) * temperature +     &
            1.3982819633333334d-08 * temperature**2d0 + (-2.504444975d-12) *   &
            temperature**3d0 + 2.4566738199999997d-16 * temperature**4d0 +     &
            29217.5791d0 / temperature, 3.1682671d0 + (-0.00163965942d0) *     &
            temperature + 2.2143546533333334d-06 * temperature**2d0 +          &
            (-1.53201656d-09) * temperature**3d0 + 4.22531942d-13 *            &
            temperature**4d0 + 29122.2592d0 / temperature, temperature >       &
            1000.0d0)
        h0_rt(4) = merge(3.28253784d0 + 0.00074154377d0 * temperature +        &
            (-2.526555563333333d-07) * temperature**2d0 + 5.236763875d-11 *    &
            temperature**3d0 + (-4.33435588d-15) * temperature**4d0 +          &
            (-1088.45772d0) / temperature, 3.78245636d0 + (-0.00149836708d0) * &
            temperature + 3.282434003333333d-06 * temperature**2d0 +           &
            (-2.4203237725d-09) * temperature**3d0 + 6.48745674d-13 *          &
            temperature**4d0 + (-1063.94356d0) / temperature, temperature >    &
            1000.0d0)
        h0_rt(5) = merge(3.09288767d0 + 0.000274214858d0 * temperature +       &
            4.216840933333333d-08 * temperature**2d0 + (-2.19865389d-11) *     &
            temperature**3d0 + 2.34824752d-15 * temperature**4d0 + 3615.85d0 / &
            temperature, 3.99201543d0 + (-0.00120065876d0) * temperature +     &
            1.5393128033333333d-06 * temperature**2d0 + (-9.702833325d-10) *   &
            temperature**3d0 + 2.7282294d-13 * temperature**4d0 + 3372.27356d0 &
            / temperature, temperature > 1000.0d0)
        h0_rt(6) = merge(3.03399249d0 + 0.00108845902d0 * temperature +        &
            (-5.469083933333333d-08) * temperature**2d0 + (-2.426049675d-11) * &
            temperature**3d0 + 3.36401984d-15 * temperature**4d0 +             &
            (-30004.2971d0) / temperature, 4.19864056d0 + (-0.00101821705d0) * &
            temperature + 2.17346737d-06 * temperature**2d0 +                  &
            (-1.371992655d-09) * temperature**3d0 + 3.54395634d-13 *           &
            temperature**4d0 + (-30293.7267d0) / temperature, temperature >    &
            1000.0d0)
        h0_rt(7) = merge(4.17228741d0 + 0.000940588135d0 * temperature +       &
            (-1.15425762d-07) * temperature**2d0 + 4.866438725d-12 *           &
            temperature**3d0 + 3.5251381d-17 * temperature**4d0 + 31.0206839d0 &
            / temperature, 4.30179807d0 + (-0.002374560485d0) * temperature +  &
            7.052763500000001d-06 * temperature**2d0 + (-6.06909785d-09) *     &
            temperature**3d0 + 1.8584504500000003d-12 * temperature**4d0 +     &
            264.018485d0 / temperature, temperature > 1000.0d0)
        h0_rt(8) = merge(4.16500285d0 + 0.00245415847d0 * temperature +        &
            (-6.337974166666666d-07) * temperature**2d0 + 9.27964965d-11 *     &
            temperature**3d0 + (-5.7581661d-15) * temperature**4d0 +           &
            (-17861.7877d0) / temperature, 4.27611269d0 + (-0.0002714112085d0) &
            * temperature + 5.5778567000000005d-06 * temperature**2d0 +        &
            (-5.394270325d-09) * temperature**3d0 + 1.724908726d-12 *          &
            temperature**4d0 + (-17702.5821d0) / temperature, temperature >    &
            1000.0d0)
        h0_rt(9) = merge(2.92664d0 + 0.0007439884d0 * temperature +            &
            (-1.8949200000000001d-07) * temperature**2d0 + 2.5242595d-11 *     &
            temperature**3d0 + (-1.3506701999999999d-15) * temperature**4d0 +  &
            (-922.7977d0) / temperature, 3.298677d0 + 0.0007041202d0 *         &
            temperature + (-1.3210739999999999d-06) * temperature**2d0 +       &
            1.41037875d-09 * temperature**3d0 + (-4.889707999999999d-13) *     &
            temperature**4d0 + (-1020.8999d0) / temperature, temperature >     &
            1000.0d0)
        h0_rt(10) = merge(2.5d0 + 0.0d0 * temperature + (-745.375d0) /         &
            temperature, 2.5d0 + 0.0d0 * temperature + (-745.375d0) /          &
            temperature, temperature > 1000.0d0)

    end subroutine get_species_enthalpies_rt

    subroutine get_species_entropies_r(temperature, s0_r)

        GPU_ROUTINE(get_species_entropies_r)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: s0_r

        s0_r(1) = merge(3.3372792d0 * log(temperature) + (-4.94024731d-05) *   &
            temperature + 2.49728389d-07 * temperature**2d0 +                  &
            (-5.985546466666667d-11) * temperature**3d0 + 5.0063844d-15 *      &
            temperature**4d0 + (-3.20502331d0), 2.34433112d0 * log(temperature)&
            + 0.00798052075d0 * temperature + (-9.7390755d-06) *               &
            temperature**2d0 + 6.7190698d-09 * temperature**3d0 +              &
            (-1.8440294025d-12) * temperature**4d0 + 0.683010238d0, temperature&
            > 1000.0d0)
        s0_r(2) = merge(2.50000001d0 * log(temperature) + (-2.30842973d-11) *  &
            temperature + 8.0780974d-15 * temperature**2d0 +                   &
            (-1.5783841166666668d-18) * temperature**3d0 + 1.2454933925d-22 *  &
            temperature**4d0 + (-0.446682914d0), 2.5d0 * log(temperature) +    &
            7.05332819d-13 * temperature + (-9.9795982d-16) * temperature**2d0 &
            + 7.669387733333333d-19 * temperature**3d0 + (-2.31933083d-22) *   &
            temperature**4d0 + (-0.446682853d0), temperature > 1000.0d0)
        s0_r(3) = merge(2.56942078d0 * log(temperature) + (-8.59741137d-05) *  &
            temperature + 2.097422945d-08 * temperature**2d0 +                 &
            (-3.3392599666666663d-12) * temperature**3d0 + 3.070842275d-16 *   &
            temperature**4d0 + 4.78433864d0, 3.1682671d0 * log(temperature) +  &
            (-0.00327931884d0) * temperature + 3.32153198d-06 *                &
            temperature**2d0 + (-2.0426887466666666d-09) * temperature**3d0 +  &
            5.281649275d-13 * temperature**4d0 + 2.05193346d0, temperature >   &
            1000.0d0)
        s0_r(4) = merge(3.28253784d0 * log(temperature) + 0.00148308754d0 *    &
            temperature + (-3.789833345d-07) * temperature**2d0 +              &
            6.982351833333333d-11 * temperature**3d0 + (-5.41794485d-15) *     &
            temperature**4d0 + 5.45323129d0, 3.78245636d0 * log(temperature) + &
            (-0.00299673416d0) * temperature + 4.923651005d-06 *               &
            temperature**2d0 + (-3.2270983633333334d-09) * temperature**3d0 +  &
            8.109320925d-13 * temperature**4d0 + 3.65767573d0, temperature >   &
            1000.0d0)
        s0_r(5) = merge(3.09288767d0 * log(temperature) + 0.000548429716d0 *   &
            temperature + 6.3252614d-08 * temperature**2d0 + (-2.93153852d-11) &
            * temperature**3d0 + 2.9353094d-15 * temperature**4d0 +            &
            4.4766961d0, 3.99201543d0 * log(temperature) + (-0.00240131752d0) *&
            temperature + 2.308969205d-06 * temperature**2d0 +                 &
            (-1.29371111d-09) * temperature**3d0 + 3.41028675d-13 *            &
            temperature**4d0 + (-0.103925458d0), temperature > 1000.0d0)
        s0_r(6) = merge(3.03399249d0 * log(temperature) + 0.00217691804d0 *    &
            temperature + (-8.2036259d-08) * temperature**2d0 +                &
            (-3.2347329d-11) * temperature**3d0 + 4.2050248d-15 *              &
            temperature**4d0 + 4.9667701d0, 4.19864056d0 * log(temperature) +  &
            (-0.0020364341d0) * temperature + 3.260201055d-06 *                &
            temperature**2d0 + (-1.82932354d-09) * temperature**3d0 +          &
            4.429945425d-13 * temperature**4d0 + (-0.849032208d0), temperature &
            > 1000.0d0)
        s0_r(7) = merge(4.17228741d0 * log(temperature) + 0.00188117627d0 *    &
            temperature + (-1.73138643d-07) * temperature**2d0 +               &
            6.4885849666666665d-12 * temperature**3d0 + 4.406422625d-17 *      &
            temperature**4d0 + 2.95767672d0, 4.30179807d0 * log(temperature) + &
            (-0.00474912097d0) * temperature + 1.057914525d-05 *               &
            temperature**2d0 + (-8.092130466666667d-09) * temperature**3d0 +   &
            2.3230630625d-12 * temperature**4d0 + 3.7166622d0, temperature >   &
            1000.0d0)
        s0_r(8) = merge(4.16500285d0 * log(temperature) + 0.00490831694d0 *    &
            temperature + (-9.50696125d-07) * temperature**2d0 +               &
            1.2372866199999999d-10 * temperature**3d0 + (-7.197707625d-15) *   &
            temperature**4d0 + 2.91615662d0, 4.27611269d0 * log(temperature) + &
            (-0.000542822417d0) * temperature + 8.36678505d-06 *               &
            temperature**2d0 + (-7.192360433333333d-09) * temperature**3d0 +   &
            2.1561359075d-12 * temperature**4d0 + 3.43505074d0, temperature >  &
            1000.0d0)
        s0_r(9) = merge(2.92664d0 * log(temperature) + 0.0014879768d0 *        &
            temperature + (-2.84238d-07) * temperature**2d0 +                  &
            3.3656793333333334d-11 * temperature**3d0 + (-1.68833775d-15) *    &
            temperature**4d0 + 5.980528d0, 3.298677d0 * log(temperature) +     &
            0.0014082404d0 * temperature + (-1.981611d-06) * temperature**2d0 +&
            1.8805050000000002d-09 * temperature**3d0 + (-6.112135d-13) *      &
            temperature**4d0 + 3.950372d0, temperature > 1000.0d0)
        s0_r(10) = merge(2.5d0 * log(temperature) + 0.0d0 * temperature +      &
            4.366d0, 2.5d0 * log(temperature) + 0.0d0 * temperature + 4.366d0, &
            temperature > 1000.0d0)

    end subroutine get_species_entropies_r

    subroutine get_species_gibbs_rt(temperature, g0_rt)

        GPU_ROUTINE(get_species_gibbs_rt)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: g0_rt

        real(dp), dimension(10) :: h0_rt
        real(dp), dimension(10) :: s0_r

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_species_entropies_r(temperature, s0_r)
        g0_rt(:) = h0_rt(:) - s0_r(:)

    end subroutine get_species_gibbs_rt

    subroutine get_equilibrium_constants(temperature, k_eq)

        GPU_ROUTINE(get_equilibrium_constants)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(31) :: k_eq

        real(dp) :: rt
        real(dp) :: c0

        real(dp), dimension(10) :: g0_rt

        rt = gas_constant * temperature
        c0 = log(one_atm/rt)

        call get_species_gibbs_rt(temperature, g0_rt)

        k_eq(1) = 0d0 + 1.0d0 * g0_rt(3) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 +  &
            1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(4))
        k_eq(2) = 0d0 + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) +  &
            1.0d0 * g0_rt(4)) + (-1d0) * (-1.0d0) * c0
        k_eq(3) = 0d0 + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) +  &
            1.0d0 * g0_rt(4)) + (-1d0) * (-1.0d0) * c0
        k_eq(4) = 0d0 + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) +  &
            1.0d0 * g0_rt(4)) + (-1d0) * (-1.0d0) * c0
        k_eq(5) = 0d0 + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) +  &
            1.0d0 * g0_rt(4)) + (-1d0) * (-1.0d0) * c0
        k_eq(6) = 0d0 + 2.0d0 * g0_rt(5) + (-1d0) * (0d0 + 1.0d0 * g0_rt(8)) + &
            (-1d0) * 1.0d0 * c0
        k_eq(7) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 +  &
            1.0d0 * g0_rt(8) + 1.0d0 * g0_rt(5))
        k_eq(8) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 +  &
            1.0d0 * g0_rt(8) + 1.0d0 * g0_rt(5))
        k_eq(9) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 +  &
            1.0d0 * g0_rt(7) + 1.0d0 * g0_rt(5))
        k_eq(10) = 0d0 + 1.0d0 * g0_rt(8) + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 + &
            2.0d0 * g0_rt(7))
        k_eq(11) = 0d0 + 1.0d0 * g0_rt(8) + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 + &
            2.0d0 * g0_rt(7))
        k_eq(12) = 0d0 + 1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(6)) + (-1d0) * 1.0d0 * c0
        k_eq(13) = 0d0 + 1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(6)) + (-1d0) * 1.0d0 * c0
        k_eq(14) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(3) + (-1d0) * (0d0 + &
            2.0d0 * g0_rt(5))
        k_eq(15) = 0d0 + 1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(1) + 1.0d0 * g0_rt(3))
        k_eq(16) = 0d0 + 1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(1) + 1.0d0 * g0_rt(3))
        k_eq(17) = 0d0 + 1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(6) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(1) + 1.0d0 * g0_rt(5))
        k_eq(18) = 0d0 + 2.0d0 * g0_rt(5) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) + &
            1.0d0 * g0_rt(7))
        k_eq(19) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(3) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(7))
        k_eq(20) = 0d0 + 1.0d0 * g0_rt(1) + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(7))
        k_eq(21) = 0d0 + 1.0d0 * g0_rt(4) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(7) + 1.0d0 * g0_rt(3))
        k_eq(22) = 0d0 + 1.0d0 * g0_rt(1) + 1.0d0 * g0_rt(7) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(8))
        k_eq(23) = 0d0 + 1.0d0 * g0_rt(6) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(2) + 1.0d0 * g0_rt(8))
        k_eq(24) = 0d0 + 1.0d0 * g0_rt(7) + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + &
            1.0d0 * g0_rt(8) + 1.0d0 * g0_rt(3))
        k_eq(25) = 0d0 + 2.0d0 * g0_rt(2) + (-1d0) * (0d0 + 1.0d0 * g0_rt(1)) +&
            (-1d0) * 1.0d0 * c0
        k_eq(26) = 0d0 + 2.0d0 * g0_rt(2) + (-1d0) * (0d0 + 1.0d0 * g0_rt(1)) +&
            (-1d0) * 1.0d0 * c0
        k_eq(27) = 0d0 + 2.0d0 * g0_rt(2) + (-1d0) * (0d0 + 1.0d0 * g0_rt(1)) +&
            (-1d0) * 1.0d0 * c0
        k_eq(28) = 0d0 + 2.0d0 * g0_rt(2) + (-1d0) * (0d0 + 1.0d0 * g0_rt(1)) +&
            (-1d0) * 1.0d0 * c0
        k_eq(29) = 0d0 + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 + 2.0d0 * g0_rt(3)) +&
            (-1d0) * (-1.0d0) * c0
        k_eq(30) = 0d0 + 1.0d0 * g0_rt(4) + (-1d0) * (0d0 + 2.0d0 * g0_rt(3)) +&
            (-1d0) * (-1.0d0) * c0
        k_eq(31) = 0d0 + 1.0d0 * g0_rt(5) + (-1d0) * (0d0 + 1.0d0 * g0_rt(2) + &
            1.0d0 * g0_rt(3)) + (-1d0) * (-1.0d0) * c0

    end subroutine get_equilibrium_constants

    subroutine get_temperature( &
        & enthalpy_or_energy, t_guess, mass_fractions, do_energy, temperature)

        GPU_ROUTINE(get_temperature)

        logical, intent(in) :: do_energy
        real(dp), intent(in) :: enthalpy_or_energy
        real(dp), intent(in) :: t_guess
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: temperature

        integer :: iter
        integer, parameter :: num_iter = 500
        real(dp), parameter :: tol = 1.0d-06

        real(dp) :: iter_temp
        real(dp) :: iter_energy
        real(dp) :: iter_energy_deriv
        real(dp) :: iter_rhs
        real(dp) :: iter_deriv

        iter_rhs = 0.d0
        iter_deriv = 1.d0
        iter_temp = t_guess

        do iter = 1, num_iter
            if(do_energy) then
                call get_mixture_specific_heat_cv_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_energy_mass(iter_temp, mass_fractions,        &
                    iter_energy)
            else
                call get_mixture_specific_heat_cp_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_enthalpy_mass(&
                    & iter_temp, mass_fractions, iter_energy)
            endif
            iter_rhs = enthalpy_or_energy - iter_energy
            iter_deriv = (-1.d0)*iter_energy_deriv
            iter_temp = iter_temp - iter_rhs / iter_deriv
            if(abs(iter_rhs/iter_deriv) .lt. tol) exit
        end do

        temperature = iter_temp

    end subroutine get_temperature

    subroutine get_falloff_rates(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_falloff_rates)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: concentrations
        real(dp), intent(out), dimension(31) :: k_fwd

        real(dp), dimension(5) :: k_high
        real(dp), dimension(5) :: k_low
        real(dp), dimension(5) :: reduced_pressure
        real(dp), dimension(5) :: falloff_center
        real(dp), dimension(5) :: falloff_factor
        real(dp), dimension(5) :: falloff_function

        k_high(1) = 55900000000.00001d0 * temperature**0.2d0
        k_high(2) = 55900000000.00001d0 * temperature**0.2d0
        k_high(3) = 55900000000.00001d0 * temperature**0.2d0
        k_high(4) = 55900000000.00001d0 * temperature**0.2d0
        k_high(5) = exp(34.3867900379128d0 + 0.0d0 * log(temperature) + (-1d0) &
            * 24436.340546700063d0 / temperature)

        k_low(1) = 26500000000000.004d0 * temperature**(-1.3d0)
        k_low(2) = 6810000000000.001d0 * temperature**(-1.2d0)
        k_low(3) = 5690000000000.001d0 * temperature**(-1.1d0)
        k_low(4) = 37000000000000.01d0 * temperature**(-1.0d0)
        k_low(5) = exp(29.88756227042119d0 + 0.0d0 * log(temperature) + (-1d0) &
            * 21237.37397224841d0 / temperature)

        reduced_pressure(1) = (0d0 + 0.0d0 * concentrations(10) + 2.5d0 *      &
            concentrations(1) + 12.0d0 * concentrations(8) + 1.0d0 * (0d0 +    &
            concentrations(2) + concentrations(3) + concentrations(5) +        &
            concentrations(7) + concentrations(9)))*k_low(1)/k_high(1)
        reduced_pressure(2) = (0d0 + 1.0d0 *                                   &
            concentrations(10))*k_low(2)/k_high(2)
        reduced_pressure(3) = (0d0 + 1.0d0 *                                   &
            concentrations(4))*k_low(3)/k_high(3)
        reduced_pressure(4) = (0d0 + 1.0d0 *                                   &
            concentrations(6))*k_low(4)/k_high(4)
        reduced_pressure(5) = (0d0 + 1.0d0 * concentrations(10) + 2.5d0 *      &
            concentrations(1) + 15.0d0 * concentrations(6) + 15.0d0 *          &
            concentrations(8) + 1.5d0 * concentrations(9) + 1.0d0 * (0d0 +     &
            concentrations(2) + concentrations(3) + concentrations(4) +        &
            concentrations(5) + concentrations(7)))*k_low(5)/k_high(5)

        falloff_center(1) = log10(0.30000000000000004d0 * exp(((-1d0) *        &
            temperature) / 1d-30) + 0.7d0 * exp(((-1d0) * temperature) /       &
            1.0000000000000002d+30) + exp((-1d+30) / temperature))
        falloff_center(2) = log10(0.30000000000000004d0 * exp(((-1d0) *        &
            temperature) / 1d-30) + 0.7d0 * exp(((-1d0) * temperature) /       &
            1.0000000000000002d+30) + exp((-1d+30) / temperature))
        falloff_center(3) = log10(0.30000000000000004d0 * exp(((-1d0) *        &
            temperature) / 1d-30) + 0.7d0 * exp(((-1d0) * temperature) /       &
            1.0000000000000002d+30) + exp((-1d+30) / temperature))
        falloff_center(4) = log10(0.19999999999999996d0 * exp(((-1d0) *        &
            temperature) / 1d-30) + 0.8d0 * exp(((-1d0) * temperature) /       &
            1.0000000000000002d+30) + exp((-1d+30) / temperature))
        falloff_center(5) = log10(0.0d0 * exp(((-1d0) * temperature) / 1d-10) +&
            1.0d0 * exp(((-1d0) * temperature) / 10000000000.0d0) +            &
            exp((-10000000000.0d0) / temperature))

        falloff_factor(1) = merge((log10(reduced_pressure(1)) + (-0.4d0) +     &
            (-1d0) * 0.67d0 * falloff_center(1)) / (0.75d0 + (-1d0) * 1.27d0 * &
            falloff_center(1) + (-1d0) * 0.14d0 * (log10(reduced_pressure(1)) +&
            (-0.4d0) + (-1d0) * 0.67d0 * falloff_center(1))),                  &
            (-7.142857142857142d0), reduced_pressure(1) > 0d0)
        falloff_factor(2) = merge((log10(reduced_pressure(2)) + (-0.4d0) +     &
            (-1d0) * 0.67d0 * falloff_center(2)) / (0.75d0 + (-1d0) * 1.27d0 * &
            falloff_center(2) + (-1d0) * 0.14d0 * (log10(reduced_pressure(2)) +&
            (-0.4d0) + (-1d0) * 0.67d0 * falloff_center(2))),                  &
            (-7.142857142857142d0), reduced_pressure(2) > 0d0)
        falloff_factor(3) = merge((log10(reduced_pressure(3)) + (-0.4d0) +     &
            (-1d0) * 0.67d0 * falloff_center(3)) / (0.75d0 + (-1d0) * 1.27d0 * &
            falloff_center(3) + (-1d0) * 0.14d0 * (log10(reduced_pressure(3)) +&
            (-0.4d0) + (-1d0) * 0.67d0 * falloff_center(3))),                  &
            (-7.142857142857142d0), reduced_pressure(3) > 0d0)
        falloff_factor(4) = merge((log10(reduced_pressure(4)) + (-0.4d0) +     &
            (-1d0) * 0.67d0 * falloff_center(4)) / (0.75d0 + (-1d0) * 1.27d0 * &
            falloff_center(4) + (-1d0) * 0.14d0 * (log10(reduced_pressure(4)) +&
            (-0.4d0) + (-1d0) * 0.67d0 * falloff_center(4))),                  &
            (-7.142857142857142d0), reduced_pressure(4) > 0d0)
        falloff_factor(5) = merge((log10(reduced_pressure(5)) + (-0.4d0) +     &
            (-1d0) * 0.67d0 * falloff_center(5)) / (0.75d0 + (-1d0) * 1.27d0 * &
            falloff_center(5) + (-1d0) * 0.14d0 * (log10(reduced_pressure(5)) +&
            (-0.4d0) + (-1d0) * 0.67d0 * falloff_center(5))),                  &
            (-7.142857142857142d0), reduced_pressure(5) > 0d0)

        falloff_function(1) = 10d0**(falloff_center(1) / (1d0 +                &
            falloff_factor(1)**2d0))
        falloff_function(2) = 10d0**(falloff_center(2) / (1d0 +                &
            falloff_factor(2)**2d0))
        falloff_function(3) = 10d0**(falloff_center(3) / (1d0 +                &
            falloff_factor(3)**2d0))
        falloff_function(4) = 10d0**(falloff_center(4) / (1d0 +                &
            falloff_factor(4)**2d0))
        falloff_function(5) = 10d0**(falloff_center(5) / (1d0 +                &
            falloff_factor(5)**2d0))

        k_fwd(2) = k_high(1)*falloff_function(1) * &
            reduced_pressure(1)/(1.d0 + reduced_pressure(1))
        k_fwd(3) = k_high(2)*falloff_function(2) * &
            reduced_pressure(2)/(1.d0 + reduced_pressure(2))
        k_fwd(4) = k_high(3)*falloff_function(3) * &
            reduced_pressure(3)/(1.d0 + reduced_pressure(3))
        k_fwd(5) = k_high(4)*falloff_function(4) * &
            reduced_pressure(4)/(1.d0 + reduced_pressure(4))
        k_fwd(6) = k_high(5)*falloff_function(5) * &
            reduced_pressure(5)/(1.d0 + reduced_pressure(5))

    end subroutine get_falloff_rates

    subroutine get_fwd_rate_coefficients(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_fwd_rate_coefficients)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: concentrations
        real(dp), intent(out), dimension(31) :: k_fwd

        real(dp), dimension(5) :: k_falloff

        k_fwd(1) = exp(25.367656736087785d0 + 0.0d0 * log(temperature) + (-1d0)&
            * 7692.213789062133d0 / temperature)
        k_fwd(2) = 0.d0
        k_fwd(3) = 0.d0
        k_fwd(4) = 0.d0
        k_fwd(5) = 0.d0
        k_fwd(6) = 0.d0
        k_fwd(7) = exp(25.052155373207164d0 + 0.0d0 * log(temperature) + (-1d0)&
            * 3657.902789002528d0 / temperature)
        k_fwd(8) = exp(21.276000863789612d0 + 0.0d0 * log(temperature) + (-1d0)&
            * 160.0238116526075d0 / temperature)
        k_fwd(9) = exp(24.087107432064798d0 + 0.0d0 * log(temperature) + (-1d0)&
            * (-251.60976674938286d0) / temperature)
        k_fwd(10) = exp(18.683045008419857d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * (-806.6609121985215d0) / temperature)
        k_fwd(11) = exp(26.763520548223827d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 6028.570011315213d0 / temperature)
        k_fwd(12) = exp(57.06375203193832d0 + (-3.31d0) * log(temperature) +   &
            (-1d0) * 60773.82306064594d0 / temperature)
        k_fwd(13) = exp(52.95945713886305d0 + (-2.44d0) * log(temperature) +   &
            (-1d0) * 60466.85914521169d0 / temperature)
        k_fwd(14) = exp(3.5751506887855933d0 + 2.4d0 * log(temperature) +      &
            (-1d0) * (-1062.2964352158945d0) / temperature)
        k_fwd(15) = exp(22.063516259564896d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 3999.58885224819d0 / temperature)
        k_fwd(16) = exp(27.502050734631588d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 9646.718457171339d0 / temperature)
        k_fwd(17) = exp(12.287652632522597d0 + 1.52d0 * log(temperature) +     &
            (-1d0) * 1739.6299273052332d0 / temperature)
        k_fwd(18) = exp(24.983124837646084d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 150.96586004962973d0 / temperature)
        k_fwd(19) = 1450000000.0000002d0 * temperature**0.0d0
        k_fwd(20) = exp(8.205218426395412d0 + 2.087d0 * log(temperature) +     &
            (-1d0) * (-729.6683235732103d0) / temperature)
        k_fwd(21) = exp(23.514430944759127d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * (-223.93269240695076d0) / temperature)
        k_fwd(22) = exp(9.400960731584833d0 + 2.0d0 * log(temperature) + (-1d0)&
            * 2616.741574193582d0 / temperature)
        k_fwd(23) = exp(23.045653557236637d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 1800.016271325085d0 / temperature)
        k_fwd(24) = exp(20.552477515966128d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 1997.7815479901d0 / temperature)
        k_fwd(25) = exp(36.30350719175082d0 + (-1.1d0) * log(temperature) +    &
            (-1d0) * 52526.054906601166d0 / temperature)
        k_fwd(26) = exp(27.528988390363395d0 + 0.0d0 * log(temperature) +      &
            (-1d0) * 48344.300583226424d0 / temperature)
        k_fwd(27) = exp(38.363060486030825d0 + (-1.4d0) * log(temperature) +   &
            (-1d0) * 52526.054906601166d0 / temperature)
        k_fwd(28) = exp(38.363060486030825d0 + (-1.4d0) * log(temperature) +   &
            (-1d0) * 52526.054906601166d0 / temperature)
        k_fwd(29) = 6160000000.000001d0 * temperature**(-0.5d0)
        k_fwd(30) = exp(16.75467248002987d0 + 0.0d0 * log(temperature) + (-1d0)&
            * (-899.7565258957932d0) / temperature)
        k_fwd(31) = 4710000000000.001d0 * temperature**(-1.0d0)

        k_fwd(12) = k_fwd(12) * ( &
            0d0 + 3.0d0 * concentrations(1) + 2.0d0 * concentrations(9) + 1.5d0&
                * concentrations(4) + 1.0d0 * (0d0 + concentrations(2) +       &
                concentrations(3) + concentrations(5) + concentrations(7) +    &
                concentrations(8) + concentrations(10)))
        k_fwd(13) = k_fwd(13) * ( &
            0d0 + 1.0d0 * concentrations(6))
        k_fwd(25) = k_fwd(25) * ( &
            0d0 + 1.0d0 * concentrations(10) + 14.4d0 * concentrations(6) +    &
                14.4d0 * concentrations(8) + 1.0d0 * (0d0 + concentrations(2) +&
                concentrations(3) + concentrations(5) + concentrations(7)))
        k_fwd(26) = k_fwd(26) * ( &
            0d0 + 1.0d0 * concentrations(1))
        k_fwd(27) = k_fwd(27) * ( &
            0d0 + 1.0d0 * concentrations(9))
        k_fwd(28) = k_fwd(28) * ( &
            0d0 + 1.0d0 * concentrations(4))
        k_fwd(29) = k_fwd(29) * ( &
            0d0 + 0.0d0 * concentrations(10) + 2.5d0 * concentrations(1) +     &
                12.0d0 * concentrations(6) + 12.0d0 * concentrations(8) + 1.0d0&
                * (0d0 + concentrations(2) + concentrations(3) +               &
                concentrations(4) + concentrations(5) + concentrations(7) +    &
                concentrations(9)))
        k_fwd(30) = k_fwd(30) * ( &
            0d0 + 1.0d0 * concentrations(10))
        k_fwd(31) = k_fwd(31) * ( &
            0d0 + 0.75d0 * concentrations(10) + 2.5d0 * concentrations(1) +    &
                12.0d0 * concentrations(6) + 12.0d0 * concentrations(8) + 1.0d0&
                * (0d0 + concentrations(2) + concentrations(3) +               &
                concentrations(4) + concentrations(5) + concentrations(7) +    &
                concentrations(9)))

        call get_falloff_rates(temperature, concentrations, k_fwd)

    end subroutine get_fwd_rate_coefficients

    subroutine get_net_rates_of_progress(temperature, concentrations, r_net)

        GPU_ROUTINE(get_net_rates_of_progress)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: concentrations
        real(dp), intent(out), dimension(31) :: r_net

        real(dp), dimension(31) :: k_fwd
        real(dp), dimension(31) :: log_k_eq

        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        r_net(1) = k_fwd(1) * (concentrations(2)**1.0d0 *                      &
            concentrations(4)**1.0d0 + (-1d0) * exp(log_k_eq(1)) *             &
            concentrations(3)**1.0d0 * concentrations(5)**1.0d0)
        r_net(2) = k_fwd(2) * (concentrations(2)**1.0d0 *                      &
            concentrations(4)**1.0d0 + (-1d0) * exp(log_k_eq(2)) *             &
            concentrations(7)**1.0d0)
        r_net(3) = k_fwd(3) * (concentrations(2)**1.0d0 *                      &
            concentrations(4)**1.0d0 + (-1d0) * exp(log_k_eq(3)) *             &
            concentrations(7)**1.0d0)
        r_net(4) = k_fwd(4) * (concentrations(2)**1.0d0 *                      &
            concentrations(4)**1.0d0 + (-1d0) * exp(log_k_eq(4)) *             &
            concentrations(7)**1.0d0)
        r_net(5) = k_fwd(5) * (concentrations(2)**1.0d0 *                      &
            concentrations(4)**1.0d0 + (-1d0) * exp(log_k_eq(5)) *             &
            concentrations(7)**1.0d0)
        r_net(6) = k_fwd(6) * (concentrations(8)**1.0d0 + (-1d0) *             &
            exp(log_k_eq(6)) * concentrations(5)**2.0d0)
        r_net(7) = k_fwd(7) * (concentrations(8)**1.0d0 *                      &
            concentrations(5)**1.0d0 + (-1d0) * exp(log_k_eq(7)) *             &
            concentrations(6)**1.0d0 * concentrations(7)**1.0d0)
        r_net(8) = k_fwd(8) * (concentrations(8)**1.0d0 *                      &
            concentrations(5)**1.0d0 + (-1d0) * exp(log_k_eq(8)) *             &
            concentrations(6)**1.0d0 * concentrations(7)**1.0d0)
        r_net(9) = k_fwd(9) * (concentrations(7)**1.0d0 *                      &
            concentrations(5)**1.0d0 + (-1d0) * exp(log_k_eq(9)) *             &
            concentrations(6)**1.0d0 * concentrations(4)**1.0d0)
        r_net(10) = k_fwd(10) * (concentrations(7)**2.0d0 + (-1d0) *           &
            exp(log_k_eq(10)) * concentrations(8)**1.0d0 *                     &
            concentrations(4)**1.0d0)
        r_net(11) = k_fwd(11) * (concentrations(7)**2.0d0 + (-1d0) *           &
            exp(log_k_eq(11)) * concentrations(8)**1.0d0 *                     &
            concentrations(4)**1.0d0)
        r_net(12) = k_fwd(12) * (concentrations(6)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(12)) * concentrations(2)**1.0d0 *                     &
            concentrations(5)**1.0d0)
        r_net(13) = k_fwd(13) * (concentrations(6)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(13)) * concentrations(2)**1.0d0 *                     &
            concentrations(5)**1.0d0)
        r_net(14) = k_fwd(14) * (concentrations(5)**2.0d0 + (-1d0) *           &
            exp(log_k_eq(14)) * concentrations(6)**1.0d0 *                     &
            concentrations(3)**1.0d0)
        r_net(15) = k_fwd(15) * (concentrations(1)**1.0d0 *                    &
            concentrations(3)**1.0d0 + (-1d0) * exp(log_k_eq(15)) *            &
            concentrations(2)**1.0d0 * concentrations(5)**1.0d0)
        r_net(16) = k_fwd(16) * (concentrations(1)**1.0d0 *                    &
            concentrations(3)**1.0d0 + (-1d0) * exp(log_k_eq(16)) *            &
            concentrations(2)**1.0d0 * concentrations(5)**1.0d0)
        r_net(17) = k_fwd(17) * (concentrations(1)**1.0d0 *                    &
            concentrations(5)**1.0d0 + (-1d0) * exp(log_k_eq(17)) *            &
            concentrations(2)**1.0d0 * concentrations(6)**1.0d0)
        r_net(18) = k_fwd(18) * (concentrations(2)**1.0d0 *                    &
            concentrations(7)**1.0d0 + (-1d0) * exp(log_k_eq(18)) *            &
            concentrations(5)**2.0d0)
        r_net(19) = k_fwd(19) * (concentrations(2)**1.0d0 *                    &
            concentrations(7)**1.0d0 + (-1d0) * exp(log_k_eq(19)) *            &
            concentrations(6)**1.0d0 * concentrations(3)**1.0d0)
        r_net(20) = k_fwd(20) * (concentrations(2)**1.0d0 *                    &
            concentrations(7)**1.0d0 + (-1d0) * exp(log_k_eq(20)) *            &
            concentrations(1)**1.0d0 * concentrations(4)**1.0d0)
        r_net(21) = k_fwd(21) * (concentrations(7)**1.0d0 *                    &
            concentrations(3)**1.0d0 + (-1d0) * exp(log_k_eq(21)) *            &
            concentrations(4)**1.0d0 * concentrations(5)**1.0d0)
        r_net(22) = k_fwd(22) * (concentrations(2)**1.0d0 *                    &
            concentrations(8)**1.0d0 + (-1d0) * exp(log_k_eq(22)) *            &
            concentrations(1)**1.0d0 * concentrations(7)**1.0d0)
        r_net(23) = k_fwd(23) * (concentrations(2)**1.0d0 *                    &
            concentrations(8)**1.0d0 + (-1d0) * exp(log_k_eq(23)) *            &
            concentrations(6)**1.0d0 * concentrations(5)**1.0d0)
        r_net(24) = k_fwd(24) * (concentrations(8)**1.0d0 *                    &
            concentrations(3)**1.0d0 + (-1d0) * exp(log_k_eq(24)) *            &
            concentrations(7)**1.0d0 * concentrations(5)**1.0d0)
        r_net(25) = k_fwd(25) * (concentrations(1)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(25)) * concentrations(2)**2.0d0)
        r_net(26) = k_fwd(26) * (concentrations(1)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(26)) * concentrations(2)**2.0d0)
        r_net(27) = k_fwd(27) * (concentrations(1)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(27)) * concentrations(2)**2.0d0)
        r_net(28) = k_fwd(28) * (concentrations(1)**1.0d0 + (-1d0) *           &
            exp(log_k_eq(28)) * concentrations(2)**2.0d0)
        r_net(29) = k_fwd(29) * (concentrations(3)**2.0d0 + (-1d0) *           &
            exp(log_k_eq(29)) * concentrations(4)**1.0d0)
        r_net(30) = k_fwd(30) * (concentrations(3)**2.0d0 + (-1d0) *           &
            exp(log_k_eq(30)) * concentrations(4)**1.0d0)
        r_net(31) = k_fwd(31) * (concentrations(2)**1.0d0 *                    &
            concentrations(3)**1.0d0 + (-1d0) * exp(log_k_eq(31)) *            &
            concentrations(5)**1.0d0)

    end subroutine get_net_rates_of_progress

    subroutine get_net_production_rates(density, temperature, mass_fractions,  &
        omega)

        GPU_ROUTINE(get_net_production_rates)

        real(dp), intent(in) :: density
        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out), dimension(10) :: omega

        real(dp), dimension(10) :: concentrations
        real(dp), dimension(31) :: r_net

        call get_concentrations(density, mass_fractions, concentrations)
        call get_net_rates_of_progress(temperature, concentrations, r_net)

        omega(1) = (0d0 + 1.0d0 * r_net(20) + 1.0d0 * r_net(22) + (-1d0) * (0d0&
            + 1.0d0 * r_net(15) + 1.0d0 * r_net(16) + 1.0d0 * r_net(17) + 1.0d0&
            * r_net(25) + 1.0d0 * r_net(26) + 1.0d0 * r_net(27) + 1.0d0 *      &
            r_net(28))) * (0d0 * r_net(1) + 1.0d0)
        omega(2) = (0d0 + 1.0d0 * r_net(12) + 1.0d0 * r_net(13) + 1.0d0 *      &
            r_net(15) + 1.0d0 * r_net(16) + 1.0d0 * r_net(17) + 2.0d0 *        &
            r_net(25) + 2.0d0 * r_net(26) + 2.0d0 * r_net(27) + 2.0d0 *        &
            r_net(28) + (-1d0) * (0d0 + 1.0d0 * r_net(1) + 1.0d0 * r_net(2) +  &
            1.0d0 * r_net(3) + 1.0d0 * r_net(4) + 1.0d0 * r_net(5) + 1.0d0 *   &
            r_net(18) + 1.0d0 * r_net(19) + 1.0d0 * r_net(20) + 1.0d0 *        &
            r_net(22) + 1.0d0 * r_net(23) + 1.0d0 * r_net(31))) * (0d0 *       &
            r_net(1) + 1.0d0)
        omega(3) = (0d0 + 1.0d0 * r_net(1) + 1.0d0 * r_net(14) + 1.0d0 *       &
            r_net(19) + (-1d0) * (0d0 + 1.0d0 * r_net(15) + 1.0d0 * r_net(16) +&
            1.0d0 * r_net(21) + 1.0d0 * r_net(24) + 2.0d0 * r_net(29) + 2.0d0 *&
            r_net(30) + 1.0d0 * r_net(31))) * (0d0 * r_net(1) + 1.0d0)
        omega(4) = (0d0 + 1.0d0 * r_net(9) + 1.0d0 * r_net(10) + 1.0d0 *       &
            r_net(11) + 1.0d0 * r_net(20) + 1.0d0 * r_net(21) + 1.0d0 *        &
            r_net(29) + 1.0d0 * r_net(30) + (-1d0) * (0d0 + 1.0d0 * r_net(1) + &
            1.0d0 * r_net(2) + 1.0d0 * r_net(3) + 1.0d0 * r_net(4) + 1.0d0 *   &
            r_net(5))) * (0d0 * r_net(1) + 1.0d0)
        omega(5) = (0d0 + 1.0d0 * r_net(1) + 2.0d0 * r_net(6) + 1.0d0 *        &
            r_net(12) + 1.0d0 * r_net(13) + 1.0d0 * r_net(15) + 1.0d0 *        &
            r_net(16) + 2.0d0 * r_net(18) + 1.0d0 * r_net(21) + 1.0d0 *        &
            r_net(23) + 1.0d0 * r_net(24) + 1.0d0 * r_net(31) + (-1d0) * (0d0 +&
            1.0d0 * r_net(7) + 1.0d0 * r_net(8) + 1.0d0 * r_net(9) + 2.0d0 *   &
            r_net(14) + 1.0d0 * r_net(17))) * (0d0 * r_net(1) + 1.0d0)
        omega(6) = (0d0 + 1.0d0 * r_net(7) + 1.0d0 * r_net(8) + 1.0d0 *        &
            r_net(9) + 1.0d0 * r_net(14) + 1.0d0 * r_net(17) + 1.0d0 *         &
            r_net(19) + 1.0d0 * r_net(23) + (-1d0) * (0d0 + 1.0d0 * r_net(12) +&
            1.0d0 * r_net(13))) * (0d0 * r_net(1) + 1.0d0)
        omega(7) = (0d0 + 1.0d0 * r_net(2) + 1.0d0 * r_net(3) + 1.0d0 *        &
            r_net(4) + 1.0d0 * r_net(5) + 1.0d0 * r_net(7) + 1.0d0 * r_net(8) +&
            1.0d0 * r_net(22) + 1.0d0 * r_net(24) + (-1d0) * (0d0 + 1.0d0 *    &
            r_net(9) + 2.0d0 * r_net(10) + 2.0d0 * r_net(11) + 1.0d0 *         &
            r_net(18) + 1.0d0 * r_net(19) + 1.0d0 * r_net(20) + 1.0d0 *        &
            r_net(21))) * (0d0 * r_net(1) + 1.0d0)
        omega(8) = (0d0 + 1.0d0 * r_net(10) + 1.0d0 * r_net(11) + (-1d0) * (0d0&
            + 1.0d0 * r_net(6) + 1.0d0 * r_net(7) + 1.0d0 * r_net(8) + 1.0d0 * &
            r_net(22) + 1.0d0 * r_net(23) + 1.0d0 * r_net(24))) * (0d0 *       &
            r_net(1) + 1.0d0)
        omega(9) = 0d0 * (0d0 * r_net(1) + 1.0d0)
        omega(10) = 0d0 * (0d0 * r_net(1) + 1.0d0)

    end subroutine get_net_production_rates

    subroutine get_species_viscosities(temperature, viscosities)

        GPU_ROUTINE(get_species_viscosities)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: viscosities

        viscosities(1) = sqrt(temperature) * ((-0.00032862351581375954d0) +    &
            0.00047402944328358616d0 * log(temperature) +                      &
            (-8.852339013572501d-05) * log(temperature)**2d0 +                 &
            8.188000383358501d-06 * log(temperature)**3d0 +                    &
            (-2.775116846638978d-07) * log(temperature)**4d0)**2d0
        viscosities(2) = sqrt(temperature) * ((-0.005094323314880651d0) +      &
            0.0028116191560081244d0 * log(temperature) +                       &
            (-0.0005132615516554624d0) * log(temperature)**2d0 +               &
            4.2510638060585776d-05 * log(temperature)**3d0 +                   &
            (-1.315280069449328d-06) * log(temperature)**4d0)**2d0
        viscosities(3) = sqrt(temperature) * ((-0.004936023407701543d0) +      &
            0.003094326393094132d0 * log(temperature) +                        &
            (-0.0006001283031241224d0) * log(temperature)**2d0 +               &
            5.318270134950558d-05 * log(temperature)**3d0 +                    &
            (-1.7573576309977012d-06) * log(temperature)**4d0)**2d0
        viscosities(4) = sqrt(temperature) * ((-0.006186428071747617d0) +      &
            0.0036188245122442887d0 * log(temperature) +                       &
            (-0.0006861983404475868d0) * log(temperature)**2d0 +               &
            5.91601270960357d-05 * log(temperature)**3d0 +                     &
            (-1.9049771784291988d-06) * log(temperature)**4d0)**2d0
        viscosities(5) = sqrt(temperature) * ((-0.005011998459951971d0) +      &
            0.0031419541269966287d0 * log(temperature) +                       &
            (-0.0006093654512129685d0) * log(temperature)**2d0 +               &
            5.400128711786429d-05 * log(temperature)**3d0 +                    &
            (-1.7844068013131771d-06) * log(temperature)**4d0)**2d0
        viscosities(6) = sqrt(temperature) * (0.009495196334902445d0 +         &
            (-0.004974400618415688d0) * log(temperature) +                     &
            0.0009719845681883221d0 * log(temperature)**2d0 +                  &
            (-7.63468726043692d-05) * log(temperature)**3d0 +                  &
            2.074120177526231d-06 * log(temperature)**4d0)**2d0
        viscosities(7) = sqrt(temperature) * ((-0.006234584035892221d0) +      &
            0.0036469939149154705d0 * log(temperature) +                       &
            (-0.0006915397979571317d0) * log(temperature)**2d0 +               &
            5.962063725257017d-05 * log(temperature)**3d0 +                    &
            (-1.919805769606859d-06) * log(temperature)**4d0)**2d0
        viscosities(8) = sqrt(temperature) * ((-0.0062816492969005565d0) +     &
            0.0036745252978457204d0 * log(temperature) +                       &
            (-0.0006967602747203512d0) * log(temperature)**2d0 +               &
            6.007071713560211d-05 * log(temperature)**3d0 +                    &
            (-1.934298502258151d-06) * log(temperature)**4d0)**2d0
        viscosities(9) = sqrt(temperature) * ((-0.005232503458857763d0) +      &
            0.0031249811202461084d0 * log(temperature) +                       &
            (-0.0005968857242794387d0) * log(temperature)**2d0 +               &
            5.190839695047226d-05 * log(temperature)**3d0 +                    &
            (-1.6850989392760908d-06) * log(temperature)**4d0)**2d0
        viscosities(10) = sqrt(temperature) * ((-0.007767485546486584d0) +     &
            0.004336606391136823d0 * log(temperature) +                        &
            (-0.0007995694940274129d0) * log(temperature)**2d0 +               &
            6.6903871218628d-05 * log(temperature)**3d0 +                      &
            (-2.0918030143678814d-06) * log(temperature)**4d0)**2d0

    end subroutine get_species_viscosities

    subroutine get_species_thermal_conductivities(temperature, conductivities)

        GPU_ROUTINE(get_species_thermal_conductivities)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10) :: conductivities

        conductivities(1) = sqrt(temperature) * ((-0.9677034329307728d0) +     &
            0.5744337660319633d0 * log(temperature) + (-0.12573711513502717d0) &
            * log(temperature)**2d0 + 0.012123569772981811d0 *                 &
            log(temperature)**3d0 + (-0.00043178207579080275d0) *              &
            log(temperature)**4d0)
        conductivities(2) = sqrt(temperature) * ((-0.19161479586118832d0) +    &
            0.09553836072736534d0 * log(temperature) +                         &
            (-0.016523473530066796d0) * log(temperature)**2d0 +                &
            0.001296639230099791d0 * log(temperature)**3d0 +                   &
            (-3.721551758450998d-05) * log(temperature)**4d0)
        conductivities(3) = sqrt(temperature) * (0.027926115403805702d0 +      &
            (-0.01635662537398194d0) * log(temperature) +                      &
            0.003849088291642481d0 * log(temperature)**2d0 +                   &
            (-0.0003927858198444532d0) * log(temperature)**3d0 +               &
            1.5080612326900335d-05 * log(temperature)**4d0)
        conductivities(4) = sqrt(temperature) * (0.10689552101591863d0 +       &
            (-0.0637671134374908d0) * log(temperature) + 0.014217756178667409d0&
            * log(temperature)**2d0 + (-0.0013908411531344694d0) *             &
            log(temperature)**3d0 + 5.0917443888674d-05 * log(temperature)**4d0)
        conductivities(5) = sqrt(temperature) * ((-0.25832873556045055d0) +    &
            0.1572393466102654d0 * log(temperature) + (-0.03525683018044662d0) &
            * log(temperature)**2d0 + 0.0034830937850860704d0 *                &
            log(temperature)**3d0 + (-0.00012701428984056754d0) *              &
            log(temperature)**4d0)
        conductivities(6) = sqrt(temperature) * ((-0.40448952246566766d0) +    &
            0.25166528584121395d0 * log(temperature) +                         &
            (-0.058238000281152676d0) * log(temperature)**2d0 +                &
            0.00593090365881573d0 * log(temperature)**3d0 +                    &
            (-0.00022233754286874003d0) * log(temperature)**4d0)
        conductivities(7) = sqrt(temperature) * ((-0.024624895175065165d0) +   &
            0.015057513850280977d0 * log(temperature) +                        &
            (-0.0034683041568395233d0) * log(temperature)**2d0 +               &
            0.00036636327985582854d0 * log(temperature)**3d0 +                 &
            (-1.389972362350799d-05) * log(temperature)**4d0)
        conductivities(8) = sqrt(temperature) * (0.006133728726525739d0 +      &
            (-0.001288424174680351d0) * log(temperature) +                     &
            (-0.0003832196624518482d0) * log(temperature)**2d0 +               &
            0.00012840152692768084d0 * log(temperature)**3d0 +                 &
            (-7.613061191922738d-06) * log(temperature)**4d0)
        conductivities(9) = sqrt(temperature) * (0.0026129214099031247d0 +     &
            0.0015932386447066321d0 * log(temperature) +                       &
            (-0.0009842775277416863d0) * log(temperature)**2d0 +               &
            0.00016507154037215444d0 * log(temperature)**3d0 +                 &
            (-8.29731531537941d-06) * log(temperature)**4d0)
        conductivities(10) = sqrt(temperature) * ((-0.0119088398319493d0) +    &
            0.006044282201818738d0 * log(temperature) +                        &
            (-0.0010691302903978483d0) * log(temperature)**2d0 +               &
            8.593637639025848d-05 * log(temperature)**3d0 +                    &
            (-2.539776667061472d-06) * log(temperature)**4d0)

    end subroutine get_species_thermal_conductivities

    subroutine get_species_binary_mass_diffusivities(temperature, diffusivities)

        GPU_ROUTINE(get_species_binary_mass_diffusivities)

        real(dp), intent(in) :: temperature
        real(dp), intent(out), dimension(10, 10)&
            :: diffusivities

        diffusivities(1, 1) = sqrt(temperature) * temperature *                &
            ((-0.006865233575642219d0) + 0.0045279883627338076d0 *             &
            log(temperature) + (-0.0008654211810261634d0) *                    &
            log(temperature)**2d0 + 7.992970847127505d-05 *                    &
            log(temperature)**3d0 + (-2.6377961782764532d-06) *                &
            log(temperature)**4d0)
        diffusivities(1, 2) = sqrt(temperature) * temperature *                &
            ((-0.02494681207468867d0) + 0.013946115930419296d0 *               &
            log(temperature) + (-0.0026011928734115094d0) *                    &
            log(temperature)**2d0 + 0.00022520946625262844d0 *                 &
            log(temperature)**3d0 + (-7.13154352858658d-06) *                  &
            log(temperature)**4d0)
        diffusivities(1, 3) = sqrt(temperature) * temperature *                &
            ((-0.009169918908865838d0) + 0.005424364464401969d0 *              &
            log(temperature) + (-0.001032209527843199d0) *                     &
            log(temperature)**2d0 + 9.207446814379493d-05 *                    &
            log(temperature)**3d0 + (-2.9857817128544263d-06) *                &
            log(temperature)**4d0)
        diffusivities(1, 4) = sqrt(temperature) * temperature *                &
            ((-0.008078545001672823d0) + 0.004640788076930059d0 *              &
            log(temperature) + (-0.0008759648030254944d0) *                    &
            log(temperature)**2d0 + 7.705769878335738d-05 *                    &
            log(temperature)**3d0 + (-2.473269089069558d-06) *                 &
            log(temperature)**4d0)
        diffusivities(1, 5) = sqrt(temperature) * temperature *                &
            ((-0.009139457743539517d0) + 0.005406345497781689d0 *              &
            log(temperature) + (-0.0010287806747209332d0) *                    &
            log(temperature)**2d0 + 9.176860986688295d-05 *                    &
            log(temperature)**3d0 + (-2.975863371011668d-06) *                 &
            log(temperature)**4d0)
        diffusivities(1, 6) = sqrt(temperature) * temperature *                &
            ((-0.009701326278444037d0) + 0.004014323899384872d0 *              &
            log(temperature) + (-0.0004679109587932634d0) *                    &
            log(temperature)**2d0 + 1.9380852658722403d-05 *                   &
            log(temperature)**3d0 + 9.241023547807315d-08 *                    &
            log(temperature)**4d0)
        diffusivities(1, 7) = sqrt(temperature) * temperature *                &
            ((-0.008071230238491819d0) + 0.004636586049739328d0 *              &
            log(temperature) + (-0.0008751716558578258d0) *                    &
            log(temperature)**2d0 + 7.698792646452423d-05 *                    &
            log(temperature)**3d0 + (-2.4710296539176154d-06) *                &
            log(temperature)**4d0)
        diffusivities(1, 8) = sqrt(temperature) * temperature *                &
            ((-0.008064342962531394d0) + 0.004632629596176147d0 *              &
            log(temperature) + (-0.0008744248615612234d0) *                    &
            log(temperature)**2d0 + 7.692223175878326d-05 *                    &
            log(temperature)**3d0 + (-2.4689210951668652d-06) *                &
            log(temperature)**4d0)
        diffusivities(1, 9) = sqrt(temperature) * temperature *                &
            ((-0.007323280548496199d0) + 0.004247486239692243d0 *              &
            log(temperature) + (-0.0008033594542376295d0) *                    &
            log(temperature)**2d0 + 7.096837773031612d-05 *                    &
            log(temperature)**3d0 + (-2.2842690699174716d-06) *                &
            log(temperature)**4d0)
        diffusivities(1, 10) = sqrt(temperature) * temperature *               &
            ((-0.009261163367810188d0) + 0.005206226859449829d0 *              &
            log(temperature) + (-0.0009753457218382934d0) *                    &
            log(temperature)**2d0 + 8.483340500241432d-05 *                    &
            log(temperature)**3d0 + (-2.69887763619359d-06) *                  &
            log(temperature)**4d0)
        diffusivities(2, 1) = sqrt(temperature) * temperature *                &
            ((-0.02494681207468867d0) + 0.013946115930419296d0 *               &
            log(temperature) + (-0.0026011928734115094d0) *                    &
            log(temperature)**2d0 + 0.00022520946625262844d0 *                 &
            log(temperature)**3d0 + (-7.13154352858658d-06) *                  &
            log(temperature)**4d0)
        diffusivities(2, 2) = sqrt(temperature) * temperature *                &
            ((-0.045124959996851836d0) + 0.02158316566126686d0 *               &
            log(temperature) + (-0.0034229936881199725d0) *                    &
            log(temperature)**2d0 + 0.00024780816599288125d0 *                 &
            log(temperature)**3d0 + (-6.298684234595243d-06) *                 &
            log(temperature)**4d0)
        diffusivities(2, 3) = sqrt(temperature) * temperature *                &
            ((-0.02669027788746811d0) + 0.01393161130704746d0 *                &
            log(temperature) + (-0.0024745908346391456d0) *                    &
            log(temperature)**2d0 + 0.00020278396526864916d0 *                 &
            log(temperature)**3d0 + (-6.077989307489837d-06) *                 &
            log(temperature)**4d0)
        diffusivities(2, 4) = sqrt(temperature) * temperature *                &
            ((-0.0197891905857488d0) + 0.009984564215031065d0 *                &
            log(temperature) + (-0.001708796464270654d0) *                     &
            log(temperature)**2d0 + 0.00013470040249434945d0 *                 &
            log(temperature)**3d0 + (-3.8579047024414366d-06) *                &
            log(temperature)**4d0)
        diffusivities(2, 5) = sqrt(temperature) * temperature *                &
            ((-0.02664335652107393d0) + 0.013907119608566644d0 *               &
            log(temperature) + (-0.0024702405171343122d0) *                    &
            log(temperature)**2d0 + 0.00020242747213820796d0 *                 &
            log(temperature)**3d0 + (-6.067304234682698d-06) *                 &
            log(temperature)**4d0)
        diffusivities(2, 6) = sqrt(temperature) * temperature *                &
            (0.031782681005665316d0 + (-0.021944709939274037d0) *              &
            log(temperature) + 0.005452754910070711d0 * log(temperature)**2d0 +&
            (-0.0005578687529700277d0) * log(temperature)**3d0 +               &
            2.081257384413998d-05 * log(temperature)**4d0)
        diffusivities(2, 7) = sqrt(temperature) * temperature *                &
            ((-0.019779959886290468d0) + 0.00997990689915249d0 *               &
            log(temperature) + (-0.0017079993934377737d0) *                    &
            log(temperature)**2d0 + 0.00013463757127695766d0 *                 &
            log(temperature)**3d0 + (-3.856105176642467d-06) *                 &
            log(temperature)**4d0)
        diffusivities(2, 8) = sqrt(temperature) * temperature *                &
            ((-0.019771272353958636d0) + 0.009975523636289315d0 *              &
            log(temperature) + (-0.0017072492250904217d0) *                    &
            log(temperature)**2d0 + 0.00013457843727251797d0 *                 &
            log(temperature)**3d0 + (-3.8544115413778065d-06) *                &
            log(temperature)**4d0)
        diffusivities(2, 9) = sqrt(temperature) * temperature *                &
            ((-0.01855279389840048d0) + 0.009455038189584385d0 *               &
            log(temperature) + (-0.0016339242326576294d0) *                    &
            log(temperature)**2d0 + 0.00013016014962606722d0 *                 &
            log(temperature)**3d0 + (-3.7729752245777146d-06) *                &
            log(temperature)**4d0)
        diffusivities(2, 10) = sqrt(temperature) * temperature *               &
            ((-0.019533868132500434d0) + 0.00948583727561595d0 *               &
            log(temperature) + (-0.001542036766965509d0) *                     &
            log(temperature)**2d0 + 0.00011488762614746123d0 *                 &
            log(temperature)**3d0 + (-3.0508636500400845d-06) *                &
            log(temperature)**4d0)
        diffusivities(3, 1) = sqrt(temperature) * temperature *                &
            ((-0.009169918908865838d0) + 0.005424364464401969d0 *              &
            log(temperature) + (-0.001032209527843199d0) *                     &
            log(temperature)**2d0 + 9.207446814379493d-05 *                    &
            log(temperature)**3d0 + (-2.9857817128544263d-06) *                &
            log(temperature)**4d0)
        diffusivities(3, 2) = sqrt(temperature) * temperature *                &
            ((-0.02669027788746811d0) + 0.01393161130704746d0 *                &
            log(temperature) + (-0.0024745908346391456d0) *                    &
            log(temperature)**2d0 + 0.00020278396526864916d0 *                 &
            log(temperature)**3d0 + (-6.077989307489837d-06) *                 &
            log(temperature)**4d0)
        diffusivities(3, 3) = sqrt(temperature) * temperature *                &
            ((-0.0062006262424959d0) + 0.003419456314967783d0 *                &
            log(temperature) + (-0.0006330719706051935d0) *                    &
            log(temperature)**2d0 + 5.4312659305034644d-05 *                   &
            log(temperature)**3d0 + (-1.705596561868811d-06) *                 &
            log(temperature)**4d0)
        diffusivities(3, 4) = sqrt(temperature) * temperature *                &
            ((-0.004562289268919991d0) + 0.002451084127434132d0 *              &
            log(temperature) + (-0.0004462621814787623d0) *                    &
            log(temperature)**2d0 + 3.755034814785211d-05 *                    &
            log(temperature)**3d0 + (-1.157471595814254d-06) *                 &
            log(temperature)**4d0)
        diffusivities(3, 5) = sqrt(temperature) * temperature *                &
            ((-0.006108057939050364d0) + 0.0033684077180350324d0 *             &
            log(temperature) + (-0.0006236209255029426d0) *                    &
            log(temperature)**2d0 + 5.3501833022145683d-05 *                   &
            log(temperature)**3d0 + (-1.6801339434275782d-06) *                &
            log(temperature)**4d0)
        diffusivities(3, 6) = sqrt(temperature) * temperature *                &
            ((-0.0005572856088396081d0) + (-0.0005667709603523218d0) *         &
            log(temperature) + 0.00032310829309159153d0 * log(temperature)**2d0&
            + (-4.306813608092417d-05) * log(temperature)**3d0 +               &
            1.8913297388084905d-06 * log(temperature)**4d0)
        diffusivities(3, 7) = sqrt(temperature) * temperature *                &
            ((-0.004539007888638684d0) + 0.00243857623538354d0 *               &
            log(temperature) + (-0.0004439849037909149d0) *                    &
            log(temperature)**2d0 + 3.735872857177228d-05 *                    &
            log(temperature)**3d0 + (-1.151565013650596d-06) *                 &
            log(temperature)**4d0)
        diffusivities(3, 8) = sqrt(temperature) * temperature *                &
            ((-0.00451699685165708d0) + 0.0024267508336622045d0 *              &
            log(temperature) + (-0.00044183188525111184d0) *                   &
            log(temperature)**2d0 + 3.717756467509114d-05 *                    &
            log(temperature)**3d0 + (-1.145980722826736d-06) *                 &
            log(temperature)**4d0)
        diffusivities(3, 9) = sqrt(temperature) * temperature *                &
            ((-0.004334076415428599d0) + 0.002348775108467442d0 *              &
            log(temperature) + (-0.00043025471406031725d0) *                   &
            log(temperature)**2d0 + 3.6453305767038437d-05 *                   &
            log(temperature)**3d0 + (-1.1313281839216372d-06) *                &
            log(temperature)**4d0)
        diffusivities(3, 10) = sqrt(temperature) * temperature *               &
            ((-0.004744228835032366d0) + 0.00249038918143051d0 *               &
            log(temperature) + (-0.0004444143922846845d0) *                    &
            log(temperature)**2d0 + 3.660331650184938d-05 *                    &
            log(temperature)**3d0 + (-1.1029546462912555d-06) *                &
            log(temperature)**4d0)
        diffusivities(4, 1) = sqrt(temperature) * temperature *                &
            ((-0.008078545001672823d0) + 0.004640788076930059d0 *              &
            log(temperature) + (-0.0008759648030254944d0) *                    &
            log(temperature)**2d0 + 7.705769878335738d-05 *                    &
            log(temperature)**3d0 + (-2.473269089069558d-06) *                 &
            log(temperature)**4d0)
        diffusivities(4, 2) = sqrt(temperature) * temperature *                &
            ((-0.0197891905857488d0) + 0.009984564215031065d0 *                &
            log(temperature) + (-0.001708796464270654d0) *                     &
            log(temperature)**2d0 + 0.00013470040249434945d0 *                 &
            log(temperature)**3d0 + (-3.8579047024414366d-06) *                &
            log(temperature)**4d0)
        diffusivities(4, 3) = sqrt(temperature) * temperature *                &
            ((-0.004562289268919991d0) + 0.002451084127434132d0 *              &
            log(temperature) + (-0.0004462621814787623d0) *                    &
            log(temperature)**2d0 + 3.755034814785211d-05 *                    &
            log(temperature)**3d0 + (-1.157471595814254d-06) *                 &
            log(temperature)**4d0)
        diffusivities(4, 4) = sqrt(temperature) * temperature *                &
            ((-0.003127289937114834d0) + 0.001633232188528128d0 *              &
            log(temperature) + (-0.00029023244731756897d0) *                   &
            log(temperature)**2d0 + 2.3795154190948053d-05 *                   &
            log(temperature)**3d0 + (-7.135754459081666d-07) *                 &
            log(temperature)**4d0)
        diffusivities(4, 5) = sqrt(temperature) * temperature *                &
            ((-0.004471245659028124d0) + 0.002402171063409123d0 *              &
            log(temperature) + (-0.00043735671372650256d0) *                   &
            log(temperature)**2d0 + 3.6801005209105434d-05 *                   &
            log(temperature)**3d0 + (-1.1343734566512901d-06) *                &
            log(temperature)**4d0)
        diffusivities(4, 6) = sqrt(temperature) * temperature *                &
            (0.0044168550074850646d0 + (-0.0031775743855656643d0) *            &
            log(temperature) + 0.0008133764546254836d0 * log(temperature)**2d0 &
            + (-8.454231010204543d-05) * log(temperature)**3d0 +               &
            3.1913607821347743d-06 * log(temperature)**4d0)
        diffusivities(4, 7) = sqrt(temperature) * temperature *                &
            ((-0.003103321303249144d0) + 0.001620714531026103d0 *              &
            log(temperature) + (-0.0002880080052593471d0) *                    &
            log(temperature)**2d0 + 2.3612779882882127d-05 *                   &
            log(temperature)**3d0 + (-7.081063563979706d-07) *                 &
            log(temperature)**4d0)
        diffusivities(4, 8) = sqrt(temperature) * temperature *                &
            ((-0.00308060305221495d0) + 0.0016088498879638456d0 *              &
            log(temperature) + (-0.00028589960670053025d0) *                   &
            log(temperature)**2d0 + 2.3439919579816093d-05 *                   &
            log(temperature)**3d0 + (-7.029225754125662d-07) *                 &
            log(temperature)**4d0)
        diffusivities(4, 9) = sqrt(temperature) * temperature *                &
            ((-0.0030300534762843713d0) + 0.0015963119455998603d0 *            &
            log(temperature) + (-0.0002855827123818112d0) *                    &
            log(temperature)**2d0 + 2.3589003022680055d-05 *                   &
            log(temperature)**3d0 + (-7.128139183189626d-07) *                 &
            log(temperature)**4d0)
        diffusivities(4, 10) = sqrt(temperature) * temperature *               &
            ((-0.003033560398652087d0) + 0.0015400149915514032d0 *             &
            log(temperature) + (-0.00026507212631149977d0) *                   &
            log(temperature)**2d0 + 2.102682747839714d-05 *                    &
            log(temperature)**3d0 + (-6.065272250095099d-07) *                 &
            log(temperature)**4d0)
        diffusivities(5, 1) = sqrt(temperature) * temperature *                &
            ((-0.009139457743539517d0) + 0.005406345497781689d0 *              &
            log(temperature) + (-0.0010287806747209332d0) *                    &
            log(temperature)**2d0 + 9.176860986688295d-05 *                    &
            log(temperature)**3d0 + (-2.975863371011668d-06) *                 &
            log(temperature)**4d0)
        diffusivities(5, 2) = sqrt(temperature) * temperature *                &
            ((-0.02664335652107393d0) + 0.013907119608566644d0 *               &
            log(temperature) + (-0.0024702405171343122d0) *                    &
            log(temperature)**2d0 + 0.00020242747213820796d0 *                 &
            log(temperature)**3d0 + (-6.067304234682698d-06) *                 &
            log(temperature)**4d0)
        diffusivities(5, 3) = sqrt(temperature) * temperature *                &
            ((-0.006108057939050364d0) + 0.0033684077180350324d0 *             &
            log(temperature) + (-0.0006236209255029426d0) *                    &
            log(temperature)**2d0 + 5.3501833022145683d-05 *                   &
            log(temperature)**3d0 + (-1.6801339434275782d-06) *                &
            log(temperature)**4d0)
        diffusivities(5, 4) = sqrt(temperature) * temperature *                &
            ((-0.004471245659028124d0) + 0.002402171063409123d0 *              &
            log(temperature) + (-0.00043735671372650256d0) *                   &
            log(temperature)**2d0 + 3.6801005209105434d-05 *                   &
            log(temperature)**3d0 + (-1.1343734566512901d-06) *                &
            log(temperature)**4d0)
        diffusivities(5, 5) = sqrt(temperature) * temperature *                &
            ((-0.006014064995815134d0) + 0.00331657347569469d0 *               &
            log(temperature) + (-0.0006140244274271003d0) *                    &
            log(temperature)**2d0 + 5.267852800350607d-05 *                    &
            log(temperature)**3d0 + (-1.6542794515412925d-06) *                &
            log(temperature)**4d0)
        diffusivities(5, 6) = sqrt(temperature) * temperature *                &
            ((-0.000548468903721082d0) + (-0.0005578041893808191d0) *          &
            log(temperature) + 0.00031799646085823745d0 * log(temperature)**2d0&
            + (-4.238676364026507d-05) * log(temperature)**3d0 +               &
            1.8614073860548128d-06 * log(temperature)**4d0)
        diffusivities(5, 7) = sqrt(temperature) * temperature *                &
            ((-0.004447487715960617d0) + 0.002389407138606764d0 *              &
            log(temperature) + (-0.0004350328208567018d0) *                    &
            log(temperature)**2d0 + 3.6605463238605725d-05 *                   &
            log(temperature)**3d0 + (-1.1283459685498635d-06) *                &
            log(temperature)**4d0)
        diffusivities(5, 8) = sqrt(temperature) * temperature *                &
            ((-0.0044250214607720354d0) + 0.0023773371714804733d0 *            &
            log(temperature) + (-0.0004328352749625562d0) *                    &
            log(temperature)**2d0 + 3.642055262591118d-05 *                    &
            log(temperature)**3d0 + (-1.122646186989161d-06) *                 &
            log(temperature)**4d0)
        diffusivities(5, 9) = sqrt(temperature) * temperature *                &
            ((-0.0042515394211532255d0) + 0.0023040456623065605d0 *            &
            log(temperature) + (-0.00042206105814221185d0) *                   &
            log(temperature)**2d0 + 3.575909874322452d-05 *                    &
            log(temperature)**3d0 + (-1.1097834719954818d-06) *                &
            log(temperature)**4d0)
        diffusivities(5, 10) = sqrt(temperature) * temperature *               &
            ((-0.004642753018435071d0) + 0.0024371214566590916d0 *             &
            log(temperature) + (-0.0004349086717689084d0) *                    &
            log(temperature)**2d0 + 3.582039654543644d-05 *                    &
            log(temperature)**3d0 + (-1.0793631992260125d-06) *                &
            log(temperature)**4d0)
        diffusivities(6, 1) = sqrt(temperature) * temperature *                &
            ((-0.009701326278444037d0) + 0.004014323899384872d0 *              &
            log(temperature) + (-0.0004679109587932634d0) *                    &
            log(temperature)**2d0 + 1.9380852658722403d-05 *                   &
            log(temperature)**3d0 + 9.241023547807315d-08 *                    &
            log(temperature)**4d0)
        diffusivities(6, 2) = sqrt(temperature) * temperature *                &
            (0.031782681005665316d0 + (-0.021944709939274037d0) *              &
            log(temperature) + 0.005452754910070711d0 * log(temperature)**2d0 +&
            (-0.0005578687529700277d0) * log(temperature)**3d0 +               &
            2.081257384413998d-05 * log(temperature)**4d0)
        diffusivities(6, 3) = sqrt(temperature) * temperature *                &
            ((-0.0005572856088396081d0) + (-0.0005667709603523218d0) *         &
            log(temperature) + 0.00032310829309159153d0 * log(temperature)**2d0&
            + (-4.306813608092417d-05) * log(temperature)**3d0 +               &
            1.8913297388084905d-06 * log(temperature)**4d0)
        diffusivities(6, 4) = sqrt(temperature) * temperature *                &
            (0.0044168550074850646d0 + (-0.0031775743855656643d0) *            &
            log(temperature) + 0.0008133764546254836d0 * log(temperature)**2d0 &
            + (-8.454231010204543d-05) * log(temperature)**3d0 +               &
            3.1913607821347743d-06 * log(temperature)**4d0)
        diffusivities(6, 5) = sqrt(temperature) * temperature *                &
            ((-0.000548468903721082d0) + (-0.0005578041893808191d0) *          &
            log(temperature) + 0.00031799646085823745d0 * log(temperature)**2d0&
            + (-4.238676364026507d-05) * log(temperature)**3d0 +               &
            1.8614073860548128d-06 * log(temperature)**4d0)
        diffusivities(6, 6) = sqrt(temperature) * temperature *                &
            (0.008153691915940649d0 + (-0.003950287262275618d0) *              &
            log(temperature) + 0.0006415133652182064d0 * log(temperature)**2d0 &
            + (-3.490182562127687d-05) * log(temperature)**3d0 +               &
            3.2184246744576303d-07 * log(temperature)**4d0)
        diffusivities(6, 7) = sqrt(temperature) * temperature *                &
            (0.00210211513073036d0 + (-0.001828149840369463d0) *               &
            log(temperature) + 0.0005252030760248455d0 * log(temperature)**2d0 &
            + (-5.765803219982472d-05) * log(temperature)**3d0 +               &
            2.2618777599206966d-06 * log(temperature)**4d0)
        diffusivities(6, 8) = sqrt(temperature) * temperature *                &
            (0.002091088185944443d0 + (-0.0018185600195909572d0) *             &
            log(temperature) + 0.000522448048367193d0 * log(temperature)**2d0 +&
            (-5.735557876673813d-05) * log(temperature)**3d0 +                 &
            2.2500127574630997d-06 * log(temperature)**4d0)
        diffusivities(6, 9) = sqrt(temperature) * temperature *                &
            (0.003278860518160964d0 + (-0.002505478451489052d0) *              &
            log(temperature) + 0.0006678165711086637d0 * log(temperature)**2d0 &
            + (-7.083588486066636d-05) * log(temperature)**3d0 +               &
            2.713492698249412d-06 * log(temperature)**4d0)
        diffusivities(6, 10) = sqrt(temperature) * temperature *               &
            (0.004679237345282072d0 + (-0.0033198362412594487d0) *             &
            log(temperature) + 0.0008414347220670881d0 * log(temperature)**2d0 &
            + (-8.700624860902155d-05) * log(temperature)**3d0 +               &
            3.2717951994463044d-06 * log(temperature)**4d0)
        diffusivities(7, 1) = sqrt(temperature) * temperature *                &
            ((-0.008071230238491819d0) + 0.004636586049739328d0 *              &
            log(temperature) + (-0.0008751716558578258d0) *                    &
            log(temperature)**2d0 + 7.698792646452423d-05 *                    &
            log(temperature)**3d0 + (-2.4710296539176154d-06) *                &
            log(temperature)**4d0)
        diffusivities(7, 2) = sqrt(temperature) * temperature *                &
            ((-0.019779959886290468d0) + 0.00997990689915249d0 *               &
            log(temperature) + (-0.0017079993934377737d0) *                    &
            log(temperature)**2d0 + 0.00013463757127695766d0 *                 &
            log(temperature)**3d0 + (-3.856105176642467d-06) *                 &
            log(temperature)**4d0)
        diffusivities(7, 3) = sqrt(temperature) * temperature *                &
            ((-0.004539007888638684d0) + 0.00243857623538354d0 *               &
            log(temperature) + (-0.0004439849037909149d0) *                    &
            log(temperature)**2d0 + 3.735872857177228d-05 *                    &
            log(temperature)**3d0 + (-1.151565013650596d-06) *                 &
            log(temperature)**4d0)
        diffusivities(7, 4) = sqrt(temperature) * temperature *                &
            ((-0.003103321303249144d0) + 0.001620714531026103d0 *              &
            log(temperature) + (-0.0002880080052593471d0) *                    &
            log(temperature)**2d0 + 2.3612779882882127d-05 *                   &
            log(temperature)**3d0 + (-7.081063563979706d-07) *                 &
            log(temperature)**4d0)
        diffusivities(7, 5) = sqrt(temperature) * temperature *                &
            ((-0.004447487715960617d0) + 0.002389407138606764d0 *              &
            log(temperature) + (-0.0004350328208567018d0) *                    &
            log(temperature)**2d0 + 3.6605463238605725d-05 *                   &
            log(temperature)**3d0 + (-1.1283459685498635d-06) *                &
            log(temperature)**4d0)
        diffusivities(7, 6) = sqrt(temperature) * temperature *                &
            (0.00210211513073036d0 + (-0.001828149840369463d0) *               &
            log(temperature) + 0.0005252030760248455d0 * log(temperature)**2d0 &
            + (-5.765803219982472d-05) * log(temperature)**3d0 +               &
            2.2618777599206966d-06 * log(temperature)**4d0)
        diffusivities(7, 7) = sqrt(temperature) * temperature *                &
            ((-0.003079166100009398d0) + 0.001608099437367917d0 *              &
            log(temperature) + (-0.00028576624837283745d0) *                   &
            log(temperature)**2d0 + 2.342898599191487d-05 *                    &
            log(temperature)**3d0 + (-7.025946960546925d-07) *                 &
            log(temperature)**4d0)
        diffusivities(7, 8) = sqrt(temperature) * temperature *                &
            ((-0.003056268300616846d0) + 0.0015961410248858012d0 *             &
            log(temperature) + (-0.00028364118658053136d0) *                   &
            log(temperature)**2d0 + 2.3254759527938263d-05 *                   &
            log(temperature)**3d0 + (-6.973699462686496d-07) *                 &
            log(temperature)**4d0)
        diffusivities(7, 9) = sqrt(temperature) * temperature *                &
            ((-0.003008377376096099d0) + 0.0015848924053389168d0 *             &
            log(temperature) + (-0.00028353973870676223d0) *                   &
            log(temperature)**2d0 + 2.3420254320084756d-05 *                   &
            log(temperature)**3d0 + (-7.077146598297605d-07) *                 &
            log(temperature)**4d0)
        diffusivities(7, 10) = sqrt(temperature) * temperature *               &
            ((-0.0030077293927809746d0) + 0.0015269016425293164d0 *            &
            log(temperature) + (-0.0002628150162657031d0) *                    &
            log(temperature)**2d0 + 2.0847782385312934d-05 *                   &
            log(temperature)**3d0 + (-6.013625978891741d-07) *                 &
            log(temperature)**4d0)
        diffusivities(8, 1) = sqrt(temperature) * temperature *                &
            ((-0.008064342962531394d0) + 0.004632629596176147d0 *              &
            log(temperature) + (-0.0008744248615612234d0) *                    &
            log(temperature)**2d0 + 7.692223175878326d-05 *                    &
            log(temperature)**3d0 + (-2.4689210951668652d-06) *                &
            log(temperature)**4d0)
        diffusivities(8, 2) = sqrt(temperature) * temperature *                &
            ((-0.019771272353958636d0) + 0.009975523636289315d0 *              &
            log(temperature) + (-0.0017072492250904217d0) *                    &
            log(temperature)**2d0 + 0.00013457843727251797d0 *                 &
            log(temperature)**3d0 + (-3.8544115413778065d-06) *                &
            log(temperature)**4d0)
        diffusivities(8, 3) = sqrt(temperature) * temperature *                &
            ((-0.00451699685165708d0) + 0.0024267508336622045d0 *              &
            log(temperature) + (-0.00044183188525111184d0) *                   &
            log(temperature)**2d0 + 3.717756467509114d-05 *                    &
            log(temperature)**3d0 + (-1.145980722826736d-06) *                 &
            log(temperature)**4d0)
        diffusivities(8, 4) = sqrt(temperature) * temperature *                &
            ((-0.00308060305221495d0) + 0.0016088498879638456d0 *              &
            log(temperature) + (-0.00028589960670053025d0) *                   &
            log(temperature)**2d0 + 2.3439919579816093d-05 *                   &
            log(temperature)**3d0 + (-7.029225754125662d-07) *                 &
            log(temperature)**4d0)
        diffusivities(8, 5) = sqrt(temperature) * temperature *                &
            ((-0.0044250214607720354d0) + 0.0023773371714804733d0 *            &
            log(temperature) + (-0.0004328352749625562d0) *                    &
            log(temperature)**2d0 + 3.642055262591118d-05 *                    &
            log(temperature)**3d0 + (-1.122646186989161d-06) *                 &
            log(temperature)**4d0)
        diffusivities(8, 6) = sqrt(temperature) * temperature *                &
            (0.002091088185944443d0 + (-0.0018185600195909572d0) *             &
            log(temperature) + 0.000522448048367193d0 * log(temperature)**2d0 +&
            (-5.735557876673813d-05) * log(temperature)**3d0 +                 &
            2.2500127574630997d-06 * log(temperature)**4d0)
        diffusivities(8, 7) = sqrt(temperature) * temperature *                &
            ((-0.003056268300616846d0) + 0.0015961410248858012d0 *             &
            log(temperature) + (-0.00028364118658053136d0) *                   &
            log(temperature)**2d0 + 2.3254759527938263d-05 *                   &
            log(temperature)**3d0 + (-6.973699462686496d-07) *                 &
            log(temperature)**4d0)
        diffusivities(8, 8) = sqrt(temperature) * temperature *                &
            ((-0.0030331976492767466d0) + 0.001584092340198947d0 *             &
            log(temperature) + (-0.0002815000830266939d0) *                    &
            log(temperature)**2d0 + 2.3079217855508333d-05 *                   &
            log(temperature)**3d0 + (-6.921057556627795d-07) *                 &
            log(temperature)**4d0)
        diffusivities(8, 9) = sqrt(temperature) * temperature *                &
            ((-0.0029878424650156953d0) + 0.001574074073545046d0 *             &
            log(temperature) + (-0.00028160432217010605d0) *                   &
            log(temperature)**2d0 + 2.326038978853447d-05 *                    &
            log(temperature)**3d0 + (-7.028838637581094d-07) *                 &
            log(temperature)**4d0)
        diffusivities(8, 10) = sqrt(temperature) * temperature *               &
            ((-0.002983225217840147d0) + 0.0015144618714995521d0 *             &
            log(temperature) + (-0.00026067384453962796d0) *                   &
            log(temperature)**2d0 + 2.0677934091146085d-05 *                   &
            log(temperature)**3d0 + (-5.964632561012826d-07) *                 &
            log(temperature)**4d0)
        diffusivities(9, 1) = sqrt(temperature) * temperature *                &
            ((-0.007323280548496199d0) + 0.004247486239692243d0 *              &
            log(temperature) + (-0.0008033594542376295d0) *                    &
            log(temperature)**2d0 + 7.096837773031612d-05 *                    &
            log(temperature)**3d0 + (-2.2842690699174716d-06) *                &
            log(temperature)**4d0)
        diffusivities(9, 2) = sqrt(temperature) * temperature *                &
            ((-0.01855279389840048d0) + 0.009455038189584385d0 *               &
            log(temperature) + (-0.0016339242326576294d0) *                    &
            log(temperature)**2d0 + 0.00013016014962606722d0 *                 &
            log(temperature)**3d0 + (-3.7729752245777146d-06) *                &
            log(temperature)**4d0)
        diffusivities(9, 3) = sqrt(temperature) * temperature *                &
            ((-0.004334076415428599d0) + 0.002348775108467442d0 *              &
            log(temperature) + (-0.00043025471406031725d0) *                   &
            log(temperature)**2d0 + 3.6453305767038437d-05 *                   &
            log(temperature)**3d0 + (-1.1313281839216372d-06) *                &
            log(temperature)**4d0)
        diffusivities(9, 4) = sqrt(temperature) * temperature *                &
            ((-0.0030300534762843713d0) + 0.0015963119455998603d0 *            &
            log(temperature) + (-0.0002855827123818112d0) *                    &
            log(temperature)**2d0 + 2.3589003022680055d-05 *                   &
            log(temperature)**3d0 + (-7.128139183189626d-07) *                 &
            log(temperature)**4d0)
        diffusivities(9, 5) = sqrt(temperature) * temperature *                &
            ((-0.0042515394211532255d0) + 0.0023040456623065605d0 *            &
            log(temperature) + (-0.00042206105814221185d0) *                   &
            log(temperature)**2d0 + 3.575909874322452d-05 *                    &
            log(temperature)**3d0 + (-1.1097834719954818d-06) *                &
            log(temperature)**4d0)
        diffusivities(9, 6) = sqrt(temperature) * temperature *                &
            (0.003278860518160964d0 + (-0.002505478451489052d0) *              &
            log(temperature) + 0.0006678165711086637d0 * log(temperature)**2d0 &
            + (-7.083588486066636d-05) * log(temperature)**3d0 +               &
            2.713492698249412d-06 * log(temperature)**4d0)
        diffusivities(9, 7) = sqrt(temperature) * temperature *                &
            ((-0.003008377376096099d0) + 0.0015848924053389168d0 *             &
            log(temperature) + (-0.00028353973870676223d0) *                   &
            log(temperature)**2d0 + 2.3420254320084756d-05 *                   &
            log(temperature)**3d0 + (-7.077146598297605d-07) *                 &
            log(temperature)**4d0)
        diffusivities(9, 8) = sqrt(temperature) * temperature *                &
            ((-0.0029878424650156953d0) + 0.001574074073545046d0 *             &
            log(temperature) + (-0.00028160432217010605d0) *                   &
            log(temperature)**2d0 + 2.326038978853447d-05 *                    &
            log(temperature)**3d0 + (-7.028838637581094d-07) *                 &
            log(temperature)**4d0)
        diffusivities(9, 9) = sqrt(temperature) * temperature *                &
            ((-0.0029567385229400263d0) + 0.0015724917419100686d0 *            &
            log(temperature) + (-0.000283699905787975d0) *                     &
            log(temperature)**2d0 + 2.364377574619869d-05 *                    &
            log(temperature)**3d0 + (-7.213510385737903d-07) *                 &
            log(temperature)**4d0)
        diffusivities(9, 10) = sqrt(temperature) * temperature *               &
            ((-0.003029480885270604d0) + 0.001555867765096798d0 *              &
            log(temperature) + (-0.00027139966842016733d0) *                   &
            log(temperature)**2d0 + 2.1828987550581687d-05 *                   &
            log(temperature)**3d0 + (-6.401769152654642d-07) *                 &
            log(temperature)**4d0)
        diffusivities(10, 1) = sqrt(temperature) * temperature *               &
            ((-0.009261163367810188d0) + 0.005206226859449829d0 *              &
            log(temperature) + (-0.0009753457218382934d0) *                    &
            log(temperature)**2d0 + 8.483340500241432d-05 *                    &
            log(temperature)**3d0 + (-2.69887763619359d-06) *                  &
            log(temperature)**4d0)
        diffusivities(10, 2) = sqrt(temperature) * temperature *               &
            ((-0.019533868132500434d0) + 0.00948583727561595d0 *               &
            log(temperature) + (-0.001542036766965509d0) *                     &
            log(temperature)**2d0 + 0.00011488762614746123d0 *                 &
            log(temperature)**3d0 + (-3.0508636500400845d-06) *                &
            log(temperature)**4d0)
        diffusivities(10, 3) = sqrt(temperature) * temperature *               &
            ((-0.004744228835032366d0) + 0.00249038918143051d0 *               &
            log(temperature) + (-0.0004444143922846845d0) *                    &
            log(temperature)**2d0 + 3.660331650184938d-05 *                    &
            log(temperature)**3d0 + (-1.1029546462912555d-06) *                &
            log(temperature)**4d0)
        diffusivities(10, 4) = sqrt(temperature) * temperature *               &
            ((-0.003033560398652087d0) + 0.0015400149915514032d0 *             &
            log(temperature) + (-0.00026507212631149977d0) *                   &
            log(temperature)**2d0 + 2.102682747839714d-05 *                    &
            log(temperature)**3d0 + (-6.065272250095099d-07) *                 &
            log(temperature)**4d0)
        diffusivities(10, 5) = sqrt(temperature) * temperature *               &
            ((-0.004642753018435071d0) + 0.0024371214566590916d0 *             &
            log(temperature) + (-0.0004349086717689084d0) *                    &
            log(temperature)**2d0 + 3.582039654543644d-05 *                    &
            log(temperature)**3d0 + (-1.0793631992260125d-06) *                &
            log(temperature)**4d0)
        diffusivities(10, 6) = sqrt(temperature) * temperature *               &
            (0.004679237345282072d0 + (-0.0033198362412594487d0) *             &
            log(temperature) + 0.0008414347220670881d0 * log(temperature)**2d0 &
            + (-8.700624860902155d-05) * log(temperature)**3d0 +               &
            3.2717951994463044d-06 * log(temperature)**4d0)
        diffusivities(10, 7) = sqrt(temperature) * temperature *               &
            ((-0.0030077293927809746d0) + 0.0015269016425293164d0 *            &
            log(temperature) + (-0.0002628150162657031d0) *                    &
            log(temperature)**2d0 + 2.0847782385312934d-05 *                   &
            log(temperature)**3d0 + (-6.013625978891741d-07) *                 &
            log(temperature)**4d0)
        diffusivities(10, 8) = sqrt(temperature) * temperature *               &
            ((-0.002983225217840147d0) + 0.0015144618714995521d0 *             &
            log(temperature) + (-0.00026067384453962796d0) *                   &
            log(temperature)**2d0 + 2.0677934091146085d-05 *                   &
            log(temperature)**3d0 + (-5.964632561012826d-07) *                 &
            log(temperature)**4d0)
        diffusivities(10, 9) = sqrt(temperature) * temperature *               &
            ((-0.003029480885270604d0) + 0.001555867765096798d0 *              &
            log(temperature) + (-0.00027139966842016733d0) *                   &
            log(temperature)**2d0 + 2.1828987550581687d-05 *                   &
            log(temperature)**3d0 + (-6.401769152654642d-07) *                 &
            log(temperature)**4d0)
        diffusivities(10, 10) = sqrt(temperature) * temperature *              &
            ((-0.002880304616567465d0) + 0.0014136357161162834d0 *             &
            log(temperature) + (-0.00023329433703720247d0) *                   &
            log(temperature)**2d0 + 1.7679090448114955d-05 *                   &
            log(temperature)**3d0 + (-4.808258646982689d-07) *                 &
            log(temperature)**4d0)

    end subroutine get_species_binary_mass_diffusivities

    subroutine get_mixture_viscosity_mixavg(&
        temperature, mass_fractions, mixture_viscosity_mixavg)

        GPU_ROUTINE(get_mixture_viscosity_mixavg)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: mixture_viscosity_mixavg

        real(dp) :: mix_mol_weight
        real(dp), dimension(10) :: &
            mole_fractions, viscosities, mix_rule_f

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_viscosities(temperature, viscosities)

        mix_rule_f(1) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(1) /&
            viscosities(1) * sqrt(1.0d0)))**2d0) / sqrt(16.0d0) +              &
            (mole_fractions(2) * (1d0 + sqrt(viscosities(1) / viscosities(2) * &
            sqrt(0.5d0)))**2d0) / sqrt(24.0d0) + (mole_fractions(3) * (1d0 +   &
            sqrt(viscosities(1) / viscosities(3) *                             &
            sqrt(7.936011904761905d0)))**2d0) / sqrt(9.008063003937746d0) +    &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(1) / viscosities(4) * &
            sqrt(15.87202380952381d0)))**2d0) / sqrt(8.504031501968873d0) +    &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(1) / viscosities(5) * &
            sqrt(8.436011904761905d0)))**2d0) / sqrt(8.948315399541364d0) +    &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(1) / viscosities(6) * &
            sqrt(8.936011904761905d0)))**2d0) / sqrt(8.895253955037468d0) +    &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(1) / viscosities(7) * &
            sqrt(16.37202380952381d0)))**2d0) / sqrt(8.488638429376477d0) +    &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(1) / viscosities(8) * &
            sqrt(16.87202380952381d0)))**2d0) / sqrt(8.474157699770682d0) +    &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(1) / viscosities(9) * &
            sqrt(13.895833333333332d0)))**2d0) / sqrt(8.575712143928037d0) +   &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(1) / viscosities(10) &
            * sqrt(19.816468253968257d0)))**2d0) / sqrt(8.403704630788486d0)
        mix_rule_f(2) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(2) /&
            viscosities(1) * sqrt(2.0d0)))**2d0) / sqrt(12.0d0) +              &
            (mole_fractions(2) * (1d0 + sqrt(viscosities(2) / viscosities(2) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(3) * (1d0 +   &
            sqrt(viscosities(2) / viscosities(3) *                             &
            sqrt(15.87202380952381d0)))**2d0) / sqrt(8.504031501968873d0) +    &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(2) / viscosities(4) * &
            sqrt(31.74404761904762d0)))**2d0) / sqrt(8.252015750984437d0) +    &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(2) / viscosities(5) * &
            sqrt(16.87202380952381d0)))**2d0) / sqrt(8.474157699770682d0) +    &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(2) / viscosities(6) * &
            sqrt(17.87202380952381d0)))**2d0) / sqrt(8.447626977518734d0) +    &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(2) / viscosities(7) * &
            sqrt(32.74404761904762d0)))**2d0) / sqrt(8.244319214688238d0) +    &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(2) / viscosities(8) * &
            sqrt(33.74404761904762d0)))**2d0) / sqrt(8.237078849885341d0) +    &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(2) / viscosities(9) * &
            sqrt(27.791666666666664d0)))**2d0) / sqrt(8.287856071964018d0) +   &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(2) / viscosities(10) &
            * sqrt(39.63293650793651d0)))**2d0) / sqrt(8.201852315394243d0)
        mix_rule_f(3) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(3) /&
            viscosities(1) * sqrt(0.12600787549221826d0)))**2d0) /             &
            sqrt(71.48809523809524d0) + (mole_fractions(2) * (1d0 +            &
            sqrt(viscosities(3) / viscosities(2) *                             &
            sqrt(0.06300393774610913d0)))**2d0) / sqrt(134.97619047619048d0) + &
            (mole_fractions(3) * (1d0 + sqrt(viscosities(3) / viscosities(3) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(4) * (1d0 +   &
            sqrt(viscosities(3) / viscosities(4) * sqrt(2.0d0)))**2d0) /       &
            sqrt(12.0d0) + (mole_fractions(5) * (1d0 + sqrt(viscosities(3) /   &
            viscosities(5) * sqrt(1.0630039377461091d0)))**2d0) /              &
            sqrt(15.525842300229318d0) + (mole_fractions(6) * (1d0 +           &
            sqrt(viscosities(3) / viscosities(6) *                             &
            sqrt(1.1260078754922183d0)))**2d0) / sqrt(15.104746044962532d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(3) / viscosities(7) * &
            sqrt(2.063003937746109d0)))**2d0) / sqrt(11.87784039265588d0) +    &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(3) / viscosities(8) * &
            sqrt(2.1260078754922183d0)))**2d0) / sqrt(11.762921150114659d0) +  &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(3) / viscosities(9) * &
            sqrt(1.7509844365272829d0)))**2d0) / sqrt(12.568858427928893d0) +  &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(3) / viscosities(10) &
            * sqrt(2.497031064441528d0)))**2d0) / sqrt(11.203804755944931d0)
        mix_rule_f(4) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(4) /&
            viscosities(1) * sqrt(0.06300393774610913d0)))**2d0) /             &
            sqrt(134.97619047619048d0) + (mole_fractions(2) * (1d0 +           &
            sqrt(viscosities(4) / viscosities(2) *                             &
            sqrt(0.031501968873054564d0)))**2d0) / sqrt(261.95238095238096d0) +&
            (mole_fractions(3) * (1d0 + sqrt(viscosities(4) / viscosities(3) * &
            sqrt(0.5d0)))**2d0) / sqrt(24.0d0) + (mole_fractions(4) * (1d0 +   &
            sqrt(viscosities(4) / viscosities(4) * sqrt(1.0d0)))**2d0) /       &
            sqrt(16.0d0) + (mole_fractions(5) * (1d0 + sqrt(viscosities(4) /   &
            viscosities(5) * sqrt(0.5315019688730546d0)))**2d0) /              &
            sqrt(23.051684600458636d0) + (mole_fractions(6) * (1d0 +           &
            sqrt(viscosities(4) / viscosities(6) *                             &
            sqrt(0.5630039377461091d0)))**2d0) / sqrt(22.209492089925064d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(4) / viscosities(7) * &
            sqrt(1.0315019688730545d0)))**2d0) / sqrt(15.75568078531176d0) +   &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(4) / viscosities(8) * &
            sqrt(1.0630039377461091d0)))**2d0) / sqrt(15.525842300229318d0) +  &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(4) / viscosities(9) * &
            sqrt(0.8754922182636414d0)))**2d0) / sqrt(17.137716855857786d0) +  &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(4) / viscosities(10) &
            * sqrt(1.248515532220764d0)))**2d0) / sqrt(14.407609511889863d0)
        mix_rule_f(5) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(5) /&
            viscosities(1) * sqrt(0.11853942494267065d0)))**2d0) /             &
            sqrt(75.48809523809524d0) + (mole_fractions(2) * (1d0 +            &
            sqrt(viscosities(5) / viscosities(2) *                             &
            sqrt(0.05926971247133533d0)))**2d0) / sqrt(142.97619047619048d0) + &
            (mole_fractions(3) * (1d0 + sqrt(viscosities(5) / viscosities(3) * &
            sqrt(0.9407302875286646d0)))**2d0) / sqrt(16.504031501968875d0) +  &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(5) / viscosities(4) * &
            sqrt(1.8814605750573292d0)))**2d0) / sqrt(12.252015750984437d0) +  &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(5) / viscosities(5) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(6) * (1d0 +   &
            sqrt(viscosities(5) / viscosities(6) *                             &
            sqrt(1.0592697124713353d0)))**2d0) / sqrt(15.552373022481266d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(5) / viscosities(7) * &
            sqrt(1.9407302875286645d0)))**2d0) / sqrt(12.12215960734412d0) +   &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(5) / viscosities(8) * &
            sqrt(2.0d0)))**2d0) / sqrt(12.0d0) + (mole_fractions(9) * (1d0 +   &
            sqrt(viscosities(5) / viscosities(9) *                             &
            sqrt(1.6472040924325275d0)))**2d0) / sqrt(12.856714499892911d0) +  &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(5) / viscosities(10) &
            * sqrt(2.349032751220086d0)))**2d0) / sqrt(11.405657071339174d0)
        mix_rule_f(6) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(6) /&
            viscosities(1) * sqrt(0.11190674437968359d0)))**2d0) /             &
            sqrt(79.48809523809524d0) + (mole_fractions(2) * (1d0 +            &
            sqrt(viscosities(6) / viscosities(2) *                             &
            sqrt(0.055953372189841796d0)))**2d0) / sqrt(150.97619047619048d0) +&
            (mole_fractions(3) * (1d0 + sqrt(viscosities(6) / viscosities(3) * &
            sqrt(0.8880932556203164d0)))**2d0) / sqrt(17.008063003937746d0) +  &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(6) / viscosities(4) * &
            sqrt(1.7761865112406328d0)))**2d0) / sqrt(12.504031501968873d0) +  &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(6) / viscosities(5) * &
            sqrt(0.9440466278101582d0)))**2d0) / sqrt(16.474157699770682d0) +  &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(6) / viscosities(6) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(7) * (1d0 +   &
            sqrt(viscosities(6) / viscosities(7) *                             &
            sqrt(1.8321398834304745d0)))**2d0) / sqrt(12.366478822032358d0) +  &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(6) / viscosities(8) * &
            sqrt(1.8880932556203165d0)))**2d0) / sqrt(12.237078849885341d0) +  &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(6) / viscosities(9) * &
            sqrt(1.55503746877602d0)))**2d0) / sqrt(13.144570571856928d0) +    &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(6) / viscosities(10) &
            * sqrt(2.2175964474049406d0)))**2d0) / sqrt(11.607509386733417d0)
        mix_rule_f(7) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(7) /&
            viscosities(1) * sqrt(0.061079803672059625d0)))**2d0) /            &
            sqrt(138.97619047619048d0) + (mole_fractions(2) * (1d0 +           &
            sqrt(viscosities(7) / viscosities(2) *                             &
            sqrt(0.030539901836029813d0)))**2d0) / sqrt(269.95238095238096d0) +&
            (mole_fractions(3) * (1d0 + sqrt(viscosities(7) / viscosities(3) * &
            sqrt(0.4847300490819851d0)))**2d0) / sqrt(24.50403150196887d0) +   &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(7) / viscosities(4) * &
            sqrt(0.9694600981639702d0)))**2d0) / sqrt(16.252015750984434d0) +  &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(7) / viscosities(5) * &
            sqrt(0.515269950918015d0)))**2d0) / sqrt(23.525842300229314d0) +   &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(7) / viscosities(6) * &
            sqrt(0.5458098527540447d0)))**2d0) / sqrt(22.657119067443794d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(7) / viscosities(7) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(8) * (1d0 +   &
            sqrt(viscosities(7) / viscosities(8) *                             &
            sqrt(1.03053990183603d0)))**2d0) / sqrt(15.762921150114657d0) +    &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(7) / viscosities(9) * &
            sqrt(0.8487547718596619d0)))**2d0) / sqrt(17.425572927821804d0) +  &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(7) / viscosities(10) &
            * sqrt(1.2103859904259833d0)))**2d0) / sqrt(14.609461827284104d0)
        mix_rule_f(8) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(8) /&
            viscosities(1) * sqrt(0.05926971247133533d0)))**2d0) /             &
            sqrt(142.97619047619048d0) + (mole_fractions(2) * (1d0 +           &
            sqrt(viscosities(8) / viscosities(2) *                             &
            sqrt(0.029634856235667664d0)))**2d0) / sqrt(277.95238095238096d0) +&
            (mole_fractions(3) * (1d0 + sqrt(viscosities(8) / viscosities(3) * &
            sqrt(0.4703651437643323d0)))**2d0) / sqrt(25.008063003937746d0) +  &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(8) / viscosities(4) * &
            sqrt(0.9407302875286646d0)))**2d0) / sqrt(16.504031501968875d0) +  &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(8) / viscosities(5) * &
            sqrt(0.5d0)))**2d0) / sqrt(24.0d0) + (mole_fractions(6) * (1d0 +   &
            sqrt(viscosities(8) / viscosities(6) *                             &
            sqrt(0.5296348562356676d0)))**2d0) / sqrt(23.104746044962532d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(8) / viscosities(7) * &
            sqrt(0.9703651437643322d0)))**2d0) / sqrt(16.24431921468824d0) +   &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(8) / viscosities(8) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(9) * (1d0 +   &
            sqrt(viscosities(8) / viscosities(9) *                             &
            sqrt(0.8236020462162638d0)))**2d0) / sqrt(17.713428999785823d0) +  &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(8) / viscosities(10) &
            * sqrt(1.174516375610043d0)))**2d0) / sqrt(14.811314142678349d0)
        mix_rule_f(9) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(9) /&
            viscosities(1) * sqrt(0.0719640179910045d0)))**2d0) /              &
            sqrt(119.16666666666666d0) + (mole_fractions(2) * (1d0 +           &
            sqrt(viscosities(9) / viscosities(2) *                             &
            sqrt(0.03598200899550225d0)))**2d0) / sqrt(230.33333333333331d0) + &
            (mole_fractions(3) * (1d0 + sqrt(viscosities(9) / viscosities(3) * &
            sqrt(0.5711073034911116d0)))**2d0) / sqrt(22.00787549221826d0) +   &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(9) / viscosities(4) * &
            sqrt(1.1422146069822232d0)))**2d0) / sqrt(15.00393774610913d0) +   &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(9) / viscosities(5) * &
            sqrt(0.6070893124866139d0)))**2d0) / sqrt(21.177632739460222d0) +  &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(9) / viscosities(6) * &
            sqrt(0.6430713214821161d0)))**2d0) / sqrt(20.440299750208162d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(9) / viscosities(7) * &
            sqrt(1.1781966159777255d0)))**2d0) / sqrt(14.790038174877296d0) +  &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(9) / viscosities(8) * &
            sqrt(1.2141786249732278d0)))**2d0) / sqrt(14.588816369730111d0) +  &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(9) / viscosities(9) * &
            sqrt(1.0d0)))**2d0) / sqrt(16.0d0) + (mole_fractions(10) * (1d0 +  &
            sqrt(viscosities(9) / viscosities(10) *                            &
            sqrt(1.426072677946741d0)))**2d0) / sqrt(13.609812265331664d0)
        mix_rule_f(10) = 0d0 + (mole_fractions(1) * (1d0 + sqrt(viscosities(10)&
            / viscosities(1) * sqrt(0.050463078848560695d0)))**2d0) /          &
            sqrt(166.53174603174605d0) + (mole_fractions(2) * (1d0 +           &
            sqrt(viscosities(10) / viscosities(2) *                            &
            sqrt(0.025231539424280348d0)))**2d0) / sqrt(325.0634920634921d0) + &
            (mole_fractions(3) * (1d0 + sqrt(viscosities(10) / viscosities(3) *&
            sqrt(0.40047559449311637d0)))**2d0) / sqrt(27.976248515532223d0) + &
            (mole_fractions(4) * (1d0 + sqrt(viscosities(10) / viscosities(4) *&
            sqrt(0.8009511889862327d0)))**2d0) / sqrt(17.98812425776611d0) +   &
            (mole_fractions(5) * (1d0 + sqrt(viscosities(10) / viscosities(5) *&
            sqrt(0.42570713391739673d0)))**2d0) / sqrt(26.792262009760687d0) + &
            (mole_fractions(6) * (1d0 + sqrt(viscosities(10) / viscosities(6) *&
            sqrt(0.4509386733416771d0)))**2d0) / sqrt(25.740771579239524d0) +  &
            (mole_fractions(7) * (1d0 + sqrt(viscosities(10) / viscosities(7) *&
            sqrt(0.8261827284105131d0)))**2d0) / sqrt(17.683087923407868d0) +  &
            (mole_fractions(8) * (1d0 + sqrt(viscosities(10) / viscosities(8) *&
            sqrt(0.8514142678347935d0)))**2d0) / sqrt(17.396131004880345d0) +  &
            (mole_fractions(9) * (1d0 + sqrt(viscosities(10) / viscosities(9) *&
            sqrt(0.701226533166458d0)))**2d0) / sqrt(19.408581423573928d0) +   &
            (mole_fractions(10) * (1d0 + sqrt(viscosities(10) / viscosities(10)&
            * sqrt(1.0d0)))**2d0) / sqrt(16.0d0)

        mixture_viscosity_mixavg = sum(mole_fractions*viscosities/mix_rule_f)

    end subroutine get_mixture_viscosity_mixavg

    subroutine get_mixture_thermal_conductivity_mixavg(temperature, &
        mass_fractions, mixture_thermal_conductivity_mixavg)

        GPU_ROUTINE(get_mixture_viscosity_mixavg)

        real(dp), intent(in) :: temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out) :: mixture_thermal_conductivity_mixavg

        real(dp) :: mix_mol_weight
        real(dp), dimension(10) :: mole_fractions, conductivities

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_thermal_conductivities(temperature, conductivities)

        mixture_thermal_conductivity_mixavg = 0.5*(&
            sum(mole_fractions*conductivities) + &
            1/sum(mole_fractions/conductivities))

    end subroutine get_mixture_thermal_conductivity_mixavg

    subroutine get_species_mass_diffusivities_mixavg(&
        pressure, temperature, mass_fractions, mass_diffusivities_mixavg)

        GPU_ROUTINE(get_species_mass_diffusivities_mixavg)

        real(dp), intent(in) :: pressure, temperature
        real(dp), intent(in), dimension(10) :: mass_fractions
        real(dp), intent(out), dimension(10) :: &
            mass_diffusivities_mixavg

        real(dp) :: mix_mol_weight
        real(dp), dimension(10) :: mole_fractions, x_sum, denom
        real(dp), dimension(10, 10) :: bdiff_ij

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_binary_mass_diffusivities(temperature, bdiff_ij)

        x_sum(1) = 0d0 + mole_fractions(1) / bdiff_ij(1, 1) + mole_fractions(2)&
            / bdiff_ij(2, 1) + mole_fractions(3) / bdiff_ij(3, 1) +            &
            mole_fractions(4) / bdiff_ij(4, 1) + mole_fractions(5) /           &
            bdiff_ij(5, 1) + mole_fractions(6) / bdiff_ij(6, 1) +              &
            mole_fractions(7) / bdiff_ij(7, 1) + mole_fractions(8) /           &
            bdiff_ij(8, 1) + mole_fractions(9) / bdiff_ij(9, 1) +              &
            mole_fractions(10) / bdiff_ij(10, 1)
        x_sum(2) = 0d0 + mole_fractions(1) / bdiff_ij(1, 2) + mole_fractions(2)&
            / bdiff_ij(2, 2) + mole_fractions(3) / bdiff_ij(3, 2) +            &
            mole_fractions(4) / bdiff_ij(4, 2) + mole_fractions(5) /           &
            bdiff_ij(5, 2) + mole_fractions(6) / bdiff_ij(6, 2) +              &
            mole_fractions(7) / bdiff_ij(7, 2) + mole_fractions(8) /           &
            bdiff_ij(8, 2) + mole_fractions(9) / bdiff_ij(9, 2) +              &
            mole_fractions(10) / bdiff_ij(10, 2)
        x_sum(3) = 0d0 + mole_fractions(1) / bdiff_ij(1, 3) + mole_fractions(2)&
            / bdiff_ij(2, 3) + mole_fractions(3) / bdiff_ij(3, 3) +            &
            mole_fractions(4) / bdiff_ij(4, 3) + mole_fractions(5) /           &
            bdiff_ij(5, 3) + mole_fractions(6) / bdiff_ij(6, 3) +              &
            mole_fractions(7) / bdiff_ij(7, 3) + mole_fractions(8) /           &
            bdiff_ij(8, 3) + mole_fractions(9) / bdiff_ij(9, 3) +              &
            mole_fractions(10) / bdiff_ij(10, 3)
        x_sum(4) = 0d0 + mole_fractions(1) / bdiff_ij(1, 4) + mole_fractions(2)&
            / bdiff_ij(2, 4) + mole_fractions(3) / bdiff_ij(3, 4) +            &
            mole_fractions(4) / bdiff_ij(4, 4) + mole_fractions(5) /           &
            bdiff_ij(5, 4) + mole_fractions(6) / bdiff_ij(6, 4) +              &
            mole_fractions(7) / bdiff_ij(7, 4) + mole_fractions(8) /           &
            bdiff_ij(8, 4) + mole_fractions(9) / bdiff_ij(9, 4) +              &
            mole_fractions(10) / bdiff_ij(10, 4)
        x_sum(5) = 0d0 + mole_fractions(1) / bdiff_ij(1, 5) + mole_fractions(2)&
            / bdiff_ij(2, 5) + mole_fractions(3) / bdiff_ij(3, 5) +            &
            mole_fractions(4) / bdiff_ij(4, 5) + mole_fractions(5) /           &
            bdiff_ij(5, 5) + mole_fractions(6) / bdiff_ij(6, 5) +              &
            mole_fractions(7) / bdiff_ij(7, 5) + mole_fractions(8) /           &
            bdiff_ij(8, 5) + mole_fractions(9) / bdiff_ij(9, 5) +              &
            mole_fractions(10) / bdiff_ij(10, 5)
        x_sum(6) = 0d0 + mole_fractions(1) / bdiff_ij(1, 6) + mole_fractions(2)&
            / bdiff_ij(2, 6) + mole_fractions(3) / bdiff_ij(3, 6) +            &
            mole_fractions(4) / bdiff_ij(4, 6) + mole_fractions(5) /           &
            bdiff_ij(5, 6) + mole_fractions(6) / bdiff_ij(6, 6) +              &
            mole_fractions(7) / bdiff_ij(7, 6) + mole_fractions(8) /           &
            bdiff_ij(8, 6) + mole_fractions(9) / bdiff_ij(9, 6) +              &
            mole_fractions(10) / bdiff_ij(10, 6)
        x_sum(7) = 0d0 + mole_fractions(1) / bdiff_ij(1, 7) + mole_fractions(2)&
            / bdiff_ij(2, 7) + mole_fractions(3) / bdiff_ij(3, 7) +            &
            mole_fractions(4) / bdiff_ij(4, 7) + mole_fractions(5) /           &
            bdiff_ij(5, 7) + mole_fractions(6) / bdiff_ij(6, 7) +              &
            mole_fractions(7) / bdiff_ij(7, 7) + mole_fractions(8) /           &
            bdiff_ij(8, 7) + mole_fractions(9) / bdiff_ij(9, 7) +              &
            mole_fractions(10) / bdiff_ij(10, 7)
        x_sum(8) = 0d0 + mole_fractions(1) / bdiff_ij(1, 8) + mole_fractions(2)&
            / bdiff_ij(2, 8) + mole_fractions(3) / bdiff_ij(3, 8) +            &
            mole_fractions(4) / bdiff_ij(4, 8) + mole_fractions(5) /           &
            bdiff_ij(5, 8) + mole_fractions(6) / bdiff_ij(6, 8) +              &
            mole_fractions(7) / bdiff_ij(7, 8) + mole_fractions(8) /           &
            bdiff_ij(8, 8) + mole_fractions(9) / bdiff_ij(9, 8) +              &
            mole_fractions(10) / bdiff_ij(10, 8)
        x_sum(9) = 0d0 + mole_fractions(1) / bdiff_ij(1, 9) + mole_fractions(2)&
            / bdiff_ij(2, 9) + mole_fractions(3) / bdiff_ij(3, 9) +            &
            mole_fractions(4) / bdiff_ij(4, 9) + mole_fractions(5) /           &
            bdiff_ij(5, 9) + mole_fractions(6) / bdiff_ij(6, 9) +              &
            mole_fractions(7) / bdiff_ij(7, 9) + mole_fractions(8) /           &
            bdiff_ij(8, 9) + mole_fractions(9) / bdiff_ij(9, 9) +              &
            mole_fractions(10) / bdiff_ij(10, 9)
        x_sum(10) = 0d0 + mole_fractions(1) / bdiff_ij(1, 10) +                &
            mole_fractions(2) / bdiff_ij(2, 10) + mole_fractions(3) /          &
            bdiff_ij(3, 10) + mole_fractions(4) / bdiff_ij(4, 10) +            &
            mole_fractions(5) / bdiff_ij(5, 10) + mole_fractions(6) /          &
            bdiff_ij(6, 10) + mole_fractions(7) / bdiff_ij(7, 10) +            &
            mole_fractions(8) / bdiff_ij(8, 10) + mole_fractions(9) /          &
            bdiff_ij(9, 10) + mole_fractions(10) / bdiff_ij(10, 10)

        denom(1) = x_sum(1) - &
            mole_fractions(1)/bdiff_ij(1, 1)
        denom(2) = x_sum(2) - &
            mole_fractions(2)/bdiff_ij(2, 2)
        denom(3) = x_sum(3) - &
            mole_fractions(3)/bdiff_ij(3, 3)
        denom(4) = x_sum(4) - &
            mole_fractions(4)/bdiff_ij(4, 4)
        denom(5) = x_sum(5) - &
            mole_fractions(5)/bdiff_ij(5, 5)
        denom(6) = x_sum(6) - &
            mole_fractions(6)/bdiff_ij(6, 6)
        denom(7) = x_sum(7) - &
            mole_fractions(7)/bdiff_ij(7, 7)
        denom(8) = x_sum(8) - &
            mole_fractions(8)/bdiff_ij(8, 8)
        denom(9) = x_sum(9) - &
            mole_fractions(9)/bdiff_ij(9, 9)
        denom(10) = x_sum(10) - &
            mole_fractions(10)/bdiff_ij(10, 10)

        if (denom(1) .gt. 0d0) then
        mass_diffusivities_mixavg(1) = &
            (mix_mol_weight - &
                mole_fractions(1)*molecular_weights(1))&
            /(pressure * mix_mol_weight * denom(1))
        else
        mass_diffusivities_mixavg(1) = &
            bdiff_ij(1, 1) / pressure
        end if
        if (denom(2) .gt. 0d0) then
        mass_diffusivities_mixavg(2) = &
            (mix_mol_weight - &
                mole_fractions(2)*molecular_weights(2))&
            /(pressure * mix_mol_weight * denom(2))
        else
        mass_diffusivities_mixavg(2) = &
            bdiff_ij(2, 2) / pressure
        end if
        if (denom(3) .gt. 0d0) then
        mass_diffusivities_mixavg(3) = &
            (mix_mol_weight - &
                mole_fractions(3)*molecular_weights(3))&
            /(pressure * mix_mol_weight * denom(3))
        else
        mass_diffusivities_mixavg(3) = &
            bdiff_ij(3, 3) / pressure
        end if
        if (denom(4) .gt. 0d0) then
        mass_diffusivities_mixavg(4) = &
            (mix_mol_weight - &
                mole_fractions(4)*molecular_weights(4))&
            /(pressure * mix_mol_weight * denom(4))
        else
        mass_diffusivities_mixavg(4) = &
            bdiff_ij(4, 4) / pressure
        end if
        if (denom(5) .gt. 0d0) then
        mass_diffusivities_mixavg(5) = &
            (mix_mol_weight - &
                mole_fractions(5)*molecular_weights(5))&
            /(pressure * mix_mol_weight * denom(5))
        else
        mass_diffusivities_mixavg(5) = &
            bdiff_ij(5, 5) / pressure
        end if
        if (denom(6) .gt. 0d0) then
        mass_diffusivities_mixavg(6) = &
            (mix_mol_weight - &
                mole_fractions(6)*molecular_weights(6))&
            /(pressure * mix_mol_weight * denom(6))
        else
        mass_diffusivities_mixavg(6) = &
            bdiff_ij(6, 6) / pressure
        end if
        if (denom(7) .gt. 0d0) then
        mass_diffusivities_mixavg(7) = &
            (mix_mol_weight - &
                mole_fractions(7)*molecular_weights(7))&
            /(pressure * mix_mol_weight * denom(7))
        else
        mass_diffusivities_mixavg(7) = &
            bdiff_ij(7, 7) / pressure
        end if
        if (denom(8) .gt. 0d0) then
        mass_diffusivities_mixavg(8) = &
            (mix_mol_weight - &
                mole_fractions(8)*molecular_weights(8))&
            /(pressure * mix_mol_weight * denom(8))
        else
        mass_diffusivities_mixavg(8) = &
            bdiff_ij(8, 8) / pressure
        end if
        if (denom(9) .gt. 0d0) then
        mass_diffusivities_mixavg(9) = &
            (mix_mol_weight - &
                mole_fractions(9)*molecular_weights(9))&
            /(pressure * mix_mol_weight * denom(9))
        else
        mass_diffusivities_mixavg(9) = &
            bdiff_ij(9, 9) / pressure
        end if
        if (denom(10) .gt. 0d0) then
        mass_diffusivities_mixavg(10) = &
            (mix_mol_weight - &
                mole_fractions(10)*molecular_weights(10))&
            /(pressure * mix_mol_weight * denom(10))
        else
        mass_diffusivities_mixavg(10) = &
            bdiff_ij(10, 10) / pressure
        end if

    end subroutine get_species_mass_diffusivities_mixavg

end module MFC
