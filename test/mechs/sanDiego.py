"""
.. autoclass:: Thermochemistry
"""


import numpy as np


class Thermochemistry:
    """
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    """

    def __init__(self, usr_np=np):
        """Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        """

        self.usr_np = usr_np
        self.model_name = 'mechs/sanDiego.cti'
        self.num_elements = 3
        self.num_species = 9
        self.num_reactions = 24
        self.num_falloff = 2

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['H2', 'H', 'O2', 'O', 'OH', 'HO2', 'H2O2', 'H2O', 'N2']
        self.species_indices = {'H2': 0, 'H': 1, 'O2': 2, 'O': 3, 'OH': 4, 'HO2': 5, 'H2O2': 6, 'H2O': 7, 'N2': 8}

        self.wts = np.array([2.016, 1.008, 31.998, 15.999, 17.007, 33.006, 34.014, 18.015, 28.014])
        self.iwts = 1/self.wts

    def _pyro_zeros_like(self, argument):
        # FIXME: This is imperfect, as a NaN will stay a NaN.
        return 0 * argument

    def _pyro_make_array(self, res_list):
        """This works around (e.g.) numpy.exp not working with object
        arrays of numpy scalars. It defaults to making object arrays, however
        if an array consists of all scalars, it makes a "plain old"
        :class:`numpy.ndarray`.

        See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
        for more context.
        """

        from numbers import Number
        all_numbers = all(isinstance(e, Number) for e in res_list)

        dtype = np.float64 if all_numbers else np.object
        result = np.empty((len(res_list),), dtype=dtype)

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for idx in range(len(res_list)):
            result[idx] = res_list[idx]

        return result

    def _pyro_norm(self, argument, normord):
        """This works around numpy.linalg norm not working with scalars.

        If the argument is a regular ole number, it uses :func:`numpy.abs`,
        otherwise it uses ``usr_np.linalg.norm``.
        """
        # Wrap norm for scalars

        from numbers import Number

        if isinstance(argument, Number):
            return np.abs(argument)
        return self.usr_np.linalg.norm(argument, normord)

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * (
                    + self.iwts[0]*mass_fractions[0]
                    + self.iwts[1]*mass_fractions[1]
                    + self.iwts[2]*mass_fractions[2]
                    + self.iwts[3]*mass_fractions[3]
                    + self.iwts[4]*mass_fractions[4]
                    + self.iwts[5]*mass_fractions[5]
                    + self.iwts[6]*mass_fractions[6]
                    + self.iwts[7]*mass_fractions[7]
                    + self.iwts[8]*mass_fractions[8]
                )

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
                    + self.iwts[0]*mass_fractions[0]
                    + self.iwts[1]*mass_fractions[1]
                    + self.iwts[2]*mass_fractions[2]
                    + self.iwts[3]*mass_fractions[3]
                    + self.iwts[4]*mass_fractions[4]
                    + self.iwts[5]*mass_fractions[5]
                    + self.iwts[6]*mass_fractions[6]
                    + self.iwts[7]*mass_fractions[7]
                    + self.iwts[8]*mass_fractions[8]
                )

    def get_concentrations(self, rho, mass_fractions):
        return self.iwts * self._pyro_make_array(rho).reshape(-1, 1) * mass_fractions

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([mass_fractions[i] * spec_property[i] * self.iwts[i]
                    for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_specific_heats_r(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -4.94024731e-05*temperature + 4.99456778e-07*temperature**2 + -1.79566394e-10*temperature**3 + 2.00255376e-14*temperature**4, 2.34433112 + 0.00798052075*temperature + -1.9478151e-05*temperature**2 + 2.01572094e-08*temperature**3 + -7.37611761e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -2.30842973e-11*temperature + 1.61561948e-14*temperature**2 + -4.73515235e-18*temperature**3 + 4.98197357e-22*temperature**4, 2.5 + 7.05332819e-13*temperature + -1.99591964e-15*temperature**2 + 2.30081632e-18*temperature**3 + -9.27732332e-22*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00148308754*temperature + -7.57966669e-07*temperature**2 + 2.09470555e-10*temperature**3 + -2.16717794e-14*temperature**4, 3.78245636 + -0.00299673416*temperature + 9.84730201e-06*temperature**2 + -9.68129509e-09*temperature**3 + 3.24372837e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -8.59741137e-05*temperature + 4.19484589e-08*temperature**2 + -1.00177799e-11*temperature**3 + 1.22833691e-15*temperature**4, 3.1682671 + -0.00327931884*temperature + 6.64306396e-06*temperature**2 + -6.12806624e-09*temperature**3 + 2.11265971e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886 + 0.00105650448*temperature + -2.59082758e-07*temperature**2 + 3.05218674e-11*temperature**3 + -1.33195876e-15*temperature**4, 4.12530561 + -0.00322544939*temperature + 6.52764691e-06*temperature**2 + -5.79853643e-09*temperature**3 + 2.06237379e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.00223982013*temperature + -6.3365815e-07*temperature**2 + 1.1424637e-10*temperature**3 + -1.07908535e-14*temperature**4, 4.30179801 + -0.00474912051*temperature + 2.11582891e-05*temperature**2 + -2.42763894e-08*temperature**3 + 9.29225124e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00490831694*temperature + -1.90139225e-06*temperature**2 + 3.71185986e-10*temperature**3 + -2.87908305e-14*temperature**4, 4.27611269 + -0.000542822417*temperature + 1.67335701e-05*temperature**2 + -2.15770813e-08*temperature**3 + 8.62454363e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00217691804*temperature + -1.64072518e-07*temperature**2 + -9.7041987e-11*temperature**3 + 1.68200992e-14*temperature**4, 4.19864056 + -0.0020364341*temperature + 6.52040211e-06*temperature**2 + -5.48797062e-09*temperature**3 + 1.77197817e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
                ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -2.470123655e-05*temperature + 1.6648559266666665e-07*temperature**2 + -4.48915985e-11*temperature**3 + 4.00510752e-15*temperature**4 + -950.158922 / temperature, 2.34433112 + 0.003990260375*temperature + -6.4927169999999995e-06*temperature**2 + 5.03930235e-09*temperature**3 + -1.4752235220000002e-12*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -1.154214865e-11*temperature + 5.385398266666667e-15*temperature**2 + -1.1837880875e-18*temperature**3 + 9.96394714e-23*temperature**4 + 25473.6599 / temperature, 2.5 + 3.526664095e-13*temperature + -6.653065466666667e-16*temperature**2 + 5.7520408e-19*temperature**3 + -1.855464664e-22*temperature**4 + 25473.6599 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00074154377*temperature + -2.526555563333333e-07*temperature**2 + 5.236763875e-11*temperature**3 + -4.33435588e-15*temperature**4 + -1088.45772 / temperature, 3.78245636 + -0.00149836708*temperature + 3.282434003333333e-06*temperature**2 + -2.4203237725e-09*temperature**3 + 6.48745674e-13*temperature**4 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -4.298705685e-05*temperature + 1.3982819633333334e-08*temperature**2 + -2.504444975e-12*temperature**3 + 2.4566738199999997e-16*temperature**4 + 29217.5791 / temperature, 3.1682671 + -0.00163965942*temperature + 2.2143546533333334e-06*temperature**2 + -1.53201656e-09*temperature**3 + 4.22531942e-13*temperature**4 + 29122.2592 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886 + 0.00052825224*temperature + -8.636091933333334e-08*temperature**2 + 7.63046685e-12*temperature**3 + -2.66391752e-16*temperature**4 + 3718.85774 / temperature, 4.12530561 + -0.001612724695*temperature + 2.1758823033333334e-06*temperature**2 + -1.4496341075e-09*temperature**3 + 4.1247475799999997e-13*temperature**4 + 3381.53812 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.001119910065*temperature + -2.1121938333333332e-07*temperature**2 + 2.85615925e-11*temperature**3 + -2.1581707e-15*temperature**4 + 111.856713 / temperature, 4.30179801 + -0.002374560255*temperature + 7.0527630333333326e-06*temperature**2 + -6.06909735e-09*temperature**3 + 1.8584502480000002e-12*temperature**4 + 294.80804 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00245415847*temperature + -6.337974166666666e-07*temperature**2 + 9.27964965e-11*temperature**3 + -5.7581661e-15*temperature**4 + -17861.7877 / temperature, 4.27611269 + -0.0002714112085*temperature + 5.5778567000000005e-06*temperature**2 + -5.394270325e-09*temperature**3 + 1.724908726e-12*temperature**4 + -17702.5821 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00108845902*temperature + -5.469083933333333e-08*temperature**2 + -2.426049675e-11*temperature**3 + 3.36401984e-15*temperature**4 + -30004.2971 / temperature, 4.19864056 + -0.00101821705*temperature + 2.17346737e-06*temperature**2 + -1.371992655e-09*temperature**3 + 3.54395634e-13*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
                ])

    def get_species_entropies_r(self, temperature):
        return self._pyro_make_array([
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792*self.usr_np.log(temperature) + -4.94024731e-05*temperature + 2.49728389e-07*temperature**2 + -5.985546466666667e-11*temperature**3 + 5.0063844e-15*temperature**4 + -3.20502331, 2.34433112*self.usr_np.log(temperature) + 0.00798052075*temperature + -9.7390755e-06*temperature**2 + 6.7190698e-09*temperature**3 + -1.8440294025e-12*temperature**4 + 0.683010238),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001*self.usr_np.log(temperature) + -2.30842973e-11*temperature + 8.0780974e-15*temperature**2 + -1.5783841166666668e-18*temperature**3 + 1.2454933925e-22*temperature**4 + -0.446682914, 2.5*self.usr_np.log(temperature) + 7.05332819e-13*temperature + -9.9795982e-16*temperature**2 + 7.669387733333333e-19*temperature**3 + -2.31933083e-22*temperature**4 + -0.446682853),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784*self.usr_np.log(temperature) + 0.00148308754*temperature + -3.789833345e-07*temperature**2 + 6.982351833333333e-11*temperature**3 + -5.41794485e-15*temperature**4 + 5.45323129, 3.78245636*self.usr_np.log(temperature) + -0.00299673416*temperature + 4.923651005e-06*temperature**2 + -3.2270983633333334e-09*temperature**3 + 8.109320925e-13*temperature**4 + 3.65767573),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078*self.usr_np.log(temperature) + -8.59741137e-05*temperature + 2.097422945e-08*temperature**2 + -3.3392599666666663e-12*temperature**3 + 3.070842275e-16*temperature**4 + 4.78433864, 3.1682671*self.usr_np.log(temperature) + -0.00327931884*temperature + 3.32153198e-06*temperature**2 + -2.0426887466666666e-09*temperature**3 + 5.281649275e-13*temperature**4 + 2.05193346),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886*self.usr_np.log(temperature) + 0.00105650448*temperature + -1.29541379e-07*temperature**2 + 1.01739558e-11*temperature**3 + -3.3298969e-16*temperature**4 + 5.70164073, 4.12530561*self.usr_np.log(temperature) + -0.00322544939*temperature + 3.263823455e-06*temperature**2 + -1.9328454766666666e-09*temperature**3 + 5.155934475e-13*temperature**4 + -0.69043296),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109*self.usr_np.log(temperature) + 0.00223982013*temperature + -3.16829075e-07*temperature**2 + 3.808212333333334e-11*temperature**3 + -2.697713375e-15*temperature**4 + 3.78510215, 4.30179801*self.usr_np.log(temperature) + -0.00474912051*temperature + 1.057914455e-05*temperature**2 + -8.0921298e-09*temperature**3 + 2.32306281e-12*temperature**4 + 3.71666245),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285*self.usr_np.log(temperature) + 0.00490831694*temperature + -9.50696125e-07*temperature**2 + 1.2372866199999999e-10*temperature**3 + -7.197707625e-15*temperature**4 + 2.91615662, 4.27611269*self.usr_np.log(temperature) + -0.000542822417*temperature + 8.36678505e-06*temperature**2 + -7.192360433333333e-09*temperature**3 + 2.1561359075e-12*temperature**4 + 3.43505074),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249*self.usr_np.log(temperature) + 0.00217691804*temperature + -8.2036259e-08*temperature**2 + -3.2347329e-11*temperature**3 + 4.2050248e-15*temperature**4 + 4.9667701, 4.19864056*self.usr_np.log(temperature) + -0.0020364341*temperature + 3.260201055e-06*temperature**2 + -1.82932354e-09*temperature**3 + 4.429945425e-13*temperature**4 + -0.849032208),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664*self.usr_np.log(temperature) + 0.0014879768*temperature + -2.84238e-07*temperature**2 + 3.3656793333333334e-11*temperature**3 + -1.68833775e-15*temperature**4 + 5.980528, 3.298677*self.usr_np.log(temperature) + 0.0014082404*temperature + -1.981611e-06*temperature**2 + 1.8805050000000002e-09*temperature**3 + -6.112135e-13*temperature**4 + 3.950372),
                ])

    def get_species_gibbs_rt(self, temperature):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
                    g0_rt[3] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[2]),
                    g0_rt[1] + g0_rt[4] + -1*(g0_rt[0] + g0_rt[3]),
                    g0_rt[1] + g0_rt[7] + -1*(g0_rt[0] + g0_rt[4]),
                    2.0*g0_rt[4] + -1*(g0_rt[7] + g0_rt[3]),
                    g0_rt[0] + -1*-1.0*c0 + -1*2.0*g0_rt[1],
                    g0_rt[7] + -1*-1.0*c0 + -1*(g0_rt[1] + g0_rt[4]),
                    g0_rt[2] + -1*-1.0*c0 + -1*2.0*g0_rt[3],
                    g0_rt[4] + -1*-1.0*c0 + -1*(g0_rt[1] + g0_rt[3]),
                    g0_rt[5] + -1*-1.0*c0 + -1*(g0_rt[3] + g0_rt[4]),
                    g0_rt[5] + -1*-1.0*c0 + -1*(g0_rt[1] + g0_rt[2]),
                    2.0*g0_rt[4] + -1*(g0_rt[1] + g0_rt[5]),
                    g0_rt[0] + g0_rt[2] + -1*(g0_rt[1] + g0_rt[5]),
                    g0_rt[7] + g0_rt[3] + -1*(g0_rt[1] + g0_rt[5]),
                    g0_rt[2] + g0_rt[4] + -1*(g0_rt[5] + g0_rt[3]),
                    g0_rt[7] + g0_rt[2] + -1*(g0_rt[5] + g0_rt[4]),
                    g0_rt[7] + g0_rt[2] + -1*(g0_rt[5] + g0_rt[4]),
                    g0_rt[6] + -1*-1.0*c0 + -1*2.0*g0_rt[4],
                    g0_rt[6] + g0_rt[2] + -1*2.0*g0_rt[5],
                    g0_rt[6] + g0_rt[2] + -1*2.0*g0_rt[5],
                    g0_rt[0] + g0_rt[5] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[7] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[7] + g0_rt[5] + -1*(g0_rt[6] + g0_rt[4]),
                    g0_rt[7] + g0_rt[5] + -1*(g0_rt[6] + g0_rt[4]),
                    g0_rt[5] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[3]),
                ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
        t_i = t_guess * ones

        for _ in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self._pyro_norm(dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    def get_falloff_rates(self, temperature, concentrations, k_fwd):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_high = self._pyro_make_array([
            4650000000.0*temperature**0.44,
            95500000000.0*temperature**-0.27,
                ])

        k_low = self._pyro_make_array([
            57500000000000.0*temperature**-1.4,
            2.76e+19*temperature**-3.2,
                ])

        reduced_pressure = self._pyro_make_array([
            (2.5*concentrations[0] + 16.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])*k_low[0]/k_high[0],
            (2.5*concentrations[0] + 6.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])*k_low[1]/k_high[1],
                            ])

        falloff_center = self._pyro_make_array([
            self.usr_np.log10(0.5*self.usr_np.exp((-1*temperature) / 1e-30) + 0.5*self.usr_np.exp((-1*temperature) / 1.0000000000000002e+30)),
            self.usr_np.log10(0.43000000000000005*self.usr_np.exp((-1*temperature) / 1.0000000000000002e+30) + 0.57*self.usr_np.exp((-1*temperature) / 1e-30)),
                        ])

        falloff_function = self._pyro_make_array([
            10**(falloff_center[0] / (1 + ((self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0]) / (0.75 + -1*1.27*falloff_center[0] + -1*0.14*(self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0])))**2)),
            10**(falloff_center[1] / (1 + ((self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1]) / (0.75 + -1*1.27*falloff_center[1] + -1*0.14*(self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1])))**2)),
                            ])*reduced_pressure/(1+reduced_pressure)

        k_fwd[9] = k_high[0]*falloff_function[0]*ones
        k_fwd[16] = k_high[1]*falloff_function[1]*ones
        return

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            self.usr_np.exp(31.192067198532598 + -0.7*self.usr_np.log(temperature) + -1*(8589.851597151493 / temperature)) * ones,
            self.usr_np.exp(3.92395157629342 + 2.67*self.usr_np.log(temperature) + -1*(3165.568384724549 / temperature)) * ones,
            self.usr_np.exp(13.97251430677394 + 1.3*self.usr_np.log(temperature) + -1*(1829.342520199863 / temperature)) * ones,
            self.usr_np.exp(6.551080335043405 + 2.33*self.usr_np.log(temperature) + -1*(7320.978251450734 / temperature)) * ones,
            1300000000000.0*temperature**-1.0 * ones,
            4e+16*temperature**-2.0 * ones,
            6170000000.0*temperature**-0.5 * ones,
            4710000000000.0*temperature**-1.0 * ones,
            8000000000.0 * ones,
            0*temperature,
            self.usr_np.exp(24.983124837646088 + -1*(148.41608612272393 / temperature)) * ones,
            self.usr_np.exp(23.532668532308907 + -1*(414.09771841210573 / temperature)) * ones,
            self.usr_np.exp(24.15725304143156 + -1*(865.9609563076275 / temperature)) * ones,
            20000000000.0 * ones,
            self.usr_np.exp(26.83251341971078 + -1*(5500.054796103862 / temperature)) * ones,
            self.usr_np.exp(24.117774230457773 + -1*(-250.16649848887016 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(19.083368717027604 + -1*(-709.00553297687 / temperature)) * ones,
            self.usr_np.exp(25.357994825176046 + -1*(5556.582802973943 / temperature)) * ones,
            self.usr_np.exp(23.85876005287556 + -1*(4000.619345786196 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(1804.0853256408905 / temperature)) * ones,
            self.usr_np.exp(25.052682521347993 + -1*(3659.8877639501534 / temperature)) * ones,
            self.usr_np.exp(21.277150950172846 + -1*(159.96223220682563 / temperature)) * ones,
            self.usr_np.exp(9.172638504792172 + 2.0*self.usr_np.log(temperature) + -1*(2008.5483292135248 / temperature)) * ones,
                ]
        self.get_falloff_rates(temperature, concentrations, k_fwd)

        k_fwd[4] *= (2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])
        k_fwd[5] *= (2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])
        k_fwd[6] *= (2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])
        k_fwd[7] *= (2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])
        k_fwd[8] *= (2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])
        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[0])*concentrations[3]*concentrations[4]),
                    k_fwd[1]*(concentrations[0]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[1])*concentrations[1]*concentrations[4]),
                    k_fwd[2]*(concentrations[0]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[2])*concentrations[1]*concentrations[7]),
                    k_fwd[3]*(concentrations[7]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[3])*concentrations[4]**2.0),
                    k_fwd[4]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[4])*concentrations[0]),
                    k_fwd[5]*(concentrations[1]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[5])*concentrations[7]),
                    k_fwd[6]*(concentrations[3]**2.0 + -1*self.usr_np.exp(log_k_eq[6])*concentrations[2]),
                    k_fwd[7]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[7])*concentrations[4]),
                    k_fwd[8]*(concentrations[3]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[8])*concentrations[5]),
                    k_fwd[9]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[9])*concentrations[5]),
                    k_fwd[10]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[10])*concentrations[4]**2.0),
                    k_fwd[11]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[11])*concentrations[0]*concentrations[2]),
                    k_fwd[12]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[12])*concentrations[7]*concentrations[3]),
                    k_fwd[13]*(concentrations[5]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[13])*concentrations[2]*concentrations[4]),
                    k_fwd[14]*(concentrations[5]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[14])*concentrations[7]*concentrations[2]),
                    k_fwd[15]*(concentrations[5]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[15])*concentrations[7]*concentrations[2]),
                    k_fwd[16]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[16])*concentrations[6]),
                    k_fwd[17]*(concentrations[5]**2.0 + -1*self.usr_np.exp(log_k_eq[17])*concentrations[6]*concentrations[2]),
                    k_fwd[18]*(concentrations[5]**2.0 + -1*self.usr_np.exp(log_k_eq[18])*concentrations[6]*concentrations[2]),
                    k_fwd[19]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[19])*concentrations[0]*concentrations[5]),
                    k_fwd[20]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[20])*concentrations[7]*concentrations[4]),
                    k_fwd[21]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[21])*concentrations[7]*concentrations[5]),
                    k_fwd[22]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[22])*concentrations[7]*concentrations[5]),
                    k_fwd[23]*(concentrations[6]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[23])*concentrations[5]*concentrations[4]),
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        if c.shape != mass_fractions.shape:
            c = c.reshape(mass_fractions.shape)
        r_net = self.get_net_rates_of_progress(temperature, self._pyro_make_array(c.T))
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                r_net[4] + r_net[11] + r_net[19] + -1*(r_net[1] + r_net[2]) * ones,
                r_net[1] + r_net[2] + -1*(r_net[0] + 2.0*r_net[4] + r_net[5] + r_net[7] + r_net[9] + r_net[10] + r_net[11] + r_net[12] + r_net[19] + r_net[20]) * ones,
                r_net[6] + r_net[11] + r_net[13] + r_net[14] + r_net[15] + r_net[17] + r_net[18] + -1*(r_net[0] + r_net[9]) * ones,
                r_net[0] + r_net[12] + -1*(r_net[1] + r_net[3] + 2.0*r_net[6] + r_net[7] + r_net[8] + r_net[13] + r_net[23]) * ones,
                r_net[0] + r_net[1] + 2.0*r_net[3] + r_net[7] + 2.0*r_net[10] + r_net[13] + r_net[20] + r_net[23] + -1*(r_net[2] + r_net[5] + r_net[8] + r_net[14] + r_net[15] + 2.0*r_net[16] + r_net[21] + r_net[22]) * ones,
                r_net[8] + r_net[9] + r_net[19] + r_net[21] + r_net[22] + r_net[23] + -1*(r_net[10] + r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[15] + 2.0*r_net[17] + 2.0*r_net[18]) * ones,
                r_net[16] + r_net[17] + r_net[18] + -1*(r_net[19] + r_net[20] + r_net[21] + r_net[22] + r_net[23]) * ones,
                r_net[2] + r_net[5] + r_net[12] + r_net[14] + r_net[15] + r_net[20] + r_net[21] + r_net[22] + -1*r_net[3] * ones,
                0.0 * ones,
               ])
