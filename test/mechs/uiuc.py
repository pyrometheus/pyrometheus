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
        self.model_name = 'mechs/uiuc.cti'
        self.num_elements = 4
        self.num_species = 7
        self.num_reactions = 3
        self.num_falloff = 0

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.wts = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
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
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.0146454151*temperature + -6.71077915e-06*temperature**2 + 1.47222923e-09*temperature**3 + -1.25706061e-13*temperature**4, 3.95920148 + -0.00757052247*temperature + 5.70990292e-05*temperature**2 + -6.91588753e-08*temperature**3 + 2.69884373e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00148308754*temperature + -7.57966669e-07*temperature**2 + 2.09470555e-10*temperature**3 + -2.16717794e-14*temperature**4, 3.78245636 + -0.00299673416*temperature + 9.84730201e-06*temperature**2 + -9.68129509e-09*temperature**3 + 3.24372837e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00441437026*temperature + -2.21481404e-06*temperature**2 + 5.23490188e-10*temperature**3 + -4.72084164e-14*temperature**4, 2.35677352 + 0.00898459677*temperature + -7.12356269e-06*temperature**2 + 2.45919022e-09*temperature**3 + -1.43699548e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.00206252743*temperature + -9.98825771e-07*temperature**2 + 2.30053008e-10*temperature**3 + -2.03647716e-14*temperature**4, 3.57953347 + -0.00061035368*temperature + 1.01681433e-06*temperature**2 + 9.07005884e-10*temperature**3 + -9.04424499e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00217691804*temperature + -1.64072518e-07*temperature**2 + -9.7041987e-11*temperature**3 + 1.68200992e-14*temperature**4, 4.19864056 + -0.0020364341*temperature + 6.52040211e-06*temperature**2 + -5.48797062e-09*temperature**3 + 1.77197817e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -4.94024731e-05*temperature + 4.99456778e-07*temperature**2 + -1.79566394e-10*temperature**3 + 2.00255376e-14*temperature**4, 2.34433112 + 0.00798052075*temperature + -1.9478151e-05*temperature**2 + 2.01572094e-08*temperature**3 + -7.37611761e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
                ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.00732270755*temperature + -2.2369263833333335e-06*temperature**2 + 3.680573075e-10*temperature**3 + -2.51412122e-14*temperature**4 + 4939.88614 / temperature, 3.95920148 + -0.003785261235*temperature + 1.9033009733333333e-05*temperature**2 + -1.7289718825e-08*temperature**3 + 5.3976874600000004e-12*temperature**4 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00074154377*temperature + -2.526555563333333e-07*temperature**2 + 5.236763875e-11*temperature**3 + -4.33435588e-15*temperature**4 + -1088.45772 / temperature, 3.78245636 + -0.00149836708*temperature + 3.282434003333333e-06*temperature**2 + -2.4203237725e-09*temperature**3 + 6.48745674e-13*temperature**4 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00220718513*temperature + -7.382713466666667e-07*temperature**2 + 1.30872547e-10*temperature**3 + -9.44168328e-15*temperature**4 + -48759.166 / temperature, 2.35677352 + 0.004492298385*temperature + -2.3745208966666665e-06*temperature**2 + 6.14797555e-10*temperature**3 + -2.8739909599999997e-14*temperature**4 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.001031263715*temperature + -3.329419236666667e-07*temperature**2 + 5.7513252e-11*temperature**3 + -4.07295432e-15*temperature**4 + -14151.8724 / temperature, 3.57953347 + -0.00030517684*temperature + 3.3893811e-07*temperature**2 + 2.26751471e-10*temperature**3 + -1.808848998e-13*temperature**4 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00108845902*temperature + -5.469083933333333e-08*temperature**2 + -2.426049675e-11*temperature**3 + 3.36401984e-15*temperature**4 + -30004.2971 / temperature, 4.19864056 + -0.00101821705*temperature + 2.17346737e-06*temperature**2 + -1.371992655e-09*temperature**3 + 3.54395634e-13*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -2.470123655e-05*temperature + 1.6648559266666665e-07*temperature**2 + -4.48915985e-11*temperature**3 + 4.00510752e-15*temperature**4 + -950.158922 / temperature, 2.34433112 + 0.003990260375*temperature + -6.4927169999999995e-06*temperature**2 + 5.03930235e-09*temperature**3 + -1.4752235220000002e-12*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
                ])

    def get_species_entropies_r(self, temperature):
        return self._pyro_make_array([
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116*self.usr_np.log(temperature) + 0.0146454151*temperature + -3.355389575e-06*temperature**2 + 4.907430766666667e-10*temperature**3 + -3.142651525e-14*temperature**4 + 10.3053693, 3.95920148*self.usr_np.log(temperature) + -0.00757052247*temperature + 2.85495146e-05*temperature**2 + -2.3052958433333332e-08*temperature**3 + 6.747109325e-12*temperature**4 + 4.09733096),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784*self.usr_np.log(temperature) + 0.00148308754*temperature + -3.789833345e-07*temperature**2 + 6.982351833333333e-11*temperature**3 + -5.41794485e-15*temperature**4 + 5.45323129, 3.78245636*self.usr_np.log(temperature) + -0.00299673416*temperature + 4.923651005e-06*temperature**2 + -3.2270983633333334e-09*temperature**3 + 8.109320925e-13*temperature**4 + 3.65767573),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029*self.usr_np.log(temperature) + 0.00441437026*temperature + -1.10740702e-06*temperature**2 + 1.7449672933333335e-10*temperature**3 + -1.18021041e-14*temperature**4 + 2.27163806, 2.35677352*self.usr_np.log(temperature) + 0.00898459677*temperature + -3.561781345e-06*temperature**2 + 8.197300733333333e-10*temperature**3 + -3.5924887e-14*temperature**4 + 9.90105222),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561*self.usr_np.log(temperature) + 0.00206252743*temperature + -4.994128855e-07*temperature**2 + 7.6684336e-11*temperature**3 + -5.0911929e-15*temperature**4 + 7.81868772, 3.57953347*self.usr_np.log(temperature) + -0.00061035368*temperature + 5.08407165e-07*temperature**2 + 3.023352946666667e-10*temperature**3 + -2.2610612475e-13*temperature**4 + 3.50840928),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249*self.usr_np.log(temperature) + 0.00217691804*temperature + -8.2036259e-08*temperature**2 + -3.2347329e-11*temperature**3 + 4.2050248e-15*temperature**4 + 4.9667701, 4.19864056*self.usr_np.log(temperature) + -0.0020364341*temperature + 3.260201055e-06*temperature**2 + -1.82932354e-09*temperature**3 + 4.429945425e-13*temperature**4 + -0.849032208),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792*self.usr_np.log(temperature) + -4.94024731e-05*temperature + 2.49728389e-07*temperature**2 + -5.985546466666667e-11*temperature**3 + 5.0063844e-15*temperature**4 + -3.20502331, 2.34433112*self.usr_np.log(temperature) + 0.00798052075*temperature + -9.7390755e-06*temperature**2 + 6.7190698e-09*temperature**3 + -1.8440294025e-12*temperature**4 + 0.683010238),
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
                    -0.17364695002734*temperature,
                    g0_rt[2] + -1*-0.5*c0 + -1*(g0_rt[3] + 0.5*g0_rt[1]),
                    g0_rt[4] + -1*-0.5*c0 + -1*(g0_rt[5] + 0.5*g0_rt[1]),
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

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            self.usr_np.exp(26.594857854425133 + -1*(17864.293439206183 / temperature)) * ones,
            self.usr_np.exp(12.693776816787125 + 0.7*self.usr_np.log(temperature) + -1*(6038.634401985189 / temperature)) * ones,
            self.usr_np.exp(18.302572655472037 + -1*(17612.683672456802 / temperature)) * ones,
                ]

        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*concentrations[0]**0.5*concentrations[1]**0.65,
                    k_fwd[1]*(concentrations[3]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[1])*concentrations[2]),
                    k_fwd[2]*(concentrations[5]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[2])*concentrations[4]),
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        if c.shape != mass_fractions.shape:
            c = c.reshape(mass_fractions.shape)
        r_net = self.get_net_rates_of_progress(temperature, self._pyro_make_array(c.T))
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                -1*r_net[0] * ones,
                -1*(r_net[0] + 0.5*r_net[1] + 0.5*r_net[2]) * ones,
                r_net[1] * ones,
                2.0*r_net[0] + -1*r_net[1] * ones,
                r_net[2] * ones,
                2.0*r_net[0] + -1*r_net[2] * ones,
                0.0 * ones,
               ])
