import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", 1)


def make_jax_pyro_class(pyro_class, usr_np):
    if usr_np != jnp:
        return pyro_class(usr_np)

    class pyro_jax_numpy(pyro_class):

        def _pyro_make_array(self, res_list):
            """This works around (e.g.) numpy.exp not working with object arrays of numpy
            scalars. It defaults to making object arrays, however if an array
            consists of all scalars, it makes a "plain old" :class:`numpy.ndarray`.

            See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
            for more context.
            """

            from numbers import Number
            # Needed to play nicely with Jax, which frequently creates
            # arrays of shape () when handed numbers
            all_numbers = all(
                isinstance(e, Number)
                or (isinstance(e, self.usr_np.ndarray) and e.shape == ())
                for e in res_list)

            if all_numbers:
                return self.usr_np.array(res_list, dtype=self.usr_np.float64)

#             result = self.usr_np.empty_like(res_list, dtype=object,
#                                             shape=(len(res_list),))
            result = self._pyro_zeros_like(self.usr_np.array(res_list))

            # 'result[:] = res_list' may look tempting, however:
            # https://github.com/numpy/numpy/issues/16564
            for idx in range(len(res_list)):
                result = result.at[idx].set(res_list[idx])

            return result

        def _pyro_norm(self, argument, normord):
            """This works around numpy.linalg norm not working with scalars.

            If the argument is a regular ole number, it uses :func:`numpy.abs`,
            otherwise it uses ``usr_np.linalg.norm``.
            """
            # Wrap norm for scalars
            from numbers import Number
            if isinstance(argument, Number):
                return self.usr_np.abs(argument)
            # Needed to play nicely with Jax, which frequently creates
            # arrays of shape () when handed numbers
            if isinstance(argument, self.usr_np.ndarray) and argument.shape == ():
                return self.usr_np.abs(argument)
            return self.usr_np.linalg.norm(argument, normord)
        
        def get_temperature_enthalpy(self, mass_fractions, enthalpy, temp_init):

            def cond_fun(temperature):
                f = enthalpy - self.get_mixture_enthalpy_mass(temperature, mass_fractions)
                j = -self.get_mixture_specific_heat_cp_mass(temperature, mass_fractions)
                return abs(f/j) > 1e-5

            def body_fun(temperature):
                f = enthalpy - self.get_mixture_enthalpy_mass(temperature, mass_fractions)
                j = -self.get_mixture_specific_heat_cp_mass(temperature, mass_fractions)
                return temperature - f/j

            return jax.lax.while_loop(cond_fun, body_fun, temp_init)
        
        def get_temperature_energy(self, mass_fractions, energy, temp_init):

            def cond_fun(temperature):
                f = energy - self.get_mixture_internal_energy_mass(temperature, mass_fractions)
                j = -self.get_mixture_specific_heat_cv_mass(temperature, mass_fractions)
                return abs(f/j) > 1e-5

            def body_fun(temperature):
                f = energy - self.get_mixture_internal_energy_mass(temperature, mass_fractions)
                j = -self.get_mixture_specific_heat_cv_mass(temperature, mass_fractions)
                return temperature - f/j

            return jax.lax.while_loop(cond_fun, body_fun, temp_init)

    return pyro_jax_numpy(usr_np=usr_np)
