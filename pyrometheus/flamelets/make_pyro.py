import numpy as np
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


_array_types = {
    jax.numpy: jax.numpy.ndarray,
    np: np.ndarray,
}


def detect_array_library(array):
    """
    Detect which array library an array belongs to.

    Returns the appropriate library module (np, jax.numpy, cp, etc.)
    """
    if isinstance(array, np.ndarray):
        return np
    elif isinstance(array, jax.numpy.ndarray):
        return jax.numpy
    else:
        raise ValueError(f"Unknown array type: {type(array)}")


def all_numbers(res_list, pyro_np):
    from numbers import Number
    return all(
        isinstance(e, Number)
        or (e.shape == () and isinstance(type(e), _array_types[pyro_np]))
        for e in res_list
    )


def _np_stack(res_list,):
    return np.stack(res_list)


def _jax_stack(res_list, pyro_np):
    array = pyro_np.empty_like(
        pyro_np.array(res_list),
    )
    for idx in range(len(res_list)):
        array = array.at[idx].set(res_list[idx])

    return array


def make_pyro_object(pyro_cls,
                     pyro_np,
                     device=None,
                     dtype=jnp.float64):
    if pyro_np == np:

        class PyroNumPy(pyro_cls):
            def __init__(self, pyro_np):
                super().__init__(pyro_np)
                self.pv_fun = self.get_mixture_specific_heat_cv_mass
                self.he_fun = self.get_mixture_internal_energy_mass

            def _pyro_make_array(self, res_list):
                return _np_stack(res_list)

            def get_temperature(self, energy, t_guess, y, tol=1e-6):
                num_iter = 500
                ones = self._pyro_zeros_like(energy) + 1.0
                iter_temp = t_guess * ones
                for _ in range(num_iter):
                    iter_rhs = energy - self.he_fun(iter_temp, y)
                    iter_deriv = -self.pv_fun(iter_temp, y)
                    dt = -iter_rhs / iter_deriv
                    iter_temp += dt
                    if self._pyro_norm(dt.ravel(), np.inf) < tol:
                        return iter_temp
                raise RuntimeError("Temperature iteration failed to converge")

        return PyroNumPy(pyro_np)

    elif pyro_np == jax.numpy:

        class PyroJAX(pyro_cls):

            def _pyro_make_array(self, res_list):
                return _jax_stack(res_list, self.pyro_np)

            def _pyro_norm(self, argument, normord):
                # Wrap norm for scalars
                from numbers import Number

                if isinstance(argument, Number):
                    return self.pyro_np.abs(argument)
                if isinstance(argument,
                              self.pyro_np.ndarray) and argument.shape == ():
                    return self.pyro_np.abs(argument)
                return self.pyro_np.linalg.norm(argument, normord)

            def get_temperature(
                self, energy, temp_init, mass_fractions, do_energy=True
            ):

                def cond_fun(temperature):
                    f = energy - self.get_mixture_internal_energy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cv_mass(
                        temperature, mass_fractions
                    )
                    return self.pyro_np.linalg.norm(f / j) > 1e-10

                def body_fun(temperature):
                    f = energy - self.get_mixture_internal_energy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cv_mass(
                        temperature, mass_fractions
                    )
                    return temperature - f / j

                return jax.lax.while_loop(cond_fun, body_fun, temp_init)

            def get_temperature_from_enthalpy(
                    self, enthalpy, mass_fractions, temp_init,
            ):

                def cond_fun(temperature):
                    f = enthalpy - self.get_mixture_enthalpy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cp_mass(
                        temperature, mass_fractions
                    )
                    return self.pyro_np.linalg.norm(f / j) > 1e-10

                def body_fun(temperature):
                    f = enthalpy - self.get_mixture_enthalpy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cp_mass(
                        temperature, mass_fractions
                    )
                    return temperature - f / j

                return jax.lax.while_loop(cond_fun, body_fun, temp_init)

        return PyroJAX(pyro_np)

    else:
        raise ValueError(
            f"Unsupported array library: {pyro_np}. "
            f"Supported libraries: numpy, jax.numpy,"
        )
