import jax
import pytato as pt
import numpy as np
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

_array_types = {
    jnp: jnp.ndarray,
    pt: pt.Array,
    np: np.ndarray
}


def all_numbers(res_list, usr_np):
    from numbers import Number
    # from itertools import product
    return all(isinstance(e, Number)
               or (e.shape == () and isinstance(type(e), _array_types[usr_np]))
               for e in res_list)


def _np_stack(res_list, usr_np):
    return usr_np.stack(res_list)


def _pt_stack(res_list, usr_np):
    array = np.empty(len(res_list), dtype=object)
    for idx in range(len(res_list)):
        array[idx] = res_list[idx]
    return array


def _jax_stack(res_list, usr_np):
    if all_numbers(res_list, usr_np):
        return usr_np.stack(res_list)

    array = usr_np.empty_like(
        usr_np.array(res_list),
    )
    for idx in range(len(res_list)):
        array = array.at[idx].set(res_list[idx])

    return array


def make_pyro_object(pyro_cls, usr_np):
    if usr_np == np:
        return pyro_cls(usr_np)

    elif usr_np == pt:

        class PyroPytato(pyro_cls):
            def __init__(self, usr_np):
                self.usr_np = usr_np
                self.num_species = pyro_cls().num_species
                self.num_reactions = pyro_cls().num_reactions
                self.one_atm = pyro_cls().one_atm
                self.gas_constant = pyro_cls().gas_constant
                self.molecular_weights = self._pyro_make_array(
                    pyro_cls().molecular_weights,
                )
                self.inv_molecular_weights = self._pyro_make_array(
                    pyro_cls().inv_molecular_weights
                )
                self.species_indices = pyro_cls().species_indices

            def _pyro_make_array(self, res_list):
                return _pt_stack(res_list, usr_np)

            def get_temperature(self, energy, temp_guess, mass_fractions):
                num_iter = 2
                iter_temp = temp_guess

                for _ in range(num_iter):
                    f = energy - self.get_mixture_internal_energy_mass(
                        iter_temp, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cv_mass(
                        iter_temp, mass_fractions
                    )
                    dt = -f/j
                    iter_temp += dt

                return iter_temp

        return PyroPytato(usr_np)

    elif usr_np == jnp:

        class PyroJaxNumpy(pyro_cls):

            def _pyro_make_array(self, res_list):
                return _jax_stack(res_list, usr_np)

            def _pyro_norm(self, argument, normord):
                # Wrap norm for scalars
                from numbers import Number
                if isinstance(argument, Number):
                    return self.usr_np.abs(argument)
                if isinstance(
                        argument, self.usr_np.ndarray
                ) and argument.shape == ():
                    return self.usr_np.abs(argument)
                return self.usr_np.linalg.norm(argument, normord)

            def get_temperature(self, energy, temp_init, mass_fractions,
                                do_energy=True):

                def cond_fun(temperature):
                    f = energy - self.get_mixture_internal_energy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cv_mass(
                        temperature, mass_fractions
                    )
                    return self.usr_np.linalg.norm(f/j) > 1e-10

                def body_fun(temperature):
                    f = energy - self.get_mixture_internal_energy_mass(
                        temperature, mass_fractions
                    )
                    j = -self.get_mixture_specific_heat_cv_mass(
                        temperature, mass_fractions
                    )
                    return temperature - f/j

                return jax.lax.while_loop(cond_fun, body_fun, temp_init)

        return PyroJaxNumpy(usr_np=usr_np)
