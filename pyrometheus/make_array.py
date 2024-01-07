__copyright__ = """
Copyright (C) 2020 Esteban Cisneros
Copyright (C) 2020 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Internal Functionality
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: all_numbers
.. autofunction:: numpy
.. autofunction:: eager_jax
.. autofunction:: torch
"""

from arraycontext import (ArrayContext,
                          NumpyArrayContext,
                          EagerJAXArrayContext,
                          PytatoJAXArrayContext,
                          TorchArrayContext)


# {{{ Type check

def all_numbers(res_list, actx: ArrayContext):
    """This function checks whether the elements of `res_list`
    are all numbers, or otherwise arrays allowed by `actx`"""

    from numbers import Number
    from itertools import product
    return all(isinstance(e, Number)
               or (e.shape == () and isinstance(type(e), t))
               for e, t in product(res_list, actx.array_types))

# }}}


# {{{ array creation routines

def numpy(res_list, actx: NumpyArrayContext):
    """This returns a :mod:`numpy.ndarray`"""

    if all_numbers(res_list, actx):
        return actx.np.stack(res_list)

    array = actx.empty((len(res_list),), dtype=object)
    for idx in range(len(res_list)):
        array[idx] = res_list[idx]

    return array


def eager_jax(res_list, actx: EagerJAXArrayContext):
    """This returns a :class:`jaxlib.xla_extensions.DeviceArrayBase`"""

    if all_numbers(res_list, actx):
        return actx.np.stack(res_list)

    array = actx.np.array(res_list)

    for idx in range(len(res_list)):
        array = array.at[idx].set(res_list[idx])

    return array


def pytato_jax(res_list, actx: PytatoJAXArrayContext):
    """This returns a :class:`pytato.array.Stack`"""
    return actx.np.stack(res_list)


def torch(res_list, actx: TorchArrayContext):
    """This returns a :class:`torch.Tensor`"""

    if all_numbers(res_list, actx):
        return actx.np.stack(res_list)

    for idx in range(len(res_list)):
        res_list[idx] = res_list[idx].squeeze()

    return actx.np.stack(res_list, 0)

# }}}
