
import jax
import jax.numpy as jnp


def block_thomas(lower, central, upper, rhs):

    # num_blocks = central.shape[0]

    # forward pass: compute central_tmp, rhs_tmp
    central_init = central[0]
    rhs_init = rhs[0]

    def fwd_step(carry, inputs):
        central_prev, rhs_prev = carry
        my_lower, my_central, my_upper, my_rhs = inputs

        s = jnp.linalg.solve(central_prev, my_upper)
        t = jnp.linalg.solve(central_prev, rhs_prev)

        central_tmp = my_central - my_lower @ s
        rhs_tmp = my_rhs - my_lower @ t

        new_carry = (central_tmp, rhs_tmp)
        outputs = (central_tmp, rhs_tmp)
        return new_carry, outputs

    inputs = (lower, central[1:], upper, rhs[1:])
    carry_init = (central_init, rhs_init)

    (_, _), (central_rest_tmp, rhs_rest_tmp) = jax.lax.scan(
        fwd_step,
        carry_init,
        inputs,
    )

    central_tmp = jnp.concatenate(
        [central_init[None, ...], central_rest_tmp],
        axis=0,
    )  # (num_blocks, nv, nv)
    rhs_tmp = jnp.concatenate(
        [rhs_init[None, ...], rhs_rest_tmp],
        axis=0,
    )  # (num_blocks, nv)

    # backward pass: solve for x
    x_last = jnp.linalg.solve(central_tmp[-1], rhs_tmp[-1])  # x[num_blocks-1]

    def bwd_step(x_next, inputs):
        central_i_tmp, my_upper, rhs_i_tmp = inputs
        x_i = jnp.linalg.solve(central_i_tmp, rhs_i_tmp - my_upper @ x_next)
        return x_i, x_i

    rev_inputs = (
        central_tmp[:-1][::-1],
        upper[::-1],
        rhs_tmp[:-1][::-1],
    )

    _, x_rev = jax.lax.scan(bwd_step, x_last, rev_inputs)

    x = jnp.concatenate([x_rev[::-1], x_last[None, :]], axis=0)
    return x
