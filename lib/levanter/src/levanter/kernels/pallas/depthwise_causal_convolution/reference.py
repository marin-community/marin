# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

import jax
import jax.numpy as jnp


def _depthwise_causal_convolution_reference(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    activation_function: str | None,
) -> tuple[jax.Array, jax.Array | None]:
    _, S, H = x.shape
    K = W.shape[-1]
    state_size = K - 1

    x = jnp.transpose(x, (0, 2, 1))
    ht = None

    if h0 is None:
        padding = [(K - 1, 0)]

        if output_state:
            ht = jnp.pad(x, ((0, 0), (0, 0), (state_size - S, 0))) if S < state_size else x[:, :, -state_size:]
    else:
        padding = [(0, 0)]

        x = jnp.concatenate([h0.astype(x.dtype), x], axis=-1)

        if output_state:
            ht = x[:, :, -state_size:]

    x = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=W[:, None, :],
        window_strides=(1,),
        padding=padding,
        feature_group_count=H,
        dimension_numbers=("NCH", "OIH", "NCH"),
    )

    if b is not None:
        x = x + b[None, :, None]

    x = jnp.transpose(x, (0, 2, 1))
    if activation_function in ["silu", "swish"]:
        x = jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)

    return x, ht
