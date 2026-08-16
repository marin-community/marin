# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

from functools import lru_cache, partial
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from .backward_kernel import _backward_core
from .forward_kernel import _ceil_div, _forward_core
from .reference import _depthwise_causal_convolution_reference

Implementation: TypeAlias = Literal["pallas_tpu", "xla"]


def _get_block_size_s(H: int) -> int:
    # the kernel stack holds several (BLOCK_SIZE_S, H) fp32 tiles and vmem overflows once
    # BLOCK_SIZE_S * H exceeds 2**19 (measured), so shrink the block for wide H. Narrow H
    # wants the block as big as possible instead: more rows per program means fewer HBM
    # round-trips, which is what sets the kernel's bandwidth at small hidden sizes
    block_size = 1 << ((1 << 19) // H).bit_length() - 1
    return min(1024, max(8, block_size))


def _pad_h0(h0: jax.Array, K: int) -> jax.Array:
    state_size = K - 1
    pad = _ceil_div(state_size, 8) * 8
    return jnp.pad(h0, ((0, 0), (pad - state_size, 0), (0, 0)))


def _forward_run(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    W = jnp.transpose(W, (1, 0))
    b = None if b is None else b[None, :]

    if h0 is not None:
        h0 = jnp.transpose(h0, (0, 2, 1)).astype(x.dtype)
        h0 = _pad_h0(h0, K=W.shape[0])

    # the per-block input states are saved from the forward as vjp residuals (a small
    # (B, ceil(S / BLOCK_SIZE_S), PAD, H) tensor) so the backward can consume them directly
    # instead of re-deriving them with an extra pass over the input
    y, h = _forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=_get_block_size_s(x.shape[-1]), ACTIVATION=ACTIVATION)

    if output_state:
        state_size = W.shape[0] - 1
        if h0 is None:
            ht = (
                jnp.pad(x, ((0, 0), (state_size - x.shape[1], 0), (0, 0)))
                if x.shape[1] < state_size
                else x[:, -state_size:, :]
            )
        else:
            ht = jnp.concatenate([h0.astype(x.dtype), x], axis=1)[:, -state_size:, :]

        ht = jnp.transpose(ht, (0, 2, 1))
    else:
        ht = None

    return y, ht, h


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _depthwise_causal_convolution_pallas(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[jax.Array, jax.Array | None]:
    y, ht, _ = _forward_run(x=x, W=W, b=b, h0=h0, output_state=output_state, ACTIVATION=ACTIVATION)
    return y, ht


def _depthwise_causal_convolution_forward(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    y, ht, h = _forward_run(x=x, W=W, b=b, h0=h0, output_state=output_state, ACTIVATION=ACTIVATION)
    return (y, ht), (x, W, b, h0, h)


def _depthwise_causal_convolution_backward(
    output_state: bool, ACTIVATION: str | None, residuals: tuple, cotangents: tuple
) -> tuple:
    x, W, b, h0, h_states = residuals
    dy, dht = cotangents

    K = W.shape[-1]
    W = jnp.transpose(W, (1, 0))

    dht = None if dht is None or not output_state else jnp.transpose(dht, (0, 2, 1))

    dx, dW, db, dh0 = _backward_core(
        x=x,
        W=W,
        b=None if b is None else b[None, :],
        h=h_states,
        dy=dy,
        dht=dht,
        BLOCK_SIZE_S=_get_block_size_s(x.shape[-1]),
        K=K,
        ACTIVATION=ACTIVATION,
    )

    dW = jnp.transpose(dW, (1, 0))
    db = None if b is None else db[0]
    dh0 = None if h0 is None else jnp.transpose(dh0[:, 1 - K :, :], (0, 2, 1))

    return dx, dW, db, dh0


_depthwise_causal_convolution_pallas.defvjp(
    _depthwise_causal_convolution_forward, _depthwise_causal_convolution_backward
)


IMPLEMENTATIONS = {
    "pallas_tpu": _depthwise_causal_convolution_pallas,
    "xla": _depthwise_causal_convolution_reference,
}


@lru_cache(maxsize=1)
def _default_implementation() -> Implementation:
    return "pallas_tpu" if jax.default_backend() == "tpu" else "xla"


def depthwise_causal_convolution(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    input_state: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    output_state: bool = False,
    activation_function: str | None = None,
    *,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """
    computes depthwise causal 1D convolution: `output[b, t, h] = act(bias[h] + sum_k weight[h, k] *
    z[b, t + k, h])` where `z` is `input` preceded by `kernel_size - 1` raw history positions taken from
    `input_state` (or 0 if `input_state` is None), i.e. `output[b, t]` only depends on
    `input[b, t - kernel_size + 1 : t + 1]`.

    :param input: input tensor of shape (B, S, H)
    :type input: jax.Array
    :param weight: depthwise convolution weight of shape (H, K), K being the kernel size
    :type weight: jax.Array
    :param bias: bias tensor of shape (H,). None means no bias is added. Defaults to None.
    :type bias: jax.Array | None
    :param input_state: the `K - 1` raw (pre-convolution) input positions preceding `input`, of shape
        (B, H, K - 1). None is equivalent to a 0 tensor. Defaults to None.
    :type input_state: jax.Array | None
    :param attention_mask: mask of shape (B, S), zeroing out padding positions before and after the
        convolution. Defaults to None.
    :type attention_mask: jax.Array | None
    :param output_state: whether to also return the trailing `K - 1` raw input positions (taken from `input`,
        falling back to `input_state` if `input` is shorter than `K - 1`) for use as `input_state` in a
        subsequent call. Defaults to False.
    :type output_state: bool
    :param activation_function: activation applied after the convolution + bias, fused into the kernel.
        Either "silu", its alias "swish", or None (no activation). Defaults to None.
    :type activation_function: str | None
    :param implementation: "pallas_tpu" uses a hand-written VPU-only Pallas TPU kernel (avoids the
        MXU/systolic array, which a tiny `kernel_size` reduction dimension underutilizes badly). "xla" uses
        the plain `jax.lax.conv_general_dilated`-based reference. None auto-detects based on the
        accelerator ("pallas_tpu" on TPU, "xla" otherwise). Defaults to None.
    :type implementation: Implementation | None
    :return: output tensor of shape (B, S, H), and the output state of shape (B, H, K - 1) if `output_state`
        is True else None.
    :rtype: tuple[jax.Array, jax.Array | None]
    """

    B, _, H = input.shape
    K = weight.shape[-1]

    assert weight.ndim == 2
    assert weight.shape[0] == H
    assert K > 1

    assert activation_function in [None, "silu", "swish"]

    if bias is not None:
        assert bias.shape == (H,)

    if input_state is not None:
        assert input_state.shape == (B, H, K - 1)

    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        input = (input * attention_mask[:, :, None]).astype(input.dtype)

    if implementation is None:
        implementation = _default_implementation()

    if implementation == "pallas_tpu":
        input, input_state = _depthwise_causal_convolution_pallas(
            x=input,
            W=weight,
            b=bias,
            h0=input_state,
            output_state=output_state,
            ACTIVATION=activation_function,
        )
    elif implementation == "xla":
        input, input_state = _depthwise_causal_convolution_reference(
            x=input,
            W=weight,
            b=bias,
            h0=input_state,
            output_state=output_state,
            activation_function=activation_function,
        )
    else:
        raise ValueError(f"unexpected implementation ({implementation})")

    return input, input_state
