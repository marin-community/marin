# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable vanilla-JAX oracle for the depthwise causal short convolution (SConv).

This is a verbatim copy of the pad-and-shift body that ``experiments/grug/*/model.py``
``ShortConv.__call__`` has always used. It is the correctness oracle *and* the
performance baseline: every other implementation must reproduce it bit for bit on the
forward pass, and every benchmark reports against its timings.

Do not "clean this up". Its exact op sequence -- ``pad`` then ``slice``, a ``where``
against the shifted ``segment_ids``, a bf16-rounded multiply, a bf16-rounded add, in
lag order -- fixes the rounding that the fused kernel is required to match.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def short_conv_reference(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"] | None = None,
) -> Float[Array, "B S C"]:
    """``out[b,t,c] = sum_lag weight[lag,c] * x[b,t-lag,c]``, zeroed across segments.

    A tap that reaches into a previous document is dropped (the shifted ``segment_ids``
    no longer match); the lag-0 (current token) tap is always kept. Positions before the
    start of the sequence read as zero.
    """
    seq_len = x.shape[1]
    out = weight[0] * x
    for lag in range(1, weight.shape[0]):
        shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :seq_len, :]
        if segment_ids is not None:
            seg_shifted = jnp.pad(segment_ids, ((0, 0), (lag, 0)), constant_values=-1)[:, :seq_len]
            shifted = jnp.where((seg_shifted == segment_ids)[..., None], shifted, 0.0)
        out = out + weight[lag] * shifted
    return out
