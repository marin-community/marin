# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the saved-logit loss and VJP with the existing streaming implementation."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.kernels.pallas.fused_cross_entropy_loss import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import linear_softmax_cross_entropy_loss_xla

from experiments.grug.belay_tpu.saved_ce_tpu import saved_linear_cross_entropy


def evaluate(function, x, labels, w, cotangent):
    output, pullback = jax.vjp(lambda a, b: function(a, labels, b), x, w)
    return output, pullback(cotangent)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("cap", [None, 10.0])
@pytest.mark.parametrize("vocab,scale", [(256, 0.1), (384, 0.1), (256, 2.0)])
def test_saved_logits_match_streaming_loss_and_input_weight_gradients(dtype, cap, vocab, scale):
    rng = np.random.RandomState(181)
    x = jnp.asarray(rng.normal(size=(64, 32)), dtype)
    w = jnp.asarray(scale * rng.normal(size=(32, vocab)), dtype)
    labels = jnp.asarray(rng.randint(0, vocab, size=64), jnp.int32)
    g_loss = jnp.asarray(rng.normal(size=64), jnp.float32).at[::3].set(0)
    g_lse = jnp.asarray(rng.normal(size=64), jnp.float32)
    source = functools.partial(
        linear_softmax_cross_entropy_loss_xla,
        block_sizes=BlockSizes(b_block_size=64, v_block_size=128),
        fast_backward=True,
        precision="highest",
        logit_soft_cap=cap,
    )
    candidate = functools.partial(saved_linear_cross_entropy, block_size=128, precision="highest", logit_soft_cap=cap)
    expected = jax.jit(functools.partial(evaluate, source))(x, labels, w, (g_loss, g_lse))
    actual = jax.jit(functools.partial(evaluate, candidate))(x, labels, w, (g_loss, g_lse))
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)
