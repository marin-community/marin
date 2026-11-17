# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``attn_res_global_kv_query``: the global layers' KV latent reads its own AttnRes gate, which starts as
the attention gate's mix and trains apart from it."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer


def test_global_kv_query_starts_as_the_attention_mix_and_steers_the_kv_latent():
    config = dataclasses.replace(tiny_config(), attn_res_global_kv_query=True)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, attn_res_global_kv_query=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-init queries: the KV gate's mix equals the attention gate's at step 0.
        np.testing.assert_allclose(forward(model, tokens), forward(plain, tokens), rtol=1e-6, atol=1e-6)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)

        # Moving only the global layers' KV queries changes the output.
        kv_queries = model.stacked_blocks.stacked.attn_res_query_kv
        moved = eqx.tree_at(
            lambda m: m.stacked_blocks.stacked.attn_res_query_kv,
            model,
            jax.random.normal(jax.random.key(2), kv_queries.shape),
        )
        moved_out = forward(moved, tokens)
        base_out = forward(model, tokens)

    assert model.kda_blocks is not None and model.kda_blocks.stacked.attn_res_query_kv is None
    assert np.isfinite(float(loss))
    kv_grad = grads.stacked_blocks.stacked.attn_res_query_kv
    assert kv_grad.shape == kv_queries.shape
    assert all(float(jnp.linalg.norm(row)) > 0 for row in kv_grad)
    assert float(jnp.max(jnp.abs(moved_out - base_out))) > 1e-3
