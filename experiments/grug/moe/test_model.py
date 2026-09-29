# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer


def test_fast_cross_entropy_backward_preserves_model_loss_and_gradients():
    # More than 128 vocabulary entries exercise the streaming custom VJP.
    config = GrugModelConfig(
        vocab_size=129,
        hidden_dim=16,
        intermediate_dim=32,
        shared_expert_intermediate_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        max_seq_len=8,
        sliding_window=4,
    )
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    weights = jnp.ones_like(tokens, dtype=jnp.float32)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        default_model = Transformer.init(config, key=jax.random.PRNGKey(0))
        fast_model = Transformer.init(
            dataclasses.replace(config, cross_entropy_implementation="xla_fast_bwd"), key=jax.random.PRNGKey(0)
        )
        loss_and_grad = eqx.filter_value_and_grad(lambda model: model.next_token_loss(tokens, weights))
        default_loss, default_grad = loss_and_grad(default_model)
        fast_loss, fast_grad = loss_and_grad(fast_model)

    np.testing.assert_array_equal(fast_loss, default_loss)
    np.testing.assert_allclose(fast_grad.output_proj, default_grad.output_proj, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fast_grad.token_embed, default_grad.token_embed, rtol=1e-5, atol=1e-5)
