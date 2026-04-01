# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax.sharding import AxisType

from experiments.grug.base.model import GrugModelConfig, Transformer
from levanter.backward_metrics import empty_sink, grad_rms_from_sink
from levanter.grug.attention import AttentionMask


def test_grug_next_token_loss_backward_sink():
    cfg = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=8,
    )
    mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2, cfg.max_seq_len), 0, cfg.vocab_size)
        loss_weight = jnp.ones((2, cfg.max_seq_len), dtype=jnp.float32)
        sink = empty_sink("grad_sumsq", "grad_count")

        def loss_fn(model, sink):
            return model.next_token_loss(
                tokens,
                loss_weight,
                mask=AttentionMask.causal(),
                backward_sink=sink,
            )

        loss, (_, backward_metrics) = jax.value_and_grad(loss_fn, argnums=(0, 1))(model, sink)

        assert jnp.isfinite(loss)
        assert backward_metrics["grad_sumsq"] > 0
        assert jnp.allclose(backward_metrics["grad_count"], float(tokens.size * cfg.hidden_dim))

        grad_rms = grad_rms_from_sink(backward_metrics)
        assert jnp.isfinite(grad_rms)
        assert grad_rms > 0
