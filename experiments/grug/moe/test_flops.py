# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import _compute_flops


@pytest.fixture
def model_config():
    return GrugModelConfig(
        vocab_size=64,
        hidden_dim=64,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=32,
        sliding_window=4,
    )


@pytest.mark.parametrize(
    ("num_layers", "expected_forward_flops"),
    [(1, 78720), (2, 141744), (4, 267792), (5, 338320), (11, 723968)],
)
def test_moe_flops_counts_windowed_layers_and_final_full_layer(model_config, num_layers, expected_forward_flops):
    # This shape costs 63,024 FLOPs per local layer, 70,528 per full layer,
    # and 8,192 for the LM head. Full layers are every fourth layer and the last.
    cfg = dataclasses.replace(model_config, num_layers=num_layers)
    flops_per_example, summary = _compute_flops(model_config=cfg)

    assert summary["throughput/flops_per_token_analytic"] == expected_forward_flops
    assert flops_per_example == 3 * cfg.max_seq_len * expected_forward_flops
    assert summary["throughput/flops_per_example_analytic"] == flops_per_example


@pytest.mark.parametrize("window", [32, 64])
def test_moe_flops_window_covering_context_counts_full_attention(model_config, window):
    cfg = dataclasses.replace(model_config, num_layers=5, sliding_window=window)
    _, summary = _compute_flops(model_config=cfg)

    assert summary["throughput/flops_per_token_analytic"] == 360832


@pytest.mark.parametrize("num_layers", [1, 2])
def test_moe_forward_window_changes_only_models_with_local_layers(model_config, num_layers):
    # The final layer is always global. Only the two-layer model has a local layer.
    cfg = dataclasses.replace(model_config, num_layers=num_layers)
    tokens = jnp.broadcast_to(jnp.arange(cfg.max_seq_len, dtype=jnp.int32), (jax.device_count(), cfg.max_seq_len))
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    full_config = dataclasses.replace(cfg, sliding_window=cfg.max_seq_len)

    with jax.set_mesh(mesh):
        windowed, _ = Transformer.init(cfg, key=jax.random.PRNGKey(0))(tokens)
        full, _ = Transformer.init(full_config, key=jax.random.PRNGKey(0))(tokens)

    if num_layers == 1:
        assert jnp.array_equal(windowed, full)
    else:
        assert not jnp.array_equal(windowed, full)
