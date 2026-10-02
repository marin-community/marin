# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for serving-format conversion of QB router state."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.science_sft.export_dr_doom import _serving_model, _snowball_config
from experiments.grug.science_sft.train import _apply_qb_betas
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def test_serving_export_materializes_effective_qb_router_bias():
    training = GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        sliding_window=8,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(training, key=jax.random.key(0))
        qb_betas = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        exported = _serving_model(model, qb_betas, _snowball_config(training))
        effective = _apply_qb_betas(model, qb_betas)

    assert exported.transformer.blocks is not None
    assert effective.stacked_blocks is not None
    expected = np.asarray(effective.stacked_blocks.stacked.mlp.router_bias)
    actual = np.stack([np.asarray(block.mlp.router_bias) for block in exported.transformer.blocks])
    np.testing.assert_array_equal(actual, expected)
