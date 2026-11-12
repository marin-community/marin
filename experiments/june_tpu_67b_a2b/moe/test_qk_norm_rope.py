# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's Pallas q/k head norm and half-RoPE (``qk_norm_rope="pallas"``) against the separate ops."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


@pytest.mark.timeout(300)
@pytest.mark.multidevice
@pytest.mark.parametrize(
    "variant",
    [
        {},
        # The long layers' own options and per-example positions.
        {"disable_long_rope": True, "qk_mult_long_scale": 1.5, "position_batch": 4},
        # One position table shared by the whole batch, which is split over devices.
        {"position_batch": 1},
    ],
)
def test_pallas_qk_norm_rope_trains_like_the_separate_ops(variant):
    # Batch and heads are sharded, so the kernels run per device under the shard_map.
    variant = dict(variant)
    position_batch = variant.pop("position_batch", 0)
    cfg = GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=16,
        sliding_window=8,
        capacity_factor=8.0,
        attention_implementation="reference",
        use_array_stacked_blocks=True,
        disable_pko=True,
        **variant,
    )
    with use_test_devices(4):
        tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
        target = jax.random.normal(jax.random.key(3), (4, cfg.max_seq_len, cfg.hidden_dim))
        position_ids = None
        if position_batch:
            position_ids = jax.random.randint(jax.random.key(2), (position_batch, cfg.max_seq_len), 0, 100)

        def loss_and_grads(config):
            with jax.set_mesh(compact_grug_mesh(model_axis_size=2)):
                params, static = eqx.partition(Transformer.init(config, key=jax.random.key(0)), eqx.is_array)

                def loss(p):
                    hidden, _ = eqx.combine(p, static)(tokens, position_ids=position_ids)
                    return jnp.mean(hidden * target)

                fn = jax.jit(jax.value_and_grad(loss))
                jaxpr = str(jax.make_jaxpr(fn)(params))
                value, grads = fn(params)
            return float(value), [np.asarray(g, np.float64) for g in jax.tree.leaves(grads)], jaxpr

        base_loss, base_grads, _ = loss_and_grads(cfg)
        loss, grads, jaxpr = loss_and_grads(dataclasses.replace(cfg, qk_norm_rope="pallas"))
    assert "shard_map" in jaxpr
    np.testing.assert_allclose(loss, base_loss, rtol=1e-5)
    for g, base in zip(grads, base_grads, strict=True):
        assert np.linalg.norm(g - base) <= 1e-4 * max(np.linalg.norm(base), 1e-30), (
            np.linalg.norm(g - base),
            np.linalg.norm(base),
        )
