# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask
from levanter.testing.devices import use_test_devices

from experiments.grug.moe_hero_ep import model


def _mesh(devices, *, context, expert):
    return Mesh(
        np.asarray(devices).reshape(1, 1, expert, context, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _tiny_config(**overrides):
    fields = dict(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    return model.GrugModelConfig(**(fields | overrides))


@pytest.mark.multidevice
def test_context_parallel_logits_match_unsharded_sequence():
    # The MLPs flatten each device's own (rows, sequence shard) block, so under context parallelism
    # the experts see tokens in a different order. With capacity for every token nothing drops, and
    # the logits must equal the unsharded-sequence run on the same parameters.
    with use_test_devices(4) as devices:
        config = _tiny_config()
        tokens = np.random.default_rng(0).integers(0, config.vocab_size, size=(4, 8), dtype=np.int32)
        logits = {}
        for context, expert in [(1, 4), (2, 2)]:
            with jax.set_mesh(_mesh(devices, context=context, expert=expert)):
                transformer = model.Transformer.init(config, key=jax.random.key(0))
                logits[context] = np.asarray(jax.jit(lambda m, ids: m.logits(ids))(transformer, jnp.asarray(tokens)))
        np.testing.assert_allclose(logits[2], logits[1], rtol=1e-5, atol=1e-5)


@pytest.mark.multidevice
def test_log_scaling_context_parallel_logits_match_unsharded_sequence():
    # The per-row scale reads global key positions. A scale built from each shard's local positions
    # would leave CP4's later shards below the floor and diverge from CP1.
    with use_test_devices(4) as devices:
        config = _tiny_config(max_seq_len=16, capacity_factor=8.0, log_scaling_n_floor=4)
        tokens = np.random.default_rng(0).integers(0, config.vocab_size, size=(4, 16), dtype=np.int32)
        # One row is unpacked, and the rest carry a document boundary or padding off a shard boundary.
        segment_ids = np.zeros((4, 16), dtype=np.int32)
        segment_ids[1, 3:] = 1
        segment_ids[2, 10:] = 1
        segment_ids[3, 13:] = -1
        mask = AttentionMask.causal().with_segment_ids(jnp.asarray(segment_ids))

        def logits(cfg, context, expert):
            with jax.set_mesh(_mesh(devices, context=context, expert=expert)):
                transformer = model.Transformer.init(cfg, key=jax.random.key(0))
                return np.asarray(jax.jit(lambda m, ids: m.logits(ids, mask))(transformer, jnp.asarray(tokens)))

        cp4 = logits(config, context=4, expert=1)
        cp1 = logits(config, context=1, expert=4)
        off = logits(dataclasses.replace(config, log_scaling_n_floor=None), context=4, expert=1)

    np.testing.assert_allclose(cp4, cp1, rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(cp4 - off)) > 1e-3
