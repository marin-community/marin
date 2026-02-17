# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh

from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import GrugMoeModelConfig, forward, init_parameters, loss_fn


def _make_moe_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(len(devices), 1, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def test_moe_forward_shapes_and_jit_compile():
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices()) * 2
    cfg = GrugMoeModelConfig(
        vocab_size=101,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=seq,
    )

    mesh = _make_moe_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, cfg.vocab_size)

        logits = forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits.shape == (batch, seq, cfg.vocab_size)

        jit_forward = jax.jit(forward, static_argnames=("cfg",))
        logits_jit = jit_forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits_jit.shape == logits.shape


def test_moe_loss_fn_runs():
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices())
    cfg = GrugMoeModelConfig(
        vocab_size=97,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=seq,
    )

    mesh = _make_moe_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(2))
        tokens = jax.random.randint(jax.random.key(3), (batch, seq), 0, cfg.vocab_size)
        weights = jnp.ones((batch, seq), dtype=jnp.float32).at[:, -1].set(0)

        loss = loss_fn(params, tokens, weights, cfg, mask=AttentionMask.causal(), reduction="mean")
        assert jnp.isfinite(loss)
        assert loss.shape == ()
