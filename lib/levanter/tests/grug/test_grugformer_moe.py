# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh

from jax._src import config as jax_config

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


def _make_dense_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_abstract_moe_mesh(*, data: int, expert: int, model: int) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(data, expert, model),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def test_moe_forward_shapes_and_jit_compile_with_ep_mesh():
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


def test_moe_ring_ep_path_lowers_on_abstract_mesh():
    cfg = GrugMoeModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=128,
        shared_expert_intermediate_dim=64,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=128,
    )

    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        key = jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32, sharding=NamedSharding(mesh, P()))
        params = jax.eval_shape(lambda k: init_parameters(cfg, key=k), key)

        batch = 8
        seq = 128
        tokens = jax.ShapeDtypeStruct(
            shape=(batch, seq),
            dtype=jnp.int32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )

        def f(p, token_ids):
            return forward(p, token_ids, cfg, mask=AttentionMask.causal())

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = jax.jit(f).trace(params, tokens).lower(lowering_platforms=(platform,))
        assert lowered is not None


def test_moe_forward_runs_without_ep_axis():
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices()) * 2
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

    mesh = _make_dense_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(2))
        tokens = jax.random.randint(jax.random.key(3), (batch, seq), 0, cfg.vocab_size)
        logits = forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits.shape == (batch, seq, cfg.vocab_size)


def test_moe_loss_fn_runs():
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices())
    cfg = GrugMoeModelConfig(
        vocab_size=89,
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
        params = init_parameters(cfg, key=jax.random.key(4))
        tokens = jax.random.randint(jax.random.key(5), (batch, seq), 0, cfg.vocab_size)
        weights = jnp.ones((batch, seq), dtype=jnp.float32).at[:, -1].set(0)

        loss = loss_fn(params, tokens, weights, cfg, mask=AttentionMask.causal(), reduction="mean")
        assert jnp.isfinite(loss)
        assert loss.shape == ()
