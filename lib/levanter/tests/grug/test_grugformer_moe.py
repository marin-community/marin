# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh

from jax._src import config as jax_config

from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import GrugMoeModelConfig, Transformer, moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


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
        model = Transformer.init(cfg, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, cfg.vocab_size)

        logits = model.logits(tokens, mask=AttentionMask.causal())
        assert logits.shape == (batch, seq, cfg.vocab_size)

        jit_forward = jax.jit(lambda m, t: m.logits(t, mask=AttentionMask.causal()))
        logits_jit = jit_forward(model, tokens)
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
        model = jax.eval_shape(lambda k: Transformer.init(cfg, key=k), key)

        batch = 8
        seq = 128
        tokens = jax.ShapeDtypeStruct(
            shape=(batch, seq),
            dtype=jnp.int32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )

        def f(m, token_ids):
            return m.logits(token_ids, mask=AttentionMask.causal())

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = jax.jit(f).trace(model, tokens).lower(lowering_platforms=(platform,))
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
        model = Transformer.init(cfg, key=jax.random.key(2))
        tokens = jax.random.randint(jax.random.key(3), (batch, seq), 0, cfg.vocab_size)
        logits = model.logits(tokens, mask=AttentionMask.causal())
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
        model = Transformer.init(cfg, key=jax.random.key(4))
        tokens = jax.random.randint(jax.random.key(5), (batch, seq), 0, cfg.vocab_size)
        weights = jnp.ones((batch, seq), dtype=jnp.float32).at[:, -1].set(0)

        loss = model.next_token_loss(tokens, weights, mask=AttentionMask.causal(), reduction="mean")
        assert jnp.isfinite(loss)
        assert loss.shape == ()


def test_moe_without_shared_expert_uses_none_weights():
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices())
    cfg = GrugMoeModelConfig(
        vocab_size=73,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=seq,
    )

    mesh = _make_dense_mesh()
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(6))
        assert model.blocks[0].mlp.shared_w13 is None
        assert model.blocks[0].mlp.shared_w2 is None

        tokens = jax.random.randint(jax.random.key(7), (batch, seq), 0, cfg.vocab_size)
        logits = model.logits(tokens, mask=AttentionMask.causal())
        assert logits.shape == (batch, seq, cfg.vocab_size)


def test_functional_moe_mlp_accepts_literal_and_callable_activation():
    cfg = GrugMoeModelConfig(
        vocab_size=67,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=0,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=8,
    )
    batch = 4
    seq = 4
    x = jax.random.normal(jax.random.key(13), (batch, seq, cfg.hidden_dim), dtype=jnp.float32)
    router = jax.random.normal(jax.random.key(14), (cfg.hidden_dim, cfg.num_experts), dtype=jnp.float32)
    w13 = jax.random.normal(
        jax.random.key(15), (cfg.num_experts, cfg.hidden_dim, 2 * cfg.intermediate_dim), dtype=jnp.float32
    )
    w2 = jax.random.normal(
        jax.random.key(16), (cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim), dtype=jnp.float32
    )

    y_enum = moe_mlp(
        x,
        router,
        w13,
        w2,
        num_experts_per_token=cfg.num_experts_per_token,
        activation=ActivationFunctionEnum.silu,
        mesh=None,
    )
    y_callable = moe_mlp(
        x,
        router,
        w13,
        w2,
        num_experts_per_token=cfg.num_experts_per_token,
        activation=lambda t: jax.nn.silu(t),
        mesh=None,
    )
    np.testing.assert_allclose(np.asarray(y_callable), np.asarray(y_enum), rtol=1e-5, atol=1e-5)
