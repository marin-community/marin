# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-stack integration checks. No downloads, training jobs or Torch required."""

import draccus

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from levanter.layers.attention import AttentionBackend
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.models.lm_model import LmExample
from levanter.optim.config import OptimizerConfig
from levanter.checkpoint import save_checkpoint, load_checkpoint

from levanter.optim.randrand import RandRANDConfig, RandRANDVariant, parameter_layout, _pack, _unpack
from levanter.optim.randrand_spectral import SpectralSettings
from levanter.optim.randrand_muon_hybrid import RandRANDMuonHybridConfig, MuonBlendConfig


def tiny_model(*, scan_layers=True):
    config = Qwen3Config(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        hybrid_norm=True,
        scan_layers=scan_layers,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    )
    return Qwen3LMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(42))


@pytest.mark.parametrize("scan_layers", [False, True])
def test_axis_roundtrip_and_full_attention_projections(scan_layers):
    params = eqx.filter(tiny_model(scan_layers=scan_layers), eqx.is_inexact_array)
    layouts, tree = parameter_layout(params)
    packed = _pack(params, layouts)
    reconstructed = _unpack(packed, params, layouts, tree)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(reconstructed), strict=True):
        np.testing.assert_array_equal(a, b)
    shapes = [
        p.shape
        for p, kind in zip(packed, [layout.kind for layout in layouts for _ in range(layout.batch_size)], strict=True)
        if kind == "matrix"
    ]
    assert shapes.count((16, 16)) == 8  # Four FULL attention projections per layer.
    assert shapes.count((32, 16)) == 4
    assert shapes.count((16, 32)) == 2


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("variant", list(RandRANDVariant))
def test_configured_qwen_updates_resume_from_native_checkpoint(tmp_path, hybrid, variant):
    constructor = RandRANDMuonHybridConfig if hybrid else RandRANDConfig
    extra = dict(switch=MuonBlendConfig(min_step=2, threshold=1.0, ramp=4)) if hybrid else {}
    config = constructor(
        **extra,
        variant=variant,
        spectral=SpectralSettings(probe_every=2),
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup=2,
    )
    assert draccus.decode(OptimizerConfig, draccus.encode(config, OptimizerConfig)) == config
    model = tiny_model()
    params, static = eqx.partition(model, eqx.is_inexact_array)
    tx = config.build(10)
    state = tx.init(params)
    tokens = hax.named(jnp.arange(8, dtype=jnp.int32), hax.Axis("position", 8))
    example = LmExample.causal(tokens)

    def loss(p):
        return eqx.combine(p, static).compute_next_token_loss(example).scalar()

    @eqx.filter_jit
    def train_step(p, s):
        value, grads = eqx.filter_value_and_grad(loss)(p)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, value

    for _ in range(4):
        params, state, value = train_step(params, state)
        assert np.isfinite(float(value))
    checkpoint = str(tmp_path / "checkpoint")
    saved = dict(params=params, state=state)
    save_checkpoint(saved, step=4, checkpoint_path=checkpoint)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
    resumed = load_checkpoint(jax.tree.map(jnp.zeros_like, saved), checkpoint, mesh=mesh, axis_mapping={})
    a = train_step(params, state)
    b = train_step(resumed["params"], resumed["state"])
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    assert int(a[1].randrand.count if hybrid else a[1].count) == 5
    if hybrid:
        assert 0 < float(a[1].blend.weight) <= 1
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(a))


@pytest.mark.multidevice
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("variant", list(RandRANDVariant))
def test_named_jit_shards_adam_buffers_and_preserves_updates(hybrid, variant):
    if len(jax.devices()) < 2:
        pytest.skip("Run with two CPU devices or multiple accelerators for the sharding check")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    constructor = RandRANDMuonHybridConfig if hybrid else RandRANDConfig
    extra = dict(switch=MuonBlendConfig(min_step=1, threshold=1.0, ramp=2)) if hybrid else {}
    config = constructor(
        **extra,
        variant=variant,
        spectral=SpectralSettings(probe_every=2),
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup=2,
    )
    params = eqx.filter(tiny_model(), eqx.is_inexact_array)
    tx = config.build(10)
    state = tx.init(params)

    def step(p, s, g):
        updates, s = tx.update(g, s, p)
        return optax.apply_updates(p, updates), s

    ordinary = eqx.filter_jit(step)
    mapped = hax.named_jit(step, axis_resources={"embed": "data"})
    with jax.set_mesh(mesh):
        distributed_p = hax.shard(params, {"embed": "data"})
        distributed_s = hax.shard(state, {"embed": "data"})
        assert any(
            not a.sharding.is_fully_replicated
            for a in jax.tree.leaves(distributed_s.randrand.first if hybrid else distributed_s.first)
        )
        for i in range(4):
            grads = jax.tree.map(lambda p, i=i: jnp.sin(p + jnp.float32(i)), params)
            params, state = ordinary(params, state, grads)
            distributed_p, distributed_s = mapped(distributed_p, distributed_s, hax.shard(grads, {"embed": "data"}))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(distributed_p), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        assert int(distributed_s.randrand.count if hybrid else distributed_s.count) == 4
        if hybrid:
            assert float(distributed_s.blend.weight) > 0
            for buffers in (distributed_s.momentum, distributed_s.previous_update):
                assert any(not a.sharding.is_fully_replicated for a in jax.tree.leaves(buffers))
            np.testing.assert_allclose(distributed_s.blend.weight, state.blend.weight, atol=1e-5, rtol=1e-5)
        assert any(
            not a.sharding.is_fully_replicated
            for a in jax.tree.leaves(distributed_s.randrand.first if hybrid else distributed_s.first)
        )
