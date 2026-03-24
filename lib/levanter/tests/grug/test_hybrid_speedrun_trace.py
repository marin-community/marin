# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax.partitioning import named_jit, set_mesh
from jax.sharding import AxisType, Mesh

from experiments.grug.hybrid_mamba3.model import (
    A_INIT_MAX,
    A_INIT_MIN,
    DT_INIT_FLOOR,
    DT_INIT_MAX,
    DT_INIT_MIN,
    Mamba3MimoMixer,
    Mamba3SisoMixer,
    Transformer,
)
from experiments.speedrun.grug_hybrid_mamba3_sweep import (
    _build_model_config,
    _build_sweep,
    _build_train_config,
)
from levanter.grug.attention import AttentionMask
from levanter.optim import AdamConfig


def _make_explicit_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize(
    ("pattern_label", "linear_mode"),
    [
        ("swa3full1", "siso"),
        ("swa-linear-swa-full", "siso"),
        ("swa-linear-swa-full", "mimo"),
    ],
)
def test_hybrid_grug_native_transformer_traces_next_token_loss(pattern_label: str, linear_mode: str):
    cfg = _build_model_config(
        width_label="d512",
        pattern_label=pattern_label,
        sliding_window=1024,
        linear_mode=linear_mode,
    )
    Batch = hax.Axis("batch", 1)
    Pos = hax.Axis("position", cfg.max_seq_len)
    token_ids = hax.named(jnp.zeros((Batch.size, Pos.size), dtype=jnp.int32), (Batch, Pos)).array
    loss_weight = hax.named(jnp.ones((Batch.size, Pos.size), dtype=jnp.float32), (Batch, Pos)).array
    mask = AttentionMask.causal()

    def compute_loss(model: Transformer, token_ids, loss_weight, mask):
        return model.next_token_loss(token_ids, loss_weight, mask=mask)

    axis_mapping = {
        "batch": "data",
        "embed": "data",
        "heads": "model",
        "mlp": "model",
        "token": "data",
        "token_repeat": "data",
    }

    with set_mesh(_make_explicit_mesh()):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        assert hasattr(model, "token_embed")
        jaxpr, _, _ = eqx.filter_make_jaxpr(
            named_jit(compute_loss, axis_resources=axis_mapping, out_axis_resources=axis_mapping)
        )(model, token_ids, loss_weight, mask)
        assert jaxpr is not None
        assert hasattr(model, "token_embed")


@pytest.mark.parametrize("linear_mode", ["siso", "mimo"])
def test_hybrid_linear_mixer_uses_mamba_style_dt_and_a_init(linear_mode: str):
    cfg = _build_model_config(
        width_label="d512",
        pattern_label="swa-linear-swa-full",
        sliding_window=1024,
        linear_mode=linear_mode,
    )
    mixer_cls = Mamba3SisoMixer if linear_mode == "siso" else Mamba3MimoMixer
    with set_mesh(_make_explicit_mesh()):
        mixer = mixer_cls.init(cfg, key=jax.random.PRNGKey(0))

    dt = jax.nn.softplus(mixer.dt_bias)
    a = jax.nn.softplus(mixer.a_bias)

    assert jnp.all(dt >= DT_INIT_FLOOR)
    assert jnp.all(dt >= DT_INIT_MIN)
    assert jnp.all(dt <= DT_INIT_MAX)
    assert jnp.all(a >= A_INIT_MIN)
    assert jnp.all(a <= A_INIT_MAX)
    assert jnp.all(mixer.d_skip == 1.0)


@pytest.mark.parametrize("linear_mode", ["siso", "mimo"])
def test_hybrid_linear_mixer_fused_in_proj_path_stays_finite(linear_mode: str):
    cfg = _build_model_config(
        width_label="d512",
        pattern_label="swa-linear-swa-full",
        sliding_window=1024,
        linear_mode=linear_mode,
    )
    mixer_cls = Mamba3SisoMixer if linear_mode == "siso" else Mamba3MimoMixer
    x = jnp.ones((2, cfg.max_seq_len, cfg.hidden_dim), dtype=jnp.bfloat16)
    with set_mesh(_make_explicit_mesh()):
        mixer = mixer_cls.init(cfg, key=jax.random.PRNGKey(0))
    projected = jnp.einsum("bsh,hd->bsd", x, mixer.w_in)

    assert hasattr(mixer, "w_in")
    assert hasattr(mixer, "out_norm")
    assert projected.shape[0] == x.shape[0]
    assert projected.shape[1] == x.shape[1]
    assert jnp.isfinite(projected).all()
    with set_mesh(_make_explicit_mesh()):
        if linear_mode == "siso":
            y = mixer.out_norm(
                jnp.ones((x.shape[0], x.shape[1], cfg.num_heads, cfg.head_dim), dtype=jnp.bfloat16),
                jnp.ones((x.shape[0], x.shape[1], cfg.num_heads, cfg.head_dim), dtype=jnp.bfloat16),
            )
        else:
            y = mixer.out_norm(
                jnp.ones((x.shape[0], x.shape[1], cfg.linear_rank, cfg.num_heads, cfg.head_dim), dtype=jnp.bfloat16),
                jnp.ones((x.shape[0], x.shape[1], cfg.linear_rank, cfg.num_heads, cfg.head_dim), dtype=jnp.bfloat16),
            )
    assert jnp.isfinite(y).all()


def test_hybrid_mimo_mixer_uses_structured_rank_init():
    cfg = _build_model_config(
        width_label="d512",
        pattern_label="swa-linear-swa-full",
        sliding_window=1024,
        linear_mode="mimo",
    )
    with set_mesh(_make_explicit_mesh()):
        mixer = Mamba3MimoMixer.init(cfg, key=jax.random.PRNGKey(0))

    assert mixer.w_rank_x.shape == (cfg.num_heads, cfg.linear_rank, cfg.head_dim)
    assert mixer.w_rank_z.shape == (cfg.num_heads, cfg.linear_rank, cfg.head_dim)
    assert mixer.w_rank_o.shape == (cfg.num_heads, cfg.linear_rank, cfg.head_dim)
    np.testing.assert_allclose(np.asarray(mixer.w_rank_x), 1.0 / cfg.linear_rank)
    np.testing.assert_allclose(np.asarray(mixer.w_rank_z), 1.0)
    np.testing.assert_allclose(np.asarray(mixer.w_rank_o), 1.0 / cfg.linear_rank)


def test_hybrid_mimo_grug_native_transformer_traces_next_token_loss_with_batch_gt_one():
    cfg = _build_model_config(
        width_label="d512",
        pattern_label="swa-linear-swa-full",
        sliding_window=1024,
        linear_mode="mimo",
    )
    Batch = hax.Axis("batch", 2)
    Pos = hax.Axis("position", cfg.max_seq_len)
    token_ids = hax.named(jnp.zeros((Batch.size, Pos.size), dtype=jnp.int32), (Batch, Pos)).array
    loss_weight = hax.named(jnp.ones((Batch.size, Pos.size), dtype=jnp.float32), (Batch, Pos)).array
    mask = AttentionMask.causal()

    def compute_loss(model: Transformer, token_ids, loss_weight, mask):
        return model.next_token_loss(token_ids, loss_weight, mask=mask)

    axis_mapping = {
        "batch": "data",
        "embed": "data",
        "heads": "model",
        "mlp": "model",
        "token": "data",
        "token_repeat": "data",
    }

    with set_mesh(_make_explicit_mesh()):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        jaxpr, _, _ = eqx.filter_make_jaxpr(
            named_jit(compute_loss, axis_resources=axis_mapping, out_axis_resources=axis_mapping)
        )(model, token_ids, loss_weight, mask)
        assert jaxpr is not None


def test_linear3full1_uses_olmo_style_layer_pattern_and_no_sliding_window_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GRUG_HYBRID_WIDTHS", "d512")
    monkeypatch.setenv("GRUG_HYBRID_PATTERNS", "linear3full1")
    monkeypatch.setenv("GRUG_HYBRID_LINEAR_MODES", "siso")
    monkeypatch.setenv("GRUG_HYBRID_SLIDING_WINDOWS", "128,1024")

    model_cfg = _build_model_config(
        width_label="d512",
        pattern_label="linear3full1",
        sliding_window=128,
        linear_mode="siso",
    )
    assert (
        model_cfg.layer_types
        == (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
        * 3
    )

    sweep = _build_sweep()
    assert len(sweep) == 1
    run_name, sweep_cfg = sweep[0]
    assert run_name == "grug-hybrid-d512-linear3full1-siso"
    assert sweep_cfg.sliding_window == 128


def test_hybrid_sweep_uses_adamw_train_config():
    model_cfg = _build_model_config(
        width_label="d512",
        pattern_label="swa-linear-swa-full",
        sliding_window=1024,
        linear_mode="mimo",
    )
    train_cfg = _build_train_config(model_cfg)
    assert isinstance(train_cfg.optimizer_config, AdamConfig)
    assert train_cfg.optimizer_config.weight_decay > 0
    assert train_cfg.train_batch_size == 192
    assert train_cfg.optimizer_config.warmup >= 1024
    assert train_cfg.optimizer_config.max_grad_norm <= 0.5
    assert train_cfg.optimizer_config.learning_rate == pytest.approx(0.001)


def test_hybrid_linear3full1_mimo_uses_more_conservative_batch_and_optimizer():
    model_cfg = _build_model_config(
        width_label="d512",
        pattern_label="linear3full1",
        sliding_window=1024,
        linear_mode="mimo",
    )
    train_cfg = _build_train_config(model_cfg)
    assert isinstance(train_cfg.optimizer_config, AdamConfig)
    assert train_cfg.train_batch_size == 128
    assert train_cfg.optimizer_config.warmup >= 1536
    assert train_cfg.optimizer_config.max_grad_norm <= 0.4
    assert train_cfg.optimizer_config.learning_rate == pytest.approx(0.000875)
