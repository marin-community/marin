# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import model as moe_model
from experiments.grug.moe.heuristic import build_from_heuristic


class _Identity(eqx.Module):
    def __call__(self, x):
        return x


class _ConstantAttention(eqx.Module):
    value: float = eqx.field(static=True)

    def __call__(self, x, mask):
        return jnp.full_like(x, self.value)


class _ConstantMlp(eqx.Module):
    value: float = eqx.field(static=True)

    def __call__(self, x):
        return jnp.full_like(x, self.value), {"router_z_loss": jnp.array(0.0)}


def test_depth_mup_residual_scale_uses_inverse_sqrt_layer_count():
    cfg = moe_model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=4,
        num_experts=4,
        num_experts_per_token=2,
        depth_mup_residual_scaling=True,
    )

    assert cfg.residual_update_scale == 0.5


def test_block_depth_mup_scales_attention_and_mlp_residual_updates():
    block = moe_model.Block(
        rms_attn=_Identity(),
        attn_gated_norm=_Identity(),
        attn=_ConstantAttention(2.0),
        rms_mlp=_Identity(),
        mlp_gated_norm=_Identity(),
        mlp=_ConstantMlp(3.0),
        shared=None,
        residual_update_scale=0.5,
    )
    x = jnp.full((1, 2, 3), 10.0, dtype=jnp.float32)

    out, _ = block(x, mask=None)

    np.testing.assert_allclose(out, jnp.full_like(x, 12.5), rtol=1e-6)


def test_depth_mup_lr_sweep_config_enables_scaling_and_scales_optimizer_lrs():
    from experiments.grug.moe import depth_mup_lr_sweep

    scale = depth_mup_lr_sweep.DEPTH_MUP_SWEEP_SCALES[0]
    lr_multiplier = 2.0
    config = depth_mup_lr_sweep.build_depth_mup_lr_sweep_config(
        scale,
        lr_multiplier,
        output_path="/tmp/moe-depth-mup-test",
    )
    _baseline_model, baseline_optimizer, baseline_batch, baseline_steps = build_from_heuristic(
        budget=scale.budget,
        hidden_dim=scale.hidden_dim,
        target_steps=depth_mup_lr_sweep.DEPTH_MUP_TARGET_STEPS,
    )

    assert config.model.depth_mup_residual_scaling is True
    np.testing.assert_allclose(config.model.residual_update_scale, 1 / math.sqrt(config.model.num_layers), rtol=1e-12)
    assert config.batch_size == baseline_batch
    assert config.steps == baseline_steps
    np.testing.assert_allclose(config.optimizer.learning_rate, baseline_optimizer.learning_rate * lr_multiplier)
    np.testing.assert_allclose(config.optimizer.adam_lr, baseline_optimizer.adam_lr * lr_multiplier)
    assert config.run_id == f"moe-depth-mup-lr-{scale.label}-lr2x"
