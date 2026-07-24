# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Anti-drift parity: Snowball (Levanter) vs the production Grug MoE Transformer.

Snowball is a pinned snapshot of ``experiments/grug/moe/model.py``. This scaffold seeds the
experiment model, exports its HF canonical weights, loads them into Snowball via
``from_state_dict``, and asserts:

- per-layer activation parity (embedding, every block output, and the final norm) so any future
  divergence is localized to a specific sublayer -- the "capture all activations" debugging
  microscope requested for bootstrapping this work;
- final-logit parity and identical greedy token IDs.

Lives on the marin side (not ``lib/levanter/tests``) because it imports ``experiments/``; the
levanter -> experiments dependency direction forbids the reverse.
"""

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

import experiments.grug.moe.model as gm

# Batch sharding for the embedding gather (matches both models' `.at[...].get(out_sharding=...)`).
_EMBED_SPEC = P(("replica_dcn", "data", "expert"))

_COMMON = dict(
    vocab_size=48,
    hidden_dim=32,
    intermediate_dim=64,
    shared_expert_intermediate_dim=48,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=5,  # 0,1,2 short; 3 => i%4==3 long; 4 => last long
    num_heads=4,
    num_kv_heads=2,
    head_dim=12,  # non-square q_proj
    max_seq_len=32,
    sliding_window=4,
)
_QK_MULT = 1.37
_EPS = 1e-5
_STD = 0.02


def _experiment_config() -> "gm.GrugModelConfig":
    return gm.GrugModelConfig(
        **_COMMON,
        qk_mult=_QK_MULT,
        layer_norm_eps=_EPS,
        initializer_std=_STD,
        moe_implementation="ring",
        disable_pko=True,
        disable_long_rope=True,
    )


def _snowball_config() -> SnowballConfig:
    return SnowballConfig(**_COMMON, qk_mult=_QK_MULT, layer_norm_eps=_EPS, initializer_std=_STD)


def _capture_experiment(model: "gm.Transformer", tokens: jax.Array) -> list[np.ndarray]:
    """Per-block hidden states for the experiment Transformer (mirrors its __call__)."""
    cfg = model.config
    short = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=None)
    long = AttentionMask(is_causal=True, sliding_window=None, segment_ids=None)
    hidden = model.token_embed.at[tokens].get(out_sharding=_EMBED_SPEC)
    hidden = model.embed_gated_norm(model.embed_norm(hidden))
    outs = [hidden]
    n = len(model.blocks)
    for i, block in enumerate(model.blocks):
        is_long = i % 4 == 3 or i == n - 1
        layer_mask = long if is_long else short
        use_pko = is_long and not cfg.disable_pko
        disable_rope = is_long and cfg.disable_long_rope
        hidden, _ = block(hidden, layer_mask, use_pko, disable_rope)
        outs.append(hidden)
    outs.append(model.final_gated_norm(model.final_norm(hidden)))
    return outs


def _capture_snowball(model: SnowballLMHeadModel, tokens: jax.Array) -> list[np.ndarray]:
    """Per-block hidden states for the Snowball transformer (mirrors its __call__)."""
    tf = model.transformer
    cfg = tf.config
    short = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=None)
    long = AttentionMask(is_causal=True, sliding_window=None, segment_ids=None)
    hidden = tf.token_embed.at[tokens].get(out_sharding=_EMBED_SPEC)
    hidden = tf.embed_gated_norm(tf.embed_norm(hidden))
    outs = [hidden]
    n = len(tf.blocks)
    for i, block in enumerate(tf.blocks):
        is_long = i % 4 == 3 or i == n - 1
        hidden = block(hidden, short, long, is_long)
        outs.append(hidden)
    outs.append(tf.final_gated_norm(tf.final_norm(hidden)))
    return outs


def test_snowball_matches_grug_experiment_per_layer_and_logits():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        exp = gm.Transformer.init(_experiment_config(), key=jax.random.key(7))
        snow = SnowballLMHeadModel.init(Axis("vocab", _COMMON["vocab_size"]), _snowball_config(), key=jax.random.key(0))
        snow = snow.from_state_dict(exp.to_state_dict())

        tokens = (jnp.arange(10, dtype=jnp.int32).reshape(1, 10)) % _COMMON["vocab_size"]

        # Per-layer activation parity (embedding, each block, final norm).
        exp_acts = [np.asarray(o) for o in _capture_experiment(exp, tokens)]
        snow_acts = [np.asarray(o) for o in _capture_snowball(snow, tokens)]
        assert len(exp_acts) == len(snow_acts) == _COMMON["num_layers"] + 2
        labels = ["embed"] + [f"block{i}" for i in range(_COMMON["num_layers"])] + ["final_norm"]
        for label, ea, sa in zip(labels, exp_acts, snow_acts, strict=True):
            max_abs = float(np.max(np.abs(ea - sa)))
            assert max_abs < 1e-5, f"activation divergence at {label}: max|diff|={max_abs}"

        # Logit + greedy parity via the standard LmHeadModel path.
        exp_logits = np.asarray(jax.jit(lambda m, t: m.logits(t))(exp, tokens))[0]
        Pos = Axis("position", tokens.shape[1])
        ids = hax.named(tokens[0], (Pos,))
        snow_logits = np.asarray(hax.named_jit(lambda m, x: m(x))(snow, ids).array)

    assert np.allclose(exp_logits, snow_logits, atol=1e-5, rtol=1e-5)
    assert np.array_equal(np.argmax(exp_logits, axis=-1), np.argmax(snow_logits, axis=-1))


@pytest.mark.parametrize("seq_len", [1, 4, 5, 16])
def test_snowball_matches_grug_across_lengths(seq_len):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        exp = gm.Transformer.init(_experiment_config(), key=jax.random.key(11))
        snow = SnowballLMHeadModel.init(Axis("vocab", _COMMON["vocab_size"]), _snowball_config(), key=jax.random.key(1))
        snow = snow.from_state_dict(exp.to_state_dict())
        tokens = (jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len) * 7 + 3) % _COMMON["vocab_size"]
        exp_logits = np.asarray(jax.jit(lambda m, t: m.logits(t))(exp, tokens))[0]
        Pos = Axis("position", seq_len)
        ids = hax.named(tokens[0], (Pos,))
        snow_logits = np.asarray(hax.named_jit(lambda m, x: m(x))(snow, ids).array)
    assert np.allclose(exp_logits, snow_logits, atol=1e-5, rtol=1e-5)
