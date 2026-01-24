# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import AttentionMask, attention, reference_attention
from levanter.grug.model import GrugModelConfig
from levanter.grug.sharding import Pbatch
from levanter.grug.model import forward, init_parameters


def _make_grug_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def test_forward_shapes_and_jit_compile():
    # On TPU, Grug uses Splash attention which requires KV sequence length to be a multiple of 128.
    seq = 128 if jax.default_backend() == "tpu" else 8
    batch = len(jax.devices()) * 2
    cfg = GrugModelConfig(
        vocab_size=101,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=seq,
    )

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, seq), 0, cfg.vocab_size)

        logits = forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits.shape == (batch, seq, cfg.vocab_size)

        jit_forward = jax.jit(forward, static_argnames=("cfg",))
        logits_jit = jit_forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits_jit.shape == (2, seq, cfg.vocab_size)


def test_parameter_sharding_specs_are_named():
    cfg = GrugModelConfig(
        vocab_size=101,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
    )

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(0))

        expected_vocab = P(None, None)
        assert params.token_embed.sharding.spec == expected_vocab
        assert params.output_proj.sharding.spec == expected_vocab
        assert getattr(params.blocks[0].attn.w_q.sharding, "spec", None) == P("data", "model")


def test_full_like_preserves_sharding_under_mesh():
    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(("data",), None))
        segment_ids = jax.device_put(jnp.array([[0, 0, 1, 1], [5, 5, 5, -1]], dtype=jnp.int32), sharding)

        batch_slice = segment_ids[:, 0]
        init_last = jnp.full_like(batch_slice, jnp.int32(-2))
        init_pos = jnp.full_like(batch_slice, jnp.int32(-1))

        assert getattr(batch_slice.sharding, "spec", None) == getattr(init_last.sharding, "spec", None)
        assert getattr(batch_slice.sharding, "spec", None) == getattr(init_pos.sharding, "spec", None)


def test_attentionmask_materialize_causal():
    mask = AttentionMask.causal()
    allowed = mask.materialize_mask(4, 4)
    expected = jnp.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=bool,
    )
    assert allowed is not None
    assert allowed.shape == (4, 4)
    assert jnp.array_equal(allowed, expected)


def test_attentionmask_materialize_sliding_window_only():
    mask = AttentionMask(is_causal=False, sliding_window=1)
    allowed = mask.materialize_mask(4, 4)
    expected = jnp.array(
        [
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    assert allowed is not None
    assert allowed.shape == (4, 4)
    assert jnp.array_equal(allowed, expected)


def test_attentionmask_materialize_segment_ids_per_batch():
    q_seg = jnp.array([[0, 0, 1], [3, 3, 3]], dtype=jnp.int32)
    k_seg = jnp.array([[0, 1, 1, 1], [3, 4, 3, 4]], dtype=jnp.int32)
    mask = AttentionMask(is_causal=False, segment_ids=(q_seg, k_seg))
    allowed = mask.materialize_mask(3, 4)
    expected = jnp.array(
        [
            [
                [True, False, False, False],
                [True, False, False, False],
                [False, True, True, True],
            ],
            [
                [True, False, True, False],
                [True, False, True, False],
                [True, False, True, False],
            ],
        ],
        dtype=bool,
    )
    assert allowed is not None
    assert allowed.shape == (2, 3, 4)
    assert jnp.array_equal(allowed, expected)


@pytest.mark.parametrize("mode", ["causal", "causal_window", "causal_window_segments"])
def test_blocksparse_attention_matches_reference_on_tiny_shapes(mode: str):
    bs = len(jax.devices())
    seq = 128 if jax.default_backend() == "tpu" else 8

    batch, heads, head_dim = bs, 2, 4
    # Keep logits in a reasonable range so this test checks semantics rather than softmax saturation
    # differences between Splash and the reference path.
    scale = 0.02
    q = jax.random.normal(jax.random.key(0), (batch, seq, heads, head_dim), dtype=jnp.float32) * scale
    k = jax.random.normal(jax.random.key(1), (batch, seq, heads, head_dim), dtype=jnp.float32) * scale
    v = jax.random.normal(jax.random.key(2), (batch, seq, heads, head_dim), dtype=jnp.float32) * scale

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        if mode == "causal":
            mask = AttentionMask.causal()
        elif mode == "causal_window":
            mask = AttentionMask.causal(sliding_window=3)
        elif mode == "causal_window_segments":
            # Segment every 16 tokens to test reset behavior while keeping Splash-compatible shapes on TPU.
            segment_ids = (jnp.arange(seq, dtype=jnp.int32) // 16)[None, :]
            segment_ids = jnp.repeat(segment_ids, repeats=bs, axis=0)
            segment_ids = jax.sharding.reshard(segment_ids, Pbatch)
            mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids, segment_ids)
        else:
            raise AssertionError(f"unknown mode: {mode}")

        q, k, v = jax.sharding.reshard((q, k, v), Pbatch)
        out_blocksparse = attention(q, k, v, mask)
        out_ref = reference_attention(q, k, v, mask, logits_dtype=None)

    assert out_blocksparse.shape == out_ref.shape
    if jax.default_backend() == "tpu":
        # Splash attention has small numeric differences vs the reference path on TPU.
        assert jnp.allclose(out_blocksparse, out_ref, rtol=1e-3, atol=1e-3)
    else:
        assert jnp.allclose(out_blocksparse, out_ref, rtol=1e-4, atol=1e-4)


def test_tpu_splash_attention_respects_causal_mask():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only (Splash attention)")

    bs = len(jax.devices())
    seq = 128
    heads, head_dim = 2, 4

    # Construct an adversarial setup:
    # - make the last key have overwhelmingly high similarity to every query
    # - set v_last to 1s, and all other v to 0s
    # Then:
    # - for q positions < last, causal mask forbids attending to the last key => output ~0
    # - for q position == last, attending to last is allowed => output ~1
    u = jnp.ones((head_dim,), dtype=jnp.float32)
    q = jnp.broadcast_to(u, (bs, seq, heads, head_dim))
    k = jnp.zeros((bs, seq, heads, head_dim), dtype=jnp.float32).at[:, -1, :, :].set(u * 50.0)
    v = jnp.zeros((bs, seq, heads, head_dim), dtype=jnp.float32).at[:, -1, :, :].set(1.0)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        q, k, v = jax.sharding.reshard((q, k, v), Pbatch)
        out = attention(q, k, v, AttentionMask.causal())

    assert out.shape == (bs, seq, heads, head_dim)
    # Early positions can't see the last key, so output should be ~0.
    assert jnp.max(jnp.abs(out[:, :-1, :, :])) < 1e-3
    # Last position can see the last key; output should be very close to 1.
    assert jnp.min(out[:, -1, :, :]) > 0.999


def test_tpu_splash_attention_respects_sliding_window():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only (Splash attention)")

    bs = len(jax.devices())
    seq = 128
    heads, head_dim = 2, 4

    # Standard sliding window semantics: W tokens including self.
    # For q position W, keys < 1 are outside the window and must be masked.
    W = 3
    u = jnp.ones((head_dim,), dtype=jnp.float32)
    q = jnp.broadcast_to(u, (bs, seq, heads, head_dim))
    k = jnp.zeros((bs, seq, heads, head_dim), dtype=jnp.float32).at[:, 0, :, :].set(u * 50.0)
    v = jnp.zeros((bs, seq, heads, head_dim), dtype=jnp.float32).at[:, 0, :, :].set(1.0)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        q, k, v = jax.sharding.reshard((q, k, v), Pbatch)
        out = attention(q, k, v, AttentionMask.causal(sliding_window=W))

    # q at positions < W still include k=0 in their window, so output ~1 (and causal allows it).
    assert jnp.min(out[:, :W, :, :]) > 0.999
    # q at position == W cannot see k=0 (outside window), so output should drop to ~0.
    assert jnp.max(jnp.abs(out[:, W:, :, :])) < 1e-3
