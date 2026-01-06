# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.grug.attention import AttentionMask, attention, reference_attention
from levanter.grug.config import GrugModelConfig
from levanter.grug.model import forward, init_parameters


def _make_grug_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(1, 1, len(devices))
    return Mesh(
        mesh_devices,
        axis_names=("replica", "data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def test_forward_shapes_and_jit_compile():
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
        tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)

        logits = forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits.shape == (2, 8, cfg.vocab_size)

        jit_forward = jax.jit(forward, static_argnames=("cfg",))
        logits_jit = jit_forward(params, tokens, cfg, mask=AttentionMask.causal())
        assert logits_jit.shape == (2, 8, cfg.vocab_size)


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

        expected_vocab = P(None, None) if jax.default_backend() == "tpu" else P("data", None)
        assert getattr(params.token_embed.sharding, "spec", None) == expected_vocab
        assert getattr(params.output_proj.sharding, "spec", None) == expected_vocab
        assert getattr(params.blocks[0].attn.w_q.sharding, "spec", None) == P("data", "model")


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
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
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
    if mode == "causal":
        mask = AttentionMask.causal()
    elif mode == "causal_window":
        mask = AttentionMask.causal(sliding_window=3)
    elif mode == "causal_window_segments":
        segment_ids = jnp.array([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=jnp.int32)
        mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids, segment_ids)
    else:
        raise AssertionError(f"unknown mode: {mode}")

    batch, seq, heads, head_dim = 1, 8, 2, 4
    q = jax.random.normal(jax.random.key(0), (batch, seq, heads, head_dim), dtype=jnp.float32)
    k = jax.random.normal(jax.random.key(1), (batch, seq, heads, head_dim), dtype=jnp.float32)
    v = jax.random.normal(jax.random.key(2), (batch, seq, heads, head_dim), dtype=jnp.float32)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        out_blocksparse = attention(q, k, v, mask)
        out_ref = reference_attention(q, k, v, mask, logits_dtype=None)

    assert out_blocksparse.shape == out_ref.shape
    assert jnp.allclose(out_blocksparse, out_ref, rtol=1e-4, atol=1e-4)
