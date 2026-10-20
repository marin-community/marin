# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny-config self-consistency tests for grug_moe_mlx.py (Mac only).

Loads the model file the same way mlx-lm does (spec_from_file_location) and
exercises behavior on a config small enough to run in seconds. All weights are
random float32; we are testing wiring (caching, masking, routing, XSA), not
learned values.
"""

import importlib.util
from pathlib import Path

import pytest

# MLX only installs on Apple Silicon; root CI collects experiments/ on Linux.
mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_map  # noqa: E402

MODEL_FILE = Path(__file__).resolve().parent / "grug_moe_mlx.py"


def load_module():
    spec = importlib.util.spec_from_file_location("custom_model", MODEL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_module()


def tiny_args(**overrides):
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_hidden_layers=5,
        sliding_window=8,
        vocab_size=101,
        rope_theta=10000.0,
        qk_mult=1.75,
        qk_mult_long_scale=1.0,
    )
    defaults.update(overrides)
    return mod.ModelArgs(**defaults)


def tiny_model(**overrides):
    mx.random.seed(0)
    model = mod.Model(tiny_args(**overrides))

    def rand(p):
        return mx.random.normal(p.shape) * 0.1

    model.update(tree_map(rand, model.parameters()))
    return model


LAYER_TYPES_5 = ["sliding_attention"] * 3 + ["full_attention"] * 2


def test_layer_types_rule():
    args = mod.ModelArgs(**{**vars(tiny_args()), "num_hidden_layers": 26})
    full = [i for i, lt in enumerate(args.layer_types) if lt == "full_attention"]
    assert full == [3, 7, 11, 15, 19, 23, 25]
    # (i+1) % 4 == 0 or last layer
    assert tiny_args(num_hidden_layers=5).layer_types == LAYER_TYPES_5


def test_unsupported_features_rejected():
    for field, value in [
        ("disable_pko", False),
        ("sconv", True),
        ("rope_fused", True),
        ("tie_word_embeddings", True),
    ]:
        with pytest.raises(ValueError, match=field):
            tiny_args(**{field: value})
    with pytest.raises(ValueError, match="latent_dim"):
        tiny_args(latent_dim=32)
    with pytest.raises(ValueError, match="num_shared_experts"):
        tiny_args(num_shared_experts=2)
    with pytest.raises(ValueError, match="grugmoe_artifact_schema_version"):
        tiny_args(grugmoe_artifact_schema_version=2)
    with pytest.raises(ValueError, match="local_kv_heads"):
        tiny_args(local_kv_heads=1, global_kv_heads=1)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        tiny_args(num_key_value_heads=3)


def test_incremental_decode_matches_full_forward():
    model = tiny_model()
    n = 40
    ids = mx.random.randint(0, 101, (1, n))

    full = model(ids)  # [1, n, vocab]

    cache = model.make_cache()
    prefill = 10
    outs = [model(ids[:, :prefill], cache)]
    for i in range(prefill, n):
        outs.append(model(ids[:, i : i + 1], cache))
    cached = mx.concatenate(outs, axis=1)

    assert cached.shape == full.shape
    # Kernel paths differ between batched prefill and single-token decode, so
    # exact comparison is impossible in f32 (float64 wiring check lives in
    # test_attention_cache_exact_in_f64). Real wiring bugs (wrong window,
    # wrong RoPE offset) produce O(0.1+) logit diffs and flip the greedy
    # choice; observed kernel noise on these random-weight logits is <~4e-4.
    diff = mx.abs(cached - full).max().item()
    assert diff < 1e-3, f"incremental decode diverged: max |diff| = {diff}"
    assert mx.array_equal(
        mx.argmax(cached, axis=-1), mx.argmax(full, axis=-1)
    ), "greedy choice differs between cached and full forward"


def test_chunked_prefill_matches_full_forward():
    model = tiny_model()
    n = 40
    ids = mx.random.randint(0, 101, (1, n))
    full = model(ids)

    cache = model.make_cache()
    outs = []
    for start in range(0, n, 7):
        outs.append(model(ids[:, start : start + 7], cache))
    chunked = mx.concatenate(outs, axis=1)

    diff = mx.abs(chunked - full).max().item()
    assert diff < 1e-3, f"chunked prefill diverged: max |diff| = {diff}"
    assert mx.array_equal(
        mx.argmax(chunked, axis=-1), mx.argmax(full, axis=-1)
    ), "greedy choice differs between chunked and full forward"


def test_sliding_window_is_real():
    # 3 layers: 0 and 1 sliding (window 8), layer 2 full (last layer).
    model = tiny_model(num_hidden_layers=3)
    n = 31
    ids = mx.random.randint(0, 101, (1, n))
    # Rebuild with token 0 changed (MLX arrays are immutable).
    other = mx.concatenate([mx.array([[42]]), ids[:, 1:]], axis=1)
    assert not mx.array_equal(ids[0, 0], other[0, 0])

    def hidden_after_layer(idx, input_ids):
        m = model.model
        h = m.embed_gated_norm(m.embed_norm(m.embed_tokens(input_ids)))
        full_mask = mod.create_attention_mask(h, None)
        swa_mask = mod.create_attention_mask(h, None, window_size=m.window_size)
        for i, layer in enumerate(m.layers[: idx + 1]):
            mask = full_mask if m.layer_types[i] == "full_attention" else swa_mask
            h = layer(h, mask)
        return h

    h1_a = hidden_after_layer(1, ids)
    h1_b = hidden_after_layer(1, other)
    # Position 30 is outside the window of both sliding layers for token 0.
    assert mx.abs(h1_a[0, 30] - h1_b[0, 30]).max().item() < 1e-5
    # But earlier positions do see the change.
    assert mx.abs(h1_a[0, 0] - h1_b[0, 0]).max().item() > 1e-5

    # The full (last) layer propagates the change to the logits at position 30.
    logits_a = model(ids)
    logits_b = model(other)
    assert mx.abs(logits_a[0, 30] - logits_b[0, 30]).max().item() > 1e-5


def test_xsa_removes_value_projection():
    """Each head's attention output is orthogonal to the current token's value."""
    model = tiny_model(num_hidden_layers=1)
    attn = model.model.layers[0].self_attn
    B, L = 1, 12
    x = mx.random.normal((B, L, 64))
    # Identity o_proj and a zero gate logit (gate = 2 * sigmoid(0) = 1) expose the
    # per-head output of Attention.__call__.
    attn.o_proj.weight = mx.eye(attn.n_heads * attn.head_dim)
    attn.attn_gate.weight = mx.zeros_like(attn.attn_gate.weight)

    y = attn(x, "causal", None).reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, attn.n_kv, attn.head_dim).transpose(0, 2, 1, 3)
    vf = mx.repeat(v, attn.n_rep, axis=1).astype(mx.float32)
    yf = y.astype(mx.float32)

    # Without XSA the coefficient of v in y is O(1) (it is 1 at position 0, where y = v).
    coefficient = mx.sum(yf * vf, axis=-1) / mx.sum(vf * vf, axis=-1)
    assert mx.abs(coefficient).max().item() < 1e-3


def test_router_bias_selection_and_renorm():
    """The router selects on biased logits but weights experts by the unbiased ones."""
    model = tiny_model(num_hidden_layers=1)
    moe = model.model.layers[0].mlp
    k = moe.top_k
    x = mx.random.normal((1, 4, 64))

    # A huge bias forces expert 7 into every top-k even though its logit is ordinary.
    moe.router.bias = moe.router.bias.astype(mx.float32).at[7].add(1e4)
    logits = x.astype(mx.float32) @ moe.router.weight.astype(mx.float32).T
    idx = mx.argpartition(-(logits + moe.router.bias), kth=k - 1, axis=-1)[..., :k]
    assert (idx == 7).any(axis=-1).all().item()

    w = mx.sigmoid(mx.take_along_axis(logits, idx, axis=-1))
    w = w * (2.5 / (mx.sum(w, axis=-1, keepdims=True) + 1e-9))
    expected = (moe.experts(x, idx).astype(mx.float32) * w[..., None]).sum(-2)
    assert mx.abs(moe(x) - expected.astype(x.dtype)).max().item() < 1e-4


def test_rotating_cache_decode_past_window_matches_full_forward():
    """Decoding 11 tokens past an 8-token window, after the rotating cache wraps,
    matches the full forward pass."""
    model = tiny_model(num_hidden_layers=1)
    n = 20
    ids = mx.random.randint(0, 101, (1, n))

    full = model(ids)[0]  # [n, vocab]
    cache = model.make_cache()
    outs = [model(ids[:, :9], cache)]
    for i in range(9, n):
        outs.append(model(ids[:, i : i + 1], cache))
    cached = mx.concatenate(outs, axis=1)[0]

    diff = mx.abs(cached - full).max().item()
    assert diff < 1e-3, f"rotating cache divergence: {diff}"
    assert mx.array_equal(
        mx.argmax(cached, axis=-1), mx.argmax(full, axis=-1)
    ), "greedy choice differs under rotating cache"


def test_attention_cache_exact_in_f64():
    """Attention alone in float64 on CPU: cached decode must equal the full
    forward to float64 noise (~1e-7). This pins window semantics, RoPE
    offsets, and KV alignment exactly, independent of f32 kernel noise."""
    with mx.stream(mx.cpu):
        args = tiny_args()
        x = mx.random.normal((1, 40, 64)).astype(mx.float64)
        for is_long in (False, True):
            mx.random.seed(2)
            attn = mod.Attention(args, is_long=is_long)
            attn.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.1, attn.parameters()))
            attn.update(tree_map(lambda p: p.astype(mx.float64), attn.parameters()))

            if is_long:
                y_full = attn(x, "causal")
                cache = mod.KVCache()
            else:
                y_full = attn(x, mod.create_attention_mask(x, None, window_size=8))
                cache = mod.RotatingKVCache(max_size=8)
            outs = []
            for s in range(0, 40, 6):
                xc = x[:, s : s + 6]
                if is_long:
                    mask = "causal" if xc.shape[1] > 1 else None
                else:
                    mask = mod.create_attention_mask(xc, cache, window_size=8)
                outs.append(attn(xc, mask, cache))
            y_cached = mx.concatenate(outs, axis=1)
            diff = mx.abs(y_cached - y_full).max().item()
            assert diff < 1e-6, f"f64 attention ({'full' if is_long else 'sliding'}): {diff}"


def test_think_markers_inferred_from_vocab():
    class Tokenizer:
        def get_vocab(self):
            return {"<|start_think|>": 11, "<|end_think|>": 12, "hello": 3}

    assert mod.tokenizer_utils._infer_thinking(Tokenizer()) == ("<|start_think|>", "<|end_think|>", (11,), (12,))
