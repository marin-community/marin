# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny GrugMoE parity harness for native tpu-inference JAX support.

Run from the Marin worktree:

    JAX_PLATFORMS=cpu \
    PYTHONPATH=../grugmoe-vllm-tpu-inference:../grugmoe-vllm-tpu-vllm \
    uv run --with-requirements ../grugmoe-vllm-tpu-inference/requirements.txt \
      --with-requirements ../grugmoe-vllm-tpu-vllm/requirements/common.txt \
      --with 'torch==2.10.0+cpu' \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      python -m experiments.grug.moe.vllm_tpu_parity \
      --tpu-inference-root ../grugmoe-vllm-tpu-inference

The component check compares tpu-inference's native JAX GrugMoE MLP against
Levanter's `moe_mlp` with the same selected experts and sigmoid combine
weights. The full check copies a tiny seeded Levanter `Transformer` into the
tpu-inference JAX model and compares final hidden states, logits from
`compute_logits`, and routed expert IDs against a dense reference using the
same Levanter parameters and equations.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import levanter.grug.grug_moe as levanter_grug_moe
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh
from levanter.grug.grug_moe import moe_mlp

import experiments.grug.moe.model as grug_moe_model
from experiments.grug.moe.model import GrugModelConfig, Transformer


class _EmptyMesh:
    empty = True
    shape: ClassVar[dict[str, int]] = {}


class _PPGroup:
    is_first_rank = True
    is_last_rank = True
    rank_in_group = 0
    world_size = 1


def _direct_moe_mlp(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate_up: jax.Array,
    w_down: jax.Array,
    *,
    activation: Any = jax.nn.silu,
    **kwargs: Any,
) -> jax.Array:
    del kwargs
    activation_fn = activation.to_jax_fn() if hasattr(activation, "to_jax_fn") else activation
    expert_w13 = w_gate_up[selected_experts]
    w13_out = jnp.einsum("td,tkdi->tki", x, expert_w13)
    intermediate_dim = w_down.shape[1]
    gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
    expert_w2 = w_down[selected_experts]
    expert_out = jnp.einsum("tki,tkid->tkd", activation_fn(gate) * up, expert_w2)
    return jnp.sum(expert_out * combine_weights[..., None].astype(expert_out.dtype), axis=1)


@contextlib.contextmanager
def _use_runtime_grug_mesh(*, for_init: bool):
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1)
    runtime_mesh = Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    grug_mesh = runtime_mesh if for_init else _EmptyMesh()
    old_get_abstract_mesh = grug_moe_model.get_abstract_mesh
    old_reshard = grug_moe_model.reshard
    old_shard_map = grug_moe_model.shard_map
    old_moe_reshard_for_init = levanter_grug_moe._reshard_for_init
    old_moe_reshard_for_shard_map = levanter_grug_moe._reshard_for_shard_map
    old_moe_current_mesh = levanter_grug_moe._current_mesh
    old_moe_mlp = levanter_grug_moe.moe_mlp
    grug_moe_model.get_abstract_mesh = lambda: grug_mesh
    grug_moe_model.reshard = lambda x, *args, **kwargs: x
    grug_moe_model.shard_map = lambda fn, **kwargs: (lambda s_ma: jnp.zeros((s_ma.shape[1],), dtype=s_ma.dtype))
    levanter_grug_moe._reshard_for_init = lambda x, spec: x
    levanter_grug_moe._reshard_for_shard_map = lambda x, mesh, spec: x
    levanter_grug_moe._current_mesh = lambda: _EmptyMesh()
    if not for_init:
        levanter_grug_moe.moe_mlp = _direct_moe_mlp
    try:
        with jax.set_mesh(runtime_mesh):
            yield grug_mesh
    finally:
        grug_moe_model.get_abstract_mesh = old_get_abstract_mesh
        grug_moe_model.reshard = old_reshard
        grug_moe_model.shard_map = old_shard_map
        levanter_grug_moe._reshard_for_init = old_moe_reshard_for_init
        levanter_grug_moe._reshard_for_shard_map = old_moe_reshard_for_shard_map
        levanter_grug_moe._current_mesh = old_moe_current_mesh
        levanter_grug_moe.moe_mlp = old_moe_mlp


@contextlib.contextmanager
def _patch_tpu_single_rank(tpu_grugmoe):
    pp_utils = importlib.import_module("tpu_inference.layers.jax.pp_utils")
    old_model_get_pp_group = tpu_grugmoe.get_pp_group
    old_pp_utils_get_pp_group = pp_utils.get_pp_group
    tpu_grugmoe.get_pp_group = lambda: _PPGroup()
    pp_utils.get_pp_group = lambda: _PPGroup()
    try:
        yield
    finally:
        tpu_grugmoe.get_pp_group = old_model_get_pp_group
        pp_utils.get_pp_group = old_pp_utils_get_pp_group


def _load_tpu_grugmoe(tpu_inference_root: Path):
    root = str(tpu_inference_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("tpu_inference.models.jax.grugmoe")


def _np(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True)


def _copy_param(param: nnx.Param, value: Any) -> None:
    param.value = jnp.asarray(value, dtype=param.value.dtype)


def _tiny_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=19,
        hidden_dim=8,
        intermediate_dim=12,
        shared_expert_intermediate_dim=10,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.02,
        moe_implementation="scatter",
    )


def _layer_sliding_window(cfg: GrugModelConfig, layer_index: int) -> int:
    short_window = cfg.sliding_window // 2
    return cfg.sliding_window if layer_index % 4 == 3 else short_window


def _hf_config(cfg: GrugModelConfig) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["GrugMoeForCausalLM"],
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        shared_expert_intermediate_dim=cfg.shared_expert_intermediate_dim,
        num_experts=cfg.num_experts,
        num_experts_per_token=cfg.num_experts_per_token,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.inferred_head_dim,
        max_seq_len=cfg.max_seq_len,
        sliding_window=cfg.sliding_window,
        layer_norm_eps=cfg.layer_norm_eps,
        initializer_std=cfg.initializer_std,
        qk_mult=cfg.qk_mult,
        rope_theta=cfg.rope.theta,
        tie_word_embeddings=False,
    )


def _vllm_config(cfg: GrugModelConfig) -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_config=_hf_config(cfg),
        dtype=jnp.float32,
    )
    return SimpleNamespace(model_config=model_config)


def _tpu_cfg(tpu_grugmoe, cfg: GrugModelConfig):
    return tpu_grugmoe.GrugMoeConfig.from_hf_config(_hf_config(cfg))


def _tpu_mesh() -> Mesh:
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    return Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))


def _route_jax(
    x: jax.Array,
    router: jax.Array,
    router_bias: jax.Array,
    *,
    num_experts_per_token: int,
) -> tuple[jax.Array, jax.Array]:
    router_logits = jnp.einsum("td,de->te", x, router).astype(jnp.float32)
    biased_logits = router_logits + router_bias.astype(jnp.float32)
    _, selected = jax.lax.top_k(biased_logits, num_experts_per_token + 1)
    selected = selected[:, :num_experts_per_token]
    unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
    combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
    return selected.astype(jnp.int32), combine_weights


def _jax_rms_norm(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    dtype = x.dtype
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    out = x_float * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight.astype(jnp.float32)
    return out.astype(dtype)


def _jax_gated_norm(norm: Any, x: jax.Array) -> jax.Array:
    gate_hidden = jnp.einsum("...d,dr->...r", x, norm.w_down)
    gate_hidden = jax.nn.silu(gate_hidden)
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, norm.w_up))
    return x * gate.astype(x.dtype)


def _jax_align_kv_heads(x: jax.Array, num_q_heads: int) -> jax.Array:
    num_kv_heads = x.shape[2]
    if num_q_heads == num_kv_heads:
        return x
    repeat = num_q_heads // num_kv_heads
    expanded = jnp.expand_dims(x, axis=3)
    expanded = jnp.broadcast_to(expanded, (*x.shape[:3], repeat, x.shape[3]))
    return expanded.reshape(*x.shape[:2], num_q_heads, x.shape[3])


def _jax_apply_rotary(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    *,
    head_dim: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    angles = positions.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    cos = jnp.cos(angles)[:, :, None, :]
    sin = jnp.sin(angles)[:, :, None, :]

    def apply(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return apply(q), apply(k)


def _jax_attention(attn: Any, x: jax.Array, positions: jax.Array, sliding_window: int) -> jax.Array:
    cfg = attn.cfg
    bsz, seq_len, _ = x.shape
    head_dim = cfg.inferred_head_dim
    q = jnp.einsum("bsh,hd->bsd", x, attn.w_q).reshape(bsz, seq_len, cfg.num_heads, head_dim)
    k = jnp.einsum("bsh,hd->bsd", x, attn.w_k).reshape(bsz, seq_len, cfg.num_kv_heads, head_dim)
    v = jnp.einsum("bsh,hd->bsd", x, attn.w_v).reshape(bsz, seq_len, cfg.num_kv_heads, head_dim)
    q = _jax_rms_norm(q)
    k = _jax_rms_norm(k)
    q, k = _jax_apply_rotary(q, k, positions, head_dim=head_dim, theta=cfg.rope.theta)
    q = q * cfg.qk_mult
    k = _jax_align_kv_heads(k, cfg.num_heads)
    v = _jax_align_kv_heads(v, cfg.num_heads)

    scores = jnp.einsum("bqhd,bkhd->bhqk", q * (head_dim**-0.5), k)
    q_pos = positions[:, :, None]
    k_pos = positions[:, None, :]
    mask = jnp.logical_and(k_pos <= q_pos, k_pos >= q_pos - (sliding_window - 1))
    scores = jnp.where(mask[:, None, :, :], scores, jnp.array(-1e9, dtype=scores.dtype))
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
    attn_out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)

    dot = jnp.sum(attn_out * v, axis=-1, keepdims=True)
    v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
    attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * v
    gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, attn.attn_gate))[..., None]
    attn_out = gate.astype(attn_out.dtype) * attn_out
    attn_out = attn_out.reshape(bsz, seq_len, cfg.num_heads * head_dim)
    return jnp.einsum("bsh,hd->bsd", attn_out, attn.w_o)


def _jax_dense_mlp(mlp: Any, x: jax.Array) -> jax.Array:
    bsz, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(bsz * seq_len, hidden_dim)
    gate = jnp.einsum("td,dm->tm", x_flat, mlp.w_gate)
    up = jnp.einsum("td,dm->tm", x_flat, mlp.w_up)
    out = jnp.einsum("tm,md->td", jax.nn.silu(gate) * up, mlp.w_down)
    return out.reshape(bsz, seq_len, hidden_dim)


def _jax_moe_mlp(mlp: Any, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    cfg = mlp.cfg
    bsz, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(bsz * seq_len, hidden_dim)
    selected, combine_weights = _route_jax(
        x_flat,
        mlp.router,
        mlp.router_bias,
        num_experts_per_token=cfg.num_experts_per_token,
    )
    out = _direct_moe_mlp(
        x_flat,
        selected,
        combine_weights,
        mlp.expert_mlp.w_gate_up,
        mlp.expert_mlp.w_down,
        activation=jax.nn.silu,
    )
    return out.reshape(bsz, seq_len, hidden_dim), selected


def _jax_full_forward(model: Transformer, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    cfg = model.config
    _bsz, seq_len = token_ids.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], token_ids.shape)
    hidden = model.token_embed[token_ids]
    hidden = _jax_gated_norm(model.embed_gated_norm, _jax_rms_norm(hidden, model.embed_norm.weight, cfg.layer_norm_eps))

    expert_ids_by_layer = []
    for i, block in enumerate(model.blocks):
        layer_window = _layer_sliding_window(cfg, i)
        attn_in = _jax_gated_norm(
            block.attn_gated_norm,
            _jax_rms_norm(hidden, block.rms_attn.weight, block.rms_attn.eps),
        )
        hidden = hidden + _jax_attention(block.attn, attn_in, positions, layer_window)
        mlp_in = _jax_gated_norm(block.mlp_gated_norm, _jax_rms_norm(hidden, block.rms_mlp.weight, block.rms_mlp.eps))
        mlp_out, expert_ids = _jax_moe_mlp(block.mlp, mlp_in)
        expert_ids_by_layer.append(expert_ids)
        if block.shared is not None:
            mlp_out = mlp_out + _jax_dense_mlp(block.shared, mlp_in)
        hidden = hidden + mlp_out

    hidden = _jax_gated_norm(
        model.final_gated_norm,
        _jax_rms_norm(hidden, model.final_norm.weight, model.final_norm.eps),
    )
    return hidden, jnp.stack(expert_ids_by_layer, axis=0)


def _jax_logits(model: Transformer, hidden: jax.Array) -> jax.Array:
    return jnp.einsum("bsh,hv->bsv", hidden, model.output_proj)


def check_moe_component(tpu_grugmoe) -> None:
    cfg = _tiny_cfg()
    native_mlp = tpu_grugmoe.GrugMoeMLP(_tpu_cfg(tpu_grugmoe, cfg), jnp.float32, nnx.Rngs(jax.random.PRNGKey(0)))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, cfg.hidden_dim)).astype(np.float32)
    router = rng.normal(size=(cfg.hidden_dim, cfg.num_experts)).astype(np.float32)
    router_bias = np.array([0.0, 0.0, 3.0, -2.0], dtype=np.float32)
    w_gate_up = rng.normal(size=(cfg.num_experts, cfg.hidden_dim, 2 * cfg.intermediate_dim)).astype(np.float32)
    w_down = rng.normal(size=(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim)).astype(np.float32)

    native_mlp.router.value = jnp.asarray(router)
    native_mlp.router_bias.value = jnp.asarray(router_bias)
    native_mlp.w_gate_up.value = jnp.asarray(w_gate_up)
    native_mlp.w_down.value = jnp.asarray(w_down)

    selected, combine_weights = _route_jax(
        jnp.asarray(x),
        jnp.asarray(router),
        jnp.asarray(router_bias),
        num_experts_per_token=cfg.num_experts_per_token,
    )
    expected = moe_mlp(
        jnp.asarray(x),
        selected,
        combine_weights,
        jnp.asarray(w_gate_up),
        jnp.asarray(w_down),
        activation=jax.nn.silu,
        implementation="scatter",
        mesh=None,
    )
    actual, actual_selected = native_mlp(jnp.asarray(x))
    np.testing.assert_array_equal(_np(actual_selected), _np(selected))
    np.testing.assert_allclose(_np(actual), _np(expected), rtol=1e-5, atol=1e-5)
    print("component: native GrugMoeMLP matches Levanter moe_mlp")


def _copy_transformer(tpu_model: Any, lev_model: Transformer) -> None:
    model = tpu_model.model
    _copy_param(model.token_embed, lev_model.token_embed)
    _copy_param(model.embed_norm.weight, lev_model.embed_norm.weight)
    _copy_param(model.embed_gated_norm.w_down, lev_model.embed_gated_norm.w_down)
    _copy_param(model.embed_gated_norm.w_up, lev_model.embed_gated_norm.w_up)
    _copy_param(model.final_norm.weight, lev_model.final_norm.weight)
    _copy_param(model.final_gated_norm.w_down, lev_model.final_gated_norm.w_down)
    _copy_param(model.final_gated_norm.w_up, lev_model.final_gated_norm.w_up)
    _copy_param(tpu_model.lm_head.weight, lev_model.output_proj)

    for tpu_block, lev_block in zip(model.layers, lev_model.blocks, strict=True):
        _copy_param(tpu_block.rms_attn.weight, lev_block.rms_attn.weight)
        _copy_param(tpu_block.attn_gated_norm.w_down, lev_block.attn_gated_norm.w_down)
        _copy_param(tpu_block.attn_gated_norm.w_up, lev_block.attn_gated_norm.w_up)
        _copy_param(tpu_block.attn.w_q, lev_block.attn.w_q)
        _copy_param(tpu_block.attn.w_k, lev_block.attn.w_k)
        _copy_param(tpu_block.attn.w_v, lev_block.attn.w_v)
        _copy_param(tpu_block.attn.w_o, lev_block.attn.w_o)
        _copy_param(tpu_block.attn.attn_gate, lev_block.attn.attn_gate)
        _copy_param(tpu_block.rms_mlp.weight, lev_block.rms_mlp.weight)
        _copy_param(tpu_block.mlp_gated_norm.w_down, lev_block.mlp_gated_norm.w_down)
        _copy_param(tpu_block.mlp_gated_norm.w_up, lev_block.mlp_gated_norm.w_up)
        _copy_param(tpu_block.mlp.router, lev_block.mlp.router)
        _copy_param(tpu_block.mlp.router_bias, lev_block.mlp.router_bias)
        _copy_param(tpu_block.mlp.w_gate_up, lev_block.mlp.expert_mlp.w_gate_up)
        _copy_param(tpu_block.mlp.w_down, lev_block.mlp.expert_mlp.w_down)
        if tpu_block.shared is not None and lev_block.shared is not None:
            _copy_param(tpu_block.shared.w_gate, lev_block.shared.w_gate)
            _copy_param(tpu_block.shared.w_up, lev_block.shared.w_up)
            _copy_param(tpu_block.shared.w_down, lev_block.shared.w_down)


def check_full_forward(tpu_grugmoe) -> None:
    attention_metadata_mod = importlib.import_module("tpu_inference.layers.common.attention_metadata")
    cfg = _tiny_cfg()
    assert [_layer_sliding_window(cfg, i) for i in range(cfg.num_layers)] == [2, 2, 2, 4]
    key = jax.random.PRNGKey(7)
    with _use_runtime_grug_mesh(for_init=True):
        lev_model = Transformer.init(cfg, key=key)

    with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
        tpu_model = tpu_grugmoe.GrugMoeForCausalLM(_vllm_config(cfg), jax.random.PRNGKey(0), _tpu_mesh())
    _copy_transformer(tpu_model, lev_model)

    token_ids = jnp.array([[1, 5, 3, 7, 2, 9]], dtype=jnp.int32)
    positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)
    metadata = attention_metadata_mod.AttentionMetadata(input_positions=positions)
    _, actual_hidden, actual_expert_ids = tpu_model.model([None] * cfg.num_layers, token_ids[0], metadata)
    expected_hidden, expected_expert_ids = jax.jit(_jax_full_forward)(lev_model, token_ids)
    actual_logits = tpu_model.compute_logits(actual_hidden)
    expected_logits = _jax_logits(lev_model, expected_hidden)[0]

    if actual_expert_ids is None:
        raise AssertionError("native GrugMoE model did not return routed expert IDs")
    np.testing.assert_allclose(_np(actual_hidden), _np(expected_hidden[0]), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(_np(actual_logits), _np(expected_logits), rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(actual_expert_ids), _np(expected_expert_ids))
    print("full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tpu-inference-root",
        type=Path,
        default=Path("../grugmoe-vllm-tpu-inference"),
        help="Path to the tpu-inference checkout containing native GrugMoE.",
    )
    parser.add_argument(
        "--component-only",
        action="store_true",
        help="Run only the MoE component parity check.",
    )
    args = parser.parse_args()

    tpu_grugmoe = _load_tpu_grugmoe(args.tpu_inference_root.resolve())
    check_moe_component(tpu_grugmoe)
    if not args.component_only:
        check_full_forward(tpu_grugmoe)


if __name__ == "__main__":
    main()
