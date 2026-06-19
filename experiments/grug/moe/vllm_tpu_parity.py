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
same Levanter parameters and equations. The artifact check saves a tiny
Levanter checkpoint, exports it through `export_lm_to_hf`, loads the resulting
canonical safetensors artifact into tpu-inference, and compares against the
manual-copy native model. It runs both the default single-file artifact and a
deliberately forced HuggingFace sharded artifact with
`model.safetensors.index.json`. The realistic GrugTrainState path also checks
small deterministic full-forward greedy generation from the loaded sharded
artifact, without using production KV-cache decode.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import importlib
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.grug.grug_moe as levanter_grug_moe
import levanter.main.export_lm_to_hf as export_lm_to_hf
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh
from levanter.checkpoint import save_checkpoint
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.jax_utils import is_inexact_arrayish
from safetensors.numpy import load_file

import experiments.grug.moe.model as grug_moe_model
from experiments.grug.moe.heuristic import DEFAULT_TARGET_STEPS, build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig, Transformer, canonical_grugmoe_tensor_names
from experiments.grug.moe.train import initial_state

jax.config.update("jax_default_matmul_precision", "float32")

_ARTIFACT_WEIGHTS_FILE = "model.safetensors"
_ARTIFACT_WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
_FORCED_SHARD_SIZE_BYTES = 1024
_REALISTIC_MAX_SHARD_SIZE_BYTES = 256 * 1024 * 1024
_SMALL_DIAGNOSTIC_MAX_SHARD_SIZE_BYTES = 64 * 1024 * 1024
_REALISTIC_GENERATION_TOKENS = 3
_TRAINING_MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
_CANARY_BUDGET = 1e18
_CANARY_HIDDEN_DIM = 1024
_ROUTING_DEBUG_VECTOR_LIMIT = 16
_FORWARD_DEBUG_VALUE_NAMES = (
    "input_ids",
    "positions",
    "query_start_loc",
    "seq_lens",
    "embedding_output",
    "layer_attn_input",
    "layer_attn_output",
    "layer_post_attn_residual",
    "layer_mlp_input",
    "layer_output",
    "final_hidden",
)
_GRUGMOE_ATTENTION_MODE_KEY = "grugmoe_attention_mode"
_DENSE_ATTENTION_MODE = "dense"
_PRODUCTION_ATTENTION_MODE = "production"
_PRODUCTION_ATTENTION_LOGITS_TOLERANCE = 5e-2
_PRODUCTION_ATTENTION_LOGPROB_TOLERANCE = 5e-2


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
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    runtime_mesh = Mesh(
        devices,
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
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


def _array_debug_summary(value: Any, *, vector_limit: int) -> dict[str, Any]:
    arr = np.asarray(value)
    flat = arr.reshape(-1)
    flat64 = flat.astype(np.float64, copy=False)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(flat.size),
    }
    if flat.size:
        summary.update(
            {
                "first_values": flat64[:vector_limit].tolist(),
                "l2": float(np.linalg.norm(flat64)),
                "max": float(np.max(flat64)),
                "mean": float(np.mean(flat64)),
                "min": float(np.min(flat64)),
            }
        )
        if flat.size <= vector_limit:
            summary["values"] = flat64.tolist()
    else:
        summary["first_values"] = []
    return summary


def _routing_debug_payload(
    *,
    source: str,
    layer: int,
    token_position: int,
    vector_limit: int,
    router_input_hidden_state: Any,
    router_logits: Any,
    router_bias: Any,
    biased_router_logits: Any,
    topk_with_boundary_logits: Any,
    topk_with_boundary_expert_ids: Any,
    topk_expert_ids: Any,
    combine_weights: Any,
) -> dict[str, Any]:
    topk_with_boundary_logits_np = np.asarray(topk_with_boundary_logits, dtype=np.float64)
    topk_with_boundary_ids_np = np.asarray(topk_with_boundary_expert_ids, dtype=np.int64)
    topk_expert_ids_np = np.asarray(topk_expert_ids, dtype=np.int64)
    top_k = int(topk_expert_ids_np.shape[0])
    payload = {
        "source": source,
        "layer": int(layer),
        "token_position": int(token_position),
        "router_input_hidden_state": _array_debug_summary(router_input_hidden_state, vector_limit=vector_limit),
        "raw_router_logits": _array_debug_summary(router_logits, vector_limit=vector_limit),
        "router_bias": _array_debug_summary(router_bias, vector_limit=vector_limit),
        "biased_router_logits": _array_debug_summary(biased_router_logits, vector_limit=vector_limit),
        "topk_expert_ids": topk_expert_ids_np.tolist(),
        "topk_with_boundary_expert_ids": topk_with_boundary_ids_np.tolist(),
        "topk_with_boundary_logits": topk_with_boundary_logits_np.tolist(),
        "combine_weights": _array_debug_summary(combine_weights, vector_limit=vector_limit),
    }
    if topk_with_boundary_logits_np.shape[0] > top_k:
        payload["boundary_expert_id"] = int(topk_with_boundary_ids_np[top_k])
        payload["boundary_logit"] = float(topk_with_boundary_logits_np[top_k])
        payload["router_margin"] = float(topk_with_boundary_logits_np[top_k - 1] - topk_with_boundary_logits_np[top_k])
    return payload


def _forward_debug_payload(
    *,
    source: str,
    layer: int,
    token_position: int,
    vector_limit: int,
    values: Sequence[Any],
) -> dict[str, Any]:
    if len(values) != len(_FORWARD_DEBUG_VALUE_NAMES):
        raise ValueError(f"expected {len(_FORWARD_DEBUG_VALUE_NAMES)} forward-debug values, got {len(values)}")
    return {
        "source": source,
        "layer": int(layer),
        "token_position": int(token_position),
        "values": {
            name: _array_debug_summary(value, vector_limit=vector_limit)
            for name, value in zip(_FORWARD_DEBUG_VALUE_NAMES, values, strict=True)
        },
    }


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


def _large_smoke_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32768,
        hidden_dim=640,
        intermediate_dim=2048,
        shared_expert_intermediate_dim=512,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        head_dim=80,
        max_seq_len=16,
        sliding_window=8,
        initializer_std=0.0,
        moe_implementation="scatter",
    )


def _scaled_realistic_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=128,
        shared_expert_intermediate_dim=256,
        num_experts=64,
        num_experts_per_token=4,
        num_layers=4,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        max_seq_len=256,
        sliding_window=128,
        initializer_std=0.5 / 16,
        qk_mult=1.3,
        moe_implementation="scatter",
    )


def _small_diagnostic_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=4096,
        hidden_dim=512,
        intermediate_dim=1024,
        shared_expert_intermediate_dim=512,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=16,
        sliding_window=8,
        initializer_std=0.5 / 16,
        qk_mult=1.3,
        moe_implementation="scatter",
    )


def _canary_cfg_optimizer_steps() -> tuple[GrugModelConfig, Any, int]:
    # Import launch lazily so simple module import/py_compile does not construct
    # the full executor step and data mixture unless the canary validation runs.
    from experiments.grug.moe.launch import GRUG_MOE_TRIAL_MODEL  # noqa: PLC0415

    cfg, optimizer_config, _batch_size, steps = build_from_heuristic(
        budget=_CANARY_BUDGET,
        hidden_dim=_CANARY_HIDDEN_DIM,
        target_steps=DEFAULT_TARGET_STEPS,
    )
    if cfg != GRUG_MOE_TRIAL_MODEL:
        raise AssertionError("heuristic-derived canary config does not match GRUG_MOE_TRIAL_MODEL")
    return cfg, optimizer_config, steps


def _assert_production_relevant_structure(cfg: GrugModelConfig) -> None:
    if cfg.num_experts != 64:
        raise AssertionError(f"realistic config must keep 64 experts, got {cfg.num_experts}")
    if cfg.num_experts_per_token != 4:
        raise AssertionError(f"realistic config must keep top-4 routing, got {cfg.num_experts_per_token}")
    if cfg.shared_expert_intermediate_dim <= 0:
        raise AssertionError("realistic config must keep the shared expert")
    if cfg.num_heads <= cfg.num_kv_heads or cfg.num_heads % cfg.num_kv_heads != 0:
        raise AssertionError(
            f"realistic config must keep GQA, got num_heads={cfg.num_heads} num_kv_heads={cfg.num_kv_heads}"
        )
    windows = {_layer_sliding_window(cfg, i) for i in range(cfg.num_layers)}
    if windows != {cfg.sliding_window // 2, cfg.sliding_window}:
        raise AssertionError(f"realistic config must exercise short and long windows, got {sorted(windows)}")


def _layer_sliding_window(cfg: GrugModelConfig, layer_index: int) -> int:
    short_window = cfg.sliding_window // 2
    return cfg.sliding_window if layer_index % 4 == 3 else short_window


def _hf_config(
    cfg: GrugModelConfig,
    *,
    attention_mode: str | None = None,
) -> SimpleNamespace:
    config_overrides = None if attention_mode is None else {_GRUGMOE_ATTENTION_MODE_KEY: attention_mode}
    return SimpleNamespace(**cfg.to_hf_config(cfg.vocab_size, config_overrides=config_overrides).to_dict())


def _vllm_config(
    cfg: GrugModelConfig,
    model_path: Path | None = None,
    *,
    attention_mode: str | None = _DENSE_ATTENTION_MODE,
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_config=_hf_config(cfg, attention_mode=attention_mode),
        dtype=jnp.float32,
        model=str(model_path) if model_path is not None else "grugmoe-canonical-test",
    )
    return SimpleNamespace(
        model_config=model_config,
        load_config=SimpleNamespace(download_dir=None),
        additional_config={},
    )


def _tpu_cfg(tpu_grugmoe, cfg: GrugModelConfig):
    return tpu_grugmoe.GrugMoeConfig.from_hf_config(_hf_config(cfg, attention_mode=_DENSE_ATTENTION_MODE))


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
    selected, combine_weights, _router_margin = _route_jax_with_margin(
        x,
        router,
        router_bias,
        num_experts_per_token=num_experts_per_token,
    )
    return selected, combine_weights


def _route_jax_with_margin(
    x: jax.Array,
    router: jax.Array,
    router_bias: jax.Array,
    *,
    num_experts_per_token: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    router_logits = jnp.einsum("td,de->te", x, router).astype(jnp.float32)
    biased_logits = router_logits + router_bias.astype(jnp.float32)
    topk_logits, selected = jax.lax.top_k(biased_logits, num_experts_per_token + 1)
    router_margin = topk_logits[:, num_experts_per_token - 1] - topk_logits[:, num_experts_per_token]
    selected = selected[:, :num_experts_per_token]
    unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
    combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
    return selected.astype(jnp.int32), combine_weights, router_margin


def _route_jax_with_margin_and_debug(
    x: jax.Array,
    router: jax.Array,
    router_bias: jax.Array,
    *,
    num_experts_per_token: int,
    token_position: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
    router_logits = jnp.einsum("td,de->te", x, router).astype(jnp.float32)
    biased_logits = router_logits + router_bias.astype(jnp.float32)
    topk_logits, selected_with_boundary = jax.lax.top_k(biased_logits, num_experts_per_token + 1)
    router_margin = topk_logits[:, num_experts_per_token - 1] - topk_logits[:, num_experts_per_token]
    selected = selected_with_boundary[:, :num_experts_per_token]
    unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
    combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
    routing_debug = (
        x[token_position],
        router_logits[token_position],
        router_bias,
        biased_logits[token_position],
        topk_logits[token_position],
        selected_with_boundary[token_position],
        selected[token_position],
        combine_weights[token_position],
    )
    return selected.astype(jnp.int32), combine_weights, router_margin, routing_debug


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


def _jax_moe_mlp_with_margin(mlp: Any, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    cfg = mlp.cfg
    bsz, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(bsz * seq_len, hidden_dim)
    selected, combine_weights, router_margin = _route_jax_with_margin(
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
    return out.reshape(bsz, seq_len, hidden_dim), selected, router_margin


def _jax_moe_mlp_with_margin_and_debug(
    mlp: Any, x: jax.Array, *, token_position: int
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
    cfg = mlp.cfg
    bsz, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(bsz * seq_len, hidden_dim)
    selected, combine_weights, router_margin, routing_debug = _route_jax_with_margin_and_debug(
        x_flat,
        mlp.router,
        mlp.router_bias,
        num_experts_per_token=cfg.num_experts_per_token,
        token_position=token_position,
    )
    out = _direct_moe_mlp(
        x_flat,
        selected,
        combine_weights,
        mlp.expert_mlp.w_gate_up,
        mlp.expert_mlp.w_down,
        activation=jax.nn.silu,
    )
    return out.reshape(bsz, seq_len, hidden_dim), selected, router_margin, routing_debug


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


def _jax_full_forward_with_router_margins(
    model: Transformer, token_ids: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    cfg = model.config
    _bsz, seq_len = token_ids.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], token_ids.shape)
    hidden = model.token_embed[token_ids]
    hidden = _jax_gated_norm(model.embed_gated_norm, _jax_rms_norm(hidden, model.embed_norm.weight, cfg.layer_norm_eps))

    expert_ids_by_layer = []
    router_margins_by_layer = []
    for i, block in enumerate(model.blocks):
        layer_window = _layer_sliding_window(cfg, i)
        attn_in = _jax_gated_norm(
            block.attn_gated_norm,
            _jax_rms_norm(hidden, block.rms_attn.weight, block.rms_attn.eps),
        )
        hidden = hidden + _jax_attention(block.attn, attn_in, positions, layer_window)
        mlp_in = _jax_gated_norm(block.mlp_gated_norm, _jax_rms_norm(hidden, block.rms_mlp.weight, block.rms_mlp.eps))
        mlp_out, expert_ids, router_margin = _jax_moe_mlp_with_margin(block.mlp, mlp_in)
        expert_ids_by_layer.append(expert_ids)
        router_margins_by_layer.append(router_margin)
        if block.shared is not None:
            mlp_out = mlp_out + _jax_dense_mlp(block.shared, mlp_in)
        hidden = hidden + mlp_out

    hidden = _jax_gated_norm(
        model.final_gated_norm,
        _jax_rms_norm(hidden, model.final_norm.weight, model.final_norm.eps),
    )
    return hidden, jnp.stack(expert_ids_by_layer, axis=0), jnp.stack(router_margins_by_layer, axis=0)


def _jax_full_forward_with_router_debug(
    model: Transformer,
    token_ids: jax.Array,
    *,
    debug_token_position: int,
    debug_layer: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    cfg = model.config
    _bsz, seq_len = token_ids.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], token_ids.shape)
    hidden = model.token_embed[token_ids]
    hidden = _jax_gated_norm(model.embed_gated_norm, _jax_rms_norm(hidden, model.embed_norm.weight, cfg.layer_norm_eps))
    embedding_output = hidden[0, debug_token_position]

    expert_ids_by_layer = []
    router_margins_by_layer = []
    routing_debug = None
    forward_layer_debug = None
    for i, block in enumerate(model.blocks):
        layer_window = _layer_sliding_window(cfg, i)
        attn_in = _jax_gated_norm(
            block.attn_gated_norm,
            _jax_rms_norm(hidden, block.rms_attn.weight, block.rms_attn.eps),
        )
        attn_out = _jax_attention(block.attn, attn_in, positions, layer_window)
        post_attn = hidden + attn_out
        hidden = post_attn
        mlp_in = _jax_gated_norm(block.mlp_gated_norm, _jax_rms_norm(hidden, block.rms_mlp.weight, block.rms_mlp.eps))
        if i == debug_layer:
            mlp_out, expert_ids, router_margin, routing_debug = _jax_moe_mlp_with_margin_and_debug(
                block.mlp, mlp_in, token_position=debug_token_position
            )
        else:
            mlp_out, expert_ids, router_margin = _jax_moe_mlp_with_margin(block.mlp, mlp_in)
        expert_ids_by_layer.append(expert_ids)
        router_margins_by_layer.append(router_margin)
        if block.shared is not None:
            mlp_out = mlp_out + _jax_dense_mlp(block.shared, mlp_in)
        hidden = hidden + mlp_out
        if i == debug_layer:
            forward_layer_debug = (
                attn_in[0, debug_token_position],
                attn_out[0, debug_token_position],
                post_attn[0, debug_token_position],
                mlp_in[0, debug_token_position],
                hidden[0, debug_token_position],
            )

    hidden = _jax_gated_norm(
        model.final_gated_norm,
        _jax_rms_norm(hidden, model.final_norm.weight, model.final_norm.eps),
    )
    assert routing_debug is not None
    assert forward_layer_debug is not None
    query_start_loc = jnp.asarray([0, seq_len], dtype=jnp.int32)
    seq_lens = jnp.asarray([seq_len], dtype=jnp.int32)
    forward_debug = (
        token_ids[0],
        positions[0],
        query_start_loc,
        seq_lens,
        embedding_output,
        *forward_layer_debug,
        hidden[0, debug_token_position],
    )
    return (
        hidden,
        jnp.stack(expert_ids_by_layer, axis=0),
        jnp.stack(router_margins_by_layer, axis=0),
        routing_debug,
        forward_debug,
    )


def _jax_logits(model: Transformer, hidden: jax.Array) -> jax.Array:
    return jnp.einsum("bsh,hv->bsv", hidden, model.output_proj)


def _jax_last_logits(model: Transformer, token_ids: jax.Array) -> jax.Array:
    hidden, _ = _jax_full_forward(model, token_ids)
    return _jax_logits(model, hidden)[0, -1]


def _append_token(token_ids: jax.Array, token_id: int) -> jax.Array:
    next_token = jnp.asarray([[token_id]], dtype=token_ids.dtype)
    return jnp.concatenate([token_ids, next_token], axis=1)


def _levanter_greedy_generation_reference(
    lev_model: Transformer,
    prompt_ids: jax.Array,
    generation_tokens: int,
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    token_ids = prompt_ids
    generated_ids = []
    next_token_logits = []
    last_logits = jax.jit(_jax_last_logits)
    for _ in range(generation_tokens):
        step_logits = last_logits(lev_model, token_ids)
        token_id = int(np.asarray(jnp.argmax(step_logits, axis=-1)))
        generated_ids.append(token_id)
        next_token_logits.append(_np(step_logits))
        token_ids = _append_token(token_ids, token_id)

    return (
        np.asarray(generated_ids, dtype=np.int32),
        np.stack(next_token_logits, axis=0),
        token_ids,
    )


def _write_installed_vllm_reference(
    lev_model: Transformer,
    prompt_ids: jax.Array,
    continuation_ids: Sequence[int],
    output_path: Path,
    *,
    routing_debug_token_position: int | None = None,
    routing_debug_layer: int | None = None,
    routing_debug_vector_limit: int = _ROUTING_DEBUG_VECTOR_LIMIT,
    forward_debug_token_position: int | None = None,
    forward_debug_layer: int | None = None,
    forward_debug_vector_limit: int = _ROUTING_DEBUG_VECTOR_LIMIT,
) -> None:
    if prompt_ids.shape[0] != 1:
        raise ValueError(f"installed-vLLM reference expects batch size 1, got {prompt_ids.shape[0]}")
    continuation = jnp.asarray([list(continuation_ids)], dtype=prompt_ids.dtype)
    score_token_ids = jnp.concatenate([prompt_ids, continuation], axis=1)
    if score_token_ids.shape[1] <= prompt_ids.shape[1]:
        raise ValueError("installed-vLLM reference needs at least one continuation token")

    routing_debug_payload = None
    forward_debug_payload = None
    debug_token_position = None
    debug_layer = None
    if (
        routing_debug_token_position is not None
        or routing_debug_layer is not None
        or forward_debug_token_position is not None
        or forward_debug_layer is not None
    ):
        if routing_debug_token_position is not None or routing_debug_layer is not None:
            if routing_debug_token_position is None or routing_debug_layer is None:
                raise ValueError("routing debug requires both token position and layer")
            debug_token_position = routing_debug_token_position
            debug_layer = routing_debug_layer
        if forward_debug_token_position is not None or forward_debug_layer is not None:
            if forward_debug_token_position is None or forward_debug_layer is None:
                raise ValueError("forward debug requires both token position and layer")
            if debug_token_position is None:
                debug_token_position = forward_debug_token_position
                debug_layer = forward_debug_layer
            elif debug_token_position != forward_debug_token_position or debug_layer != forward_debug_layer:
                raise ValueError("routing and forward debug must target the same token/layer for one reference pass")
        assert debug_token_position is not None
        assert debug_layer is not None
        if not (0 <= debug_token_position < int(score_token_ids.shape[1])):
            raise ValueError(
                f"debug token position {debug_token_position} is outside score length {score_token_ids.shape[1]}"
            )
        if not (0 <= debug_layer < int(lev_model.config.num_layers)):
            raise ValueError(f"debug layer {debug_layer} is outside num_layers {lev_model.config.num_layers}")
        debug_forward = jax.jit(
            _jax_full_forward_with_router_debug,
            static_argnames=("debug_token_position", "debug_layer"),
        )
        hidden, expert_ids, router_margins, routing_debug, forward_debug = debug_forward(
            lev_model,
            score_token_ids,
            debug_token_position=debug_token_position,
            debug_layer=debug_layer,
        )
        if routing_debug_token_position is not None:
            (
                router_input_hidden_state,
                router_logits,
                router_bias,
                biased_router_logits,
                topk_with_boundary_logits,
                topk_with_boundary_expert_ids,
                topk_expert_ids,
                combine_weights,
            ) = (_np(value) for value in routing_debug)
            routing_debug_payload = _routing_debug_payload(
                source="Levanter manual reference",
                layer=debug_layer,
                token_position=debug_token_position,
                vector_limit=routing_debug_vector_limit,
                router_input_hidden_state=router_input_hidden_state,
                router_logits=router_logits,
                router_bias=router_bias,
                biased_router_logits=biased_router_logits,
                topk_with_boundary_logits=topk_with_boundary_logits,
                topk_with_boundary_expert_ids=topk_with_boundary_expert_ids,
                topk_expert_ids=topk_expert_ids,
                combine_weights=combine_weights,
            )
        if forward_debug_token_position is not None:
            forward_debug_payload = _forward_debug_payload(
                source="Levanter manual reference",
                layer=debug_layer,
                token_position=debug_token_position,
                vector_limit=forward_debug_vector_limit,
                values=tuple(_np(value) for value in forward_debug),
            )
    else:
        hidden, expert_ids, router_margins = jax.jit(_jax_full_forward_with_router_margins)(lev_model, score_token_ids)
    logits = _jax_logits(lev_model, hidden)[0]
    log_probs = jax.nn.log_softmax(logits[:-1].astype(jnp.float32), axis=-1)
    targets = score_token_ids[0, 1:]
    selected_logprobs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(axis=-1)

    prompt_len = int(prompt_ids.shape[1])
    score_len = int(score_token_ids.shape[1])
    continuation_positions = list(range(prompt_len, score_len))
    full_selected_logprobs_np = _np(selected_logprobs)
    continuation_logprobs_np = full_selected_logprobs_np[prompt_len - 1 :]
    expert_ids_np = _np(expert_ids)
    router_margins_np = _np(router_margins)
    if expert_ids_np.shape[1] != score_len:
        raise AssertionError(f"reference expert IDs have shape {expert_ids_np.shape}, expected sequence {score_len}")
    if router_margins_np.shape[1] != score_len:
        raise AssertionError(
            f"reference router margins have shape {router_margins_np.shape}, expected sequence {score_len}"
        )

    payload = {
        "model_config": json.loads(json.dumps(dataclasses.asdict(lev_model.config), default=str)),
        "prompt_ids": np.asarray(prompt_ids).tolist()[0],
        "continuation_ids": [int(token_id) for token_id in continuation_ids],
        "score_token_ids": np.asarray(score_token_ids).tolist()[0],
        "logprob_token_positions": continuation_positions,
        "levanter_full_selected_logprobs": full_selected_logprobs_np.astype(np.float64).tolist(),
        "levanter_continuation_logprobs": continuation_logprobs_np.astype(np.float64).tolist(),
        "levanter_routed_experts": np.transpose(expert_ids_np, (1, 0, 2)).astype(np.int64).tolist(),
        "levanter_router_margin": np.transpose(router_margins_np, (1, 0)).astype(np.float64).tolist(),
    }
    if routing_debug_payload is not None:
        payload["levanter_routing_debug"] = routing_debug_payload
    if forward_debug_payload is not None:
        payload["levanter_forward_debug"] = forward_debug_payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, sort_keys=True)
    print(f"reference_json={output_path}")
    print(f"reference_score_token_ids={payload['score_token_ids']}")
    print(f"reference_continuation_logprobs={payload['levanter_continuation_logprobs']}")
    if routing_debug_payload is not None:
        print("levanter_routing_debug=" + json.dumps(routing_debug_payload, sort_keys=True))
    if forward_debug_payload is not None:
        print("levanter_forward_debug=" + json.dumps(forward_debug_payload, sort_keys=True))


def _native_greedy_generation(
    tpu_model: Any,
    prompt_ids: jax.Array,
    attention_metadata_mod: Any,
    generation_tokens: int,
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    token_ids = prompt_ids
    generated_ids = []
    next_token_logits = []
    for _ in range(generation_tokens):
        _, logits, _ = _native_forward(tpu_model, token_ids, attention_metadata_mod)
        step_logits = logits[-1]
        token_id = int(np.asarray(jnp.argmax(step_logits, axis=-1)))
        generated_ids.append(token_id)
        next_token_logits.append(_np(step_logits))
        token_ids = _append_token(token_ids, token_id)

    return (
        np.asarray(generated_ids, dtype=np.int32),
        np.stack(next_token_logits, axis=0),
        token_ids,
    )


def _assert_generation_matches_reference(
    *,
    name: str,
    actual_generated_ids: np.ndarray,
    actual_logits: np.ndarray,
    actual_token_ids: jax.Array,
    expected_generated_ids: np.ndarray,
    expected_logits: np.ndarray,
    expected_token_ids: jax.Array,
) -> None:
    np.testing.assert_array_equal(actual_generated_ids, expected_generated_ids)
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(actual_token_ids), _np(expected_token_ids))
    print(
        f"realistic-generation: {name} full-forward greedy generation matched "
        f"Levanter reference for token IDs and logits across {len(expected_generated_ids)} steps"
    )


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


def _levanter_export_mesh() -> Mesh:
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    return Mesh(
        devices,
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def export_saved_checkpoint_via_levanter(
    lev_model: Transformer,
    checkpoint_dir: Path,
    artifact_dir: Path,
    *,
    max_shard_size: int | None = None,
) -> None:
    trainable, _ = eqx.partition(lev_model, is_inexact_arrayish)
    save_checkpoint({"params": trainable}, 0, checkpoint_dir)
    convert_config_kwargs = {}
    if max_shard_size is not None:
        convert_config_kwargs["max_shard_size"] = max_shard_size
    export_lm_to_hf.main(
        export_lm_to_hf.ConvertLmConfig(
            trainer=SimpleNamespace(
                device_mesh=jax.set_mesh(_levanter_export_mesh()),
                parameter_axis_mapping={},
            ),
            checkpoint_path=str(checkpoint_dir),
            output_dir=str(artifact_dir),
            checkpoint_subpath="params",
            model=lev_model.config,
            save_tokenizer=False,
            use_cpu=False,
            **convert_config_kwargs,
        )
    )


def export_training_state_checkpoint_via_levanter(
    cfg: GrugModelConfig,
    checkpoint_dir: Path,
    artifact_dir: Path,
    *,
    max_shard_size: int,
) -> None:
    export_lm_to_hf.main(
        export_lm_to_hf.ConvertLmConfig(
            trainer=SimpleNamespace(
                device_mesh=jax.set_mesh(_levanter_export_mesh()),
                parameter_axis_mapping={},
            ),
            checkpoint_path=str(checkpoint_dir),
            output_dir=str(artifact_dir),
            checkpoint_subpath="params",
            model=cfg,
            save_tokenizer=False,
            use_cpu=False,
            max_shard_size=max_shard_size,
        )
    )


def _exported_tensor_names(artifact_dir: Path, *, expect_sharded: bool) -> frozenset[str]:
    index_path = artifact_dir / _ARTIFACT_WEIGHTS_INDEX_FILE
    weights_path = artifact_dir / _ARTIFACT_WEIGHTS_FILE
    if not expect_sharded:
        if index_path.exists():
            raise AssertionError(f"single-file export unexpectedly wrote {index_path.name}")
        return frozenset(load_file(str(weights_path)))

    if not index_path.exists():
        raise AssertionError(f"forced sharded export did not write {index_path.name}")
    if weights_path.exists():
        raise AssertionError(f"forced sharded export unexpectedly wrote {weights_path.name}")

    with index_path.open() as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AssertionError(f"{index_path.name} does not contain a non-empty weight_map")

    shard_names = set(weight_map.values())
    if len(shard_names) < 2:
        raise AssertionError(f"forced sharded export wrote only {len(shard_names)} shard")

    for shard_name in sorted(shard_names):
        expected_in_shard = {name for name, shard in weight_map.items() if shard == shard_name}
        actual_in_shard = set(load_file(str(artifact_dir / shard_name)))
        if actual_in_shard != expected_in_shard:
            raise AssertionError(
                f"shard {shard_name} does not match the HF index: "
                f"missing={sorted(expected_in_shard - actual_in_shard)} "
                f"extra={sorted(actual_in_shard - expected_in_shard)}"
            )

    return frozenset(weight_map)


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _shard_count(artifact_dir: Path) -> int:
    return len(list(artifact_dir.glob("model-*-of-*.safetensors")))


def _realistic_prompt_ids(cfg: GrugModelConfig) -> jax.Array:
    prompt = np.array([1, 42, 128, 2048, 17, 3072, 5, 63], dtype=np.int32)
    return jnp.asarray((prompt % cfg.vocab_size)[None, :])


def _realistic_cfg_optimizer_steps(config_name: str) -> tuple[GrugModelConfig, Any, int, str]:
    canary_cfg, optimizer_config, steps = _canary_cfg_optimizer_steps()
    if config_name == "canary":
        return canary_cfg, optimizer_config, steps, "GRUG_MOE_TRIAL_MODEL"
    if config_name == "scaled":
        return _scaled_realistic_cfg(), optimizer_config, steps, "scaled production-structured fallback"
    if config_name == "small-diagnostic":
        return _small_diagnostic_cfg(), optimizer_config, steps, "small diagnostic GrugMoE"
    raise ValueError(f"unknown realistic config {config_name!r}")


def _default_realistic_max_shard_size(config_name: str) -> int:
    if config_name == "small-diagnostic":
        return _SMALL_DIAGNOSTIC_MAX_SHARD_SIZE_BYTES
    return _REALISTIC_MAX_SHARD_SIZE_BYTES


def _save_seeded_training_state_checkpoint(
    cfg: GrugModelConfig,
    checkpoint_dir: Path,
    *,
    optimizer_config: Any,
    num_train_steps: int,
    seed: int,
) -> Transformer:
    optimizer = optimizer_config.build(num_train_steps)
    mp = jmp.get_policy(_TRAINING_MP_POLICY)
    with _use_runtime_grug_mesh(for_init=True):
        state = initial_state(
            cfg,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(seed),
            ema_beta=None,
        )

    nonzero_probe = jnp.asarray(
        [
            state.params.token_embed[0, 0],
            state.params.output_proj[0, 0],
            state.params.blocks[0].mlp.expert_mlp.w_gate_up[0, 0, 0],
        ]
    )
    if not bool(np.any(np.asarray(jax.device_get(nonzero_probe)) != 0.0)):
        raise AssertionError("seeded training-state checkpoint probe was all zeros")

    save_checkpoint(state, 0, checkpoint_dir)
    lev_model = state.params
    del state
    gc.collect()
    return lev_model


def _clear_jax_memory() -> None:
    gc.collect()
    jax.clear_caches()


def _native_forward(
    tpu_model: Any,
    token_ids: jax.Array,
    attention_metadata_mod: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)
    metadata = attention_metadata_mod.AttentionMetadata(input_positions=positions)
    _, hidden, expert_ids = tpu_model.model([None] * tpu_model.model.config.num_layers, token_ids[0], metadata)
    logits = tpu_model.compute_logits(hidden)
    if expert_ids is None:
        raise AssertionError("native GrugMoE model did not return routed expert IDs")
    return hidden, logits, expert_ids


def _production_attention_metadata(
    attention_metadata_mod: Any,
    *,
    seq_len: int,
    block_size: int,
) -> Any:
    block_count = max(1, (seq_len + block_size - 1) // block_size)
    return attention_metadata_mod.AttentionMetadata(
        input_positions=jnp.arange(seq_len, dtype=jnp.int32),
        block_tables=jnp.arange(block_count, dtype=jnp.int32),
        seq_lens=jnp.asarray([seq_len], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, seq_len], dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
    )


def _native_production_forward(
    tpu_model: Any,
    token_ids: jax.Array,
    attention_metadata_mod: Any,
    *,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    from tpu_inference.runner.kv_cache import create_kv_caches  # noqa: PLC0415

    cfg = tpu_model.model.config
    seq_len = int(token_ids.shape[1])
    block_count = max(4, (seq_len + block_size - 1) // block_size)
    kv_caches = create_kv_caches(
        num_blocks=block_count,
        block_size=block_size,
        num_kv_heads=cfg.num_kv_heads,
        head_size=cfg.inferred_head_dim,
        mesh=_tpu_mesh(),
        layer_names=["layer"] * cfg.num_layers,
        cache_dtype=jnp.float32,
    )
    metadata = _production_attention_metadata(
        attention_metadata_mod,
        seq_len=seq_len,
        block_size=block_size,
    )
    _, hidden, expert_ids = tpu_model.model(kv_caches, token_ids[0], metadata)
    logits = tpu_model.compute_logits(hidden)
    if expert_ids is None:
        raise AssertionError("production GrugMoE model did not return routed expert IDs")
    return hidden, logits, expert_ids


def _selected_next_token_logprobs(
    logits: jax.Array,
    token_ids: jax.Array,
) -> jax.Array:
    targets = token_ids[0, 1:]
    logprobs = jax.nn.log_softmax(logits[:-1].astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]


def _delta_summary(actual: Any, expected: Any) -> dict[str, float]:
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {
        "max_abs_delta": float(np.max(delta)) if delta.size else 0.0,
        "mean_abs_delta": float(np.mean(delta)) if delta.size else 0.0,
    }


def _assert_delta_with_summary(
    *,
    label: str,
    actual: Any,
    expected: Any,
    tolerance: float,
) -> dict[str, float]:
    summary = _delta_summary(actual, expected)
    if summary["max_abs_delta"] > tolerance:
        raise AssertionError(f"{label} exceeded tolerance {tolerance}: {summary}")
    return summary


def check_production_attention_against_dense(tpu_grugmoe) -> None:
    attention_metadata_mod = importlib.import_module("tpu_inference.layers.common.attention_metadata")
    cfg = _small_diagnostic_cfg()
    token_ids = _realistic_prompt_ids(cfg)
    key = jax.random.PRNGKey(11)
    with _use_runtime_grug_mesh(for_init=True):
        lev_model = Transformer.init(cfg, key=key)

    with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
        dense_model = tpu_grugmoe.GrugMoeForCausalLM(
            _vllm_config(cfg, attention_mode=_DENSE_ATTENTION_MODE),
            jax.random.PRNGKey(12),
            _tpu_mesh(),
        )
        production_model = tpu_grugmoe.GrugMoeForCausalLM(
            _vllm_config(cfg, attention_mode=_PRODUCTION_ATTENTION_MODE),
            jax.random.PRNGKey(13),
            _tpu_mesh(),
        )
    _copy_transformer(dense_model, lev_model)
    _copy_transformer(production_model, lev_model)

    dense_hidden, dense_logits, dense_expert_ids = _native_forward(
        dense_model,
        token_ids,
        attention_metadata_mod,
    )
    (
        production_hidden,
        production_logits,
        production_expert_ids,
    ) = _native_production_forward(
        production_model,
        token_ids,
        attention_metadata_mod,
        block_size=16,
    )

    hidden_summary = _assert_delta_with_summary(
        label="production-attention hidden",
        actual=_np(production_hidden),
        expected=_np(dense_hidden),
        tolerance=_PRODUCTION_ATTENTION_LOGITS_TOLERANCE,
    )
    logits_summary = _assert_delta_with_summary(
        label="production-attention logits",
        actual=_np(production_logits),
        expected=_np(dense_logits),
        tolerance=_PRODUCTION_ATTENTION_LOGITS_TOLERANCE,
    )
    logprob_summary = _assert_delta_with_summary(
        label="production-attention selected next-token logprobs",
        actual=_np(_selected_next_token_logprobs(production_logits, token_ids)),
        expected=_np(_selected_next_token_logprobs(dense_logits, token_ids)),
        tolerance=_PRODUCTION_ATTENTION_LOGPROB_TOLERANCE,
    )
    np.testing.assert_array_equal(
        _np(production_expert_ids),
        _np(dense_expert_ids),
    )
    print(
        "production-attention-smoke: "
        f"mode={_PRODUCTION_ATTENTION_MODE} reference={_DENSE_ATTENTION_MODE} "
        f"token_ids={np.asarray(token_ids).tolist()[0]} "
        f"hidden_delta={json.dumps(hidden_summary, sort_keys=True)} "
        f"logits_delta={json.dumps(logits_summary, sort_keys=True)} "
        f"logprob_delta={json.dumps(logprob_summary, sort_keys=True)}"
    )


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
    actual_hidden, actual_logits, actual_expert_ids = _native_forward(tpu_model, token_ids, attention_metadata_mod)
    expected_hidden, expected_expert_ids = jax.jit(_jax_full_forward)(lev_model, token_ids)
    expected_logits = _jax_logits(lev_model, expected_hidden)[0]

    np.testing.assert_allclose(_np(actual_hidden), _np(expected_hidden[0]), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(_np(actual_logits), _np(expected_logits), rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(actual_expert_ids), _np(expected_expert_ids))
    print("full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference")


def _check_inference_artifact_roundtrip(
    tpu_grugmoe,
    *,
    expect_sharded: bool,
) -> None:
    attention_metadata_mod = importlib.import_module("tpu_inference.layers.common.attention_metadata")
    cfg = _tiny_cfg()
    key = jax.random.PRNGKey(7)
    with _use_runtime_grug_mesh(for_init=True):
        lev_model = Transformer.init(cfg, key=key)

    token_ids = jnp.array([[1, 5, 3, 7, 2, 9]], dtype=jnp.int32)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "grugmoe-inference"
        checkpoint_dir = Path(tmp) / "checkpoints"
        export_saved_checkpoint_via_levanter(
            lev_model,
            checkpoint_dir,
            artifact_dir,
            max_shard_size=_FORCED_SHARD_SIZE_BYTES if expect_sharded else None,
        )
        expected_names = canonical_grugmoe_tensor_names(cfg)
        exported_names = _exported_tensor_names(artifact_dir, expect_sharded=expect_sharded)
        if exported_names != expected_names:
            raise AssertionError(
                "exported artifact tensors do not match the canonical schema: "
                f"missing={sorted(expected_names - exported_names)} "
                f"extra={sorted(exported_names - expected_names)}"
            )
        with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
            manual_model = tpu_grugmoe.GrugMoeForCausalLM(_vllm_config(cfg), jax.random.PRNGKey(0), _tpu_mesh())
            loaded_model = tpu_grugmoe.GrugMoeForCausalLM(
                _vllm_config(cfg, artifact_dir),
                jax.random.PRNGKey(1),
                _tpu_mesh(),
            )

        _copy_transformer(manual_model, lev_model)
        report = loaded_model.load_weights(jax.random.PRNGKey(2))
        if report.consumed != expected_names:
            raise AssertionError(
                "artifact load report does not match exported tensors: "
                f"missing={sorted(expected_names - report.consumed)} "
                f"extra={sorted(report.consumed - expected_names)}"
            )
        if report.missing or report.unexpected:
            raise AssertionError(
                f"artifact load report was not strict: missing={report.missing} unexpected={report.unexpected}"
            )

        manual_hidden, manual_logits, manual_expert_ids = _native_forward(
            manual_model,
            token_ids,
            attention_metadata_mod,
        )
        loaded_hidden, loaded_logits, loaded_expert_ids = _native_forward(
            loaded_model,
            token_ids,
            attention_metadata_mod,
        )

    np.testing.assert_allclose(_np(loaded_hidden), _np(manual_hidden), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(_np(loaded_logits), _np(manual_logits), rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(loaded_expert_ids), _np(manual_expert_ids))
    artifact_layout = "sharded" if expect_sharded else "single-file"
    print(
        f"artifact-{artifact_layout}: saved-checkpoint Levanter export matches manual-copy hidden states, "
        "logits, and routed expert IDs"
    )


def check_inference_artifact_roundtrip(tpu_grugmoe) -> None:
    _check_inference_artifact_roundtrip(tpu_grugmoe, expect_sharded=False)
    _check_inference_artifact_roundtrip(tpu_grugmoe, expect_sharded=True)


def check_large_sharded_artifact_smoke(tpu_grugmoe) -> None:
    cfg = _large_smoke_cfg()
    key = jax.random.PRNGKey(11)
    with _use_runtime_grug_mesh(for_init=True):
        lev_model = Transformer.init(cfg, key=key)

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "grugmoe-large-inference"
        checkpoint_dir = Path(tmp) / "checkpoints"
        export_saved_checkpoint_via_levanter(
            lev_model,
            checkpoint_dir,
            artifact_dir,
            max_shard_size=64 * 1024 * 1024,
        )
        expected_names = canonical_grugmoe_tensor_names(cfg)
        exported_names = _exported_tensor_names(artifact_dir, expect_sharded=True)
        if exported_names != expected_names:
            raise AssertionError(
                "large export tensors do not match the canonical schema: "
                f"missing={sorted(expected_names - exported_names)} "
                f"extra={sorted(exported_names - expected_names)}"
            )

        with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
            loaded_model = tpu_grugmoe.GrugMoeForCausalLM(
                _vllm_config(cfg, artifact_dir),
                jax.random.PRNGKey(12),
                _tpu_mesh(),
            )
        report = loaded_model.load_weights(jax.random.PRNGKey(13))
        if report.consumed != expected_names:
            raise AssertionError(
                "large artifact load report does not match exported tensors: "
                f"missing={sorted(expected_names - report.consumed)} "
                f"extra={sorted(report.consumed - expected_names)}"
            )
        if report.missing or report.unexpected:
            raise AssertionError(
                f"large artifact load report was not strict: missing={report.missing} unexpected={report.unexpected}"
            )

        artifact_size = _directory_size_bytes(artifact_dir)
        if not (1_000_000_000 <= artifact_size <= 5_000_000_000):
            raise AssertionError(f"large smoke artifact size {artifact_size} bytes is outside the 1-5GB target")
        shard_count = len(list(artifact_dir.glob("model-*-of-*.safetensors")))

    print(
        "artifact-large-sharded: zero-weight Levanter export loaded in native tpu-inference "
        f"({artifact_size} bytes, {shard_count} shards)"
    )


def check_realistic_training_state_roundtrip(
    tpu_grugmoe,
    *,
    config_name: str,
    output_dir: Path | None,
    max_shard_size: int,
    generation_tokens: int,
    reference_output_path: Path | None = None,
    routing_debug_token_position: int | None = None,
    routing_debug_layer: int | None = None,
    routing_debug_vector_limit: int = _ROUTING_DEBUG_VECTOR_LIMIT,
    forward_debug_token_position: int | None = None,
    forward_debug_layer: int | None = None,
    forward_debug_vector_limit: int = _ROUTING_DEBUG_VECTOR_LIMIT,
) -> None:
    attention_metadata_mod = importlib.import_module("tpu_inference.layers.common.attention_metadata")
    cfg, optimizer_config, num_train_steps, config_source = _realistic_cfg_optimizer_steps(config_name)
    if config_name != "small-diagnostic":
        _assert_production_relevant_structure(cfg)
    token_ids = _realistic_prompt_ids(cfg)
    if generation_tokens <= 0:
        raise ValueError(f"generation_tokens must be positive, got {generation_tokens}")
    if token_ids.shape[1] + generation_tokens > cfg.max_seq_len:
        raise ValueError(
            "fixed prompt plus generation tokens must fit in the configured context: "
            f"prompt={token_ids.shape[1]} generation_tokens={generation_tokens} max_seq_len={cfg.max_seq_len}"
        )
    expected_names = canonical_grugmoe_tensor_names(cfg)

    root = output_dir or Path(tempfile.mkdtemp(prefix="grugmoe-realistic-roundtrip-"))
    checkpoint_dir = root / "checkpoints"
    artifact_dir = root / "grugmoe-inference"
    if checkpoint_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"realistic roundtrip output path must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    print(f"realistic-roundtrip: output_dir={root}")
    print(f"realistic-roundtrip: config_source={config_source}")
    print(f"realistic-roundtrip: config={json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str)}")
    print(f"realistic-roundtrip: dtype_policy={_TRAINING_MP_POLICY}; native_forward_dtype=float32")
    print(f"realistic-roundtrip: prompt_ids={np.asarray(token_ids).tolist()[0]}")
    print(f"realistic-generation: greedy_new_tokens={generation_tokens}; sampling=false; kv_cache=false")

    lev_model = _save_seeded_training_state_checkpoint(
        cfg,
        checkpoint_dir,
        optimizer_config=optimizer_config,
        num_train_steps=num_train_steps,
        seed=23,
    )
    checkpoint_size = _directory_size_bytes(checkpoint_dir)

    expected_hidden, expected_expert_ids = jax.jit(_jax_full_forward)(lev_model, token_ids)
    expected_logits = _jax_logits(lev_model, expected_hidden)[0]
    expected_hidden_np = _np(expected_hidden[0])
    expected_logits_np = _np(expected_logits)
    expected_expert_ids_np = _np(expected_expert_ids)

    with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
        manual_model = tpu_grugmoe.GrugMoeForCausalLM(
            _vllm_config(cfg),
            jax.random.PRNGKey(24),
            _tpu_mesh(),
        )
    _copy_transformer(manual_model, lev_model)
    manual_hidden, manual_logits, manual_expert_ids = _native_forward(
        manual_model,
        token_ids,
        attention_metadata_mod,
    )
    np.testing.assert_allclose(_np(manual_hidden), expected_hidden_np, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(_np(manual_logits), expected_logits_np, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(manual_expert_ids), expected_expert_ids_np)
    print("realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts")

    (
        expected_generated_ids,
        expected_generation_logits,
        expected_generation_token_ids,
    ) = _levanter_greedy_generation_reference(lev_model, token_ids, generation_tokens)
    manual_generated_ids, manual_generation_logits, manual_generation_token_ids = _native_greedy_generation(
        manual_model,
        token_ids,
        attention_metadata_mod,
        generation_tokens,
    )
    _assert_generation_matches_reference(
        name="manual-copy native reference",
        actual_generated_ids=manual_generated_ids,
        actual_logits=manual_generation_logits,
        actual_token_ids=manual_generation_token_ids,
        expected_generated_ids=expected_generated_ids,
        expected_logits=expected_generation_logits,
        expected_token_ids=expected_generation_token_ids,
    )
    print(
        "realistic-generation: "
        f"prompt_ids={np.asarray(token_ids).tolist()[0]} "
        f"generated_ids={expected_generated_ids.tolist()} "
        f"final_token_ids={_np(expected_generation_token_ids).tolist()[0]}"
    )
    if reference_output_path is not None:
        _write_installed_vllm_reference(
            lev_model,
            token_ids,
            expected_generated_ids.tolist(),
            reference_output_path,
            routing_debug_token_position=routing_debug_token_position,
            routing_debug_layer=routing_debug_layer,
            routing_debug_vector_limit=routing_debug_vector_limit,
            forward_debug_token_position=forward_debug_token_position,
            forward_debug_layer=forward_debug_layer,
            forward_debug_vector_limit=forward_debug_vector_limit,
        )

    del manual_model
    del lev_model
    _clear_jax_memory()

    export_training_state_checkpoint_via_levanter(
        cfg,
        checkpoint_dir,
        artifact_dir,
        max_shard_size=max_shard_size,
    )
    exported_names = _exported_tensor_names(artifact_dir, expect_sharded=True)
    if exported_names != expected_names:
        raise AssertionError(
            "realistic export tensors do not match the canonical schema: "
            f"missing={sorted(expected_names - exported_names)} "
            f"extra={sorted(exported_names - expected_names)}"
        )

    with _patch_tpu_single_rank(tpu_grugmoe), jax.set_mesh(_tpu_mesh()):
        loaded_model = tpu_grugmoe.GrugMoeForCausalLM(
            _vllm_config(cfg, artifact_dir),
            jax.random.PRNGKey(25),
            _tpu_mesh(),
        )
    report = loaded_model.load_weights(jax.random.PRNGKey(26))
    if report.consumed != expected_names:
        raise AssertionError(
            "realistic artifact load report does not match exported tensors: "
            f"missing={sorted(expected_names - report.consumed)} "
            f"extra={sorted(report.consumed - expected_names)}"
        )
    if report.missing or report.unexpected:
        raise AssertionError(
            f"realistic artifact load report was not strict: missing={report.missing} unexpected={report.unexpected}"
        )

    loaded_hidden, loaded_logits, loaded_expert_ids = _native_forward(
        loaded_model,
        token_ids,
        attention_metadata_mod,
    )
    np.testing.assert_allclose(_np(loaded_hidden), expected_hidden_np, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(_np(loaded_logits), expected_logits_np, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(_np(loaded_expert_ids), expected_expert_ids_np)

    loaded_generated_ids, loaded_generation_logits, loaded_generation_token_ids = _native_greedy_generation(
        loaded_model,
        token_ids,
        attention_metadata_mod,
        generation_tokens,
    )
    _assert_generation_matches_reference(
        name="loaded native artifact",
        actual_generated_ids=loaded_generated_ids,
        actual_logits=loaded_generation_logits,
        actual_token_ids=loaded_generation_token_ids,
        expected_generated_ids=expected_generated_ids,
        expected_logits=expected_generation_logits,
        expected_token_ids=expected_generation_token_ids,
    )

    artifact_size = _directory_size_bytes(artifact_dir)
    shard_count = _shard_count(artifact_dir)
    print(
        "realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched "
        "Levanter/manual-copy hidden states, logits, and routed expert IDs"
    )
    print(
        "realistic-roundtrip: "
        f"checkpoint_dir={checkpoint_dir} checkpoint_bytes={checkpoint_size} "
        f"artifact_dir={artifact_dir} artifact_bytes={artifact_size} "
        f"shard_count={shard_count} max_shard_size={max_shard_size} "
        f"expected_tensors={len(expected_names)} consumed_tensors={len(report.consumed)} "
        f"missing={sorted(report.missing)} unexpected={sorted(report.unexpected)} "
        f"generation_tokens={generation_tokens} generated_ids={expected_generated_ids.tolist()}"
    )


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
    parser.add_argument(
        "--large-smoke",
        action="store_true",
        help="Also run a 1-5GB zero-weight sharded export/load smoke.",
    )
    parser.add_argument(
        "--realistic-roundtrip",
        action="store_true",
        help="Run the seeded non-zero GrugTrainState checkpoint -> sharded HF -> native tpu-inference parity check.",
    )
    parser.add_argument(
        "--production-attention-smoke",
        action="store_true",
        help=("Run the small production-attention KV-cache path against the dense reference."),
    )
    parser.add_argument(
        "--realistic-config",
        choices=("canary", "scaled", "small-diagnostic"),
        default="canary",
        help=(
            "Use the full GRUG_MOE_TRIAL_MODEL canary config, a smaller config "
            "that preserves its production MoE/GQA structure, or a compact diagnostic config."
        ),
    )
    parser.add_argument(
        "--realistic-output-dir",
        type=Path,
        default=None,
        help="Directory to leave the realistic checkpoint and HF artifact in. Defaults to a new /tmp directory.",
    )
    parser.add_argument(
        "--realistic-max-shard-size",
        type=int,
        default=None,
        help="Forced max safetensors shard size for the realistic HF export.",
    )
    parser.add_argument(
        "--realistic-generation-tokens",
        type=int,
        default=_REALISTIC_GENERATION_TOKENS,
        help="Number of deterministic greedy full-forward tokens to generate in the realistic roundtrip.",
    )
    parser.add_argument(
        "--routing-debug-token-position",
        type=int,
        default=None,
        help="When writing an installed-vLLM reference JSON, include routing math for this token position.",
    )
    parser.add_argument(
        "--routing-debug-layer",
        type=int,
        default=None,
        help="When writing an installed-vLLM reference JSON, include routing math for this decoder layer.",
    )
    parser.add_argument(
        "--routing-debug-vector-limit",
        type=int,
        default=_ROUTING_DEBUG_VECTOR_LIMIT,
        help="Maximum number of values to include inline for large routing-debug vectors.",
    )
    parser.add_argument(
        "--forward-debug-token-position",
        type=int,
        default=None,
        help="When writing an installed-vLLM reference JSON, include forward-pass stages for this token position.",
    )
    parser.add_argument(
        "--forward-debug-layer",
        type=int,
        default=None,
        help="When writing an installed-vLLM reference JSON, include forward-pass stages for this decoder layer.",
    )
    parser.add_argument(
        "--forward-debug-vector-limit",
        type=int,
        default=_ROUTING_DEBUG_VECTOR_LIMIT,
        help="Maximum number of values to include inline for large forward-debug vectors.",
    )
    args = parser.parse_args()

    tpu_grugmoe = _load_tpu_grugmoe(args.tpu_inference_root.resolve())
    check_moe_component(tpu_grugmoe)
    if not args.component_only:
        check_full_forward(tpu_grugmoe)
        if args.production_attention_smoke:
            check_production_attention_against_dense(tpu_grugmoe)
        check_inference_artifact_roundtrip(tpu_grugmoe)
        if args.large_smoke:
            check_large_sharded_artifact_smoke(tpu_grugmoe)
    if args.realistic_roundtrip:
        realistic_max_shard_size = (
            args.realistic_max_shard_size
            if args.realistic_max_shard_size is not None
            else _default_realistic_max_shard_size(args.realistic_config)
        )
        check_realistic_training_state_roundtrip(
            tpu_grugmoe,
            config_name=args.realistic_config,
            output_dir=args.realistic_output_dir,
            max_shard_size=realistic_max_shard_size,
            generation_tokens=args.realistic_generation_tokens,
            routing_debug_token_position=args.routing_debug_token_position,
            routing_debug_layer=args.routing_debug_layer,
            routing_debug_vector_limit=args.routing_debug_vector_limit,
            forward_debug_token_position=args.forward_debug_token_position,
            forward_debug_layer=args.forward_debug_layer,
            forward_debug_vector_limit=args.forward_debug_vector_limit,
        )


if __name__ == "__main__":
    main()
