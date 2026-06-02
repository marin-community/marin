# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny GrugMoE parity harness for the vLLM TPU support branch.

Run from the Marin worktree:

    uv run python -m experiments.grug.moe.vllm_tpu_parity \
      --vllm-root ../grugmoe-vllm-tpu-vllm

The component check compares vLLM's PyTorch GrugMoE MLP against Levanter's
`moe_mlp` with the same selected experts and sigmoid combine weights. The full
check copies a tiny Levanter `Transformer` into vLLM's correctness-first model
and compares hidden states.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import levanter.grug.grug_moe as levanter_grug_moe
import numpy as np
import torch
from jax.sharding import AxisType, Mesh
from levanter.grug.grug_moe import moe_mlp

import experiments.grug.moe.model as grug_moe_model
from experiments.grug.moe.model import (
    GrugModelConfig,
    Transformer,
)


class _EmptyMesh:
    empty = True
    shape: ClassVar[dict[str, int]] = {}


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
    grug_moe_model.shard_map = lambda fn, **kwargs: (
        lambda s_ma: jnp.zeros((s_ma.shape[1],), dtype=s_ma.dtype)
    )
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


def _default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


def _install_vllm_import_stubs() -> None:
    """Install enough vLLM modules to import grugmoe.py directly.

    This keeps the parity harness independent from a full vLLM editable install.
    It does not exercise vLLM registry loading; use the vLLM test command for
    that path.
    """
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = object
    model_executor_mod = types.ModuleType("vllm.model_executor")
    model_loader_mod = types.ModuleType("vllm.model_executor.model_loader")
    weight_utils_mod = types.ModuleType(
        "vllm.model_executor.model_loader.weight_utils"
    )
    weight_utils_mod.default_weight_loader = _default_weight_loader
    sequence_mod = types.ModuleType("vllm.sequence")
    sequence_mod.IntermediateTensors = object

    sys.modules.update(
        {
            "vllm": vllm_mod,
            "vllm.config": config_mod,
            "vllm.model_executor": model_executor_mod,
            "vllm.model_executor.model_loader": model_loader_mod,
            "vllm.model_executor.model_loader.weight_utils": weight_utils_mod,
            "vllm.sequence": sequence_mod,
        }
    )


def _load_vllm_grugmoe(vllm_root: Path):
    module_path = vllm_root / "vllm" / "model_executor" / "models" / "grugmoe.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing vLLM grugmoe.py at {module_path}")

    _install_vllm_import_stubs()
    spec = importlib.util.spec_from_file_location("grugmoe_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _np(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True)


def _copy_param(param: torch.nn.Parameter, value: Any) -> None:
    with torch.no_grad():
        param.copy_(torch.as_tensor(_np(value), dtype=param.dtype))


def _tiny_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=19,
        hidden_dim=8,
        intermediate_dim=12,
        shared_expert_intermediate_dim=10,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.02,
        moe_implementation="scatter",
    )


def _vllm_cfg(vllm_grugmoe, cfg: GrugModelConfig):
    return vllm_grugmoe.GrugMoeConfig(
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
    ).validate()


def _route_jax(
    x: jax.Array,
    router: jax.Array,
    router_bias: jax.Array,
    *,
    num_experts_per_token: int,
) -> tuple[jax.Array, jax.Array]:
    router_logits = jnp.einsum("td,de->te", x, router).astype(jnp.float32)
    biased_logits = router_logits + router_bias.astype(jnp.float32)
    topk = min(router.shape[1], num_experts_per_token + 1)
    _, selected = jax.lax.top_k(biased_logits, topk)
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

    aligned_v = _jax_align_kv_heads(v, cfg.num_heads)
    dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
    v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
    attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
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


def _jax_moe_mlp(mlp: Any, x: jax.Array) -> jax.Array:
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
    return out.reshape(bsz, seq_len, hidden_dim)


def _jax_full_forward(model: Transformer, token_ids: jax.Array) -> jax.Array:
    cfg = model.config
    _bsz, seq_len = token_ids.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], token_ids.shape)
    hidden = model.token_embed[token_ids]
    hidden = _jax_gated_norm(model.embed_gated_norm, _jax_rms_norm(hidden, model.embed_norm.weight, cfg.layer_norm_eps))

    short_window = cfg.sliding_window // 2
    for i, block in enumerate(model.blocks):
        layer_window = cfg.sliding_window if i % 4 == 3 else short_window
        attn_in = _jax_gated_norm(
            block.attn_gated_norm,
            _jax_rms_norm(hidden, block.rms_attn.weight, block.rms_attn.eps),
        )
        hidden = hidden + _jax_attention(block.attn, attn_in, positions, layer_window)
        mlp_in = _jax_gated_norm(block.mlp_gated_norm, _jax_rms_norm(hidden, block.rms_mlp.weight, block.rms_mlp.eps))
        mlp_out = _jax_moe_mlp(block.mlp, mlp_in)
        if block.shared is not None:
            mlp_out = mlp_out + _jax_dense_mlp(block.shared, mlp_in)
        hidden = hidden + mlp_out

    return _jax_gated_norm(model.final_gated_norm, _jax_rms_norm(hidden, model.final_norm.weight, model.final_norm.eps))


def check_moe_component(vllm_grugmoe) -> None:
    cfg = _tiny_cfg()
    v_cfg = _vllm_cfg(vllm_grugmoe, cfg)
    torch_mlp = vllm_grugmoe.GrugMoeMLP(v_cfg)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, cfg.hidden_dim)).astype(np.float32)
    router = rng.normal(size=(cfg.hidden_dim, cfg.num_experts)).astype(np.float32)
    router_bias = np.array([0.0, 0.0, 3.0, -2.0], dtype=np.float32)
    w_gate_up = rng.normal(
        size=(cfg.num_experts, cfg.hidden_dim, 2 * cfg.intermediate_dim)
    ).astype(np.float32)
    w_down = rng.normal(
        size=(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim)
    ).astype(np.float32)

    with torch.no_grad():
        torch_mlp.router.copy_(torch.from_numpy(router))
        torch_mlp.router_bias.copy_(torch.from_numpy(router_bias))
        torch_mlp.w_gate_up.copy_(torch.from_numpy(w_gate_up))
        torch_mlp.w_down.copy_(torch.from_numpy(w_down))

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
    actual = torch_mlp(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(actual, _np(expected), rtol=1e-5, atol=1e-5)
    print("component: GrugMoeMLP matches Levanter moe_mlp")


def _copy_transformer(v_model: torch.nn.Module, lev_model: Transformer) -> None:
    _copy_param(v_model.token_embed, lev_model.token_embed)
    _copy_param(v_model.output_proj, lev_model.output_proj)
    _copy_param(v_model.embed_norm.weight, lev_model.embed_norm.weight)
    _copy_param(v_model.embed_gated_norm.w_down, lev_model.embed_gated_norm.w_down)
    _copy_param(v_model.embed_gated_norm.w_up, lev_model.embed_gated_norm.w_up)
    _copy_param(v_model.final_norm.weight, lev_model.final_norm.weight)
    _copy_param(v_model.final_gated_norm.w_down, lev_model.final_gated_norm.w_down)
    _copy_param(v_model.final_gated_norm.w_up, lev_model.final_gated_norm.w_up)

    for v_block, l_block in zip(v_model.blocks, lev_model.blocks, strict=True):
        _copy_param(v_block.rms_attn.weight, l_block.rms_attn.weight)
        _copy_param(v_block.attn_gated_norm.w_down, l_block.attn_gated_norm.w_down)
        _copy_param(v_block.attn_gated_norm.w_up, l_block.attn_gated_norm.w_up)
        _copy_param(v_block.attn.w_q, l_block.attn.w_q)
        _copy_param(v_block.attn.w_k, l_block.attn.w_k)
        _copy_param(v_block.attn.w_v, l_block.attn.w_v)
        _copy_param(v_block.attn.w_o, l_block.attn.w_o)
        _copy_param(v_block.attn.attn_gate, l_block.attn.attn_gate)
        _copy_param(v_block.rms_mlp.weight, l_block.rms_mlp.weight)
        _copy_param(v_block.mlp_gated_norm.w_down, l_block.mlp_gated_norm.w_down)
        _copy_param(v_block.mlp_gated_norm.w_up, l_block.mlp_gated_norm.w_up)
        _copy_param(v_block.mlp.router, l_block.mlp.router)
        _copy_param(v_block.mlp.router_bias, l_block.mlp.router_bias)
        _copy_param(v_block.mlp.w_gate_up, l_block.mlp.expert_mlp.w_gate_up)
        _copy_param(v_block.mlp.w_down, l_block.mlp.expert_mlp.w_down)
        if v_block.shared is not None and l_block.shared is not None:
            _copy_param(v_block.shared.w_gate, l_block.shared.w_gate)
            _copy_param(v_block.shared.w_up, l_block.shared.w_up)
            _copy_param(v_block.shared.w_down, l_block.shared.w_down)


def check_full_forward(vllm_grugmoe) -> None:
    cfg = _tiny_cfg()
    key = jax.random.PRNGKey(7)
    with _use_runtime_grug_mesh(for_init=True):
        lev_model = Transformer.init(cfg, key=key)

    v_model = vllm_grugmoe.GrugMoeModel(_vllm_cfg(vllm_grugmoe, cfg))
    _copy_transformer(v_model, lev_model)

    token_ids = np.array([[1, 5, 3, 7]], dtype=np.int32)
    positions = torch.arange(token_ids.shape[1], dtype=torch.int64)[None, :]
    with torch.no_grad():
        actual = v_model(torch.from_numpy(token_ids), positions).detach().numpy()

    expected = jax.jit(_jax_full_forward)(lev_model, jnp.asarray(token_ids))

    np.testing.assert_allclose(actual, _np(expected), rtol=5e-4, atol=5e-4)
    print("full: GrugMoeModel hidden states match Levanter Transformer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=Path("../grugmoe-vllm-tpu-vllm"),
        help="Path to the vLLM checkout containing grugmoe.py.",
    )
    parser.add_argument(
        "--component-only",
        action="store_true",
        help="Run only the MoE component parity check.",
    )
    args = parser.parse_args()

    vllm_grugmoe = _load_vllm_grugmoe(args.vllm_root.resolve())
    check_moe_component(vllm_grugmoe)
    if not args.component_only:
        check_full_forward(vllm_grugmoe)


if __name__ == "__main__":
    main()
