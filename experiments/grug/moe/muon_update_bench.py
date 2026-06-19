# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Grug MoE MuonH update benchmark.

This isolates the optimizer update from the full Grug forward/backward path so
grouped Muon changes can be lowered and timed quickly on a single GPU node.

Typical CoreWeave usage:

```
RUN_ID=MUON-BENCH-... \
MUON_BENCH_KINDS=muonh_update,muon_direction,ns4d_data_group,ns4d_dotonly_einsum \
MUON_BENCH_ORTHOGONALIZATION_LAYOUT=stack_batch_4d_sharded \
MUON_BENCH_SWEEP_BACKEND_STEPS=1,5 \
MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 \
scratch/launch_muon_update_bench_executor_n1.sh
```

The JSONL summary includes compile time, execution time, HLO dot/collective
counts, and an approximate Newton-Schulz dot-kernel speed-of-light estimate.
"""

import argparse
import json
import math
import re
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
from jax import lax, shard_map
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, reshard, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import grouped_moe_mlp
from levanter.optim.grugmuon import (
    DEFAULT_MAX_GROUPED_STACK_SIZE,
    STACK_BATCH_SHARDED,
    STACK_PADDING_MAX_OVERHEAD,
    _grug_scale_with_muon,
    _target_sharding,
    _zeropower_via_newtonschulz_batched_stack_sharded,
)
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.optimizer import (
    _expert_momentum_sharding,
    _scale_invariant_hyperball_updates,
    _uses_adamh_baseline_adam_group,
    scale_with_grug_muonh,
)

MAY_HIDDEN_DIM = 2560
MAY_INTERMEDIATE_DIM = 1280
MAY_SHARED_EXPERT_INTERMEDIATE_DIM = MAY_HIDDEN_DIM
MAY_NUM_EXPERTS = 256
MAY_GATED_NORM_RANK = 128
MAY_HIDDEN_HEAD_RATIO = 128
MAY_GQA_RATIO = 4
MAY_MODEL_AXIS = 1
MAY_EXPERT_AXIS = 8
MAY_REPLICA_AXIS = 1
MAY_BACKEND_STEPS = 1
MAY_LEARNING_RATE = 0.02
MAY_MOMENTUM = 0.95
MAY_MUON_EPSILON = 1e-8
DEFAULT_WARMUP = 1
DEFAULT_ITERS = 3
NOMINAL_H100_BF16_DENSE_TFLOPS = 989.0
DOT_GENERAL_RE = re.compile(r"stablehlo\.dot_general")
BATCHED_STACK_DOT_RE = re.compile(r"batching_dims = \[0\] x \[0\]")
TWO_BATCH_AXIS_DOT_RE = re.compile(r"batching_dims = \[0, 1\] x \[0, 1\]")
CUSTOM_CALL_RE = re.compile(r"\bcustom-call\b|stablehlo\.custom_call|mhlo\.custom_call|custom_call")
GPU_GEMM_CUSTOM_CALL_RE = re.compile(r"(?:cublas\$gemm|gemm|cublas|cutlass|triton)", re.IGNORECASE)
ALL_GATHER_RE = re.compile(r"\ball-gather\b|stablehlo\.all_gather|mhlo\.all_gather")
ALL_REDUCE_RE = re.compile(r"\ball-reduce\b|stablehlo\.all_reduce|mhlo\.all_reduce")
REDUCE_SCATTER_RE = re.compile(r"\breduce-scatter\b|stablehlo\.reduce_scatter|mhlo\.reduce_scatter")
ALL_TO_ALL_RE = re.compile(r"\ball-to-all\b|stablehlo\.all_to_all|mhlo\.all_to_all")
COLLECTIVE_PERMUTE_RE = re.compile(r"\bcollective-permute\b|stablehlo\.collective_permute|mhlo\.collective_permute")
MUONH_UPDATE_BENCH = "muonh_update"
MUON_DIRECTION_BENCH = "muon_direction"
HYPERBALL_ONLY_BENCH = "hyperball_only"
NS4D_REPLICATED_GROUP_BENCH = "ns4d_replicated_group"
NS4D_DATA_GROUP_BENCH = "ns4d_data_group"
NS4D_DATA_GROUP_APPLY_BENCH = "ns4d_data_group_apply"
EXPERT_GROUPED_APPLY_BOUNDARY_BENCH = "expert_grouped_apply_boundary"
EXPERT_GROUPED_LAYER_SLICE_BENCH = "expert_grouped_layer_slice_boundary"
EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH = "expert_grouped_single_layer_slice_boundary"
EXPERT_GROUPED_BANK_CONSUMER_BENCH = "expert_grouped_bank_consumer"
EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH = "expert_grouped_moe_mlp_consumer"
EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH = "expert_grouped_optimizer_apply"
EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "expert_grouped_muonh_optimizer_apply"
EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "expert_only_grouped_muonh_optimizer_apply"
EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH = "expert_fsdp_grouped_apply_boundary"
EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH = "expert_fsdp_grouped_restore_boundary"
EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH = "expert_fsdp_grouped_target_restore_boundary"
EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH = "expert_fsdp_grouped_target_apply_boundary"
EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH = "expert_fsdp_grouped_explicit_restore_boundary"
EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH = "expert_fsdp_grouped_explicit_apply_boundary"
EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH = "expert_fsdp_grouped_explicit_a2a_apply_boundary"
EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH = "expert_fsdp_grouped_updates_muonh_updates"
EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH = "expert_fsdp_grouped_updates_muonh_apply"
EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "expert_fsdp_grouped_muonh_optimizer_apply"
FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH = "full_production_muonh_optimizer_apply"
FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH = "full_production_grouped_2d_muonh_optimizer_apply"
FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH = "full_production_grouped_2d_direct_apply"
FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH = "full_production_grouped_2d_persistent_apply"
ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH = "ordinary_2d_muonh_optimizer_apply"
ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "ordinary_2d_grouped_muonh_optimizer_apply"
ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH = "ordinary_2d_grouped_direct_apply"
ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH = "ordinary_2d_grouped_persistent_apply"
ORDINARY_2D_GROUPED_STACK_NS_BENCH = "ordinary_2d_grouped_stack_ns"
ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH = "ordinary_2d_grouped_restore_split"
FULL_PRODUCTION_APPLY_ONLY_BENCH = "full_production_apply_only"
NS4D_DATA_RESHARD_RESTORE_BENCH = "ns4d_data_reshard_restore"
NS4D_DATA_INDEX_RESTORE_BENCH = "ns4d_data_index_restore"
NS4D_DOTONLY_EINSUM_BENCH = "ns4d_dotonly_einsum"
NS4D_DOTONLY_MATMUL_BENCH = "ns4d_dotonly_matmul"
NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH = "ns4d_dotonly_lax_dot_general"
NS4D_PADDED_GROUP_BENCH = "ns4d_padded_group"
NS4D_DOTONLY_MATMUL_PADDED_BENCH = "ns4d_dotonly_matmul_padded"
BENCH_KINDS = (
    MUONH_UPDATE_BENCH,
    MUON_DIRECTION_BENCH,
    HYPERBALL_ONLY_BENCH,
    NS4D_REPLICATED_GROUP_BENCH,
    NS4D_DATA_GROUP_BENCH,
    NS4D_DATA_GROUP_APPLY_BENCH,
    EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
    ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
    ORDINARY_2D_GROUPED_STACK_NS_BENCH,
    ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
    FULL_PRODUCTION_APPLY_ONLY_BENCH,
    NS4D_DATA_RESHARD_RESTORE_BENCH,
    NS4D_DATA_INDEX_RESTORE_BENCH,
    NS4D_DOTONLY_EINSUM_BENCH,
    NS4D_DOTONLY_MATMUL_BENCH,
    NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH,
    NS4D_PADDED_GROUP_BENCH,
    NS4D_DOTONLY_MATMUL_PADDED_BENCH,
)
NS4D_DATA_SHARDED_BENCHES = (
    NS4D_DATA_GROUP_BENCH,
    NS4D_DATA_GROUP_APPLY_BENCH,
    EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    FULL_PRODUCTION_APPLY_ONLY_BENCH,
    NS4D_DATA_RESHARD_RESTORE_BENCH,
    NS4D_DATA_INDEX_RESTORE_BENCH,
    NS4D_DOTONLY_EINSUM_BENCH,
    NS4D_DOTONLY_MATMUL_BENCH,
    NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH,
)
NS4D_PADDED_BENCHES = (NS4D_PADDED_GROUP_BENCH, NS4D_DOTONLY_MATMUL_PADDED_BENCH)
GROUPED_APPLY_BOUNDARY_BENCHES = (
    NS4D_DATA_GROUP_APPLY_BENCH,
    EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
    FULL_PRODUCTION_APPLY_ONLY_BENCH,
)
GROUPED_PARAM_INPUT_BENCHES = (
    EXPERT_GROUPED_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
)
GROUPED_OPTIMIZER_APPLY_BENCHES = (
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
)
FULL_PRODUCTION_MUONH_BENCHES = (
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
)
ORDINARY_2D_DECOMPOSITION_BENCHES = (
    ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
    ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    ORDINARY_2D_GROUPED_STACK_NS_BENCH,
    ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
)
NON_NS_APPLY_BENCHES = (FULL_PRODUCTION_APPLY_ONLY_BENCH,)
GROUPED_EXPERT_PATHS = (
    "blocks[*].mlp.expert_mlp.w_gate_up",
    "blocks[*].mlp.expert_mlp.w_down",
)
DEFAULT_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT = 1
FULL_PRODUCTION_MUONH_PATHS = (
    "ordinary_blocks[*].attn.w_q",
    "ordinary_blocks[*].attn.w_k",
    "ordinary_blocks[*].attn.w_v",
    "ordinary_blocks[*].attn.w_o",
    "ordinary_blocks[*].attn_gated_norm.w_down",
    "ordinary_blocks[*].attn_gated_norm.w_up",
    "ordinary_blocks[*].mlp_gated_norm.w_down",
    "ordinary_blocks[*].mlp_gated_norm.w_up",
    "ordinary_blocks[*].shared.w_gate",
    "ordinary_blocks[*].shared.w_up",
    "ordinary_blocks[*].shared.w_down",
    *GROUPED_EXPERT_PATHS,
)
ORDINARY_PARAM_PATHS = (
    "ordinary_blocks[*].mlp.w_in",
    "ordinary_blocks[*].router.bias",
    "token_embed",
)


def is_full_production_muonh_bench(bench_kind: str) -> bool:
    return bench_kind in FULL_PRODUCTION_MUONH_BENCHES


def groups_2d_muonh_leaves(bench_kind: str) -> bool:
    return bench_kind in (
        FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    )


@dataclass(frozen=True)
class BenchConfig:
    layers: int
    ns4d_group_size: int | None
    ns4d_group_axis: str
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    dtype: str
    backend_steps: int
    orthogonalization_layout: str
    max_grouped_stack_size: int
    replica_axis: int
    data_axis: int
    expert_axis: int
    model_axis: int
    learning_rate: float
    ns_compute_dtype: str = "input"
    grouped_expert_consumer_tokens_per_expert: int = DEFAULT_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT


@dataclass(frozen=True)
class GroupEstimate:
    leaf_shape: tuple[int, int, int]
    leaves: int
    chunks: list[int]
    grouped_chunks: int


@dataclass(frozen=True)
class Grouped2DEstimate:
    leaf_shape: tuple[int, int]
    sharding_spec: str
    leaves: int
    chunks: list[int]
    grouped_chunks: int


@dataclass(frozen=True)
class PersistentGrouped2DMetadata:
    source_paths: tuple[str, ...]
    leaf_shape: tuple[int, int]
    source_sharding_spec: str
    target_sharding_spec: str
    valid_stack_length: int
    padded_stack_length: int


@dataclass(frozen=True)
class HloSummary:
    characters: int
    dot_general: int
    batched_stack_dot_general: int
    two_batch_axis_dot_general: int
    custom_call: int
    gpu_gemm_custom_call: int
    all_gather: int
    all_reduce: int
    reduce_scatter: int
    all_to_all: int
    collective_permute: int
    grouped_scope_mentions: int
    stack_sharded_scope_mentions: int
    pad_scope_mentions: int
    slice_scope_mentions: int


@dataclass(frozen=True)
class TimingSummary:
    compile_seconds: float | None
    compiled_hlo: HloSummary | None
    times: list[float]
    median_seconds: float | None
    mean_seconds: float | None
    min_seconds: float | None
    stdev_seconds: float | None


def emit_jsonl(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def dtype_from_name(name: str) -> jnp.dtype:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if normalized in ("fp32", "float32"):
        return jnp.float32
    if normalized in ("fp16", "float16"):
        return jnp.float16
    raise ValueError(f"Unsupported dtype={name!r}")


def dtype_name(dtype: jnp.dtype) -> str:
    return str(jnp.dtype(dtype))


def numpy_dtype_from_name(name: str) -> np.dtype:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return np.dtype(ml_dtypes.bfloat16)
    if normalized in ("fp32", "float32"):
        return np.dtype(np.float32)
    if normalized in ("fp16", "float16"):
        return np.dtype(np.float16)
    raise ValueError(f"Unsupported dtype={name!r}")


def ns_compute_dtype_from_name(name: str, input_dtype: jnp.dtype) -> jnp.dtype:
    if name == "input":
        return jnp.dtype(input_dtype)
    return dtype_from_name(name)


def ns_compute_dtype_name(name: str, input_dtype: jnp.dtype) -> str:
    if name == "input":
        return "input"
    return dtype_name(ns_compute_dtype_from_name(name, input_dtype))


def cast_for_ns_compute(x: jax.Array, config: BenchConfig) -> tuple[jax.Array, jnp.dtype]:
    original_dtype = jnp.dtype(x.dtype)
    compute_dtype = ns_compute_dtype_from_name(config.ns_compute_dtype, original_dtype)
    if compute_dtype == original_dtype:
        return x, original_dtype
    with jax.named_scope("muon_update_bench/ns_compute_dtype/cast_input"):
        return x.astype(compute_dtype), original_dtype


def restore_ns_compute_dtype(x: jax.Array, original_dtype: jnp.dtype) -> jax.Array:
    if jnp.dtype(x.dtype) == original_dtype:
        return x
    with jax.named_scope("muon_update_bench/ns_compute_dtype/cast_output"):
        return x.astype(original_dtype)


def zeropower_via_newtonschulz_4d_for_config(update: jax.Array, config: BenchConfig) -> jax.Array:
    update, original_dtype = cast_for_ns_compute(update, config)
    update = zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON)
    return restore_ns_compute_dtype(update, original_dtype)


def zeropower_via_newtonschulz_3d_stack_for_config(
    update: jax.Array,
    config: BenchConfig,
    target_pspec: NamedSharding | P | None,
) -> jax.Array:
    update, original_dtype = cast_for_ns_compute(update, config)
    update = _zeropower_via_newtonschulz_batched_stack_sharded(
        update,
        config.backend_steps,
        MAY_MUON_EPSILON,
        target_pspec=target_pspec,
    )
    return restore_ns_compute_dtype(update, original_dtype)


def newtonschulz_4d_dotonly_for_config(update: jax.Array, config: BenchConfig, implementation: str) -> jax.Array:
    update, original_dtype = cast_for_ns_compute(update, config)
    if implementation == "einsum":
        update = newtonschulz_4d_dotonly_einsum(update, config.backend_steps)
    elif implementation == "matmul":
        update = newtonschulz_4d_dotonly_matmul(update, config.backend_steps)
    elif implementation == "lax_dot_general":
        update = newtonschulz_4d_dotonly_lax_dot_general(update, config.backend_steps)
    else:
        raise ValueError(f"Unknown dot-only Newton-Schulz implementation {implementation!r}.")
    return restore_ns_compute_dtype(update, original_dtype)


def create_mesh(replica_axis: int, data_axis: int, expert_axis: int, model_axis: int) -> Mesh:
    expected_devices = replica_axis * data_axis * expert_axis * model_axis
    devices = np.asarray(jax.devices())
    if devices.size != expected_devices:
        raise ValueError(
            "Device count must match replica_axis * data_axis * expert_axis * model_axis; "
            f"got {devices.size} devices and expected {expected_devices} for "
            f"{replica_axis} * {data_axis} * {expert_axis} * {model_axis}."
        )
    return Mesh(
        devices.reshape((replica_axis, data_axis, expert_axis, model_axis)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def create_abstract_mesh(config: BenchConfig) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(config.replica_axis, config.data_axis, config.expert_axis, config.model_axis),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def maybe_abstract_mesh(config: BenchConfig, enabled: bool):
    if not enabled:
        return nullcontext()
    return use_abstract_mesh(create_abstract_mesh(config))


def expert_param_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("expert", None, None))


def expert_fsdp_param_sharding(mesh: Mesh, name: str) -> NamedSharding:
    """Production-like Grug MoE expert parameter sharding for model_axis=1."""
    if name == "w_gate_up":
        return NamedSharding(mesh, P("expert", "data", "model"))
    if name == "w_down":
        return NamedSharding(mesh, P("expert", "model", "data"))
    raise ValueError(f"Unknown expert parameter name {name!r}.")


def require_model_axis_one(config: BenchConfig) -> None:
    if config.model_axis != 1:
        raise ValueError(f"Muon update benchmark is single-node/model_axis=1 only, got model_axis={config.model_axis}.")


def synthetic_shapes(config: BenchConfig) -> dict[str, tuple[int, int, int]]:
    return {
        "w_gate_up": (config.num_experts, config.hidden_dim, 2 * config.intermediate_dim),
        "w_down": (config.num_experts, config.intermediate_dim, config.hidden_dim),
    }


def may_attention_dims(config: BenchConfig) -> tuple[int, int, int]:
    num_heads = max(1, config.hidden_dim // MAY_HIDDEN_HEAD_RATIO)
    target_kv_heads = max(1, num_heads // MAY_GQA_RATIO)
    num_kv_heads = next(kv_heads for kv_heads in range(target_kv_heads, 0, -1) if num_heads % kv_heads == 0)
    head_dim = config.hidden_dim // num_heads
    return num_heads, num_kv_heads, head_dim


def production_muonh_2d_shapes(config: BenchConfig) -> dict[str, tuple[int, int]]:
    num_heads, num_kv_heads, head_dim = may_attention_dims(config)
    hidden_dim = config.hidden_dim
    shared_dim = MAY_SHARED_EXPERT_INTERMEDIATE_DIM if hidden_dim == MAY_HIDDEN_DIM else hidden_dim
    return {
        "attn.w_q": (hidden_dim, num_heads * head_dim),
        "attn.w_k": (hidden_dim, num_kv_heads * head_dim),
        "attn.w_v": (hidden_dim, num_kv_heads * head_dim),
        "attn.w_o": (num_heads * head_dim, hidden_dim),
        "attn_gated_norm.w_down": (hidden_dim, MAY_GATED_NORM_RANK),
        "attn_gated_norm.w_up": (MAY_GATED_NORM_RANK, hidden_dim),
        "mlp_gated_norm.w_down": (hidden_dim, MAY_GATED_NORM_RANK),
        "mlp_gated_norm.w_up": (MAY_GATED_NORM_RANK, hidden_dim),
        "shared.w_gate": (hidden_dim, shared_dim),
        "shared.w_up": (hidden_dim, shared_dim),
        "shared.w_down": (shared_dim, hidden_dim),
    }


def synthetic_shardings(mesh: Mesh, config: BenchConfig) -> Any:
    sharding = expert_param_sharding(mesh)
    return {
        "layers": tuple(
            {"mlp": {"expert_mlp": {name: sharding for name in synthetic_shapes(config)}}} for _ in range(config.layers)
        )
    }


def synthetic_fsdp_expert_shardings(mesh: Mesh, config: BenchConfig) -> Any:
    return {
        "layers": tuple(
            {"mlp": {"expert_mlp": {name: expert_fsdp_param_sharding(mesh, name) for name in synthetic_shapes(config)}}}
            for _ in range(config.layers)
        )
    }


def mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
    shape = getattr(mesh, "shape", None)
    if shape is not None:
        return int(shape.get(axis_name, 1))
    return int(dict(zip(mesh.axis_names, mesh.devices.shape, strict=True)).get(axis_name, 1))


def ns4d_group_axis(config: BenchConfig) -> str | tuple[str, ...] | None:
    if config.ns4d_group_axis == "none":
        return None
    if config.ns4d_group_axis == "replica_dcn,data":
        return ("replica_dcn", "data")
    if config.ns4d_group_axis == "data,replica_dcn":
        return ("data", "replica_dcn")
    if config.ns4d_group_axis not in ("data", "replica_dcn"):
        raise ValueError(
            "ns4d_group_axis must be one of 'data', 'replica_dcn', 'replica_dcn,data', "
            f"'data,replica_dcn', or 'none'; got {config.ns4d_group_axis!r}."
        )
    return config.ns4d_group_axis


def ns4d_axis_size(config: BenchConfig) -> int:
    group_axis = ns4d_group_axis(config)
    if group_axis is None:
        return 1
    if isinstance(group_axis, tuple):
        return math.prod(config.data_axis if axis == "data" else config.replica_axis for axis in group_axis)
    if group_axis == "data":
        return config.data_axis
    if group_axis == "replica_dcn":
        return config.replica_axis
    return 1


def padded_ns4d_group_size(config: BenchConfig, bench_kind: str) -> int:
    group_size = ns4d_group_size(config)
    if bench_kind in (
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ):
        axis_size = ns4d_axis_size(config)
        if axis_size <= 1:
            return group_size
        return math.ceil(group_size / axis_size) * axis_size
    if bench_kind not in NS4D_PADDED_BENCHES:
        return group_size
    axis_size = ns4d_axis_size(config)
    if axis_size <= 1:
        return group_size
    return math.ceil(group_size / axis_size) * axis_size


def ns4d_compute_sharding(mesh: Mesh, config: BenchConfig, bench_kind: str) -> NamedSharding:
    group_size = (
        padded_ns4d_group_size(config, bench_kind)
        if bench_kind
        in (
            EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
            EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        )
        else ns4d_group_size(config)
    )
    return ns4d_compute_sharding_for_group_size(mesh, config, bench_kind, group_size)


def ns4d_compute_sharding_for_group_size(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    group_size: int,
) -> NamedSharding:
    if bench_kind == NS4D_REPLICATED_GROUP_BENCH:
        return NamedSharding(mesh, P(None, "expert", None, None))
    if bench_kind in (*NS4D_DATA_SHARDED_BENCHES, *NS4D_PADDED_BENCHES):
        requested_group_axis = ns4d_group_axis(config)
        if isinstance(requested_group_axis, tuple):
            live_axes = tuple(axis for axis in requested_group_axis if mesh_axis_size(mesh, axis) > 1)
            group_axis = live_axes if len(live_axes) > 1 else live_axes[0] if live_axes else None
        else:
            group_axis = (
                requested_group_axis
                if requested_group_axis is not None and mesh_axis_size(mesh, requested_group_axis) > 1
                else None
            )
        if group_axis is not None:
            axis_names = group_axis if isinstance(group_axis, tuple) else (group_axis,)
            axis_size = math.prod(mesh_axis_size(mesh, axis_name) for axis_name in axis_names)
            if group_size % axis_size != 0:
                group_axis = None
        return NamedSharding(mesh, P(group_axis, "expert", None, None))
    raise ValueError(f"Unsupported NS4D bench kind: {bench_kind!r}")


def ns4d_input_sharding(mesh: Mesh, config: BenchConfig, bench_kind: str) -> NamedSharding:
    if bench_kind in NS4D_PADDED_BENCHES:
        return NamedSharding(mesh, P(None, "expert", None, None))
    return ns4d_compute_sharding(mesh, config, bench_kind)


def ns4d_input_sharding_for_group_size(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    group_size: int,
) -> NamedSharding:
    if bench_kind in NS4D_PADDED_BENCHES:
        return NamedSharding(mesh, P(None, "expert", None, None))
    return ns4d_compute_sharding_for_group_size(mesh, config, bench_kind, group_size)


def ns4d_result_sharding(mesh: Mesh, config: BenchConfig, bench_kind: str) -> NamedSharding | None:
    if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
        return None
    if bench_kind == NS4D_DATA_RESHARD_RESTORE_BENCH:
        return NamedSharding(mesh, P("expert", None, None))
    if bench_kind in (
        EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ):
        return None
    if bench_kind == NS4D_DATA_INDEX_RESTORE_BENCH:
        return None
    if bench_kind in NS4D_PADDED_BENCHES:
        return NamedSharding(mesh, P(None, "expert", None, None))
    return ns4d_compute_sharding(mesh, config, bench_kind)


def grouped_expert_group_sharding(mesh: Mesh, config: BenchConfig, bench_kind: str, group_size: int) -> NamedSharding:
    return ns4d_input_sharding_for_group_size(mesh, config, bench_kind, group_size)


def sharding_spec_string(sharding: NamedSharding) -> str:
    return str(sharding.spec)


def synthetic_specs(mesh: Mesh, config: BenchConfig) -> Any:
    dtype = dtype_from_name(config.dtype)
    sharding = expert_param_sharding(mesh)
    shapes = synthetic_shapes(config)
    return {
        "layers": tuple(
            {
                "mlp": {
                    "expert_mlp": {
                        name: jax.ShapeDtypeStruct(shape, dtype, sharding=sharding) for name, shape in shapes.items()
                    }
                }
            }
            for _ in range(config.layers)
        )
    }


def synthetic_fsdp_expert_specs(mesh: Mesh, config: BenchConfig) -> Any:
    dtype = dtype_from_name(config.dtype)
    shapes = synthetic_shapes(config)
    return {
        "layers": tuple(
            {
                "mlp": {
                    "expert_mlp": {
                        name: jax.ShapeDtypeStruct(
                            shape,
                            dtype,
                            sharding=expert_fsdp_param_sharding(mesh, name),
                        )
                        for name, shape in shapes.items()
                    }
                }
            }
            for _ in range(config.layers)
        )
    }


def synthetic_ns4d_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    sharding = ns4d_input_sharding(mesh, config, bench_kind)
    return {
        name: jax.ShapeDtypeStruct((ns4d_group_size(config), *shape), dtype, sharding=sharding)
        for name, shape in synthetic_shapes(config).items()
    }


def grouped_expert_group_count(config: BenchConfig) -> int:
    return len(grouped_expert_group_sizes(config))


def grouped_expert_group_sizes(config: BenchConfig) -> tuple[int, ...]:
    group_size = ns4d_group_size(config)
    full_groups, remainder = divmod(config.layers, group_size)
    sizes = [group_size] * full_groups
    if remainder:
        sizes.append(remainder)
    return tuple(sizes)


def grouped_expert_group_sizes_for_bench(config: BenchConfig, bench_kind: str) -> tuple[int, ...]:
    sizes = grouped_expert_group_sizes(config)
    if bench_kind not in (
        EXPERT_GROUPED_LAYER_SLICE_BENCH,
        EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
        EXPERT_GROUPED_BANK_CONSUMER_BENCH,
        EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ):
        return sizes
    axis_size = ns4d_axis_size(config)
    if axis_size <= 1:
        return sizes
    return tuple(math.ceil(size / axis_size) * axis_size for size in sizes)


def synthetic_grouped_expert_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    shapes = synthetic_shapes(config)
    return {
        "blocks": tuple(
            {
                "mlp": {
                    "expert_mlp": {
                        name: jax.ShapeDtypeStruct(
                            (group_size, *shape),
                            dtype,
                            sharding=grouped_expert_group_sharding(mesh, config, bench_kind, group_size),
                        )
                        for name, shape in shapes.items()
                    }
                }
            }
            for group_size in grouped_expert_group_sizes_for_bench(config, bench_kind)
        )
    }


def synthetic_grouped_expert_consumer_input_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    return {
        "blocks": tuple(
            {
                "x": jax.ShapeDtypeStruct(
                    (
                        group_size,
                        config.num_experts,
                        config.grouped_expert_consumer_tokens_per_expert,
                        config.hidden_dim,
                    ),
                    dtype,
                    sharding=grouped_expert_group_sharding(mesh, config, bench_kind, group_size),
                )
            }
            for group_size in grouped_expert_group_sizes_for_bench(config, bench_kind)
        )
    }


def grouped_moe_consumer_sharding(mesh: Mesh, config: BenchConfig, bench_kind: str, group_size: int) -> NamedSharding:
    group_axis = grouped_expert_group_sharding(mesh, config, bench_kind, group_size).spec[0]
    return NamedSharding(mesh, P(group_axis, None, None))


def synthetic_grouped_moe_mlp_consumer_input_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    tokens = config.num_experts * config.grouped_expert_consumer_tokens_per_expert
    top_k = 1
    return {
        "blocks": tuple(
            {
                "x": jax.ShapeDtypeStruct(
                    (group_size, tokens, config.hidden_dim),
                    dtype,
                    sharding=grouped_moe_consumer_sharding(mesh, config, bench_kind, group_size),
                ),
                "selected_experts": jax.ShapeDtypeStruct(
                    (group_size, tokens, top_k),
                    jnp.dtype(jnp.int32),
                    sharding=grouped_moe_consumer_sharding(mesh, config, bench_kind, group_size),
                ),
                "combine_weights": jax.ShapeDtypeStruct(
                    (group_size, tokens, top_k),
                    dtype,
                    sharding=grouped_moe_consumer_sharding(mesh, config, bench_kind, group_size),
                ),
            }
            for group_size in grouped_expert_group_sizes_for_bench(config, bench_kind)
        )
    }


def synthetic_productionish_grouped_expert_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    matrix_sharding = NamedSharding(mesh, P(None, None))
    vector_sharding = NamedSharding(mesh, P(None))
    tree = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    tree["ordinary_blocks"] = tuple(
        {
            "mlp": {
                "w_in": jax.ShapeDtypeStruct((config.hidden_dim, config.hidden_dim), dtype, sharding=matrix_sharding)
            },
            "router": {"bias": jax.ShapeDtypeStruct((config.num_experts,), dtype, sharding=vector_sharding)},
        }
        for _ in range(config.layers)
    )
    tree["token_embed"] = jax.ShapeDtypeStruct((config.hidden_dim, config.hidden_dim), dtype, sharding=matrix_sharding)
    return tree


def production_2d_sharding(mesh: Mesh, path: str) -> NamedSharding:
    data_axis = "data" if mesh_axis_size(mesh, "data") > 1 else None
    model_axis = "model" if mesh_axis_size(mesh, "model") > 1 else None
    if path in ("attn.w_o", "shared.w_down"):
        return NamedSharding(mesh, P(model_axis, data_axis))
    if path.endswith("_gated_norm.w_down") or path.endswith("_gated_norm.w_up"):
        return NamedSharding(mesh, P(None, None))
    return NamedSharding(mesh, P(data_axis, model_axis))


def synthetic_full_production_muonh_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    dtype = dtype_from_name(config.dtype)
    matrix_sharding = NamedSharding(mesh, P(None, None))
    vector_sharding = NamedSharding(mesh, P(None))
    tree = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    muon_2d_shapes = production_muonh_2d_shapes(config)
    tree["ordinary_blocks"] = tuple(
        {
            "attn": (
                {
                    name.removeprefix("attn."): jax.ShapeDtypeStruct(
                        shape,
                        dtype,
                        sharding=production_2d_sharding(mesh, name),
                    )
                    for name, shape in muon_2d_shapes.items()
                    if name.startswith("attn.")
                }
                | {
                    "attn_gate": jax.ShapeDtypeStruct(
                        (config.hidden_dim, max(1, config.hidden_dim // MAY_HIDDEN_HEAD_RATIO)),
                        dtype,
                        sharding=matrix_sharding,
                    )
                }
            ),
            "attn_gated_norm": {
                name.removeprefix("attn_gated_norm."): jax.ShapeDtypeStruct(
                    shape,
                    dtype,
                    sharding=production_2d_sharding(mesh, name),
                )
                for name, shape in muon_2d_shapes.items()
                if name.startswith("attn_gated_norm.")
            },
            "mlp_gated_norm": {
                name.removeprefix("mlp_gated_norm."): jax.ShapeDtypeStruct(
                    shape,
                    dtype,
                    sharding=production_2d_sharding(mesh, name),
                )
                for name, shape in muon_2d_shapes.items()
                if name.startswith("mlp_gated_norm.")
            },
            "mlp": {
                "router": jax.ShapeDtypeStruct(
                    (config.hidden_dim, config.num_experts),
                    dtype,
                    sharding=matrix_sharding,
                ),
                "router_bias": jax.ShapeDtypeStruct((config.num_experts,), dtype, sharding=vector_sharding),
            },
            "shared": {
                name.removeprefix("shared."): jax.ShapeDtypeStruct(
                    shape,
                    dtype,
                    sharding=production_2d_sharding(mesh, name),
                )
                for name, shape in muon_2d_shapes.items()
                if name.startswith("shared.")
            },
        }
        for _ in range(config.layers)
    )
    tree["token_embed"] = jax.ShapeDtypeStruct((config.hidden_dim, config.hidden_dim), dtype, sharding=matrix_sharding)
    tree["output_proj"] = jax.ShapeDtypeStruct((config.hidden_dim, config.hidden_dim), dtype, sharding=matrix_sharding)
    return tree


def synthetic_ordinary_2d_muonh_specs(mesh: Mesh, config: BenchConfig) -> Any:
    dtype = dtype_from_name(config.dtype)
    muon_2d_shapes = production_muonh_2d_shapes(config)
    return {
        "ordinary_blocks": tuple(
            {
                "attn": {
                    name.removeprefix("attn."): jax.ShapeDtypeStruct(
                        shape,
                        dtype,
                        sharding=production_2d_sharding(mesh, name),
                    )
                    for name, shape in muon_2d_shapes.items()
                    if name.startswith("attn.")
                },
                "attn_gated_norm": {
                    name.removeprefix("attn_gated_norm."): jax.ShapeDtypeStruct(
                        shape,
                        dtype,
                        sharding=production_2d_sharding(mesh, name),
                    )
                    for name, shape in muon_2d_shapes.items()
                    if name.startswith("attn_gated_norm.")
                },
                "mlp_gated_norm": {
                    name.removeprefix("mlp_gated_norm."): jax.ShapeDtypeStruct(
                        shape,
                        dtype,
                        sharding=production_2d_sharding(mesh, name),
                    )
                    for name, shape in muon_2d_shapes.items()
                    if name.startswith("mlp_gated_norm.")
                },
                "shared": {
                    name.removeprefix("shared."): jax.ShapeDtypeStruct(
                        shape,
                        dtype,
                        sharding=production_2d_sharding(mesh, name),
                    )
                    for name, shape in muon_2d_shapes.items()
                    if name.startswith("shared.")
                },
            }
            for _ in range(config.layers)
        )
    }


def _grouped_2d_persistent_specs_from_specs(
    specs: Any,
    *,
    full_production_tree: bool,
    max_grouped_stack_size: int,
) -> tuple[Any, ...]:
    groups = _persistent_grouped_2d_entries_from_specs(
        specs,
        full_production_tree=full_production_tree,
        max_grouped_stack_size=max_grouped_stack_size,
    )
    return tuple(grouped_spec for _, grouped_spec in groups)


def _persistent_grouped_2d_entries_from_specs(
    specs: Any,
    *,
    full_production_tree: bool,
    max_grouped_stack_size: int,
) -> tuple[tuple[PersistentGrouped2DMetadata, jax.ShapeDtypeStruct], ...]:
    if full_production_tree:
        mask = full_production_muonh_mask(specs)
    else:
        mask = jax.tree.map(lambda leaf: "muonh" if hasattr(leaf, "ndim") and leaf.ndim == 2 else "ordinary", specs)
    mask_by_path = dict(leaf_items(mask))

    groups: dict[tuple[tuple[int, int], str, str], list[tuple[str, Any]]] = {}
    for path, leaf in leaf_items(specs):
        if mask_by_path[path] != "muonh" or not hasattr(leaf, "ndim") or leaf.ndim != 2:
            continue
        param_spec = getattr(getattr(leaf, "sharding", None), "spec", None)
        key = (tuple(leaf.shape), str(leaf.dtype), str(param_spec))
        groups.setdefault(key, []).append((path, leaf))

    grouped_entries = []
    for entries in groups.values():
        for chunk_start in range(0, len(entries), max_grouped_stack_size):
            entry_chunk = entries[chunk_start : chunk_start + max_grouped_stack_size]
            sample = entry_chunk[0][1]
            source_paths = tuple(path for path, _ in entry_chunk)
            valid_stack_length = len(entry_chunk)
            shape = (valid_stack_length, *sample.shape)
            sharding = _stacked_2d_target(shape, sample)
            padded_stack_length = valid_stack_length
            if isinstance(sharding, NamedSharding):
                stack_axis = sharding.spec[0]
                if stack_axis is not None:
                    axis_names = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
                    axis_size = math.prod(mesh_axis_size(sharding.mesh, axis_name) for axis_name in axis_names)
                    padded_stack_length = math.ceil(valid_stack_length / axis_size) * axis_size
                    shape = (padded_stack_length, *shape[1:])
            if sharding is None:
                sample_sharding = getattr(sample, "sharding", None)
                if isinstance(sample_sharding, NamedSharding):
                    sharding = NamedSharding(sample_sharding.mesh, P(None, None, None))
            source_sharding_spec = str(getattr(getattr(sample, "sharding", None), "spec", None))
            target_sharding_spec = str(getattr(sharding, "spec", None))
            metadata = PersistentGrouped2DMetadata(
                source_paths=source_paths,
                leaf_shape=tuple(sample.shape),
                source_sharding_spec=source_sharding_spec,
                target_sharding_spec=target_sharding_spec,
                valid_stack_length=valid_stack_length,
                padded_stack_length=padded_stack_length,
            )
            grouped_entries.append((metadata, jax.ShapeDtypeStruct(shape, sample.dtype, sharding=sharding)))
    return tuple(grouped_entries)


def persistent_grouped_2d_metadata_from_specs(
    specs: Any,
    *,
    full_production_tree: bool,
    max_grouped_stack_size: int,
) -> tuple[PersistentGrouped2DMetadata, ...]:
    return tuple(
        metadata
        for metadata, _ in _persistent_grouped_2d_entries_from_specs(
            specs,
            full_production_tree=full_production_tree,
            max_grouped_stack_size=max_grouped_stack_size,
        )
    )


def synthetic_ordinary_2d_grouped_persistent_specs(mesh: Mesh, config: BenchConfig) -> Any:
    specs = synthetic_ordinary_2d_muonh_specs(mesh, config)
    return {
        "ordinary_2d_groups": _grouped_2d_persistent_specs_from_specs(
            specs,
            full_production_tree=False,
            max_grouped_stack_size=config.max_grouped_stack_size,
        )
    }


def synthetic_full_production_grouped_persistent_specs(mesh: Mesh, config: BenchConfig, bench_kind: str) -> Any:
    full_specs = synthetic_full_production_muonh_specs(mesh, config, bench_kind)
    return {
        "blocks": full_specs["blocks"],
        "ordinary_2d_groups": _grouped_2d_persistent_specs_from_specs(
            full_specs,
            full_production_tree=True,
            max_grouped_stack_size=config.max_grouped_stack_size,
        ),
    }


def make_array_tree(config: BenchConfig, shardings: Any, seed: int) -> Any:
    dtype = numpy_dtype_from_name(config.dtype)
    shapes = synthetic_shapes(config)

    def make_leaf(layer_index: int, name: str, shape: tuple[int, int, int]) -> jax.Array:
        fill_value = np.asarray(((seed + 1) * (layer_index + 1) * (len(name) + 1)) % 17 + 1, dtype=np.float32)
        fill_value = fill_value / np.asarray(1000, dtype=np.float32)
        sharding = shardings["layers"][layer_index]["mlp"]["expert_mlp"][name]

        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(shape, sharding, callback)

    return {
        "layers": tuple(
            {"mlp": {"expert_mlp": {name: make_leaf(layer_index, name, shape) for name, shape in shapes.items()}}}
            for layer_index in range(config.layers)
        )
    }


def make_ns4d_array_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    dtype = numpy_dtype_from_name(config.dtype)
    sharding = ns4d_input_sharding(mesh, config, bench_kind)

    def make_leaf(name: str, shape: tuple[int, int, int]) -> jax.Array:
        full_shape = (ns4d_group_size(config), *shape)
        fill_value = np.asarray(((seed + 1) * (len(name) + 1)) % 17 + 1, dtype=np.float32)
        fill_value = fill_value / np.asarray(1000, dtype=np.float32)

        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = full_shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(full_shape, sharding, callback)

    return {name: make_leaf(name, shape) for name, shape in synthetic_shapes(config).items()}


def make_grouped_expert_array_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    dtype = numpy_dtype_from_name(config.dtype)
    shapes = synthetic_shapes(config)

    def make_leaf(group_index: int, group_size: int, name: str, shape: tuple[int, int, int]) -> jax.Array:
        full_shape = (group_size, *shape)
        sharding = grouped_expert_group_sharding(mesh, config, bench_kind, group_size)
        fill_value = np.asarray(((seed + 1) * (group_index + 1) * (len(name) + 1)) % 17 + 1, dtype=np.float32)
        fill_value = fill_value / np.asarray(1000, dtype=np.float32)

        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = full_shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(full_shape, sharding, callback)

    return {
        "blocks": tuple(
            {
                "mlp": {
                    "expert_mlp": {
                        name: make_leaf(group_index, group_size, name, shape) for name, shape in shapes.items()
                    }
                }
            }
            for group_index, group_size in enumerate(grouped_expert_group_sizes_for_bench(config, bench_kind))
        )
    }


def make_grouped_expert_consumer_input_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    specs = synthetic_grouped_expert_consumer_input_specs(mesh, config, bench_kind)
    return make_array_from_spec_tree(specs, config, seed)


def make_grouped_moe_mlp_consumer_input_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    specs = synthetic_grouped_moe_mlp_consumer_input_specs(mesh, config, bench_kind)
    dtype = numpy_dtype_from_name(config.dtype)

    def make_float_array(spec: jax.ShapeDtypeStruct, fill_value: np.ndarray) -> jax.Array:
        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = spec.shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(spec.shape, spec.sharding, callback)

    def make_selected_experts(spec: jax.ShapeDtypeStruct) -> jax.Array:
        def callback(index: tuple[slice, ...]) -> np.ndarray:
            slices = []
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = spec.shape[axis] if axis_index.stop is None else axis_index.stop
                slices.append((start, stop))
                local_shape.append(stop - start)
            token_ids = np.arange(slices[1][0], slices[1][1], dtype=np.int32) % config.num_experts
            values = token_ids[None, :, None]
            return np.broadcast_to(values, tuple(local_shape)).astype(np.int32)

        return jax.make_array_from_callback(spec.shape, spec.sharding, callback)

    blocks = []
    for group_index, block_specs in enumerate(specs["blocks"]):
        x_fill = np.asarray(((seed + 1) * (group_index + 1)) % 31 + 1, dtype=np.float32) / 1000
        blocks.append(
            {
                "x": make_float_array(block_specs["x"], x_fill),
                "selected_experts": make_selected_experts(block_specs["selected_experts"]),
                "combine_weights": make_float_array(
                    block_specs["combine_weights"],
                    np.asarray(1.0, dtype=np.float32),
                ),
            }
        )
    return {"blocks": tuple(blocks)}


def make_productionish_grouped_expert_array_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    dtype = numpy_dtype_from_name(config.dtype)
    matrix_sharding = NamedSharding(mesh, P(None, None))
    vector_sharding = NamedSharding(mesh, P(None))
    tree = make_grouped_expert_array_tree(mesh, config, bench_kind, seed)

    def make_array(shape: tuple[int, ...], sharding: NamedSharding, fill_value: np.ndarray) -> jax.Array:
        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(shape, sharding, callback)

    tree["ordinary_blocks"] = tuple(
        {
            "mlp": {
                "w_in": make_array(
                    (config.hidden_dim, config.hidden_dim),
                    matrix_sharding,
                    np.asarray(((seed + 3) * (layer_index + 1)) % 19 + 1, dtype=np.float32) / 1000,
                )
            },
            "router": {
                "bias": make_array(
                    (config.num_experts,),
                    vector_sharding,
                    np.asarray(((seed + 5) * (layer_index + 1)) % 23 + 1, dtype=np.float32) / 1000,
                )
            },
        }
        for layer_index in range(config.layers)
    )
    tree["token_embed"] = make_array(
        (config.hidden_dim, config.hidden_dim),
        matrix_sharding,
        np.asarray((seed + 7) % 29 + 1, dtype=np.float32) / 1000,
    )
    return tree


def make_array_from_spec_tree(specs: Any, config: BenchConfig, seed: int) -> Any:
    def make_leaf(path: str, spec: jax.ShapeDtypeStruct) -> jax.Array:
        dtype = spec.dtype
        if np.issubdtype(np.dtype(dtype), np.integer):
            fill_value = np.asarray(((seed + 1) * (len(path) + 1)) % max(config.num_experts, 1), dtype=dtype)
        else:
            fill_value = np.asarray(((seed + 1) * (len(path) + 1)) % 31 + 1, dtype=np.float32) / 1000

        def callback(index: tuple[slice, ...]) -> np.ndarray:
            local_shape = []
            for axis, axis_index in enumerate(index):
                start = 0 if axis_index.start is None else axis_index.start
                stop = spec.shape[axis] if axis_index.stop is None else axis_index.stop
                local_shape.append(stop - start)
            return np.full(local_shape, fill_value, dtype=dtype)

        return jax.make_array_from_callback(spec.shape, spec.sharding, callback)

    def visit(prefix: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: visit(f"{prefix}.{key}" if prefix else str(key), child) for key, child in value.items()}
        if isinstance(value, tuple):
            return tuple(
                visit(f"{prefix}.{index}" if prefix else str(index), child) for index, child in enumerate(value)
            )
        return make_leaf(prefix, value)

    return visit("", specs)


def make_full_production_muonh_array_tree(mesh: Mesh, config: BenchConfig, bench_kind: str, seed: int) -> Any:
    specs = synthetic_full_production_muonh_specs(mesh, config, bench_kind)
    return make_array_from_spec_tree(specs, config, seed)


def make_full_production_grouped_persistent_array_tree(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    seed: int,
) -> Any:
    specs = synthetic_full_production_grouped_persistent_specs(mesh, config, bench_kind)
    return make_array_from_spec_tree(specs, config, seed)


def make_ordinary_2d_muonh_array_tree(mesh: Mesh, config: BenchConfig, seed: int) -> Any:
    specs = synthetic_ordinary_2d_muonh_specs(mesh, config)
    return make_array_from_spec_tree(specs, config, seed)


def make_ordinary_2d_grouped_persistent_array_tree(mesh: Mesh, config: BenchConfig, seed: int) -> Any:
    specs = synthetic_ordinary_2d_grouped_persistent_specs(mesh, config)
    return make_array_from_spec_tree(specs, config, seed)


def ns4d_group_size(config: BenchConfig) -> int:
    return config.layers if config.ns4d_group_size is None else config.ns4d_group_size


def build_muonh_transform(config: BenchConfig):
    return scale_with_grug_muonh(
        momentum=MAY_MOMENTUM,
        nesterov=True,
        steps=config.backend_steps,
        muon_eps=MAY_MUON_EPSILON,
        learning_rate=config.learning_rate,
        momentum_sharding_fn=_expert_momentum_sharding,
        orthogonalization_layout=config.orthogonalization_layout,
        max_grouped_stack_size=config.max_grouped_stack_size,
        ns_compute_dtype=config.ns_compute_dtype,
    )


def build_direction_transform(config: BenchConfig):
    return _grug_scale_with_muon(
        momentum=MAY_MOMENTUM,
        nesterov=True,
        steps=config.backend_steps,
        muon_eps=MAY_MUON_EPSILON,
        use_kimi_scaling=False,
        momentum_sharding_fn=_expert_momentum_sharding,
        orthogonalization_layout=config.orthogonalization_layout,
        max_grouped_stack_size=config.max_grouped_stack_size,
        ns_compute_dtype=config.ns_compute_dtype,
    )


def assert_expert_stack_sharding(tree: Any, expected_spec: P, label: str) -> None:
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 3:
            continue
        sharding = getattr(leaf, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} lost NamedSharding: got {sharding!r}.")
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")
        stack_axis = sharding.spec[0]
        stack_axes = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
        if "expert" not in stack_axes:
            raise AssertionError(f"{label} at {path} lost expert-axis stack sharding: got {sharding.spec}.")


def expected_expert_fsdp_spec(path: str) -> P:
    if path == "w_gate_up" or path.endswith(".w_gate_up"):
        return P("expert", "data", "model")
    if path == "w_down" or path.endswith(".w_down"):
        return P("expert", "model", "data")
    raise AssertionError(f"Unknown expert FSDP leaf path {path!r}.")


def assert_expert_fsdp_sharding(tree: Any, label: str) -> None:
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 3:
            continue
        sharding = getattr(leaf, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} lost NamedSharding: got {sharding!r}.")
        expected_spec = expected_expert_fsdp_spec(path)
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")
        stack_axis = sharding.spec[0]
        stack_axes = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
        if "expert" not in stack_axes:
            raise AssertionError(f"{label} at {path} lost expert-axis stack sharding: got {sharding.spec}.")


def assert_grouped_expert_target_fsdp_sharding(tree: Any, label: str) -> None:
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 4:
            continue
        sharding = getattr(leaf, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} lost NamedSharding: got {sharding!r}.")
        name = path.rsplit(".", 1)[-1]
        expected_spec = P(None, *expected_expert_fsdp_spec(name))
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")
        stack_axis = sharding.spec[1]
        stack_axes = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
        if "expert" not in stack_axes:
            raise AssertionError(f"{label} at {path} lost expert-axis stack sharding: got {sharding.spec}.")


def assert_expert_ep_sharding(tree: Any, label: str) -> None:
    expected_spec = P("expert", None, None)
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 3:
            continue
        sharding = getattr(leaf, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} lost NamedSharding: got {sharding!r}.")
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")


def assert_expert_stack_sharding_if_present(tree: Any, expected_spec: P, label: str) -> None:
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 3:
            continue
        sharding = getattr(leaf, "sharding", None)
        if sharding is None:
            continue
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} has unexpected sharding object: got {sharding!r}.")
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")
        stack_axis = sharding.spec[0]
        stack_axes = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
        if "expert" not in stack_axes:
            raise AssertionError(f"{label} at {path} lost expert-axis stack sharding: got {sharding.spec}.")


def assert_runtime_expert_stack_sharding(tree: Any, expected_spec: P, label: str, config: BenchConfig) -> None:
    if config.expert_axis <= 1:
        return
    assert_expert_stack_sharding(tree, expected_spec, label)


def assert_runtime_expert_fsdp_sharding(tree: Any, label: str, config: BenchConfig) -> None:
    if config.expert_axis <= 1:
        return
    assert_expert_fsdp_sharding(tree, label)


def assert_ns4d_sharding(tree: Any, expected_spec: P, label: str) -> None:
    for path, leaf in leaf_items(tree):
        if not hasattr(leaf, "ndim") or leaf.ndim != 4:
            continue
        sharding = getattr(leaf, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise AssertionError(f"{label} at {path} lost NamedSharding: got {sharding!r}.")
        if sharding.spec != expected_spec:
            raise AssertionError(f"{label} at {path} expected {expected_spec}, got {sharding.spec}.")


def assert_grouped_expert_sharding(tree: Any, mesh: Mesh, config: BenchConfig, bench_kind: str, label: str) -> None:
    if not isinstance(tree, dict) or "blocks" not in tree:
        raise AssertionError(f"{label} expected a grouped expert tree with a blocks field.")
    for group_index, (block, group_size) in enumerate(
        zip(tree["blocks"], grouped_expert_group_sizes_for_bench(config, bench_kind), strict=True)
    ):
        expected_spec = grouped_expert_group_sharding(mesh, config, bench_kind, group_size).spec
        assert_ns4d_sharding(block, expected_spec, f"{label} block {group_index}")


def assert_grouped_moe_consumer_sharding(
    tree: Any, mesh: Mesh, config: BenchConfig, bench_kind: str, label: str
) -> None:
    if not isinstance(tree, dict) or "blocks" not in tree:
        raise AssertionError(f"{label} expected a grouped MoE consumer tree with a blocks field.")
    for group_index, (block, group_size) in enumerate(
        zip(tree["blocks"], grouped_expert_group_sizes_for_bench(config, bench_kind), strict=True)
    ):
        expected_spec = grouped_moe_consumer_sharding(mesh, config, bench_kind, group_size).spec
        assert_ns4d_sharding(block, expected_spec, f"{label} block {group_index}")


def assert_grouped_or_uniform_ns4d_sharding(
    tree: Any,
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    expected_spec: P,
    label: str,
) -> None:
    if bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES:
        assert_grouped_expert_sharding(tree, mesh, config, bench_kind, label)
        return
    assert_ns4d_sharding(tree, expected_spec, label)


def momentum_expected_spec(config: BenchConfig) -> P:
    if config.replica_axis > 1 and config.num_experts % (config.replica_axis * config.expert_axis) == 0:
        return P(("replica_dcn", "expert"), None, None)
    return P("expert", None, None)


def leaf_items(tree: Any) -> list[tuple[str, Any]]:
    items = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child_prefix, child)
            return
        if isinstance(value, tuple):
            for index, child in enumerate(value):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                visit(child_prefix, child)
            return
        items.append((prefix, value))

    visit("", tree)
    return items


def update_step_factory(config: BenchConfig):
    transform = build_muonh_transform(config)

    def update_step(params, updates, state):
        with jax.named_scope("muon_update_bench/update_step"):
            next_updates, next_state = transform.update(updates, state, params)
            return next_updates, next_state

    return update_step


def direction_step_factory(config: BenchConfig):
    transform = build_direction_transform(config)

    def update_step(params, updates, state):
        with jax.named_scope("muon_update_bench/direction_update_step"):
            next_updates, next_state = transform.update(updates, state, params)
            return next_updates, next_state

    return update_step


def hyperball_step_factory(config: BenchConfig):
    def update_step(params, direction_updates):
        with jax.named_scope("muon_update_bench/hyperball_only_step"):
            hyperball_updates = _scale_invariant_hyperball_updates(
                params,
                direction_updates,
                config.learning_rate,
                label="Muon update benchmark hyperball-only",
            )
            return hyperball_updates

    return update_step


def ns4d_step_factory(mesh: Mesh, config: BenchConfig, bench_kind: str):
    def update_step(updates):
        with jax.named_scope("muon_update_bench/ns4d_update_step"):
            if bench_kind in NS4D_PADDED_BENCHES:
                compute_sharding = ns4d_compute_sharding(mesh, config, bench_kind)
                next_updates = jax.tree.map(
                    lambda update: padded_ns4d_update(update, config, bench_kind, compute_sharding),
                    updates,
                )
            elif bench_kind == NS4D_DOTONLY_EINSUM_BENCH:
                next_updates = jax.tree.map(
                    lambda update: newtonschulz_4d_dotonly_for_config(update, config, "einsum"),
                    updates,
                )
            elif bench_kind == NS4D_DOTONLY_MATMUL_BENCH:
                next_updates = jax.tree.map(
                    lambda update: newtonschulz_4d_dotonly_for_config(update, config, "matmul"),
                    updates,
                )
            elif bench_kind == NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH:
                next_updates = jax.tree.map(
                    lambda update: newtonschulz_4d_dotonly_for_config(update, config, "lax_dot_general"),
                    updates,
                )
            else:
                next_updates = jax.tree.map(
                    lambda update: zeropower_via_newtonschulz_4d_for_config(update, config),
                    updates,
                )
            if bench_kind == NS4D_DATA_RESHARD_RESTORE_BENCH:
                next_updates = restore_ns4d_group_with_reshard(mesh, next_updates)
            elif bench_kind == NS4D_DATA_INDEX_RESTORE_BENCH:
                next_updates = restore_ns4d_group_with_indexing(next_updates)
            return next_updates

    return update_step


def ns4d_grouped_apply_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/ns4d_grouped_apply_step"):
            next_updates = jax.tree.map(
                lambda update: zeropower_via_newtonschulz_4d_for_config(update, config),
                updates,
            )
            next_updates = jax.tree.map(lambda update: scale_ns4d_update(update, config), next_updates)
            return jax.tree.map(lambda param, update: param + update, params, next_updates)

    return update_step


def grouped_expert_apply_boundary_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/expert_grouped_apply_boundary_step"):
            next_updates = jax.tree.map(
                lambda update: zeropower_via_newtonschulz_4d_for_config(update, config),
                updates,
            )
            next_updates = jax.tree.map(lambda update: scale_ns4d_update(update, config), next_updates)
            with jax.named_scope("muon_update_bench/expert_grouped_apply_boundary/optax_apply_updates"):
                return optax.apply_updates(params, next_updates)

    return update_step


def expert_grouped_layer_slices(mesh: Mesh, config: BenchConfig, grouped_params):
    """Slice persistent grouped expert banks into per-layer EP-consumable leaves."""
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    ep_sharding = NamedSharding(mesh, P("expert", None, None))
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        padded_group_size = grouped_expert_group_sizes_for_bench(config, EXPERT_GROUPED_LAYER_SLICE_BENCH)[group_index]
        compute_sharding = grouped_expert_group_sharding(
            mesh,
            config,
            EXPERT_GROUPED_LAYER_SLICE_BENCH,
            padded_group_size,
        )
        for name in synthetic_shapes(config):
            grouped_param = grouped_params["blocks"][group_index]["mlp"]["expert_mlp"][name]
            grouped_param = _restore_grouped_expert_update_for_split(
                mesh,
                grouped_param,
                valid_group_size,
                compute_sharding,
            )
            with jax.named_scope(f"muon_update_bench/expert_grouped_layer_slice/{name}/split_to_ep_leaves"):
                param_parts = [
                    jnp.squeeze(param_part, axis=0) for param_part in jnp.split(grouped_param, valid_group_size, axis=0)
                ]
            for local_index, param_part in enumerate(param_parts):
                layer_index = layer_offset + local_index
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = reshard(param_part, ep_sharding)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_grouped_single_layer_slice(mesh: Mesh, config: BenchConfig, grouped_params, layer_index: int = 0):
    """Slice one layer's expert weights out of persistent grouped expert banks."""
    if layer_index < 0 or layer_index >= config.layers:
        raise ValueError(f"layer_index={layer_index} must be in [0, {config.layers}).")

    ep_sharding = NamedSharding(mesh, P("expert", None, None))
    layer_offset = 0
    group_index = 0
    local_layer_index = layer_index
    for candidate_group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        if layer_index < layer_offset + valid_group_size:
            group_index = candidate_group_index
            local_layer_index = layer_index - layer_offset
            break
        layer_offset += valid_group_size
    else:
        raise AssertionError("layer_index bounds check should have selected a grouped expert block.")

    padded_group_size = grouped_expert_group_sizes_for_bench(config, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH)[
        group_index
    ]
    compute_sharding = grouped_expert_group_sharding(
        mesh,
        config,
        EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
        padded_group_size,
    )
    expert_mlp = {}
    for name in synthetic_shapes(config):
        grouped_param = grouped_params["blocks"][group_index]["mlp"]["expert_mlp"][name]
        grouped_param = _restore_grouped_expert_update_for_split(
            mesh,
            grouped_param,
            grouped_expert_group_sizes(config)[group_index],
            compute_sharding,
        )
        with jax.named_scope(f"muon_update_bench/expert_grouped_single_layer_slice/{name}/index_ep_leaf"):
            expert_mlp[name] = reshard(grouped_param[local_layer_index], ep_sharding)
    return {"mlp": {"expert_mlp": expert_mlp}}


def expert_grouped_layer_slice_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(grouped_params):
        with jax.named_scope("muon_update_bench/expert_grouped_layer_slice_step"):
            return expert_grouped_layer_slices(mesh, config, grouped_params)

    return update_step


def expert_grouped_single_layer_slice_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(grouped_params):
        with jax.named_scope("muon_update_bench/expert_grouped_single_layer_slice_step"):
            return expert_grouped_single_layer_slice(mesh, config, grouped_params)

    return update_step


def grouped_expert_bank_consumer_outputs(config: BenchConfig, grouped_params, activations):
    """Run a grouped expert-bank MLP consumer without materializing per-layer expert leaves."""
    output_blocks = []
    for group_index, _valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        expert_mlp = grouped_params["blocks"][group_index]["mlp"]["expert_mlp"]
        x = activations["blocks"][group_index]["x"]
        with jax.named_scope("muon_update_bench/expert_grouped_bank_consumer/gate_up"):
            gate_up = jnp.einsum("getd,gedi->geti", x, expert_mlp["w_gate_up"])
        with jax.named_scope("muon_update_bench/expert_grouped_bank_consumer/activation"):
            gate, up = jnp.split(gate_up, 2, axis=-1)
            hidden = jax.nn.silu(gate) * up
        with jax.named_scope("muon_update_bench/expert_grouped_bank_consumer/down"):
            output_blocks.append({"x": jnp.einsum("geti,geid->getd", hidden, expert_mlp["w_down"])})
    return {"blocks": tuple(output_blocks)}


def grouped_expert_bank_consumer_step_factory(config: BenchConfig):
    def update_step(grouped_params, activations):
        with jax.named_scope("muon_update_bench/expert_grouped_bank_consumer_step"):
            return grouped_expert_bank_consumer_outputs(config, grouped_params, activations)

    return update_step


def grouped_moe_mlp_consumer_outputs(mesh: Mesh, config: BenchConfig, grouped_params, routed_inputs):
    """Run the public grouped MoE helper over grouped expert-bank weights."""
    output_blocks = []
    implementation = "ring" if config.expert_axis > 1 else "scatter"
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        expert_mlp = grouped_params["blocks"][group_index]["mlp"]["expert_mlp"]
        block_inputs = routed_inputs["blocks"][group_index]
        with jax.named_scope("muon_update_bench/expert_grouped_moe_mlp_consumer"):
            output_blocks.append(
                {
                    "x": grouped_moe_mlp(
                        block_inputs["x"],
                        block_inputs["selected_experts"],
                        block_inputs["combine_weights"],
                        expert_mlp["w_gate_up"],
                        expert_mlp["w_down"],
                        valid_group_size=valid_group_size,
                        implementation=implementation,
                        mesh=mesh,
                    )
                }
            )
    return {"blocks": tuple(output_blocks)}


def grouped_moe_mlp_consumer_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(grouped_params, routed_inputs):
        with jax.named_scope("muon_update_bench/expert_grouped_moe_mlp_consumer_step"):
            return grouped_moe_mlp_consumer_outputs(mesh, config, grouped_params, routed_inputs)

    return update_step


def scale_with_grouped_4d_muon(config: BenchConfig, *, use_hyperball: bool) -> optax.GradientTransformation:
    """Muon-style transform for already-grouped `[group, expert, fan_in, fan_out]` leaves."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if use_hyperball and params is None:
            raise ValueError("Grouped MuonH requires params for per-matrix hyperball projection")

        def transform(update, param):
            if update is None or not hasattr(update, "ndim") or update.ndim != 4:
                return update
            with jax.named_scope("muon_update_bench/expert_grouped_optimizer/muon_4d_update"):
                update = zeropower_via_newtonschulz_4d_for_config(update, config)
                direction = scale_ns4d_direction(update)
                if use_hyperball:
                    return grouped_4d_hyperball_update(param, direction, config)
                return -config.learning_rate * direction

        if params is None:
            params = updates
        return jax.tree.map(transform, updates, params, is_leaf=lambda x: x is None), state

    return optax.GradientTransformation(init_fn, update_fn)


def grouped_expert_productionish_mask(params):
    paths = leaf_key_paths(params)

    def mask_fn(param, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        path_lower = path_str.lower()
        if ".mlp.expert_mlp.w_" in path_lower and hasattr(param, "ndim") and param.ndim == 4:
            return "grouped_muon"
        return "ordinary"

    return jax.tree.map(mask_fn, params, paths)


def full_production_muonh_mask(params):
    paths = leaf_key_paths(params)

    def mask_fn(param, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        path_lower = path_str.lower()
        if ".mlp.expert_mlp.w_" in path_lower and hasattr(param, "ndim") and param.ndim == 4:
            return "grouped_muonh"
        if _uses_adamh_baseline_adam_group(path_lower):
            return "ordinary"
        if "output_proj" in path_lower or "lm_head" in path_lower:
            return "ordinary"
        if "gated_norm" in path_lower:
            return "muonh"
        if hasattr(param, "ndim") and param.ndim in (2, 3):
            return "muonh"
        return "ordinary"

    return jax.tree.map(mask_fn, params, paths)


def build_grouped_expert_productionish_optimizer(
    config: BenchConfig,
    *,
    use_hyperball: bool = False,
) -> optax.GradientTransformation:
    grouped_muon = optax.chain(
        optax.trace(MAY_MOMENTUM, nesterov=True),
        scale_with_grouped_4d_muon(config, use_hyperball=use_hyperball),
    )
    ordinary = optax.scale(-config.learning_rate)
    return optax.multi_transform(
        {
            "grouped_muon": grouped_muon,
            "ordinary": ordinary,
        },
        grouped_expert_productionish_mask,
    )


def build_full_production_muonh_optimizer(
    config: BenchConfig,
    *,
    group_2d_muonh: bool = False,
) -> optax.GradientTransformation:
    grouped_muonh = optax.chain(
        optax.trace(MAY_MOMENTUM, nesterov=True),
        scale_with_grouped_4d_muon(config, use_hyperball=True),
    )
    muonh = (
        optax.chain(optax.trace(MAY_MOMENTUM, nesterov=True), scale_with_grouped_2d_muonh(config))
        if group_2d_muonh
        else build_muonh_transform(config)
    )
    return optax.multi_transform(
        {
            "grouped_muonh": grouped_muonh,
            "muonh": muonh,
            "ordinary": optax.scale(-config.learning_rate),
        },
        full_production_muonh_mask,
    )


def build_ordinary_2d_muonh_optimizer(
    config: BenchConfig,
    *,
    group_2d_muonh: bool = False,
) -> optax.GradientTransformation:
    muonh = scale_with_grouped_2d_muonh(config) if group_2d_muonh else build_muonh_transform(config)
    return optax.chain(optax.trace(MAY_MOMENTUM, nesterov=True), muonh)


def grouped_expert_optimizer_apply_step_factory(config: BenchConfig, *, use_hyperball: bool = False):
    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=use_hyperball)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/expert_grouped_optimizer_apply_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/expert_grouped_optimizer_apply/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state, next_updates

    return update_step


def grouped_expert_optimizer_apply_timing_step_factory(config: BenchConfig, *, use_hyperball: bool = False):
    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=use_hyperball)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/expert_grouped_optimizer_apply_timing_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/expert_grouped_optimizer_apply_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state

    return update_step


def _restore_grouped_expert_update_for_split(
    mesh: Mesh,
    update: jax.Array,
    valid_group_size: int,
    compute_sharding: NamedSharding,
) -> jax.Array:
    if compute_sharding.spec[0] is not None:
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh/restore_group_axis_replicated"):
            update = reshard(update, NamedSharding(mesh, P(None, "expert", None, None)))
    if valid_group_size != update.shape[0]:
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh/slice_valid_group_axis"):
            update = update[:valid_group_size]
    return update


def expert_fsdp_grouped_muonh_updates(mesh: Mesh, config: BenchConfig, params, updates):
    """Compute expert MuonH in grouped 4D form, then restore ordinary FSDP-shaped updates."""

    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for valid_group_size in grouped_expert_group_sizes(config):
        padded_group_size = grouped_expert_group_sizes_for_bench(
            config, EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH
        )[layer_offset // ns4d_group_size(config)]
        compute_sharding = grouped_expert_group_sharding(
            mesh,
            config,
            EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
            padded_group_size,
        )
        group_slice = slice(layer_offset, layer_offset + valid_group_size)
        for name in synthetic_shapes(config):
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/stack_group"):
                stacked_updates = jnp.stack(
                    [
                        updates["layers"][layer_index]["mlp"]["expert_mlp"][name]
                        for layer_index in range(group_slice.start, group_slice.stop)
                    ],
                    axis=0,
                )
            if padded_group_size != valid_group_size:
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/pad_group_axis"):
                    pad_width = [(0, padded_group_size - valid_group_size), *[(0, 0) for _ in stacked_updates.shape[1:]]]
                    stacked_updates = jnp.pad(stacked_updates, pad_width)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/reshard_grouped_compute"):
                stacked_updates = reshard(stacked_updates, compute_sharding)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/grouped_ns_direction"):
                direction = zeropower_via_newtonschulz_4d_for_config(stacked_updates, config)
                direction = scale_ns4d_direction(direction)
            direction = _restore_grouped_expert_update_for_split(
                mesh,
                direction,
                valid_group_size,
                compute_sharding,
            )
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/split_directions_to_fsdp_leaves"):
                direction_parts = [
                    jnp.squeeze(direction_part, axis=0)
                    for direction_part in jnp.split(direction, valid_group_size, axis=0)
                ]
            for local_index, direction_part in enumerate(direction_parts):
                layer_index = layer_offset + local_index
                param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                direction_part = _restore_param_sharding(direction_part, param)
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/fsdp_hyperball"):
                    output_layers[layer_index]["mlp"]["expert_mlp"][name] = grouped_3d_hyperball_update(
                        param,
                        direction_part,
                        config,
                    )
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_boundary_updates(mesh: Mesh, config: BenchConfig, params, grouped_updates):
    """Restore grouped 4D expert updates to ordinary FSDP-shaped per-layer updates."""
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        padded_group_size = grouped_expert_group_sizes_for_bench(
            config,
            EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        )[group_index]
        compute_sharding = grouped_expert_group_sharding(
            mesh,
            config,
            EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
            padded_group_size,
        )
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            update = _restore_grouped_expert_update_for_split(mesh, update, valid_group_size, compute_sharding)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_boundary/{name}/split_to_fsdp_leaves"):
                update_parts = [
                    jnp.squeeze(update_part, axis=0) for update_part in jnp.split(update, valid_group_size, axis=0)
                ]
            for local_index, update_part in enumerate(update_parts):
                layer_index = layer_offset + local_index
                param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = _restore_param_sharding(update_part, param)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_target_boundary_updates(config: BenchConfig, params, grouped_updates):
    """Restore grouped 4D expert updates through a 4D FSDP-compatible target layout."""
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            first_param = params["layers"][layer_offset]["mlp"]["expert_mlp"][name]
            target_sharding = _target_sharding(first_param)
            if isinstance(target_sharding, NamedSharding):
                target_group_sharding = NamedSharding(target_sharding.mesh, P(None, *target_sharding.spec))
                with jax.named_scope(
                    f"muon_update_bench/expert_fsdp_grouped_target_boundary/{name}/reshard_group_to_fsdp_layout"
                ):
                    update = reshard(update, target_group_sharding)
            if valid_group_size != update.shape[0]:
                with jax.named_scope(
                    f"muon_update_bench/expert_fsdp_grouped_target_boundary/{name}/slice_valid_group_axis"
                ):
                    update = update[:valid_group_size]
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_target_boundary/{name}/split_to_fsdp_leaves"):
                update_parts = [
                    jnp.squeeze(update_part, axis=0) for update_part in jnp.split(update, valid_group_size, axis=0)
                ]
            for local_index, update_part in enumerate(update_parts):
                layer_index = layer_offset + local_index
                param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = _restore_param_sharding(update_part, param)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_target_apply_outputs(config: BenchConfig, params, grouped_updates):
    """Apply grouped updates after converting them to a grouped FSDP-compatible layout."""
    output_blocks = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}}
        for _ in grouped_expert_group_sizes(config)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            grouped_param = jnp.stack(
                [
                    params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                    for layer_index in range(layer_offset, layer_offset + valid_group_size)
                ],
                axis=0,
            )
            first_param = params["layers"][layer_offset]["mlp"]["expert_mlp"][name]
            target_sharding = _target_sharding(first_param)
            if isinstance(target_sharding, NamedSharding):
                target_group_sharding = NamedSharding(target_sharding.mesh, P(None, *target_sharding.spec))
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_target_apply/{name}/reshard_param_group"):
                    grouped_param = reshard(grouped_param, target_group_sharding)
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_target_apply/{name}/reshard_update_group"):
                    update = reshard(update, target_group_sharding)
            if valid_group_size != update.shape[0]:
                with jax.named_scope(
                    f"muon_update_bench/expert_fsdp_grouped_target_apply/{name}/slice_valid_group_axis"
                ):
                    update = update[:valid_group_size]
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_target_apply/{name}/apply_group"):
                output_blocks[group_index]["mlp"]["expert_mlp"][name] = optax.apply_updates(grouped_param, update)
        layer_offset += valid_group_size
    return {"blocks": tuple(output_blocks)}


def _live_group_axis(mesh: Mesh, config: BenchConfig) -> str | tuple[str, ...] | None:
    requested_group_axis = ns4d_group_axis(config)
    if isinstance(requested_group_axis, tuple):
        live_axes = tuple(axis for axis in requested_group_axis if mesh_axis_size(mesh, axis) > 1)
        if len(live_axes) > 1:
            return live_axes
        if live_axes:
            return live_axes[0]
        return None
    if requested_group_axis is not None and mesh_axis_size(mesh, requested_group_axis) > 1:
        return requested_group_axis
    return None


def _explicit_grouped_update_to_fsdp_group(
    mesh: Mesh,
    config: BenchConfig,
    name: str,
    update: jax.Array,
) -> jax.Array:
    """Convert a grouped update to grouped FSDP layout with an explicit shard_map boundary."""
    require_model_axis_one(config)
    group_axis = _live_group_axis(mesh, config)
    input_spec = P(group_axis, "expert", None, None)
    if name == "w_gate_up":
        output_spec = P(None, "expert", "data", "model")
        data_sharded_axis = 2
    elif name == "w_down":
        output_spec = P(None, "expert", "model", "data")
        data_sharded_axis = 3
    else:
        raise ValueError(f"Unknown expert parameter name {name!r}.")

    data_axis_size = mesh_axis_size(mesh, "data")
    if update.shape[data_sharded_axis] % data_axis_size != 0:
        raise ValueError(
            f"{name} axis {data_sharded_axis} has size {update.shape[data_sharded_axis]}, "
            f"which must be divisible by data_axis={data_axis_size}."
        )

    def restore_group(local_update):
        if group_axis is None:
            gathered = local_update
        else:
            gathered = lax.all_gather(local_update, axis_name=group_axis, axis=0, tiled=True)
        if data_axis_size <= 1:
            return gathered
        data_index = lax.axis_index("data")
        local_axis_size = gathered.shape[data_sharded_axis] // data_axis_size
        start = data_index * local_axis_size
        return lax.dynamic_slice_in_dim(gathered, start, local_axis_size, axis=data_sharded_axis)

    return shard_map(
        restore_group,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        check_vma=False,
    )(update)


def _reorder_data_major_group_axis(update: jax.Array, replica_axis_size: int, data_axis_size: int) -> jax.Array:
    """Undo the data-major group order produced by a data-axis all_to_all."""
    group_axis_size = update.shape[0]
    shard_group_size = group_axis_size // (replica_axis_size * data_axis_size)
    reshaped = update.reshape((data_axis_size, replica_axis_size, shard_group_size, *update.shape[1:]))
    transposed = jnp.transpose(reshaped, (1, 0, 2, *range(3, reshaped.ndim)))
    return transposed.reshape(update.shape)


def _explicit_grouped_update_to_fsdp_group_a2a(
    mesh: Mesh,
    config: BenchConfig,
    name: str,
    update: jax.Array,
) -> jax.Array:
    """Convert grouped updates to grouped FSDP layout using data all-to-all instead of gather-and-slice."""
    require_model_axis_one(config)
    group_axis = _live_group_axis(mesh, config)
    if group_axis != ("replica_dcn", "data"):
        return _explicit_grouped_update_to_fsdp_group(mesh, config, name, update)

    input_spec = P(group_axis, "expert", None, None)
    if name == "w_gate_up":
        output_spec = P(None, "expert", "data", "model")
        data_sharded_axis = 2
    elif name == "w_down":
        output_spec = P(None, "expert", "model", "data")
        data_sharded_axis = 3
    else:
        raise ValueError(f"Unknown expert parameter name {name!r}.")

    replica_axis_size = mesh_axis_size(mesh, "replica_dcn")
    data_axis_size = mesh_axis_size(mesh, "data")
    if update.shape[0] % (replica_axis_size * data_axis_size) != 0:
        raise ValueError(
            f"Grouped axis has size {update.shape[0]}, which must be divisible by "
            f"replica_axis * data_axis = {replica_axis_size * data_axis_size}."
        )
    if update.shape[data_sharded_axis] % data_axis_size != 0:
        raise ValueError(
            f"{name} axis {data_sharded_axis} has size {update.shape[data_sharded_axis]}, "
            f"which must be divisible by data_axis={data_axis_size}."
        )

    def restore_group(local_update):
        gathered = lax.all_gather(local_update, axis_name="replica_dcn", axis=0, tiled=True)
        restored = lax.all_to_all(
            gathered,
            axis_name="data",
            split_axis=data_sharded_axis,
            concat_axis=0,
            tiled=True,
        )
        return _reorder_data_major_group_axis(restored, replica_axis_size, data_axis_size)

    return shard_map(
        restore_group,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        check_vma=False,
    )(update)


def expert_fsdp_grouped_explicit_boundary_updates(mesh: Mesh, config: BenchConfig, params, grouped_updates):
    """Restore grouped expert updates to FSDP leaves using an explicit shard_map boundary."""
    del params
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_explicit_boundary/{name}/restore_group"):
                update = _explicit_grouped_update_to_fsdp_group(mesh, config, name, update)
            if valid_group_size != update.shape[0]:
                with jax.named_scope(
                    f"muon_update_bench/expert_fsdp_grouped_explicit_boundary/{name}/slice_valid_group_axis"
                ):
                    update = update[:valid_group_size]
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_explicit_boundary/{name}/split_to_fsdp_leaves"):
                update_parts = [
                    jnp.squeeze(update_part, axis=0) for update_part in jnp.split(update, valid_group_size, axis=0)
                ]
            for local_index, update_part in enumerate(update_parts):
                layer_index = layer_offset + local_index
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = update_part
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_explicit_apply_outputs(mesh: Mesh, config: BenchConfig, params, grouped_updates):
    """Apply grouped expert updates to FSDP params using an explicit shard_map boundary."""
    return expert_fsdp_grouped_explicit_apply_outputs_with_restore(
        mesh,
        config,
        params,
        grouped_updates,
        _explicit_grouped_update_to_fsdp_group,
        "expert_fsdp_grouped_explicit_apply",
    )


def expert_fsdp_grouped_explicit_a2a_apply_outputs(mesh: Mesh, config: BenchConfig, params, grouped_updates):
    """Apply grouped expert updates to FSDP params using an explicit all-to-all boundary."""
    return expert_fsdp_grouped_explicit_apply_outputs_with_restore(
        mesh,
        config,
        params,
        grouped_updates,
        _explicit_grouped_update_to_fsdp_group_a2a,
        "expert_fsdp_grouped_explicit_a2a_apply",
    )


def expert_fsdp_grouped_explicit_apply_outputs_with_restore(
    mesh: Mesh,
    config: BenchConfig,
    params,
    grouped_updates,
    restore_update,
    scope_name: str,
):
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            with jax.named_scope(f"muon_update_bench/{scope_name}/{name}/restore_group"):
                update = restore_update(mesh, config, name, update)
            if valid_group_size != update.shape[0]:
                with jax.named_scope(f"muon_update_bench/{scope_name}/{name}/slice_valid_group_axis"):
                    update = update[:valid_group_size]
            with jax.named_scope(f"muon_update_bench/{scope_name}/{name}/split_apply"):
                update_parts = [
                    jnp.squeeze(update_part, axis=0) for update_part in jnp.split(update, valid_group_size, axis=0)
                ]
                for local_index, update_part in enumerate(update_parts):
                    layer_index = layer_offset + local_index
                    param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                    output_layers[layer_index]["mlp"]["expert_mlp"][name] = optax.apply_updates(param, update_part)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_explicit_restore_boundary_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_restore_boundary_step"):
            return expert_fsdp_grouped_explicit_boundary_updates(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_explicit_restore_boundary_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_restore_boundary_timing_step"):
            return expert_fsdp_grouped_explicit_boundary_updates(mesh, config, None, grouped_updates)

    return update_step


def expert_fsdp_grouped_explicit_apply_boundary_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_apply_boundary_step"):
            return expert_fsdp_grouped_explicit_apply_outputs(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_explicit_apply_boundary_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_apply_boundary_timing_step"):
            return expert_fsdp_grouped_explicit_apply_outputs(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_explicit_a2a_apply_boundary_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_a2a_apply_boundary_step"):
            return expert_fsdp_grouped_explicit_a2a_apply_outputs(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_explicit_a2a_apply_boundary_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_explicit_a2a_apply_boundary_timing_step"):
            return expert_fsdp_grouped_explicit_a2a_apply_outputs(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_apply_boundary_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_apply_boundary_step"):
            next_updates = expert_fsdp_grouped_boundary_updates(mesh, config, params, grouped_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_boundary/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_updates

    return update_step


def expert_fsdp_grouped_restore_boundary_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_restore_boundary_step"):
            return expert_fsdp_grouped_boundary_updates(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_restore_boundary_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_restore_boundary_timing_step"):
            return expert_fsdp_grouped_boundary_updates(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_target_apply_boundary_step_factory(config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_target_apply_boundary_step"):
            return expert_fsdp_grouped_target_apply_outputs(config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_target_apply_boundary_timing_step_factory(config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_target_apply_boundary_timing_step"):
            return expert_fsdp_grouped_target_apply_outputs(config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_target_restore_boundary_step_factory(config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_target_restore_boundary_step"):
            return expert_fsdp_grouped_target_boundary_updates(config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_target_restore_boundary_timing_step_factory(config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_target_restore_boundary_timing_step"):
            return expert_fsdp_grouped_target_boundary_updates(config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_apply_boundary_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_apply_boundary_timing_step"):
            next_updates = expert_fsdp_grouped_boundary_updates(mesh, config, params, grouped_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_boundary_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params

    return update_step


def expert_fsdp_grouped_updates_muonh_updates(mesh: Mesh, config: BenchConfig, params, grouped_updates):
    """Compute NS from already-grouped expert updates, then restore MuonH updates to FSDP leaves."""
    output_layers = [
        {"mlp": {"expert_mlp": {name: None for name in synthetic_shapes(config)}}} for _ in range(config.layers)
    ]
    layer_offset = 0
    for group_index, valid_group_size in enumerate(grouped_expert_group_sizes(config)):
        padded_group_size = grouped_expert_group_sizes_for_bench(
            config,
            EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        )[group_index]
        compute_sharding = grouped_expert_group_sharding(
            mesh,
            config,
            EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
            padded_group_size,
        )
        for name in synthetic_shapes(config):
            update = grouped_updates["blocks"][group_index]["mlp"]["expert_mlp"][name]
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_updates_muonh/{name}/stack_fsdp_params"):
                stacked_params = jnp.stack(
                    [
                        params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                        for layer_index in range(layer_offset, layer_offset + valid_group_size)
                    ],
                    axis=0,
                )
            if padded_group_size != valid_group_size:
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_updates_muonh/{name}/pad_param_group_axis"):
                    pad_width = [(0, padded_group_size - valid_group_size), *[(0, 0) for _ in stacked_params.shape[1:]]]
                    stacked_params = jnp.pad(stacked_params, pad_width)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_updates_muonh/{name}/reshard_params_grouped"):
                stacked_params = reshard(stacked_params, compute_sharding)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_updates_muonh/{name}/grouped_ns_hyperball"):
                direction = zeropower_via_newtonschulz_4d_for_config(update, config)
                direction = scale_ns4d_direction(direction)
                update = grouped_4d_hyperball_update(stacked_params, direction, config)
            update = _restore_grouped_expert_update_for_split(
                mesh,
                update,
                valid_group_size,
                compute_sharding,
            )
            with jax.named_scope(
                f"muon_update_bench/expert_fsdp_grouped_updates_muonh/{name}/split_updates_to_fsdp_leaves"
            ):
                update_parts = [
                    jnp.squeeze(update_part, axis=0) for update_part in jnp.split(update, valid_group_size, axis=0)
                ]
            for local_index, update_part in enumerate(update_parts):
                layer_index = layer_offset + local_index
                param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = _restore_param_sharding(update_part, param)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


def expert_fsdp_grouped_updates_muonh_updates_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh_updates_step"):
            return expert_fsdp_grouped_updates_muonh_updates(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_updates_muonh_updates_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh_updates_timing_step"):
            return expert_fsdp_grouped_updates_muonh_updates(mesh, config, params, grouped_updates)

    return update_step


def expert_fsdp_grouped_updates_muonh_apply_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh_apply_step"):
            next_updates = expert_fsdp_grouped_updates_muonh_updates(mesh, config, params, grouped_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_updates

    return update_step


def expert_fsdp_grouped_updates_muonh_apply_timing_step_factory(mesh: Mesh, config: BenchConfig):
    def update_step(params, grouped_updates):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh_apply_timing_step"):
            next_updates = expert_fsdp_grouped_updates_muonh_updates(mesh, config, params, grouped_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_updates_muonh_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params

    return update_step


def expert_fsdp_grouped_muonh_optimizer_apply_step_factory(mesh: Mesh, config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh_optimizer_apply_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_updates = expert_fsdp_grouped_muonh_updates(mesh, config, params, traced_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state, next_updates

    return update_step


def expert_fsdp_grouped_muonh_optimizer_apply_timing_step_factory(mesh: Mesh, config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh_optimizer_apply_timing_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_updates = expert_fsdp_grouped_muonh_updates(mesh, config, params, traced_updates)
            with jax.named_scope("muon_update_bench/expert_fsdp_grouped_muonh_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state

    return update_step


def full_production_muonh_optimizer_apply_step_factory(config: BenchConfig, *, group_2d_muonh: bool = False):
    optimizer = build_full_production_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/full_production_muonh_optimizer_apply_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/full_production_muonh_optimizer_apply/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state, next_updates

    return update_step


def full_production_muonh_optimizer_apply_timing_step_factory(config: BenchConfig, *, group_2d_muonh: bool = False):
    optimizer = build_full_production_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/full_production_muonh_optimizer_apply_timing_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/full_production_muonh_optimizer_apply_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state

    return update_step


def full_production_grouped_2d_direct_apply_timing_step_factory(config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/full_production_grouped_2d_direct_apply_timing_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_params = full_production_grouped_2d_direct_apply_outputs(params, traced_updates, config)
            return next_params, next_state

    return update_step


def full_production_grouped_2d_persistent_apply_timing_step_factory(config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/full_production_grouped_2d_persistent_apply_timing_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_params = full_production_grouped_persistent_apply_outputs(params, traced_updates, config)
            return next_params, next_state

    return update_step


def ordinary_2d_muonh_optimizer_apply_step_factory(config: BenchConfig, *, group_2d_muonh: bool = False):
    optimizer = build_ordinary_2d_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/ordinary_2d_muonh_optimizer_apply_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/ordinary_2d_muonh_optimizer_apply/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state, next_updates

    return update_step


def ordinary_2d_muonh_optimizer_apply_timing_step_factory(config: BenchConfig, *, group_2d_muonh: bool = False):
    optimizer = build_ordinary_2d_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/ordinary_2d_muonh_optimizer_apply_timing_step"):
            next_updates, next_state = optimizer.update(grads, state, params)
            with jax.named_scope("muon_update_bench/ordinary_2d_muonh_optimizer_apply_timing/optax_apply_updates"):
                next_params = optax.apply_updates(params, next_updates)
            return next_params, next_state

    return update_step


def ordinary_2d_grouped_direct_apply_timing_step_factory(config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/ordinary_2d_grouped_direct_apply_timing_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_params = grouped_2d_direct_apply_outputs(params, traced_updates, config)
            return next_params, next_state

    return update_step


def ordinary_2d_grouped_persistent_apply_timing_step_factory(config: BenchConfig):
    momentum = optax.trace(MAY_MOMENTUM, nesterov=True)

    def update_step(params, grads, state):
        with jax.named_scope("muon_update_bench/ordinary_2d_grouped_persistent_apply_timing_step"):
            traced_updates, next_state = momentum.update(grads, state, params)
            next_params = grouped_2d_persistent_apply_outputs(params, traced_updates, config)
            return next_params, next_state

    return update_step


def grouped_2d_stack_ns_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/ordinary_2d_grouped_stack_ns_step"):
            return grouped_2d_stack_ns_outputs(params, updates, config)

    return update_step


def grouped_2d_restore_split_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/ordinary_2d_grouped_restore_split_step"):
            return grouped_2d_restore_split_outputs(params, updates, config)

    return update_step


def full_production_apply_only_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/full_production_apply_only_step"):
            scaled_updates = jax.tree.map(lambda update: -config.learning_rate * update, updates)
            with jax.named_scope("muon_update_bench/full_production_apply_only/optax_apply_updates"):
                return optax.apply_updates(params, scaled_updates)

    return update_step


def ns4d_bench_uses_grouped_params(bench_kind: str) -> bool:
    return bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES or bench_kind in GROUPED_PARAM_INPUT_BENCHES


def ns4d_bench_returns_4d_updates(bench_kind: str) -> bool:
    return bench_kind in (
        NS4D_REPLICATED_GROUP_BENCH,
        NS4D_DATA_GROUP_BENCH,
        NS4D_DATA_GROUP_APPLY_BENCH,
        EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_GROUPED_BANK_CONSUMER_BENCH,
        FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        NS4D_DOTONLY_EINSUM_BENCH,
        NS4D_DOTONLY_MATMUL_BENCH,
        NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH,
        NS4D_PADDED_GROUP_BENCH,
        NS4D_DOTONLY_MATMUL_PADDED_BENCH,
    )


def ns4d_boundary_status(config: BenchConfig, bench_kind: str) -> str | None:
    if is_tree_update_bench(bench_kind):
        return None
    if bench_kind == ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH:
        return "ordinary_2d_muonh_optimizer_updates_apply"
    if bench_kind == ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "ordinary_2d_grouped_muonh_optimizer_updates_apply"
    if bench_kind == ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH:
        return "ordinary_2d_grouped_direct_apply"
    if bench_kind == ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH:
        return "ordinary_2d_grouped_persistent_apply"
    if bench_kind == ORDINARY_2D_GROUPED_STACK_NS_BENCH:
        return "ordinary_2d_grouped_stack_ns_only"
    if bench_kind == ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH:
        return "ordinary_2d_grouped_restore_split_only"
    if bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
        return "full_production_apply_only"
    if bench_kind == FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH:
        return "full_production_grouped_2d_persistent_apply"
    if bench_kind == NS4D_DATA_GROUP_APPLY_BENCH:
        return "grouped_params_updates_apply"
    if bench_kind == EXPERT_GROUPED_APPLY_BOUNDARY_BENCH:
        return "grouped_blocks_expert_params_updates_apply"
    if bench_kind == EXPERT_GROUPED_LAYER_SLICE_BENCH:
        return "grouped_blocks_expert_params_slice_to_ep_leaves"
    if bench_kind == EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH:
        return "grouped_blocks_expert_params_slice_one_ep_leaf"
    if bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
        return "grouped_blocks_expert_bank_consumer"
    if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
        return "grouped_blocks_public_moe_mlp_consumer"
    if bench_kind == EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH:
        return "grouped_blocks_expert_direction_optimizer_updates_apply"
    if bench_kind == EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "grouped_blocks_expert_muonh_optimizer_updates_apply"
    if bench_kind == EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "expert_only_grouped_blocks_muonh_optimizer_updates_apply"
    if bench_kind == EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_restore_then_apply"
    if bench_kind == EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_restore_only"
    if bench_kind == EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_target_restore_only"
    if bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_target_apply_grouped"
    if bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_explicit_restore_only"
    if bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_explicit_apply"
    if bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH:
        return "expert_fsdp_params_grouped_updates_explicit_a2a_apply"
    if bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH:
        return "expert_fsdp_params_grouped_updates_muonh_restore_only"
    if bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH:
        return "expert_fsdp_params_grouped_updates_muonh_restore_then_apply"
    if bench_kind == EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "expert_fsdp_params_grouped_muonh_restore_then_apply"
    if is_full_production_muonh_bench(bench_kind):
        if bench_kind == FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH:
            return "full_production_grouped_2d_direct_apply"
        return "full_production_muonh_optimizer_updates_apply"
    if bench_kind == NS4D_DATA_RESHARD_RESTORE_BENCH:
        return "restore_then_split"
    if bench_kind == NS4D_DATA_INDEX_RESTORE_BENCH:
        return "direct_index_restore"
    if bench_kind in NS4D_PADDED_BENCHES and padded_ns4d_group_size(config, bench_kind) != ns4d_group_size(config):
        return "pad_compute_restore_then_slice"
    return "grouped_updates_only"


def assert_grouped_apply_has_no_boundary_collectives(hlo_summary: HloSummary, label: str) -> None:
    nonzero_counts = grouped_apply_boundary_collectives(hlo_summary)
    if nonzero_counts:
        raise AssertionError(
            f"{label} forced grouped apply boundary collectives {nonzero_counts}; "
            "this means grouped params/updates did not survive the apply boundary."
        )


def grouped_apply_boundary_collectives(hlo_summary: HloSummary) -> dict[str, int]:
    collective_counts = {
        "all_gather": hlo_summary.all_gather,
        "reduce_scatter": hlo_summary.reduce_scatter,
        "all_reduce": hlo_summary.all_reduce,
        "all_to_all": hlo_summary.all_to_all,
    }
    return {name: count for name, count in collective_counts.items() if count}


def is_expert_only_grouped_muonh_bench(bench_kind: str) -> bool:
    return bench_kind == EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH


def is_expert_fsdp_grouped_muonh_bench(bench_kind: str) -> bool:
    return bench_kind == EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH


def is_expert_fsdp_grouped_updates_muonh_bench(bench_kind: str) -> bool:
    return bench_kind in (
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    )


def is_expert_fsdp_grouped_boundary_bench(bench_kind: str) -> bool:
    return bench_kind in (
        EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    )


def is_expert_fsdp_grouped_bench(bench_kind: str) -> bool:
    return bench_kind in (
        EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    )


def scale_ns4d_update(update: jax.Array, config: BenchConfig) -> jax.Array:
    with jax.named_scope("muon_update_bench/ns4d_grouped_apply/scale_update"):
        return -config.learning_rate * scale_ns4d_direction(update)


def scale_ns4d_direction(update: jax.Array) -> jax.Array:
    fan_in, fan_out = update.shape[-2:]
    with jax.named_scope("muon_update_bench/ns4d_grouped_apply/scale_direction"):
        scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
        return scale * update


def grouped_4d_hyperball_update(param: jax.Array, direction: jax.Array, config: BenchConfig) -> jax.Array:
    """Project grouped 4D Muon directions independently per `[group, expert]` matrix."""
    if param is None:
        raise ValueError("Grouped MuonH hyperball requires params")
    axes = (-2, -1)
    with jax.named_scope("muon_update_bench/expert_grouped_muonh/hyperball_matrix_norms"):
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(direction), axis=axes, keepdims=True))
        step_scale = config.learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
        dot = jnp.sum(param * direction, axis=axes, keepdims=True)
    with jax.named_scope("muon_update_bench/expert_grouped_muonh/hyperball_matrix_projection"):
        new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
        new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
        rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
        return (rescale - 1) * param - rescale * step_scale * direction


def grouped_3d_hyperball_update(param: jax.Array, direction: jax.Array, config: BenchConfig) -> jax.Array:
    """Project grouped 3D Muon directions independently per `[group, fan_in, fan_out]` matrix."""
    if param is None:
        raise ValueError("Grouped 2D MuonH hyperball requires params")
    if param.ndim != 3 or direction.ndim != 3:
        raise ValueError(
            "Grouped 2D MuonH hyperball expects stacked 3D params/directions; "
            f"got param.ndim={param.ndim}, direction.ndim={direction.ndim}."
        )

    with jax.named_scope("muon_update_bench/grouped_2d_muonh/hyperball_matrix_norms"):
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=(-2, -1), keepdims=True))
        direction_norm = jnp.sqrt(jnp.sum(jnp.square(direction), axis=(-2, -1), keepdims=True))
        step_scale = config.learning_rate * param_norm / jnp.maximum(direction_norm, 1e-10)
        dot = jnp.sum(param * direction, axis=(-2, -1), keepdims=True)
    with jax.named_scope("muon_update_bench/grouped_2d_muonh/hyperball_matrix_projection"):
        new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * direction_norm**2
        new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
        rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
        return (rescale - 1) * param - rescale * step_scale * direction


def _stacked_2d_target(shape: tuple[int, int, int], sample: jax.Array) -> NamedSharding | None:
    sharding = _target_sharding(sample)
    if not isinstance(sharding, NamedSharding) or sharding.mesh.empty:
        return None

    stack_size = shape[0]
    candidates: list[tuple[int, str | tuple[str, ...]]] = []
    live_axes = tuple(axis for axis in ("replica_dcn", "data") if mesh_axis_size(sharding.mesh, axis) > 1)
    for axis_mask in range(1, 1 << len(live_axes)):
        axes = tuple(axis for index, axis in enumerate(live_axes) if axis_mask & (1 << index))
        axis_size = math.prod(mesh_axis_size(sharding.mesh, axis) for axis in axes)
        padded_stack_size = math.ceil(stack_size / axis_size) * axis_size
        if padded_stack_size / stack_size <= STACK_PADDING_MAX_OVERHEAD:
            candidates.append((axis_size, axes[0] if len(axes) == 1 else axes))
    if not candidates:
        return None

    _, stack_axis = max(candidates, key=lambda item: item[0])
    return NamedSharding(sharding.mesh, P(stack_axis, None, None))


def _restore_stacked_2d_for_split(stacked: jax.Array, target_sharding: NamedSharding | None) -> jax.Array:
    if target_sharding is None or target_sharding.spec[0] is None:
        return stacked
    return reshard(stacked, NamedSharding(target_sharding.mesh, P(None, None, None)))


def _restore_param_sharding(update: jax.Array, param: jax.Array) -> jax.Array:
    target_sharding = _target_sharding(param)
    if target_sharding is None:
        return update
    return reshard(update, target_sharding)


def _group_2d_entries(updates, params):
    update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
    param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
    if treedef != param_treedef:
        raise ValueError("Grouped 2D MuonH requires updates and params to have matching tree structure.")

    output_leaves = [None] * len(update_leaves)
    groups: dict[tuple[tuple[int, int], str, str], list[tuple[int, jax.Array, jax.Array]]] = {}
    for index, (update, param) in enumerate(zip(update_leaves, param_leaves, strict=True)):
        if update is None:
            output_leaves[index] = None
            continue
        if hasattr(update, "ndim") and update.ndim == 2:
            param_spec = getattr(_target_sharding(param), "spec", None)
            key = (tuple(update.shape), str(update.dtype), str(param_spec))
            groups.setdefault(key, []).append((index, update, param))
            continue
        output_leaves[index] = update
    return treedef, output_leaves, groups


def _grouped_2d_chunks(
    groups: dict[tuple[tuple[int, int], str, str], list[tuple[int, jax.Array, jax.Array]]],
    max_grouped_stack_size: int,
):
    for entries in groups.values():
        for chunk_start in range(0, len(entries), max_grouped_stack_size):
            yield entries[chunk_start : chunk_start + max_grouped_stack_size]


def grouped_2d_stack_ns_outputs(params, updates, config: BenchConfig):
    """Return grouped 2D stack+Newton-Schulz outputs without split/hyperball/apply."""
    _treedef, _output_leaves, groups = _group_2d_entries(updates, params)
    outputs = []
    for entry_chunk in _grouped_2d_chunks(groups, config.max_grouped_stack_size):
        sample_update = entry_chunk[0][1]
        stack_shape = (len(entry_chunk), *sample_update.shape)
        target_pspec = _stacked_2d_target(stack_shape, sample_update)
        with jax.named_scope("muon_update_bench/grouped_2d_decompose/stack_ns"):
            stacked_updates = jnp.stack([update for _, update, _ in entry_chunk], axis=0)
            directions = zeropower_via_newtonschulz_3d_stack_for_config(stacked_updates, config, target_pspec)
            outputs.append(scale_ns4d_direction(directions))
    return tuple(outputs)


def grouped_2d_restore_split_outputs(params, updates, config: BenchConfig):
    """Return a tree after only stack, split, and target-sharding restore."""
    treedef, output_leaves, groups = _group_2d_entries(updates, params)
    for entry_chunk in _grouped_2d_chunks(groups, config.max_grouped_stack_size):
        sample_update = entry_chunk[0][1]
        stack_shape = (len(entry_chunk), *sample_update.shape)
        target_pspec = _stacked_2d_target(stack_shape, sample_update)
        with jax.named_scope("muon_update_bench/grouped_2d_decompose/restore_split"):
            stacked_updates = jnp.stack([update for _, update, _ in entry_chunk], axis=0)
            stacked_updates = _restore_stacked_2d_for_split(stacked_updates, target_pspec)
            update_parts = [
                jnp.squeeze(update_part, axis=0) for update_part in jnp.split(stacked_updates, len(entry_chunk), axis=0)
            ]
        for (index, _, param), update in zip(entry_chunk, update_parts, strict=True):
            output_leaves[index] = _restore_param_sharding(update, param)
    return jax.tree.unflatten(treedef, output_leaves)


def _apply_grouped_2d_direct_chunks(
    output_leaves: list[Any],
    groups: dict[tuple[tuple[int, int], str, str], list[tuple[int, jax.Array, jax.Array]]],
    config: BenchConfig,
) -> None:
    for entry_chunk in _grouped_2d_chunks(groups, config.max_grouped_stack_size):
        sample_update = entry_chunk[0][1]
        stack_shape = (len(entry_chunk), *sample_update.shape)
        target_pspec = _stacked_2d_target(stack_shape, sample_update)

        with jax.named_scope("muon_update_bench/grouped_2d_direct_apply/stacked_hyperball_apply"):
            stacked_updates = jnp.stack([update for _, update, _ in entry_chunk], axis=0)
            stacked_params = jnp.stack([param for _, _, param in entry_chunk], axis=0)
            if target_pspec is not None:
                stacked_params = reshard(stacked_params, target_pspec)
            directions = zeropower_via_newtonschulz_3d_stack_for_config(stacked_updates, config, target_pspec)
            directions = scale_ns4d_direction(directions)
            stacked_updates = grouped_3d_hyperball_update(stacked_params, directions, config)
            next_stacked_params = stacked_params + stacked_updates
            next_param_parts = [
                jnp.squeeze(param_part, axis=0)
                for param_part in jnp.split(next_stacked_params, len(entry_chunk), axis=0)
            ]

        for (index, _, _), next_param in zip(entry_chunk, next_param_parts, strict=True):
            output_leaves[index] = next_param


def grouped_2d_direct_apply_outputs(params, updates, config: BenchConfig):
    """Apply grouped 2D MuonH in stacked form, then split directly back to the param tree."""
    update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
    param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
    if treedef != param_treedef:
        raise ValueError("Grouped 2D direct apply requires updates and params to have matching tree structure.")

    output_leaves: list[Any] = [None] * len(update_leaves)
    groups: dict[tuple[tuple[int, int], str, str], list[tuple[int, jax.Array, jax.Array]]] = {}
    for index, (update, param) in enumerate(zip(update_leaves, param_leaves, strict=True)):
        if update is None:
            output_leaves[index] = param
            continue
        if hasattr(update, "ndim") and update.ndim == 2:
            param_spec = getattr(_target_sharding(param), "spec", None)
            key = (tuple(update.shape), str(update.dtype), str(param_spec))
            groups.setdefault(key, []).append((index, update, param))
            continue
        output_leaves[index] = param + update

    _apply_grouped_2d_direct_chunks(output_leaves, groups, config)
    return jax.tree.unflatten(treedef, output_leaves)


def full_production_grouped_2d_direct_apply_outputs(params, updates, config: BenchConfig):
    """Apply production MuonH groups without restoring grouped 2D updates before apply."""
    update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
    param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
    mask_leaves, mask_treedef = jax.tree.flatten(full_production_muonh_mask(params), is_leaf=lambda x: x is None)
    if treedef != param_treedef or treedef != mask_treedef:
        raise ValueError("Full-production direct apply requires matching update, param, and mask trees.")

    output_leaves: list[Any] = [None] * len(update_leaves)
    groups: dict[tuple[tuple[int, int], str, str], list[tuple[int, jax.Array, jax.Array]]] = {}
    for index, (update, param, mask) in enumerate(zip(update_leaves, param_leaves, mask_leaves, strict=True)):
        if update is None:
            output_leaves[index] = param
            continue
        if mask == "grouped_muonh" and hasattr(update, "ndim") and update.ndim == 4:
            with jax.named_scope("muon_update_bench/full_production_direct_apply/grouped_4d_expert"):
                direction = zeropower_via_newtonschulz_4d_for_config(update, config)
                direction = scale_ns4d_direction(direction)
                output_leaves[index] = param + grouped_4d_hyperball_update(param, direction, config)
            continue
        if mask == "muonh" and hasattr(update, "ndim") and update.ndim == 2:
            param_spec = getattr(_target_sharding(param), "spec", None)
            key = (tuple(update.shape), str(update.dtype), str(param_spec))
            groups.setdefault(key, []).append((index, update, param))
            continue
        output_leaves[index] = param - config.learning_rate * update

    _apply_grouped_2d_direct_chunks(output_leaves, groups, config)
    return jax.tree.unflatten(treedef, output_leaves)


def grouped_2d_persistent_apply_outputs(params, updates, config: BenchConfig):
    """Apply MuonH to persistent grouped `[group, fan_in, fan_out]` leaves."""

    def apply_group(param, update):
        if update is None:
            return param
        if not hasattr(update, "ndim") or update.ndim != 3:
            return param - config.learning_rate * update
        target_pspec = _target_sharding(update)
        with jax.named_scope("muon_update_bench/grouped_2d_persistent_apply/grouped_hyperball_apply"):
            if target_pspec is not None:
                param = reshard(param, target_pspec)
            direction = zeropower_via_newtonschulz_3d_stack_for_config(update, config, target_pspec)
            direction = scale_ns4d_direction(direction)
            return param + grouped_3d_hyperball_update(param, direction, config)

    return jax.tree.map(apply_group, params, updates, is_leaf=lambda x: x is None)


def full_production_grouped_persistent_apply_outputs(params, updates, config: BenchConfig):
    """Apply production MuonH with ordinary 2D groups kept as persistent stacked leaves."""

    def apply_group(param, update):
        if update is None:
            return param
        if not hasattr(update, "ndim"):
            return param - config.learning_rate * update
        if update.ndim == 4:
            with jax.named_scope("muon_update_bench/full_production_persistent_apply/grouped_4d_expert"):
                direction = zeropower_via_newtonschulz_4d_for_config(update, config)
                direction = scale_ns4d_direction(direction)
                return param + grouped_4d_hyperball_update(param, direction, config)
        if update.ndim == 3:
            target_pspec = _target_sharding(update)
            with jax.named_scope("muon_update_bench/full_production_persistent_apply/grouped_2d"):
                if target_pspec is not None:
                    param = reshard(param, target_pspec)
                direction = zeropower_via_newtonschulz_3d_stack_for_config(update, config, target_pspec)
                direction = scale_ns4d_direction(direction)
                return param + grouped_3d_hyperball_update(param, direction, config)
        return param - config.learning_rate * update

    return jax.tree.map(apply_group, params, updates, is_leaf=lambda x: x is None)


def scale_with_grouped_2d_muonh(config: BenchConfig) -> optax.GradientTransformation:
    """MuonH transform that batches same-shaped 2D leaves before Newton-Schulz."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Grouped 2D MuonH requires params for hyperball projection")

        treedef, output_leaves, groups = _group_2d_entries(updates, params)
        for entry_chunk in _grouped_2d_chunks(groups, config.max_grouped_stack_size):
            sample_update = entry_chunk[0][1]
            stack_shape = (len(entry_chunk), *sample_update.shape)
            target_pspec = _stacked_2d_target(stack_shape, sample_update)

            with jax.named_scope("muon_update_bench/grouped_2d_muonh/orthogonalize_stack"):
                stacked_updates = jnp.stack([update for _, update, _ in entry_chunk], axis=0)
                stacked_params = jnp.stack([param for _, _, param in entry_chunk], axis=0)
                directions = zeropower_via_newtonschulz_3d_stack_for_config(stacked_updates, config, target_pspec)
                directions = scale_ns4d_direction(directions)
                stacked_updates = grouped_3d_hyperball_update(stacked_params, directions, config)
                stacked_updates = _restore_stacked_2d_for_split(stacked_updates, target_pspec)
                update_parts = [
                    jnp.squeeze(update_part, axis=0)
                    for update_part in jnp.split(stacked_updates, len(entry_chunk), axis=0)
                ]

            for (index, _, param), update in zip(entry_chunk, update_parts, strict=True):
                output_leaves[index] = _restore_param_sharding(update, param)

        return jax.tree.unflatten(treedef, output_leaves), state

    return optax.GradientTransformation(init_fn, update_fn)


def padded_ns4d_update(
    update: jax.Array,
    config: BenchConfig,
    bench_kind: str,
    compute_sharding: NamedSharding,
) -> jax.Array:
    original_group_size = update.shape[0]
    target_group_size = padded_ns4d_group_size(config, bench_kind)
    if target_group_size != original_group_size:
        with jax.named_scope("muon_update_bench/ns4d_padded/pad_group_axis"):
            pad_width = [(0, target_group_size - original_group_size), *[(0, 0) for _ in update.shape[1:]]]
            update = jnp.pad(update, pad_width)

    with jax.named_scope("muon_update_bench/ns4d_padded/reshard_padded_group_axis"):
        update = reshard(update, compute_sharding)

    if bench_kind == NS4D_DOTONLY_MATMUL_PADDED_BENCH:
        update = newtonschulz_4d_dotonly_for_config(update, config, "matmul")
    elif bench_kind == NS4D_PADDED_GROUP_BENCH:
        update = zeropower_via_newtonschulz_4d_for_config(update, config)
    else:
        raise ValueError(f"Unsupported padded NS4D bench kind: {bench_kind!r}")

    if target_group_size != original_group_size:
        with jax.named_scope("muon_update_bench/ns4d_padded/restore_group_axis_before_slice"):
            update = reshard(update, NamedSharding(compute_sharding.mesh, P(None, "expert", None, None)))
        with jax.named_scope("muon_update_bench/ns4d_padded/slice_group_axis"):
            update = update[:original_group_size]
    return update


@jax.named_call
def zeropower_via_newtonschulz_4d(
    x: jax.Array,
    steps: int,
    eps: float,
) -> jax.Array:
    """Newton-Schulz over `[group, expert, m, n]` without flattening batch axes."""
    assert x.ndim == 4
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]
    with jax.named_scope("newton_schulz_4d/normalize_input"):
        x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)

    transpose = False
    if x.shape[-2] > x.shape[-1]:
        with jax.named_scope("newton_schulz_4d/transpose_tall_matrices"):
            x = jnp.swapaxes(x, -1, -2)
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_4d/iter_{i}/gram"):
            gram = jnp.einsum("...ik,...jk->...ij", x, x)
        with jax.named_scope(f"newton_schulz_4d/iter_{i}/polynomial"):
            polynomial = b * gram + c * jnp.einsum("...ik,...kj->...ij", gram, gram)
        with jax.named_scope(f"newton_schulz_4d/iter_{i}/apply"):
            x = a * x + jnp.einsum("...ik,...kj->...ij", polynomial, x)

    if transpose:
        with jax.named_scope("newton_schulz_4d/restore_transpose"):
            x = jnp.swapaxes(x, -1, -2)
    return x


@jax.named_call
def newtonschulz_4d_dotonly_einsum(x: jax.Array, steps: int) -> jax.Array:
    """Run only the three Newton-Schulz dot shapes per step using einsum."""
    assert x.ndim == 4
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]

    for i in range(steps):
        _, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_4d_dotonly_einsum/iter_{i}/gram"):
            gram = jnp.einsum("...ik,...jk->...ij", x, x)
        with jax.named_scope(f"newton_schulz_4d_dotonly_einsum/iter_{i}/polynomial"):
            polynomial = b * gram + c * jnp.einsum("...ik,...kj->...ij", gram, gram)
        with jax.named_scope(f"newton_schulz_4d_dotonly_einsum/iter_{i}/apply"):
            x = jnp.einsum("...ik,...kj->...ij", polynomial, x)
    return x


@jax.named_call
def newtonschulz_4d_dotonly_matmul(x: jax.Array, steps: int) -> jax.Array:
    """Run only the three Newton-Schulz dot shapes per step using jnp.matmul."""
    assert x.ndim == 4
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]

    for i in range(steps):
        _, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_4d_dotonly_matmul/iter_{i}/gram"):
            gram = jnp.matmul(x, jnp.swapaxes(x, -1, -2))
        with jax.named_scope(f"newton_schulz_4d_dotonly_matmul/iter_{i}/polynomial"):
            polynomial = b * gram + c * jnp.matmul(gram, gram)
        with jax.named_scope(f"newton_schulz_4d_dotonly_matmul/iter_{i}/apply"):
            x = jnp.matmul(polynomial, x)
    return x


@jax.named_call
def newtonschulz_4d_dotonly_lax_dot_general(x: jax.Array, steps: int) -> jax.Array:
    """Run only the three Newton-Schulz dot shapes per step using lax.dot_general."""
    assert x.ndim == 4
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]
    batch_axes = ((0, 1), (0, 1))
    matrix_contract = ((3,), (2,))
    dimension_numbers = (matrix_contract, batch_axes)

    for i in range(steps):
        _, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_4d_dotonly_lax_dot_general/iter_{i}/gram"):
            gram = lax.dot_general(x, jnp.swapaxes(x, -1, -2), dimension_numbers)
        with jax.named_scope(f"newton_schulz_4d_dotonly_lax_dot_general/iter_{i}/polynomial"):
            polynomial = b * gram + c * lax.dot_general(gram, gram, dimension_numbers)
        with jax.named_scope(f"newton_schulz_4d_dotonly_lax_dot_general/iter_{i}/apply"):
            x = lax.dot_general(polynomial, x, dimension_numbers)
    return x


def restore_ns4d_group_with_reshard(mesh: Mesh, updates: Any) -> Any:
    """Mirror the production 4D path that removes data sharding before splitting."""

    def restore(update):
        with jax.named_scope("muon_update_bench/ns4d_restore/reshard_group_axis_replicated"):
            update = reshard(update, NamedSharding(mesh, P(None, "expert", None, None)))
        with jax.named_scope("muon_update_bench/ns4d_restore/split_group_axis"):
            return tuple(jnp.squeeze(part, axis=0) for part in jnp.split(update, update.shape[0], axis=0))

    return jax.tree.map(restore, updates)


def restore_ns4d_group_with_indexing(updates: Any) -> Any:
    """Try direct group-axis indexing to test whether XLA can avoid the explicit pre-split reshard."""

    def restore(update):
        with jax.named_scope("muon_update_bench/ns4d_restore/index_group_axis"):
            return tuple(update[index] for index in range(update.shape[0]))

    return jax.tree.map(restore, updates)


def estimate_grouping(config: BenchConfig) -> list[GroupEstimate]:
    estimates = []
    for shape in synthetic_shapes(config).values():
        chunks = []
        current = 0
        for _ in range(config.layers):
            stack_size = shape[0]
            if current and current + stack_size > config.max_grouped_stack_size:
                chunks.append(current)
                current = 0
            current += stack_size
        if current:
            chunks.append(current)
        estimates.append(
            GroupEstimate(
                leaf_shape=shape,
                leaves=config.layers,
                chunks=chunks,
                grouped_chunks=sum(chunk > shape[0] for chunk in chunks),
            )
        )
    return estimates


def estimate_grouped_2d_muonh(mesh: Mesh, config: BenchConfig, *, full_production_tree: bool) -> list[Grouped2DEstimate]:
    specs = (
        synthetic_full_production_muonh_specs(mesh, config, FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH)
        if full_production_tree
        else synthetic_ordinary_2d_muonh_specs(mesh, config)
    )
    if full_production_tree:
        mask = full_production_muonh_mask(specs)
    else:
        mask = jax.tree.map(lambda leaf: "muonh" if hasattr(leaf, "ndim") and leaf.ndim == 2 else "ordinary", specs)
    mask_by_path = dict(leaf_items(mask))

    groups: dict[tuple[tuple[int, int], str, str], int] = {}
    for path, leaf in leaf_items(specs):
        leaf_mask = mask_by_path[path]
        if leaf_mask != "muonh" or not hasattr(leaf, "ndim") or leaf.ndim != 2:
            continue
        sharding_spec = getattr(getattr(leaf, "sharding", None), "spec", None)
        key = (tuple(leaf.shape), str(leaf.dtype), str(sharding_spec))
        groups[key] = groups.get(key, 0) + 1

    estimates = []
    for (shape, _dtype, sharding_spec), leaves in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2])):
        chunks = [
            min(config.max_grouped_stack_size, leaves - chunk_start)
            for chunk_start in range(0, leaves, config.max_grouped_stack_size)
        ]
        estimates.append(
            Grouped2DEstimate(
                leaf_shape=shape,
                sharding_spec=sharding_spec,
                leaves=leaves,
                chunks=chunks,
                grouped_chunks=sum(chunk > 1 for chunk in chunks),
            )
        )
    return estimates


def summarize_hlo(hlo_text: str) -> HloSummary:
    return HloSummary(
        characters=len(hlo_text),
        dot_general=len(DOT_GENERAL_RE.findall(hlo_text)),
        batched_stack_dot_general=len(BATCHED_STACK_DOT_RE.findall(hlo_text)),
        two_batch_axis_dot_general=len(TWO_BATCH_AXIS_DOT_RE.findall(hlo_text)),
        custom_call=len(CUSTOM_CALL_RE.findall(hlo_text)),
        gpu_gemm_custom_call=len(GPU_GEMM_CUSTOM_CALL_RE.findall(hlo_text)),
        all_gather=len(ALL_GATHER_RE.findall(hlo_text)),
        all_reduce=len(ALL_REDUCE_RE.findall(hlo_text)),
        reduce_scatter=len(REDUCE_SCATTER_RE.findall(hlo_text)),
        all_to_all=len(ALL_TO_ALL_RE.findall(hlo_text)),
        collective_permute=len(COLLECTIVE_PERMUTE_RE.findall(hlo_text)),
        grouped_scope_mentions=hlo_text.count("orthogonalize_3d_grouped_stack"),
        stack_sharded_scope_mentions=hlo_text.count("orthogonalize_3d_stack_sharded"),
        pad_scope_mentions=hlo_text.count("pad_stack_axis"),
        slice_scope_mentions=hlo_text.count("slice_padded_stack_axis"),
    )


def maybe_write_text(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def is_tree_update_bench(bench_kind: str) -> bool:
    return bench_kind in (MUONH_UPDATE_BENCH, MUON_DIRECTION_BENCH, HYPERBALL_ONLY_BENCH)


def is_ordinary_2d_decomposition_bench(bench_kind: str) -> bool:
    return bench_kind in (*ORDINARY_2D_DECOMPOSITION_BENCHES, *NON_NS_APPLY_BENCHES)


def bench_skip_reason(config: BenchConfig, bench_kind: str) -> str | None:
    if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH and config.expert_axis <= 1:
        return (
            f"{bench_kind} validates the expert-parallel grouped MoE path and requires expert_axis > 1; "
            "use expert_grouped_bank_consumer for the no-EP dense grouped-bank proxy."
        )
    if bench_kind == NS4D_DATA_INDEX_RESTORE_BENCH:
        group_axis = ns4d_group_axis(config)
        if group_axis is not None and ns4d_axis_size(config) > 1:
            return (
                f"{bench_kind} cannot directly index group axis shards for {group_axis}={ns4d_axis_size(config)}; "
                "JAX requires the sliced output dimension to remain divisible by the mesh axis."
            )
    if (
        bench_kind in NS4D_DATA_SHARDED_BENCHES
        and bench_kind not in NS4D_PADDED_BENCHES
        and bench_kind not in GROUPED_APPLY_BOUNDARY_BENCHES
        and not is_expert_fsdp_grouped_muonh_bench(bench_kind)
    ):
        group_axis = ns4d_group_axis(config)
        if group_axis is None:
            return None
        axis_size = ns4d_axis_size(config)
        group_size = ns4d_group_size(config)
        if group_size % axis_size != 0:
            return (
                f"{bench_kind} requires ns4d group axis {group_size} to be divisible by "
                f"{group_axis}={axis_size} for P({group_axis}, expert, None, None)."
            )
    return None


def ns_dot_flops_for_shape(shape: tuple[int, int, int], group_size: int, steps: int) -> int:
    experts, fan_in, fan_out = shape
    return group_size * experts * ns_dot_flops_for_matrix_shape((fan_in, fan_out), steps)


def ns_dot_flops_for_matrix_shape(shape: tuple[int, int], steps: int) -> int:
    fan_in, fan_out = shape
    rows = min(fan_in, fan_out)
    cols = max(fan_in, fan_out)
    gram_flops = 2 * rows * cols * rows
    polynomial_flops = 2 * rows * rows * rows
    apply_flops = 2 * rows * rows * cols
    return steps * (gram_flops + polynomial_flops + apply_flops)


def grouped_expert_bank_consumer_flops(config: BenchConfig) -> int:
    tokens_per_expert = config.grouped_expert_consumer_tokens_per_expert
    # Two dense expert MLP projections over grouped expert banks:
    # x @ w_gate_up plus hidden @ w_down. Elementwise gate work is excluded.
    per_layer = (
        config.num_experts
        * tokens_per_expert
        * (2 * config.hidden_dim * (2 * config.intermediate_dim) + 2 * config.intermediate_dim * config.hidden_dim)
    )
    return config.layers * per_layer


def estimated_full_production_muonh_ns_dot_flops(config: BenchConfig) -> int:
    per_layer_2d = sum(
        ns_dot_flops_for_matrix_shape(shape, config.backend_steps)
        for shape in production_muonh_2d_shapes(config).values()
    )
    per_layer_expert = sum(
        ns_dot_flops_for_shape(shape, 1, config.backend_steps) for shape in synthetic_shapes(config).values()
    )
    return config.layers * (per_layer_2d + per_layer_expert)


def estimated_ordinary_2d_muonh_ns_dot_flops(config: BenchConfig) -> int:
    per_layer_2d = sum(
        ns_dot_flops_for_matrix_shape(shape, config.backend_steps)
        for shape in production_muonh_2d_shapes(config).values()
    )
    return config.layers * per_layer_2d


def estimated_ns_dot_flops(config: BenchConfig, bench_kind: str) -> int:
    if bench_kind in (EXPERT_GROUPED_BANK_CONSUMER_BENCH, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH):
        return grouped_expert_bank_consumer_flops(config)
    if bench_kind in (
        HYPERBALL_ONLY_BENCH,
        ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
        FULL_PRODUCTION_APPLY_ONLY_BENCH,
        EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    ):
        return 0
    if bench_kind == FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH:
        return estimated_full_production_muonh_ns_dot_flops(config)
    if bench_kind in (
        ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_STACK_NS_BENCH,
    ):
        return estimated_ordinary_2d_muonh_ns_dot_flops(config)
    if is_full_production_muonh_bench(bench_kind):
        return estimated_full_production_muonh_ns_dot_flops(config)
    if bench_kind in (
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ):
        return sum(
            ns_dot_flops_for_shape(shape, group_size, config.backend_steps)
            for group_size in grouped_expert_group_sizes_for_bench(config, bench_kind)
            for shape in synthetic_shapes(config).values()
        )
    if is_tree_update_bench(bench_kind):
        group_size = config.layers
    elif bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES:
        group_size = config.layers
    else:
        group_size = padded_ns4d_group_size(config, bench_kind)
    return sum(
        ns_dot_flops_for_shape(shape, group_size, config.backend_steps) for shape in synthetic_shapes(config).values()
    )


def estimated_matrix_count(config: BenchConfig, bench_kind: str) -> int:
    if bench_kind in (EXPERT_GROUPED_BANK_CONSUMER_BENCH, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH):
        return config.layers * 2 * config.num_experts
    if bench_kind in (
        ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_STACK_NS_BENCH,
        ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
    ):
        return config.layers * len(production_muonh_2d_shapes(config))
    if bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
        return 0
    if bench_kind == FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH:
        per_layer_2d = len(production_muonh_2d_shapes(config))
        per_layer_expert = sum(shape[0] for shape in synthetic_shapes(config).values())
        return config.layers * (per_layer_2d + per_layer_expert)
    if is_full_production_muonh_bench(bench_kind):
        per_layer_2d = len(production_muonh_2d_shapes(config))
        per_layer_expert = sum(shape[0] for shape in synthetic_shapes(config).values())
        return config.layers * (per_layer_2d + per_layer_expert)
    if bench_kind in (
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ):
        return sum(grouped_expert_group_sizes_for_bench(config, bench_kind)) * sum(
            shape[0] for shape in synthetic_shapes(config).values()
        )
    if is_tree_update_bench(bench_kind) or bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES:
        group_size = config.layers
    else:
        group_size = padded_ns4d_group_size(config, bench_kind)
    return group_size * sum(shape[0] for shape in synthetic_shapes(config).values())


def estimated_expert_update_global_bytes(config: BenchConfig) -> int:
    dtype = dtype_from_name(config.dtype)
    bytes_per_element = int(np.dtype(dtype).itemsize)
    elements_per_layer = sum(math.prod(shape) for shape in synthetic_shapes(config).values())
    return config.layers * elements_per_layer * bytes_per_element


def estimated_boundary_byte_estimates(config: BenchConfig, bench_kind: str) -> dict[str, float] | None:
    if not is_expert_fsdp_grouped_boundary_bench(bench_kind):
        return None
    global_bytes = estimated_expert_update_global_bytes(config)
    expert_axis = max(1, config.expert_axis)
    data_axis = max(1, config.data_axis)
    group_axis = max(1, ns4d_axis_size(config))
    grouped_input_per_device = global_bytes / (expert_axis * group_axis)
    fsdp_output_per_device = global_bytes / (expert_axis * data_axis)
    all_gather_slice_peak_per_device = global_bytes / expert_axis
    return {
        "global_update_bytes": float(global_bytes),
        "grouped_input_per_device_bytes": float(grouped_input_per_device),
        "fsdp_output_per_device_bytes": float(fsdp_output_per_device),
        "all_gather_slice_peak_per_device_bytes": float(all_gather_slice_peak_per_device),
        "fsdp_output_to_grouped_input_ratio": float(fsdp_output_per_device / grouped_input_per_device),
        "all_gather_slice_peak_to_grouped_input_ratio": float(
            all_gather_slice_peak_per_device / grouped_input_per_device
        ),
    }


def estimated_tflops(flops: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return flops / seconds / 1e12


def percent_h100_bf16_peak(tflops: float | None, devices: int) -> float | None:
    if tflops is None:
        return None
    return 100 * tflops / (devices * NOMINAL_H100_BF16_DENSE_TFLOPS)


def lower_tree_update(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    hlo_output: Path | None,
    abstract_mesh_enabled: bool,
) -> HloSummary:
    specs = synthetic_specs(mesh, config)
    expected_spec = P("expert", None, None)
    expected_momentum_spec = momentum_expected_spec(config)
    assert_expert_stack_sharding(specs, expected_spec, "params/update specs")

    if bench_kind == MUONH_UPDATE_BENCH:
        transform = build_muonh_transform(config)
        update_step = jax.jit(update_step_factory(config))
        update_label = "MuonH update specs"
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            state_spec = jax.eval_shape(transform.init, specs)
            next_update_specs, next_state_spec = jax.eval_shape(update_step, specs, specs, state_spec)
            lowered = update_step.lower(specs, specs, state_spec)
        assert_expert_stack_sharding_if_present(state_spec.momentum_buffer, expected_momentum_spec, "momentum specs")
        assert_expert_stack_sharding_if_present(
            next_state_spec.momentum_buffer, expected_momentum_spec, "next momentum specs"
        )
    elif bench_kind == MUON_DIRECTION_BENCH:
        transform = build_direction_transform(config)
        update_step = jax.jit(direction_step_factory(config))
        update_label = "Muon direction update specs"
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            state_spec = jax.eval_shape(transform.init, specs)
            next_update_specs, next_state_spec = jax.eval_shape(update_step, specs, specs, state_spec)
            lowered = update_step.lower(specs, specs, state_spec)
        assert_expert_stack_sharding_if_present(state_spec.momentum_buffer, expected_momentum_spec, "momentum specs")
        assert_expert_stack_sharding_if_present(
            next_state_spec.momentum_buffer, expected_momentum_spec, "next momentum specs"
        )
    elif bench_kind == HYPERBALL_ONLY_BENCH:
        update_step = jax.jit(hyperball_step_factory(config))
        update_label = "MuonH hyperball-only update specs"
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            next_update_specs = jax.eval_shape(update_step, specs, specs)
            lowered = update_step.lower(specs, specs)
    else:
        raise ValueError(f"Unsupported tree update benchmark kind: {bench_kind!r}")

    assert_expert_stack_sharding_if_present(next_update_specs, expected_spec, update_label)
    hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    maybe_write_text(hlo_output, hlo_text)
    return summarize_hlo(hlo_text)


def lower_ns4d(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    hlo_output: Path | None,
    abstract_mesh_enabled: bool,
    allow_boundary_collectives: bool,
) -> HloSummary:
    if is_full_production_muonh_bench(bench_kind):
        specs = synthetic_full_production_muonh_specs(mesh, config, bench_kind)
    elif is_expert_fsdp_grouped_bench(bench_kind):
        specs = synthetic_fsdp_expert_specs(mesh, config)
    elif is_expert_only_grouped_muonh_bench(bench_kind):
        specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
        specs = synthetic_productionish_grouped_expert_specs(mesh, config, bench_kind)
    elif bench_kind in (
        EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_GROUPED_LAYER_SLICE_BENCH,
        EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
        EXPERT_GROUPED_BANK_CONSUMER_BENCH,
        EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    ):
        specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    else:
        specs = synthetic_ns4d_specs(mesh, config, bench_kind)
    input_spec = ns4d_input_sharding(mesh, config, bench_kind).spec
    result_sharding = ns4d_result_sharding(mesh, config, bench_kind)
    if is_expert_fsdp_grouped_bench(bench_kind):
        assert_expert_fsdp_sharding(specs, "expert FSDP input specs")
    else:
        assert_grouped_or_uniform_ns4d_sharding(specs, mesh, config, bench_kind, input_spec, "NS4D input specs")
    if ns4d_bench_uses_grouped_params(bench_kind):
        if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            activation_specs = synthetic_grouped_moe_mlp_consumer_input_specs(mesh, config, bench_kind)
            update_step = jax.jit(grouped_moe_mlp_consumer_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, grouped_specs, activation_specs)
                lowered = update_step.lower(grouped_specs, activation_specs)
            assert_grouped_expert_sharding(grouped_specs, mesh, config, bench_kind, "grouped expert MoE params")
            assert_grouped_moe_consumer_sharding(
                activation_specs,
                mesh,
                config,
                bench_kind,
                "grouped MoE consumer inputs",
            )
            assert_grouped_moe_consumer_sharding(
                result_specs,
                mesh,
                config,
                bench_kind,
                "grouped MoE consumer result",
            )
            lower_args = None
        elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            activation_specs = synthetic_grouped_expert_consumer_input_specs(mesh, config, bench_kind)
            update_step = jax.jit(grouped_expert_bank_consumer_step_factory(config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, grouped_specs, activation_specs)
                lowered = update_step.lower(grouped_specs, activation_specs)
            assert_grouped_expert_sharding(grouped_specs, mesh, config, bench_kind, "grouped expert bank params")
            assert_ns4d_sharding(activation_specs, input_spec, "grouped expert bank consumer inputs")
            if result_sharding is not None:
                assert_ns4d_sharding(result_specs, result_sharding.spec, "grouped expert bank consumer result")
            lower_args = None
        elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            if bench_kind == EXPERT_GROUPED_LAYER_SLICE_BENCH:
                update_step = jax.jit(expert_grouped_layer_slice_step_factory(mesh, config))
            else:
                update_step = jax.jit(expert_grouped_single_layer_slice_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, grouped_specs)
                lowered = update_step.lower(grouped_specs)
            assert_grouped_expert_sharding(grouped_specs, mesh, config, bench_kind, "grouped expert bank params")
            assert_expert_ep_sharding(result_specs, "grouped expert layer slice result")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_apply_boundary_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs, update_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs, mesh, config, bench_kind, "expert FSDP boundary grouped updates"
            )
            assert_expert_fsdp_sharding(update_specs, "expert FSDP boundary restored updates")
            assert_expert_fsdp_sharding(result_specs, "expert FSDP boundary apply result")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_restore_boundary_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs, mesh, config, bench_kind, "expert FSDP restore-only grouped updates"
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP restore-only restored updates")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_target_restore_boundary_step_factory(config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs, mesh, config, bench_kind, "expert FSDP target restore grouped updates"
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP target restore restored updates")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_target_apply_boundary_step_factory(config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP target apply grouped updates",
            )
            assert_grouped_expert_target_fsdp_sharding(
                result_specs,
                "expert FSDP target apply grouped result",
            )
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_explicit_restore_boundary_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP explicit restore grouped updates",
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP explicit restore updates")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_explicit_apply_boundary_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP explicit apply grouped updates",
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP explicit apply result")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_explicit_a2a_apply_boundary_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP explicit a2a apply grouped updates",
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP explicit a2a apply result")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_updates_muonh_updates_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP grouped-updates MuonH update-only grouped updates",
            )
            assert_expert_fsdp_sharding(result_specs, "expert FSDP grouped-updates MuonH update-only updates")
            lower_args = None
        elif bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH:
            grouped_specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
            update_step = jax.jit(expert_fsdp_grouped_updates_muonh_apply_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                result_specs, update_specs = jax.eval_shape(update_step, specs, grouped_specs)
                lowered = update_step.lower(specs, grouped_specs)
            assert_grouped_expert_sharding(
                grouped_specs,
                mesh,
                config,
                bench_kind,
                "expert FSDP grouped-updates MuonH grouped updates",
            )
            assert_expert_fsdp_sharding(update_specs, "expert FSDP grouped-updates MuonH restored updates")
            assert_expert_fsdp_sharding(result_specs, "expert FSDP grouped-updates MuonH apply result")
            lower_args = None
        elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
            optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
            update_step = jax.jit(expert_fsdp_grouped_muonh_optimizer_apply_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                result_specs, _next_state_specs, update_specs = jax.eval_shape(update_step, specs, specs, state_specs)
                lowered = update_step.lower(specs, specs, state_specs)
            assert_expert_fsdp_sharding(update_specs, "expert FSDP grouped MuonH restored updates")
            assert_expert_fsdp_sharding(result_specs, "expert FSDP grouped MuonH apply result")
            lower_args = None
        elif is_full_production_muonh_bench(bench_kind):
            if bench_kind == FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH:
                optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                update_step = jax.jit(full_production_grouped_2d_direct_apply_timing_step_factory(config))
            else:
                group_2d_muonh = groups_2d_muonh_leaves(bench_kind)
                optimizer = build_full_production_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)
                update_step = jax.jit(
                    full_production_muonh_optimizer_apply_step_factory(config, group_2d_muonh=group_2d_muonh)
                )
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                eval_result = jax.eval_shape(update_step, specs, specs, state_specs)
                if bench_kind == FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH:
                    result_specs, _next_state_specs = eval_result
                    update_specs = specs
                else:
                    result_specs, _next_state_specs, update_specs = eval_result
                lowered = update_step.lower(specs, specs, state_specs)
            assert_grouped_expert_sharding(
                update_specs, mesh, config, bench_kind, "full production grouped expert updates"
            )
            lower_args = None
        elif is_expert_only_grouped_muonh_bench(bench_kind):
            optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
            update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config, use_hyperball=True))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                result_specs, _next_state_specs, update_specs = jax.eval_shape(update_step, specs, specs, state_specs)
                lowered = update_step.lower(specs, specs, state_specs)
            assert_grouped_expert_sharding(update_specs, mesh, config, bench_kind, "expert-only grouped MuonH updates")
            lower_args = None
        elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
            use_hyperball = bench_kind == EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH
            optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=use_hyperball)
            update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config, use_hyperball=use_hyperball))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                result_specs, _next_state_specs, update_specs = jax.eval_shape(update_step, specs, specs, state_specs)
                lowered = update_step.lower(specs, specs, state_specs)
            assert_grouped_expert_sharding(
                update_specs, mesh, config, bench_kind, "production-ish optimizer grouped expert updates"
            )
            lower_args = None
        elif bench_kind == EXPERT_GROUPED_APPLY_BOUNDARY_BENCH:
            update_step = jax.jit(grouped_expert_apply_boundary_step_factory(config))
            lower_args = (specs, specs)
        else:
            update_step = jax.jit(ns4d_grouped_apply_step_factory(config))
            lower_args = (specs, specs)
    else:
        update_step = jax.jit(ns4d_step_factory(mesh, config, bench_kind))
        lower_args = (specs,)
    if lower_args is not None:
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            result_specs = jax.eval_shape(update_step, *lower_args)
            lowered = update_step.lower(*lower_args)
    if is_expert_fsdp_grouped_bench(bench_kind):
        assert_expert_fsdp_sharding(result_specs, "expert FSDP grouped MuonH result specs")
    elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
        assert_expert_ep_sharding(result_specs, "grouped expert layer slice result specs")
    elif result_sharding is not None and ns4d_bench_returns_4d_updates(bench_kind):
        assert_grouped_or_uniform_ns4d_sharding(
            result_specs, mesh, config, bench_kind, result_sharding.spec, "NS4D result specs"
        )
    hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    maybe_write_text(hlo_output, hlo_text)
    hlo_summary = summarize_hlo(hlo_text)
    if (
        bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES
        and not is_expert_fsdp_grouped_bench(bench_kind)
        and not allow_boundary_collectives
    ):
        assert_grouped_apply_has_no_boundary_collectives(hlo_summary, "lowered NS4D grouped apply")
    if bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH and not allow_boundary_collectives:
        assert_grouped_apply_has_no_boundary_collectives(hlo_summary, "lowered grouped expert bank consumer")
    return hlo_summary


def lower_ordinary_2d_decomposition(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    hlo_output: Path | None,
    abstract_mesh_enabled: bool,
) -> HloSummary:
    if bench_kind == FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH:
        specs = synthetic_full_production_grouped_persistent_specs(mesh, config, bench_kind)
        optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
        update_step = jax.jit(full_production_grouped_2d_persistent_apply_timing_step_factory(config))
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            state_specs = jax.eval_shape(optimizer.init, specs)
            jax.eval_shape(update_step, specs, specs, state_specs)
            lowered = update_step.lower(specs, specs, state_specs)
        lower_args = None
    elif bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
        specs = synthetic_full_production_muonh_specs(mesh, config, bench_kind)
        update_step = jax.jit(full_production_apply_only_step_factory(config))
        lower_args = (specs, specs)
    else:
        if bench_kind == ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH:
            specs = synthetic_ordinary_2d_grouped_persistent_specs(mesh, config)
        else:
            specs = synthetic_ordinary_2d_muonh_specs(mesh, config)
        if bench_kind in (
            ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
            ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
            ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
            ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
        ):
            if bench_kind == ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH:
                optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                update_step = jax.jit(ordinary_2d_grouped_direct_apply_timing_step_factory(config))
            elif bench_kind == ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH:
                optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                update_step = jax.jit(ordinary_2d_grouped_persistent_apply_timing_step_factory(config))
            else:
                group_2d_muonh = bench_kind == ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH
                optimizer = build_ordinary_2d_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)
                update_step = jax.jit(
                    ordinary_2d_muonh_optimizer_apply_step_factory(config, group_2d_muonh=group_2d_muonh)
                )
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                jax.eval_shape(update_step, specs, specs, state_specs)
                lowered = update_step.lower(specs, specs, state_specs)
            lower_args = None
        elif bench_kind == ORDINARY_2D_GROUPED_STACK_NS_BENCH:
            update_step = jax.jit(grouped_2d_stack_ns_step_factory(config))
            lower_args = (specs, specs)
        elif bench_kind == ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH:
            update_step = jax.jit(grouped_2d_restore_split_step_factory(config))
            lower_args = (specs, specs)
        else:
            raise ValueError(f"Unsupported ordinary 2D decomposition benchmark kind: {bench_kind!r}")

    if lower_args is not None:
        with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
            lowered = update_step.lower(*lower_args)

    hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    maybe_write_text(hlo_output, hlo_text)
    return summarize_hlo(hlo_text)


def block_until_ready_tree(value: Any) -> None:
    jax.block_until_ready(value)


def time_tree_update(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    warmup: int,
    iters: int,
    compile_only: bool,
    abstract_mesh_enabled: bool,
) -> TimingSummary:
    shardings = synthetic_shardings(mesh, config)
    with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
        params = make_array_tree(config, shardings, seed=0)
        updates = make_array_tree(config, shardings, seed=1)
        expected_spec = P("expert", None, None)
        expected_momentum_spec = momentum_expected_spec(config)
        assert_expert_stack_sharding(params, expected_spec, "params")
        assert_expert_stack_sharding(updates, expected_spec, "updates")

        if bench_kind == MUONH_UPDATE_BENCH:
            transform = build_muonh_transform(config)
            state = transform.init(params)
            block_until_ready_tree((params, updates, state))
            assert_expert_stack_sharding(state.momentum_buffer, expected_momentum_spec, "momentum")
            update_step = jax.jit(update_step_factory(config))
            compile_start = time.perf_counter()
            compiled = update_step.lower(params, updates, state).compile()
        elif bench_kind == MUON_DIRECTION_BENCH:
            transform = build_direction_transform(config)
            state = transform.init(params)
            block_until_ready_tree((params, updates, state))
            assert_expert_stack_sharding(state.momentum_buffer, expected_momentum_spec, "momentum")
            update_step = jax.jit(direction_step_factory(config))
            compile_start = time.perf_counter()
            compiled = update_step.lower(params, updates, state).compile()
        elif bench_kind == HYPERBALL_ONLY_BENCH:
            state = None
            block_until_ready_tree((params, updates))
            update_step = jax.jit(hyperball_step_factory(config))
            compile_start = time.perf_counter()
            compiled = update_step.lower(params, updates).compile()
        else:
            raise ValueError(f"Unsupported tree update benchmark kind: {bench_kind!r}")
        compile_seconds = time.perf_counter() - compile_start
        compiled_hlo = summarize_hlo(compiled.as_text())
        if compile_only:
            return TimingSummary(compile_seconds, compiled_hlo, [], None, None, None, None)

        for _ in range(warmup):
            if bench_kind == HYPERBALL_ONLY_BENCH:
                next_updates = compiled(params, updates)
                block_until_ready_tree(next_updates)
            else:
                next_updates, state = compiled(params, updates, state)
                block_until_ready_tree((next_updates, state))
            assert_runtime_expert_stack_sharding(next_updates, expected_spec, f"warmup {bench_kind} updates", config)
            if state is not None:
                assert_runtime_expert_stack_sharding(
                    state.momentum_buffer, expected_momentum_spec, "warmup momentum", config
                )
            if bench_kind != HYPERBALL_ONLY_BENCH:
                updates = next_updates

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            if bench_kind == HYPERBALL_ONLY_BENCH:
                next_updates = compiled(params, updates)
                block_until_ready_tree(next_updates)
            else:
                next_updates, state = compiled(params, updates, state)
                block_until_ready_tree((next_updates, state))
            assert_runtime_expert_stack_sharding(next_updates, expected_spec, f"{bench_kind} updates", config)
            if state is not None:
                assert_runtime_expert_stack_sharding(state.momentum_buffer, expected_momentum_spec, "momentum", config)
            times.append(time.perf_counter() - start)
            if bench_kind != HYPERBALL_ONLY_BENCH:
                updates = next_updates

    return TimingSummary(
        compile_seconds=compile_seconds,
        compiled_hlo=compiled_hlo,
        times=times,
        median_seconds=statistics.median(times) if times else None,
        mean_seconds=statistics.mean(times) if times else None,
        min_seconds=min(times) if times else None,
        stdev_seconds=statistics.stdev(times) if len(times) > 1 else None,
    )


def time_ns4d(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    warmup: int,
    iters: int,
    compile_only: bool,
    abstract_mesh_enabled: bool,
    allow_boundary_collectives: bool,
) -> TimingSummary:
    with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
        optimizer_state = None
        if is_full_production_muonh_bench(bench_kind):
            updates = make_full_production_muonh_array_tree(mesh, config, bench_kind, seed=1)
        elif is_expert_fsdp_grouped_boundary_bench(bench_kind) or is_expert_fsdp_grouped_updates_muonh_bench(bench_kind):
            updates = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
            updates = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)
        elif is_expert_only_grouped_muonh_bench(bench_kind):
            updates = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
            updates = make_productionish_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        elif bench_kind in (
            EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
            EXPERT_GROUPED_LAYER_SLICE_BENCH,
            EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
            EXPERT_GROUPED_BANK_CONSUMER_BENCH,
            EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
        ):
            if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                updates = make_grouped_moe_mlp_consumer_input_tree(mesh, config, bench_kind, seed=1)
            elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
                updates = make_grouped_expert_consumer_input_tree(mesh, config, bench_kind, seed=1)
            else:
                updates = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        else:
            updates = make_ns4d_array_tree(mesh, config, bench_kind, seed=1)
        input_spec = ns4d_input_sharding(mesh, config, bench_kind).spec
        result_sharding = ns4d_result_sharding(mesh, config, bench_kind)
        if is_expert_fsdp_grouped_boundary_bench(bench_kind) or is_expert_fsdp_grouped_updates_muonh_bench(bench_kind):
            assert_grouped_expert_sharding(updates, mesh, config, bench_kind, "expert FSDP grouped updates")
        elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
            assert_expert_fsdp_sharding(updates, "expert FSDP updates")
        elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
            assert_ns4d_sharding(updates, input_spec, "grouped expert bank consumer inputs")
        elif bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
            assert_grouped_moe_consumer_sharding(
                updates,
                mesh,
                config,
                bench_kind,
                "grouped MoE consumer inputs",
            )
        else:
            assert_grouped_or_uniform_ns4d_sharding(updates, mesh, config, bench_kind, input_spec, "NS4D updates")
        if ns4d_bench_uses_grouped_params(bench_kind):
            if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                params = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=0)
                update_step = jax.jit(grouped_moe_mlp_consumer_step_factory(mesh, config))
                lower_args = (params, updates)
            elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
                params = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=0)
                update_step = jax.jit(grouped_expert_bank_consumer_step_factory(config))
                lower_args = (params, updates)
            elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                params = None
                if bench_kind == EXPERT_GROUPED_LAYER_SLICE_BENCH:
                    update_step = jax.jit(expert_grouped_layer_slice_step_factory(mesh, config))
                else:
                    update_step = jax.jit(expert_grouped_single_layer_slice_step_factory(mesh, config))
                lower_args = (updates,)
            elif bench_kind == EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_apply_boundary_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_restore_boundary_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_target_restore_boundary_timing_step_factory(config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(expert_fsdp_grouped_target_apply_boundary_timing_step_factory(config))
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH:
                params = None
                update_step = jax.jit(expert_fsdp_grouped_explicit_restore_boundary_timing_step_factory(mesh, config))
                lower_args = (updates,)
            elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_explicit_apply_boundary_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_explicit_a2a_apply_boundary_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_updates_muonh_updates_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif bench_kind == EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH:
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                update_step = jax.jit(
                    expert_fsdp_grouped_updates_muonh_apply_timing_step_factory(mesh, config),
                    donate_argnums=(0,),
                )
                lower_args = (params, updates)
            elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
                params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
                optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                optimizer_state = optimizer.init(params)
                update_step = jax.jit(
                    expert_fsdp_grouped_muonh_optimizer_apply_timing_step_factory(mesh, config),
                    donate_argnums=(0, 2),
                )
                lower_args = (params, updates, optimizer_state)
            elif is_full_production_muonh_bench(bench_kind):
                params = make_full_production_muonh_array_tree(mesh, config, bench_kind, seed=0)
                if bench_kind == FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH:
                    optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                    update_step = jax.jit(
                        full_production_grouped_2d_direct_apply_timing_step_factory(config),
                        donate_argnums=(0, 2),
                    )
                else:
                    group_2d_muonh = groups_2d_muonh_leaves(bench_kind)
                    optimizer = build_full_production_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)
                    update_step = jax.jit(
                        full_production_muonh_optimizer_apply_timing_step_factory(
                            config,
                            group_2d_muonh=group_2d_muonh,
                        ),
                        donate_argnums=(0, 2),
                    )
                optimizer_state = optimizer.init(params)
                lower_args = (params, updates, optimizer_state)
            elif is_expert_only_grouped_muonh_bench(bench_kind):
                params = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=0)
                optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
                optimizer_state = optimizer.init(params)
                update_step = jax.jit(
                    grouped_expert_optimizer_apply_timing_step_factory(config, use_hyperball=True),
                    donate_argnums=(0, 2),
                )
                lower_args = (params, updates, optimizer_state)
            elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
                params = make_productionish_grouped_expert_array_tree(mesh, config, bench_kind, seed=0)
                use_hyperball = bench_kind == EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH
                optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=use_hyperball)
                optimizer_state = optimizer.init(params)
                update_step = jax.jit(
                    grouped_expert_optimizer_apply_timing_step_factory(config, use_hyperball=use_hyperball),
                    donate_argnums=(0, 2),
                )
                lower_args = (params, updates, optimizer_state)
            elif bench_kind == EXPERT_GROUPED_APPLY_BOUNDARY_BENCH:
                params = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=0)
                update_step = jax.jit(grouped_expert_apply_boundary_step_factory(config))
                lower_args = (params, updates)
            else:
                params = make_ns4d_array_tree(mesh, config, bench_kind, seed=0)
                update_step = jax.jit(ns4d_grouped_apply_step_factory(config))
                lower_args = (params, updates)
            if bench_kind == EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH:
                pass
            elif is_expert_fsdp_grouped_bench(bench_kind):
                assert_expert_fsdp_sharding(params, "expert FSDP params")
            elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
                assert_grouped_expert_sharding(params, mesh, config, bench_kind, "grouped expert bank params")
            elif bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                assert_grouped_expert_sharding(params, mesh, config, bench_kind, "grouped expert MoE params")
            elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                assert_grouped_expert_sharding(updates, mesh, config, bench_kind, "grouped expert bank params")
            else:
                assert_grouped_or_uniform_ns4d_sharding(params, mesh, config, bench_kind, input_spec, "NS4D params")
            if bench_kind in (
                EXPERT_GROUPED_LAYER_SLICE_BENCH,
                EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
                EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
            ):
                block_until_ready_tree(updates)
            else:
                block_until_ready_tree((params, updates, optimizer_state))
        else:
            params = None
            block_until_ready_tree(updates)
            update_step = jax.jit(ns4d_step_factory(mesh, config, bench_kind))
            lower_args = (updates,)

        compile_start = time.perf_counter()
        compiled = update_step.lower(*lower_args).compile()
        compile_seconds = time.perf_counter() - compile_start
        compiled_hlo = summarize_hlo(compiled.as_text())
        if (
            bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES
            and not is_expert_fsdp_grouped_bench(bench_kind)
            and not allow_boundary_collectives
        ):
            assert_grouped_apply_has_no_boundary_collectives(compiled_hlo, "compiled NS4D grouped apply")
        if bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH and not allow_boundary_collectives:
            assert_grouped_apply_has_no_boundary_collectives(compiled_hlo, "compiled grouped expert bank consumer")
        if compile_only:
            return TimingSummary(compile_seconds, compiled_hlo, [], None, None, None, None)

        for _ in range(warmup):
            if params is None:
                next_updates = compiled(updates)
                block_until_ready_tree(next_updates)
            elif (
                is_expert_fsdp_grouped_bench(bench_kind)
                or is_expert_only_grouped_muonh_bench(bench_kind)
                or bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES
                or bench_kind
                in (
                    EXPERT_GROUPED_LAYER_SLICE_BENCH,
                    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
                    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
                    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
                )
            ):
                if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                    consumer_outputs = compiled(params, updates)
                    block_until_ready_tree(consumer_outputs)
                elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
                    consumer_outputs = compiled(params, updates)
                    block_until_ready_tree(consumer_outputs)
                elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                    params = compiled(updates)
                    block_until_ready_tree(params)
                elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
                    grouped_target_params = compiled(params, updates)
                    block_until_ready_tree(grouped_target_params)
                elif is_expert_fsdp_grouped_boundary_bench(bench_kind) or is_expert_fsdp_grouped_updates_muonh_bench(
                    bench_kind
                ):
                    params = compiled(params, updates)
                    block_until_ready_tree(params)
                else:
                    params, optimizer_state = compiled(params, updates, optimizer_state)
                    block_until_ready_tree((params, optimizer_state))
                if bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
                    assert_grouped_expert_target_fsdp_sharding(
                        grouped_target_params,
                        "warmup expert FSDP target-apply grouped params",
                    )
                elif is_expert_fsdp_grouped_bench(bench_kind):
                    assert_runtime_expert_fsdp_sharding(
                        params,
                        "warmup expert FSDP grouped params",
                        config,
                    )
                elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                    assert_expert_ep_sharding(params, "warmup grouped expert layer slices")
                elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH and result_sharding is not None:
                    assert_ns4d_sharding(
                        consumer_outputs,
                        result_sharding.spec,
                        "warmup grouped expert bank consumer outputs",
                    )
                elif bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                    assert_grouped_moe_consumer_sharding(
                        consumer_outputs,
                        mesh,
                        config,
                        bench_kind,
                        "warmup grouped MoE consumer outputs",
                    )
                elif result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        params,
                        mesh,
                        config,
                        bench_kind,
                        result_sharding.spec,
                        "warmup production-ish grouped params",
                    )
                next_updates = None
            else:
                params = compiled(params, updates)
                block_until_ready_tree(params)
                if result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        params,
                        mesh,
                        config,
                        bench_kind,
                        result_sharding.spec,
                        "warmup NS4D grouped apply result",
                    )
                next_updates = None
            if next_updates is not None and ns4d_bench_returns_4d_updates(bench_kind):
                if result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        next_updates, mesh, config, bench_kind, result_sharding.spec, "warmup NS4D updates"
                    )
                updates = next_updates

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            if params is None:
                next_updates = compiled(updates)
                block_until_ready_tree(next_updates)
            elif (
                is_expert_fsdp_grouped_bench(bench_kind)
                or is_expert_only_grouped_muonh_bench(bench_kind)
                or bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES
                or bench_kind
                in (
                    EXPERT_GROUPED_LAYER_SLICE_BENCH,
                    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
                    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
                    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
                )
            ):
                if bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                    consumer_outputs = compiled(params, updates)
                    block_until_ready_tree(consumer_outputs)
                elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH:
                    consumer_outputs = compiled(params, updates)
                    block_until_ready_tree(consumer_outputs)
                elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                    params = compiled(updates)
                    block_until_ready_tree(params)
                elif bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
                    grouped_target_params = compiled(params, updates)
                    block_until_ready_tree(grouped_target_params)
                elif is_expert_fsdp_grouped_boundary_bench(bench_kind) or is_expert_fsdp_grouped_updates_muonh_bench(
                    bench_kind
                ):
                    params = compiled(params, updates)
                    block_until_ready_tree(params)
                else:
                    params, optimizer_state = compiled(params, updates, optimizer_state)
                    block_until_ready_tree((params, optimizer_state))
                if bench_kind == EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH:
                    assert_grouped_expert_target_fsdp_sharding(
                        grouped_target_params,
                        "expert FSDP target-apply grouped params",
                    )
                elif is_expert_fsdp_grouped_bench(bench_kind):
                    assert_runtime_expert_fsdp_sharding(
                        params,
                        "expert FSDP grouped params",
                        config,
                    )
                elif bench_kind in (EXPERT_GROUPED_LAYER_SLICE_BENCH, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH):
                    assert_expert_ep_sharding(params, "grouped expert layer slices")
                elif bench_kind == EXPERT_GROUPED_BANK_CONSUMER_BENCH and result_sharding is not None:
                    assert_ns4d_sharding(
                        consumer_outputs,
                        result_sharding.spec,
                        "grouped expert bank consumer outputs",
                    )
                elif bench_kind == EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH:
                    assert_grouped_moe_consumer_sharding(
                        consumer_outputs,
                        mesh,
                        config,
                        bench_kind,
                        "grouped MoE consumer outputs",
                    )
                elif result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        params,
                        mesh,
                        config,
                        bench_kind,
                        result_sharding.spec,
                        "production-ish grouped params",
                    )
                next_updates = None
            else:
                params = compiled(params, updates)
                block_until_ready_tree(params)
                if result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        params,
                        mesh,
                        config,
                        bench_kind,
                        result_sharding.spec,
                        "NS4D grouped apply result",
                    )
                next_updates = None
            times.append(time.perf_counter() - start)
            if next_updates is not None and ns4d_bench_returns_4d_updates(bench_kind):
                if result_sharding is not None:
                    assert_grouped_or_uniform_ns4d_sharding(
                        next_updates, mesh, config, bench_kind, result_sharding.spec, "NS4D updates"
                    )
                updates = next_updates

    return TimingSummary(
        compile_seconds=compile_seconds,
        compiled_hlo=compiled_hlo,
        times=times,
        median_seconds=statistics.median(times) if times else None,
        mean_seconds=statistics.mean(times) if times else None,
        min_seconds=min(times) if times else None,
        stdev_seconds=statistics.stdev(times) if len(times) > 1 else None,
    )


def time_ordinary_2d_decomposition(
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    warmup: int,
    iters: int,
    compile_only: bool,
    abstract_mesh_enabled: bool,
) -> TimingSummary:
    with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
        optimizer_state = None
        if bench_kind == FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH:
            params = make_full_production_grouped_persistent_array_tree(mesh, config, bench_kind, seed=0)
            updates = make_full_production_grouped_persistent_array_tree(mesh, config, bench_kind, seed=1)
            optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
            optimizer_state = optimizer.init(params)
            update_step = jax.jit(
                full_production_grouped_2d_persistent_apply_timing_step_factory(config),
                donate_argnums=(0, 2),
            )
            lower_args = (params, updates, optimizer_state)
            block_until_ready_tree((params, updates, optimizer_state))
        elif bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
            params = make_full_production_muonh_array_tree(mesh, config, bench_kind, seed=0)
            updates = make_full_production_muonh_array_tree(mesh, config, bench_kind, seed=1)
            update_step = jax.jit(full_production_apply_only_step_factory(config), donate_argnums=(0,))
            lower_args = (params, updates)
            block_until_ready_tree((params, updates))
        else:
            if bench_kind == ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH:
                params = make_ordinary_2d_grouped_persistent_array_tree(mesh, config, seed=0)
                updates = make_ordinary_2d_grouped_persistent_array_tree(mesh, config, seed=1)
            else:
                params = make_ordinary_2d_muonh_array_tree(mesh, config, seed=0)
                updates = make_ordinary_2d_muonh_array_tree(mesh, config, seed=1)
            if bench_kind in (
                ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
                ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
            ):
                if bench_kind == ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH:
                    optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                    update_step = jax.jit(
                        ordinary_2d_grouped_direct_apply_timing_step_factory(config),
                        donate_argnums=(0, 2),
                    )
                elif bench_kind == ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH:
                    optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
                    update_step = jax.jit(
                        ordinary_2d_grouped_persistent_apply_timing_step_factory(config),
                        donate_argnums=(0, 2),
                    )
                else:
                    group_2d_muonh = bench_kind == ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH
                    optimizer = build_ordinary_2d_muonh_optimizer(config, group_2d_muonh=group_2d_muonh)
                    update_step = jax.jit(
                        ordinary_2d_muonh_optimizer_apply_timing_step_factory(
                            config,
                            group_2d_muonh=group_2d_muonh,
                        ),
                        donate_argnums=(0, 2),
                    )
                optimizer_state = optimizer.init(params)
                lower_args = (params, updates, optimizer_state)
                block_until_ready_tree((params, updates, optimizer_state))
            elif bench_kind == ORDINARY_2D_GROUPED_STACK_NS_BENCH:
                update_step = jax.jit(grouped_2d_stack_ns_step_factory(config))
                lower_args = (params, updates)
                block_until_ready_tree((params, updates))
            elif bench_kind == ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH:
                update_step = jax.jit(grouped_2d_restore_split_step_factory(config))
                lower_args = (params, updates)
                block_until_ready_tree((params, updates))
            else:
                raise ValueError(f"Unsupported ordinary 2D decomposition benchmark kind: {bench_kind!r}")

        compile_start = time.perf_counter()
        compiled = update_step.lower(*lower_args).compile()
        compile_seconds = time.perf_counter() - compile_start
        compiled_hlo = summarize_hlo(compiled.as_text())
        if compile_only:
            return TimingSummary(compile_seconds, compiled_hlo, [], None, None, None, None)

        for _ in range(warmup):
            if bench_kind in (
                ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
                ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
            ):
                params, optimizer_state = compiled(params, updates, optimizer_state)
                block_until_ready_tree((params, optimizer_state))
            elif bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
                params = compiled(params, updates)
                block_until_ready_tree(params)
            else:
                next_updates = compiled(params, updates)
                block_until_ready_tree(next_updates)

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            if bench_kind in (
                ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
                ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
            ):
                params, optimizer_state = compiled(params, updates, optimizer_state)
                block_until_ready_tree((params, optimizer_state))
            elif bench_kind == FULL_PRODUCTION_APPLY_ONLY_BENCH:
                params = compiled(params, updates)
                block_until_ready_tree(params)
            else:
                next_updates = compiled(params, updates)
                block_until_ready_tree(next_updates)
            times.append(time.perf_counter() - start)

    return TimingSummary(
        compile_seconds=compile_seconds,
        compiled_hlo=compiled_hlo,
        times=times,
        median_seconds=statistics.median(times) if times else None,
        mean_seconds=statistics.mean(times) if times else None,
        min_seconds=min(times) if times else None,
        stdev_seconds=statistics.stdev(times) if len(times) > 1 else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--ns4d-group-size",
        type=int,
        help="Override the pure NS4D group axis length without changing production tree layer count.",
    )
    parser.add_argument(
        "--ns4d-group-axis",
        default="data",
        choices=("data", "replica_dcn", "replica_dcn,data", "data,replica_dcn", "none"),
        help="Mesh axis used for pure NS4D group-axis sharding in data-sharded NS4D bench kinds.",
    )
    parser.add_argument("--hidden-dim", type=int, default=MAY_HIDDEN_DIM)
    parser.add_argument("--intermediate-dim", type=int, default=MAY_INTERMEDIATE_DIM)
    parser.add_argument("--num-experts", type=int, default=MAY_NUM_EXPERTS)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "bfloat16", "fp32", "float32", "fp16", "float16"))
    parser.add_argument(
        "--ns-compute-dtype",
        default="input",
        choices=("input", "bf16", "bfloat16", "fp32", "float32", "fp16", "float16"),
        help="Harness-only Newton-Schulz compute dtype; casts NS inputs and restores the original output dtype.",
    )
    parser.add_argument("--backend-steps", type=int, default=MAY_BACKEND_STEPS)
    parser.add_argument("--sweep-backend-steps", help="Comma-separated backend step counts, e.g. 1,5.")
    parser.add_argument("--orthogonalization-layout", default=STACK_BATCH_SHARDED)
    parser.add_argument("--max-grouped-stack-size", type=int, default=DEFAULT_MAX_GROUPED_STACK_SIZE)
    parser.add_argument("--sweep-max-grouped-stack-sizes", help="Comma-separated stack caps, e.g. 256,512,832.")
    parser.add_argument(
        "--grouped-expert-consumer-tokens-per-expert",
        type=int,
        default=DEFAULT_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT,
        help=(
            "Synthetic routed-token load for expert_grouped_bank_consumer. "
            "This controls the [group, expert, token, hidden] activation shape."
        ),
    )
    parser.add_argument(
        "--bench-kinds",
        default=MUONH_UPDATE_BENCH,
        help=f"Comma-separated benchmark kinds. Valid: {','.join(BENCH_KINDS)}.",
    )
    parser.add_argument("--replica-axis", type=int, default=MAY_REPLICA_AXIS)
    parser.add_argument("--data-axis", type=int, default=1)
    parser.add_argument("--expert-axis", type=int, default=MAY_EXPERT_AXIS)
    parser.add_argument("--model-axis", type=int, default=MAY_MODEL_AXIS)
    parser.add_argument("--learning-rate", type=float, default=MAY_LEARNING_RATE)
    parser.add_argument("--mode", choices=("lower", "run", "both"), default="both")
    parser.add_argument(
        "--disable-abstract-mesh",
        action="store_true",
        help="Disable JAX abstract mesh context for local one-device HLO inspection only.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--allow-boundary-collectives",
        action="store_true",
        help=(
            "Do not fail grouped apply benchmarks when compiled AG/AR/RS appear. "
            "Use only for decomposition/debug profiles; summary rows still report the counts."
        ),
    )
    parser.add_argument("--hlo-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_int_csv(raw: str | None) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}.")
    return values


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one value in {raw!r}.")
    return values


def config_from_args(args: argparse.Namespace) -> BenchConfig:
    input_dtype = dtype_from_name(args.dtype)
    if args.grouped_expert_consumer_tokens_per_expert < 1:
        raise ValueError("--grouped-expert-consumer-tokens-per-expert must be >= 1.")
    return BenchConfig(
        layers=args.layers,
        ns4d_group_size=args.ns4d_group_size,
        ns4d_group_axis=args.ns4d_group_axis,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        dtype=dtype_name(input_dtype),
        backend_steps=args.backend_steps,
        orthogonalization_layout=args.orthogonalization_layout,
        max_grouped_stack_size=args.max_grouped_stack_size,
        replica_axis=args.replica_axis,
        data_axis=args.data_axis,
        expert_axis=args.expert_axis,
        model_axis=args.model_axis,
        learning_rate=args.learning_rate,
        ns_compute_dtype=ns_compute_dtype_name(args.ns_compute_dtype, input_dtype),
        grouped_expert_consumer_tokens_per_expert=args.grouped_expert_consumer_tokens_per_expert,
    )


def config_variants_from_args(args: argparse.Namespace) -> list[BenchConfig]:
    base = config_from_args(args)
    caps = parse_int_csv(args.sweep_max_grouped_stack_sizes) or (base.max_grouped_stack_size,)
    backend_steps = parse_int_csv(args.sweep_backend_steps) or (base.backend_steps,)
    return [replace(base, max_grouped_stack_size=cap, backend_steps=steps) for steps in backend_steps for cap in caps]


def config_label(config: BenchConfig) -> str:
    return f"h{config.backend_steps}_cap{config.max_grouped_stack_size}"


def variant_label(config: BenchConfig, bench_kind: str) -> str:
    if bench_kind == MUONH_UPDATE_BENCH:
        return config_label(config)
    if bench_kind in (MUON_DIRECTION_BENCH, HYPERBALL_ONLY_BENCH):
        return f"{bench_kind}_{config_label(config)}"
    return f"{bench_kind}_h{config.backend_steps}"


def output_path_for_config(path: Path | None, label: str, total_configs: int) -> Path | None:
    if path is None or total_configs == 1:
        return path
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def run_config(
    args: argparse.Namespace,
    mesh: Mesh,
    config: BenchConfig,
    bench_kind: str,
    total_configs: int,
) -> dict[str, Any]:
    label = variant_label(config, bench_kind)
    ns4d_input_spec = None
    ns4d_compute_spec = None
    ns4d_result_spec = None
    ns4d_boundary = ns4d_boundary_status(config, bench_kind)
    if not is_tree_update_bench(bench_kind) and not is_ordinary_2d_decomposition_bench(bench_kind):
        ns4d_input_spec = sharding_spec_string(ns4d_input_sharding(mesh, config, bench_kind))
        ns4d_compute_spec = sharding_spec_string(ns4d_compute_sharding(mesh, config, bench_kind))
        result_sharding = ns4d_result_sharding(mesh, config, bench_kind)
        ns4d_result_spec = sharding_spec_string(result_sharding) if result_sharding is not None else None
    grouped_2d_estimates = None
    if bench_kind in (
        ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
        ORDINARY_2D_GROUPED_STACK_NS_BENCH,
        ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
    ):
        grouped_2d_estimates = [
            asdict(estimate) for estimate in estimate_grouped_2d_muonh(mesh, config, full_production_tree=False)
        ]
    elif bench_kind in (
        FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
        FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
        FULL_PRODUCTION_APPLY_ONLY_BENCH,
    ):
        grouped_2d_estimates = [
            asdict(estimate) for estimate in estimate_grouped_2d_muonh(mesh, config, full_production_tree=True)
        ]
    metadata = {
        "event": "metadata",
        "label": label,
        "bench_kind": bench_kind,
        "config": asdict(config),
        "abstract_mesh_enabled": not args.disable_abstract_mesh,
        "boundary_collectives_allowed": args.allow_boundary_collectives,
        "devices": int(np.asarray(jax.devices()).size),
        "device_kinds": sorted({getattr(device, "device_kind", "") for device in jax.devices()}),
        "synthetic_shapes": synthetic_shapes(config),
        "ns4d_group_size": ns4d_group_size(config),
        "ns4d_group_axis": ns4d_group_axis(config),
        "ns4d_padded_group_size": padded_ns4d_group_size(config, bench_kind),
        "ns4d_input_sharding_spec": ns4d_input_spec,
        "ns4d_compute_sharding_spec": ns4d_compute_spec,
        "ns4d_result_sharding_spec": ns4d_result_spec,
        "ns4d_boundary_status": ns4d_boundary,
        "grouped_expert_paths": (
            list(GROUPED_EXPERT_PATHS)
            if bench_kind
            in (
                EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
                EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
                EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                EXPERT_GROUPED_BANK_CONSUMER_BENCH,
                EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
                EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
                EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
                EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
                FULL_PRODUCTION_APPLY_ONLY_BENCH,
            )
            else None
        ),
        "full_production_muonh_paths": (
            list(FULL_PRODUCTION_MUONH_PATHS)
            if bench_kind
            in (
                FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
                FULL_PRODUCTION_APPLY_ONLY_BENCH,
            )
            else None
        ),
        "ordinary_2d_muonh_paths": (
            list(production_muonh_2d_shapes(config))
            if bench_kind
            in (
                ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                ORDINARY_2D_GROUPED_DIRECT_APPLY_BENCH,
                ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
                ORDINARY_2D_GROUPED_STACK_NS_BENCH,
                ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
            )
            else None
        ),
        "ordinary_param_paths": list(ORDINARY_PARAM_PATHS) if bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES else None,
        "grouped_expert_group_count": (
            grouped_expert_group_count(config)
            if bench_kind
            in (
                EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
                EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
                EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                EXPERT_GROUPED_BANK_CONSUMER_BENCH,
                EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
                EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
                EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
                EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
                EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
                FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
                FULL_PRODUCTION_APPLY_ONLY_BENCH,
            )
            and bench_skip_reason(config, bench_kind) is None
            else None
        ),
        "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config)],
        "grouped_2d_estimates": grouped_2d_estimates,
    }
    emit_jsonl(metadata)

    result: dict[str, Any] = {"metadata": metadata}
    skip_reason = bench_skip_reason(config, bench_kind)
    if skip_reason is not None:
        skip_payload = {"event": "skipped", "label": label, "reason": skip_reason}
        emit_jsonl(skip_payload)
        result["skipped"] = skip_payload
        return result

    if args.mode in ("lower", "both"):
        start = time.perf_counter()
        hlo_path = output_path_for_config(args.hlo_output, label, total_configs)
        if is_tree_update_bench(bench_kind):
            hlo_summary = lower_tree_update(mesh, config, bench_kind, hlo_path, not args.disable_abstract_mesh)
        elif is_ordinary_2d_decomposition_bench(bench_kind):
            hlo_summary = lower_ordinary_2d_decomposition(
                mesh,
                config,
                bench_kind,
                hlo_path,
                not args.disable_abstract_mesh,
            )
        else:
            hlo_summary = lower_ns4d(
                mesh,
                config,
                bench_kind,
                hlo_path,
                not args.disable_abstract_mesh,
                args.allow_boundary_collectives,
            )
        lower_seconds = time.perf_counter() - start
        lower_payload = {
            "event": "lowered",
            "label": label,
            "lower_seconds": lower_seconds,
            "hlo": asdict(hlo_summary),
        }
        emit_jsonl(lower_payload)
        result["lowered"] = lower_payload

    if args.mode in ("run", "both"):
        if is_tree_update_bench(bench_kind):
            timing = time_tree_update(
                mesh,
                config,
                bench_kind,
                args.warmup,
                args.iters,
                args.compile_only,
                not args.disable_abstract_mesh,
            )
        elif is_ordinary_2d_decomposition_bench(bench_kind):
            timing = time_ordinary_2d_decomposition(
                mesh,
                config,
                bench_kind,
                args.warmup,
                args.iters,
                args.compile_only,
                not args.disable_abstract_mesh,
            )
        else:
            timing = time_ns4d(
                mesh,
                config,
                bench_kind,
                args.warmup,
                args.iters,
                args.compile_only,
                not args.disable_abstract_mesh,
                args.allow_boundary_collectives,
            )
        timing_payload = {"event": "timing", "label": label, "timing": asdict(timing)}
        emit_jsonl(timing_payload)
        result["timing"] = timing_payload

    return result


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    config = result["metadata"]["config"]
    bench_kind = result["metadata"]["bench_kind"]
    devices = result["metadata"]["devices"]
    estimates = result["metadata"]["group_estimates"]
    grouped_2d_estimates = result["metadata"].get("grouped_2d_estimates") or []
    bench_config = BenchConfig(**config)
    flops = estimated_ns_dot_flops(bench_config, bench_kind)
    boundary_bytes = estimated_boundary_byte_estimates(bench_config, bench_kind) or {}
    row = {
        "label": result["metadata"]["label"],
        "bench_kind": bench_kind,
        "backend_steps": config["backend_steps"],
        "max_grouped_stack_size": config["max_grouped_stack_size"],
        "grouped_expert_consumer_tokens_per_expert": config["grouped_expert_consumer_tokens_per_expert"],
        "ns4d_group_size": result["metadata"]["ns4d_group_size"],
        "ns4d_padded_group_size": result["metadata"]["ns4d_padded_group_size"],
        "ns4d_input_sharding_spec": result["metadata"]["ns4d_input_sharding_spec"],
        "ns4d_compute_sharding_spec": result["metadata"]["ns4d_compute_sharding_spec"],
        "ns4d_result_sharding_spec": result["metadata"]["ns4d_result_sharding_spec"],
        "ns4d_boundary_status": result["metadata"]["ns4d_boundary_status"],
        "boundary_collectives_allowed": result["metadata"]["boundary_collectives_allowed"],
        "estimated_ns_dot_flops": flops,
        "estimated_matrix_count": estimated_matrix_count(bench_config, bench_kind),
        "estimated_boundary_global_update_bytes": boundary_bytes.get("global_update_bytes"),
        "estimated_boundary_grouped_input_per_device_bytes": boundary_bytes.get("grouped_input_per_device_bytes"),
        "estimated_boundary_fsdp_output_per_device_bytes": boundary_bytes.get("fsdp_output_per_device_bytes"),
        "estimated_boundary_all_gather_slice_peak_per_device_bytes": boundary_bytes.get(
            "all_gather_slice_peak_per_device_bytes"
        ),
        "estimated_boundary_fsdp_output_to_grouped_input_ratio": boundary_bytes.get(
            "fsdp_output_to_grouped_input_ratio"
        ),
        "estimated_boundary_all_gather_slice_peak_to_grouped_input_ratio": boundary_bytes.get(
            "all_gather_slice_peak_to_grouped_input_ratio"
        ),
        "grouped_expert_group_count": result["metadata"]["grouped_expert_group_count"],
        "grouped_chunks": sum(estimate["grouped_chunks"] for estimate in estimates),
        "chunks": [estimate["chunks"] for estimate in estimates],
        "group_estimates": estimates,
        "grouped_2d_chunks": sum(estimate["grouped_chunks"] for estimate in grouped_2d_estimates),
        "grouped_2d_chunk_sizes": [estimate["chunks"] for estimate in grouped_2d_estimates],
        "grouped_2d_estimates": grouped_2d_estimates,
    }
    if "skipped" in result:
        row.update({"skipped": True, "skip_reason": result["skipped"]["reason"]})
        return row
    if "lowered" in result:
        row.update(
            {
                "dot_general": result["lowered"]["hlo"]["dot_general"],
                "batched_stack_dot_general": result["lowered"]["hlo"]["batched_stack_dot_general"],
                "two_batch_axis_dot_general": result["lowered"]["hlo"]["two_batch_axis_dot_general"],
                "custom_call": result["lowered"]["hlo"]["custom_call"],
                "gpu_gemm_custom_call": result["lowered"]["hlo"]["gpu_gemm_custom_call"],
                "all_gather": result["lowered"]["hlo"]["all_gather"],
                "all_reduce": result["lowered"]["hlo"]["all_reduce"],
                "reduce_scatter": result["lowered"]["hlo"]["reduce_scatter"],
                "all_to_all": result["lowered"]["hlo"]["all_to_all"],
                "collective_permute": result["lowered"]["hlo"]["collective_permute"],
                "lower_seconds": result["lowered"]["lower_seconds"],
            }
        )
    if "timing" in result:
        timing = result["timing"]["timing"]
        compiled_hlo = timing["compiled_hlo"]
        mean_tflops = estimated_tflops(flops, timing["mean_seconds"])
        median_tflops = estimated_tflops(flops, timing["median_seconds"])
        row.update(
            {
                "compile_seconds": timing["compile_seconds"],
                "compiled_hlo_dot_general": compiled_hlo["dot_general"] if compiled_hlo else None,
                "compiled_hlo_two_batch_axis_dot_general": (
                    compiled_hlo["two_batch_axis_dot_general"] if compiled_hlo else None
                ),
                "compiled_hlo_custom_call": compiled_hlo["custom_call"] if compiled_hlo else None,
                "compiled_hlo_gpu_gemm_custom_call": compiled_hlo["gpu_gemm_custom_call"] if compiled_hlo else None,
                "compiled_hlo_all_gather": compiled_hlo["all_gather"] if compiled_hlo else None,
                "compiled_hlo_all_reduce": compiled_hlo["all_reduce"] if compiled_hlo else None,
                "compiled_hlo_reduce_scatter": compiled_hlo["reduce_scatter"] if compiled_hlo else None,
                "compiled_hlo_all_to_all": compiled_hlo["all_to_all"] if compiled_hlo else None,
                "compiled_hlo_collective_permute": compiled_hlo["collective_permute"] if compiled_hlo else None,
                "median_seconds": timing["median_seconds"],
                "mean_seconds": timing["mean_seconds"],
                "min_seconds": timing["min_seconds"],
                "mean_estimated_tflops": mean_tflops,
                "median_estimated_tflops": median_tflops,
                "mean_h100_bf16_peak_pct": percent_h100_bf16_peak(mean_tflops, devices),
                "median_h100_bf16_peak_pct": percent_h100_bf16_peak(median_tflops, devices),
            }
        )
    return row


def main() -> None:
    args = parse_args()
    configs = config_variants_from_args(args)
    bench_kinds = parse_str_csv(args.bench_kinds)
    invalid_bench_kinds = sorted(set(bench_kinds) - set(BENCH_KINDS))
    if invalid_bench_kinds:
        raise ValueError(f"Unknown benchmark kinds {invalid_bench_kinds}; expected one of {BENCH_KINDS}.")
    for config in configs:
        require_model_axis_one(config)
    first_config = configs[0]
    mesh = create_mesh(
        first_config.replica_axis,
        first_config.data_axis,
        first_config.expert_axis,
        first_config.model_axis,
    )
    total_variants = len(configs) * len(bench_kinds)
    results = [
        run_config(args, mesh, config, bench_kind, total_variants) for config in configs for bench_kind in bench_kinds
    ]
    summary_rows = [summary_row(result) for result in results]
    emit_jsonl({"event": "summary_table", "rows": summary_rows})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"results": results, "summary_table": summary_rows},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_jsonl({"event": "failed", "error": repr(exc)})
        raise
