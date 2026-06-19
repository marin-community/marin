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
from jax import lax
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, reshard, use_abstract_mesh
from jax.sharding import PartitionSpec as P
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
EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH = "expert_grouped_optimizer_apply"
EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "expert_grouped_muonh_optimizer_apply"
EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH = "expert_only_grouped_muonh_optimizer_apply"
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
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
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
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
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
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_DIRECT_APPLY_BENCH,
    FULL_PRODUCTION_APPLY_ONLY_BENCH,
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
    if bench_kind == NS4D_DATA_RESHARD_RESTORE_BENCH:
        return NamedSharding(mesh, P("expert", None, None))
    if bench_kind == EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return expert_param_sharding(mesh)
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
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
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
    dtype = numpy_dtype_from_name(config.dtype)

    def make_leaf(path: str, spec: jax.ShapeDtypeStruct) -> jax.Array:
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
                    lambda update: newtonschulz_4d_dotonly_einsum(update, config.backend_steps),
                    updates,
                )
            elif bench_kind == NS4D_DOTONLY_MATMUL_BENCH:
                next_updates = jax.tree.map(
                    lambda update: newtonschulz_4d_dotonly_matmul(update, config.backend_steps),
                    updates,
                )
            elif bench_kind == NS4D_DOTONLY_LAX_DOT_GENERAL_BENCH:
                next_updates = jax.tree.map(
                    lambda update: newtonschulz_4d_dotonly_lax_dot_general(update, config.backend_steps),
                    updates,
                )
            else:
                next_updates = jax.tree.map(
                    lambda update: zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON),
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
                lambda update: zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON),
                updates,
            )
            next_updates = jax.tree.map(lambda update: scale_ns4d_update(update, config), next_updates)
            return jax.tree.map(lambda param, update: param + update, params, next_updates)

    return update_step


def grouped_expert_apply_boundary_step_factory(config: BenchConfig):
    def update_step(params, updates):
        with jax.named_scope("muon_update_bench/expert_grouped_apply_boundary_step"):
            next_updates = jax.tree.map(
                lambda update: zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON),
                updates,
            )
            next_updates = jax.tree.map(lambda update: scale_ns4d_update(update, config), next_updates)
            with jax.named_scope("muon_update_bench/expert_grouped_apply_boundary/optax_apply_updates"):
                return optax.apply_updates(params, next_updates)

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
                update = zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON)
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
                stacked_params = jnp.stack(
                    [
                        params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                        for layer_index in range(group_slice.start, group_slice.stop)
                    ],
                    axis=0,
                )
            if padded_group_size != valid_group_size:
                with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/pad_group_axis"):
                    pad_width = [(0, padded_group_size - valid_group_size), *[(0, 0) for _ in stacked_updates.shape[1:]]]
                    stacked_updates = jnp.pad(stacked_updates, pad_width)
                    stacked_params = jnp.pad(stacked_params, pad_width)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/reshard_grouped_compute"):
                stacked_updates = reshard(stacked_updates, compute_sharding)
                stacked_params = reshard(stacked_params, compute_sharding)
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/grouped_ns_hyperball"):
                direction = zeropower_via_newtonschulz_4d(
                    stacked_updates,
                    config.backend_steps,
                    MAY_MUON_EPSILON,
                )
                direction = scale_ns4d_direction(direction)
                grouped_updates = grouped_4d_hyperball_update(stacked_params, direction, config)
            grouped_updates = _restore_grouped_expert_update_for_split(
                mesh,
                grouped_updates,
                valid_group_size,
                compute_sharding,
            )
            with jax.named_scope(f"muon_update_bench/expert_fsdp_grouped_muonh/{name}/split_to_fsdp_leaves"):
                update_parts = [
                    jnp.squeeze(update_part, axis=0)
                    for update_part in jnp.split(grouped_updates, valid_group_size, axis=0)
                ]
            for local_index, update in enumerate(update_parts):
                layer_index = layer_offset + local_index
                param = params["layers"][layer_index]["mlp"]["expert_mlp"][name]
                output_layers[layer_index]["mlp"]["expert_mlp"][name] = _restore_param_sharding(update, param)
        layer_offset += valid_group_size
    return {"layers": tuple(output_layers)}


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
    return bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES


def ns4d_bench_returns_4d_updates(bench_kind: str) -> bool:
    return bench_kind in (
        NS4D_REPLICATED_GROUP_BENCH,
        NS4D_DATA_GROUP_BENCH,
        NS4D_DATA_GROUP_APPLY_BENCH,
        EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
        EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
        EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
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
    if bench_kind == EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH:
        return "grouped_blocks_expert_direction_optimizer_updates_apply"
    if bench_kind == EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "grouped_blocks_expert_muonh_optimizer_updates_apply"
    if bench_kind == EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH:
        return "expert_only_grouped_blocks_muonh_optimizer_updates_apply"
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
            directions = _zeropower_via_newtonschulz_batched_stack_sharded(
                stacked_updates,
                config.backend_steps,
                MAY_MUON_EPSILON,
                target_pspec=target_pspec,
            )
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
            directions = _zeropower_via_newtonschulz_batched_stack_sharded(
                stacked_updates,
                config.backend_steps,
                MAY_MUON_EPSILON,
                target_pspec=target_pspec,
            )
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
                direction = zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON)
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
            direction = _zeropower_via_newtonschulz_batched_stack_sharded(
                update,
                config.backend_steps,
                MAY_MUON_EPSILON,
                target_pspec=target_pspec,
            )
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
                direction = zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON)
                direction = scale_ns4d_direction(direction)
                return param + grouped_4d_hyperball_update(param, direction, config)
        if update.ndim == 3:
            target_pspec = _target_sharding(update)
            with jax.named_scope("muon_update_bench/full_production_persistent_apply/grouped_2d"):
                if target_pspec is not None:
                    param = reshard(param, target_pspec)
                direction = _zeropower_via_newtonschulz_batched_stack_sharded(
                    update,
                    config.backend_steps,
                    MAY_MUON_EPSILON,
                    target_pspec=target_pspec,
                )
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
                directions = _zeropower_via_newtonschulz_batched_stack_sharded(
                    stacked_updates,
                    config.backend_steps,
                    MAY_MUON_EPSILON,
                    target_pspec=target_pspec,
                )
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
        update = newtonschulz_4d_dotonly_matmul(update, config.backend_steps)
    elif bench_kind == NS4D_PADDED_GROUP_BENCH:
        update = zeropower_via_newtonschulz_4d(update, config.backend_steps, MAY_MUON_EPSILON)
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
    if bench_kind in (HYPERBALL_ONLY_BENCH, ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH, FULL_PRODUCTION_APPLY_ONLY_BENCH):
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
    elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
        specs = synthetic_specs(mesh, config)
    elif is_expert_only_grouped_muonh_bench(bench_kind):
        specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
        specs = synthetic_productionish_grouped_expert_specs(mesh, config, bench_kind)
    elif bench_kind == EXPERT_GROUPED_APPLY_BOUNDARY_BENCH:
        specs = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    else:
        specs = synthetic_ns4d_specs(mesh, config, bench_kind)
    input_spec = ns4d_input_sharding(mesh, config, bench_kind).spec
    result_sharding = ns4d_result_sharding(mesh, config, bench_kind)
    if is_expert_fsdp_grouped_muonh_bench(bench_kind):
        assert_expert_stack_sharding(specs, P("expert", None, None), "expert FSDP input specs")
    else:
        assert_grouped_or_uniform_ns4d_sharding(specs, mesh, config, bench_kind, input_spec, "NS4D input specs")
    if ns4d_bench_uses_grouped_params(bench_kind):
        if is_expert_fsdp_grouped_muonh_bench(bench_kind):
            optimizer = optax.trace(MAY_MOMENTUM, nesterov=True)
            update_step = jax.jit(expert_fsdp_grouped_muonh_optimizer_apply_step_factory(mesh, config))
            with mesh, maybe_abstract_mesh(config, abstract_mesh_enabled):
                state_specs = jax.eval_shape(optimizer.init, specs)
                result_specs, _next_state_specs, update_specs = jax.eval_shape(update_step, specs, specs, state_specs)
                lowered = update_step.lower(specs, specs, state_specs)
            assert_expert_stack_sharding(
                update_specs,
                P("expert", None, None),
                "expert FSDP grouped MuonH restored updates",
            )
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
    if is_expert_fsdp_grouped_muonh_bench(bench_kind):
        assert_expert_stack_sharding(result_specs, result_sharding.spec, "expert FSDP grouped MuonH result specs")
    elif result_sharding is not None and ns4d_bench_returns_4d_updates(bench_kind):
        assert_grouped_or_uniform_ns4d_sharding(
            result_specs, mesh, config, bench_kind, result_sharding.spec, "NS4D result specs"
        )
    hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    maybe_write_text(hlo_output, hlo_text)
    hlo_summary = summarize_hlo(hlo_text)
    if (
        bench_kind in GROUPED_APPLY_BOUNDARY_BENCHES
        and not is_expert_fsdp_grouped_muonh_bench(bench_kind)
        and not allow_boundary_collectives
    ):
        assert_grouped_apply_has_no_boundary_collectives(hlo_summary, "lowered NS4D grouped apply")
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
        elif is_expert_fsdp_grouped_muonh_bench(bench_kind):
            updates = make_array_tree(config, synthetic_shardings(mesh, config), seed=1)
        elif is_expert_only_grouped_muonh_bench(bench_kind):
            updates = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        elif bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES:
            updates = make_productionish_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        elif bench_kind == EXPERT_GROUPED_APPLY_BOUNDARY_BENCH:
            updates = make_grouped_expert_array_tree(mesh, config, bench_kind, seed=1)
        else:
            updates = make_ns4d_array_tree(mesh, config, bench_kind, seed=1)
        input_spec = ns4d_input_sharding(mesh, config, bench_kind).spec
        result_sharding = ns4d_result_sharding(mesh, config, bench_kind)
        if is_expert_fsdp_grouped_muonh_bench(bench_kind):
            assert_expert_stack_sharding(updates, P("expert", None, None), "expert FSDP updates")
        else:
            assert_grouped_or_uniform_ns4d_sharding(updates, mesh, config, bench_kind, input_spec, "NS4D updates")
        if ns4d_bench_uses_grouped_params(bench_kind):
            if is_expert_fsdp_grouped_muonh_bench(bench_kind):
                params = make_array_tree(config, synthetic_shardings(mesh, config), seed=0)
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
            if is_expert_fsdp_grouped_muonh_bench(bench_kind):
                assert_expert_stack_sharding(params, P("expert", None, None), "expert FSDP params")
            else:
                assert_grouped_or_uniform_ns4d_sharding(params, mesh, config, bench_kind, input_spec, "NS4D params")
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
            and not is_expert_fsdp_grouped_muonh_bench(bench_kind)
            and not allow_boundary_collectives
        ):
            assert_grouped_apply_has_no_boundary_collectives(compiled_hlo, "compiled NS4D grouped apply")
        if compile_only:
            return TimingSummary(compile_seconds, compiled_hlo, [], None, None, None, None)

        for _ in range(warmup):
            if params is None:
                next_updates = compiled(updates)
                block_until_ready_tree(next_updates)
            elif (
                is_expert_fsdp_grouped_muonh_bench(bench_kind)
                or is_expert_only_grouped_muonh_bench(bench_kind)
                or bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES
            ):
                params, optimizer_state = compiled(params, updates, optimizer_state)
                block_until_ready_tree((params, optimizer_state))
                if is_expert_fsdp_grouped_muonh_bench(bench_kind):
                    assert_runtime_expert_stack_sharding(
                        params,
                        result_sharding.spec,
                        "warmup expert FSDP grouped params",
                        config,
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
                is_expert_fsdp_grouped_muonh_bench(bench_kind)
                or is_expert_only_grouped_muonh_bench(bench_kind)
                or bench_kind in GROUPED_OPTIMIZER_APPLY_BENCHES
            ):
                params, optimizer_state = compiled(params, updates, optimizer_state)
                block_until_ready_tree((params, optimizer_state))
                if is_expert_fsdp_grouped_muonh_bench(bench_kind):
                    assert_runtime_expert_stack_sharding(
                        params,
                        result_sharding.spec,
                        "expert FSDP grouped params",
                        config,
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
    parser.add_argument("--backend-steps", type=int, default=MAY_BACKEND_STEPS)
    parser.add_argument("--sweep-backend-steps", help="Comma-separated backend step counts, e.g. 1,5.")
    parser.add_argument("--orthogonalization-layout", default=STACK_BATCH_SHARDED)
    parser.add_argument("--max-grouped-stack-size", type=int, default=DEFAULT_MAX_GROUPED_STACK_SIZE)
    parser.add_argument("--sweep-max-grouped-stack-sizes", help="Comma-separated stack caps, e.g. 256,512,832.")
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
    return BenchConfig(
        layers=args.layers,
        ns4d_group_size=args.ns4d_group_size,
        ns4d_group_axis=args.ns4d_group_axis,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        dtype=dtype_name(dtype_from_name(args.dtype)),
        backend_steps=args.backend_steps,
        orthogonalization_layout=args.orthogonalization_layout,
        max_grouped_stack_size=args.max_grouped_stack_size,
        replica_axis=args.replica_axis,
        data_axis=args.data_axis,
        expert_axis=args.expert_axis,
        model_axis=args.model_axis,
        learning_rate=args.learning_rate,
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
    row = {
        "label": result["metadata"]["label"],
        "bench_kind": bench_kind,
        "backend_steps": config["backend_steps"],
        "max_grouped_stack_size": config["max_grouped_stack_size"],
        "ns4d_group_size": result["metadata"]["ns4d_group_size"],
        "ns4d_padded_group_size": result["metadata"]["ns4d_padded_group_size"],
        "ns4d_input_sharding_spec": result["metadata"]["ns4d_input_sharding_spec"],
        "ns4d_compute_sharding_spec": result["metadata"]["ns4d_compute_sharding_spec"],
        "ns4d_result_sharding_spec": result["metadata"]["ns4d_result_sharding_spec"],
        "ns4d_boundary_status": result["metadata"]["ns4d_boundary_status"],
        "boundary_collectives_allowed": result["metadata"]["boundary_collectives_allowed"],
        "estimated_ns_dot_flops": flops,
        "estimated_matrix_count": estimated_matrix_count(bench_config, bench_kind),
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
