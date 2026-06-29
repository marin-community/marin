# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
import json
import os
import signal
import sys
import threading
import time
from typing import Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.pallas_mgpu import (
    _DISPATCH_COPY_SCHEDULES,
    MoeMgpuConfig,
    _MoeMgpuUpMetadata,
    _group_sizes_with_padding,
    _permute_up_tiled_metadata_mgpu_kernel,
    _permute_up_tiled_values_with_schedule,
    _moe_mgpu_dispatch_w13_activation,
    _receiver_capacity,
    combine_bwd_mgpu,
    combine_slots_mgpu,
    down_unpermute_mgpu,
    dx_unpermute_vector_mgpu,
    local_producer_consumer_copy_mgpu,
    local_split_wg_w13_mgpu,
    moe_mlp_pallas_mgpu_staged,
    permute_mgpu,
    permute_up_mgpu,
    prepare_mgpu_receive_plan,
    pull_combine_mgpu,
    pull_combine_vector_mgpu,
    ragged_w2_mgpu,
    return_combine_mgpu,
    return_slots_mgpu,
)
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum

from haliax.nn.ragged_dot import ragged_dot


_H100_SXM_BF16_PEAK_FLOPS = 989e12

_IMPLEMENTATION_CHOICES = ("none", "ring", "ragged_all_to_all", "pallas_mgpu")

_PALLAS_STAGE_CHOICES = (
    "producer_consumer_copy",
    "split_wg_w13",
    "permute_up",
    "permute_up_compare",
    "permute_up_compare_split",
    "permute_metadata",
    "permute_values",
    "permute",
    "w13",
    "w2",
    "return_slots",
    "combine_slots",
    "return_combine",
    "pull_combine",
    "pull_combine_vector",
    "down_unpermute",
    "staged_forward",
    "backward_prereq",
    "combine_bwd",
    "w2_bwd",
    "w13_bwd",
    "dx_unpermute_vector",
    "dx_pull_combine_vector",
    "saved_backward_pipeline",
    "manual_backward_pipeline",
    "manual_forward_backward_pipeline",
)

_PASS_MODE_CHOICES = ("forward", "forward_backward")


@dataclass(frozen=True, slots=True)
class PallasStageDependencies:
    needs_permute: bool
    needs_permute_up: bool
    needs_w2: bool
    needs_backward_prereq: bool
    needs_combine_bwd: bool
    needs_w2_bwd: bool
    needs_w13_bwd: bool


def _pallas_stage_requested(pallas_stages: frozenset[str] | None, stage: str) -> bool:
    return pallas_stages is None or stage in pallas_stages


def _pallas_stage_dependencies(pallas_stages: frozenset[str] | None) -> PallasStageDependencies:
    needs_w2 = any(
        _pallas_stage_requested(pallas_stages, stage)
        for stage in (
            "w2",
            "return_slots",
            "combine_slots",
            "return_combine",
            "pull_combine",
            "pull_combine_vector",
        )
    )
    needs_permute = _pallas_stage_requested(pallas_stages, "permute") or _pallas_stage_requested(pallas_stages, "w13")
    needs_permute_up = (
        _pallas_stage_requested(pallas_stages, "permute_up")
        or needs_w2
        or _pallas_stage_requested(pallas_stages, "down_unpermute")
    )
    needs_w13_bwd = (
        _pallas_stage_requested(pallas_stages, "w13_bwd")
        or _pallas_stage_requested(pallas_stages, "dx_unpermute_vector")
        or _pallas_stage_requested(pallas_stages, "dx_pull_combine_vector")
        or _pallas_stage_requested(pallas_stages, "saved_backward_pipeline")
    )
    needs_w2_bwd = _pallas_stage_requested(pallas_stages, "w2_bwd") or needs_w13_bwd
    needs_combine_bwd = _pallas_stage_requested(pallas_stages, "combine_bwd") or needs_w2_bwd
    needs_backward_prereq = _pallas_stage_requested(pallas_stages, "backward_prereq") or needs_combine_bwd
    return PallasStageDependencies(
        needs_permute=needs_permute,
        needs_permute_up=needs_permute_up,
        needs_w2=needs_w2,
        needs_backward_prereq=needs_backward_prereq,
        needs_combine_bwd=needs_combine_bwd,
        needs_w2_bwd=needs_w2_bwd,
        needs_w13_bwd=needs_w13_bwd,
    )


_PALLAS_BACKWARD_STAGES = frozenset(
    {
        "backward_prereq",
        "combine_bwd",
        "w2_bwd",
        "w13_bwd",
        "dx_unpermute_vector",
        "dx_pull_combine_vector",
        "saved_backward_pipeline",
        "manual_backward_pipeline",
        "manual_forward_backward_pipeline",
    }
)


def _pallas_backward_stage_requested(pallas_stages: frozenset[str] | None) -> bool:
    if pallas_stages is None:
        return True
    return bool(pallas_stages & _PALLAS_BACKWARD_STAGES)


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _normalize_pallas_stage_args(raw_stages: list[str] | None) -> list[str] | None:
    if raw_stages is None:
        return None
    stages = [stage.strip() for value in raw_stages for stage in value.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(_PALLAS_STAGE_CHOICES))
    if unknown:
        choices = ", ".join(_PALLAS_STAGE_CHOICES)
        unknown_values = ", ".join(unknown)
        raise ValueError(f"unknown Pallas stage(s): {unknown_values}; choose from {choices}")
    duplicates = _duplicate_values(stages)
    if duplicates:
        duplicate_values = ", ".join(duplicates)
        raise ValueError(f"duplicate Pallas stage(s): {duplicate_values}")
    return stages


def _validate_chunked_permute_up_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.dispatch_chunked_permute_up:
        return
    if args.routing != "balanced":
        parser.error("--dispatch-chunked-permute-up currently requires --routing balanced")
    for field in (
        "ep_size",
        "experts_per_rank",
        "topk",
        "block_m",
        "block_n",
        "block_k",
        "dispatch_expert_group_size",
        "dispatch_chunk_copy_tile",
        "dispatch_chunk_copy_rows",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--dispatch-chunked-permute-up requires {field.replace('_', '-')} to be positive")

    global_experts = args.ep_size * args.experts_per_rank
    assignments_per_rank = args.tokens_per_rank * args.topk
    if assignments_per_rank % global_experts != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires tokens_per_rank * topk to be divisible by "
            f"ep_size * experts_per_rank; got {assignments_per_rank=} and {global_experts=}"
        )
    if args.experts_per_rank % args.dispatch_expert_group_size != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires experts_per_rank to be divisible by "
            "dispatch_expert_group_size; got "
            f"experts_per_rank={args.experts_per_rank} and "
            f"dispatch_expert_group_size={args.dispatch_expert_group_size}"
        )

    rows_per_source_expert = assignments_per_rank // global_experts
    if rows_per_source_expert % args.block_m != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires rows per source/expert to be divisible by block_m; "
            f"got rows_per_source_expert={rows_per_source_expert} and block_m={args.block_m}"
        )
    if args.hidden_dim % args.dispatch_chunk_copy_tile != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires hidden_dim to be divisible by dispatch_chunk_copy_tile; "
            f"got hidden_dim={args.hidden_dim} and dispatch_chunk_copy_tile={args.dispatch_chunk_copy_tile}"
        )
    if rows_per_source_expert % args.dispatch_chunk_copy_rows != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires rows per source/expert to be divisible by "
            "dispatch_chunk_copy_rows; got "
            f"rows_per_source_expert={rows_per_source_expert} and "
            f"dispatch_chunk_copy_rows={args.dispatch_chunk_copy_rows}"
        )
    if args.hidden_dim % args.block_k != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires hidden_dim to be divisible by block_k; "
            f"got hidden_dim={args.hidden_dim} and block_k={args.block_k}"
        )
    if args.intermediate_dim % args.block_n != 0:
        parser.error(
            "--dispatch-chunked-permute-up requires intermediate_dim to be divisible by block_n; "
            f"got intermediate_dim={args.intermediate_dim} and block_n={args.block_n}"
        )


@dataclass(frozen=True, slots=True)
class BenchShape:
    tokens_per_rank: int
    hidden_dim: int
    intermediate_dim: int
    experts_per_rank: int
    topk: int
    ep_size: int
    capacity_factor: float


@dataclass(frozen=True, slots=True)
class BenchResult:
    kernel: str
    implementation: str
    shape: str
    dtype: str
    backend: str
    device_type: str
    device_count: int
    block_sizes: str
    measurement_key: str
    compile_time: float | None
    steady_state_time: float | None
    status: str
    error: str | None
    git_sha: str
    xla_flags: str
    backend_env: str
    routing: str
    warmup: int
    steps: int
    candidate_timeout_seconds: float | None
    assignments_per_rank: int
    requested_receiver_capacity_per_rank: int
    receiver_capacity_per_rank: int
    receiver_capacity_padding_per_rank: int
    estimated_flops_per_rank: float
    effective_tflops_per_rank: float | None
    estimated_dispatch_bytes_per_rank: float
    estimated_return_bytes_per_rank: float
    estimated_memory_footprint_per_rank: float
    roofline_fraction_per_rank: float | None
    dropped_routes: int | None
    baseline_implementation: str | None
    max_abs_diff_vs_baseline: float | None
    mean_abs_diff_vs_baseline: float | None
    allclose_rtol: float
    allclose_atol: float
    matches_baseline: bool | None


@dataclass(frozen=True, slots=True)
class BenchInputs:
    x: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array
    w_up_gate: jax.Array
    w_down: jax.Array

    def as_jit_args(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return (self.x, self.selected_experts, self.combine_weights, self.w_up_gate, self.w_down)


@dataclass(frozen=True, slots=True)
class TimingResult:
    compile_time: float
    steady_state_time: float
    output: Any

    def __iter__(self) -> Iterator[Any]:
        yield self.compile_time
        yield self.steady_state_time
        yield self.output


@dataclass(frozen=True, slots=True)
class BenchEstimates:
    flops: float
    dispatch_bytes: float
    return_bytes: float
    memory_footprint: float


def _git_sha(override: str | None) -> str:
    if override is not None:
        return override
    env_sha = os.environ.get("MARIN_GIT_SHA")
    if env_sha:
        return env_sha
    head = os.popen("git rev-parse --short=12 HEAD 2>/dev/null").read().strip()
    return head or "unknown"


def _backend_env() -> str:
    keys = (
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "NCCL_DEBUG",
    )
    values = {key: os.environ[key] for key in keys if key in os.environ}
    return json.dumps(values, sort_keys=True)


def _shape_key(shape: BenchShape) -> str:
    return (
        f"T={shape.tokens_per_rank},D={shape.hidden_dim},I={shape.intermediate_dim},"
        f"E_local={shape.experts_per_rank},K={shape.topk},EP={shape.ep_size},"
        f"capacity_factor={shape.capacity_factor}"
    )


def _block_sizes(config: MoeMgpuConfig) -> str:
    return json.dumps(
        {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "max_concurrent_steps": config.max_concurrent_steps,
            "grid_block_n": config.grid_block_n,
            "capacity_factor": config.capacity_factor,
            "deterministic": config.deterministic,
            "dispatch_copy_schedule": config.dispatch_copy_schedule,
            "dispatch_expert_group_size": config.dispatch_expert_group_size,
            "dispatch_chunk_copy_tile": config.dispatch_chunk_copy_tile,
            "dispatch_chunk_copy_rows": config.dispatch_chunk_copy_rows,
            "dispatch_chunk_vectorized_copy_rows": config.dispatch_chunk_vectorized_copy_rows,
            "dispatch_fuse_metadata": config.dispatch_fuse_metadata,
            "dispatch_chunked_permute_up": config.dispatch_chunked_permute_up,
            "dispatch_split_wg_permute_up": config.dispatch_split_wg_permute_up,
            "dispatch_split_wg_overlap_permute_up": config.dispatch_split_wg_overlap_permute_up,
            "combine_bwd_block_n": config.combine_bwd_block_n,
            "dx_unpermute_block_n": config.dx_unpermute_block_n,
        },
        sort_keys=True,
    )


def _measurement_key(
    *,
    kernel: str,
    implementation: str,
    shape: str,
    dtype: str,
    backend: str,
    device_type: str,
    device_count: int,
    block_sizes: str,
    routing: str,
) -> str:
    return json.dumps(
        {
            "kernel": kernel,
            "implementation": implementation,
            "shape": shape,
            "dtype": dtype,
            "backend": backend,
            "device_type": device_type,
            "device_count": device_count,
            "block_sizes": block_sizes,
            "routing": routing,
        },
        sort_keys=True,
    )


def _row_status(*, error: str | None, matches_baseline: bool | None) -> str:
    if error is not None:
        return "error"
    if matches_baseline is False:
        return "mismatch"
    return "ok"


def _expected_result_count(
    *,
    implementations: list[str],
    include_pallas_stages: bool,
    pallas_stages: frozenset[str] | None,
) -> int:
    count = 0
    if implementations != ["none"]:
        count += len(implementations)
    if include_pallas_stages:
        count += len(_PALLAS_STAGE_CHOICES if pallas_stages is None else pallas_stages)
    return count


def _raise_for_unsuccessful_results(rows: list[BenchResult]) -> None:
    failed_rows = [row for row in rows if row.status != "ok"]
    if not failed_rows:
        return
    summaries = [f"{row.kernel}/{row.implementation} status={row.status} error={row.error!r}" for row in failed_rows]
    raise ValueError("benchmark emitted non-ok result row(s): " + "; ".join(summaries))


def _requested_receiver_capacity(shape: BenchShape) -> int:
    assignments = shape.tokens_per_rank * shape.topk
    return max(shape.experts_per_rank, int(np.ceil(assignments * shape.capacity_factor)))


def _receiver_capacity_for_shape(shape: BenchShape) -> int:
    return _receiver_capacity(
        shape.tokens_per_rank,
        shape.topk,
        shape.experts_per_rank,
        shape.capacity_factor,
    )


def _make_mesh(ep_size: int) -> Mesh:
    devices = jax.devices()
    if len(devices) < ep_size:
        raise ValueError(f"requested EP={ep_size}, but only {len(devices)} JAX devices are visible")
    mesh_devices = np.array(devices[:ep_size]).reshape(1, ep_size, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_selected_experts(
    *,
    key: jax.Array,
    shape: BenchShape,
    routing: str,
) -> jax.Array:
    global_tokens = shape.tokens_per_rank * shape.ep_size
    num_experts = shape.experts_per_rank * shape.ep_size
    if routing == "uniform":
        return jax.random.randint(key, (global_tokens, shape.topk), 0, num_experts, dtype=jnp.int32)
    if routing == "balanced":
        assignments_per_rank = shape.tokens_per_rank * shape.topk
        if assignments_per_rank % num_experts != 0:
            raise ValueError(
                "balanced routing requires tokens_per_rank * topk to be divisible by "
                f"global experts; got {assignments_per_rank=} and {num_experts=}"
            )
        per_rank_flat = jnp.tile(
            jnp.arange(num_experts, dtype=jnp.int32),
            assignments_per_rank // num_experts,
        )
        return jnp.tile(per_rank_flat[None, :], (shape.ep_size, 1)).reshape(global_tokens, shape.topk)

    token_rank = jnp.arange(global_tokens, dtype=jnp.int32)[:, None] // shape.tokens_per_rank
    route_slot = jnp.arange(shape.topk, dtype=jnp.int32)[None, :]
    if routing == "all_local":
        dst_rank = token_rank
    elif routing == "all_to_rank0":
        dst_rank = jnp.zeros_like(token_rank)
    elif routing == "all_remote_next":
        dst_rank = (token_rank + 1) % shape.ep_size
    else:
        raise ValueError(f"unknown routing pattern {routing!r}")
    local_expert = route_slot % shape.experts_per_rank
    return (dst_rank * shape.experts_per_rank + local_expert).astype(jnp.int32)


def _make_inputs(
    *,
    shape: BenchShape,
    routing: str,
    dtype: jnp.dtype,
    scale: float,
    seed: int,
) -> BenchInputs:
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(seed), 5)
    global_tokens = shape.tokens_per_rank * shape.ep_size
    num_experts = shape.experts_per_rank * shape.ep_size

    x = jax.random.normal(k_x, (global_tokens, shape.hidden_dim), dtype=jnp.float32) * scale
    selected_experts = _make_selected_experts(key=k_sel, shape=shape, routing=routing)
    combine_logits = jax.random.normal(k_logits, (global_tokens, shape.topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1)
    w_up_gate = (
        jax.random.normal(k_w13, (num_experts, shape.hidden_dim, 2 * shape.intermediate_dim), dtype=jnp.float32)
        * scale
    )
    w_down = (
        jax.random.normal(k_w2, (num_experts, shape.intermediate_dim, shape.hidden_dim), dtype=jnp.float32) * scale
    )

    return BenchInputs(
        x=x.astype(dtype),
        selected_experts=selected_experts,
        combine_weights=combine_weights.astype(dtype),
        w_up_gate=w_up_gate.astype(dtype),
        w_down=w_down.astype(dtype),
    )


def _reshard_inputs(
    mesh: Mesh,
    inputs: BenchInputs,
) -> BenchInputs:
    batch_sharding = NamedSharding(mesh, P("expert", None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    return BenchInputs(
        x=jax.sharding.reshard(inputs.x, batch_sharding),
        selected_experts=jax.sharding.reshard(inputs.selected_experts, batch_sharding),
        combine_weights=jax.sharding.reshard(inputs.combine_weights, batch_sharding),
        w_up_gate=jax.sharding.reshard(inputs.w_up_gate, expert_sharding),
        w_down=jax.sharding.reshard(inputs.w_down, expert_sharding),
    )


def _make_target(*, shape: BenchShape, dtype: jnp.dtype, seed: int) -> jax.Array:
    global_tokens = shape.tokens_per_rank * shape.ep_size
    return jax.random.normal(
        jax.random.key(seed),
        (global_tokens, shape.hidden_dim),
        dtype=jnp.float32,
    ).astype(dtype)


def _reshard_target(mesh: Mesh, target: jax.Array) -> jax.Array:
    return jax.sharding.reshard(target, NamedSharding(mesh, P("expert", None)))


def _time_jitted(fn: Callable[..., Any], *args: Any, warmup: int, steps: int) -> TimingResult:
    started = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - started

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    started = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        jax.block_until_ready(out)
    steady_state_time = (time.perf_counter() - started) / steps
    return TimingResult(
        compile_time=compile_time,
        steady_state_time=steady_state_time,
        output=out,
    )


@contextlib.contextmanager
def _candidate_timeout(seconds: float | None):
    if seconds is None:
        yield
        return
    if seconds <= 0.0:
        raise ValueError(f"candidate timeout must be positive when set, got {seconds}")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("candidate timeout uses SIGALRM and must run on the main Python thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"benchmark candidate exceeded {seconds:.3f}s timeout")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0 or previous_timer[1] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _grad_diff_stats(
    actual_grads: tuple[jax.Array, ...],
    baseline_grads: tuple[jax.Array, ...],
) -> tuple[float, float]:
    max_abs_diff = 0.0
    diff_sum = 0.0
    diff_count = 0
    for actual, baseline in zip(actual_grads, baseline_grads, strict=True):
        diff = jnp.abs(actual.astype(jnp.float32) - baseline.astype(jnp.float32))
        max_abs_diff = max(max_abs_diff, float(jnp.max(diff).block_until_ready()))
        diff_sum += float(jnp.sum(diff).block_until_ready())
        diff_count += diff.size
    if diff_count == 0:
        return max_abs_diff, 0.0
    return max_abs_diff, diff_sum / diff_count


def _grads_allclose(
    actual_grads: tuple[jax.Array, ...],
    baseline_grads: tuple[jax.Array, ...],
    *,
    rtol: float,
    atol: float,
) -> bool:
    for actual, baseline in zip(actual_grads, baseline_grads, strict=True):
        diff = jnp.abs(actual.astype(jnp.float32) - baseline.astype(jnp.float32))
        tolerance = atol + rtol * jnp.abs(baseline.astype(jnp.float32))
        if not bool(jnp.all(diff <= tolerance).block_until_ready()):
            return False
    return True


def _estimates(shape: BenchShape, dtype: jnp.dtype) -> BenchEstimates:
    assignments = shape.tokens_per_rank * shape.topk
    received = _receiver_capacity_for_shape(shape)
    flops = 6.0 * received * shape.hidden_dim * shape.intermediate_dim
    bytes_per_element = jnp.dtype(dtype).itemsize
    dispatch_bytes = assignments * shape.hidden_dim * bytes_per_element
    return_bytes = assignments * shape.hidden_dim * bytes_per_element
    memory_footprint = (
        shape.tokens_per_rank * shape.hidden_dim
        + received * shape.hidden_dim
        + received * shape.intermediate_dim
        + assignments * shape.hidden_dim
    ) * bytes_per_element
    return BenchEstimates(
        flops=flops,
        dispatch_bytes=dispatch_bytes,
        return_bytes=return_bytes,
        memory_footprint=memory_footprint,
    )


def _stage_flops(shape: BenchShape, stage: str) -> float:
    assignments = shape.tokens_per_rank * shape.topk
    received = _receiver_capacity_for_shape(shape)
    if stage in ("producer_consumer_copy", "permute", "permute_metadata", "permute_values"):
        return 0.0
    if stage == "split_wg_w13":
        return 4.0 * shape.tokens_per_rank * shape.hidden_dim * shape.intermediate_dim
    if stage == "permute_up":
        return 4.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage in ("permute_up_compare", "permute_up_compare_split"):
        return 8.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "w13":
        return 4.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "down_unpermute":
        return 2.0 * received * shape.intermediate_dim * shape.hidden_dim
    if stage == "staged_forward":
        return 6.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "w2":
        return 2.0 * received * shape.intermediate_dim * shape.hidden_dim
    if stage == "return_slots":
        return 0.0
    if stage == "combine_slots":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "return_combine":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "pull_combine":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "pull_combine_vector":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "backward_prereq":
        return 6.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "combine_bwd":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "w2_bwd":
        return 4.0 * received * shape.intermediate_dim * shape.hidden_dim
    if stage == "w13_bwd":
        return 8.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "dx_unpermute_vector":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "dx_pull_combine_vector":
        return 2.0 * assignments * shape.hidden_dim
    if stage == "saved_backward_pipeline":
        return (
            _stage_flops(shape, "combine_bwd")
            + _stage_flops(shape, "w2_bwd")
            + _stage_flops(shape, "w13_bwd")
            + _stage_flops(shape, "dx_unpermute_vector")
        )
    if stage == "manual_backward_pipeline":
        return 3.0 * 6.0 * received * shape.hidden_dim * shape.intermediate_dim
    if stage == "manual_forward_backward_pipeline":
        return 4.0 * 6.0 * received * shape.hidden_dim * shape.intermediate_dim
    raise ValueError(f"unknown stage {stage!r}")


def _result_row(
    *,
    kernel: str,
    implementation: str,
    shape: BenchShape,
    config: MoeMgpuConfig,
    dtype: jnp.dtype,
    routing: str,
    warmup: int,
    steps: int,
    git_sha: str,
    flops: float,
    dispatch_bytes: float,
    return_bytes: float,
    memory_footprint: float,
    compile_time: float | None,
    steady_state_time: float | None,
    error: str | None,
    dropped_routes: int | None,
    baseline_implementation: str | None,
    max_abs_diff: float | None,
    mean_abs_diff: float | None,
    matches_baseline: bool | None,
    allclose_rtol: float,
    allclose_atol: float,
    candidate_timeout_seconds: float | None = None,
) -> BenchResult:
    device = jax.devices()[0]
    assignments = shape.tokens_per_rank * shape.topk
    requested_receiver_capacity = _requested_receiver_capacity(shape)
    receiver_capacity = _receiver_capacity_for_shape(shape)
    shape_key = _shape_key(shape)
    dtype_key = str(jnp.dtype(dtype))
    backend = device.platform
    device_type = getattr(device, "device_kind", "unknown")
    block_sizes = _block_sizes(config)
    effective_tflops = None
    roofline_fraction = None
    if steady_state_time is not None:
        effective_tflops = flops / steady_state_time / 1e12
        if "h100" in getattr(device, "device_kind", "").lower():
            roofline_fraction = effective_tflops * 1e12 / _H100_SXM_BF16_PEAK_FLOPS
    return BenchResult(
        kernel=kernel,
        implementation=implementation,
        shape=shape_key,
        dtype=dtype_key,
        backend=backend,
        device_type=device_type,
        device_count=shape.ep_size,
        block_sizes=block_sizes,
        measurement_key=_measurement_key(
            kernel=kernel,
            implementation=implementation,
            shape=shape_key,
            dtype=dtype_key,
            backend=backend,
            device_type=device_type,
            device_count=shape.ep_size,
            block_sizes=block_sizes,
            routing=routing,
        ),
        compile_time=compile_time,
        steady_state_time=steady_state_time,
        status=_row_status(error=error, matches_baseline=matches_baseline),
        error=error,
        git_sha=git_sha,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        backend_env=_backend_env(),
        routing=routing,
        warmup=warmup,
        steps=steps,
        candidate_timeout_seconds=candidate_timeout_seconds,
        assignments_per_rank=assignments,
        requested_receiver_capacity_per_rank=requested_receiver_capacity,
        receiver_capacity_per_rank=receiver_capacity,
        receiver_capacity_padding_per_rank=receiver_capacity - requested_receiver_capacity,
        estimated_flops_per_rank=flops,
        effective_tflops_per_rank=effective_tflops,
        estimated_dispatch_bytes_per_rank=dispatch_bytes,
        estimated_return_bytes_per_rank=return_bytes,
        estimated_memory_footprint_per_rank=memory_footprint,
        roofline_fraction_per_rank=roofline_fraction,
        dropped_routes=dropped_routes,
        baseline_implementation=baseline_implementation,
        max_abs_diff_vs_baseline=max_abs_diff,
        mean_abs_diff_vs_baseline=mean_abs_diff,
        allclose_rtol=allclose_rtol,
        allclose_atol=allclose_atol,
        matches_baseline=matches_baseline,
    )


def _emit_result(
    row: BenchResult,
    jsonl_handle: TextIO | None,
    seen_measurement_keys: set[str] | None = None,
) -> None:
    if seen_measurement_keys is not None:
        if row.measurement_key in seen_measurement_keys:
            raise ValueError(f"duplicate benchmark measurement_key emitted: {row.measurement_key}")
        seen_measurement_keys.add(row.measurement_key)
    serialized = json.dumps(asdict(row), sort_keys=True)
    print(serialized, flush=True)
    if jsonl_handle is not None:
        jsonl_handle.write(serialized)
        jsonl_handle.write("\n")
        jsonl_handle.flush()


def _emit_progress(
    *,
    kernel: str,
    implementation: str,
    shape: BenchShape,
    config: MoeMgpuConfig | None = None,
    dtype: jnp.dtype | None = None,
    routing: str | None = None,
) -> None:
    shape_key = _shape_key(shape)
    event: dict[str, Any] = {
        "event": "starting",
        "kernel": kernel,
        "implementation": implementation,
        "shape": shape_key,
        "ts": time.time(),
    }
    if config is not None and dtype is not None and routing is not None:
        device = jax.devices()[0]
        dtype_key = str(jnp.dtype(dtype))
        backend = device.platform
        device_type = getattr(device, "device_kind", "unknown")
        block_sizes = _block_sizes(config)
        event.update(
            {
                "dtype": dtype_key,
                "backend": backend,
                "device_type": device_type,
                "device_count": shape.ep_size,
                "block_sizes": block_sizes,
                "routing": routing,
                "measurement_key": _measurement_key(
                    kernel=kernel,
                    implementation=implementation,
                    shape=shape_key,
                    dtype=dtype_key,
                    backend=backend,
                    device_type=device_type,
                    device_count=shape.ep_size,
                    block_sizes=block_sizes,
                    routing=routing,
                ),
            }
        )
    print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)


def _benchmark_implementation(
    *,
    implementation: str,
    shape: BenchShape,
    config: MoeMgpuConfig,
    mesh: Mesh,
    inputs: BenchInputs,
    target: jax.Array | None,
    pass_mode: str,
    baseline_output: Any | None,
    baseline_implementation: str | None,
    routing: str,
    dtype: jnp.dtype,
    warmup: int,
    steps: int,
    git_sha: str,
    allclose_rtol: float,
    allclose_atol: float,
    candidate_timeout_seconds: float | None,
) -> tuple[BenchResult, Any | None]:
    estimates = _estimates(shape, dtype)
    flops = estimates.flops
    kernel_name = "grug_moe_mlp_forward" if pass_mode == "forward" else "grug_moe_mlp_forward_backward"
    if pass_mode == "forward_backward":
        # Approximate dense training-step math: forward plus input/weight backward
        # GEMMs for W13 and W2. This keeps the TFLOP metric comparable enough for
        # regression tracking without claiming an exact sparse-router flop count.
        flops *= 3.0

    def call_moe(x, selected_experts, combine_weights, w_up_gate, w_down):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=config.capacity_factor,
            report_capacity_overflow=True,
        )

    if pass_mode == "forward_backward" and target is None:
        raise ValueError("target is required for forward_backward benchmarking")

    def call_moe_loss(x, selected_experts, combine_weights, w_up_gate, w_down, target):
        out, dropped = call_moe(x, selected_experts, combine_weights, w_up_gate, w_down)
        loss = jnp.sum(out.astype(jnp.float32) * target.astype(jnp.float32))
        return loss, dropped

    grad_fn = jax.value_and_grad(call_moe_loss, argnums=(0, 2, 3, 4), has_aux=True)

    try:
        _emit_progress(
            kernel=kernel_name,
            implementation=implementation,
            shape=shape,
            config=config,
            dtype=dtype,
            routing=routing,
        )
        with jax.set_mesh(mesh), _candidate_timeout(candidate_timeout_seconds):
            if pass_mode == "forward":
                timing = _time_jitted(
                    jax.jit(call_moe),
                    *inputs.as_jit_args(),
                    warmup=warmup,
                    steps=steps,
                )
                compile_time = timing.compile_time
                steady_state_time = timing.steady_state_time
                out = timing.output
                dropped = int(np.asarray(out[1]))
                output = out[0]
            else:
                timing = _time_jitted(
                    jax.jit(grad_fn),
                    *inputs.as_jit_args(),
                    target,
                    warmup=warmup,
                    steps=steps,
                )
                compile_time = timing.compile_time
                steady_state_time = timing.steady_state_time
                out = timing.output
                (loss, dropped_array), grads = out
                dropped = int(np.asarray(dropped_array))
                output = (loss, grads)
        max_abs_diff = None
        mean_abs_diff = None
        matches_baseline = None
        if baseline_output is not None:
            if pass_mode == "forward":
                diff = jnp.abs(output.astype(jnp.float32) - baseline_output.astype(jnp.float32))
                max_abs_diff = float(jnp.max(diff).block_until_ready())
                mean_abs_diff = float(jnp.mean(diff).block_until_ready())
                tolerance = allclose_atol + allclose_rtol * jnp.abs(baseline_output.astype(jnp.float32))
                matches_baseline = bool(jnp.all(diff <= tolerance).block_until_ready())
            else:
                loss, grads = output
                baseline_loss, baseline_grads = baseline_output
                loss_diff = jnp.abs(loss.astype(jnp.float32) - baseline_loss.astype(jnp.float32))
                grad_max_diff, grad_mean_diff = _grad_diff_stats(grads, baseline_grads)
                max_abs_diff = max(float(loss_diff.block_until_ready()), grad_max_diff)
                mean_abs_diff = grad_mean_diff
                loss_tolerance = allclose_atol + allclose_rtol * jnp.abs(baseline_loss.astype(jnp.float32))
                loss_matches = bool((loss_diff <= loss_tolerance).block_until_ready())
                matches_baseline = loss_matches and _grads_allclose(
                    grads,
                    baseline_grads,
                    rtol=allclose_rtol,
                    atol=allclose_atol,
                )
        error = None
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        compile_time = None
        steady_state_time = None
        dropped = None
        output = None
        max_abs_diff = None
        mean_abs_diff = None
        matches_baseline = None
        error = f"{type(exc).__name__}: {exc}"

    return (
        _result_row(
            kernel=kernel_name,
            implementation=implementation,
            shape=shape,
            config=config,
            dtype=dtype,
            routing=routing,
            warmup=warmup,
            steps=steps,
            git_sha=git_sha,
            flops=flops,
            dispatch_bytes=estimates.dispatch_bytes,
            return_bytes=estimates.return_bytes,
            memory_footprint=estimates.memory_footprint,
            compile_time=compile_time,
            steady_state_time=steady_state_time,
            error=error,
            dropped_routes=dropped,
            baseline_implementation=baseline_implementation,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            matches_baseline=matches_baseline,
            allclose_rtol=allclose_rtol,
            allclose_atol=allclose_atol,
            candidate_timeout_seconds=candidate_timeout_seconds,
        ),
        output,
    )


def _benchmark_pallas_stages(
    *,
    shape: BenchShape,
    config: MoeMgpuConfig,
    mesh: Mesh,
    inputs: BenchInputs,
    target: jax.Array | None,
    routing: str,
    dtype: jnp.dtype,
    warmup: int,
    steps: int,
    git_sha: str,
    allclose_rtol: float,
    allclose_atol: float,
    pallas_stages: frozenset[str] | None,
) -> list[BenchResult]:
    x = inputs.x
    selected_experts = inputs.selected_experts
    combine_weights = inputs.combine_weights
    w_up_gate = inputs.w_up_gate
    w_down = inputs.w_down
    target_for_stages = None
    if target is not None:
        target_for_stages = jax.sharding.reshard(target, NamedSharding(mesh, P("expert", None)))
    estimates = _estimates(shape, dtype)
    dispatch_bytes = estimates.dispatch_bytes
    return_bytes = estimates.return_bytes
    memory_footprint = estimates.memory_footprint
    stage_dependencies = _pallas_stage_dependencies(pallas_stages)

    def requested(stage: str) -> bool:
        return _pallas_stage_requested(pallas_stages, stage)

    def emit_stage_progress(kernel: str) -> None:
        _emit_progress(
            kernel=kernel,
            implementation="pallas_mgpu",
            shape=shape,
            config=config,
            dtype=dtype,
            routing=routing,
        )

    def shard_permute_inputs(selected_experts_local):
        plan = prepare_mgpu_receive_plan(
            selected_experts_local,
            local_experts=shape.experts_per_rank,
            config=config,
        )
        return plan

    def shard_permute_metadata(selected_experts_local):
        plan = shard_permute_inputs(selected_experts_local)
        recv_src_rank, recv_src_assignment = _permute_up_tiled_metadata_mgpu_kernel(
            plan.metadata.assignment_ids_sorted,
            plan.metadata.dst_ranks_sorted,
            plan.remote_rows_sorted,
            plan.keep_sorted,
            capacity=plan.capacity,
            ep_size=plan.ep_size,
            expert_axis="expert",
            config=config,
        )
        return recv_src_rank, recv_src_assignment, plan.dropped

    permute_metadata_fn = shard_map(
        shard_permute_metadata,
        mesh=mesh,
        in_specs=P("expert", None),
        out_specs=(P("expert"), P("expert"), P()),
        check_vma=False,
    )

    def shard_permute_values(x_local, selected_experts_local):
        plan = shard_permute_inputs(selected_experts_local)
        recv_x = _permute_up_tiled_values_with_schedule(
            x_local,
            plan.metadata,
            plan.remote_rows_sorted,
            plan.keep_sorted,
            rank=plan.rank,
            capacity=plan.capacity,
            ep_size=plan.ep_size,
            local_experts=shape.experts_per_rank,
            config=config,
        )
        return recv_x, plan.dropped

    permute_values_fn = shard_map(
        shard_permute_values,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None)),
        out_specs=(P("expert", None), P()),
        check_vma=False,
    )

    def shard_permute_up(x_local, selected_experts_local, w_up_gate_local):
        layout = permute_up_mgpu(
            x_local,
            selected_experts_local,
            w_up_gate_local,
            local_experts=shape.experts_per_rank,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=config,
        )
        return (
            layout.hidden,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            layout.rows_per_expert,
            layout.dropped,
        )

    permute_up_fn = shard_map(
        shard_permute_up,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
        out_specs=(P("expert", None), P("expert"), P("expert"), P("expert"), P()),
        check_vma=False,
    )

    def shard_permute_up_baseline(x_local, selected_experts_local, w_up_gate_local):
        baseline_config = replace(
            config,
            dispatch_chunk_copy_tile=MoeMgpuConfig().dispatch_chunk_copy_tile,
            dispatch_chunk_copy_rows=MoeMgpuConfig().dispatch_chunk_copy_rows,
            dispatch_chunk_vectorized_copy_rows=False,
            dispatch_chunked_permute_up=False,
            dispatch_split_wg_permute_up=False,
            dispatch_split_wg_overlap_permute_up=False,
        )
        layout = permute_up_mgpu(
            x_local,
            selected_experts_local,
            w_up_gate_local,
            local_experts=shape.experts_per_rank,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=baseline_config,
        )
        return (
            layout.hidden,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            layout.rows_per_expert,
            layout.dropped,
        )

    permute_up_baseline_fn = shard_map(
        shard_permute_up_baseline,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
        out_specs=(P("expert", None), P("expert"), P("expert"), P("expert"), P()),
        check_vma=False,
    )

    def shard_permute_up_compare(x_local, selected_experts_local, w_up_gate_local):
        baseline_config = replace(
            config,
            dispatch_chunk_copy_tile=MoeMgpuConfig().dispatch_chunk_copy_tile,
            dispatch_chunk_copy_rows=MoeMgpuConfig().dispatch_chunk_copy_rows,
            dispatch_chunk_vectorized_copy_rows=False,
            dispatch_chunked_permute_up=False,
            dispatch_split_wg_permute_up=False,
            dispatch_split_wg_overlap_permute_up=False,
        )
        baseline = permute_up_mgpu(
            x_local,
            selected_experts_local,
            w_up_gate_local,
            local_experts=shape.experts_per_rank,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=baseline_config,
        )
        candidate = permute_up_mgpu(
            x_local,
            selected_experts_local,
            w_up_gate_local,
            local_experts=shape.experts_per_rank,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=config,
        )
        hidden_diff = jnp.abs(candidate.hidden.astype(jnp.float32) - baseline.hidden.astype(jnp.float32))
        local_max = jnp.max(hidden_diff)
        local_sum = jnp.sum(hidden_diff)
        local_count = jnp.array(hidden_diff.size, dtype=jnp.float32)
        rank_mismatches = jnp.sum(candidate.recv_src_rank != baseline.recv_src_rank, dtype=jnp.int32)
        assignment_mismatches = jnp.sum(
            candidate.recv_src_assignment != baseline.recv_src_assignment,
            dtype=jnp.int32,
        )
        return (
            jax.lax.pmax(local_max, "expert"),
            jax.lax.psum(local_sum, "expert") / jax.lax.psum(local_count, "expert"),
            jax.lax.psum(rank_mismatches, "expert"),
            jax.lax.psum(assignment_mismatches, "expert"),
            baseline.dropped,
            candidate.dropped,
        )

    permute_up_compare_fn = shard_map(
        shard_permute_up_compare,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
        out_specs=(P(), P(), P(), P(), P(), P()),
        check_vma=False,
    )

    def shard_producer_consumer_copy(x_local):
        copied = local_producer_consumer_copy_mgpu(x_local, config=config)
        diff = jnp.abs(copied.astype(jnp.float32) - x_local.astype(jnp.float32))
        local_max = jnp.max(diff)
        local_sum = jnp.sum(diff)
        local_count = jnp.array(diff.size, dtype=jnp.float32)
        return (
            jax.lax.pmax(local_max, "expert"),
            jax.lax.psum(local_sum, "expert") / jax.lax.psum(local_count, "expert"),
        )

    producer_consumer_copy_fn = shard_map(
        shard_producer_consumer_copy,
        mesh=mesh,
        in_specs=P("expert", None),
        out_specs=(P(), P()),
        check_vma=False,
    )

    def shard_split_wg_w13(x_local, w_up_gate_local):
        w_local = w_up_gate_local[0]
        actual = local_split_wg_w13_mgpu(
            x_local,
            w_local,
            ActivationFunctionEnum.silu.to_jax_fn(),
            config=config,
        )
        gate = x_local.astype(jnp.float32) @ w_local[:, : shape.intermediate_dim].astype(jnp.float32)
        up = x_local.astype(jnp.float32) @ w_local[:, shape.intermediate_dim :].astype(jnp.float32)
        expected = (jax.nn.silu(gate) * up).astype(actual.dtype)
        diff = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
        local_max = jnp.max(diff)
        local_sum = jnp.sum(diff)
        local_count = jnp.array(diff.size, dtype=jnp.float32)
        return (
            jax.lax.pmax(local_max, "expert"),
            jax.lax.psum(local_sum, "expert") / jax.lax.psum(local_count, "expert"),
        )

    split_wg_w13_fn = shard_map(
        shard_split_wg_w13,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None, None)),
        out_specs=(P(), P()),
        check_vma=False,
    )

    def shard_permute(x_local, selected_experts_local):
        layout = permute_mgpu(
            x_local,
            selected_experts_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )
        return (
            layout.recv_x,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            layout.rows_per_expert,
            layout.dropped,
        )

    permute_fn = shard_map(
        shard_permute,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None)),
        out_specs=(P("expert", None), P("expert"), P("expert"), P("expert"), P()),
        check_vma=False,
    )

    def shard_w13(recv_x, rows_per_expert, w_up_gate_local):
        rows_with_padding = rows_per_expert.at[-1].add(recv_x.shape[0] - jnp.sum(rows_per_expert, dtype=jnp.int32))
        metadata = _MoeMgpuUpMetadata(global_expert_counts=rows_with_padding[jnp.newaxis, :])
        return _moe_mgpu_dispatch_w13_activation(
            recv_x,
            w_up_gate_local,
            ActivationFunctionEnum.silu.to_jax_fn(),
            metadata,
            config,
        )

    w13_fn = shard_map(
        shard_w13,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert", None, None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_down_unpermute(
        hidden,
        recv_src_rank,
        recv_src_assignment,
        rows_per_expert,
        w_down_local,
        weights_local,
        selected_experts_local,
    ):
        return down_unpermute_mgpu(
            hidden,
            recv_src_rank,
            recv_src_assignment,
            rows_per_expert,
            w_down_local,
            weights_local,
            selected_experts_local,
            expert_axis="expert",
            config=config,
        )

    down_unpermute_fn = shard_map(
        shard_down_unpermute,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert"),
            P("expert", None, None),
            P("expert", None),
            P("expert", None),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_w2(hidden, rows_per_expert, w_down_local):
        return ragged_w2_mgpu(
            hidden,
            w_down_local,
            rows_per_expert,
            config=config,
        )

    w2_fn = shard_map(
        shard_w2,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert", None, None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_return_combine(y_dispatch, recv_src_rank, recv_src_assignment, weights_local):
        return return_combine_mgpu(
            y_dispatch,
            recv_src_rank,
            recv_src_assignment,
            weights_local,
            expert_axis="expert",
            config=config,
        )

    return_combine_fn = shard_map(
        shard_return_combine,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert"), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_pull_combine(y_dispatch, selected_experts_local, weights_local):
        return pull_combine_mgpu(
            y_dispatch,
            selected_experts_local,
            weights_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )

    pull_combine_fn = shard_map(
        shard_pull_combine,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_pull_combine_vector(y_dispatch, selected_experts_local, weights_local):
        return pull_combine_vector_mgpu(
            y_dispatch,
            selected_experts_local,
            weights_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )

    pull_combine_vector_fn = shard_map(
        shard_pull_combine_vector,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_return_slots(y_dispatch, recv_src_rank, recv_src_assignment, weights_local):
        return return_slots_mgpu(
            y_dispatch,
            recv_src_rank,
            recv_src_assignment,
            weights_local,
            expert_axis="expert",
            config=config,
        )

    return_slots_fn = shard_map(
        shard_return_slots,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert"), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_combine_slots(return_slots, weights_local):
        return combine_slots_mgpu(return_slots, weights_local, config=config)

    combine_slots_fn = shard_map(
        shard_combine_slots,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_staged_forward(x_local, selected_experts_local, weights_local, w_up_gate_local, w_down_local):
        return moe_mlp_pallas_mgpu_staged(
            x_local,
            selected_experts_local,
            weights_local,
            w_up_gate_local,
            w_down_local,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=config,
        )

    staged_forward_fn = shard_map(
        shard_staged_forward,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None, None),
            P("expert", None, None),
        ),
        out_specs=(P("expert", None), P()),
        check_vma=False,
    )

    def shard_backward_prereq(x_local, selected_experts_local, w_up_gate_local, w_down_local):
        layout = permute_mgpu(
            x_local,
            selected_experts_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )
        compute_group_sizes = _group_sizes_with_padding(layout.rows_per_expert, total_size=layout.recv_x.shape[0])
        w13_out = ragged_dot(layout.recv_x, w_up_gate_local, compute_group_sizes)
        gate, up = jnp.split(w13_out, [shape.intermediate_dim], axis=-1)
        hidden = (ActivationFunctionEnum.silu.to_jax_fn()(gate) * up).astype(layout.recv_x.dtype)
        y_dispatch = ragged_dot(hidden, w_down_local, compute_group_sizes).astype(hidden.dtype)
        return (
            layout.recv_x,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            layout.rows_per_expert,
            hidden,
            y_dispatch,
            layout.dropped,
        )

    backward_prereq_fn = shard_map(
        shard_backward_prereq,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None, None), P("expert", None, None)),
        out_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P(),
        ),
        check_vma=False,
    )

    def shard_combine_bwd(y_dispatch, out_bar_local, selected_experts_local, weights_local):
        return combine_bwd_mgpu(
            y_dispatch,
            out_bar_local,
            selected_experts_local,
            weights_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )

    combine_bwd_fn = shard_map(
        shard_combine_bwd,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None), P("expert", None)),
        out_specs=(P("expert", None), P("expert", None)),
        check_vma=False,
    )

    def shard_w2_bwd(hidden, rows_per_expert, w_down_local, dy_dispatch):
        compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=hidden.shape[0])

        def w2_forward(hidden_arg, w_down_arg):
            return ragged_dot(hidden_arg, w_down_arg, compute_group_sizes).astype(hidden_arg.dtype)

        _y_dispatch, pullback = jax.vjp(w2_forward, hidden, w_down_local)
        return pullback(dy_dispatch)

    w2_bwd_fn = shard_map(
        shard_w2_bwd,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert", None, None), P("expert", None)),
        out_specs=(P("expert", None), P("expert", None, None)),
        check_vma=False,
    )

    def shard_w13_bwd(recv_x, rows_per_expert, w_up_gate_local, dhidden):
        compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=recv_x.shape[0])

        def w13_activation(recv_x_arg, w_up_gate_arg):
            w13_out = ragged_dot(recv_x_arg, w_up_gate_arg, compute_group_sizes)
            gate, up = jnp.split(w13_out, [shape.intermediate_dim], axis=-1)
            return (ActivationFunctionEnum.silu.to_jax_fn()(gate) * up).astype(recv_x_arg.dtype)

        _hidden, pullback = jax.vjp(w13_activation, recv_x, w_up_gate_local)
        return pullback(dhidden)

    w13_bwd_fn = shard_map(
        shard_w13_bwd,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert", None, None), P("expert", None)),
        out_specs=(P("expert", None), P("expert", None, None)),
        check_vma=False,
    )

    def shard_dx_unpermute_vector(drecv_x, recv_src_rank, recv_src_assignment, weights_local):
        return dx_unpermute_vector_mgpu(
            drecv_x,
            recv_src_rank,
            recv_src_assignment,
            jnp.ones_like(weights_local, dtype=drecv_x.dtype),
            expert_axis="expert",
            config=config,
        )

    dx_unpermute_vector_fn = shard_map(
        shard_dx_unpermute_vector,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert"), P("expert"), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_dx_pull_combine_vector(drecv_x, selected_experts_local, weights_local):
        return pull_combine_vector_mgpu(
            drecv_x,
            selected_experts_local,
            jnp.ones_like(weights_local, dtype=drecv_x.dtype),
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )

    dx_pull_combine_vector_fn = shard_map(
        shard_dx_pull_combine_vector,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None)),
        out_specs=P("expert", None),
        check_vma=False,
    )

    def shard_saved_backward_pipeline(
        recv_x,
        recv_src_rank,
        recv_src_assignment,
        rows_per_expert,
        hidden,
        y_dispatch,
        selected_experts_local,
        weights_local,
        w_up_gate_local,
        w_down_local,
        out_bar_local,
    ):
        compute_group_sizes = _group_sizes_with_padding(rows_per_expert, total_size=recv_x.shape[0])

        def w13_activation(recv_x_arg, w_up_gate_arg):
            w13_out = ragged_dot(recv_x_arg, w_up_gate_arg, compute_group_sizes)
            gate, up = jnp.split(w13_out, [shape.intermediate_dim], axis=-1)
            return (ActivationFunctionEnum.silu.to_jax_fn()(gate) * up).astype(recv_x_arg.dtype)

        _hidden, w13_pullback = jax.vjp(w13_activation, recv_x, w_up_gate_local)

        def w2_forward(hidden_arg, w_down_arg):
            return ragged_dot(hidden_arg, w_down_arg, compute_group_sizes).astype(hidden_arg.dtype)

        _y_dispatch, w2_pullback = jax.vjp(w2_forward, hidden, w_down_local)
        dy_dispatch, dcombine_weights = combine_bwd_mgpu(
            y_dispatch,
            out_bar_local.astype(y_dispatch.dtype),
            selected_experts_local,
            weights_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )
        dhidden, dmoe_w2 = w2_pullback(dy_dispatch)
        drecv_x, dmoe_w13 = w13_pullback(dhidden)
        dx = dx_unpermute_vector_mgpu(
            drecv_x,
            recv_src_rank,
            recv_src_assignment,
            jnp.ones_like(weights_local, dtype=drecv_x.dtype),
            expert_axis="expert",
            config=config,
        )
        return dx, dcombine_weights, dmoe_w13, dmoe_w2

    saved_backward_pipeline_fn = shard_map(
        shard_saved_backward_pipeline,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None, None),
            P("expert", None, None),
            P("expert", None),
        ),
        out_specs=(P("expert", None), P("expert", None), P("expert", None, None), P("expert", None, None)),
        check_vma=False,
    )

    def shard_manual_backward_pipeline(
        x_local,
        selected_experts_local,
        weights_local,
        w_up_gate_local,
        w_down_local,
        out_bar_local,
    ):
        layout = permute_mgpu(
            x_local,
            selected_experts_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )
        compute_group_sizes = _group_sizes_with_padding(layout.rows_per_expert, total_size=layout.recv_x.shape[0])

        def w13_activation(recv_x_arg, w_up_gate_arg):
            w13_out = ragged_dot(recv_x_arg, w_up_gate_arg, compute_group_sizes)
            gate, up = jnp.split(w13_out, [shape.intermediate_dim], axis=-1)
            return (ActivationFunctionEnum.silu.to_jax_fn()(gate) * up).astype(recv_x_arg.dtype)

        hidden, w13_pullback = jax.vjp(w13_activation, layout.recv_x, w_up_gate_local)

        def w2_forward(hidden_arg, w_down_arg):
            return ragged_dot(hidden_arg, w_down_arg, compute_group_sizes).astype(hidden_arg.dtype)

        y_dispatch, w2_pullback = jax.vjp(w2_forward, hidden, w_down_local)
        dy_dispatch, dcombine_weights = combine_bwd_mgpu(
            y_dispatch,
            out_bar_local.astype(y_dispatch.dtype),
            selected_experts_local,
            weights_local,
            local_experts=shape.experts_per_rank,
            expert_axis="expert",
            config=config,
        )
        dhidden, dmoe_w2 = w2_pullback(dy_dispatch)
        drecv_x, dmoe_w13 = w13_pullback(dhidden)
        dx = dx_unpermute_vector_mgpu(
            drecv_x,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            jnp.ones_like(weights_local, dtype=drecv_x.dtype),
            expert_axis="expert",
            config=config,
        )
        return dx, dcombine_weights, dmoe_w13, dmoe_w2, layout.dropped

    manual_backward_pipeline_fn = shard_map(
        shard_manual_backward_pipeline,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None, None),
            P("expert", None, None),
            P("expert", None),
        ),
        out_specs=(P("expert", None), P("expert", None), P("expert", None, None), P("expert", None, None), P()),
        check_vma=False,
    )

    def shard_manual_forward_backward_pipeline(
        x_local,
        selected_experts_local,
        weights_local,
        w_up_gate_local,
        w_down_local,
        out_bar_local,
    ):
        forward_out, forward_dropped = moe_mlp_pallas_mgpu_staged(
            x_local,
            selected_experts_local,
            weights_local,
            w_up_gate_local,
            w_down_local,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            expert_axis="expert",
            config=config,
        )
        dx, dcombine_weights, dmoe_w13, dmoe_w2, backward_dropped = shard_manual_backward_pipeline(
            x_local,
            selected_experts_local,
            weights_local,
            w_up_gate_local,
            w_down_local,
            out_bar_local,
        )
        dx = dx + jnp.zeros_like(forward_out, dtype=dx.dtype)
        return dx, dcombine_weights, dmoe_w13, dmoe_w2, forward_dropped + backward_dropped

    manual_forward_backward_pipeline_fn = shard_map(
        shard_manual_forward_backward_pipeline,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None, None),
            P("expert", None, None),
            P("expert", None),
        ),
        out_specs=(P("expert", None), P("expert", None), P("expert", None, None), P("expert", None, None), P()),
        check_vma=False,
    )

    needs_w2 = stage_dependencies.needs_w2
    needs_permute = stage_dependencies.needs_permute
    needs_permute_up = stage_dependencies.needs_permute_up
    needs_backward_prereq = stage_dependencies.needs_backward_prereq
    needs_combine_bwd = stage_dependencies.needs_combine_bwd
    needs_w2_bwd = stage_dependencies.needs_w2_bwd
    needs_w13_bwd = stage_dependencies.needs_w13_bwd

    rows: list[BenchResult] = []
    if requested("staged_forward"):
        try:
            emit_stage_progress("grug_moe_mlp_staged_forward")
            with jax.set_mesh(mesh):
                staged_compile, staged_time, staged_out = _time_jitted(
                    jax.jit(staged_forward_fn),
                    x,
                    selected_experts,
                    combine_weights,
                    w_up_gate,
                    w_down,
                    warmup=warmup,
                    steps=steps,
                )
            staged_dropped = int(np.asarray(staged_out[1]))
            staged_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            staged_compile = None
            staged_time = None
            staged_dropped = None
            staged_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_staged_forward",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "staged_forward"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=staged_compile,
                steady_state_time=staged_time,
                error=staged_error,
                dropped_routes=staged_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("permute_metadata"):
        try:
            emit_stage_progress("grug_moe_mlp_permute_metadata")
            with jax.set_mesh(mesh):
                metadata_compile, metadata_time, metadata_out = _time_jitted(
                    jax.jit(permute_metadata_fn),
                    selected_experts,
                    warmup=warmup,
                    steps=steps,
                )
            metadata_dropped = int(np.asarray(metadata_out[-1]))
            metadata_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            metadata_compile = None
            metadata_time = None
            metadata_dropped = None
            metadata_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_permute_metadata",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "permute_metadata"),
                dispatch_bytes=0.0,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=metadata_compile,
                steady_state_time=metadata_time,
                error=metadata_error,
                dropped_routes=metadata_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("permute_values"):
        try:
            emit_stage_progress("grug_moe_mlp_permute_values")
            with jax.set_mesh(mesh):
                values_compile, values_time, values_out = _time_jitted(
                    jax.jit(permute_values_fn),
                    x,
                    selected_experts,
                    warmup=warmup,
                    steps=steps,
                )
            values_dropped = int(np.asarray(values_out[-1]))
            values_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            values_compile = None
            values_time = None
            values_dropped = None
            values_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_permute_values",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "permute_values"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=values_compile,
                steady_state_time=values_time,
                error=values_error,
                dropped_routes=values_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("producer_consumer_copy"):
        try:
            emit_stage_progress("grug_moe_mlp_producer_consumer_copy")
            with jax.set_mesh(mesh):
                pc_compile, pc_time, pc_out = _time_jitted(
                    jax.jit(producer_consumer_copy_fn),
                    x,
                    warmup=warmup,
                    steps=steps,
                )
            pc_max_abs = float(np.asarray(pc_out[0]))
            pc_mean_abs = float(np.asarray(pc_out[1]))
            pc_matches = pc_max_abs == 0.0
            pc_error = None if pc_matches else f"max_abs_diff={pc_max_abs}"
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            pc_compile = None
            pc_time = None
            pc_error = f"{type(exc).__name__}: {exc}"
            pc_max_abs = None
            pc_mean_abs = None
            pc_matches = None

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_producer_consumer_copy",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "producer_consumer_copy"),
                dispatch_bytes=shape.tokens_per_rank * shape.hidden_dim * jnp.dtype(dtype).itemsize,
                return_bytes=shape.tokens_per_rank * shape.hidden_dim * jnp.dtype(dtype).itemsize,
                memory_footprint=memory_footprint,
                compile_time=pc_compile,
                steady_state_time=pc_time,
                error=pc_error,
                dropped_routes=None,
                baseline_implementation="input_copy",
                max_abs_diff=pc_max_abs,
                mean_abs_diff=pc_mean_abs,
                matches_baseline=pc_matches,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("split_wg_w13"):
        try:
            emit_stage_progress("grug_moe_mlp_split_wg_w13")
            with jax.set_mesh(mesh):
                split_compile, split_time, split_out = _time_jitted(
                    jax.jit(split_wg_w13_fn),
                    x,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
            split_max_abs = float(np.asarray(split_out[0]))
            split_mean_abs = float(np.asarray(split_out[1]))
            split_matches = split_max_abs <= allclose_atol
            split_error = None if split_matches else f"max_abs_diff={split_max_abs}"
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            split_compile = None
            split_time = None
            split_error = f"{type(exc).__name__}: {exc}"
            split_max_abs = None
            split_mean_abs = None
            split_matches = None

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_split_wg_w13",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "split_wg_w13"),
                dispatch_bytes=0.0,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=split_compile,
                steady_state_time=split_time,
                error=split_error,
                dropped_routes=None,
                baseline_implementation="jax_dot",
                max_abs_diff=split_max_abs,
                mean_abs_diff=split_mean_abs,
                matches_baseline=split_matches,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    up_out = None
    dropped = None
    if needs_permute_up:
        try:
            emit_stage_progress("grug_moe_mlp_permute_up")
            with jax.set_mesh(mesh):
                up_compile, up_time, up_out = _time_jitted(
                    jax.jit(permute_up_fn),
                    x,
                    selected_experts,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
            dropped = int(np.asarray(up_out[-1]))
            up_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            up_compile = None
            up_time = None
            up_out = None
            up_error = f"{type(exc).__name__}: {exc}"

        if requested("permute_up"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_permute_up",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "permute_up"),
                    dispatch_bytes=dispatch_bytes,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=up_compile,
                    steady_state_time=up_time,
                    error=up_error,
                    dropped_routes=dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    if requested("permute_up_compare"):
        try:
            emit_stage_progress("grug_moe_mlp_permute_up_compare")
            with jax.set_mesh(mesh):
                compare_compile, compare_time, compare_out = _time_jitted(
                    jax.jit(permute_up_compare_fn),
                    x,
                    selected_experts,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
            compare_max_abs = float(np.asarray(compare_out[0]))
            compare_mean_abs = float(np.asarray(compare_out[1]))
            rank_mismatches = int(np.asarray(compare_out[2]))
            assignment_mismatches = int(np.asarray(compare_out[3]))
            base_dropped = int(np.asarray(compare_out[4]))
            candidate_dropped = int(np.asarray(compare_out[5]))
            compare_error = None
            compare_matches = (
                rank_mismatches == 0
                and assignment_mismatches == 0
                and base_dropped == candidate_dropped
                and compare_max_abs <= allclose_atol
            )
            if not compare_matches:
                compare_error = (
                    f"rank_mismatches={rank_mismatches}, assignment_mismatches={assignment_mismatches}, "
                    f"base_dropped={base_dropped}, candidate_dropped={candidate_dropped}"
                )
            compare_dropped = candidate_dropped
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            compare_compile = None
            compare_time = None
            compare_error = f"{type(exc).__name__}: {exc}"
            compare_max_abs = None
            compare_mean_abs = None
            compare_matches = None
            compare_dropped = None

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_permute_up_compare",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "permute_up_compare"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=compare_compile,
                steady_state_time=compare_time,
                error=compare_error,
                dropped_routes=compare_dropped,
                baseline_implementation="pallas_mgpu_staged",
                max_abs_diff=compare_max_abs,
                mean_abs_diff=compare_mean_abs,
                matches_baseline=compare_matches,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("permute_up_compare_split"):
        try:
            emit_stage_progress("grug_moe_mlp_permute_up_compare_split")
            with jax.set_mesh(mesh):
                baseline_compile, baseline_time, baseline_out = _time_jitted(
                    jax.jit(permute_up_baseline_fn),
                    x,
                    selected_experts,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
                candidate_compile, candidate_time, candidate_out = _time_jitted(
                    jax.jit(permute_up_fn),
                    x,
                    selected_experts,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
            baseline_hidden, baseline_src_rank, baseline_assignment, _baseline_rows, baseline_dropped = baseline_out
            candidate_hidden, candidate_src_rank, candidate_assignment, _candidate_rows, candidate_dropped = (
                candidate_out
            )
            hidden_diff = np.abs(
                np.asarray(candidate_hidden, dtype=np.float32) - np.asarray(baseline_hidden, dtype=np.float32)
            )
            split_compare_max_abs = float(np.max(hidden_diff))
            split_compare_mean_abs = float(np.mean(hidden_diff))
            rank_mismatches = int(np.sum(np.asarray(candidate_src_rank) != np.asarray(baseline_src_rank)))
            assignment_mismatches = int(np.sum(np.asarray(candidate_assignment) != np.asarray(baseline_assignment)))
            base_dropped = int(np.asarray(baseline_dropped))
            candidate_dropped_int = int(np.asarray(candidate_dropped))
            split_compare_error = None
            split_compare_matches = (
                rank_mismatches == 0
                and assignment_mismatches == 0
                and base_dropped == candidate_dropped_int
                and split_compare_max_abs <= allclose_atol
            )
            if not split_compare_matches:
                split_compare_error = (
                    f"rank_mismatches={rank_mismatches}, assignment_mismatches={assignment_mismatches}, "
                    f"base_dropped={base_dropped}, candidate_dropped={candidate_dropped_int}"
                )
            split_compare_compile = baseline_compile + candidate_compile
            split_compare_time = baseline_time + candidate_time
            split_compare_dropped = candidate_dropped_int
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            split_compare_compile = None
            split_compare_time = None
            split_compare_error = f"{type(exc).__name__}: {exc}"
            split_compare_max_abs = None
            split_compare_mean_abs = None
            split_compare_matches = None
            split_compare_dropped = None

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_permute_up_compare_split",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "permute_up_compare_split"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=split_compare_compile,
                steady_state_time=split_compare_time,
                error=split_compare_error,
                dropped_routes=split_compare_dropped,
                baseline_implementation="pallas_mgpu_staged",
                max_abs_diff=split_compare_max_abs,
                mean_abs_diff=split_compare_mean_abs,
                matches_baseline=split_compare_matches,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    permute_out = None
    permute_dropped = None
    if needs_permute:
        try:
            emit_stage_progress("grug_moe_mlp_permute")
            with jax.set_mesh(mesh):
                permute_compile, permute_time, permute_out = _time_jitted(
                    jax.jit(permute_fn),
                    x,
                    selected_experts,
                    warmup=warmup,
                    steps=steps,
                )
            permute_dropped = int(np.asarray(permute_out[-1]))
            permute_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            permute_compile = None
            permute_time = None
            permute_out = None
            permute_error = f"{type(exc).__name__}: {exc}"

        if requested("permute"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_permute",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "permute"),
                    dispatch_bytes=dispatch_bytes,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=permute_compile,
                    steady_state_time=permute_time,
                    error=permute_error,
                    dropped_routes=permute_dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    if requested("w13"):
        try:
            if permute_out is None:
                raise ValueError("permute stage failed; W13-only stage was skipped")
            recv_x_only, _permute_src_rank, _permute_src_assignment, permute_rows_per_expert, _permute_dropped = (
                permute_out
            )
            emit_stage_progress("grug_moe_mlp_w13")
            with jax.set_mesh(mesh):
                w13_compile, w13_time, _w13_out = _time_jitted(
                    jax.jit(w13_fn),
                    recv_x_only,
                    permute_rows_per_expert,
                    w_up_gate,
                    warmup=warmup,
                    steps=steps,
                )
            w13_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            w13_compile = None
            w13_time = None
            w13_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_w13",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "w13"),
                dispatch_bytes=0.0,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=w13_compile,
                steady_state_time=w13_time,
                error=w13_error,
                dropped_routes=permute_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if up_out is None:
        hidden = None
        recv_src_rank = None
        recv_src_assignment = None
        rows_per_expert = None
    else:
        hidden, recv_src_rank, recv_src_assignment, rows_per_expert, _dropped = up_out
    _w2_out = None
    if needs_w2:
        try:
            emit_stage_progress("grug_moe_mlp_w2")
            with jax.set_mesh(mesh):
                w2_compile, w2_time, _w2_out = _time_jitted(
                    jax.jit(w2_fn),
                    hidden,
                    rows_per_expert,
                    w_down,
                    warmup=warmup,
                    steps=steps,
                )
            w2_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            w2_compile = None
            w2_time = None
            _w2_out = None
            w2_error = f"{type(exc).__name__}: {exc}"

        if requested("w2"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_w2",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "w2"),
                    dispatch_bytes=0.0,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=w2_compile,
                    steady_state_time=w2_time,
                    error=w2_error,
                    dropped_routes=dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    _return_slots_out = None
    if requested("return_slots") or requested("combine_slots"):
        try:
            if _w2_out is None:
                raise ValueError("w2 stage failed; return-slots stage was skipped")
            emit_stage_progress("grug_moe_mlp_return_slots")
            with jax.set_mesh(mesh):
                return_slots_compile, return_slots_time, _return_slots_out = _time_jitted(
                    jax.jit(return_slots_fn),
                    _w2_out,
                    recv_src_rank,
                    recv_src_assignment,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            return_slots_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            return_slots_compile = None
            return_slots_time = None
            _return_slots_out = None
            return_slots_error = f"{type(exc).__name__}: {exc}"

        if requested("return_slots"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_return_slots",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "return_slots"),
                    dispatch_bytes=0.0,
                    return_bytes=return_bytes,
                    memory_footprint=memory_footprint,
                    compile_time=return_slots_compile,
                    steady_state_time=return_slots_time,
                    error=return_slots_error,
                    dropped_routes=dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    if requested("combine_slots"):
        try:
            if _return_slots_out is None:
                raise ValueError("return-slots stage failed; combine-only stage was skipped")
            emit_stage_progress("grug_moe_mlp_combine_slots")
            with jax.set_mesh(mesh):
                combine_slots_compile, combine_slots_time, _combine_slots_out = _time_jitted(
                    jax.jit(combine_slots_fn),
                    _return_slots_out,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            combine_slots_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            combine_slots_compile = None
            combine_slots_time = None
            combine_slots_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_combine_slots",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "combine_slots"),
                dispatch_bytes=0.0,
                return_bytes=0.0,
                memory_footprint=memory_footprint,
                compile_time=combine_slots_compile,
                steady_state_time=combine_slots_time,
                error=combine_slots_error,
                dropped_routes=dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    _return_combine_out = None
    if requested("return_combine") or requested("pull_combine"):
        try:
            if _w2_out is None:
                raise ValueError("w2 stage failed; return/combine stage was skipped")
            emit_stage_progress("grug_moe_mlp_return_combine")
            with jax.set_mesh(mesh):
                return_combine_compile, return_combine_time, _return_combine_out = _time_jitted(
                    jax.jit(return_combine_fn),
                    _w2_out,
                    recv_src_rank,
                    recv_src_assignment,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            return_combine_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            return_combine_compile = None
            return_combine_time = None
            _return_combine_out = None
            return_combine_error = f"{type(exc).__name__}: {exc}"

        if requested("return_combine"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_return_combine",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "return_combine"),
                    dispatch_bytes=0.0,
                    return_bytes=return_bytes,
                    memory_footprint=memory_footprint,
                    compile_time=return_combine_compile,
                    steady_state_time=return_combine_time,
                    error=return_combine_error,
                    dropped_routes=dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    _pull_combine_out = None
    if requested("pull_combine") or requested("pull_combine_vector"):
        try:
            if _w2_out is None:
                raise ValueError("w2 stage failed; pull/combine stage was skipped")
            emit_stage_progress("grug_moe_mlp_pull_combine")
            with jax.set_mesh(mesh):
                pull_combine_compile, pull_combine_time, _pull_combine_out = _time_jitted(
                    jax.jit(pull_combine_fn),
                    _w2_out,
                    selected_experts,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            pull_combine_error = None
            pull_max_abs_diff = None
            pull_mean_abs_diff = None
            pull_matches = None
            if _return_combine_out is not None:
                diff = jnp.abs(_pull_combine_out.astype(jnp.float32) - _return_combine_out.astype(jnp.float32))
                pull_max_abs_diff = float(jnp.max(diff).block_until_ready())
                pull_mean_abs_diff = float(jnp.mean(diff).block_until_ready())
                tolerance = allclose_atol + allclose_rtol * jnp.abs(_return_combine_out.astype(jnp.float32))
                pull_matches = bool(jnp.all(diff <= tolerance).block_until_ready())
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            pull_combine_compile = None
            pull_combine_time = None
            _pull_combine_out = None
            pull_combine_error = f"{type(exc).__name__}: {exc}"
            pull_max_abs_diff = None
            pull_mean_abs_diff = None
            pull_matches = None

        if requested("pull_combine"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_pull_combine",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "pull_combine"),
                    dispatch_bytes=0.0,
                    return_bytes=return_bytes,
                    memory_footprint=memory_footprint,
                    compile_time=pull_combine_compile,
                    steady_state_time=pull_combine_time,
                    error=pull_combine_error,
                    dropped_routes=dropped,
                    baseline_implementation="grug_moe_mlp_return_combine" if _return_combine_out is not None else None,
                    max_abs_diff=pull_max_abs_diff,
                    mean_abs_diff=pull_mean_abs_diff,
                    matches_baseline=pull_matches,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    if requested("pull_combine_vector"):
        try:
            if _w2_out is None:
                raise ValueError("w2 stage failed; vector pull/combine stage was skipped")
            emit_stage_progress("grug_moe_mlp_pull_combine_vector")
            with jax.set_mesh(mesh):
                pull_vector_compile, pull_vector_time, _pull_vector_out = _time_jitted(
                    jax.jit(pull_combine_vector_fn),
                    _w2_out,
                    selected_experts,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            pull_vector_error = None
            pull_vector_max_abs_diff = None
            pull_vector_mean_abs_diff = None
            pull_vector_matches = None
            if _pull_combine_out is not None:
                diff = jnp.abs(_pull_vector_out.astype(jnp.float32) - _pull_combine_out.astype(jnp.float32))
                pull_vector_max_abs_diff = float(jnp.max(diff).block_until_ready())
                pull_vector_mean_abs_diff = float(jnp.mean(diff).block_until_ready())
                tolerance = allclose_atol + allclose_rtol * jnp.abs(_pull_combine_out.astype(jnp.float32))
                pull_vector_matches = bool(jnp.all(diff <= tolerance).block_until_ready())
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            pull_vector_compile = None
            pull_vector_time = None
            pull_vector_error = f"{type(exc).__name__}: {exc}"
            pull_vector_max_abs_diff = None
            pull_vector_mean_abs_diff = None
            pull_vector_matches = None

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_pull_combine_vector",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "pull_combine_vector"),
                dispatch_bytes=0.0,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=pull_vector_compile,
                steady_state_time=pull_vector_time,
                error=pull_vector_error,
                dropped_routes=dropped,
                baseline_implementation="grug_moe_mlp_pull_combine" if _pull_combine_out is not None else None,
                max_abs_diff=pull_vector_max_abs_diff,
                mean_abs_diff=pull_vector_mean_abs_diff,
                matches_baseline=pull_vector_matches,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("down_unpermute"):
        try:
            emit_stage_progress("grug_moe_mlp_down_unpermute")
            with jax.set_mesh(mesh):
                down_compile, down_time, _down_out = _time_jitted(
                    jax.jit(down_unpermute_fn),
                    hidden,
                    recv_src_rank,
                    recv_src_assignment,
                    rows_per_expert,
                    w_down,
                    combine_weights,
                    selected_experts,
                    warmup=warmup,
                    steps=steps,
                )
            down_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            down_compile = None
            down_time = None
            down_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_down_unpermute",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "down_unpermute"),
                dispatch_bytes=0.0,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=down_compile,
                steady_state_time=down_time,
                error=down_error,
                dropped_routes=dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    backward_prereq_out = None
    backward_dropped = None
    if needs_backward_prereq:
        try:
            emit_stage_progress("grug_moe_mlp_backward_prereq")
            with jax.set_mesh(mesh):
                backward_prereq_compile, backward_prereq_time, backward_prereq_out = _time_jitted(
                    jax.jit(backward_prereq_fn),
                    x,
                    selected_experts,
                    w_up_gate,
                    w_down,
                    warmup=warmup,
                    steps=steps,
                )
            backward_dropped = int(np.asarray(backward_prereq_out[-1]))
            backward_prereq_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            backward_prereq_compile = None
            backward_prereq_time = None
            backward_prereq_out = None
            backward_prereq_error = f"{type(exc).__name__}: {exc}"

        if requested("backward_prereq"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_backward_prereq",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "backward_prereq"),
                    dispatch_bytes=dispatch_bytes,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=backward_prereq_compile,
                    steady_state_time=backward_prereq_time,
                    error=backward_prereq_error,
                    dropped_routes=backward_dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    dy_dispatch = None
    if needs_combine_bwd:
        try:
            if backward_prereq_out is None:
                raise ValueError("backward prereq stage failed; combine-bwd stage was skipped")
            if target_for_stages is None:
                raise ValueError("target is required for combine-bwd stage")
            (
                _backward_recv_x,
                _backward_recv_src_rank,
                _backward_recv_src_assignment,
                _backward_rows_per_expert,
                _backward_hidden,
                backward_y_dispatch,
                _backward_dropped_array,
            ) = backward_prereq_out
            emit_stage_progress("grug_moe_mlp_combine_bwd")
            with jax.set_mesh(mesh):
                combine_bwd_compile, combine_bwd_time, combine_bwd_out = _time_jitted(
                    jax.jit(combine_bwd_fn),
                    backward_y_dispatch,
                    target_for_stages,
                    selected_experts,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            dy_dispatch, _dcombine_weights = combine_bwd_out
            combine_bwd_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            combine_bwd_compile = None
            combine_bwd_time = None
            dy_dispatch = None
            combine_bwd_error = f"{type(exc).__name__}: {exc}"

        if requested("combine_bwd"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_combine_bwd",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "combine_bwd"),
                    dispatch_bytes=0.0,
                    return_bytes=return_bytes,
                    memory_footprint=memory_footprint,
                    compile_time=combine_bwd_compile,
                    steady_state_time=combine_bwd_time,
                    error=combine_bwd_error,
                    dropped_routes=backward_dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    dhidden = None
    if needs_w2_bwd:
        try:
            if backward_prereq_out is None or dy_dispatch is None:
                raise ValueError("combine-bwd stage failed; W2-bwd stage was skipped")
            (
                _backward_recv_x,
                _backward_recv_src_rank,
                _backward_recv_src_assignment,
                backward_rows_per_expert,
                backward_hidden,
                _backward_y_dispatch,
                _backward_dropped_array,
            ) = backward_prereq_out
            emit_stage_progress("grug_moe_mlp_w2_bwd")
            with jax.set_mesh(mesh):
                w2_bwd_compile, w2_bwd_time, w2_bwd_out = _time_jitted(
                    jax.jit(w2_bwd_fn),
                    backward_hidden,
                    backward_rows_per_expert,
                    w_down,
                    dy_dispatch,
                    warmup=warmup,
                    steps=steps,
                )
            dhidden, _dmoe_w2 = w2_bwd_out
            w2_bwd_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            w2_bwd_compile = None
            w2_bwd_time = None
            dhidden = None
            w2_bwd_error = f"{type(exc).__name__}: {exc}"

        if requested("w2_bwd"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_w2_bwd",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "w2_bwd"),
                    dispatch_bytes=0.0,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=w2_bwd_compile,
                    steady_state_time=w2_bwd_time,
                    error=w2_bwd_error,
                    dropped_routes=backward_dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    drecv_x = None
    if needs_w13_bwd:
        try:
            if backward_prereq_out is None or dhidden is None:
                raise ValueError("W2-bwd stage failed; W13-bwd stage was skipped")
            (
                backward_recv_x,
                _backward_recv_src_rank,
                _backward_recv_src_assignment,
                backward_rows_per_expert,
                _backward_hidden,
                _backward_y_dispatch,
                _backward_dropped_array,
            ) = backward_prereq_out
            emit_stage_progress("grug_moe_mlp_w13_bwd")
            with jax.set_mesh(mesh):
                w13_bwd_compile, w13_bwd_time, w13_bwd_out = _time_jitted(
                    jax.jit(w13_bwd_fn),
                    backward_recv_x,
                    backward_rows_per_expert,
                    w_up_gate,
                    dhidden,
                    warmup=warmup,
                    steps=steps,
                )
            drecv_x, _dmoe_w13 = w13_bwd_out
            w13_bwd_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            w13_bwd_compile = None
            w13_bwd_time = None
            drecv_x = None
            w13_bwd_error = f"{type(exc).__name__}: {exc}"

        if requested("w13_bwd"):
            rows.append(
                _result_row(
                    kernel="grug_moe_mlp_w13_bwd",
                    implementation="pallas_mgpu",
                    shape=shape,
                    config=config,
                    dtype=dtype,
                    routing=routing,
                    warmup=warmup,
                    steps=steps,
                    git_sha=git_sha,
                    flops=_stage_flops(shape, "w13_bwd"),
                    dispatch_bytes=0.0,
                    return_bytes=0.0,
                    memory_footprint=memory_footprint,
                    compile_time=w13_bwd_compile,
                    steady_state_time=w13_bwd_time,
                    error=w13_bwd_error,
                    dropped_routes=backward_dropped,
                    baseline_implementation=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    matches_baseline=None,
                    allclose_rtol=allclose_rtol,
                    allclose_atol=allclose_atol,
                )
            )

    if requested("dx_unpermute_vector"):
        try:
            if backward_prereq_out is None or drecv_x is None:
                raise ValueError("W13-bwd stage failed; dx-unpermute-vector stage was skipped")
            (
                _backward_recv_x,
                backward_recv_src_rank,
                backward_recv_src_assignment,
                _backward_rows_per_expert,
                _backward_hidden,
                _backward_y_dispatch,
                _backward_dropped_array,
            ) = backward_prereq_out
            emit_stage_progress("grug_moe_mlp_dx_unpermute_vector")
            with jax.set_mesh(mesh):
                dx_compile, dx_time, _dx_out = _time_jitted(
                    jax.jit(dx_unpermute_vector_fn),
                    drecv_x,
                    backward_recv_src_rank,
                    backward_recv_src_assignment,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            dx_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            dx_compile = None
            dx_time = None
            dx_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_dx_unpermute_vector",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "dx_unpermute_vector"),
                dispatch_bytes=0.0,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=dx_compile,
                steady_state_time=dx_time,
                error=dx_error,
                dropped_routes=backward_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("dx_pull_combine_vector"):
        try:
            if drecv_x is None:
                raise ValueError("W13-bwd stage failed; dx-pull-combine-vector stage was skipped")
            emit_stage_progress("grug_moe_mlp_dx_pull_combine_vector")
            with jax.set_mesh(mesh):
                dx_pull_compile, dx_pull_time, _dx_pull_out = _time_jitted(
                    jax.jit(dx_pull_combine_vector_fn),
                    drecv_x,
                    selected_experts,
                    combine_weights,
                    warmup=warmup,
                    steps=steps,
                )
            dx_pull_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            dx_pull_compile = None
            dx_pull_time = None
            dx_pull_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_dx_pull_combine_vector",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "dx_pull_combine_vector"),
                dispatch_bytes=0.0,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=dx_pull_compile,
                steady_state_time=dx_pull_time,
                error=dx_pull_error,
                dropped_routes=backward_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("saved_backward_pipeline"):
        try:
            if backward_prereq_out is None:
                raise ValueError("backward prereq stage failed; saved-backward-pipeline stage was skipped")
            if target_for_stages is None:
                raise ValueError("target is required for saved backward pipeline stage")
            (
                backward_recv_x,
                backward_recv_src_rank,
                backward_recv_src_assignment,
                backward_rows_per_expert,
                backward_hidden,
                backward_y_dispatch,
                _backward_dropped_array,
            ) = backward_prereq_out
            emit_stage_progress("grug_moe_mlp_saved_backward_pipeline")
            with jax.set_mesh(mesh):
                saved_bwd_compile, saved_bwd_time, _saved_bwd_out = _time_jitted(
                    jax.jit(saved_backward_pipeline_fn),
                    backward_recv_x,
                    backward_recv_src_rank,
                    backward_recv_src_assignment,
                    backward_rows_per_expert,
                    backward_hidden,
                    backward_y_dispatch,
                    selected_experts,
                    combine_weights,
                    w_up_gate,
                    w_down,
                    target_for_stages,
                    warmup=warmup,
                    steps=steps,
                )
            saved_bwd_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            saved_bwd_compile = None
            saved_bwd_time = None
            saved_bwd_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_saved_backward_pipeline",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "saved_backward_pipeline"),
                dispatch_bytes=0.0,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=saved_bwd_compile,
                steady_state_time=saved_bwd_time,
                error=saved_bwd_error,
                dropped_routes=backward_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("manual_backward_pipeline"):
        try:
            if target_for_stages is None:
                raise ValueError("target is required for manual backward pipeline stage")
            emit_stage_progress("grug_moe_mlp_manual_backward_pipeline")
            with jax.set_mesh(mesh):
                manual_compile, manual_time, manual_out = _time_jitted(
                    jax.jit(manual_backward_pipeline_fn),
                    x,
                    selected_experts,
                    combine_weights,
                    w_up_gate,
                    w_down,
                    target_for_stages,
                    warmup=warmup,
                    steps=steps,
                )
            manual_dropped = int(np.asarray(manual_out[-1]))
            manual_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            manual_compile = None
            manual_time = None
            manual_dropped = None
            manual_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_manual_backward_pipeline",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "manual_backward_pipeline"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=manual_compile,
                steady_state_time=manual_time,
                error=manual_error,
                dropped_routes=manual_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )

    if requested("manual_forward_backward_pipeline"):
        try:
            if target_for_stages is None:
                raise ValueError("target is required for manual forward/backward pipeline stage")
            emit_stage_progress("grug_moe_mlp_manual_forward_backward_pipeline")
            with jax.set_mesh(mesh):
                manual_fb_compile, manual_fb_time, manual_fb_out = _time_jitted(
                    jax.jit(manual_forward_backward_pipeline_fn),
                    x,
                    selected_experts,
                    combine_weights,
                    w_up_gate,
                    w_down,
                    target_for_stages,
                    warmup=warmup,
                    steps=steps,
                )
            manual_fb_dropped = int(np.asarray(manual_fb_out[-1]))
            manual_fb_error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
            manual_fb_compile = None
            manual_fb_time = None
            manual_fb_dropped = None
            manual_fb_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            _result_row(
                kernel="grug_moe_mlp_manual_forward_backward_pipeline",
                implementation="pallas_mgpu",
                shape=shape,
                config=config,
                dtype=dtype,
                routing=routing,
                warmup=warmup,
                steps=steps,
                git_sha=git_sha,
                flops=_stage_flops(shape, "manual_forward_backward_pipeline"),
                dispatch_bytes=dispatch_bytes,
                return_bytes=return_bytes,
                memory_footprint=memory_footprint,
                compile_time=manual_fb_compile,
                steady_state_time=manual_fb_time,
                error=manual_fb_error,
                dropped_routes=manual_fb_dropped,
                baseline_implementation=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                matches_baseline=None,
                allclose_rtol=allclose_rtol,
                allclose_atol=allclose_atol,
            )
        )
    return rows


def _parse_args() -> argparse.Namespace:
    default_config = MoeMgpuConfig()
    parser = argparse.ArgumentParser(description="Benchmark Grug MoE EP backends, including Hopper Pallas MGPU.")
    parser.add_argument("--tokens-per-rank", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--intermediate-dim", type=int, default=128)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=default_config.capacity_factor)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--max-concurrent-steps", type=int, default=default_config.max_concurrent_steps)
    parser.add_argument("--grid-block-n", type=int, default=default_config.grid_block_n)
    parser.add_argument(
        "--routing",
        choices=("uniform", "balanced", "all_local", "all_to_rank0", "all_remote_next"),
        default="uniform",
    )
    parser.add_argument(
        "--dispatch-copy-schedule",
        choices=sorted(_DISPATCH_COPY_SCHEDULES),
        default="assignment_major",
    )
    parser.add_argument("--dispatch-expert-group-size", type=int, default=8)
    parser.add_argument("--dispatch-chunk-copy-tile", type=int, default=default_config.dispatch_chunk_copy_tile)
    parser.add_argument("--dispatch-chunk-copy-rows", type=int, default=default_config.dispatch_chunk_copy_rows)
    parser.add_argument(
        "--dispatch-chunk-vectorized-copy-rows",
        action=argparse.BooleanOptionalAction,
        default=default_config.dispatch_chunk_vectorized_copy_rows,
    )
    parser.add_argument(
        "--dispatch-fuse-metadata",
        action=argparse.BooleanOptionalAction,
        default=default_config.dispatch_fuse_metadata,
    )
    parser.add_argument(
        "--dispatch-chunked-permute-up",
        action=argparse.BooleanOptionalAction,
        default=default_config.dispatch_chunked_permute_up,
    )
    parser.add_argument(
        "--dispatch-split-wg-permute-up",
        action=argparse.BooleanOptionalAction,
        default=default_config.dispatch_split_wg_permute_up,
    )
    parser.add_argument(
        "--dispatch-split-wg-overlap-permute-up",
        action=argparse.BooleanOptionalAction,
        default=default_config.dispatch_split_wg_overlap_permute_up,
    )
    parser.add_argument("--combine-bwd-block-n", type=int, default=default_config.combine_bwd_block_n)
    parser.add_argument("--dx-unpermute-block-n", type=int, default=default_config.dx_unpermute_block_n)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=_IMPLEMENTATION_CHOICES,
        default=["ragged_all_to_all", "pallas_mgpu"],
        help="Forward implementations to benchmark. Use 'none' with --include-pallas-stages for stage-only runs.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--candidate-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional best-effort wall-clock timeout for each top-level implementation candidate. "
            "Timeouts are emitted as structured error rows; native XLA/JAX work may not interrupt immediately."
        ),
    )
    parser.add_argument(
        "--pass-mode",
        choices=_PASS_MODE_CHOICES,
        default="forward",
        help="Benchmark forward only or scalar-loss forward+backward for each implementation.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit nonzero after emitting rows if any benchmark result row has status other than ok.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--allclose-rtol", type=float, default=1e-2)
    parser.add_argument("--allclose-atol", type=float, default=0.1)
    parser.add_argument("--include-pallas-stages", action="store_true")
    parser.add_argument(
        "--pallas-stages",
        nargs="+",
        default=None,
        metavar="STAGE",
        help=(
            "Optional subset of Pallas diagnostic stages to emit when --include-pallas-stages is set. "
            "Values may be space-separated or comma-separated."
        ),
    )
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    args = parser.parse_args()
    duplicate_implementations = _duplicate_values(args.implementations)
    if duplicate_implementations:
        duplicate_values = ", ".join(duplicate_implementations)
        parser.error(f"--implementations contains duplicate value(s): {duplicate_values}")
    if "none" in args.implementations and len(args.implementations) > 1:
        parser.error("--implementations none cannot be combined with other implementations")
    if args.implementations == ["none"] and not args.include_pallas_stages:
        parser.error("--implementations none requires --include-pallas-stages")
    _validate_chunked_permute_up_args(args, parser)
    if args.dispatch_split_wg_permute_up and not args.dispatch_chunked_permute_up:
        parser.error("--dispatch-split-wg-permute-up requires --dispatch-chunked-permute-up")
    if args.dispatch_split_wg_overlap_permute_up and not args.dispatch_split_wg_permute_up:
        parser.error("--dispatch-split-wg-overlap-permute-up requires --dispatch-split-wg-permute-up")
    if args.candidate_timeout_seconds is not None and args.candidate_timeout_seconds <= 0.0:
        parser.error("--candidate-timeout-seconds must be positive when set")
    try:
        args.pallas_stages = _normalize_pallas_stage_args(args.pallas_stages)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = _parse_args()
    shape = BenchShape(
        tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        experts_per_rank=args.experts_per_rank,
        topk=args.topk,
        ep_size=args.ep_size,
        capacity_factor=args.capacity_factor,
    )
    config = MoeMgpuConfig(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        max_concurrent_steps=args.max_concurrent_steps,
        grid_block_n=args.grid_block_n,
        capacity_factor=args.capacity_factor,
        dispatch_copy_schedule=args.dispatch_copy_schedule,
        dispatch_expert_group_size=args.dispatch_expert_group_size,
        dispatch_chunk_copy_tile=args.dispatch_chunk_copy_tile,
        dispatch_chunk_copy_rows=args.dispatch_chunk_copy_rows,
        dispatch_chunk_vectorized_copy_rows=args.dispatch_chunk_vectorized_copy_rows,
        dispatch_fuse_metadata=args.dispatch_fuse_metadata,
        dispatch_chunked_permute_up=args.dispatch_chunked_permute_up,
        dispatch_split_wg_permute_up=args.dispatch_split_wg_permute_up,
        dispatch_split_wg_overlap_permute_up=args.dispatch_split_wg_overlap_permute_up,
        combine_bwd_block_n=args.combine_bwd_block_n,
        dx_unpermute_block_n=args.dx_unpermute_block_n,
    )
    dtype = jnp.bfloat16
    mesh = _make_mesh(shape.ep_size)
    inputs = _reshard_inputs(
        mesh,
        _make_inputs(shape=shape, routing=args.routing, dtype=dtype, scale=args.scale, seed=args.seed),
    )
    pallas_stages = frozenset(args.pallas_stages) if args.pallas_stages is not None else None
    target = None
    if args.pass_mode == "forward_backward" or (
        args.include_pallas_stages and _pallas_backward_stage_requested(pallas_stages)
    ):
        target = _reshard_target(mesh, _make_target(shape=shape, dtype=dtype, seed=args.seed + 10_000))
    git_sha = _git_sha(args.git_sha)

    jsonl_handle = None
    if args.jsonl is not None:
        parent = os.path.dirname(os.path.abspath(args.jsonl))
        if parent:
            os.makedirs(parent, exist_ok=True)
        jsonl_handle = open(args.jsonl, "w", encoding="utf-8")

    baseline_output = None
    baseline_implementation = None
    seen_measurement_keys: set[str] = set()
    result_rows: list[BenchResult] = []
    emitted_rows = 0
    expected_rows = _expected_result_count(
        implementations=args.implementations,
        include_pallas_stages=args.include_pallas_stages,
        pallas_stages=pallas_stages,
    )
    try:
        if args.implementations != ["none"]:
            for implementation in args.implementations:
                row, output = _benchmark_implementation(
                    implementation=implementation,
                    shape=shape,
                    config=config,
                    mesh=mesh,
                    inputs=inputs,
                    target=target,
                    pass_mode=args.pass_mode,
                    baseline_output=baseline_output,
                    baseline_implementation=baseline_implementation,
                    routing=args.routing,
                    dtype=dtype,
                    warmup=args.warmup,
                    steps=args.steps,
                    git_sha=git_sha,
                    allclose_rtol=args.allclose_rtol,
                    allclose_atol=args.allclose_atol,
                    candidate_timeout_seconds=args.candidate_timeout_seconds,
                )
                _emit_result(row, jsonl_handle, seen_measurement_keys)
                result_rows.append(row)
                emitted_rows += 1
                if baseline_output is None and output is not None:
                    baseline_output = output
                    baseline_implementation = implementation
        if args.include_pallas_stages:
            for row in _benchmark_pallas_stages(
                shape=shape,
                config=config,
                mesh=mesh,
                inputs=inputs,
                target=target,
                routing=args.routing,
                dtype=dtype,
                warmup=args.warmup,
                steps=args.steps,
                git_sha=git_sha,
                allclose_rtol=args.allclose_rtol,
                allclose_atol=args.allclose_atol,
                pallas_stages=pallas_stages,
            ):
                _emit_result(row, jsonl_handle, seen_measurement_keys)
                result_rows.append(row)
                emitted_rows += 1
        if emitted_rows != expected_rows:
            raise ValueError(f"benchmark emitted {emitted_rows} result rows, expected {expected_rows}")
        if args.fail_on_error:
            _raise_for_unsuccessful_results(result_rows)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()


if __name__ == "__main__":
    main()
