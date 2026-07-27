# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct EP8 transport gate for NCCL UB-X against Marin's exact Ring layout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import ubx
except (ImportError, OSError):
    torch = None
    dist = None
    ubx = None

UBX_SOURCE_COMMIT = "db0c814185a0415cc2e23dca387fecb9282de551"
EP_SIZE = 8
DEFAULT_TOKENS_PER_RANK = 16_384
DEFAULT_NUM_EXPERTS = 64
DEFAULT_TOP_K = 4
DEFAULT_HIDDEN_DIM = 2_560
DEFAULT_SKEW_ALPHA = 1.2
DEFAULT_RELATIVE_L2_LIMIT = 0.002
DEFAULT_REQUIRED_SPEEDUP = 1.10
_POOL_ALIGNMENT = 2 * 1024 * 1024
_POOL_HEADROOM = 256 * 1024 * 1024


@dataclass(frozen=True)
class BenchmarkConfig:
    tokens_per_rank: int = DEFAULT_TOKENS_PER_RANK
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_k: int = DEFAULT_TOP_K
    capacity_factor: float = 1.0
    routing: str = "balanced"
    skew_alpha: float = DEFAULT_SKEW_ALPHA
    seed: int = 0
    warmup: int = 5
    iterations: int = 30
    relative_l2_limit: float = DEFAULT_RELATIVE_L2_LIMIT
    required_speedup: float = DEFAULT_REQUIRED_SPEEDUP

    @property
    def global_tokens(self) -> int:
        return self.tokens_per_rank * EP_SIZE

    @property
    def assignments_per_rank(self) -> int:
        return self.tokens_per_rank * self.top_k

    @property
    def global_assignments(self) -> int:
        return self.global_tokens * self.top_k

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // EP_SIZE

    @property
    def capacity_per_expert_rank(self) -> int:
        return max(
            self.experts_per_rank,
            math.ceil(self.capacity_factor * self.global_assignments / EP_SIZE),
        )


@dataclass(frozen=True)
class RoutePlan:
    selected_experts: np.ndarray
    assignment_weights: np.ndarray
    accepted_assignments: np.ndarray
    routing: np.ndarray
    gate_weights_by_expert: np.ndarray
    original_counts: np.ndarray
    accepted_counts: np.ndarray
    drops_by_expert_rank: np.ndarray
    route_digest: str


@dataclass(frozen=True)
class ReferenceMaps:
    token_offsets: np.ndarray
    dispatch_topk_expert: np.ndarray
    dispatch_topk_slot: np.ndarray
    inverse_map: np.ndarray
    topk_idx: np.ndarray
    valid_slots: np.ndarray
    source_tokens: np.ndarray
    max_tokens_per_rank: int


def _validate_config(config: BenchmarkConfig) -> None:
    positive_ints = {
        "tokens_per_rank": config.tokens_per_rank,
        "hidden_dim": config.hidden_dim,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "iterations": config.iterations,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {config.warmup}")
    if config.num_experts % EP_SIZE:
        raise ValueError(f"num_experts={config.num_experts} must be divisible by EP size {EP_SIZE}")
    if config.top_k > config.num_experts:
        raise ValueError(f"top_k={config.top_k} must be <= num_experts={config.num_experts}")
    if config.hidden_dim % 32:
        raise ValueError(f"hidden_dim={config.hidden_dim} must be divisible by 32 for UB-X")
    if config.capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {config.capacity_factor}")
    if config.skew_alpha <= 0:
        raise ValueError(f"skew_alpha must be positive, got {config.skew_alpha}")
    if config.routing not in {"balanced", "learned_skew"}:
        raise ValueError(f"unknown routing mode {config.routing!r}")
    if config.relative_l2_limit <= 0:
        raise ValueError("relative_l2_limit must be positive")
    if config.required_speedup <= 0:
        raise ValueError("required_speedup must be positive")


def _balanced_selected_experts(config: BenchmarkConfig) -> np.ndarray:
    assignments = np.arange(config.global_assignments, dtype=np.int64)
    return (assignments % config.num_experts).reshape(config.global_tokens, config.top_k).astype(np.int32)


def _learned_skew_selected_experts(config: BenchmarkConfig) -> np.ndarray:
    """Generate distinct top-k choices from deterministic skewed router logits."""
    rng = np.random.default_rng(config.seed)
    popularity_order = rng.permutation(config.num_experts)
    popularity_rank = np.empty(config.num_experts, dtype=np.int64)
    popularity_rank[popularity_order] = np.arange(1, config.num_experts + 1)
    expert_bias = -config.skew_alpha * np.log(popularity_rank.astype(np.float64))

    selected = np.empty((config.global_tokens, config.top_k), dtype=np.int32)
    chunk_size = 4096
    for start in range(0, config.global_tokens, chunk_size):
        stop = min(start + chunk_size, config.global_tokens)
        # Gumbel top-k is weighted sampling without replacement. It is a
        # deterministic proxy for learned router logits with a persistent
        # expert-popularity bias, while preserving top-k distinctness.
        logits = expert_bias[None, :] + rng.gumbel(size=(stop - start, config.num_experts))
        candidates = np.argpartition(logits, -config.top_k, axis=1)[:, -config.top_k :]
        candidate_logits = np.take_along_axis(logits, candidates, axis=1)
        order = np.argsort(-candidate_logits, axis=1)
        selected[start:stop] = np.take_along_axis(candidates, order, axis=1).astype(np.int32)
    return selected


def _assignment_weights(config: BenchmarkConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 17)
    logits = rng.normal(size=(config.global_tokens, config.top_k)).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _cap_routes_like_ring(selected_experts: np.ndarray, config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Apply Ring's expert-order, source-major prefix cap independently per EP rank."""
    flat_selected = selected_experts.reshape(-1)
    accepted_flat = np.zeros(flat_selected.shape, dtype=np.bool_)
    drops_by_rank = np.zeros(EP_SIZE, dtype=np.int64)

    for expert_rank in range(EP_SIZE):
        remaining = config.capacity_per_expert_rank
        original_on_rank = 0
        expert_start = expert_rank * config.experts_per_rank
        for expert in range(expert_start, expert_start + config.experts_per_rank):
            positions = np.flatnonzero(flat_selected == expert)
            original_on_rank += positions.size
            take = min(positions.size, remaining)
            accepted_flat[positions[:take]] = True
            remaining -= take
        drops_by_rank[expert_rank] = original_on_rank - (config.capacity_per_expert_rank - remaining)

    return accepted_flat.reshape(selected_experts.shape), drops_by_rank


def build_route_plan(config: BenchmarkConfig) -> RoutePlan:
    _validate_config(config)
    if config.routing == "balanced":
        selected = _balanced_selected_experts(config)
    else:
        selected = _learned_skew_selected_experts(config)

    if np.any(np.sort(selected, axis=1)[:, 1:] == np.sort(selected, axis=1)[:, :-1]):
        raise AssertionError("routing fixture produced duplicate experts within a token")

    weights = _assignment_weights(config)
    accepted, drops_by_rank = _cap_routes_like_ring(selected, config)
    flat_selected = selected.reshape(-1)
    flat_accepted = accepted.reshape(-1)
    token_rows = np.repeat(np.arange(config.global_tokens, dtype=np.int64), config.top_k)

    routing = np.zeros((config.global_tokens, config.num_experts), dtype=np.uint8)
    routing[token_rows[flat_accepted], flat_selected[flat_accepted]] = 1
    gate_weights = np.zeros((config.global_tokens, config.num_experts), dtype=np.float32)
    gate_weights[token_rows[flat_accepted], flat_selected[flat_accepted]] = weights.reshape(-1)[flat_accepted]

    original_counts = np.bincount(flat_selected, minlength=config.num_experts).astype(np.int64)
    accepted_counts = routing.sum(axis=0, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(selected.tobytes())
    digest.update(accepted.tobytes())

    if int(accepted_counts.sum() + drops_by_rank.sum()) != config.global_assignments:
        raise AssertionError("accepted and dropped assignments do not reconstruct the original routing")
    for rank in range(EP_SIZE):
        expert_slice = slice(rank * config.experts_per_rank, (rank + 1) * config.experts_per_rank)
        expected_drop = int(original_counts[expert_slice].sum() - accepted_counts[expert_slice].sum())
        if int(drops_by_rank[rank]) != expected_drop:
            raise AssertionError(f"drop accounting mismatch on expert rank {rank}")

    return RoutePlan(
        selected_experts=selected,
        assignment_weights=weights,
        accepted_assignments=accepted,
        routing=routing,
        gate_weights_by_expert=gate_weights,
        original_counts=original_counts,
        accepted_counts=accepted_counts,
        drops_by_expert_rank=drops_by_rank,
        route_digest=digest.hexdigest(),
    )


def route_plan_summary(plan: RoutePlan, config: BenchmarkConfig) -> dict[str, Any]:
    accepted_by_rank = plan.accepted_counts.reshape(EP_SIZE, config.experts_per_rank).sum(axis=1)
    return {
        "routing": config.routing,
        "route_digest": plan.route_digest,
        "global_tokens": config.global_tokens,
        "tokens_per_source_rank": config.tokens_per_rank,
        "global_assignments": config.global_assignments,
        "assignments_per_source_rank": config.assignments_per_rank,
        "capacity_per_expert_rank": config.capacity_per_expert_rank,
        "original_count_min": int(plan.original_counts.min()),
        "original_count_max": int(plan.original_counts.max()),
        "accepted_count_min": int(plan.accepted_counts.min()),
        "accepted_count_max": int(plan.accepted_counts.max()),
        "accepted_assignments": int(plan.accepted_counts.sum()),
        "accepted_by_expert_rank": accepted_by_rank.tolist(),
        "drops_by_expert_rank": plan.drops_by_expert_rank.tolist(),
        "dropped_assignments": int(plan.drops_by_expert_rank.sum()),
    }


def reference_maps(plan: RoutePlan, config: BenchmarkConfig, rank: int) -> ReferenceMaps:
    """Build an independent oracle for UB-X dispatch and PUSH-combine maps."""
    routing = plan.routing.astype(np.bool_)
    counts = routing.sum(axis=0, dtype=np.int64)
    max_slots = int(counts.max(initial=0))
    max_tokens_per_rank = config.experts_per_rank * max_slots
    prefix = np.cumsum(routing, axis=0, dtype=np.int64) - routing

    local_start = rank * config.tokens_per_rank
    local_stop = local_start + config.tokens_per_rank
    token_offsets = np.full((config.tokens_per_rank, config.num_experts), -1, dtype=np.int32)
    for local_token, global_token in enumerate(range(local_start, local_stop)):
        for expert in np.flatnonzero(routing[global_token]):
            token_offsets[local_token, expert] = int(
                (expert % config.experts_per_rank) * max_slots + prefix[global_token, expert]
            )

    topk_max = int(routing.sum(axis=1).max(initial=0))
    topk_width = max(topk_max, 1)
    dispatch_expert = np.full((config.tokens_per_rank, topk_width), -1, dtype=np.int32)
    dispatch_slot = np.full((config.tokens_per_rank, topk_width), -1, dtype=np.int32)
    topk_idx = np.full((config.tokens_per_rank, topk_width), -1, dtype=np.int32)
    for local_token, global_token in enumerate(range(local_start, local_stop)):
        experts = np.flatnonzero(routing[global_token])
        dispatch_expert[local_token, : experts.size] = experts
        topk_idx[local_token, : experts.size] = experts
        dispatch_slot[local_token, : experts.size] = token_offsets[local_token, experts]

    inverse_map = np.zeros((max_tokens_per_rank, 4), dtype=np.int32)
    expert_start = rank * config.experts_per_rank
    valid_slots: list[int] = []
    source_tokens: list[int] = []
    for global_token in range(config.global_tokens):
        accepted_experts = np.flatnonzero(routing[global_token])
        for k_index, expert in enumerate(accepted_experts):
            if not expert_start <= expert < expert_start + config.experts_per_rank:
                continue
            slot = int((expert % config.experts_per_rank) * max_slots + prefix[global_token, expert])
            inverse_map[slot] = (
                global_token // config.tokens_per_rank,
                global_token % config.tokens_per_rank,
                k_index,
                1,
            )
            valid_slots.append(slot)
            source_tokens.append(global_token)

    return ReferenceMaps(
        token_offsets=token_offsets,
        dispatch_topk_expert=dispatch_expert,
        dispatch_topk_slot=dispatch_slot,
        inverse_map=inverse_map,
        topk_idx=topk_idx,
        valid_slots=np.asarray(valid_slots, dtype=np.int64),
        source_tokens=np.asarray(source_tokens, dtype=np.int64),
        max_tokens_per_rank=max_tokens_per_rank,
    )


def ring_assignment_indices(plan: RoutePlan, config: BenchmarkConfig, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Ring's expert-major/source-major accepted assignment list and validity."""
    selected_flat = plan.selected_experts.reshape(-1)
    accepted_flat = plan.accepted_assignments.reshape(-1)
    positions: list[np.ndarray] = []
    expert_start = rank * config.experts_per_rank
    for expert in range(expert_start, expert_start + config.experts_per_rank):
        positions.append(np.flatnonzero((selected_flat == expert) & accepted_flat))
    accepted_positions = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    if accepted_positions.size > config.capacity_per_expert_rank:
        raise AssertionError("Ring route cap exceeded its fixed expert-rank capacity")
    padded = np.zeros(config.capacity_per_expert_rank, dtype=np.int64)
    valid = np.zeros(config.capacity_per_expert_rank, dtype=np.bool_)
    padded[: accepted_positions.size] = accepted_positions
    valid[: accepted_positions.size] = True
    return padded, valid


def admission_result(
    *,
    route_exact: bool,
    output_relative_l2: dict[str, float],
    ring_p50_ms: float,
    ubx_p50_ms: float,
    relative_l2_limit: float,
    required_speedup: float,
) -> dict[str, Any]:
    speedup = ring_p50_ms / ubx_p50_ms
    floating_output_passed = all(
        math.isfinite(value) and value <= relative_l2_limit for value in output_relative_l2.values()
    )
    return {
        "passed": route_exact and floating_output_passed and speedup >= required_speedup,
        "route_count_drop_exact": route_exact,
        "floating_output_passed": floating_output_passed,
        "relative_l2_limit": relative_l2_limit,
        "required_speedup": required_speedup,
        "transport_speedup_vs_ring": speedup,
        "transport_speedup_passed": speedup >= required_speedup,
    }


def inspect_ubx_source(source: Path) -> dict[str, Any]:
    source = source.resolve()

    def git_text(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    commit = git_text("rev-parse", "HEAD").strip()
    if commit != UBX_SOURCE_COMMIT:
        raise RuntimeError(f"UB-X source must be {UBX_SOURCE_COMMIT}, found {commit}")
    tracked_changes = git_text("status", "--short", "--untracked-files=no").strip()
    if tracked_changes:
        raise RuntimeError(f"UB-X source has tracked modifications:\n{tracked_changes}")

    def pinned_file(path: str) -> str:
        return git_text("show", f"HEAD:{path}")

    version_file = pinned_file("makefiles/version.mk")
    expected_version_fields = ("NCCL_MAJOR   := 2", "NCCL_MINOR   := 30", "NCCL_PATCH   := 7", "PKG_REVISION := 1")
    if not all(field in version_file for field in expected_version_fields):
        raise RuntimeError("pinned NCCL source does not report version 2.30.7-1")

    allocator_source = pinned_file("contrib/nccl_ubx/ubx/allocator.py")
    ops_source = pinned_file("contrib/nccl_ubx/ubx/ops.py")
    setup_source = pinned_file("contrib/nccl_ubx/setup.py")
    registry_source = pinned_file("contrib/nccl_ubx/ubx/_api_registry.py")
    required_api = (
        "a2av_token_bf16_bf16_topk",
        "combine_push3_bf16_bf16",
        "compute_token_offsets",
        "compute_dispatch_topk_map",
        "compute_combine_push_map",
    )
    combined_source = allocator_source + ops_source
    missing = [name for name in required_api if name not in combined_source]
    if missing:
        raise RuntimeError(f"pinned UB-X source is missing required API names: {missing}")
    if 'libraries=["nccl"]' not in setup_source:
        raise RuntimeError("UB-X extension no longer declares its libnccl link")
    if "RECOMMENDED MoE combine" not in registry_source:
        raise RuntimeError("pinned UB-X source no longer marks combine_push3 as the recommended MoE combine")

    return {
        "repository": "https://github.com/NVIDIA/nccl",
        "commit": commit,
        "nccl_version": "2.30.7-1",
        "ubx_path": "contrib/nccl_ubx",
        "nccl4py_path": "bindings/nccl4py",
        "required_api": list(required_api),
    }


def _aligned_pool_size(required_bytes: int) -> int:
    # Eager allocations get 90% of the pool when UBX_GRAPH_POOL_SHARE=0.1.
    requested = math.ceil(required_bytes / 0.88) + _POOL_HEADROOM
    return math.ceil(requested / _POOL_ALIGNMENT) * _POOL_ALIGNMENT


def _relative_l2_torch(actual: Any, reference: Any) -> dict[str, Any]:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    difference = actual_f32 - reference_f32
    error_l2 = float(difference.square().sum(dtype=actual_f32.dtype).sqrt().item())
    reference_l2 = float(reference_f32.square().sum(dtype=reference_f32.dtype).sqrt().item())
    relative_l2 = error_l2 / reference_l2 if reference_l2 else (0.0 if error_l2 == 0.0 else float("inf"))
    return {
        "relative_l2_error": relative_l2,
        "error_l2": error_l2,
        "reference_l2": reference_l2,
        "max_abs": float(difference.abs().max().item()),
        "finite": bool(actual_f32.isfinite().all().item() and reference_f32.isfinite().all().item()),
    }


def _timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "samples": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p10_ms": float(np.quantile(values, 0.1)),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "aggregation": "p50_of_per_iteration_slowest_rank_cuda_event_ms",
    }


def _check_runtime_api(ubx: Any, source: Path) -> dict[str, Any]:
    required_module_api = (
        "SymmAllocator",
        "compute_token_offsets",
        "compute_dispatch_topk_map",
        "compute_combine_push_map",
    )
    missing = [name for name in required_module_api if not hasattr(ubx, name)]
    if missing:
        raise RuntimeError(f"installed UB-X is missing APIs: {missing}")
    required_allocator_api = ("a2av_token_bf16_bf16_topk", "combine_push3_bf16_bf16")
    missing_allocator = [name for name in required_allocator_api if not hasattr(ubx.SymmAllocator, name)]
    if missing_allocator:
        raise RuntimeError(f"installed UB-X allocator is missing APIs: {missing_allocator}")
    package_path = Path(ubx.__file__).resolve()
    expected_package_root = (source / "contrib/nccl_ubx").resolve()
    if not package_path.is_relative_to(expected_package_root):
        raise RuntimeError(f"installed UB-X package {package_path} is not from {expected_package_root}")
    extension_path = Path(importlib.import_module("ubx._C").__file__).resolve()
    linkage = subprocess.run(
        ["ldd", str(extension_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    expected_library_root = str((source / "build/lib").resolve())
    nccl_linkage = [line.strip() for line in linkage.splitlines() if "libnccl.so" in line]
    if not nccl_linkage or expected_library_root not in nccl_linkage[0]:
        raise RuntimeError(f"UB-X extension must link libnccl from {expected_library_root}; ldd reported {nccl_linkage}")
    return {
        "module": required_module_api,
        "allocator": required_allocator_api,
        "combine_signature": str(inspect.signature(ubx.SymmAllocator.combine_push3_bf16_bf16)),
        "package_path": str(package_path),
        "extension_path": str(extension_path),
        "nccl_linkage": nccl_linkage,
    }


def _all_ranks_true(torch: Any, dist: Any, value: bool, device: Any) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _all_rank_output_metrics(torch: Any, dist: Any, metrics: dict[str, dict[str, Any]], device: Any) -> None:
    for values in metrics.values():
        maxima = torch.tensor(
            [values["relative_l2_error"], values["max_abs"]],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        finite = torch.tensor(int(values["finite"]), dtype=torch.int32, device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        values["all_rank_max_relative_l2_error"] = float(maxima[0].item())
        values["all_rank_max_abs"] = float(maxima[1].item())
        values["all_rank_finite"] = bool(finite.item())


def _measure_slowest_rank_ms(torch: Any, dist: Any, op: Any, device: Any) -> float:
    dist.barrier()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    op()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(float(start.elapsed_time(end)), dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def _run_gpu(config: BenchmarkConfig, source: Path) -> tuple[dict[str, Any], bool]:
    if torch is None or dist is None or ubx is None:
        raise RuntimeError("the GPU gate requires PyTorch and the pinned UB-X package")

    source_info = inspect_ubx_source(source)
    runtime_api = _check_runtime_api(ubx, source)
    graph_pool_share = float(os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1"))
    if graph_pool_share != 0.1:
        raise RuntimeError(f"UBX_GRAPH_POOL_SHARE must be 0.1 for this eager gate, got {graph_pool_share}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != EP_SIZE:
        raise RuntimeError(f"UB-X direct gate requires exactly {EP_SIZE} ranks, found {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError(f"UB-X H100 gate requires SM90, found {torch.cuda.get_device_capability(device)}")
    torch_nccl_version = torch.cuda.nccl.version()
    torch_nccl_components = (
        tuple(torch_nccl_version[:3])
        if isinstance(torch_nccl_version, tuple)
        else (torch_nccl_version // 10_000, (torch_nccl_version // 100) % 100, torch_nccl_version % 100)
    )
    if torch_nccl_components != (2, 30, 7):
        raise RuntimeError(f"PyTorch must resolve NCCL 2.30.7, found {torch_nccl_version}")
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    plan = build_route_plan(config)
    maps = reference_maps(plan, config, rank)
    local_start = rank * config.tokens_per_rank
    local_stop = local_start + config.tokens_per_rank

    generator = torch.Generator(device=device).manual_seed(config.seed + 1000 + rank)
    tokens = torch.randn(
        config.tokens_per_rank,
        config.hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    selected_local = torch.as_tensor(plan.selected_experts[local_start:local_stop], dtype=torch.int32, device=device)
    assignment_weights_local = torch.as_tensor(
        plan.assignment_weights[local_start:local_stop],
        dtype=torch.float32,
        device=device,
    )
    gate_weights_local = torch.as_tensor(
        plan.gate_weights_by_expert[local_start:local_stop],
        dtype=torch.float32,
        device=device,
    )
    routing = torch.as_tensor(plan.routing, dtype=torch.uint8, device=device)

    global_tokens = torch.empty(
        config.global_tokens,
        config.hidden_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    dist.all_gather_into_tensor(global_tokens, tokens)

    token_offsets, max_tokens_per_rank, tokens_per_expert, _ = ubx.compute_token_offsets(
        routing,
        config.experts_per_rank,
        rank,
        world_size,
    )
    dispatch_topk_expert, dispatch_topk_slot, _ = ubx.compute_dispatch_topk_map(
        routing,
        token_offsets,
        config.experts_per_rank,
        rank,
        world_size,
    )
    inverse_map, topk_idx, combine_max_tokens = ubx.compute_combine_push_map(
        routing,
        config.experts_per_rank,
        rank,
        world_size,
    )

    map_checks = {
        "token_offsets": np.array_equal(token_offsets.cpu().numpy(), maps.token_offsets),
        "dispatch_topk_expert": np.array_equal(
            dispatch_topk_expert.cpu().numpy(),
            maps.dispatch_topk_expert,
        ),
        "dispatch_topk_slot": np.array_equal(dispatch_topk_slot.cpu().numpy(), maps.dispatch_topk_slot),
        "inverse_map": np.array_equal(inverse_map.cpu().numpy(), maps.inverse_map),
        "topk_idx": np.array_equal(topk_idx.cpu().numpy(), maps.topk_idx),
        "tokens_per_expert": np.array_equal(tokens_per_expert.cpu().numpy(), plan.accepted_counts),
        "max_tokens_per_rank": int(max_tokens_per_rank) == maps.max_tokens_per_rank,
        "combine_max_tokens_per_rank": int(combine_max_tokens) == maps.max_tokens_per_rank,
    }
    maps_exact = all(map_checks.values())

    dispatch_bytes = maps.max_tokens_per_rank * config.hidden_dim * 2
    combine_destination_bytes = config.tokens_per_rank * config.top_k * config.hidden_dim * 2
    pool_size = _aligned_pool_size(dispatch_bytes + 2 * combine_destination_bytes)
    allocator = ubx.SymmAllocator(pool_size, device, dist.group.WORLD)
    dispatch_output = allocator.create_tensor(
        [maps.max_tokens_per_rank, config.hidden_dim],
        torch.bfloat16,
    )

    def ubx_transport() -> Any:
        allocator.a2av_token_bf16_bf16_topk(
            tokens,
            dispatch_topk_expert,
            dispatch_topk_slot,
            config.experts_per_rank,
            dispatch_output,
        )
        return allocator.combine_push3_bf16_bf16(
            dispatch_output,
            inverse_map,
            topk_idx,
            config.experts_per_rank,
            maps.max_tokens_per_rank,
            gate_weights=gate_weights_local,
        )

    # Exercise both halves of push3's double-buffered destination before
    # correctness checks and timing.
    ubx_output = ubx_transport()
    ubx_output = ubx_transport()
    torch.cuda.synchronize(device)

    valid_slots = torch.as_tensor(maps.valid_slots, dtype=torch.int64, device=device)
    source_tokens = torch.as_tensor(maps.source_tokens, dtype=torch.int64, device=device)
    received_valid = dispatch_output.index_select(0, valid_slots)
    expected_valid = global_tokens.index_select(0, source_tokens)
    dispatch_bitwise = bool(torch.equal(received_valid, expected_valid))
    dispatch_metrics = _relative_l2_torch(received_valid, expected_valid)

    assignment_indices_np, ring_valid_np = ring_assignment_indices(plan, config, rank)
    assignment_indices = torch.as_tensor(assignment_indices_np, dtype=torch.int64, device=device)
    token_indices = torch.div(assignment_indices, config.top_k, rounding_mode="floor")
    ring_valid = torch.as_tensor(ring_valid_np, dtype=torch.bool, device=device)

    ring_global_selected = torch.empty(
        config.global_tokens,
        config.top_k,
        dtype=torch.int32,
        device=device,
    )
    ring_global_weights = torch.empty(
        config.global_tokens,
        config.top_k,
        dtype=torch.float32,
        device=device,
    )
    ring_dispatch = torch.empty(
        config.capacity_per_expert_rank,
        config.hidden_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    ring_weight_f32 = torch.empty(config.capacity_per_expert_rank, dtype=torch.float32, device=device)
    ring_weight_bf16 = torch.empty(config.capacity_per_expert_rank, dtype=torch.bfloat16, device=device)
    ring_weighted = torch.empty_like(ring_dispatch)
    ring_dense_output = torch.empty_like(global_tokens)
    ring_output = torch.empty_like(tokens)

    def ring_transport() -> Any:
        dist.all_gather_into_tensor(global_tokens, tokens)
        dist.all_gather_into_tensor(ring_global_selected, selected_local)
        dist.all_gather_into_tensor(ring_global_weights, assignment_weights_local)
        torch.index_select(global_tokens, 0, token_indices, out=ring_dispatch)
        torch.index_select(ring_global_weights.reshape(-1), 0, assignment_indices, out=ring_weight_f32)
        ring_weight_bf16.copy_(ring_weight_f32)
        ring_dispatch.masked_fill_(~ring_valid[:, None], 0)
        ring_weight_bf16.masked_fill_(~ring_valid, 0)
        torch.mul(ring_dispatch, ring_weight_bf16[:, None], out=ring_weighted)
        ring_dense_output.zero_()
        ring_dense_output.index_add_(0, token_indices, ring_weighted)
        dist.reduce_scatter_tensor(ring_output, ring_dense_output, op=dist.ReduceOp.SUM)
        return ring_output

    ring_output = ring_transport()
    torch.cuda.synchronize(device)
    gathered_routes_exact = bool(
        torch.equal(
            ring_global_selected,
            torch.as_tensor(plan.selected_experts, dtype=torch.int32, device=device),
        )
    )

    accepted_gate_sum = torch.as_tensor(
        plan.gate_weights_by_expert[local_start:local_stop].sum(axis=1),
        dtype=torch.float32,
        device=device,
    )
    reference_output = tokens.float() * accepted_gate_sum[:, None]
    output_metrics = {
        "dispatch_vs_reference": dispatch_metrics,
        "ring_vs_fp32_identity_reference": _relative_l2_torch(ring_output, reference_output),
        "ubx_vs_fp32_identity_reference": _relative_l2_torch(ubx_output, reference_output),
        "ubx_vs_ring": _relative_l2_torch(ubx_output, ring_output),
    }
    _all_rank_output_metrics(torch, dist, output_metrics, device)
    route_exact_local = maps_exact and gathered_routes_exact and dispatch_bitwise
    route_exact = _all_ranks_true(torch, dist, route_exact_local, device)

    for warmup_index in range(config.warmup):
        arms = (ring_transport, ubx_transport) if warmup_index % 2 == 0 else (ubx_transport, ring_transport)
        for arm in arms:
            _measure_slowest_rank_ms(torch, dist, arm, device)

    timing_samples = {"ring": [], "ubx": []}
    for iteration in range(config.iterations):
        arms = (("ring", ring_transport), ("ubx", ubx_transport))
        if iteration % 2:
            arms = tuple(reversed(arms))
        for name, arm in arms:
            timing_samples[name].append(_measure_slowest_rank_ms(torch, dist, arm, device))
    timings = {name: _timing_summary(samples) for name, samples in timing_samples.items()}

    output_relative_l2 = {name: metrics["all_rank_max_relative_l2_error"] for name, metrics in output_metrics.items()}
    admission = admission_result(
        route_exact=route_exact,
        output_relative_l2=output_relative_l2,
        ring_p50_ms=timings["ring"]["p50_ms"],
        ubx_p50_ms=timings["ubx"]["p50_ms"],
        relative_l2_limit=config.relative_l2_limit,
        required_speedup=config.required_speedup,
    )
    admission["passed"] = _all_ranks_true(torch, dist, admission["passed"], device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "event": "nccl_ubx_direct_ep8_gate",
        "source": source_info,
        "runtime_api": runtime_api,
        "runtime": {
            "torch": torch.__version__,
            "torch_nccl": torch_nccl_version,
            "ubx": ubx.get_version(),
            "device": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "world_size": world_size,
        },
        "config": asdict(config),
        "shape": {
            "ep_size": EP_SIZE,
            "experts_per_rank": config.experts_per_rank,
            "global_tokens": config.global_tokens,
            "assignments_per_source_rank": config.assignments_per_rank,
            "global_assignments": config.global_assignments,
            "dtype": "bfloat16",
            "expert": "identity_transport_oracle",
        },
        "route_plan": route_plan_summary(plan, config),
        "correctness": {
            "map_checks": map_checks,
            "maps_exact_on_rank0": maps_exact if rank == 0 else None,
            "gathered_selected_experts_exact_on_rank0": gathered_routes_exact if rank == 0 else None,
            "dispatch_bitwise_exact_on_rank0": dispatch_bitwise if rank == 0 else None,
            "route_count_drop_exact_all_ranks": route_exact,
            "floating_outputs": output_metrics,
        },
        "memory": {
            "ubx_pool_bytes": pool_size,
            "dispatch_buffer_bytes": dispatch_bytes,
            "combine_destination_bytes_each": combine_destination_bytes,
            "device_free_bytes_after_benchmark": int(free_bytes),
            "device_total_bytes": int(total_bytes),
        },
        "timings": timings,
        "admission": admission,
    }

    dist.barrier()
    allocator.close()
    dist.destroy_process_group()
    return result, bool(admission["passed"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", choices=("balanced", "learned_skew"), default="balanced")
    parser.add_argument("--tokens-per-rank", type=int, default=DEFAULT_TOKENS_PER_RANK)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--skew-alpha", type=float, default=DEFAULT_SKEW_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--relative-l2-limit", type=float, default=DEFAULT_RELATIVE_L2_LIMIT)
    parser.add_argument("--required-speedup", type=float, default=DEFAULT_REQUIRED_SPEEDUP)
    parser.add_argument("--ubx-source", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = BenchmarkConfig(
        tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        routing=args.routing,
        skew_alpha=args.skew_alpha,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        relative_l2_limit=args.relative_l2_limit,
        required_speedup=args.required_speedup,
    )
    plan = build_route_plan(config)
    if args.plan_only:
        result = {
            "event": "nccl_ubx_direct_ep8_plan",
            "config": asdict(config),
            "shape": {
                "ep_size": EP_SIZE,
                "global_tokens": config.global_tokens,
                "assignments_per_source_rank": config.assignments_per_rank,
                "global_assignments": config.global_assignments,
            },
            "route_plan": route_plan_summary(plan, config),
            "source": inspect_ubx_source(args.ubx_source) if args.ubx_source else None,
        }
        print(json.dumps(result, sort_keys=True))
        return

    if args.ubx_source is None:
        raise SystemExit("--ubx-source is required for a GPU gate")
    result, passed = _run_gpu(config, args.ubx_source)
    if int(os.environ["RANK"]) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
