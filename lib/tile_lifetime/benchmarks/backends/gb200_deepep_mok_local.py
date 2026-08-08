# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a DeepEP-shaped local route boundary with the MoK grouped-GEMM probe.

This benchmark deliberately starts and ends at an already-dispatched local
boundary. It reproduces the tensors that DeepEP transport exposes:

    coalesced received tokens -> local expert grouping -> 256 padding
    -> MoK W13 probes -> standalone SwiGLU -> MoK W2 probe
    -> fixed route-slot merge -> inverse coalesced return -> shared-output add

It does not import MoK's task graph or complete forward megakernel.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
PACKAGE_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(PACKAGE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from gb200_mok_gmm_probe import (  # noqa: E402
    MOK_COMMIT,
    _build_extension,
    _load_extension,
    _validate_checkout,
)

from tile_lifetime import build_relation_plan  # noqa: E402
from tile_lifetime.cuda_map_fold_codegen import CudaMapFoldProgram, shuttle_map_fold_program  # noqa: E402

DEEPEP_COMMIT = "7febc6e25660af0f54d95dd781ecdcd62265ecca"
DEFAULT_LOCAL_TOKENS = 2_048
DEFAULT_HIDDEN_SIZE = 7_168
DEFAULT_INTERMEDIATE_SIZE = 3_072
DEFAULT_GLOBAL_EXPERTS = 384
DEFAULT_TOP_K = 6
DEFAULT_WORLD_SIZE = 4
TILE_ROWS = 256


def _validate_generated_map_fold_extension(
    module: ModuleType,
    program: CudaMapFoldProgram | None = None,
) -> str:
    """Require the loaded CUDA extension to match the selected generic scalar IR."""
    expected = (program or shuttle_map_fold_program()).fingerprint
    observed = module.generated_map_fold_program_sha256()
    if observed != expected:
        raise ValueError(f"CUDA Map/Fold program is {observed}; selected Shuttle program is {expected}")
    return expected


@dataclass(frozen=True)
class LocalRoutePlan:
    """Host metadata for one receiver rank's coalesced and padded route relation."""

    receiver_source_tokens: np.ndarray
    assignment_receiver_rows: np.ndarray
    assignment_route_slots: np.ndarray
    assignment_padded_rows: np.ndarray
    padded_receiver_rows: np.ndarray
    route_padded_rows: np.ndarray
    actual_counts: np.ndarray
    padded_counts: np.ndarray

    @property
    def assignments(self) -> int:
        return int(self.assignment_padded_rows.size)

    @property
    def received_tokens(self) -> int:
        return int(self.receiver_source_tokens.size)

    @property
    def padded_rows(self) -> int:
        return int(self.padded_counts.sum())


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mok-root",
        type=Path,
        default=Path(os.environ["MOK_ROOT"]) if "MOK_ROOT" in os.environ else None,
    )
    parser.add_argument(
        "--deepep-root",
        type=Path,
        default=(Path(os.environ["DEEPEP_ROOT"]) if "DEEPEP_ROOT" in os.environ else None),
    )
    parser.add_argument("--route-fixture", type=Path, required=True)
    parser.add_argument("--probe-extension", type=Path)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/tmp/tile_lifetime_mok_gmm_probe") / MOK_COMMIT[:12],
    )
    parser.add_argument("--nvcc", default=os.environ.get("MOK_NVCC", "nvcc"))
    parser.add_argument("--owner-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--local-tokens", type=int, default=DEFAULT_LOCAL_TOKENS)
    parser.add_argument("--global-experts", type=int, default=DEFAULT_GLOBAL_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json-output", type=Path)
    return parser


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.mok_root is None:
        raise ValueError("--mok-root or MOK_ROOT is required")
    if args.deepep_root is None:
        raise ValueError("--deepep-root or DEEPEP_ROOT is required to record the pinned transport source")
    if not 0 <= args.owner_rank < args.world_size:
        raise ValueError(f"owner-rank must be in [0, {args.world_size}), got {args.owner_rank}")
    if args.global_experts <= 0 or args.global_experts % args.world_size:
        raise ValueError("global-experts must be positive and divisible by world-size")
    if args.local_tokens <= 0 or args.top_k <= 0:
        raise ValueError("local-tokens and top-k must be positive")
    if args.hidden_size <= 0 or args.hidden_size % 256:
        raise ValueError("hidden-size must be positive and divisible by 256")
    if args.intermediate_size <= 0 or args.intermediate_size % 256:
        raise ValueError("intermediate-size must be positive and divisible by 256")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")


def _load_routes(path: Path, *, global_tokens: int, top_k: int) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path) as fixture:
        selected_experts = np.asarray(fixture["selected_experts"], dtype=np.int64)
        combine_weights = np.asarray(fixture["combine_weights"], dtype=np.float32)
    expected_shape = (global_tokens, top_k)
    if selected_experts.shape != expected_shape or combine_weights.shape != expected_shape:
        raise ValueError(
            f"route fixture tensors must both have shape {expected_shape}, got "
            f"{selected_experts.shape} and {combine_weights.shape}"
        )
    if not np.isfinite(combine_weights).all():
        raise ValueError("route fixture weights must be finite")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return selected_experts, combine_weights, digest


def _local_route_plan(
    selected_experts: np.ndarray,
    combine_weights: np.ndarray,
    *,
    owner_rank: int,
    world_size: int,
    global_experts: int,
    alignment: int = TILE_ROWS,
) -> LocalRoutePlan:
    """Project the compiler's global relation into one DeepEP receiver-rank view."""
    local_experts = global_experts // world_size
    destination_items = np.arange(global_experts, dtype=np.int32)
    relation = build_relation_plan(
        selected_experts.astype(np.int32, copy=False),
        combine_weights,
        destination_rank_by_item=destination_items // local_experts,
        destination_local_item_by_item=destination_items % local_experts,
        padding_quantum=alignment,
    )
    exchange_mask = relation.exchange_destination_rank == owner_rank
    receiver_source_tokens = relation.exchange_source_item[exchange_mask].astype(np.int64, copy=False)
    source_to_receiver = np.full(selected_experts.shape[0], -1, dtype=np.int64)
    source_to_receiver[receiver_source_tokens] = np.arange(receiver_source_tokens.size, dtype=np.int64)

    owner_rows = np.flatnonzero(relation.row_destination_rank == owner_rank)
    owner_row_valid = relation.row_valid[owner_rows]
    valid_owner_rows = owner_rows[owner_row_valid]
    source_tokens = relation.row_source_item[valid_owner_rows]
    route_slots = relation.row_route_slot[valid_owner_rows]
    assignment_receiver_rows = source_to_receiver[source_tokens]
    assignment_padded_rows = np.flatnonzero(owner_row_valid).astype(np.int64)

    owner_groups = relation.group_destination_rank == owner_rank
    actual_counts = relation.group_count[owner_groups].astype(np.int32, copy=False)
    padded_counts = relation.group_padded_count[owner_groups].astype(np.int32, copy=False)

    route_padded_rows = np.full((receiver_source_tokens.size, selected_experts.shape[1]), -1, dtype=np.int64)
    route_padded_rows[assignment_receiver_rows, route_slots] = assignment_padded_rows
    padded_receiver_rows = np.full(int(padded_counts.sum()), -1, dtype=np.int64)
    padded_receiver_rows[assignment_padded_rows] = assignment_receiver_rows
    plan = LocalRoutePlan(
        receiver_source_tokens=receiver_source_tokens,
        assignment_receiver_rows=assignment_receiver_rows,
        assignment_route_slots=route_slots.astype(np.int64),
        assignment_padded_rows=assignment_padded_rows,
        padded_receiver_rows=padded_receiver_rows,
        route_padded_rows=route_padded_rows,
        actual_counts=actual_counts,
        padded_counts=padded_counts,
    )
    _validate_route_plan(
        plan,
        selected_experts,
        owner_rank=owner_rank,
        world_size=world_size,
        global_experts=global_experts,
    )
    return plan


def _validate_route_plan(
    plan: LocalRoutePlan,
    selected_experts: np.ndarray,
    *,
    owner_rank: int,
    world_size: int,
    global_experts: int,
) -> dict[str, int | bool | str]:
    """Check the compiler projection against a direct route-table construction."""
    local_experts = global_experts // world_size
    owners = selected_experts // local_experts
    expected_assignments = int((owners == owner_rank).sum())
    if plan.assignments != expected_assignments:
        raise AssertionError(f"assignment count mismatch: {plan.assignments} != {expected_assignments}")
    if int(plan.actual_counts.sum()) != plan.assignments:
        raise AssertionError("local expert counts do not sum to assignments")
    if np.any(plan.padded_counts % TILE_ROWS):
        raise AssertionError("padded expert counts are not 256-aligned")
    valid_rows = plan.route_padded_rows[plan.route_padded_rows >= 0]
    if valid_rows.size != plan.assignments or np.unique(valid_rows).size != valid_rows.size:
        raise AssertionError("route-to-padded-row relation is not one-to-one")
    if valid_rows.size and (valid_rows.min() < 0 or valid_rows.max() >= plan.padded_rows):
        raise AssertionError("route-to-padded-row relation is out of bounds")
    inverse_rows = plan.padded_receiver_rows[plan.assignment_padded_rows]
    if not np.array_equal(inverse_rows, plan.assignment_receiver_rows):
        raise AssertionError("padded-row inverse does not recover receiver rows")

    expected_receiver_source_tokens = np.flatnonzero(np.any(owners == owner_rank, axis=1)).astype(np.int64)
    if not np.array_equal(plan.receiver_source_tokens, expected_receiver_source_tokens):
        raise AssertionError("coalesced receiver-token order is not ascending source-token order")

    expected_actual_counts = np.bincount(
        (selected_experts[owners == owner_rank] % local_experts).astype(np.int64),
        minlength=local_experts,
    ).astype(np.int32)
    expected_padded_counts = ((expected_actual_counts + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS).astype(np.int32)
    if not np.array_equal(plan.actual_counts, expected_actual_counts):
        raise AssertionError("per-expert assignment counts disagree with the route table")
    if not np.array_equal(plan.padded_counts, expected_padded_counts):
        raise AssertionError("per-expert padded counts disagree with the direct construction")

    source_to_receiver = np.full(selected_experts.shape[0], -1, dtype=np.int64)
    source_to_receiver[expected_receiver_source_tokens] = np.arange(expected_receiver_source_tokens.size)
    expected_route_padded_rows = np.full_like(plan.route_padded_rows, -1)
    expected_padded_receiver_rows = np.full(plan.padded_rows, -1, dtype=np.int64)
    expected_assignment_receiver_rows: list[int] = []
    expected_assignment_route_slots: list[int] = []
    expected_assignment_padded_rows: list[int] = []
    group_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(expected_padded_counts[:-1], dtype=np.int64)))
    group_next_rows = group_offsets.copy()
    for source_token in range(selected_experts.shape[0]):
        receiver_row = int(source_to_receiver[source_token])
        for route_slot in range(selected_experts.shape[1]):
            global_expert = int(selected_experts[source_token, route_slot])
            if global_expert // local_experts != owner_rank:
                continue
            local_expert = global_expert % local_experts
            padded_row = int(group_next_rows[local_expert])
            group_next_rows[local_expert] += 1
            expected_assignment_receiver_rows.append(receiver_row)
            expected_assignment_route_slots.append(route_slot)
            expected_assignment_padded_rows.append(padded_row)
            expected_route_padded_rows[receiver_row, route_slot] = padded_row
            expected_padded_receiver_rows[padded_row] = receiver_row

    expected_receiver_rows = np.asarray(expected_assignment_receiver_rows, dtype=np.int64)
    expected_route_slots = np.asarray(expected_assignment_route_slots, dtype=np.int64)
    expected_padded_rows = np.asarray(expected_assignment_padded_rows, dtype=np.int64)
    padded_order = np.argsort(expected_padded_rows, kind="stable")
    expected_receiver_rows = expected_receiver_rows[padded_order]
    expected_route_slots = expected_route_slots[padded_order]
    expected_padded_rows = expected_padded_rows[padded_order]
    if not np.array_equal(plan.assignment_receiver_rows, expected_receiver_rows):
        raise AssertionError("assignment-to-receiver mapping disagrees with the direct route-table construction")
    if not np.array_equal(plan.assignment_route_slots, expected_route_slots):
        raise AssertionError("assignment route slots disagree with the direct route-table construction")
    if not np.array_equal(plan.assignment_padded_rows, expected_padded_rows):
        raise AssertionError("assignment-to-padded mapping disagrees with the direct expert grouping")
    if not np.array_equal(plan.route_padded_rows, expected_route_padded_rows):
        raise AssertionError("route-to-padded mapping disagrees with the direct route-table construction")
    if not np.array_equal(plan.padded_receiver_rows, expected_padded_receiver_rows):
        raise AssertionError("padded-row inverse disagrees with the direct expert grouping")
    return {
        "exact": True,
        "construction": "direct source-token/route-slot scan grouped by local expert",
        "receiver_tokens_checked": plan.received_tokens,
        "assignments_checked": plan.assignments,
        "padded_rows_checked": plan.padded_rows,
    }


def _device_index(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.int64, device=device)


class LocalPhysicalRuntime:
    """Reusable Torch buffers for one already-dispatched receiver-local forward."""

    def __init__(
        self,
        module: ModuleType,
        plan: LocalRoutePlan,
        combine_weights: np.ndarray,
        *,
        owner_rank: int,
        local_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.module = module
        self.plan = plan
        self.owner_rank = owner_rank
        self.local_tokens = local_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        local_experts = plan.padded_counts.size
        global_tokens = combine_weights.shape[0]

        torch.manual_seed(seed)
        self.x = torch.full((global_tokens, hidden_size), 0.01, dtype=torch.bfloat16, device=device)
        self.shared_output = torch.full((local_tokens, hidden_size), 0.02, dtype=torch.bfloat16, device=device)
        weight_scale_w13 = 1.0 / math.sqrt(hidden_size)
        weight_scale_w2 = 1.0 / math.sqrt(intermediate_size)
        self.gate_weights = torch.empty(
            (local_experts, intermediate_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ).normal_(0.0, weight_scale_w13)
        self.up_weights = torch.empty_like(self.gate_weights).normal_(0.0, weight_scale_w13)
        self.gate_up_weights = torch.cat((self.gate_weights, self.up_weights), dim=1).contiguous()
        self.down_weights = torch.empty(
            (local_experts, hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ).normal_(0.0, weight_scale_w2)

        self.receiver_source_tokens = _device_index(plan.receiver_source_tokens, device)
        self.assignment_receiver_rows = _device_index(plan.assignment_receiver_rows, device)
        self.assignment_padded_rows = _device_index(plan.assignment_padded_rows, device)
        self.padded_receiver_rows = _device_index(plan.padded_receiver_rows, device)
        self.route_padded_rows = _device_index(plan.route_padded_rows, device)
        self.padded_counts = torch.as_tensor(plan.padded_counts, dtype=torch.int32, device=device)
        receiver_weights = combine_weights[plan.receiver_source_tokens]
        self.receiver_weights = torch.as_tensor(receiver_weights, dtype=torch.float32, device=device)
        source_start = owner_rank * local_tokens
        source_tokens = np.arange(source_start, source_start + local_tokens, dtype=np.int64)
        receiver_candidates = np.searchsorted(plan.receiver_source_tokens, source_tokens)
        safe_candidates = np.minimum(receiver_candidates, plan.receiver_source_tokens.size - 1)
        source_present = (receiver_candidates < plan.receiver_source_tokens.size) & (
            plan.receiver_source_tokens[safe_candidates] == source_tokens
        )
        source_receiver_rows = np.where(source_present, safe_candidates, -1)
        self.source_receiver_rows = _device_index(source_receiver_rows, device)

        self.recv_x = torch.empty((plan.received_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        self.assignment_x = torch.empty((plan.assignments, hidden_size), dtype=torch.bfloat16, device=device)
        self.padded_x = torch.empty((plan.padded_rows, hidden_size), dtype=torch.bfloat16, device=device)
        self.gate = torch.empty((plan.padded_rows, intermediate_size), dtype=torch.bfloat16, device=device)
        self.up = torch.empty_like(self.gate)
        self.gate_up = torch.empty(
            (plan.padded_rows, 2 * intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self.swiglu = torch.empty_like(self.gate)
        self.down = torch.empty((plan.padded_rows, hidden_size), dtype=torch.bfloat16, device=device)
        self.recv_merged = torch.empty((plan.received_tokens, hidden_size), dtype=torch.float32, device=device)
        self.recv_merged_bf16 = torch.empty((plan.received_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        self.returned = torch.empty((global_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        self.output = torch.empty((local_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    def coalesced_dispatch(self) -> None:
        torch.index_select(self.x, 0, self.receiver_source_tokens, out=self.recv_x)

    def padded_pack_torch(self) -> None:
        torch.index_select(self.recv_x, 0, self.assignment_receiver_rows, out=self.assignment_x)
        self.padded_x.zero_()
        self.padded_x.index_copy_(0, self.assignment_padded_rows, self.assignment_x)

    def padded_pack_cuda(self) -> None:
        self.module.padded_pack_bf16_out(self.recv_x, self.padded_receiver_rows, self.padded_x)

    def w13(self) -> None:
        self.module.grouped_gemm_out(self.padded_x, self.gate_weights, self.padded_counts, self.gate)
        self.module.grouped_gemm_out(self.padded_x, self.up_weights, self.padded_counts, self.up)

    def w13_concatenated(self) -> None:
        self.module.grouped_gemm_out(self.padded_x, self.gate_up_weights, self.padded_counts, self.gate_up)

    def apply_swiglu_torch(self) -> None:
        self.swiglu.copy_(torch_functional.silu(self.gate.float()).mul_(self.up.float()).bfloat16())

    def apply_swiglu_cuda(self) -> None:
        self.module.adjacent_pair_map_bf16_out(self.gate, self.up, self.swiglu)

    def apply_swiglu_concatenated_cuda(self) -> None:
        self.module.row_halves_pair_map_bf16_out(self.gate_up, self.swiglu)

    def w13_swiglu_generated(self) -> None:
        self.w13()
        self.apply_swiglu_cuda()

    def w13_swiglu_concatenated(self) -> None:
        self.w13_concatenated()
        self.apply_swiglu_concatenated_cuda()

    def w2(self) -> None:
        self.module.grouped_gemm_out(self.swiglu, self.down_weights, self.padded_counts, self.down)

    def merge_received_torch(self) -> None:
        self.recv_merged.zero_()
        for route_slot in range(self.route_padded_rows.shape[1]):
            padded_rows = self.route_padded_rows[:, route_slot]
            present = padded_rows >= 0
            route_output = self.down.index_select(0, padded_rows.clamp_min(0)).float()
            contribution = route_output * self.receiver_weights[:, route_slot, None]
            self.recv_merged.add_(torch.where(present[:, None], contribution, 0))
        self.recv_merged_bf16.copy_(self.recv_merged.bfloat16())

    def merge_received_cuda(self) -> None:
        self.module.indexed_weighted_ordered_fold_bf16_out(
            self.down,
            self.route_padded_rows,
            self.receiver_weights,
            self.recv_merged_bf16,
        )

    def merge_received_cuda_fma(self) -> None:
        self.module.indexed_weighted_ordered_fold_relaxed_bf16_out(
            self.down,
            self.route_padded_rows,
            self.receiver_weights,
            self.recv_merged_bf16,
        )

    def return_merge_and_add_shared_torch(self) -> None:
        self.merge_received_torch()
        self.returned.zero_()
        self.returned.index_copy_(0, self.receiver_source_tokens, self.recv_merged_bf16)
        source_start = self.owner_rank * self.local_tokens
        source_end = source_start + self.local_tokens
        torch.add(self.returned[source_start:source_end], self.shared_output, out=self.output)

    def return_merge_and_add_shared_cuda(self) -> None:
        self.module.indirect_weighted_fold_base_map_out(
            self.down,
            self.route_padded_rows,
            self.receiver_weights,
            self.source_receiver_rows,
            self.shared_output,
            self.output,
        )

    def run_torch_reference(self) -> None:
        self.coalesced_dispatch()
        self.padded_pack_torch()
        self.w13()
        self.apply_swiglu_torch()
        self.w2()
        self.return_merge_and_add_shared_torch()

    def run_precombine_torch(self) -> None:
        self.coalesced_dispatch()
        self.padded_pack_torch()
        self.w13()
        self.apply_swiglu_torch()
        self.w2()
        self.merge_received_torch()

    def run_precombine_generated(self) -> None:
        self.coalesced_dispatch()
        self.padded_pack_cuda()
        self.w13()
        self.apply_swiglu_cuda()
        self.w2()
        self.merge_received_cuda()

    def run_precombine_concatenated(self) -> None:
        self.coalesced_dispatch()
        self.padded_pack_cuda()
        self.w13_concatenated()
        self.apply_swiglu_concatenated_cuda()
        self.w2()
        self.merge_received_cuda()

    def run_generated(self) -> None:
        self.coalesced_dispatch()
        self.padded_pack_cuda()
        self.w13()
        self.apply_swiglu_cuda()
        self.w2()
        self.return_merge_and_add_shared_cuda()


def _measure(function, *, warmup: int, iterations: int, device: torch.device) -> tuple[list[float], TimingSummary]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in events]
    return samples, TimingSummary(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
    )


def _measure_interleaved(
    first,
    second,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[tuple[list[float], TimingSummary], tuple[list[float], TimingSummary]]:
    for iteration in range(warmup):
        if iteration % 2 == 0:
            first()
            second()
        else:
            second()
            first()
    torch.cuda.synchronize(device)
    first_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)
    ]
    second_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)
    ]
    for iteration in range(iterations):
        first_start, first_end = first_events[iteration]
        second_start, second_end = second_events[iteration]
        if iteration % 2 == 0:
            first_start.record()
            first()
            first_end.record()
            second_start.record()
            second()
            second_end.record()
        else:
            second_start.record()
            second()
            second_end.record()
            first_start.record()
            first()
            first_end.record()
    torch.cuda.synchronize(device)
    first_samples = [start.elapsed_time(end) for start, end in first_events]
    second_samples = [start.elapsed_time(end) for start, end in second_events]

    def summarize(samples: list[float]) -> TimingSummary:
        return TimingSummary(
            median_ms=statistics.median(samples),
            mean_ms=statistics.fmean(samples),
            minimum_ms=min(samples),
            maximum_ms=max(samples),
        )

    return (first_samples, summarize(first_samples)), (second_samples, summarize(second_samples))


def _source_status(args: argparse.Namespace) -> dict[str, Any]:
    assert args.mok_root is not None and args.deepep_root is not None
    mok = _validate_checkout(args.mok_root.resolve())
    deepep_revision = _git_revision(args.deepep_root.resolve())
    if deepep_revision != DEEPEP_COMMIT:
        raise ValueError(f"DeepEP checkout is {deepep_revision}; expected {DEEPEP_COMMIT}")
    torch_deepep_importable = importlib.util.find_spec("deep_ep") is not None
    return {
        **mok,
        "deepep_commit": deepep_revision,
        "deepep_torch_importable_in_benchmark_process": torch_deepep_importable,
        "transport_scope": "excluded_already_dispatched_local_benchmark",
    }


def _extension(args: argparse.Namespace) -> Path:
    assert args.mok_root is not None
    if args.probe_extension is not None:
        extension = args.probe_extension.resolve()
        if not extension.is_file():
            raise FileNotFoundError(f"probe extension not found: {extension}")
        return extension
    return _build_extension(args.mok_root.resolve(), args.build_dir.resolve(), args.nvcc)


def main() -> None:
    args = _parser().parse_args()
    _validate_arguments(args)
    source = _source_status(args)
    global_tokens = args.local_tokens * args.world_size
    selected_experts, combine_weights, fixture_sha256 = _load_routes(
        args.route_fixture,
        global_tokens=global_tokens,
        top_k=args.top_k,
    )
    if selected_experts.min() < 0 or selected_experts.max() >= args.global_experts:
        raise ValueError("route fixture contains an out-of-range expert")
    plan = _local_route_plan(
        selected_experts,
        combine_weights,
        owner_rank=args.owner_rank,
        world_size=args.world_size,
        global_experts=args.global_experts,
    )
    metadata_correspondence = _validate_route_plan(
        plan,
        selected_experts,
        owner_rank=args.owner_rank,
        world_size=args.world_size,
        global_experts=args.global_experts,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension_path = _extension(args)
    module = _load_extension(extension_path)
    _validate_generated_map_fold_extension(module)
    runtime = LocalPhysicalRuntime(
        module,
        plan,
        combine_weights,
        owner_rank=args.owner_rank,
        local_tokens=args.local_tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
        seed=args.seed,
    )

    runtime.coalesced_dispatch()
    runtime.padded_pack_torch()
    torch.cuda.synchronize(device)
    padded_torch = runtime.padded_x.clone()
    runtime.padded_pack_cuda()
    torch.cuda.synchronize(device)
    padded_pack_matches = torch.equal(padded_torch, runtime.padded_x)

    runtime.w13()
    runtime.w13_concatenated()
    torch.cuda.synchronize(device)
    concatenated_gate = runtime.gate_up[:, : args.intermediate_size]
    concatenated_up = runtime.gate_up[:, args.intermediate_size :]
    concatenated_gate_error = (runtime.gate.float() - concatenated_gate.float()).abs()
    concatenated_up_error = (runtime.up.float() - concatenated_up.float()).abs()
    concatenated_w13_matches = torch.allclose(
        runtime.gate, concatenated_gate, atol=0.015625, rtol=0.01
    ) and torch.allclose(
        runtime.up,
        concatenated_up,
        atol=0.015625,
        rtol=0.01,
    )
    concatenated_w13_bitwise = torch.equal(runtime.gate, concatenated_gate) and torch.equal(runtime.up, concatenated_up)
    runtime.apply_swiglu_torch()
    torch.cuda.synchronize(device)
    swiglu_torch = runtime.swiglu.clone()
    runtime.apply_swiglu_cuda()
    torch.cuda.synchronize(device)
    swiglu_absolute_error = (swiglu_torch.float() - runtime.swiglu.float()).abs()
    swiglu_matches = torch.allclose(swiglu_torch, runtime.swiglu, atol=0.015625, rtol=0.01)
    separate_swiglu = runtime.swiglu.clone()
    runtime.apply_swiglu_concatenated_cuda()
    torch.cuda.synchronize(device)
    concatenated_swiglu_absolute_error = (separate_swiglu.float() - runtime.swiglu.float()).abs()
    concatenated_swiglu_matches = torch.allclose(separate_swiglu, runtime.swiglu, atol=0.015625, rtol=0.01)
    concatenated_swiglu_bitwise = torch.equal(separate_swiglu, runtime.swiglu)

    runtime.w2()
    runtime.merge_received_torch()
    torch.cuda.synchronize(device)
    received_torch = runtime.recv_merged_bf16.clone()
    runtime.merge_received_cuda_fma()
    torch.cuda.synchronize(device)
    received_fma = runtime.recv_merged_bf16.clone()
    received_fma_absolute_error = (received_torch.float() - received_fma.float()).abs()
    received_fma_matches = torch.allclose(received_torch, received_fma, atol=0.015625, rtol=0.01)
    runtime.merge_received_cuda()
    torch.cuda.synchronize(device)
    received_explicit = runtime.recv_merged_bf16.clone()
    received_explicit_absolute_error = (received_torch.float() - received_explicit.float()).abs()
    received_explicit_matches = torch.allclose(received_torch, received_explicit, atol=0.015625, rtol=0.01)
    received_explicit_bitwise = torch.equal(received_torch, received_explicit)
    runtime.merge_received_cuda()
    torch.cuda.synchronize(device)
    received_explicit_deterministic = torch.equal(received_explicit, runtime.recv_merged_bf16)
    runtime.merge_received_cuda_fma()
    torch.cuda.synchronize(device)
    received_fma_deterministic = torch.equal(received_fma, runtime.recv_merged_bf16)

    runtime.run_precombine_generated()
    torch.cuda.synchronize(device)
    separate_precombine = runtime.recv_merged_bf16.clone()
    runtime.run_precombine_concatenated()
    torch.cuda.synchronize(device)
    concatenated_precombine = runtime.recv_merged_bf16.clone()
    concatenated_precombine_absolute_error = (separate_precombine.float() - concatenated_precombine.float()).abs()
    concatenated_precombine_matches = torch.allclose(
        separate_precombine,
        concatenated_precombine,
        atol=0.015625,
        rtol=0.01,
    )
    concatenated_precombine_bitwise = torch.equal(separate_precombine, concatenated_precombine)
    runtime.run_precombine_concatenated()
    torch.cuda.synchronize(device)
    concatenated_precombine_deterministic = torch.equal(concatenated_precombine, runtime.recv_merged_bf16)

    runtime.return_merge_and_add_shared_torch()
    torch.cuda.synchronize(device)
    local_slice_torch = runtime.output.clone()
    runtime.return_merge_and_add_shared_cuda()
    torch.cuda.synchronize(device)
    local_slice_merge_matches = torch.equal(local_slice_torch, runtime.output)

    runtime.run_torch_reference()
    torch.cuda.synchronize(device)
    full_torch = runtime.output.clone()
    runtime.run_generated()
    torch.cuda.synchronize(device)
    full_absolute_error = (full_torch.float() - runtime.output.float()).abs()
    full_matches = torch.allclose(full_torch, runtime.output, atol=0.03125, rtol=0.01)
    first_generated_output = runtime.output.clone()
    runtime.run_generated()
    torch.cuda.synchronize(device)
    deterministic = torch.equal(first_generated_output, runtime.output)
    finite = bool(torch.isfinite(runtime.output).all().item())
    if not all(
        (
            padded_pack_matches,
            concatenated_w13_matches,
            swiglu_matches,
            concatenated_swiglu_matches,
            received_fma_matches,
            received_explicit_matches,
            received_fma_deterministic,
            received_explicit_deterministic,
            concatenated_precombine_matches,
            concatenated_precombine_deterministic,
            local_slice_merge_matches,
            full_matches,
            deterministic,
            finite,
        )
    ):
        raise RuntimeError(
            "correctness smoke failed: "
            f"padded_pack={padded_pack_matches}, concatenated_w13={concatenated_w13_matches}, "
            f"swiglu={swiglu_matches}, concatenated_swiglu={concatenated_swiglu_matches}, "
            f"received_fma={received_fma_matches}, received_explicit={received_explicit_matches}, "
            f"received_fma_deterministic={received_fma_deterministic}, "
            f"received_explicit_deterministic={received_explicit_deterministic}, "
            f"concatenated_precombine={concatenated_precombine_matches}, "
            f"concatenated_precombine_deterministic={concatenated_precombine_deterministic}, "
            f"local_slice_merge={local_slice_merge_matches}, "
            f"full={full_matches}, deterministic={deterministic}, finite={finite}"
        )

    stage_functions = {
        "coalesced_dispatch_local_simulation": runtime.coalesced_dispatch,
        "padded_pack_torch": runtime.padded_pack_torch,
        "padded_pack_cuda": runtime.padded_pack_cuda,
        "w13": runtime.w13,
        "w13_concatenated": runtime.w13_concatenated,
        "standalone_swiglu_torch": runtime.apply_swiglu_torch,
        "standalone_swiglu_cuda": runtime.apply_swiglu_cuda,
        "standalone_swiglu_concatenated_cuda": runtime.apply_swiglu_concatenated_cuda,
        "w2": runtime.w2,
        "precombine_received_merge_torch": runtime.merge_received_torch,
        "precombine_received_merge_cuda_fma": runtime.merge_received_cuda_fma,
        "precombine_received_merge_cuda": runtime.merge_received_cuda,
        "local_slice_merge_shared_add_torch_diagnostic": runtime.return_merge_and_add_shared_torch,
        "local_slice_merge_shared_add_cuda_diagnostic": runtime.return_merge_and_add_shared_cuda,
        "precombine_local_composition_torch": runtime.run_precombine_torch,
        "precombine_local_composition_generated": runtime.run_precombine_generated,
        "precombine_local_composition_concatenated": runtime.run_precombine_concatenated,
        "local_slice_composition_torch_diagnostic": runtime.run_torch_reference,
        "local_slice_composition_generated_diagnostic": runtime.run_generated,
    }
    timings: dict[str, dict[str, Any]] = {}
    for name, function in stage_functions.items():
        samples, summary = _measure(function, warmup=args.warmup, iterations=args.iterations, device=device)
        timings[name] = {"samples_ms": samples, **asdict(summary)}

    interleaved_functions = {
        "w13": (runtime.w13, runtime.w13_concatenated),
        "w13_swiglu": (runtime.w13_swiglu_generated, runtime.w13_swiglu_concatenated),
        "precombine_local_composition": (runtime.run_precombine_generated, runtime.run_precombine_concatenated),
    }
    interleaved_timings: dict[str, dict[str, dict[str, Any]]] = {}
    for name, (separate_function, concatenated_function) in interleaved_functions.items():
        separate_measurement, concatenated_measurement = _measure_interleaved(
            separate_function,
            concatenated_function,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        separate_samples, separate_summary = separate_measurement
        concatenated_samples, concatenated_summary = concatenated_measurement
        delta_samples = [
            concatenated_sample - separate_sample
            for separate_sample, concatenated_sample in zip(separate_samples, concatenated_samples, strict=True)
        ]
        delta_summary = TimingSummary(
            median_ms=statistics.median(delta_samples),
            mean_ms=statistics.fmean(delta_samples),
            minimum_ms=min(delta_samples),
            maximum_ms=max(delta_samples),
        )
        interleaved_timings[name] = {
            "separate": {"samples_ms": separate_samples, **asdict(separate_summary)},
            "concatenated": {"samples_ms": concatenated_samples, **asdict(concatenated_summary)},
            "concatenated_minus_separate": {"samples_ms": delta_samples, **asdict(delta_summary)},
        }

    properties = torch.cuda.get_device_properties(device)
    logical_flops = 6 * plan.assignments * args.hidden_size * args.intermediate_size
    padded_flops = 6 * plan.padded_rows * args.hidden_size * args.intermediate_size
    full_median = timings["precombine_local_composition_generated"]["median_ms"]
    concatenated_full_median = timings["precombine_local_composition_concatenated"]["median_ms"]
    interleaved_full = interleaved_timings["precombine_local_composition"]
    interleaved_separate_full_median = interleaved_full["separate"]["median_ms"]
    interleaved_concatenated_full_median = interleaved_full["concatenated"]["median_ms"]
    interleaved_full_delta_median = interleaved_full["concatenated_minus_separate"]["median_ms"]
    selected_gate_up_layout = "concatenated" if interleaved_full_delta_median < 0.0 else "separate"
    result = {
        "schema_version": 1,
        "benchmark": "deepep_boundary_mok_gmm_local_composition",
        "status": "ok_already_dispatched_local",
        "scope": {
            "transport": "simulated_exact_coalesced_local_boundary",
            "compute": "standalone_mok_grouped_gemm_probe",
            "scheduler": "ordinary_torch_launch_sequence_not_mok_event_graph",
            "shared_expert": "precomputed_output_add_only",
            "primary_timing": (
                "receiver-local coalesced gather, padded pack, W13, SwiGLU, W2, and deterministic pre-combine "
                "merge over every received token; official DeepEP combine and shared expert compute are excluded"
            ),
            "local_slice_diagnostic": (
                "the fused merge/shared-add diagnostic computes only owner_rank's source-token slice and is not a "
                "replacement for cross-rank DeepEP combine"
            ),
            "distributed_followup": "validate this receiver-local plan against official pinned DeepEP dispatch/combine",
            "official_combine_numerical_policy": (
                "pinned intranode combine enumerates contributing ranks in ascending rank order and adds them "
                "sequentially in FP32 before BF16 conversion; it does not use atomic accumulation"
            ),
        },
        "source": source,
        "route_fixture": {"path": str(args.route_fixture), "sha256": fixture_sha256},
        "shape": {
            "owner_rank": args.owner_rank,
            "world_size": args.world_size,
            "local_tokens": args.local_tokens,
            "global_tokens": global_tokens,
            "global_experts": args.global_experts,
            "local_experts": args.global_experts // args.world_size,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "received_coalesced_tokens": plan.received_tokens,
            "local_assignments": plan.assignments,
            "padded_rows": plan.padded_rows,
            "minimum_actual_expert_rows": int(plan.actual_counts.min()),
            "maximum_actual_expert_rows": int(plan.actual_counts.max()),
            "minimum_padded_expert_rows": int(plan.padded_counts.min()),
            "maximum_padded_expert_rows": int(plan.padded_counts.max()),
        },
        "configuration": {
            "alignment": TILE_ROWS,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "probe_extension": str(extension_path),
        },
        "correctness_smoke": {
            "finite": finite,
            "padded_pack_matches_torch_bitwise": padded_pack_matches,
            "concatenated_w13_matches_separate": bool(concatenated_w13_matches),
            "concatenated_w13_matches_separate_bitwise": concatenated_w13_bitwise,
            "concatenated_w13_maximum_absolute_error": float(
                max(concatenated_gate_error.max().item(), concatenated_up_error.max().item())
            ),
            "concatenated_w13_mean_absolute_error": float(
                (concatenated_gate_error.sum() + concatenated_up_error.sum()).item()
                / (concatenated_gate_error.numel() + concatenated_up_error.numel())
            ),
            "swiglu_matches_torch": bool(swiglu_matches),
            "swiglu_maximum_absolute_error": float(swiglu_absolute_error.max().item()),
            "swiglu_mean_absolute_error": float(swiglu_absolute_error.mean().item()),
            "concatenated_swiglu_matches_separate": bool(concatenated_swiglu_matches),
            "concatenated_swiglu_matches_separate_bitwise": concatenated_swiglu_bitwise,
            "concatenated_swiglu_maximum_absolute_error": float(concatenated_swiglu_absolute_error.max().item()),
            "concatenated_swiglu_mean_absolute_error": float(concatenated_swiglu_absolute_error.mean().item()),
            "received_merge_fma_matches_torch": bool(received_fma_matches),
            "received_merge_fma_matches_torch_bitwise": torch.equal(received_torch, received_fma),
            "received_merge_fma_maximum_absolute_error": float(received_fma_absolute_error.max().item()),
            "received_merge_fma_mean_absolute_error": float(received_fma_absolute_error.mean().item()),
            "received_merge_fma_repeat_bitwise_equal": received_fma_deterministic,
            "received_merge_explicit_matches_torch": bool(received_explicit_matches),
            "received_merge_explicit_matches_torch_bitwise": received_explicit_bitwise,
            "received_merge_explicit_maximum_absolute_error": float(received_explicit_absolute_error.max().item()),
            "received_merge_explicit_mean_absolute_error": float(received_explicit_absolute_error.mean().item()),
            "received_merge_explicit_repeat_bitwise_equal": received_explicit_deterministic,
            "concatenated_precombine_matches_separate": bool(concatenated_precombine_matches),
            "concatenated_precombine_matches_separate_bitwise": concatenated_precombine_bitwise,
            "concatenated_precombine_maximum_absolute_error": float(concatenated_precombine_absolute_error.max().item()),
            "concatenated_precombine_mean_absolute_error": float(concatenated_precombine_absolute_error.mean().item()),
            "concatenated_precombine_repeat_bitwise_equal": concatenated_precombine_deterministic,
            "local_slice_merge_matches_torch_bitwise": local_slice_merge_matches,
            "full_generated_matches_torch": bool(full_matches),
            "full_maximum_absolute_error": float(full_absolute_error.max().item()),
            "full_mean_absolute_error": float(full_absolute_error.mean().item()),
            "repeat_bitwise_equal": deterministic,
            "output_mean": float(runtime.output.float().mean().item()),
            "output_max_abs": float(runtime.output.float().abs().max().item()),
        },
        "route_metadata_correspondence": metadata_correspondence,
        "gate_up_layout_comparison": {
            "selected_by_interleaved_receiver_local_median": selected_gate_up_layout,
            "separate_w13_median_ms": timings["w13"]["median_ms"],
            "concatenated_w13_median_ms": timings["w13_concatenated"]["median_ms"],
            "separate_swiglu_median_ms": timings["standalone_swiglu_cuda"]["median_ms"],
            "concatenated_swiglu_median_ms": timings["standalone_swiglu_concatenated_cuda"]["median_ms"],
            "separate_precombine_median_ms": full_median,
            "concatenated_precombine_median_ms": concatenated_full_median,
            "concatenated_precombine_delta_ms": concatenated_full_median - full_median,
            "concatenated_precombine_relative_change": concatenated_full_median / full_median - 1.0,
            "interleaved_separate_precombine_median_ms": interleaved_separate_full_median,
            "interleaved_concatenated_precombine_median_ms": interleaved_concatenated_full_median,
            "interleaved_concatenated_precombine_delta_ms": (
                interleaved_concatenated_full_median - interleaved_separate_full_median
            ),
            "interleaved_paired_precombine_delta_median_ms": interleaved_full_delta_median,
            "interleaved_concatenated_precombine_relative_change": (
                interleaved_concatenated_full_median / interleaved_separate_full_median - 1.0
            ),
        },
        "timing": timings,
        "interleaved_timing": interleaved_timings,
        "throughput": {
            "logical_flops": logical_flops,
            "padded_flops": padded_flops,
            "logical_tflops": logical_flops / (full_median * 1e9),
            "padded_tflops": padded_flops / (full_median * 1e9),
            "concatenated_logical_tflops": logical_flops / (concatenated_full_median * 1e9),
            "concatenated_padded_tflops": padded_flops / (concatenated_full_median * 1e9),
        },
        "environment": {
            "device": str(device),
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
