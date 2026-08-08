# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark a compiler-planned four-rank DeepEP plus MoK-primitive runtime."""

import argparse
import base64
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as torch_functional

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from backends.gb200_deepep_mok_local import (  # noqa: E402  # pyrefly: ignore[missing-import]
    DEEPEP_COMMIT,
    TILE_ROWS,
    LocalRoutePlan,
    _git_revision,
    _load_routes,
    _local_route_plan,
    _validate_generated_map_fold_extension,
)
from benchmark_metadata import (  # noqa: E402  # pyrefly: ignore[missing-import]
    canonical_json_sha256,
    command_record,
    file_sha256,
    framed_tensor_sha256,
    nvidia_smi_snapshot,
    toolchain_snapshot,
)
from gb200_mok_gmm_probe import (  # noqa: E402  # pyrefly: ignore[missing-import]
    _load_extension,
    _validate_checkout,
)

from tile_lifetime import (  # noqa: E402
    DType,
    ExpertParallelConfig,
    ExpertParallelPlan,
    ExpertParallelStageKind,
    NumericalPolicy,
    compile_stablehlo_expert_parallel_region,
)
from tile_lifetime.cuda_map_fold_codegen import shuttle_map_fold_program  # noqa: E402
from tile_lifetime.moe_reference import MOE_REGION_INPUT_NAMES  # noqa: E402
from tile_lifetime.relation import build_partitioned_merge_rows  # noqa: E402

SCHEMA_VERSION = 2
SHUTTLE_BENCHMARK = "four_rank_deepep_compiler_relation_mok_primitives"


@dataclass(frozen=True)
class TimingSummary:
    """Rank-max timing statistics in milliseconds."""

    median_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


class GateUpLayout(StrEnum):
    """Physical layout for routed and shared gate/up projections."""

    CONCATENATED_E_2I_K = "concatenated_e_2i_k"
    SEPARATE_E_I_K = "separate_e_i_k"


class RoutingSource(StrEnum):
    """Source of routing metadata for a benchmark execution."""

    FIXTURE = "fixture"
    NATURAL_STABLEHLO = "natural_stablehlo"


class AcceptanceOrder(StrEnum):
    """Launch order for the matched Shuttle and oracle timing phases."""

    SHUTTLE_FIRST = "shuttle_first"
    ORACLE_FIRST = "oracle_first"


@dataclass(frozen=True)
class TransportLegalization:
    """Map DeepEP receiver order to the compiler relation's receiver order."""

    deep_global_source_tokens: np.ndarray
    relation_receiver_row_for_deep_row: np.ndarray
    deep_receiver_row_for_relation_row: np.ndarray
    padded_deep_receiver_rows: np.ndarray
    deep_route_padded_rows: np.ndarray
    deep_route_weights: np.ndarray
    source_rank_counts: np.ndarray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-fixture", type=Path)
    parser.add_argument(
        "--routing-source",
        type=RoutingSource,
        choices=list(RoutingSource),
        default=RoutingSource.FIXTURE,
    )
    parser.add_argument(
        "--stablehlo-fixture",
        type=Path,
        help="Base64 portable StableHLO exported from the ordinary JAX MoE region.",
    )
    parser.add_argument("--probe-extension", type=Path, required=True)
    parser.add_argument("--mok-root", type=Path, required=True)
    parser.add_argument("--deepep-root", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--clock-policy", default="cluster_default_unpinned")
    parser.add_argument("--local-tokens", type=int, default=2_048)
    parser.add_argument("--global-experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=7_168)
    parser.add_argument("--intermediate-size", type=int, default=3_072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-bytes", type=int, default=int(2e9))
    parser.add_argument("--deepep-sms", type=int, default=24)
    parser.add_argument(
        "--gate-up-layout",
        type=GateUpLayout,
        choices=list(GateUpLayout),
        default=GateUpLayout.CONCATENATED_E_2I_K,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--acceptance-order",
        type=AcceptanceOrder,
        choices=list(AcceptanceOrder),
        default=AcceptanceOrder.SHUTTLE_FIRST,
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--semantic-fixture-output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if world_size != 4:
        raise ValueError(f"this benchmark requires exactly four ranks, got {world_size}")
    if args.global_experts <= 0 or args.global_experts % world_size:
        raise ValueError("global-experts must be positive and divisible by world size")
    if args.local_tokens <= 0 or args.local_tokens % TILE_ROWS:
        raise ValueError(f"local-tokens must be positive and divisible by {TILE_ROWS}")
    if args.hidden_size <= 0 or args.hidden_size % 256:
        raise ValueError("hidden-size must be positive and divisible by 256")
    if args.intermediate_size <= 0 or args.intermediate_size % 256:
        raise ValueError("intermediate-size must be positive and divisible by 256")
    if args.top_k <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("top-k/iterations must be positive and warmup non-negative")
    if _git_revision(args.deepep_root.resolve()) != DEEPEP_COMMIT:
        raise ValueError(f"DeepEP checkout must be pinned at {DEEPEP_COMMIT}")
    _validate_checkout(args.mok_root.resolve())
    if not args.probe_extension.is_file():
        raise FileNotFoundError(args.probe_extension)
    if args.routing_source is RoutingSource.FIXTURE and args.route_fixture is None:
        raise ValueError("--route-fixture is required for fixture routing")
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        if args.stablehlo_fixture is None or not args.stablehlo_fixture.is_file():
            raise ValueError("--stablehlo-fixture is required for natural StableHLO routing")


class NaturalRouterRuntime:
    """Execute the recovered router Contract and normalized top-k Fold/Maps."""

    def __init__(
        self,
        x: torch.Tensor,
        *,
        plan: ExpertParallelPlan,
        seed: int,
    ) -> None:
        self.x = x
        self.plan = plan
        relation = plan.route_relation
        if relation.token_count != x.shape[0]:
            raise ValueError(f"recovered Relation expects {relation.token_count} tokens, runtime has {x.shape[0]}")
        plan.stage(ExpertParallelStageKind.ROUTE_RELATION)
        self.top_k = relation.slots_per_token
        torch.manual_seed(seed)
        self.weight = torch.empty(
            (relation.global_expert_count, x.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        ).normal_(0.0, 1.0 / math.sqrt(x.shape[1]))
        self.logits = torch.empty(
            (x.shape[0], relation.global_expert_count),
            dtype=torch.bfloat16,
            device=x.device,
        )
        self.top_values = torch.empty((x.shape[0], self.top_k), dtype=torch.bfloat16, device=x.device)
        self.expert_indices = torch.empty((x.shape[0], self.top_k), dtype=torch.int64, device=x.device)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.mm(self.x, self.weight.T, out=self.logits)
        torch.topk(
            self.logits,
            self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
            out=(self.top_values, self.expert_indices),
        )
        route_weights = torch.softmax(self.top_values.float(), dim=-1)
        return self.expert_indices, route_weights


def _compile_natural_stablehlo_plan(args: argparse.Namespace) -> tuple[ExpertParallelPlan, dict[str, Any]]:
    """Validate that the benchmark shape enters through ordinary JAX StableHLO."""
    assert args.stablehlo_fixture is not None
    artifact = base64.b64decode(args.stablehlo_fixture.read_text())
    plan = compile_stablehlo_expert_parallel_region(
        artifact,
        input_names=MOE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        config=ExpertParallelConfig(
            expert_parallel_size=4,
            exchange_workers=args.deepep_sms,
        ),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    expected = (args.local_tokens, args.top_k, args.global_experts)
    observed = (
        plan.route_relation.token_count,
        plan.route_relation.slots_per_token,
        plan.route_relation.global_expert_count,
    )
    if observed != expected:
        raise ValueError(f"StableHLO plan shape {observed} does not match benchmark {expected}")
    return (
        plan,
        {
            "portable_artifact_sha256": file_sha256(args.stablehlo_fixture),
            "semantic_input": "ordinary JAX shared-plus-routed MoE",
            "semantic_operations": ["Contract", "selection", "Relation", "SegmentedContract", "Map", "Fold"],
            "runtime_lineage": "router shapes and route Relation are read from the recovered StableHLO plan",
            "physical_stage_count": len(plan.stages),
            "forward_transport": plan.schedule.forward_transport.implementation,
            "reverse_transport": plan.schedule.reverse_transport.implementation,
            "merge": plan.schedule.merge_implementation,
        },
    )


def _legalize_transport_order(
    plan: LocalRoutePlan,
    combine_weights: np.ndarray,
    *,
    rank: int,
    local_tokens: int,
    rank_prefix_matrix: torch.Tensor,
    recv_src_idx: torch.Tensor,
) -> TransportLegalization:
    prefix = rank_prefix_matrix[:, rank].cpu().numpy().astype(np.int64, copy=False)
    source_rank_counts = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), prefix)))
    deep_source_ranks = np.repeat(np.arange(source_rank_counts.size, dtype=np.int64), source_rank_counts)
    deep_source_local_tokens = recv_src_idx.cpu().numpy().astype(np.int64, copy=False)
    if deep_source_ranks.size != deep_source_local_tokens.size:
        raise AssertionError("DeepEP rank prefixes and recv_src_idx disagree")
    deep_global_source_tokens = deep_source_ranks * local_tokens + deep_source_local_tokens

    relation_receiver_row_for_deep_row = np.searchsorted(plan.receiver_source_tokens, deep_global_source_tokens)
    safe_rows = np.minimum(relation_receiver_row_for_deep_row, plan.receiver_source_tokens.size - 1)
    exact = (relation_receiver_row_for_deep_row < plan.receiver_source_tokens.size) & (
        plan.receiver_source_tokens[safe_rows] == deep_global_source_tokens
    )
    if not np.all(exact):
        raise AssertionError("DeepEP receiver rows are not present in the compiler coalesced relation")
    if np.unique(relation_receiver_row_for_deep_row).size != plan.received_tokens:
        raise AssertionError("DeepEP-to-relation receiver mapping is not one-to-one")

    deep_receiver_row_for_relation_row = np.empty(plan.received_tokens, dtype=np.int64)
    deep_receiver_row_for_relation_row[relation_receiver_row_for_deep_row] = np.arange(
        plan.received_tokens,
        dtype=np.int64,
    )
    padded_deep_receiver_rows = np.full(plan.padded_receiver_rows.shape, -1, dtype=np.int64)
    valid_padded = plan.padded_receiver_rows >= 0
    padded_deep_receiver_rows[valid_padded] = deep_receiver_row_for_relation_row[plan.padded_receiver_rows[valid_padded]]
    return TransportLegalization(
        deep_global_source_tokens=deep_global_source_tokens,
        relation_receiver_row_for_deep_row=relation_receiver_row_for_deep_row,
        deep_receiver_row_for_relation_row=deep_receiver_row_for_relation_row,
        padded_deep_receiver_rows=padded_deep_receiver_rows,
        deep_route_padded_rows=plan.route_padded_rows[relation_receiver_row_for_deep_row],
        deep_route_weights=combine_weights[deep_global_source_tokens],
        source_rank_counts=source_rank_counts,
    )


def _device_long(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.int64, device=device)


class DistributedPhysicalRuntime:
    """Buffers and launches for one rank of the generated physical plan."""

    def __init__(
        self,
        module: ModuleType,
        deep_ep: ModuleType,
        buffer: Any,
        handle: tuple[Any, ...],
        x: torch.Tensor,
        plan: LocalRoutePlan,
        legalization: TransportLegalization,
        *,
        rank: int,
        hidden_size: int,
        intermediate_size: int,
        seed: int,
        gate_up_layout: GateUpLayout,
    ) -> None:
        self.module = module
        self.deep_ep = deep_ep
        self.buffer = buffer
        self.handle = handle
        self.x = x
        self.dispatch_config = deep_ep.Buffer.get_dispatch_config(dist.get_world_size())
        self.combine_config = deep_ep.Buffer.get_combine_config(dist.get_world_size())
        self.device = x.device
        self.gate_up_layout = gate_up_layout
        self.last_output = torch.empty_like(x)

        local_experts = plan.padded_counts.size
        self.global_experts = local_experts * dist.get_world_size()
        weight_scale_w13 = 1.0 / math.sqrt(hidden_size)
        weight_scale_w2 = 1.0 / math.sqrt(intermediate_size)
        torch.manual_seed(seed + 10_000 + rank)
        gate_weights = torch.empty(
            (local_experts, intermediate_size, hidden_size), dtype=torch.bfloat16, device=x.device
        ).normal_(0.0, weight_scale_w13)
        up_weights = torch.empty_like(gate_weights).normal_(0.0, weight_scale_w13)
        if gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            self.gate_up_weights = torch.cat((gate_weights, up_weights), dim=1).contiguous()
            self.gate_weights = self.gate_up_weights[:, :intermediate_size]
            self.up_weights = self.gate_up_weights[:, intermediate_size:]
        else:
            self.gate_up_weights = None
            self.gate_weights = gate_weights
            self.up_weights = up_weights
        self.down_weights = torch.empty(
            (local_experts, hidden_size, intermediate_size), dtype=torch.bfloat16, device=x.device
        ).normal_(0.0, weight_scale_w2)

        torch.manual_seed(seed + 20_000)
        shared_gate_weights = torch.empty(
            (1, intermediate_size, hidden_size), dtype=torch.bfloat16, device=x.device
        ).normal_(0.0, weight_scale_w13)
        shared_up_weights = torch.empty_like(shared_gate_weights).normal_(0.0, weight_scale_w13)
        if gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            self.shared_gate_up_weights = torch.cat((shared_gate_weights, shared_up_weights), dim=1).contiguous()
            self.shared_gate_weights = self.shared_gate_up_weights[:, :intermediate_size]
            self.shared_up_weights = self.shared_gate_up_weights[:, intermediate_size:]
        else:
            self.shared_gate_up_weights = None
            self.shared_gate_weights = shared_gate_weights
            self.shared_up_weights = shared_up_weights
        self.shared_down_weights = torch.empty(
            (1, hidden_size, intermediate_size), dtype=torch.bfloat16, device=x.device
        ).normal_(0.0, weight_scale_w2)

        self.padded_counts = torch.as_tensor(plan.padded_counts, dtype=torch.int32, device=x.device)
        self.padded_deep_receiver_rows = _device_long(legalization.padded_deep_receiver_rows, x.device)
        self.deep_route_padded_rows = _device_long(legalization.deep_route_padded_rows, x.device)
        self.deep_route_weights = torch.as_tensor(legalization.deep_route_weights, dtype=torch.float32, device=x.device)
        self.dynamic_group_counts = torch.empty(local_experts, dtype=torch.int32, device=x.device)
        self.dynamic_relation_overflow = torch.empty(local_experts, dtype=torch.int32, device=x.device)
        self.dynamic_fixed_group_capacity = bool(
            plan.padded_counts.size > 0 and np.all(plan.padded_counts == plan.padded_counts[0])
        )
        self.shared_counts = torch.tensor([x.shape[0]], dtype=torch.int32, device=x.device)

        self.reverse_send_counts = [int(count) for count in legalization.source_rank_counts]
        send_counts = torch.as_tensor(self.reverse_send_counts, dtype=torch.int64, device=x.device)
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(receive_counts, send_counts)
        self.reverse_receive_counts = [int(count) for count in receive_counts.cpu().tolist()]
        send_source_tokens = torch.as_tensor(
            legalization.deep_global_source_tokens % x.shape[0],
            dtype=torch.int64,
            device=x.device,
        )
        returned_source_tokens = torch.empty(
            sum(self.reverse_receive_counts),
            dtype=torch.int64,
            device=x.device,
        )
        dist.all_to_all_single(
            returned_source_tokens,
            send_source_tokens,
            output_split_sizes=self.reverse_receive_counts,
            input_split_sizes=self.reverse_send_counts,
        )
        rank_token_rows = build_partitioned_merge_rows(
            returned_source_tokens.cpu().numpy(),
            np.asarray(self.reverse_receive_counts, dtype=np.int64),
            source_item_count=x.shape[0],
        )
        self.rank_token_rows = _device_long(rank_token_rows, x.device)

        padded_rows = plan.padded_rows
        received_tokens = plan.received_tokens
        self.padded_x = torch.empty((padded_rows, hidden_size), dtype=torch.bfloat16, device=x.device)
        self.gate = torch.empty((padded_rows, intermediate_size), dtype=torch.bfloat16, device=x.device)
        self.up = torch.empty_like(self.gate)
        self.gate_up = torch.empty((padded_rows, 2 * intermediate_size), dtype=torch.bfloat16, device=x.device)
        self.swiglu = torch.empty_like(self.gate)
        self.down = torch.empty((padded_rows, hidden_size), dtype=torch.bfloat16, device=x.device)
        self.recv_merged = torch.empty((received_tokens, hidden_size), dtype=torch.bfloat16, device=x.device)
        self.returned_rank_partials = torch.empty(
            (sum(self.reverse_receive_counts), hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        self.shared_gate = torch.empty((x.shape[0], intermediate_size), dtype=torch.bfloat16, device=x.device)
        self.shared_up = torch.empty_like(self.shared_gate)
        self.shared_gate_up = torch.empty(
            (x.shape[0], 2 * intermediate_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        self.shared_swiglu = torch.empty_like(self.shared_gate)
        self.shared_output = torch.empty_like(x)
        self.coarse_recv_x = torch.empty((received_tokens, hidden_size), dtype=torch.bfloat16, device=x.device)
        self.coarse_padded_x = torch.empty_like(self.padded_x)
        self.coarse_gate = torch.empty_like(self.gate)
        self.coarse_up = torch.empty_like(self.up)
        self.coarse_gate_up = torch.empty_like(self.gate_up)
        self.coarse_swiglu = torch.empty_like(self.swiglu)
        self.coarse_down = torch.empty_like(self.down)
        self.coarse_shared_x = torch.empty_like(x)
        self.coarse_shared_gate = torch.empty_like(self.shared_gate)
        self.coarse_shared_up = torch.empty_like(self.shared_up)
        self.coarse_shared_gate_up = torch.empty_like(self.shared_gate_up)
        self.coarse_shared_swiglu = torch.empty_like(self.shared_swiglu)

    def generate_receiver_relation(
        self,
        received_expert_indices: torch.Tensor,
        received_route_weights: torch.Tensor,
    ) -> None:
        """Generate the bounded destination-group index plane on the device."""
        if not self.dynamic_fixed_group_capacity:
            raise ValueError("the generated fixed-capacity RelationPlan requires equal padded group capacities")
        self.module.fixed_capacity_relation_plan_out(
            received_expert_indices,
            received_route_weights,
            self.padded_deep_receiver_rows,
            self.deep_route_padded_rows,
            self.deep_route_weights,
            self.dynamic_group_counts,
            self.dynamic_relation_overflow,
        )

    def dispatch_natural_routes(
        self,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        asynchronous: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[Any, ...], Any]:
        """Dispatch payload using relation edges produced by the natural frontend."""
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = self.buffer.get_dispatch_layout(
            expert_indices,
            self.global_experts,
        )
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, event = self.buffer.dispatch(
            x=self.x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=expert_indices,
            topk_weights=route_weights,
            expert_alignment=1,
            config=self.dispatch_config,
            async_finish=asynchronous,
        )
        self.handle = handle
        return recv_x, recv_topk_idx, recv_topk_weights, handle, event

    def dispatch(self, *, asynchronous: bool) -> tuple[torch.Tensor, Any]:
        recv_x, _, _, _, _, event = self.buffer.dispatch(
            x=self.x,
            handle=self.handle,
            expert_alignment=1,
            config=self.dispatch_config,
            async_finish=asynchronous,
        )
        return recv_x, event

    def routed(self, recv_x: torch.Tensor) -> None:
        self.module.padded_pack_bf16_out(recv_x, self.padded_deep_receiver_rows, self.padded_x)
        if self.gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            assert self.gate_up_weights is not None
            self.module.grouped_gemm_out(self.padded_x, self.gate_up_weights, self.padded_counts, self.gate_up)
            self.module.row_halves_pair_map_bf16_out(self.gate_up, self.swiglu)
        else:
            self.module.grouped_gemm_out(self.padded_x, self.gate_weights, self.padded_counts, self.gate)
            self.module.grouped_gemm_out(self.padded_x, self.up_weights, self.padded_counts, self.up)
            self.module.adjacent_pair_map_bf16_out(self.gate, self.up, self.swiglu)
        self.module.grouped_gemm_out(self.swiglu, self.down_weights, self.padded_counts, self.down)
        self.module.indexed_weighted_ordered_fold_bf16_out(
            self.down,
            self.deep_route_padded_rows,
            self.deep_route_weights,
            self.recv_merged,
        )

    def shared(self) -> None:
        if self.gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            assert self.shared_gate_up_weights is not None
            self.module.grouped_gemm_out(
                self.x,
                self.shared_gate_up_weights,
                self.shared_counts,
                self.shared_gate_up,
            )
            self.module.row_halves_pair_map_bf16_out(self.shared_gate_up, self.shared_swiglu)
        else:
            self.module.grouped_gemm_out(self.x, self.shared_gate_weights, self.shared_counts, self.shared_gate)
            self.module.grouped_gemm_out(self.x, self.shared_up_weights, self.shared_counts, self.shared_up)
            self.module.adjacent_pair_map_bf16_out(self.shared_gate, self.shared_up, self.shared_swiglu)
        self.module.grouped_gemm_out(
            self.shared_swiglu,
            self.shared_down_weights,
            self.shared_counts,
            self.shared_output,
        )

    def combine(self) -> None:
        self.last_output, _, _ = self.buffer.combine(
            x=self.recv_merged,
            handle=self.handle,
            bias=self.shared_output,
            config=self.combine_config,
        )

    def reverse_transport_and_generated_merge(self) -> None:
        """Return rank partials as payload, then execute the compiler-owned fold."""
        dist.all_to_all_single(
            self.returned_rank_partials,
            self.recv_merged,
            output_split_sizes=self.reverse_receive_counts,
            input_split_sizes=self.reverse_send_counts,
        )
        self.module.partitioned_ordered_fold_base_map_bf16_out(
            self.returned_rank_partials,
            self.rank_token_rows,
            self.shared_output,
            self.last_output,
        )

    def sequential(self) -> None:
        recv_x, _ = self.dispatch(asynchronous=False)
        self.routed(recv_x)
        self.shared()
        self.combine()

    def overlap_shared_with_dispatch(self) -> None:
        recv_x, event = self.dispatch(asynchronous=True)
        self.shared()
        event.current_stream_wait()
        self.routed(recv_x)
        self.combine()

    def generated_sequential(self) -> None:
        recv_x, _ = self.dispatch(asynchronous=False)
        self.routed(recv_x)
        self.shared()
        self.reverse_transport_and_generated_merge()

    def generated_overlap_shared_with_dispatch(self) -> None:
        recv_x, event = self.dispatch(asynchronous=True)
        self.shared()
        event.current_stream_wait()
        self.routed(recv_x)
        self.reverse_transport_and_generated_merge()

    def generated_natural_post_route(
        self,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> None:
        """Generate runtime relation metadata and execute from natural route tensors."""
        recv_x, recv_topk_idx, recv_topk_weights, _, event = self.dispatch_natural_routes(
            expert_indices,
            route_weights,
            asynchronous=True,
        )
        self.shared()
        event.current_stream_wait()
        self.generate_receiver_relation(recv_topk_idx, recv_topk_weights)
        self.routed(recv_x)
        self.reverse_transport_and_generated_merge()

    def generated_natural_end_to_end(self, router: NaturalRouterRuntime) -> None:
        """Execute router through generated merge without route or task fixtures."""
        expert_indices, route_weights = router()
        self.generated_natural_post_route(expert_indices, route_weights)

    def routed_coarse_materialized(self, recv_x: torch.Tensor) -> None:
        """Execute routed work with deliberate activation copies between phases."""
        self.coarse_recv_x.copy_(recv_x)
        self.module.padded_pack_bf16_out(self.coarse_recv_x, self.padded_deep_receiver_rows, self.padded_x)
        self.coarse_padded_x.copy_(self.padded_x)
        if self.gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            assert self.gate_up_weights is not None
            self.module.grouped_gemm_out(
                self.coarse_padded_x,
                self.gate_up_weights,
                self.padded_counts,
                self.gate_up,
            )
            self.coarse_gate_up.copy_(self.gate_up)
            self.module.row_halves_pair_map_bf16_out(self.coarse_gate_up, self.swiglu)
        else:
            self.module.grouped_gemm_out(
                self.coarse_padded_x,
                self.gate_weights,
                self.padded_counts,
                self.gate,
            )
            self.module.grouped_gemm_out(
                self.coarse_padded_x,
                self.up_weights,
                self.padded_counts,
                self.up,
            )
            self.coarse_gate.copy_(self.gate)
            self.coarse_up.copy_(self.up)
            self.module.adjacent_pair_map_bf16_out(self.coarse_gate, self.coarse_up, self.swiglu)
        self.coarse_swiglu.copy_(self.swiglu)
        self.module.grouped_gemm_out(
            self.coarse_swiglu,
            self.down_weights,
            self.padded_counts,
            self.down,
        )
        self.coarse_down.copy_(self.down)
        self.module.indexed_weighted_ordered_fold_bf16_out(
            self.coarse_down,
            self.deep_route_padded_rows,
            self.deep_route_weights,
            self.recv_merged,
        )

    def shared_coarse_materialized(self) -> None:
        """Execute the shared expert with explicit W13 and SwiGLU copies."""
        self.coarse_shared_x.copy_(self.x)
        if self.gate_up_layout == GateUpLayout.CONCATENATED_E_2I_K:
            assert self.shared_gate_up_weights is not None
            self.module.grouped_gemm_out(
                self.coarse_shared_x,
                self.shared_gate_up_weights,
                self.shared_counts,
                self.shared_gate_up,
            )
            self.coarse_shared_gate_up.copy_(self.shared_gate_up)
            self.module.row_halves_pair_map_bf16_out(self.coarse_shared_gate_up, self.shared_swiglu)
        else:
            self.module.grouped_gemm_out(
                self.coarse_shared_x,
                self.shared_gate_weights,
                self.shared_counts,
                self.shared_gate,
            )
            self.module.grouped_gemm_out(
                self.coarse_shared_x,
                self.shared_up_weights,
                self.shared_counts,
                self.shared_up,
            )
            self.coarse_shared_gate.copy_(self.shared_gate)
            self.coarse_shared_up.copy_(self.shared_up)
            self.module.adjacent_pair_map_bf16_out(self.coarse_shared_gate, self.coarse_shared_up, self.shared_swiglu)
        self.coarse_shared_swiglu.copy_(self.shared_swiglu)
        self.module.grouped_gemm_out(
            self.coarse_shared_swiglu,
            self.shared_down_weights,
            self.shared_counts,
            self.shared_output,
        )

    def coarse_materialized_sequential(self) -> None:
        """Execute a legal schedule with intentionally coarse HBM boundaries."""
        recv_x, _ = self.dispatch(asynchronous=False)
        self.routed_coarse_materialized(recv_x)
        self.shared_coarse_materialized()
        self.combine()

    def identity_transport(self) -> None:
        recv_x, _ = self.dispatch(asynchronous=False)
        self.last_output, _, _ = self.buffer.combine(x=recv_x, handle=self.handle, config=self.combine_config)


def _rank_max_measure(
    function: Any,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[list[float], list[list[float]], TimingSummary]:
    for _ in range(warmup):
        dist.barrier()
        function()
        torch.cuda.synchronize(device)
    samples: list[float] = []
    local_samples: list[float] = []
    for _ in range(iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        local_elapsed = start.elapsed_time(end)
        rank_max = torch.tensor([local_elapsed], dtype=torch.float32, device=device)
        dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
        samples.append(float(rank_max.item()))
        local_samples.append(float(local_elapsed))
    gathered_samples: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_samples, local_samples)
    per_rank_samples = [rank_samples for rank_samples in gathered_samples if rank_samples is not None]
    return (
        samples,
        per_rank_samples,
        TimingSummary(
            median_ms=statistics.median(samples),
            mean_ms=statistics.fmean(samples),
            minimum_ms=min(samples),
            maximum_ms=max(samples),
        ),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape, and C-order bytes."""
    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.bfloat16:
        payload = contiguous.view(torch.uint16).numpy().tobytes(order="C")
    else:
        payload = contiguous.numpy().tobytes(order="C")
    return framed_tensor_sha256(str(contiguous.dtype), tuple(contiguous.shape), payload)


def _mapping_checks(
    legalization: TransportLegalization,
    plan: LocalRoutePlan,
    selected_experts: np.ndarray,
    combine_weights: np.ndarray,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    x: torch.Tensor,
    *,
    rank: int,
    local_tokens: int,
    global_experts: int,
) -> dict[str, Any]:
    local_experts = global_experts // dist.get_world_size()
    source_tokens = legalization.deep_global_source_tokens
    expected_global_experts = selected_experts[source_tokens]
    expected_local_experts = np.where(
        expected_global_experts // local_experts == rank,
        expected_global_experts % local_experts,
        -1,
    )
    topk_exact = np.array_equal(recv_topk_idx.cpu().numpy(), expected_local_experts)
    valid_routes = expected_local_experts >= 0
    weights_exact = np.array_equal(
        recv_topk_weights.cpu().numpy()[valid_routes],
        combine_weights[source_tokens][valid_routes],
    )
    counts_exact = np.array_equal(
        np.bincount(expected_local_experts[expected_local_experts >= 0], minlength=local_experts),
        plan.actual_counts,
    )

    global_x = torch.empty((local_tokens * dist.get_world_size(), x.shape[1]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(global_x, x)
    source_index = torch.as_tensor(source_tokens, dtype=torch.int64, device=x.device)
    payload_exact = torch.equal(recv_x, global_x.index_select(0, source_index))
    return {
        "exact": bool(topk_exact and weights_exact and counts_exact and payload_exact),
        "deep_receiver_rows": int(source_tokens.size),
        "compiler_receiver_rows": plan.received_tokens,
        "compiler_assignments": plan.assignments,
        "compiler_padded_rows": plan.padded_rows,
        "source_rank_counts": legalization.source_rank_counts.tolist(),
        "topk_local_expert_exact": topk_exact,
        "topk_valid_route_weight_exact": weights_exact,
        "topk_invalid_route_weights": "ignored because DeepEP marks these routes with local expert -1",
        "expert_counts_exact": counts_exact,
        "payload_exact": payload_exact,
        "legalization": "DeepEP source-rank prefix plus recv_src_idx to compiler global source-token relation",
    }


def _reference_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = (x.float() @ gate_weight.float().T).bfloat16()
    up = (x.float() @ up_weight.float().T).bfloat16()
    activated = (torch_functional.silu(gate.float()) * up.float()).bfloat16()
    return (activated.float() @ down_weight.float().T).bfloat16()


def _small_semantic_reference(
    module: ModuleType,
    deep_ep: ModuleType,
    buffer: Any,
    *,
    rank: int,
    device: torch.device,
    seed: int,
    gate_up_layout: GateUpLayout,
    fixture_output: Path,
) -> dict[str, Any]:
    """Compare the physical runtime with an independent source-ordered Torch MoE."""
    world_size = dist.get_world_size()
    local_tokens = 256
    hidden_size = 256
    intermediate_size = 256
    global_experts = 8
    top_k = 2
    global_tokens = local_tokens * world_size
    source_items = np.arange(global_tokens, dtype=np.int64)
    selected_experts = np.stack((source_items % global_experts, (source_items + 3) % global_experts), axis=1)
    combine_weights = np.broadcast_to(
        np.asarray([0.375, 0.625], dtype=np.float32),
        (global_tokens, top_k),
    ).copy()
    plan = _local_route_plan(
        selected_experts,
        combine_weights,
        owner_rank=rank,
        world_size=world_size,
        global_experts=global_experts,
    )
    source_start = rank * local_tokens
    source_end = source_start + local_tokens
    local_topk_idx = torch.as_tensor(selected_experts[source_start:source_end], dtype=torch.int64, device=device)
    local_topk_weights = torch.as_tensor(combine_weights[source_start:source_end], dtype=torch.float32, device=device)
    torch.manual_seed(seed + 30_000 + rank)
    x = torch.empty((local_tokens, hidden_size), dtype=torch.bfloat16, device=device).normal_(0.0, 0.1)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
        local_topk_idx,
        global_experts,
    )
    _recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=local_topk_idx,
        topk_weights=local_topk_weights,
        expert_alignment=1,
        config=deep_ep.Buffer.get_dispatch_config(world_size),
    )
    legalization = _legalize_transport_order(
        plan,
        combine_weights,
        rank=rank,
        local_tokens=local_tokens,
        rank_prefix_matrix=handle[0],
        recv_src_idx=handle[3],
    )
    runtime = DistributedPhysicalRuntime(
        module,
        deep_ep,
        buffer,
        handle,
        x,
        plan,
        legalization,
        rank=rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed + 40_000,
        gate_up_layout=gate_up_layout,
    )
    runtime.generated_sequential()
    torch.cuda.synchronize(device)
    generated = runtime.last_output.clone()

    def gather_experts(local_weights: torch.Tensor) -> torch.Tensor:
        local_weights = local_weights.contiguous()
        gathered = torch.empty(
            (local_weights.shape[0] * world_size, *local_weights.shape[1:]),
            dtype=local_weights.dtype,
            device=device,
        )
        dist.all_gather_into_tensor(gathered, local_weights)
        return gathered

    global_gate_weights = gather_experts(runtime.gate_weights)
    global_up_weights = gather_experts(runtime.up_weights)
    global_down_weights = gather_experts(runtime.down_weights)
    reference_routed = torch.zeros_like(x, dtype=torch.float32)
    for route_slot in range(top_k):
        slot_output = torch.empty_like(x)
        slot_experts = local_topk_idx[:, route_slot]
        for expert in range(global_experts):
            token_mask = slot_experts == expert
            if token_mask.any():
                slot_output[token_mask] = _reference_mlp(
                    x[token_mask],
                    global_gate_weights[expert],
                    global_up_weights[expert],
                    global_down_weights[expert],
                )
        reference_routed.add_(slot_output.float() * local_topk_weights[:, route_slot, None])
    reference_shared = _reference_mlp(
        x,
        runtime.shared_gate_weights[0],
        runtime.shared_up_weights[0],
        runtime.shared_down_weights[0],
    )
    reference = (reference_routed + reference_shared.float()).bfloat16()
    absolute_error = (generated.float() - reference.float()).abs()
    passed = bool(torch.allclose(generated, reference, atol=0.125, rtol=0.05))
    fixture_output.parent.mkdir(parents=True, exist_ok=True)
    rank_fixture = fixture_output.with_name(f"{fixture_output.stem}-rank{rank}{fixture_output.suffix}")
    np.savez(
        rank_fixture,
        input_bf16_bits=x.view(torch.uint16).cpu().numpy(),
        selected_experts=selected_experts[source_start:source_end],
        combine_weights=combine_weights[source_start:source_end],
        generated_bf16_bits=generated.view(torch.uint16).cpu().numpy(),
        reference_bf16_bits=reference.view(torch.uint16).cpu().numpy(),
    )
    return {
        "passed": passed,
        "independent": True,
        "reference": "source-ordered Torch BF16 boundaries with FP32 route-slot accumulation",
        "physical_ordering_difference": (
            "generated path first collapses routes per owner to BF16, then returns payload with all-to-all and "
            "combines owners in fixed rank order"
        ),
        "shape": {
            "local_tokens": local_tokens,
            "global_experts": global_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "atol": 0.125,
        "rtol": 0.05,
        "maximum_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "finite": bool(torch.isfinite(generated).all().item()),
        "generated_output_sha256": _tensor_sha256(generated),
        "reference_output_sha256": _tensor_sha256(reference),
        "fixture": {
            "path": str(rank_fixture),
            "sha256": file_sha256(rank_fixture),
            "weight_generation": {
                "routed_seed": seed + 50_000 + rank,
                "shared_seed": seed + 60_000,
            },
            "routed_weight_sha256": {
                "gate": _tensor_sha256(runtime.gate_weights),
                "up": _tensor_sha256(runtime.up_weights),
                "down": _tensor_sha256(runtime.down_weights),
            },
            "shared_weight_sha256": {
                "gate": _tensor_sha256(runtime.shared_gate_weights),
                "up": _tensor_sha256(runtime.shared_up_weights),
                "down": _tensor_sha256(runtime.shared_down_weights),
            },
        },
    }


def main() -> None:
    args = _parser().parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _validate_args(args, world_size)

    import deep_ep  # noqa: PLC0415  # pyrefly: ignore[missing-import]

    deep_ep.Buffer.set_num_sms(args.deepep_sms)
    buffer = deep_ep.Buffer(dist.group.WORLD, args.buffer_bytes, explicitly_destroy=True)
    natural_compilation = (
        _compile_natural_stablehlo_plan(args) if args.routing_source is RoutingSource.NATURAL_STABLEHLO else None
    )
    natural_plan = natural_compilation[0] if natural_compilation is not None else None
    natural_frontend = natural_compilation[1] if natural_compilation is not None else None
    torch.manual_seed(args.seed + rank)
    x = torch.empty((args.local_tokens, args.hidden_size), dtype=torch.bfloat16, device=device).normal_(0.0, 0.1)
    router = NaturalRouterRuntime(x, plan=natural_plan, seed=args.seed + 30_000) if natural_plan is not None else None
    source_start = rank * args.local_tokens
    source_end = source_start + args.local_tokens
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        assert router is not None
        local_topk_idx, local_topk_weights = router()
        gathered_experts = torch.empty(
            (args.local_tokens * world_size, args.top_k),
            dtype=torch.int64,
            device=device,
        )
        gathered_weights = torch.empty(
            (args.local_tokens * world_size, args.top_k),
            dtype=torch.float32,
            device=device,
        )
        dist.all_gather_into_tensor(gathered_experts, local_topk_idx)
        dist.all_gather_into_tensor(gathered_weights, local_topk_weights)
        torch.cuda.synchronize(device)
        selected_experts = gathered_experts.cpu().numpy()
        combine_weights = gathered_weights.cpu().numpy()
        fixture_sha256 = None
    else:
        assert args.route_fixture is not None
        selected_experts, combine_weights, fixture_sha256 = _load_routes(
            args.route_fixture,
            global_tokens=args.local_tokens * world_size,
            top_k=args.top_k,
        )
        local_topk_idx = torch.as_tensor(
            selected_experts[source_start:source_end],
            dtype=torch.int64,
            device=device,
        )
        local_topk_weights = torch.as_tensor(
            combine_weights[source_start:source_end],
            dtype=torch.float32,
            device=device,
        )
    plan = _local_route_plan(
        selected_experts,
        combine_weights,
        owner_rank=rank,
        world_size=world_size,
        global_experts=args.global_experts,
    )

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
        local_topk_idx,
        args.global_experts,
    )
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=local_topk_idx,
        topk_weights=local_topk_weights,
        expert_alignment=1,
        config=deep_ep.Buffer.get_dispatch_config(world_size),
    )
    legalization = _legalize_transport_order(
        plan,
        combine_weights,
        rank=rank,
        local_tokens=args.local_tokens,
        rank_prefix_matrix=handle[0],
        recv_src_idx=handle[3],
    )
    mapping = _mapping_checks(
        legalization,
        plan,
        selected_experts,
        combine_weights,
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        x,
        rank=rank,
        local_tokens=args.local_tokens,
        global_experts=args.global_experts,
    )
    mapping["deep_ep_reported_expert_counts_exact"] = list(recv_counts) == plan.actual_counts.tolist()
    mapping["exact"] = bool(mapping["exact"] and mapping["deep_ep_reported_expert_counts_exact"])
    if not mapping["exact"]:
        raise AssertionError(f"rank {rank} transport legalization failed: {mapping}")

    module = _load_extension(args.probe_extension.resolve())
    selected_map_fold_program = shuttle_map_fold_program(
        natural_plan.map_fold_semantics if natural_plan is not None else None
    )
    map_fold_program_sha256 = _validate_generated_map_fold_extension(module, selected_map_fold_program)
    runtime = DistributedPhysicalRuntime(
        module,
        deep_ep,
        buffer,
        handle,
        x,
        plan,
        legalization,
        rank=rank,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seed=args.seed,
        gate_up_layout=args.gate_up_layout,
    )
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        runtime.generate_receiver_relation(recv_topk_idx, recv_topk_weights)
        torch.cuda.synchronize(device)
        dynamic_relation = {
            "overflow": int(runtime.dynamic_relation_overflow.sum().item()),
            "group_counts_exact": torch.equal(
                runtime.dynamic_group_counts,
                torch.as_tensor(plan.actual_counts, dtype=torch.int32, device=device),
            ),
            "padded_source_rows_exact": torch.equal(
                runtime.padded_deep_receiver_rows,
                torch.as_tensor(legalization.padded_deep_receiver_rows, dtype=torch.int64, device=device),
            ),
            "edge_destination_rows_exact": torch.equal(
                runtime.deep_route_padded_rows,
                torch.as_tensor(legalization.deep_route_padded_rows, dtype=torch.int64, device=device),
            ),
            "edge_weights_exact": torch.equal(
                runtime.deep_route_weights[recv_topk_idx >= 0],
                recv_topk_weights[recv_topk_idx >= 0],
            ),
        }
        dynamic_relation["exact"] = bool(
            dynamic_relation["overflow"] == 0
            and dynamic_relation["group_counts_exact"]
            and dynamic_relation["padded_source_rows_exact"]
            and dynamic_relation["edge_destination_rows_exact"]
            and dynamic_relation["edge_weights_exact"]
        )
        if not dynamic_relation["exact"]:
            raise AssertionError(f"rank {rank} generated device relation mismatch: {dynamic_relation}")
    else:
        dynamic_relation = None

    mok_oracle = None
    mok_oracle_post_route = None
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        from mok import functional as mok_functional  # noqa: PLC0415  # pyrefly: ignore[missing-import]

        mok_config = mok_functional.MoKConfig(
            fwd_num_comm_sms=20,
            minibatch_size=2_048,
            macrobatch_size=65_536,
        )
        mok_workspace = mok_functional.get_workspace(
            mok_config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=args.local_tokens,
            hidden_size=args.hidden_size,
            topk=args.top_k,
        )
        oracle_shared_gate = runtime.shared_gate_weights[0].contiguous()
        oracle_shared_up = runtime.shared_up_weights[0].contiguous()
        oracle_shared_down = runtime.shared_down_weights[0].contiguous()
        oracle_routed_gate = runtime.gate_weights.contiguous()
        oracle_routed_up = runtime.up_weights.contiguous()
        oracle_routed_down = runtime.down_weights.contiguous()

        def run_mok_post_route(
            expert_indices: torch.Tensor = local_topk_idx,
            route_weights: torch.Tensor = local_topk_weights,
        ) -> torch.Tensor:
            schedule = mok_functional.build_schedule(
                mok_workspace,
                mok_config,
                expert_indices,
                num_local_experts=args.global_experts // world_size,
            )
            return mok_functional.forward(
                mok_config,
                mok_workspace,
                schedule,
                x,
                route_weights,
                oracle_shared_gate,
                oracle_shared_up,
                oracle_shared_down,
                oracle_routed_gate,
                oracle_routed_up,
                oracle_routed_down,
            )[0]

        def run_mok_natural_end_to_end() -> torch.Tensor:
            assert router is not None
            expert_indices, route_weights = router()
            return run_mok_post_route(expert_indices, route_weights)

        mok_oracle_post_route = run_mok_post_route
        mok_oracle = run_mok_natural_end_to_end
    runtime.sequential()
    torch.cuda.synchronize(device)
    sequential_output = runtime.last_output.clone()
    runtime.overlap_shared_with_dispatch()
    torch.cuda.synchronize(device)
    overlap_output = runtime.last_output.clone()
    runtime.overlap_shared_with_dispatch()
    torch.cuda.synchronize(device)
    repeated_overlap_output = runtime.last_output.clone()
    runtime.generated_overlap_shared_with_dispatch()
    torch.cuda.synchronize(device)
    generated_overlap_output = runtime.last_output.clone()
    runtime.generated_overlap_shared_with_dispatch()
    torch.cuda.synchronize(device)
    repeated_generated_overlap_output = runtime.last_output.clone()
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        assert router is not None
        runtime.generated_natural_end_to_end(router)
        torch.cuda.synchronize(device)
        natural_generated_output = runtime.last_output.clone()
        runtime.generated_natural_end_to_end(router)
        torch.cuda.synchronize(device)
        repeated_natural_generated_output = runtime.last_output.clone()
        assert mok_oracle is not None
        natural_mok_output = mok_oracle()
        torch.cuda.synchronize(device)
    else:
        natural_generated_output = generated_overlap_output
        repeated_natural_generated_output = repeated_generated_overlap_output
        natural_mok_output = overlap_output
    runtime.coarse_materialized_sequential()
    torch.cuda.synchronize(device)
    coarse_output = runtime.last_output.clone()
    correctness = {
        "finite": bool(torch.isfinite(overlap_output).all().item()),
        "sequential_overlap_bitwise_equal": torch.equal(sequential_output, overlap_output),
        "overlap_repeat_bitwise_equal": torch.equal(overlap_output, repeated_overlap_output),
        "coarse_selected_bitwise_equal": torch.equal(coarse_output, overlap_output),
        "generated_repeat_bitwise_equal": torch.equal(
            generated_overlap_output,
            repeated_generated_overlap_output,
        ),
        "natural_generated_repeat_bitwise_equal": torch.equal(
            natural_generated_output,
            repeated_natural_generated_output,
        ),
        "natural_shuttle_mok_maximum_absolute_error": float(
            (natural_generated_output.float() - natural_mok_output.float()).abs().max().item()
        ),
        "natural_shuttle_mok_mean_absolute_error": float(
            (natural_generated_output.float() - natural_mok_output.float()).abs().mean().item()
        ),
        "generated_official_maximum_absolute_error": float(
            (generated_overlap_output.float() - overlap_output.float()).abs().max().item()
        ),
        "maximum_absolute_error": float((sequential_output.float() - overlap_output.float()).abs().max().item()),
        "mean_absolute_error": float((sequential_output.float() - overlap_output.float()).abs().mean().item()),
        "sequential_output_sha256": _tensor_sha256(sequential_output),
        "overlap_output_sha256": _tensor_sha256(overlap_output),
        "repeated_overlap_output_sha256": _tensor_sha256(repeated_overlap_output),
        "coarse_output_sha256": _tensor_sha256(coarse_output),
        "generated_overlap_output_sha256": _tensor_sha256(generated_overlap_output),
        "repeated_generated_overlap_output_sha256": _tensor_sha256(repeated_generated_overlap_output),
        "natural_generated_output_sha256": _tensor_sha256(natural_generated_output),
        "repeated_natural_generated_output_sha256": _tensor_sha256(repeated_natural_generated_output),
        "natural_mok_output_sha256": _tensor_sha256(natural_mok_output),
    }
    if not all(
        (
            correctness["finite"],
            correctness["sequential_overlap_bitwise_equal"],
            correctness["overlap_repeat_bitwise_equal"],
            correctness["coarse_selected_bitwise_equal"],
            correctness["generated_repeat_bitwise_equal"],
            correctness["natural_generated_repeat_bitwise_equal"],
        )
    ):
        raise AssertionError(f"rank {rank} distributed correctness failed: {correctness}")

    runtime.routed(recv_x)
    runtime.shared()
    phases = {
        "official_dispatch_combine_identity": runtime.identity_transport,
        "compiler_routed_already_dispatched": lambda: runtime.routed(recv_x),
        "generated_shared_expert": runtime.shared,
        "official_combine_with_shared_bias": runtime.combine,
        "payload_reverse_transport_and_generated_merge": runtime.reverse_transport_and_generated_merge,
        "full_sequential": runtime.sequential,
        "full_overlap_shared_with_async_dispatch": runtime.overlap_shared_with_dispatch,
        "full_generated_sequential": runtime.generated_sequential,
        "full_generated_overlap_shared_with_async_dispatch": runtime.generated_overlap_shared_with_dispatch,
        "full_coarse_materialized_sequential": runtime.coarse_materialized_sequential,
    }
    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        assert router is not None and mok_oracle is not None and mok_oracle_post_route is not None
        phases.update(
            {
                "natural_router_contract_topk_fold": router,
                "natural_runtime_layout_and_payload_dispatch": lambda: runtime.dispatch_natural_routes(
                    local_topk_idx,
                    local_topk_weights,
                    asynchronous=False,
                ),
                "generated_receiver_relation_plan": lambda: runtime.generate_receiver_relation(
                    recv_topk_idx,
                    recv_topk_weights,
                ),
                "natural_shuttle_post_route": lambda: runtime.generated_natural_post_route(
                    local_topk_idx,
                    local_topk_weights,
                ),
            }
        )
        if args.acceptance_order is AcceptanceOrder.SHUTTLE_FIRST:
            phases["natural_end_to_end_shuttle"] = lambda: runtime.generated_natural_end_to_end(router)
            phases["natural_end_to_end_mok_oracle"] = mok_oracle
        else:
            phases["natural_end_to_end_mok_oracle"] = mok_oracle
            phases["natural_end_to_end_shuttle"] = lambda: runtime.generated_natural_end_to_end(router)
        phases["mok_oracle_post_route"] = mok_oracle_post_route
    timing: dict[str, dict[str, Any]] = {}
    phase_telemetry: dict[str, Any] = {}
    for name, function in phases.items():
        telemetry_before = nvidia_smi_snapshot() if rank == 0 else None
        samples, per_rank_samples, summary = _rank_max_measure(
            function,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        telemetry_after = nvidia_smi_snapshot() if rank == 0 else None
        timing[name] = {
            "rank_max_samples_ms": samples,
            "per_rank_samples_ms": per_rank_samples,
            **asdict(summary),
        }
        if rank == 0:
            phase_telemetry[name] = {"before": telemetry_before, "after": telemetry_after}

    semantic_reference = _small_semantic_reference(
        module,
        deep_ep,
        buffer,
        rank=rank,
        device=device,
        seed=args.seed,
        gate_up_layout=args.gate_up_layout,
        fixture_output=args.semantic_fixture_output,
    )
    if not semantic_reference["passed"]:
        raise AssertionError(f"rank {rank} independent semantic reference failed: {semantic_reference}")
    local_record = {
        "rank": rank,
        "mapping": mapping,
        "correctness": correctness,
        "independent_semantic_reference": semantic_reference,
        "received_tokens": plan.received_tokens,
        "assignments": plan.assignments,
        "padded_rows": plan.padded_rows,
        "generated_device_relation": dynamic_relation,
    }
    records: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(records, local_record)
    if rank == 0:
        candidate = {
            "exchange_implementation": "deepep",
            "forward_transport": "deepep_dispatch_payload_permutation",
            "reverse_transport": "torch_distributed_all_to_all_single_payload_permutation",
            "semantic_merge": "generated_fixed_rank_fold",
            "segmented_contraction_implementation": "standalone_sm100_grouped_gemm",
            "exchange_workers": args.deepep_sms,
            "gate_up_layout": args.gate_up_layout.value,
            "overlap_policy": "shared_expert_with_async_dispatch",
            "materialization_policy": "tile_flow_boundaries",
        }
        source = _validate_checkout(args.mok_root.resolve())
        extension = args.probe_extension.resolve()
        matched_boundary = None
        if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
            shuttle_end_to_end = timing["natural_end_to_end_shuttle"]["median_ms"]
            oracle_end_to_end = timing["natural_end_to_end_mok_oracle"]["median_ms"]
            shuttle_post_route = timing["natural_shuttle_post_route"]["median_ms"]
            oracle_post_route = timing["mok_oracle_post_route"]["median_ms"]
            matched_boundary = {
                "acceptance_metric": "median rank-maximum natural-program end-to-end latency",
                "shuttle_included": [
                    "BF16 router Contract",
                    "top-k selection",
                    "FP32 normalized route-weight Fold/Maps",
                    "DeepEP payload-only dispatch and runtime layout",
                    "generated fixed-capacity receiver RelationPlan",
                    "generic grouped W13",
                    "generated SwiGLU Map",
                    "generic grouped W2",
                    "payload-only reverse all_to_all_single",
                    "generated deterministic rank-ordered Fold merge and shared Map",
                ],
                "oracle_included": [
                    "identical BF16 router Contract",
                    "identical top-k selection",
                    "identical FP32 normalized route-weight Fold/Maps",
                    "MoK schedule construction",
                    "complete MoK BF16 forward",
                ],
                "common_excluded": [
                    "host input creation",
                    "weight creation",
                    "StableHLO import and compile time",
                    "workspace allocation",
                    "correctness hashing",
                ],
                "shuttle_median_ms": shuttle_end_to_end,
                "oracle_median_ms": oracle_end_to_end,
                "ratio": shuttle_end_to_end / oracle_end_to_end,
                "passes_1_2x": shuttle_end_to_end <= 1.2 * oracle_end_to_end,
                "launch_order": args.acceptance_order.value,
                "post_route": {
                    "shuttle_median_ms": shuttle_post_route,
                    "oracle_median_ms": oracle_post_route,
                    "ratio": shuttle_post_route / oracle_post_route,
                },
                "route_fixture_used": False,
            }
        result = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": SHUTTLE_BENCHMARK,
            "status": "ok",
            "source": {
                "shuttle_revision": args.shuttle_revision,
                "deepep_commit": DEEPEP_COMMIT,
                **source,
                "probe_extension": str(extension),
                "probe_extension_sha256": file_sha256(extension),
                "probe_source_sha256": file_sha256(
                    Path(__file__).resolve().parents[2] / "backends" / "sm100" / "mok_gmm_probe" / "mok_gmm_probe.cu"
                ),
                "generated_map_fold_program_sha256": map_fold_program_sha256,
                "generated_map_fold_include_sha256": file_sha256(
                    Path(__file__).resolve().parents[2]
                    / "backends"
                    / "sm100"
                    / "mok_gmm_probe"
                    / "generated_map_fold.inc"
                ),
                "generated_map_fold_generator_sha256": file_sha256(
                    Path(__file__).resolve().parents[2] / "src" / "tile_lifetime" / "cuda_map_fold_codegen.py"
                ),
            },
            "routing_frontend": {
                "source": args.routing_source.value,
                "route_fixture_path": str(args.route_fixture) if args.route_fixture is not None else None,
                "route_fixture_sha256": fixture_sha256,
                "natural_stablehlo": natural_frontend,
                "router_weight_sha256": _tensor_sha256(router.weight) if router is not None else None,
                "selected_experts_sha256": canonical_json_sha256(selected_experts.tolist()),
                "route_weights_sha256": framed_tensor_sha256(
                    str(combine_weights.dtype),
                    tuple(combine_weights.shape),
                    combine_weights.tobytes(order="C"),
                ),
            },
            "shape": {
                "world_size": world_size,
                "local_tokens": args.local_tokens,
                "global_experts": args.global_experts,
                "local_experts": args.global_experts // world_size,
                "top_k": args.top_k,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
            },
            "schedule": {
                "merge": "fixed route-slot order, explicit FP32 RN multiply then RN add, no atomics",
                "cross_rank_combine": (
                    "not used by accepted natural Shuttle phase; legacy control only"
                    if args.routing_source is RoutingSource.NATURAL_STABLEHLO
                    else "official DeepEP fixed owner-rank order"
                ),
                "clean_cross_rank_return": "all_to_all_single payload permutation; no reduction in transport",
                "clean_semantic_merge": "generated fixed owner-rank FP32 fold plus shared add; no atomics",
                "shared_add": (
                    "generated Map after the source-ordered Fold"
                    if args.routing_source is RoutingSource.NATURAL_STABLEHLO
                    else "DeepEP combine bias"
                ),
                "overlap": "generated shared expert on current stream while async DeepEP dispatch runs",
                "deepep_sms": args.deepep_sms,
                "padding_quantum": TILE_ROWS,
                "gate_up_layout": args.gate_up_layout.value,
                "coarse_materialization_ablation": (
                    "explicit copies after dispatch packing, W13, SwiGLU, and W2 on routed and shared paths"
                ),
            },
            "candidate": {**candidate, "fingerprint_sha256": canonical_json_sha256(candidate)},
            "generated_code_audit": {
                "accepted_path_external_semantic_kernels": [],
                "generic_compute_primitives": ["standalone_sm100_grouped_gemm"],
                "generic_communication_primitives": [
                    "DeepEP payload dispatch/permutation",
                    "torch.distributed.all_to_all_single payload return",
                ],
                "generated_shuttle_kernels": [
                    "fixed-capacity runtime RelationPlan skeleton",
                    "generic adjacent-pair Map skeleton with generated ScalarExpression",
                    "generic indexed ordered-Fold skeleton with generated contribution/update expressions",
                    "generic partitioned ordered-Fold plus base-Map skeleton with generated expressions",
                ],
                "semantic_body_generation": {
                    "selected_program_sha256": map_fold_program_sha256,
                    "compile_guard": "generated CUDA include must exactly match selected Map/Fold IR before build",
                    "runtime_guard": "loaded extension exports and matches selected Map/Fold IR digest",
                },
                "atomic_audit": {
                    "generated_relation_plan": (
                        "none; stable warp-prefix placement and one deterministic overflow value per group"
                    ),
                    "generated_route_slot_fold": "none",
                    "generated_rank_fold": "none",
                    "semantic_accumulation": "none",
                    "grouped_gemm_control": (
                        "backend may use queue/semaphore primitives; each semantic output tile has unique ownership"
                    ),
                    "transport_control": (
                        "DeepEP may use internal readiness/counter atomics; these do not choose semantic order or "
                        "accumulate semantic values"
                    ),
                },
                "expert_oracle_only": ["complete MoK BF16 forward"],
                "legacy_control_only": ["DeepEP semantic combine"],
            },
            "rank_records": records,
            "timing": timing,
            "matched_boundary": matched_boundary,
            "environment": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "gpu": torch.cuda.get_device_properties(device).name,
                "clock_policy": args.clock_policy,
                "command": command_record(),
                "toolchain": toolchain_snapshot(os.environ.get("MOK_NVCC", "nvcc")),
                "gpu_telemetry": {
                    "initial": nvidia_smi_snapshot(),
                    "by_phase": phase_telemetry,
                    "final": nvidia_smi_snapshot(),
                },
            },
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")

    if args.routing_source is RoutingSource.NATURAL_STABLEHLO:
        from mok import functional as mok_functional  # noqa: PLC0415  # pyrefly: ignore[missing-import]

        mok_functional.clear_workspace_cache()
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
