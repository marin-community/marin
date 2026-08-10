# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime import (
    ExchangeRowMode,
    ExpertMaterializationSchedule,
    ExpertOverlapPolicy,
    ExpertParallelConfig,
    ExpertParallelStageKind,
    GateUpPhysicalLayout,
    NumericalPolicy,
    ReadinessGranularity,
    TileStorage,
    TransportSelection,
    TransportSemantics,
    build_expert_parallel_relation_plan,
)
from tile_lifetime.expert_parallel import ExpertParallelLegalityError, compile_expert_parallel_region
from tile_lifetime.ir import TensorGraph
from tile_lifetime.tile_program import TilePrimitive


def _global_moe_region(
    *,
    global_experts: int = 384,
    routed_weight_experts: int = 384,
) -> TensorGraph:
    tokens, hidden, intermediate, top_k = 2048, 7168, 3072, 6
    graph = TensorGraph()
    x = graph.input("x", shape=(tokens, hidden), dtype=DType.BF16)
    router_weight = graph.parameter("router_weight", shape=(hidden, global_experts), dtype=DType.BF16)
    shared_gate = graph.parameter("shared_gate", shape=(intermediate, hidden), dtype=DType.BF16)
    shared_up = graph.parameter("shared_up", shape=(intermediate, hidden), dtype=DType.BF16)
    shared_down = graph.parameter("shared_down", shape=(hidden, intermediate), dtype=DType.BF16)
    routed_gate = graph.parameter("routed_gate", shape=(routed_weight_experts, intermediate, hidden), dtype=DType.BF16)
    routed_up = graph.parameter("routed_up", shape=(routed_weight_experts, intermediate, hidden), dtype=DType.BF16)
    routed_down = graph.parameter("routed_down", shape=(routed_weight_experts, hidden, intermediate), dtype=DType.BF16)
    logits = graph.linear(x, router_weight, name="router_logits", accumulation_dtype=DType.FP32)
    expert_indices, route_weights = graph.top_k_router(logits, name="routes", top_k=top_k)
    shared_output = graph.shared_expert_mlp(
        x,
        shared_gate,
        shared_up,
        shared_down,
        name="shared_output",
        accumulation_dtype=DType.FP32,
    )
    routed_output = graph.routed_expert_mlp(
        x,
        expert_indices,
        routed_gate,
        routed_up,
        routed_down,
        name="routed_output",
        accumulation_dtype=DType.FP32,
    )
    graph.weighted_expert_combine(shared_output, routed_output, route_weights, name="moe_output")
    return graph


def test_generic_ep_lowering_exposes_atomic_global_to_local_dataflow() -> None:
    plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert plan.ownership.global_expert_count == 384
    assert plan.ownership.local_expert_count == 96
    assert plan.ownership.owner(0) == (0, 0)
    assert plan.ownership.owner(95) == (0, 95)
    assert plan.ownership.owner(96) == (1, 0)
    assert plan.ownership.owner(383) == (3, 95)
    assert plan.route_relation.slots_per_token == 6
    assert plan.segments.keys == ("local_expert",)
    assert plan.segments.padding_quantum == 256
    assert plan.capacity.receiver_assignment_capacity == 15_360
    assert plan.capacity.padded_local_capacity == 39_840
    assert "exact fallback" in plan.capacity.overflow_policy

    assert [stage.kind for stage in plan.stages] == [
        ExpertParallelStageKind.ROUTE_RELATION,
        ExpertParallelStageKind.EXPERT_OWNERSHIP,
        ExpertParallelStageKind.LEGALIZE_GATE_UP_LAYOUT,
        ExpertParallelStageKind.GROUP_BY_OWNER,
        ExpertParallelStageKind.PROJECT_EXCHANGE_ROWS,
        ExpertParallelStageKind.FORWARD_EXCHANGE,
        ExpertParallelStageKind.EXPAND_LOCAL_ASSIGNMENTS,
        ExpertParallelStageKind.SEGMENT_BY_LOCAL_EXPERT,
        ExpertParallelStageKind.PAD_LOCAL_SEGMENTS,
        ExpertParallelStageKind.SHARED_DENSE_GATE_UP,
        ExpertParallelStageKind.ROUTED_SEGMENTED_GATE_UP,
        ExpertParallelStageKind.SHARED_PAIRWISE_SWIGLU,
        ExpertParallelStageKind.ROUTED_PAIRWISE_SWIGLU,
        ExpertParallelStageKind.SHARED_DENSE_DOWN,
        ExpertParallelStageKind.ROUTED_SEGMENTED_DOWN,
        ExpertParallelStageKind.REVERSE_EXCHANGE,
        ExpertParallelStageKind.WEIGHTED_SCATTER_REDUCE,
        ExpertParallelStageKind.SHARED_ADD,
    ]
    scatter = plan.stage(ExpertParallelStageKind.WEIGHTED_SCATTER_REDUCE)
    shared_add = plan.stage(ExpertParallelStageKind.SHARED_ADD)
    assert scatter.outputs == ("moe_output.routed_scatter_output",)
    assert shared_add.inputs == ("moe_output.routed_scatter_output", "shared_output")
    exchange_workers = next(pool for pool in plan.schedule.worker_pools if pool.name == "exchange")
    assert exchange_workers.workers == 56
    assert plan.schedule.exchange_implementation == "deepep"
    assert plan.schedule.exchange_implementation_candidates == ("deepep", "ragged_all_to_all")
    assert plan.schedule.forward_transport == TransportSelection(
        "deepep_dispatch",
        TransportSemantics.PAYLOAD_PERMUTATION,
    )
    assert plan.schedule.reverse_transport == TransportSelection(
        "all_to_all_single",
        TransportSemantics.PAYLOAD_PERMUTATION,
    )
    assert plan.schedule.merge_implementation == "generated_source_ordered_fold"
    assert tuple(operation.primitive for operation in plan.merge_program.operations) == (
        TilePrimitive.LOAD_STATE,
        TilePrimitive.LOAD_TILE,
        TilePrimitive.ADD,
        TilePrimitive.LOAD_TILE,
        TilePrimitive.ADD,
        TilePrimitive.LOAD_TILE,
        TilePrimitive.ADD,
        TilePrimitive.LOAD_TILE,
        TilePrimitive.ADD,
        TilePrimitive.CONVERT,
        TilePrimitive.STORE,
    )
    assert plan.schedule.segmented_contraction_implementation == "standalone_sm100_grouped_gemm"
    assert plan.schedule.segmented_contraction_candidates == ("standalone_sm100_grouped_gemm", "ragged_dot")
    assert plan.schedule.exchange_worker_candidates == (12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96)
    assert plan.schedule.overlap_policy is ExpertOverlapPolicy.SHARED_WITH_ASYNC_DISPATCH
    assert plan.schedule.overlap_policy_candidates == (
        ExpertOverlapPolicy.SHARED_WITH_ASYNC_DISPATCH,
        ExpertOverlapPolicy.SEQUENTIAL,
    )
    assert plan.schedule.materialization_schedule is ExpertMaterializationSchedule.TILE_FLOW_BOUNDARIES
    assert plan.schedule.materialization_schedule_candidates == (
        ExpertMaterializationSchedule.TILE_FLOW_BOUNDARIES,
        ExpertMaterializationSchedule.COARSE_ACTIVATION_BOUNDARIES,
    )


def test_generic_ep_lowering_exposes_exchange_and_gate_up_layout_alternatives() -> None:
    plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert {projection.mode for projection in plan.exchange_projections} == {
        ExchangeRowMode.ASSIGNMENT,
        ExchangeRowMode.COALESCED_TOKEN_OWNER,
    }
    assert plan.selected_exchange_projection.mode is ExchangeRowMode.COALESCED_TOKEN_OWNER
    projection = plan.stage(ExpertParallelStageKind.PROJECT_EXCHANGE_ROWS)
    assert "source_token" in projection.operation
    assert plan.gate_up_layout.selected is GateUpPhysicalLayout.CONCATENATED_E_2I_K
    assert "[local_expert, 2*intermediate, hidden]" in plan.gate_up_layout.legalization

    flows = {flow.value: flow for flow in plan.tile_flows}
    sent_x = flows["moe_output.exchange_x_rows"]
    assert sent_x.shape == (2048 * 4, 7168)
    assert sent_x.logical_layout == "coalesced_token_owner_send_rows"
    assert sent_x.tile_shape == (128, 512)
    assert sent_x.readiness is ReadinessGranularity.MINIBATCH
    packed_weight = flows["moe_output.local_gate_up_weight"]
    assert packed_weight.shape == (96, 6144, 7168)
    assert packed_weight.logical_layout == "concatenated_e_2i_k"

    separate_plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(
            expert_parallel_size=4,
            gate_up_layout=GateUpPhysicalLayout.SEPARATE_E_I_K,
        ),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    separate_flows = {flow.value: flow for flow in separate_plan.tile_flows}
    assert separate_flows["moe_output.local_gate_up_weight"].shape == (96, 2, 3072, 7168)
    assert separate_flows["moe_output.routed_gate_up"].shape[1:] == (2, 3072)


def test_generic_ep_buffer_lifetimes_are_derived_from_tile_flow_edges() -> None:
    plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    flows = {flow.value: flow for flow in plan.tile_flows}
    buffers = {buffer.value: buffer for buffer in plan.buffers}
    materializations = {record.value: record for record in plan.materializations}
    assert buffers.keys() == flows.keys() == materializations.keys()
    for value, flow in flows.items():
        buffer = buffers[value]
        assert buffer.shape == flow.shape
        assert buffer.logical_layout == flow.logical_layout
        assert buffer.tile_shape == flow.tile_shape
        assert buffer.storage is flow.storage
        assert flow.fanout == len(flow.consumers)
    assert buffers["moe_output.route_weights"].live_until == "weighted_scatter_reduce"
    assert buffers["moe_output.local_gate_up_weight"].storage is TileStorage.GLOBAL_BUFFER
    assert buffers["moe_output"].live_until == "region_output"


def test_generic_ep_assignment_row_exchange_is_a_distinct_legal_plan() -> None:
    plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(expert_parallel_size=4, exchange_row_mode=ExchangeRowMode.ASSIGNMENT),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    sent_x = next(flow for flow in plan.tile_flows if flow.value == "moe_output.exchange_x_rows")
    assignments = 2048 * 6
    assert sent_x.shape == (assignments, 7168)
    assert sent_x.logical_layout == "assignment_send_rows"
    assert plan.selected_exchange_projection.mode is ExchangeRowMode.ASSIGNMENT
    assert max(flow.shape[0] for flow in plan.tile_flows if len(flow.shape) >= 2) == 39_840


def test_generic_ep_lowering_rejects_local_weight_semantics_and_invalid_schedule() -> None:
    with pytest.raises(ExpertParallelLegalityError) as exc_info:
        compile_expert_parallel_region(
            _global_moe_region(routed_weight_experts=96),
            config=ExpertParallelConfig(
                expert_parallel_size=4,
                segment_padding=0,
                exchange_workers=0,
                exchange_worker_candidates=(12, 56),
                exchange_implementation="unknown_exchange",
                reverse_transport=TransportSelection(
                    "deepep_combine",
                    TransportSemantics.PAYLOAD_PERMUTATION_AND_REDUCTION,
                ),
                segmented_contraction_implementation="unknown_contraction",
                overlap_policy_candidates=(ExpertOverlapPolicy.SEQUENTIAL,),
                materialization_schedule_candidates=(ExpertMaterializationSchedule.COARSE_ACTIVATION_BOUNDARIES,),
            ),
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
        )

    reasons = exc_info.value.reasons
    assert any("global expert axis (384), got 96" in reason for reason in reasons)
    assert any("segment padding must be positive" in reason for reason in reasons)
    assert any("worker pool" in reason for reason in reasons)
    assert any("selected exchange implementation" in reason for reason in reasons)
    assert any("reverse transport must only permute payload" in reason for reason in reasons)
    assert any("selected segmented contraction" in reason for reason in reasons)
    assert any("selected exchange worker count" in reason for reason in reasons)
    assert any("selected overlap policy" in reason for reason in reasons)
    assert any("selected materialization schedule" in reason for reason in reasons)
    assert any("reorder floating-point" in reason for reason in reasons)


def test_generic_ep_capacity_drives_seeded_global_route_relation() -> None:
    plan = compile_expert_parallel_region(
        _global_moe_region(),
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    rng = np.random.default_rng(20260806)
    destination_indices = rng.integers(0, 384, size=(2048, 6), dtype=np.int32)
    weights = rng.random((2048, 6), dtype=np.float32)
    weights /= np.sum(weights, axis=1, keepdims=True)

    relation = build_expert_parallel_relation_plan(plan, destination_indices, weights)

    rank_route_counts = np.bincount(relation.destination_rank, minlength=4)
    rank_padded_counts = np.bincount(relation.row_destination_rank, minlength=4)
    assert np.all(rank_route_counts <= plan.capacity.receiver_assignment_capacity)
    assert np.all(rank_padded_counts <= plan.capacity.padded_local_capacity)
    assert relation.route_count == 2048 * 6
    assert relation.merge_order.endswith("FP32 accumulation")
