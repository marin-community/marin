# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime import (
    DType,
    NumericalPolicy,
    TensorGraph,
)
from tile_lifetime.moe import MoELegalityError, MoERoutedPrecision, MoKOracleConfig, compile_mok_oracle_region
from tile_lifetime.plan import OpaqueMoKOracleSkeleton, PersistentTaskPlacement


def _moe_region(
    *,
    tokens: int = 2048,
    hidden: int = 7168,
    intermediate: int = 3072,
    global_experts: int = 384,
    local_experts: int = 96,
    top_k: int = 6,
    shared_up_dtype: DType = DType.BF16,
    routed_down_dtype: DType = DType.BF16,
) -> TensorGraph:
    graph = TensorGraph()
    x = graph.input("x", shape=(tokens, hidden), dtype=DType.BF16)
    router_weight = graph.parameter("router_weight", shape=(hidden, global_experts), dtype=DType.BF16)
    shared_gate_weight = graph.parameter("shared_gate_weight", shape=(intermediate, hidden), dtype=DType.BF16)
    shared_up_weight = graph.parameter("shared_up_weight", shape=(intermediate, hidden), dtype=shared_up_dtype)
    shared_down_weight = graph.parameter("shared_down_weight", shape=(hidden, intermediate), dtype=DType.BF16)
    routed_gate_weight = graph.parameter(
        "routed_gate_weight", shape=(local_experts, intermediate, hidden), dtype=DType.BF16
    )
    routed_up_weight = graph.parameter("routed_up_weight", shape=(local_experts, intermediate, hidden), dtype=DType.BF16)
    routed_down_weight = graph.parameter(
        "routed_down_weight", shape=(local_experts, hidden, intermediate), dtype=routed_down_dtype
    )

    router_logits = graph.linear(x, router_weight, name="router_logits", accumulation_dtype=DType.FP32)
    expert_indices, router_weights = graph.top_k_router(router_logits, name="routes", top_k=top_k)
    shared = graph.shared_expert_mlp(
        x,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        name="shared_expert",
        accumulation_dtype=DType.FP32,
    )
    routed = graph.routed_expert_mlp(
        x,
        expert_indices,
        routed_gate_weight,
        routed_up_weight,
        routed_down_weight,
        name="routed_experts",
        accumulation_dtype=DType.FP32,
    )
    graph.weighted_expert_combine(shared, routed, router_weights, name="moe_output")
    return graph


def test_compile_mok_oracle_region_recovers_explicit_gb200_oracle_structure() -> None:
    plan = compile_mok_oracle_region(
        _moe_region(),
        config=MoKOracleConfig(
            expert_parallel_size=4,
            communication_sm_count=20,
            minibatch_size=2048,
            macrobatch_size=65_536,
            routed_precision=MoERoutedPrecision.BF16,
        ),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert len(plan.skeletons) == 1
    skeleton = plan.skeletons[0]
    assert isinstance(skeleton, OpaqueMoKOracleSkeleton)
    assert (skeleton.global_experts, skeleton.local_experts, skeleton.shared_experts) == (384, 96, 1)
    assert skeleton.top_k == 6
    assert skeleton.routed_precision == "bf16"
    assert (skeleton.local_token_count, skeleton.hidden_size, skeleton.intermediate_size) == (2048, 7168, 3072)
    assert skeleton.schedule.expert_padding == 256
    assert skeleton.schedule.capacity_factor == 2
    assert skeleton.schedule.capacity == 2048 * 6 * 2
    assert (skeleton.communication_sm_count, skeleton.minibatch_size, skeleton.macrobatch_size) == (
        20,
        2048,
        65_536,
    )
    assert skeleton.grouped_gemm_tile == (256, 256, 64)
    assert skeleton.swiglu_tile == (128, 128)
    assert skeleton.swiglu_operation == "pairwise_swiglu"
    assert skeleton.dispatch_tile == (128, 512)
    assert skeleton.combine_tile == (16, 1024)
    assert skeleton.cluster_size == 2
    assert skeleton.threads_per_cluster_block == 256
    assert skeleton.backend_revision == "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"

    tasks = {task.name: task for task in skeleton.task_roles}
    assert tuple(tasks) == (
        "dispatch",
        "shared_gate_gemm",
        "shared_up_gemm",
        "shared_swiglu",
        "shared_down_gemm",
        "routed_gate_gemm",
        "routed_up_gemm",
        "routed_swiglu",
        "routed_down_gemm",
        "combine",
    )
    assert tasks["shared_swiglu"].placement is PersistentTaskPlacement.CTA_LOCAL
    assert tasks["routed_swiglu"].placement is PersistentTaskPlacement.CTA_LOCAL
    assert tasks["dispatch"].placement is PersistentTaskPlacement.COMMUNICATION_SM
    assert tasks["routed_gate_gemm"].waits_for == ("moe_output.x_routed_ready",)
    assert tasks["combine"].waits_for == ("moe_output.y_routed_ready",)
    assert tasks["shared_swiglu"].inputs == ("shared_expert.gate", "shared_expert.up")
    assert tasks["routed_swiglu"].inputs == ("routed_experts.gate", "routed_experts.up")

    events = {event.name for event in skeleton.readiness_events}
    assert events == {
        "moe_output.x_routed_ready",
        "moe_output.gate_up_tile_ready",
        "moe_output.hidden_row_block_ready",
        "moe_output.y_routed_ready",
        "moe_output.y_routed_done",
    }
    assert [(worker.name, worker.count) for worker in skeleton.worker_roles] == [
        ("communication_cta", 20),
        ("gemm_consumer_warpgroup", 1),
        ("gemm_producer_warpgroup", 1),
    ]
    assert skeleton.dispatch_send_buffer != skeleton.routed_input_buffer
    assert skeleton.routed_output_buffer != skeleton.combine_receive_buffer
    assert plan.rewrites[0].applied


def test_compile_mok_oracle_region_rejects_mxfp8_without_scale_tensor_semantics() -> None:
    with pytest.raises(MoELegalityError) as exc_info:
        compile_mok_oracle_region(
            _moe_region(
                tokens=512,
                hidden=256,
                intermediate=256,
                global_experts=16,
                local_experts=4,
                top_k=2,
            ),
            config=MoKOracleConfig(expert_parallel_size=4, routed_precision=MoERoutedPrecision.MXFP8),
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )

    assert exc_info.value.reasons == ("MXFP8 scale tensor semantics are not modeled by the first MoK compiler slice",)


def test_compile_mok_oracle_region_validates_every_shared_and_routed_weight_dtype() -> None:
    with pytest.raises(MoELegalityError) as exc_info:
        compile_mok_oracle_region(
            _moe_region(
                tokens=512,
                hidden=256,
                intermediate=256,
                global_experts=16,
                local_experts=4,
                top_k=2,
                shared_up_dtype=DType.FP32,
                routed_down_dtype=DType.FP32,
            ),
            config=MoKOracleConfig(expert_parallel_size=4),
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )

    assert exc_info.value.reasons == (
        "shared gate, up, and down weights must all have BF16 dtype",
        "BF16 routed precision requires routed gate, up, and down weights to all have BF16 dtype",
    )


def test_compile_mok_oracle_region_reports_all_physical_legality_failures() -> None:
    with pytest.raises(MoELegalityError) as exc_info:
        compile_mok_oracle_region(
            _moe_region(tokens=768, global_experts=384, local_experts=96),
            config=MoKOracleConfig(
                expert_parallel_size=8,
                communication_sm_count=23,
                minibatch_size=1000,
                macrobatch_size=1500,
            ),
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
        )

    reasons = exc_info.value.reasons
    assert any("finite-precision" in reason for reason in reasons)
    assert any("require 768" in reason for reason in reasons)
    assert any("communication SM count" in reason for reason in reasons)
    assert any("minibatch size" in reason for reason in reasons)
    assert any("macrobatch size" in reason for reason in reasons)


def test_compile_mok_oracle_region_rejects_unmodeled_consumer() -> None:
    graph = _moe_region(tokens=512, hidden=256, intermediate=256, global_experts=16, local_experts=4, top_k=2)
    output = graph.values[-1]
    graph.view(output, shape=output.shape, name="observed_output")

    with pytest.raises(MoELegalityError) as exc_info:
        compile_mok_oracle_region(
            graph,
            config=MoKOracleConfig(expert_parallel_size=4),
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )

    assert any("additional semantic operations" in reason for reason in exc_info.value.reasons)
