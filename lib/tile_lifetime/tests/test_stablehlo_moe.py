# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime import DType, MoESemanticRecoveryError, recover_stablehlo_moe_region
from tile_lifetime.ir import (
    LinearOp,
    RoutedExpertMLPOp,
    SharedExpertMLPOp,
    TopKRouterOp,
    WeightedExpertCombineOp,
)
from tile_lifetime.moe_recovery import recover_moe_region
from tile_lifetime.moe_reference import (
    MOE_REGION_INPUT_NAMES,
    MOE_UNIMPORTED_PRIVATE_OPERATIONS,
    MoEDebugConfig,
    export_debug_moe_region,
)
from tile_lifetime.stablehlo_import import (
    CompositeAttributes,
    GatherAttributes,
    import_stablehlo,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "moe_region_v1_14_1.mlir.bc.b64"


def _fixture_artifact() -> bytes:
    return base64.b64decode(FIXTURE.read_text())


def test_moe_export_uses_versioned_top_k_and_static_expert_gathers() -> None:
    graph = import_stablehlo(_fixture_artifact(), input_names=MOE_REGION_INPUT_NAMES)

    composites = tuple(operation for operation in graph.operations if operation.kind == "composite")
    gathers = tuple(operation for operation in graph.operations if operation.kind == "gather")
    assert len(graph.operations) == 70
    assert len(composites) == 1
    assert composites[0].attributes == CompositeAttributes(
        name="chlo.top_k",
        attributes=(("k", "2 : i64"),),
        version=1,
    )
    assert len(gathers) == 3
    assert all(isinstance(operation.attributes, GatherAttributes) for operation in gathers)
    assert not any(operation.kind == "sort" for operation in graph.operations)
    assert MOE_UNIMPORTED_PRIVATE_OPERATIONS == ("stablehlo.sort",)
    assert all("moe_reference.py" in operation.source_location for operation in (*composites, *gathers))


def test_public_stablehlo_path_recovers_global_semantic_moe_region() -> None:
    recovered = recover_stablehlo_moe_region(
        _fixture_artifact(),
        input_names=MOE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
    )

    assert [type(operation) for operation in recovered.graph.operations] == [
        LinearOp,
        TopKRouterOp,
        SharedExpertMLPOp,
        RoutedExpertMLPOp,
        WeightedExpertCombineOp,
    ]
    router = recovered.graph.operations[1]
    routed = recovered.graph.operations[3]
    assert isinstance(router, TopKRouterOp)
    assert isinstance(routed, RoutedExpertMLPOp)
    assert router.top_k == 2
    assert router.normalize_weights
    assert router.logits.shape == (8, 4)
    assert routed.gate_weight.shape == (4, 32, 16)
    assert routed.gate_weight.shape[0] == router.logits.shape[1]
    assert routed.expert_indices == router.expert_indices
    assert recovered.source_operation_ids == tuple(range(70))
    assert all(operation.source_location is not None for operation in recovered.graph.operations)


def test_parameterized_moe_export_keeps_global_weights_as_inputs() -> None:
    config = MoEDebugConfig(tokens=6, hidden=8, intermediate=12, experts=3, top_k=1)
    graph = import_stablehlo(export_debug_moe_region(config), input_names=MOE_REGION_INPUT_NAMES)

    assert [graph.value(value_id).shape for value_id in graph.inputs] == [
        (6, 8),
        (8, 3),
        (12, 8),
        (12, 8),
        (8, 12),
        (3, 12, 8),
        (3, 12, 8),
        (3, 8, 12),
    ]
    assert [graph.value(value_id).shape for value_id in graph.outputs] == [(6, 8), (6, 1), (6, 1)]


def test_moe_recovery_reports_top_k_composite_mismatch_by_stage() -> None:
    graph = import_stablehlo(_fixture_artifact(), input_names=MOE_REGION_INPUT_NAMES)
    top_k = next(operation for operation in graph.operations if operation.kind == "composite")
    assert isinstance(top_k.attributes, CompositeAttributes)
    mutated = replace(top_k, attributes=replace(top_k.attributes, version=2))
    operations = tuple(mutated if operation.id == top_k.id else operation for operation in graph.operations)

    with pytest.raises(MoESemanticRecoveryError) as exc_info:
        recover_moe_region(replace(graph, operations=operations), gemm_accumulation_dtype=DType.FP32)

    assert exc_info.value.stage == "router"
    assert "version-1 chlo.top_k" in exc_info.value.reason
