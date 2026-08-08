# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime import (
    DType,
    GemmSkeleton,
    NumericalPolicy,
    PlanRuntimeError,
    ReductionSkeleton,
    RMSScalePlacement,
    RuntimeBufferSpec,
    RuntimeDiagnosticCode,
    StreamingAttentionSkeleton,
    TensorBinding,
    compile_stablehlo_dense_transformer_region,
    execute_region_plan,
    required_input_specs,
    validate_region_plan,
)
from tile_lifetime.plan import RegionPlan
from tile_lifetime.reference import (
    DENSE_REGION_INPUT_NAMES,
    DenseDebugConfig,
    export_debug_dense_region,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "dense_region_v1_14_1.mlir.bc.b64"


class RecordingBackend:
    """Backend fake that records the validated public dispatch boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.allocations: list[RuntimeBufferSpec] = []

    def allocate(self, spec: RuntimeBufferSpec) -> TensorBinding:
        self.allocations.append(spec)
        return TensorBinding(handle=("allocation", spec.name), shape=spec.shape, dtype=spec.dtype)

    def alias(self, spec: RuntimeBufferSpec, source: TensorBinding) -> TensorBinding:
        return TensorBinding(handle=source.handle, shape=spec.shape, dtype=spec.dtype)

    def run_gemm(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        self.calls.append(("gemm", skeleton.name))

    def run_attention(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        self.calls.append(("attention", skeleton.name))

    def run_reduction(self, skeleton: ReductionSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        self.calls.append(("reduction", skeleton.name))


def _plan(placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_PROLOGUE) -> RegionPlan:
    artifact = base64.b64decode(FIXTURE.read_text())
    return compile_stablehlo_dense_transformer_region(
        artifact,
        input_names=DENSE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=placement,
    )


def _inputs(plan: RegionPlan) -> dict[str, TensorBinding]:
    return {
        name: TensorBinding(handle=("input", name), shape=spec.shape, dtype=spec.dtype)
        for name, spec in required_input_specs(plan).items()
    }


@pytest.mark.parametrize(
    "placement",
    [RMSScalePlacement.CONSUMER_PROLOGUE, RMSScalePlacement.CONSUMER_EPILOGUE],
)
def test_runtime_dispatches_connected_plan_in_dependency_order(placement: RMSScalePlacement) -> None:
    plan = _plan(placement)
    backend = RecordingBackend()

    result = execute_region_plan(plan, _inputs(plan), backend)

    gate_name = f"contract_{placement.value}_row_scale_pairwise_map"
    assert backend.calls == [
        ("gemm", "contract_partition_pairwise_linear_maps"),
        ("attention", "streaming_normalized_weighted_fold"),
        ("gemm", "contract_maps_and_fold_partials"),
        ("reduction", "combine_fold_partials"),
        ("gemm", gate_name),
        ("gemm", "contract_maps_and_fold_partials"),
        ("reduction", "combine_fold_partials"),
        ("gemm", "contract_partition_pairwise_linear_maps"),
    ]
    assert result.bindings["x_bsh"].handle == result.bindings["x"].handle
    assert result.bindings["attention_flat"].handle == result.bindings["attention"].handle
    assert result.bindings["rotated.query"].handle == result.bindings["qkv"].handle
    assert result.bindings["rotated.key"].handle == result.bindings["qkv"].handle
    assert result.bindings["qkv.value"].handle == result.bindings["qkv"].handle
    assert "next_rotated.query" in result.bindings
    assert "next_rotated.key" in result.bindings
    assert "next_qkv.value" in result.bindings
    assert result.bindings["next_rotated.query"].handle == result.bindings["next_qkv"].handle
    assert result.bindings["next_rotated.key"].handle == result.bindings["next_qkv"].handle
    assert result.bindings["next_qkv.value"].handle == result.bindings["next_qkv"].handle


def test_runtime_validates_primary_shape_contract_and_reduces_one_partial_per_n_tile() -> None:
    config = DenseDebugConfig(
        sequence=2,
        hidden=4096,
        intermediate=14336,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
    )
    plan = compile_stablehlo_dense_transformer_region(
        export_debug_dense_region(config),
        input_names=DENSE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    backend = RecordingBackend()

    execute_region_plan(plan, _inputs(plan), backend)

    assert plan.skeletons[0].cluster_shape == (1, 2, 1)
    assert plan.skeletons[4].cluster_shape == (1, 2, 1)
    first_reduction = plan.skeletons[3]
    second_reduction = plan.skeletons[6]
    assert isinstance(first_reduction, ReductionSkeleton)
    assert isinstance(second_reduction, ReductionSkeleton)
    assert plan.materialization(first_reduction.input).shape == (config.tokens, config.hidden // 256)
    assert plan.materialization(second_reduction.input).shape == (config.tokens, config.hidden // 256)
    allocation_shapes = {spec.name: spec.shape for spec in backend.allocations}
    assert allocation_shapes[first_reduction.output] == (config.tokens,)
    assert allocation_shapes[second_reduction.output] == (config.tokens,)


def test_runtime_rejects_unsupported_backend_layout_attachment_and_resources() -> None:
    plan = _plan()
    first = plan.skeletons[0]
    assert isinstance(first, GemmSkeleton)
    bad_attachment = replace(first.epilogue[0], operation="unsupported_partition")
    mutations = (
        (replace(first, backend="generic_cuda"), RuntimeDiagnosticCode.BACKEND_CONTRACT, "backend"),
        (replace(first, output_layout="row_major"), RuntimeDiagnosticCode.LAYOUT_CONTRACT, "output_layout"),
        (
            replace(first, epilogue=(bad_attachment, *first.epilogue[1:])),
            RuntimeDiagnosticCode.ATTACHMENT_CONTRACT,
            "tile_program",
        ),
        (replace(first, cluster_shape=(2, 1, 1)), RuntimeDiagnosticCode.RESOURCE_CONTRACT, "physical_config"),
    )

    for mutated, code, field in mutations:
        rejected = replace(plan, skeletons=(mutated, *plan.skeletons[1:]))
        with pytest.raises(PlanRuntimeError) as exc_info:
            validate_region_plan(rejected)
        assert any(
            diagnostic.code is code and diagnostic.skeleton_index == 0 and diagnostic.field == field
            for diagnostic in exc_info.value.diagnostics
        )


def test_runtime_reports_missing_and_mismatched_input_bindings() -> None:
    plan = _plan()
    inputs = _inputs(plan)
    del inputs["x"]
    qkv_weight = inputs["qkv_weight"]
    inputs["qkv_weight"] = replace(qkv_weight, shape=(1,))

    with pytest.raises(PlanRuntimeError) as exc_info:
        execute_region_plan(plan, inputs, RecordingBackend())

    diagnostics = exc_info.value.diagnostics
    assert any(
        diagnostic.code is RuntimeDiagnosticCode.MISSING_BINDING and diagnostic.field == "x"
        for diagnostic in diagnostics
    )
    assert any(
        diagnostic.code is RuntimeDiagnosticCode.BINDING_METADATA and diagnostic.field == "qkv_weight"
        for diagnostic in diagnostics
    )


def test_runtime_rejects_mismatched_rms_placement_families() -> None:
    prologue_plan = _plan(RMSScalePlacement.CONSUMER_PROLOGUE)
    epilogue_plan = _plan(RMSScalePlacement.CONSUMER_EPILOGUE)
    mixed = replace(
        prologue_plan,
        skeletons=(*prologue_plan.skeletons[:7], epilogue_plan.skeletons[7]),
    )

    with pytest.raises(PlanRuntimeError) as exc_info:
        validate_region_plan(mixed)

    assert any(
        diagnostic.code is RuntimeDiagnosticCode.BACKEND_CONTRACT and diagnostic.field == "rms_scale_placement"
        for diagnostic in exc_info.value.diagnostics
    )
