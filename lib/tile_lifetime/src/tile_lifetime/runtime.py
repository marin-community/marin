# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-testable validation and dispatch for the bounded dense region plan."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND, GemmProgram, compile_gemm_program
from tile_lifetime.ir import DType
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    ReductionSkeleton,
    RegionPlan,
    StreamingAttentionSkeleton,
)
from tile_lifetime.tile_program import TilePrimitive, TileProgramError, TileProgramStage


class RuntimeDiagnosticCode(StrEnum):
    """Stable category for a rejected plan or binding."""

    PLAN_SHAPE = "plan_shape"
    SKELETON_TYPE = "skeleton_type"
    BACKEND_CONTRACT = "backend_contract"
    LAYOUT_CONTRACT = "layout_contract"
    ATTACHMENT_CONTRACT = "attachment_contract"
    RESOURCE_CONTRACT = "resource_contract"
    MATERIALIZATION_CONTRACT = "materialization_contract"
    MISSING_BINDING = "missing_binding"
    BINDING_METADATA = "binding_metadata"
    DEPENDENCY_ORDER = "dependency_order"
    BACKEND_RESULT = "backend_result"


@dataclass(frozen=True)
class RuntimeDiagnostic:
    """One structured runtime validation failure."""

    code: RuntimeDiagnosticCode
    message: str
    skeleton_index: int | None = None
    field: str | None = None
    expected: str | None = None
    actual: str | None = None


class PlanRuntimeError(ValueError):
    """Raised with structured diagnostics when a plan cannot be dispatched."""

    def __init__(self, diagnostics: tuple[RuntimeDiagnostic, ...]):
        if not diagnostics:
            raise ValueError("PlanRuntimeError requires at least one diagnostic")
        self.diagnostics = diagnostics
        super().__init__("; ".join(diagnostic.message for diagnostic in diagnostics))


@dataclass(frozen=True)
class RuntimeBufferSpec:
    """Shape and dtype required for one named runtime binding."""

    name: str
    shape: tuple[int, ...]
    dtype: DType


@dataclass(frozen=True)
class TensorBinding:
    """Backend-owned tensor handle with compiler-visible metadata."""

    handle: object
    shape: tuple[int, ...]
    dtype: DType


class RegionBackend(Protocol):
    """Injected physical backend used by the CPU-independent plan runtime."""

    def allocate(self, spec: RuntimeBufferSpec) -> TensorBinding:
        """Allocate one materialized or partial-statistic buffer."""

    def alias(self, spec: RuntimeBufferSpec, source: TensorBinding) -> TensorBinding:
        """Create a zero-copy logical view of an existing binding."""

    def run_gemm(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        """Dispatch one validated GEMM skeleton."""

    def run_attention(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        """Dispatch one validated streaming-attention skeleton."""

    def run_reduction(self, skeleton: ReductionSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        """Dispatch one validated auxiliary reduction."""


@dataclass(frozen=True)
class RuntimeResult:
    """Bindings produced after dispatching the complete region."""

    bindings: Mapping[str, TensorBinding]


EXPECTED_SKELETON_TYPES = (
    GemmSkeleton,
    StreamingAttentionSkeleton,
    GemmSkeleton,
    ReductionSkeleton,
    GemmSkeleton,
    GemmSkeleton,
    ReductionSkeleton,
    GemmSkeleton,
)
H100_GEMM_TILE_SHAPE = (128, 256, 64)
GENERATED_STREAMING_BACKEND = "h100_streaming_contract_fold"


def validate_region_plan(plan: RegionPlan) -> None:
    """Validate the exact dense H100 skeleton, backend, layout, and attachment contracts."""
    diagnostics: list[RuntimeDiagnostic] = []
    if len(plan.skeletons) != len(EXPECTED_SKELETON_TYPES):
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.PLAN_SHAPE,
                message=f"dense runtime expects eight skeletons, found {len(plan.skeletons)}",
                field="skeletons",
                expected="8",
                actual=str(len(plan.skeletons)),
            )
        )
    for index, expected_type in enumerate(EXPECTED_SKELETON_TYPES):
        if index >= len(plan.skeletons):
            break
        skeleton = plan.skeletons[index]
        if not isinstance(skeleton, expected_type):
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.SKELETON_TYPE,
                    message=f"skeleton {index} must be {expected_type.__name__}, found {type(skeleton).__name__}",
                    skeleton_index=index,
                    field="type",
                    expected=expected_type.__name__,
                    actual=type(skeleton).__name__,
                )
            )
    if diagnostics:
        raise PlanRuntimeError(tuple(diagnostics))

    first_qkv, attention, output_projection, mlp_reduction, gate_up, down, next_reduction, next_qkv = plan.skeletons
    assert isinstance(first_qkv, GemmSkeleton)
    assert isinstance(attention, StreamingAttentionSkeleton)
    assert isinstance(output_projection, GemmSkeleton)
    assert isinstance(mlp_reduction, ReductionSkeleton)
    assert isinstance(gate_up, GemmSkeleton)
    assert isinstance(down, GemmSkeleton)
    assert isinstance(next_reduction, ReductionSkeleton)
    assert isinstance(next_qkv, GemmSkeleton)

    _validate_gemm(
        diagnostics,
        first_qkv,
        index=0,
        input_layout="bsh_contiguous",
        output_layout="fa3_bshd_last_dimension_contiguous",
        cluster_shape=_qkv_cluster_shape(first_qkv.shape[1]),
    )
    _expect(
        diagnostics,
        attention.backend,
        GENERATED_STREAMING_BACKEND,
        code=RuntimeDiagnosticCode.BACKEND_CONTRACT,
        index=1,
        field="backend",
    )
    _expect(
        diagnostics,
        attention.input_layout,
        first_qkv.output_layout,
        code=RuntimeDiagnosticCode.LAYOUT_CONTRACT,
        index=1,
        field="input_layout",
    )
    _expect(
        diagnostics,
        attention.output_layout,
        "bshd_contiguous",
        code=RuntimeDiagnosticCode.LAYOUT_CONTRACT,
        index=1,
        field="output_layout",
    )
    _validate_attachment_ops(
        diagnostics,
        attention.attachments,
        (
            "score_map",
            "domain_restriction",
            "online_fold_update",
            "fold_finalize",
        ),
        (
            AttachmentSite.ATTENTION_SCORE_TRANSFORM,
            AttachmentSite.ATTENTION_SCORE_TRANSFORM,
            AttachmentSite.ATTENTION_ONLINE_UPDATE,
            AttachmentSite.ATTENTION_OUTPUT_TRANSFORM,
        ),
        index=1,
        field="attachments",
    )
    expected_query_block = 192 if attention.head_dimension == 64 else 128
    expected_key_value_block = 128 if attention.causal else (192 if attention.head_dimension == 64 else 176)
    supported_attention_shape = (
        attention.head_dimension in (64, 128)
        and attention.query_heads > 0
        and attention.key_value_heads > 0
        and attention.query_heads % attention.key_value_heads == 0
    )
    if (
        not supported_attention_shape
        or attention.query_block_size != expected_query_block
        or attention.key_value_block_size != expected_key_value_block
        or not math.isclose(attention.scale, attention.head_dimension**-0.5, rel_tol=1e-6)
        or attention.pipeline_stages != 2
        or attention.producer_threads != 32
        or attention.consumer_threads != (expected_query_block // 64) * 128
        or attention.pack_gqa != (attention.query_heads != attention.key_value_heads)
        or attention.mma_pv_is_rs != attention.causal
        or not attention.intra_warpgroup_overlap
        or not attention.persistent_scheduler
        or attention.register_estimate != (168 if attention.head_dimension == 128 else None)
        or len(attention.online_state) != 3
        or attention.attachments[2].outputs != attention.online_state
        or attention.attachments[3].inputs != (attention.online_state[2], attention.online_state[1])
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.RESOURCE_CONTRACT,
                message="streaming Contract/Fold schedule does not match a supported SM90 configuration",
                skeleton_index=1,
                field="physical_schedule",
            )
        )
    _validate_gemm(
        diagnostics,
        output_projection,
        index=2,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        cluster_shape=(1, 1, 1),
    )
    _validate_reduction(diagnostics, mlp_reduction, index=3)

    gate_program = _compile_gemm(diagnostics, gate_up, index=4)
    gate_is_prologue = _has_primitive(gate_program, TileProgramStage.PREPARATION, TilePrimitive.SCALE_ROW)
    gate_is_epilogue = _has_primitive(gate_program, TileProgramStage.FINALIZATION, TilePrimitive.SCALE_ROW)
    _validate_gemm(
        diagnostics,
        gate_up,
        index=4,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        cluster_shape=_wide_projection_cluster_shape(gate_up.shape[1]),
    )
    _validate_gemm(
        diagnostics,
        down,
        index=5,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        cluster_shape=(1, 1, 1),
    )
    _validate_reduction(diagnostics, next_reduction, index=6)

    next_program = _compile_gemm(diagnostics, next_qkv, index=7)
    next_is_prologue = _has_primitive(next_program, TileProgramStage.PREPARATION, TilePrimitive.SCALE_ROW)
    next_is_epilogue = _has_primitive(next_program, TileProgramStage.FINALIZATION, TilePrimitive.SCALE_ROW)
    _validate_gemm(
        diagnostics,
        next_qkv,
        index=7,
        input_layout="row_major_mk",
        output_layout="fa3_bshd_last_dimension_contiguous",
        cluster_shape=_qkv_cluster_shape(next_qkv.shape[1]),
    )
    if gate_is_prologue != next_is_prologue or gate_is_epilogue != next_is_epilogue:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.BACKEND_CONTRACT,
                message="gate/up and next-QKV skeletons select different RMS placement families",
                field="rms_scale_placement",
            )
        )
    _validate_materializations(plan, diagnostics)
    if plan.sequence_squared_materializations:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.MATERIALIZATION_CONTRACT,
                message="dense runtime rejects sequence-squared materializations",
                field="materializations",
            )
        )
    if diagnostics:
        raise PlanRuntimeError(tuple(diagnostics))


def required_input_specs(plan: RegionPlan) -> Mapping[str, RuntimeBufferSpec]:
    """Return the exact named inputs required by a validated dense plan."""
    validate_region_plan(plan)
    return _required_input_specs_unchecked(plan)


def execute_region_plan(
    plan: RegionPlan,
    inputs: Mapping[str, TensorBinding],
    backend: RegionBackend,
) -> RuntimeResult:
    """Validate, allocate, bind aliases, and dispatch a dense region in dependency order."""
    validate_region_plan(plan)
    required = _required_input_specs_unchecked(plan)
    diagnostics: list[RuntimeDiagnostic] = []
    for name, spec in required.items():
        binding = inputs.get(name)
        if binding is None:
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.MISSING_BINDING,
                    message=f"missing required input binding {name}",
                    field=name,
                )
            )
            continue
        if binding.shape != spec.shape or binding.dtype is not spec.dtype:
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.BINDING_METADATA,
                    message=f"binding {name} has incompatible shape or dtype",
                    field=name,
                    expected=f"{spec.shape} {spec.dtype.value}",
                    actual=f"{binding.shape} {binding.dtype.value}",
                )
            )
    if diagnostics:
        raise PlanRuntimeError(tuple(diagnostics))

    bindings = dict(inputs)
    records = {record.value: record for record in plan.materializations}
    for record in plan.materializations:
        if record.disposition not in {
            MaterializationDisposition.MATERIALIZE,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
        }:
            continue
        spec = RuntimeBufferSpec(record.value, record.shape, record.dtype)
        bindings[record.value] = _validated_backend_binding(backend.allocate(spec), spec)
    for skeleton in plan.skeletons:
        if not isinstance(skeleton, ReductionSkeleton):
            continue
        input_record = records[skeleton.input]
        spec = RuntimeBufferSpec(skeleton.output, (input_record.shape[0],), skeleton.reduction_dtype)
        bindings[skeleton.output] = _validated_backend_binding(backend.allocate(spec), spec)

    available = set(required)
    _resolve_aliases(plan, bindings, available, backend)
    for index, skeleton in enumerate(plan.skeletons):
        required_names, produced_names = _dispatch_dependencies(skeleton, records)
        missing = tuple(sorted(required_names - available))
        if missing:
            raise PlanRuntimeError(
                (
                    RuntimeDiagnostic(
                        code=RuntimeDiagnosticCode.DEPENDENCY_ORDER,
                        message=f"skeleton {index} reads unavailable values {missing}",
                        skeleton_index=index,
                        field="inputs",
                        actual=repr(missing),
                    ),
                )
            )
        if isinstance(skeleton, GemmSkeleton):
            backend.run_gemm(skeleton, bindings)
        elif isinstance(skeleton, StreamingAttentionSkeleton):
            backend.run_attention(skeleton, bindings)
        else:
            assert isinstance(skeleton, ReductionSkeleton)
            backend.run_reduction(skeleton, bindings)
        available.update(name for name in produced_names if name in bindings)
        _resolve_aliases(plan, bindings, available, backend)
    return RuntimeResult(bindings=bindings)


def _validate_gemm(
    diagnostics: list[RuntimeDiagnostic],
    skeleton: GemmSkeleton,
    *,
    index: int,
    input_layout: str,
    output_layout: str,
    cluster_shape: tuple[int, int, int],
) -> None:
    if skeleton.backend != GENERIC_H100_GEMM_BACKEND:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.BACKEND_CONTRACT,
                message=f"skeleton {index} backend {skeleton.backend!r} is unsupported",
                skeleton_index=index,
                field="backend",
                expected=repr(GENERIC_H100_GEMM_BACKEND),
                actual=repr(skeleton.backend),
            )
        )
    _expect(
        diagnostics,
        skeleton.input_layout,
        input_layout,
        code=RuntimeDiagnosticCode.LAYOUT_CONTRACT,
        index=index,
        field="input_layout",
    )
    _expect(
        diagnostics,
        skeleton.output_layout,
        output_layout,
        code=RuntimeDiagnosticCode.LAYOUT_CONTRACT,
        index=index,
        field="output_layout",
    )
    _compile_gemm(diagnostics, skeleton, index=index)
    if skeleton.accumulation_dtype is not DType.FP32:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.RESOURCE_CONTRACT,
                message=f"skeleton {index} must accumulate in FP32",
                skeleton_index=index,
                field="accumulation_dtype",
            )
        )
    if (
        skeleton.physical_tile_shape != H100_GEMM_TILE_SHAPE
        or skeleton.cluster_shape != cluster_shape
        or skeleton.pingpong is not False
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.RESOURCE_CONTRACT,
                message=f"skeleton {index} does not match the supported SM90 physical configuration",
                skeleton_index=index,
                field="physical_config",
            )
        )


def _compile_gemm(diagnostics: list[RuntimeDiagnostic], skeleton: GemmSkeleton, *, index: int) -> GemmProgram | None:
    try:
        return compile_gemm_program(skeleton)
    except TileProgramError as error:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.ATTACHMENT_CONTRACT,
                message=f"skeleton {index} tile program is invalid: {error}",
                skeleton_index=index,
                field="tile_program",
            )
        )
        return None


def _has_primitive(program: GemmProgram | None, stage: TileProgramStage, primitive: TilePrimitive) -> bool:
    return program is not None and primitive in program.tile_program.primitives_at(stage)


def _validate_reduction(
    diagnostics: list[RuntimeDiagnostic],
    skeleton: ReductionSkeleton,
    *,
    index: int,
) -> None:
    if skeleton.reduction_dtype is not DType.FP32 or not skeleton.operator.startswith("rsqrt(sum / "):
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.RESOURCE_CONTRACT,
                message=f"skeleton {index} is not the supported FP32 inverse-RMS reduction",
                skeleton_index=index,
                field="reduction",
            )
        )


def _validate_attachment_ops(
    diagnostics: list[RuntimeDiagnostic],
    attachments: tuple[Attachment, ...],
    expected_operations: tuple[str, ...],
    expected_sites: tuple[AttachmentSite, ...],
    *,
    index: int,
    field: str,
) -> None:
    actual_operations = tuple(attachment.operation for attachment in attachments)
    actual_sites = tuple(attachment.site for attachment in attachments)
    if actual_operations != expected_operations or actual_sites != expected_sites:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.ATTACHMENT_CONTRACT,
                message=f"skeleton {index} {field} does not match the supported attachment program",
                skeleton_index=index,
                field=field,
                expected=repr((expected_operations, expected_sites)),
                actual=repr((actual_operations, actual_sites)),
            )
        )


def _expect(
    diagnostics: list[RuntimeDiagnostic],
    actual: object,
    expected: object,
    *,
    code: RuntimeDiagnosticCode,
    index: int,
    field: str,
) -> None:
    if actual != expected:
        diagnostics.append(
            RuntimeDiagnostic(
                code=code,
                message=f"skeleton {index} {field} is {actual!r}, expected {expected!r}",
                skeleton_index=index,
                field=field,
                expected=repr(expected),
                actual=repr(actual),
            )
        )


def _validate_materializations(plan: RegionPlan, diagnostics: list[RuntimeDiagnostic]) -> None:
    names = [record.value for record in plan.materializations]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        diagnostics.append(
            RuntimeDiagnostic(
                code=RuntimeDiagnosticCode.MATERIALIZATION_CONTRACT,
                message=f"plan contains duplicate materialization records {duplicates}",
                field="materializations",
            )
        )
    for record in plan.materializations:
        if record.disposition is MaterializationDisposition.ALIAS and record.alias_of is None:
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.MATERIALIZATION_CONTRACT,
                    message=f"alias {record.value} does not identify its source",
                    field=record.value,
                )
            )
    records = {record.value: record for record in plan.materializations}
    for index in (0, 7):
        skeleton = plan.skeletons[index]
        assert isinstance(skeleton, GemmSkeleton)
        packed = records.get(skeleton.output)
        try:
            alias_names = compile_gemm_program(skeleton).stored_values
        except TileProgramError:
            alias_names = ()
        aliases = tuple(records.get(name) for name in alias_names)
        if (
            packed is None
            or packed.disposition is not MaterializationDisposition.MATERIALIZE
            or len(alias_names) != 3
            or any(
                alias is None
                or alias.disposition is not MaterializationDisposition.ALIAS
                or alias.alias_of != skeleton.output
                for alias in aliases
            )
        ):
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.MATERIALIZATION_CONTRACT,
                    message=f"skeleton {index} must expose Q, K, and V as views of one packed QKV allocation",
                    skeleton_index=index,
                    field="packed_qkv",
                )
            )
    for index in (3, 6):
        skeleton = plan.skeletons[index]
        assert isinstance(skeleton, ReductionSkeleton)
        partial = records.get(skeleton.input)
        if (
            partial is None
            or partial.disposition is not MaterializationDisposition.PARTIAL_REDUCTION_ONLY
            or len(partial.shape) != 2
        ):
            diagnostics.append(
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.MATERIALIZATION_CONTRACT,
                    message=f"skeleton {index} must reduce a two-dimensional row-by-tile partial buffer",
                    skeleton_index=index,
                    field="input_partials",
                )
            )


def _wide_projection_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return 1, 2 if output_width >= 16_384 else 1, 1


def _qkv_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return 1, 2 if output_width >= 6_144 else 1, 1


def _required_input_specs_unchecked(plan: RegionPlan) -> Mapping[str, RuntimeBufferSpec]:
    first_qkv = plan.skeletons[0]
    attention = plan.skeletons[1]
    output_projection = plan.skeletons[2]
    gate_up = plan.skeletons[4]
    down = plan.skeletons[5]
    next_qkv = plan.skeletons[7]
    assert isinstance(first_qkv, GemmSkeleton)
    assert isinstance(attention, StreamingAttentionSkeleton)
    assert isinstance(output_projection, GemmSkeleton)
    assert isinstance(gate_up, GemmSkeleton)
    assert isinstance(down, GemmSkeleton)
    assert isinstance(next_qkv, GemmSkeleton)
    records = {record.value: record for record in plan.materializations}
    x_alias = records[first_qkv.input]
    assert x_alias.alias_of is not None
    x_shape = records[output_projection.epilogue[0].outputs[0]].shape
    sine_name, cosine_name = first_qkv.epilogue[1].inputs[1:]
    sequence = records[attention.query].shape[1]
    rope_shape = (sequence, attention.head_dimension // 2)
    specs = (
        RuntimeBufferSpec(x_alias.alias_of, x_shape, DType.BF16),
        RuntimeBufferSpec(first_qkv.weight, (first_qkv.shape[2], first_qkv.shape[1]), DType.BF16),
        RuntimeBufferSpec(
            output_projection.weight, (output_projection.shape[2], output_projection.shape[1]), DType.BF16
        ),
        RuntimeBufferSpec(output_projection.epilogue[1].inputs[1], (output_projection.shape[1],), DType.BF16),
        RuntimeBufferSpec(gate_up.weight, (gate_up.shape[2], gate_up.shape[1]), DType.BF16),
        RuntimeBufferSpec(down.weight, (down.shape[2], down.shape[1]), DType.BF16),
        RuntimeBufferSpec(down.epilogue[1].inputs[1], (down.shape[1],), DType.BF16),
        RuntimeBufferSpec(next_qkv.weight, (next_qkv.shape[2], next_qkv.shape[1]), DType.BF16),
        RuntimeBufferSpec(sine_name, rope_shape, DType.BF16),
        RuntimeBufferSpec(cosine_name, rope_shape, DType.BF16),
    )
    return {spec.name: spec for spec in specs}


def _validated_backend_binding(binding: TensorBinding, spec: RuntimeBufferSpec) -> TensorBinding:
    if binding.shape != spec.shape or binding.dtype is not spec.dtype:
        raise PlanRuntimeError(
            (
                RuntimeDiagnostic(
                    code=RuntimeDiagnosticCode.BACKEND_RESULT,
                    message=f"backend returned incompatible metadata for {spec.name}",
                    field=spec.name,
                    expected=f"{spec.shape} {spec.dtype.value}",
                    actual=f"{binding.shape} {binding.dtype.value}",
                ),
            )
        )
    return binding


def _resolve_aliases(
    plan: RegionPlan,
    bindings: dict[str, TensorBinding],
    available: set[str],
    backend: RegionBackend,
) -> None:
    for record in plan.materializations:
        if record.disposition is not MaterializationDisposition.ALIAS or record.value in available:
            continue
        assert record.alias_of is not None
        if record.alias_of not in available:
            continue
        spec = RuntimeBufferSpec(record.value, record.shape, record.dtype)
        bindings[record.value] = _validated_backend_binding(backend.alias(spec, bindings[record.alias_of]), spec)
        available.add(record.value)


def _dispatch_dependencies(
    skeleton: GemmSkeleton | StreamingAttentionSkeleton | ReductionSkeleton,
    records: Mapping[str, MaterializationRecord],
) -> tuple[set[str], set[str]]:
    if isinstance(skeleton, StreamingAttentionSkeleton):
        return {skeleton.query, skeleton.key, skeleton.value}, {skeleton.output}
    if isinstance(skeleton, ReductionSkeleton):
        return {skeleton.input}, {skeleton.output}

    required = {skeleton.input, skeleton.weight}
    local = {skeleton.output}
    epilogue_names = {name for attachment in skeleton.epilogue for name in (*attachment.inputs, *attachment.outputs)}
    local.update(
        name
        for name in epilogue_names
        if name in records and records[name].disposition is MaterializationDisposition.EPILOGUE_ONLY
    )
    for attachment in skeleton.prologue:
        required.update(name for name in attachment.inputs if name not in local)
        local.update(attachment.outputs)
    for attachment in skeleton.epilogue:
        required.update(name for name in attachment.inputs if name not in local)
        local.update(attachment.outputs)
    return required, local
