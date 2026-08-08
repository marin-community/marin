# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural proof that named frontend semantics erase before scheduling."""

from dataclasses import dataclass, replace

from tile_lifetime.plan import (
    GemmSkeleton,
    ReductionSkeleton,
    RegionPlan,
    SemanticErasureReport,
    SemanticLoweringStep,
    StreamingAttentionSkeleton,
    TransformSkeleton,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    MapPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorProgram,
)


class SemanticErasureError(ValueError):
    """A named semantic construct survived into physical candidate selection."""


_GENERIC_SEMANTIC_PRIMITIVES = frozenset(
    {
        "Map",
        "Contract",
        "Fold",
        "Scan",
        "Relation",
        "RelationPlan",
        "Selection",
        "SegmentedContract",
        "DomainRestriction",
        "Transport",
        "Materialize",
    }
)
_FORBIDDEN_SCHEDULING_TOKENS = (
    "rmsnorm",
    "layernorm",
    "flashattention",
    "scaled_dot_product_attention",
    "causal_attention",
    "swiglu",
    "rope",
    "mixture_of_kittens",
    "mok_forward",
    "moe_forward",
    "gated_deltanet",
    "gdn",
    "mamba",
)
_GENERIC_ATTACHMENT_OPERATIONS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "scale_row",
        "partial_sum",
        "partial_sum_square",
        "pairwise_map",
        "pairwise_linear_map",
        "view",
        "partition",
        "convert",
        "load_tile",
        "load_row",
        "store_tile",
        "score_map",
        "domain_restriction",
        "online_fold_update",
        "fold_finalize",
    }
)


@dataclass(frozen=True)
class ErasedTensorProgram:
    """A generic tensor program paired with independently derived erasure evidence."""

    program: TensorProgram
    report: SemanticErasureReport

    def with_program(self, program: TensorProgram) -> "ErasedTensorProgram":
        """Replace generic math and recompute its scheduling signatures."""
        return ErasedTensorProgram(
            program=program,
            report=replace(self.report, scheduling_keys=tensor_program_scheduling_keys(program)),
        )


def build_tensor_erasure_report(
    program: TensorProgram,
    *,
    source_semantics: tuple[str, ...],
    lowering_steps: tuple[SemanticLoweringStep, ...],
) -> SemanticErasureReport:
    """Build a report whose schedule keys are derived from generic structure."""
    provisional = SemanticErasureReport(
        source_semantics=source_semantics,
        lowering_steps=lowering_steps,
        scheduling_keys=tensor_program_scheduling_keys(program),
    )
    errors = semantic_erasure_errors(provisional)
    return replace(provisional, validation_errors=errors)


def tensor_program_scheduling_keys(program: TensorProgram) -> tuple[str, ...]:
    """Derive name-free candidate-selection keys from a generic tensor program."""
    keys: list[str] = []
    for operation in program.operations:
        if isinstance(operation, ContractPrimitive):
            keys.append(
                "contract:"
                f"inputs={len(operation.inputs)}:"
                f"output_rank={len(operation.output.axes)}:"
                f"reduction_rank={len(operation.reduction_axes)}:"
                f"accumulate={operation.accumulation_dtype.value}"
            )
        elif isinstance(operation, FoldPrimitive):
            keys.append(
                f"fold:{operation.reducer.value}:reduction_rank={len(operation.reduction_axes)}:"
                f"accumulate={operation.accumulation_dtype.value}"
            )
        else:
            assert isinstance(operation, MapPrimitive)
            keys.append(f"map:{_expression_signature(operation.expression)}")
    return tuple(keys)


def validate_erased_tensor_program(erased: ErasedTensorProgram) -> None:
    """Reject stale or named reports before physical candidates are enumerated."""
    expected_keys = tensor_program_scheduling_keys(erased.program)
    errors = list(semantic_erasure_errors(erased.report))
    if erased.report.scheduling_keys != expected_keys:
        errors.append("scheduling keys do not match the supplied generic tensor program")
    if errors:
        raise SemanticErasureError("; ".join(errors))


def semantic_erasure_errors(report: SemanticErasureReport) -> tuple[str, ...]:
    """Return structural report violations without inspecting diagnostic names."""
    errors = list(report.validation_errors)
    for step in report.lowering_steps:
        unsupported = tuple(
            primitive for primitive in step.generic_primitives if primitive not in _GENERIC_SEMANTIC_PRIMITIVES
        )
        if unsupported:
            errors.append(f"{step.source_semantic!r} lowers to unsupported semantic primitives {unsupported}")
    for key in report.scheduling_keys:
        normalized = key.lower().replace("-", "_").replace(" ", "_")
        matched = tuple(token for token in _FORBIDDEN_SCHEDULING_TOKENS if token in normalized)
        if matched:
            errors.append(f"scheduling key {key!r} retains named semantics {matched}")
    return tuple(dict.fromkeys(errors))


def validate_plan_semantic_erasure(plan: RegionPlan) -> None:
    """Validate the accepted physical path contains no named semantic dispatch."""
    report = plan.semantic_erasure_report
    if report is None:
        raise SemanticErasureError("region plan has no semantic-erasure report")
    errors = list(semantic_erasure_errors(report))
    for skeleton in plan.skeletons:
        normalized_name = skeleton.name.lower().replace("-", "_")
        if any(token in normalized_name for token in _FORBIDDEN_SCHEDULING_TOKENS):
            errors.append(f"physical skeleton name {skeleton.name!r} retains a workload name")
        if isinstance(skeleton, TransformSkeleton):
            errors.append(f"materialized transform {skeleton.operation!r} remains in the accepted plan")
            continue
        if isinstance(skeleton, GemmSkeleton):
            for attachment in (*skeleton.prologue, *skeleton.epilogue):
                if attachment.operation not in _GENERIC_ATTACHMENT_OPERATIONS:
                    errors.append(f"GEMM attachment {attachment.operation!r} is not a generic tile primitive")
            if skeleton.backend is not None:
                normalized_backend = skeleton.backend.lower().replace("-", "_")
                if any(token in normalized_backend for token in _FORBIDDEN_SCHEDULING_TOKENS):
                    errors.append(f"GEMM backend {skeleton.backend!r} retains a workload name")
        elif isinstance(skeleton, ReductionSkeleton):
            normalized_operator = skeleton.operator.lower().replace("-", "_")
            if any(token in normalized_operator for token in _FORBIDDEN_SCHEDULING_TOKENS):
                errors.append(f"reduction operator {skeleton.operator!r} retains a workload name")
        elif isinstance(skeleton, StreamingAttentionSkeleton):
            for attachment in skeleton.attachments:
                if attachment.operation not in _GENERIC_ATTACHMENT_OPERATIONS:
                    errors.append(f"streaming attachment {attachment.operation!r} is not a generic semantic primitive")
            normalized_backend = skeleton.backend.lower().replace("-", "_")
            if any(token in normalized_backend for token in _FORBIDDEN_SCHEDULING_TOKENS):
                errors.append(f"streaming backend {skeleton.backend!r} retains a workload name")
    if errors:
        raise SemanticErasureError("; ".join(dict.fromkeys(errors)))


def _expression_signature(expression: ScalarExpression) -> str:
    if expression.kind is ScalarExpressionKind.INPUT:
        return "input"
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return "constant"
    operands = ",".join(_expression_signature(operand) for operand in expression.operands)
    return f"{expression.kind.value}({operands})"
