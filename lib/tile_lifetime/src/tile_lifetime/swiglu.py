# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical named-TensorGraph SwiGLU planner."""

from dataclasses import dataclass

from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.ir import (
    LinearOp,
    PairwiseSwiGLUOp,
    SemanticOp,
    SwiGLUOp,
    TensorGraph,
    TensorValue,
    operation_inputs,
)
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    RegionPlan,
    RewriteExplanation,
    TransformSkeleton,
)


@dataclass(frozen=True)
class _SwiGLURegion:
    activation: SwiGLUOp | PairwiseSwiGLUOp
    projections: tuple[LinearOp, ...]
    down_projection: LinearOp
    input: TensorValue
    combined_weight: str
    mainloop_output_width: int


def compile_reference_swiglu_region(graph: TensorGraph, *, numerical_policy: NumericalPolicy) -> RegionPlan:
    """Compile one hand-built named SwiGLU graph for reference tests."""
    region, rejection_reasons = _find_swiglu_region(graph, numerical_policy=numerical_policy)
    if region is None:
        return _materialized_swiglu_fallback(graph, rejection_reasons=rejection_reasons)
    return _swiglu_plan(region)


def _find_swiglu_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
) -> tuple[_SwiGLURegion | None, tuple[str, ...]]:
    activations = tuple(
        operation for operation in graph.operations if isinstance(operation, (SwiGLUOp, PairwiseSwiGLUOp))
    )
    if len(activations) != 1:
        return None, (f"expected one SwiGLU operation, found {len(activations)}",)

    activation = activations[0]
    reasons: list[str] = []
    if isinstance(activation, SwiGLUOp):
        gate_projection = graph.producer(activation.gate)
        up_projection = graph.producer(activation.up)
        if not isinstance(gate_projection, LinearOp) or not isinstance(up_projection, LinearOp):
            reasons.append("separate SwiGLU inputs are not both produced by linear projections")
            projections = ()
            input_value = None
            combined_weight = ""
            mainloop_output_width = 0
        else:
            projections = (gate_projection, up_projection)
            input_value = gate_projection.input
            combined_weight = f"interleave_adjacent({gate_projection.weight.name},{up_projection.weight.name})"
            mainloop_output_width = gate_projection.output.shape[-1] * 2
            if gate_projection.id == up_projection.id:
                reasons.append("gate and up must be distinct linear projections")
            if gate_projection.input != up_projection.input:
                reasons.append("gate and up projections do not share the same input")
            if gate_projection.accumulation_dtype != up_projection.accumulation_dtype:
                reasons.append("gate and up projections use different accumulation dtypes")
            for projection in projections:
                consumers = graph.consumers(projection.output)
                if consumers != (activation,):
                    reasons.append(
                        f"{projection.output.name} has {len(consumers)} consumers; epilogue fusion requires only SwiGLU"
                    )
    else:
        combined_projection = graph.producer(activation.input)
        if not isinstance(combined_projection, LinearOp):
            reasons.append("pairwise SwiGLU input is not produced by a linear projection")
            projections = ()
            input_value = None
            combined_weight = ""
            mainloop_output_width = 0
        else:
            projections = (combined_projection,)
            input_value = combined_projection.input
            combined_weight = combined_projection.weight.name
            mainloop_output_width = combined_projection.output.shape[-1]
            consumers = graph.consumers(combined_projection.output)
            if consumers != (activation,):
                reasons.append(
                    f"{combined_projection.output.name} has {len(consumers)} consumers; "
                    "epilogue fusion requires only SwiGLU"
                )

    activation_consumers = graph.consumers(activation.output)
    if len(activation_consumers) != 1:
        reasons.append(
            f"SwiGLU output has {len(activation_consumers)} consumers; direct down-projection planning requires one"
        )
        down_projection = None
    else:
        candidate = activation_consumers[0]
        if not isinstance(candidate, LinearOp) or candidate.input != activation.output:
            reasons.append("SwiGLU output's only consumer is not the down-projection GEMM")
            down_projection = None
        else:
            down_projection = candidate

    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        reasons.append("epilogue SwiGLU bypasses materialized BF16 gate/up rounding under the bitwise-exact policy")

    if reasons or not projections or input_value is None or down_projection is None:
        return None, tuple(reasons)
    return (
        _SwiGLURegion(
            activation=activation,
            projections=projections,
            down_projection=down_projection,
            input=input_value,
            combined_weight=combined_weight,
            mainloop_output_width=mainloop_output_width,
        ),
        (),
    )


def _swiglu_plan(region: _SwiGLURegion) -> RegionPlan:
    activation = region.activation
    first_projection = region.projections[0]
    down = region.down_projection
    projection_outputs = tuple(projection.output.name for projection in region.projections)

    gate_up = GemmSkeleton(
        name="gate_up_projection_pairwise_swiglu",
        input=region.input.name,
        weight=region.combined_weight,
        output=activation.output.name,
        shape=(activation.output.shape[0], region.mainloop_output_width, region.input.shape[1]),
        accumulation_dtype=first_projection.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn_pair_reduced",
        epilogue=(
            Attachment(
                operation="pairwise_swiglu",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=projection_outputs,
                outputs=(activation.output.name,),
            ),
        ),
    )
    down_gemm = GemmSkeleton(
        name="down_projection",
        input=activation.output.name,
        weight=down.weight.name,
        output=down.output.name,
        shape=(down.output.shape[0], down.output.shape[1], down.input.shape[1]),
        accumulation_dtype=down.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
    )

    projection_materializations = tuple(
        MaterializationRecord(
            value=projection.output.name,
            shape=projection.output.shape,
            dtype=projection.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="gate/up accumulator values are consumed as adjacent pairs before the GEMM tile is stored",
        )
        for projection in region.projections
    )
    materializations = (
        *projection_materializations,
        MaterializationRecord(
            value=activation.output.name,
            shape=activation.output.shape,
            dtype=activation.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="dimension-reduced SwiGLU output is the down-projection mainloop input",
        ),
        MaterializationRecord(
            value=down.output.name,
            shape=down.output.shape,
            dtype=down.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="region output",
        ),
    )
    source_form = "separate gate/up projections" if isinstance(activation, SwiGLUOp) else "combined projection"
    explanation = RewriteExplanation(
        name="fuse_pairwise_swiglu_into_gate_up_epilogue",
        applied=True,
        original_fragment=(*projection_outputs, "silu(gate) * up", down.output.name),
        transformed_fragment=(
            f"{source_form} -> adjacent (gate, up) accumulator pairs",
            "apply pairwise SwiGLU in the gate/up GEMM epilogue",
            f"store only {activation.output.name} for the down projection",
        ),
        semantic_properties=(
            "gate and up are pairwise-local",
            "SwiGLU reduces each adjacent gate/up pair to one feature",
            "the down projection consumes the reduced feature dimension",
        ),
        legality_checks=(
            "gate/up projections share one input and accumulation dtype",
            "gate/up accumulator values have no external consumers",
            "combined projection pairs are adjacent in (gate, up) order",
            "SwiGLU output has exactly one down-projection consumer",
            "numerical policy permits bypassing materialized preactivation rounding",
        ),
        estimated_benefit="eliminates the expanded gate/up write and read plus a standalone SwiGLU kernel",
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=(
            "SwiGLU consumes FP32 GEMM accumulators before the BF16 store, changing rounding relative to "
            "materialized BF16 gate and up tensors"
        ),
    )
    return RegionPlan(
        skeletons=(gate_up, down_gemm),
        materializations=materializations,
        rewrites=(explanation,),
    )


def _materialized_swiglu_fallback(
    graph: TensorGraph,
    *,
    rejection_reasons: tuple[str, ...],
) -> RegionPlan:
    skeletons = tuple(_fallback_skeleton(operation) for operation in graph.operations)
    materializations = tuple(
        MaterializationRecord(
            value=operation.output.name,
            shape=operation.output.shape,
            dtype=operation.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="standalone fallback operation boundary",
        )
        for operation in graph.operations
    )
    explanation = RewriteExplanation(
        name="fuse_pairwise_swiglu_into_gate_up_epilogue",
        applied=False,
        original_fragment=tuple(type(operation).__name__ for operation in graph.operations),
        transformed_fragment=(),
        semantic_properties=("pairwise-local SwiGLU",),
        legality_checks=(),
        estimated_benefit="none; the materialized reference path is retained",
        numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
        numerical_effect="none",
        rejection_reasons=rejection_reasons,
    )
    return RegionPlan(skeletons=skeletons, materializations=materializations, rewrites=(explanation,))


def _fallback_skeleton(operation: SemanticOp) -> GemmSkeleton | TransformSkeleton:
    if isinstance(operation, LinearOp):
        return GemmSkeleton(
            name=f"materialized_{operation.output.name}",
            input=operation.input.name,
            weight=operation.weight.name,
            output=operation.output.name,
            shape=(operation.output.shape[0], operation.output.shape[1], operation.input.shape[1]),
            accumulation_dtype=operation.accumulation_dtype,
        )
    if isinstance(operation, SwiGLUOp):
        operation_name = "swiglu"
    elif isinstance(operation, PairwiseSwiGLUOp):
        operation_name = "pairwise_swiglu"
    else:
        operation_name = type(operation).__name__.removesuffix("Op").lower()
    return TransformSkeleton(
        name=f"materialized_{operation.output.name}",
        operation=operation_name,
        inputs=tuple(value.name for value in operation_inputs(operation)),
        output=operation.output.name,
    )
