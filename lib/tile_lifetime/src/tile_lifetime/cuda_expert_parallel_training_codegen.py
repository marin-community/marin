# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate scalar reverse bodies for relation-segmented training schedules."""

from pathlib import Path

from tile_lifetime.cuda_map_fold_codegen import (
    CudaArithmeticMode,
    CudaMapFoldProgram,
    CudaScalarFunction,
    render_cuda_scalar_program_include,
)
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.expert_parallel_training import ExpertParallelTrainingPlan
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    scalar_binary,
    scalar_expression_inputs,
    scalar_input,
)


def expert_parallel_training_scalar_program(plan: ExpertParallelTrainingPlan) -> CudaMapFoldProgram:
    """Lower generic reverse Map/Fold expressions to CUDA scalar functions."""
    return expert_training_scalar_program_from_pair_map(plan.forward.map_fold_semantics.pair_map)


def expert_training_scalar_program_from_pair_map(pair_map: ScalarExpression) -> CudaMapFoldProgram:
    """Generate reverse bodies from one erased pair-Map expression."""
    pair_map_left_vjp = scalar_expression_vjp(pair_map, input_name="left", cotangent_name="cotangent")
    pair_map_right_vjp = scalar_expression_vjp(pair_map, input_name="right", cotangent_name="cotangent")
    multiply = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input("value"),
        scalar_input("weight"),
    )
    add = scalar_binary(
        ScalarExpressionKind.ADD,
        scalar_input("state"),
        scalar_input("contribution"),
    )
    return CudaMapFoldProgram(
        functions=(
            CudaScalarFunction(
                "generated_pair_left_vjp",
                _vjp_arguments(pair_map_left_vjp),
                pair_map_left_vjp,
            ),
            CudaScalarFunction(
                "generated_pair_right_vjp",
                _vjp_arguments(pair_map_right_vjp),
                pair_map_right_vjp,
            ),
            CudaScalarFunction(
                "generated_edge_cotangent_map",
                ("value", "weight"),
                multiply,
                CudaArithmeticMode.EXPLICIT_RN,
            ),
            CudaScalarFunction(
                "generated_route_weight_fold_contribution",
                ("value", "weight"),
                multiply,
                CudaArithmeticMode.EXPLICIT_RN,
            ),
            CudaScalarFunction(
                "generated_route_weight_fold_update",
                ("state", "contribution"),
                add,
                CudaArithmeticMode.EXPLICIT_RN,
            ),
            CudaScalarFunction(
                "generated_source_input_fold_update",
                ("state", "contribution"),
                add,
                CudaArithmeticMode.EXPLICIT_RN,
            ),
        )
    )


def _vjp_arguments(expression: ScalarExpression) -> tuple[str, ...]:
    inputs = scalar_expression_inputs(expression)
    return tuple(name for name in ("left", "right", "cotangent") if name in inputs)


def render_cuda_expert_parallel_training_include(plan: ExpertParallelTrainingPlan) -> str:
    """Render generated reverse expressions for generic CUDA loop skeletons."""
    return render_cuda_expert_training_program_include(expert_parallel_training_scalar_program(plan))


def render_cuda_expert_training_program_include(program: CudaMapFoldProgram) -> str:
    """Render reverse scalar bodies plus a uniform pair-VJP skeleton interface."""
    source = render_cuda_scalar_program_include(
        program,
        fingerprint_macro="SHUTTLE_EXPERT_TRAINING_PROGRAM_SHA256",
        generated_by="tile_lifetime.cuda_expert_parallel_training_codegen",
    )
    functions = {function.symbol: function for function in program.functions}
    wrappers = []
    for side in ("left", "right"):
        function = functions[f"generated_pair_{side}_vjp"]
        arguments = ", ".join(function.arguments)
        wrappers.extend(
            (
                "",
                (
                    f"static __device__ __forceinline__ float generated_pair_{side}_vjp_uniform("
                    "float left, float right, float cotangent) {"
                ),
                f"    return generated_pair_{side}_vjp({arguments});",
                "}",
            )
        )
    return "\n".join((source, *wrappers, ""))


def verify_cuda_expert_training_include(path: Path, program: CudaMapFoldProgram) -> None:
    """Reject a checked-in reverse include that drifted from scalar semantics."""
    expected = render_cuda_expert_training_program_include(program)
    if path.read_text() != expected:
        raise ValueError(f"generated CUDA reverse include {path} does not match program {program.fingerprint}")
