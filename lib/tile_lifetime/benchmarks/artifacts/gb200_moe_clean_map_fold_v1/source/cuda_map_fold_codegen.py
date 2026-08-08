# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate CUDA scalar bodies for generic Map and ordered-Fold skeletons."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tile_lifetime.expert_parallel_plan import MapFoldSemantics
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_expression_inputs,
    scalar_input,
    scalar_unary,
    serialize_scalar_expression,
)


class CudaArithmeticMode(StrEnum):
    """Floating-point operator spelling used by one generated scalar body."""

    CUDA_EXPRESSION = "cuda_expression"
    EXPLICIT_RN = "explicit_rn"


@dataclass(frozen=True)
class CudaScalarFunction:
    """One backend-neutral scalar expression specialized as a CUDA device function."""

    symbol: str
    arguments: tuple[str, ...]
    expression: ScalarExpression
    arithmetic_mode: CudaArithmeticMode = CudaArithmeticMode.CUDA_EXPRESSION

    def __post_init__(self) -> None:
        if not self.symbol.isidentifier():
            raise ValueError(f"CUDA scalar symbol must be an identifier: {self.symbol!r}")
        if not self.arguments or len(set(self.arguments)) != len(self.arguments):
            raise ValueError("CUDA scalar arguments must be nonempty and unique")
        if any(not argument.isidentifier() for argument in self.arguments):
            raise ValueError(f"CUDA scalar arguments must be identifiers: {self.arguments!r}")
        referenced = scalar_expression_inputs(self.expression)
        if referenced != set(self.arguments):
            raise ValueError(
                f"CUDA scalar function {self.symbol!r} arguments {sorted(self.arguments)} "
                f"do not match expression inputs {sorted(referenced)}"
            )


@dataclass(frozen=True)
class CudaMapFoldProgram:
    """Scalar semantics plugged into generic pointwise and ordered-Fold loop skeletons."""

    functions: tuple[CudaScalarFunction, ...]

    def __post_init__(self) -> None:
        symbols = tuple(function.symbol for function in self.functions)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("generated CUDA scalar function symbols must be nonempty and unique")

    @property
    def fingerprint(self) -> str:
        """Return a stable digest of the semantic ASTs and numerical modes."""
        return hashlib.sha256(_serialized_program(self).encode()).hexdigest()


def default_map_fold_semantics() -> MapFoldSemantics:
    """Build the standalone probe's backend-neutral scalar semantics."""
    left = scalar_input("left")
    right = scalar_input("right")
    sigmoid = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_constant(1.0),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            scalar_unary(
                ScalarExpressionKind.EXP,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(-1.0), left),
            ),
        ),
    )
    return MapFoldSemantics(
        pair_map=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_binary(ScalarExpressionKind.MULTIPLY, left, sigmoid),
            right,
        ),
        fold_contribution=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("value"),
            scalar_input("weight"),
        ),
        fold_update=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input("state"),
            scalar_input("contribution"),
        ),
        post_fold_map=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input("folded"),
            scalar_input("base"),
        ),
        explicit_rounding_functions=frozenset({"fold_contribution", "fold_update"}),
    )


def shuttle_map_fold_program(semantics: MapFoldSemantics | None = None) -> CudaMapFoldProgram:
    """Lower backend-neutral Map/Fold scalar semantics to CUDA device functions."""
    selected = semantics or default_map_fold_semantics()

    def arithmetic_mode(name: str) -> CudaArithmeticMode:
        if name in selected.explicit_rounding_functions:
            return CudaArithmeticMode.EXPLICIT_RN
        return CudaArithmeticMode.CUDA_EXPRESSION

    return CudaMapFoldProgram(
        functions=(
            CudaScalarFunction(
                "generated_pair_map",
                ("left", "right"),
                selected.pair_map,
                arithmetic_mode("pair_map"),
            ),
            CudaScalarFunction(
                "generated_fold_contribution",
                ("value", "weight"),
                selected.fold_contribution,
                arithmetic_mode("fold_contribution"),
            ),
            CudaScalarFunction(
                "generated_fold_update",
                ("state", "contribution"),
                selected.fold_update,
                arithmetic_mode("fold_update"),
            ),
            CudaScalarFunction(
                "generated_fold_contribution_relaxed",
                ("value", "weight"),
                selected.fold_contribution,
            ),
            CudaScalarFunction(
                "generated_fold_update_relaxed",
                ("state", "contribution"),
                selected.fold_update,
            ),
            CudaScalarFunction(
                "generated_post_fold_map",
                ("folded", "base"),
                selected.post_fold_map,
                arithmetic_mode("post_fold_map"),
            ),
        )
    )


def render_cuda_map_fold_include(program: CudaMapFoldProgram) -> str:
    """Render a self-contained include for generic CUDA loop skeletons."""
    lines = [
        "// Generated by tile_lifetime.cuda_map_fold_codegen; do not edit.",
        f'#define SHUTTLE_MAP_FOLD_PROGRAM_SHA256 "{program.fingerprint}"',
        "",
    ]
    for function in program.functions:
        arguments = ", ".join(f"float {argument}" for argument in function.arguments)
        expression = _cuda_expression(
            function.expression,
            {argument: argument for argument in function.arguments},
            function.arithmetic_mode,
        )
        lines.extend(
            (
                f"static __device__ __forceinline__ float {function.symbol}({arguments}) {{",
                f"    return {expression};",
                "}",
                "",
            )
        )
    return "\n".join(lines)


def verify_cuda_map_fold_include(path: Path, program: CudaMapFoldProgram) -> None:
    """Reject a checked-in CUDA include that drifted from the selected scalar IR."""
    expected = render_cuda_map_fold_include(program)
    observed = path.read_text()
    if observed != expected:
        raise ValueError(
            f"generated CUDA scalar include {path} does not match program {program.fingerprint}; "
            "regenerate it from render_cuda_map_fold_include()"
        )


def evaluate_scalar_expression(expression: ScalarExpression, inputs: Mapping[str, float]) -> float | bool:
    """Evaluate a scalar AST for backend-independent generator tests."""
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    values = tuple(evaluate_scalar_expression(operand, inputs) for operand in expression.operands)
    if kind is ScalarExpressionKind.ADD:
        return float(values[0]) + float(values[1])
    if kind is ScalarExpressionKind.SUBTRACT:
        return float(values[0]) - float(values[1])
    if kind is ScalarExpressionKind.MULTIPLY:
        return float(values[0]) * float(values[1])
    if kind is ScalarExpressionKind.DIVIDE:
        return float(values[0]) / float(values[1])
    if kind is ScalarExpressionKind.EXP:
        return math.exp(float(values[0]))
    if kind is ScalarExpressionKind.RSQRT:
        return 1.0 / math.sqrt(float(values[0]))
    if kind is ScalarExpressionKind.TANH:
        return math.tanh(float(values[0]))
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return float(values[0]) <= float(values[1])
    if kind is ScalarExpressionKind.SELECT:
        return values[1] if bool(values[0]) else values[2]
    raise AssertionError(f"unhandled scalar expression kind {kind}")


def _serialized_program(program: CudaMapFoldProgram) -> str:
    encoded = [
        {
            "symbol": function.symbol,
            "arguments": function.arguments,
            "expression": json.loads(serialize_scalar_expression(function.expression)),
            "arithmetic_mode": function.arithmetic_mode.value,
        }
        for function in program.functions
    ]
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"))


def _cuda_expression(
    expression: ScalarExpression,
    inputs: Mapping[str, str],
    mode: CudaArithmeticMode,
) -> str:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return "true" if expression.constant else "false"
        return _cuda_float(float(expression.constant))
    operands = tuple(_cuda_expression(operand, inputs, mode) for operand in expression.operands)
    if kind in {
        ScalarExpressionKind.ADD,
        ScalarExpressionKind.SUBTRACT,
        ScalarExpressionKind.MULTIPLY,
    }:
        if mode is CudaArithmeticMode.EXPLICIT_RN:
            intrinsic = {
                ScalarExpressionKind.ADD: "__fadd_rn",
                ScalarExpressionKind.SUBTRACT: "__fsub_rn",
                ScalarExpressionKind.MULTIPLY: "__fmul_rn",
            }[kind]
            return f"{intrinsic}({operands[0]}, {operands[1]})"
        operator = {
            ScalarExpressionKind.ADD: "+",
            ScalarExpressionKind.SUBTRACT: "-",
            ScalarExpressionKind.MULTIPLY: "*",
        }[kind]
        return f"({operands[0]} {operator} {operands[1]})"
    if kind is ScalarExpressionKind.DIVIDE:
        return f"({operands[0]} / {operands[1]})"
    if kind is ScalarExpressionKind.EXP:
        return f"expf({operands[0]})"
    if kind is ScalarExpressionKind.RSQRT:
        return f"rsqrtf({operands[0]})"
    if kind is ScalarExpressionKind.TANH:
        return f"tanhf({operands[0]})"
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return f"({operands[0]} <= {operands[1]})"
    if kind is ScalarExpressionKind.SELECT:
        return f"({operands[0]} ? {operands[1]} : {operands[2]})"
    raise AssertionError(f"unhandled scalar expression kind {kind}")


def _cuda_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CUDA scalar literals must be finite")
    rendered = repr(value)
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return f"{rendered}f"
