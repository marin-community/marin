# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cast-aware scalar programs imported from physical tensor dataflow."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class CastScalarKind(StrEnum):
    """Operations accepted by the generic cast-aware scalar generator."""

    INPUT = "input"
    CONSTANT = "constant"
    CONVERT = "convert"
    NEGATE = "negate"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    TANH = "tanh"
    SELECT = "select"


class CastScalarDType(StrEnum):
    """Scalar types whose conversion behavior is represented explicitly."""

    BF16 = "bf16"
    F32 = "f32"
    PRED = "pred"


class CastScalarNumericalPolicy(StrEnum):
    """Finite-precision ordering required by a generated scalar body."""

    SOURCE_ORDERED = "source_ordered"


@dataclass(frozen=True)
class ScalarIndexRelation:
    """Affine source coordinate for one scalar input at an output coordinate."""

    row_offset: int
    feature_offset: int


@dataclass(frozen=True)
class CastScalarExpression:
    """One scalar expression node with an explicit result dtype."""

    kind: CastScalarKind
    dtype: CastScalarDType
    operands: tuple[CastScalarExpression, ...] = ()
    input_name: str | None = None
    input_index: ScalarIndexRelation | None = None
    constant: float | bool | None = None

    def __post_init__(self) -> None:
        if self.kind is CastScalarKind.INPUT:
            if self.input_name is None or self.input_index is None or self.operands or self.constant is not None:
                raise ValueError("scalar input requires a name and index relation only")
            return
        if self.kind is CastScalarKind.CONSTANT:
            if self.constant is None or self.operands or self.input_name is not None or self.input_index is not None:
                raise ValueError("scalar constant requires a literal only")
            return
        arity = {
            CastScalarKind.CONVERT: 1,
            CastScalarKind.NEGATE: 1,
            CastScalarKind.EXP: 1,
            CastScalarKind.TANH: 1,
            CastScalarKind.ADD: 2,
            CastScalarKind.SUBTRACT: 2,
            CastScalarKind.MULTIPLY: 2,
            CastScalarKind.DIVIDE: 2,
            CastScalarKind.SELECT: 3,
        }[self.kind]
        if (
            len(self.operands) != arity
            or self.input_name is not None
            or self.input_index is not None
            or self.constant is not None
        ):
            raise ValueError(f"scalar {self.kind.value} requires {arity} operands")


@dataclass(frozen=True)
class CastScalarProgram:
    """A pointwise scalar body plus its physical input index relations."""

    expression: CastScalarExpression
    numerical_policy: CastScalarNumericalPolicy = CastScalarNumericalPolicy.SOURCE_ORDERED

    @property
    def inputs(self) -> tuple[CastScalarExpression, ...]:
        """Return distinct scalar leaves ordered by their index relation."""
        leaves: dict[tuple[str, ScalarIndexRelation], CastScalarExpression] = {}

        def visit(expression: CastScalarExpression) -> None:
            if expression.kind is CastScalarKind.INPUT:
                assert expression.input_name is not None and expression.input_index is not None
                leaves[(expression.input_name, expression.input_index)] = expression
            for operand in expression.operands:
                visit(operand)

        visit(self.expression)
        return tuple(
            leaves[key]
            for key in sorted(
                leaves,
                key=lambda value: (value[1].row_offset, value[1].feature_offset, value[0]),
            )
        )

    @property
    def serialized(self) -> str:
        """Return a canonical, source-name-independent scalar encoding."""
        return json.dumps(
            {
                "numerical_policy": self.numerical_policy.value,
                "expression": _encode_expression(self.expression),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        """Return a stable semantic digest of operations, casts, and index maps."""
        return hashlib.sha256(self.serialized.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Encode this scalar body for a compiler ownership report."""
        encoded = json.loads(self.serialized)
        return {
            "digest": self.digest,
            "numerical_policy": self.numerical_policy.value,
            "expression": encoded["expression"],
            "inputs": [
                {
                    "name": value.input_name,
                    "dtype": value.dtype.value,
                    "row_offset": value.input_index.row_offset,
                    "feature_offset": value.input_index.feature_offset,
                }
                for value in self.inputs
                if value.input_index is not None
            ],
        }


@dataclass(frozen=True)
class GeneratedCudaScalarBody:
    """A generic CUDA device function rendered from a cast-aware scalar AST."""

    symbol: str
    source: str
    semantic_digest: str
    source_digest: str

    def to_dict(self) -> dict[str, str]:
        """Encode generated source and both provenance digests."""
        return {
            "symbol": self.symbol,
            "semantic_digest": self.semantic_digest,
            "source_digest": self.source_digest,
            "source": self.source,
        }


def generate_cuda_scalar_body(
    program: CastScalarProgram, *, symbol: str = "generated_scalar_map"
) -> GeneratedCudaScalarBody:
    """Render a generic CUDA scalar function preserving explicit BF16 casts."""
    if not symbol.isidentifier():
        raise ValueError(f"CUDA scalar symbol must be an identifier: {symbol!r}")
    arguments = ", ".join(f"float {value.input_name}" for value in program.inputs)
    expression = _render_cuda_expression(program.expression)
    source = (
        "// Generated by tile_lifetime.cast_scalar_program; do not edit.\n"
        "#include <cuda_bf16.h>\n\n"
        f"// Scalar semantic SHA256: {program.digest}\n"
        f"static __device__ __forceinline__ float {symbol}({arguments}) {{\n"
        f"    return {expression};\n"
        "}\n"
    )
    return GeneratedCudaScalarBody(
        symbol=symbol,
        source=source,
        semantic_digest=program.digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def evaluate_cast_scalar_program(program: CastScalarProgram, inputs: Mapping[str, float]) -> float | bool:
    """Evaluate a generated scalar body with explicit BF16 round-to-nearest-even."""

    def evaluate(expression: CastScalarExpression) -> float | bool:
        if expression.kind is CastScalarKind.INPUT:
            assert expression.input_name is not None
            value = inputs[expression.input_name]
            if expression.dtype is CastScalarDType.BF16:
                return _round_float32_to_bfloat16(value)
            if expression.dtype is CastScalarDType.F32:
                return _round_float32(value)
            return bool(value)
        if expression.kind is CastScalarKind.CONSTANT:
            assert expression.constant is not None
            return expression.constant
        values = tuple(evaluate(operand) for operand in expression.operands)
        if expression.kind is CastScalarKind.CONVERT:
            value = float(values[0])
        elif expression.kind is CastScalarKind.NEGATE:
            value = -float(values[0])
        elif expression.kind is CastScalarKind.ADD:
            value = float(values[0]) + float(values[1])
        elif expression.kind is CastScalarKind.SUBTRACT:
            value = float(values[0]) - float(values[1])
        elif expression.kind is CastScalarKind.MULTIPLY:
            value = float(values[0]) * float(values[1])
        elif expression.kind is CastScalarKind.DIVIDE:
            value = float(values[0]) / float(values[1])
        elif expression.kind is CastScalarKind.EXP:
            value = math.exp(float(values[0]))
        elif expression.kind is CastScalarKind.TANH:
            value = math.tanh(float(values[0]))
        else:
            assert expression.kind is CastScalarKind.SELECT
            value = values[1] if bool(values[0]) else values[2]
        if expression.dtype is CastScalarDType.BF16:
            return _round_float32_to_bfloat16(float(value))
        if expression.dtype is CastScalarDType.PRED:
            return bool(value)
        return _round_float32(float(value))

    return evaluate(program.expression)


def _encode_expression(expression: CastScalarExpression) -> dict[str, object]:
    encoded: dict[str, object] = {"kind": expression.kind.value, "dtype": expression.dtype.value}
    if expression.kind is CastScalarKind.INPUT:
        assert expression.input_name is not None and expression.input_index is not None
        encoded.update(
            {
                "input": expression.input_name,
                "index": {
                    "row_offset": expression.input_index.row_offset,
                    "feature_offset": expression.input_index.feature_offset,
                },
            }
        )
    elif expression.kind is CastScalarKind.CONSTANT:
        encoded["constant"] = expression.constant
    else:
        encoded["operands"] = [_encode_expression(operand) for operand in expression.operands]
    return encoded


def _render_cuda_expression(expression: CastScalarExpression) -> str:
    if expression.kind is CastScalarKind.INPUT:
        assert expression.input_name is not None
        return expression.input_name
    if expression.kind is CastScalarKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return "true" if expression.constant else "false"
        return _cuda_float(float(expression.constant))
    operands = tuple(_render_cuda_expression(operand) for operand in expression.operands)
    if expression.kind is CastScalarKind.CONVERT:
        if expression.dtype is CastScalarDType.PRED:
            raise ValueError("CUDA scalar conversion to predicate is unsupported")
        rendered = operands[0]
    elif expression.kind is CastScalarKind.NEGATE:
        rendered = f"(-{operands[0]})"
    elif expression.kind in {
        CastScalarKind.ADD,
        CastScalarKind.SUBTRACT,
        CastScalarKind.MULTIPLY,
        CastScalarKind.DIVIDE,
    }:
        intrinsic = {
            CastScalarKind.ADD: "__fadd_rn",
            CastScalarKind.SUBTRACT: "__fsub_rn",
            CastScalarKind.MULTIPLY: "__fmul_rn",
            CastScalarKind.DIVIDE: "__fdiv_rn",
        }[expression.kind]
        rendered = f"{intrinsic}({operands[0]}, {operands[1]})"
    elif expression.kind is CastScalarKind.EXP:
        rendered = f"expf({operands[0]})"
    elif expression.kind is CastScalarKind.TANH:
        rendered = f"tanhf({operands[0]})"
    elif expression.kind is CastScalarKind.SELECT:
        rendered = f"({operands[0]} ? {operands[1]} : {operands[2]})"
    else:
        raise AssertionError(f"unhandled scalar kind {expression.kind.value}")
    if expression.dtype is CastScalarDType.BF16:
        return f"__bfloat162float(__float2bfloat16_rn({rendered}))"
    return rendered


def _round_float32_to_bfloat16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
        return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF0000) | 0x00400000))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def _round_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _cuda_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CUDA scalar literals must be finite")
    rendered = repr(value)
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return f"{rendered}f"
