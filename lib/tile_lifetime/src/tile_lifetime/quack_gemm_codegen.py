# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate QuACK/CuTe GEMM programs from backend-neutral tile dataflow."""

from __future__ import annotations

import hashlib
import keyword
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.gemm_program import GemmProgram
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind, deserialize_scalar_expression
from tile_lifetime.tile_program import TileOp, TilePrimitive, TileProgramError, TileProgramStage


class QuackOperandKind(StrEnum):
    """Physical delivery mechanism for an external tile-program value."""

    TILE = "tile"
    ROW = "row"
    COLUMN = "column"
    PAIR_COEFFICIENT_TILE = "pair_coefficient_tile"


@dataclass(frozen=True)
class QuackOperand:
    """One generated function argument and its logical source values."""

    parameter: str
    sources: tuple[str, ...]
    kind: QuackOperandKind
    stage: TileProgramStage


@dataclass(frozen=True)
class QuackOutput:
    """One auxiliary output emitted by the generated epilogue."""

    parameter: str
    destination: str
    reduction: bool


@dataclass(frozen=True)
class GeneratedQuackGemm:
    """Canonical importable source plus the runtime bindings it expects."""

    source: str
    digest: str
    operands: tuple[QuackOperand, ...]
    outputs: tuple[QuackOutput, ...]
    c_source: str | None
    has_transform: bool
    writes_main_output: bool


def generate_quack_gemm(program: GemmProgram) -> GeneratedQuackGemm:
    """Compile a supported tile dataflow into plain QuACK authoring functions.

    The generated module contains no Transformer or skeleton name. QuACK's
    reusable GEMM mainloop traces these functions into the A-fragment path and
    accumulator epilogue.
    """
    builder = _SourceBuilder(program)
    source = builder.source()
    return GeneratedQuackGemm(
        source=source,
        digest=hashlib.sha256(source.encode()).hexdigest(),
        operands=tuple(builder.operands),
        outputs=tuple(builder.outputs),
        c_source=builder.c_source,
        has_transform=builder.has_transform,
        writes_main_output=builder.writes_main_output,
    )


class _SourceBuilder:
    def __init__(self, program: GemmProgram) -> None:
        self.program = program
        self.operands: list[QuackOperand] = []
        self.outputs: list[QuackOutput] = []
        self._operand_by_sources: dict[tuple[tuple[str, ...], TileProgramStage], str] = {}
        self.c_source: str | None = None
        self.has_transform = bool(program.preparation)
        self.writes_main_output = True

    def source(self) -> str:
        transform = self._transform_source()
        epilogue = self._epilogue_source()
        imports = (
            "import cutlass.cute as cute\n"
            "from quack.epilogue import gemm_epilogue, pack, unpack\n"
            "from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad\n"
            "from quack.operand_transform import a_transform\n"
        )
        if any(
            operation.primitive is TilePrimitive.PAIRWISE_SWIGLU
            for operation in (*self.program.preparation, *self.program.finalization)
        ):
            imports = f"from quack.activation import swiglu\n{imports}"
        return f"{imports}\n{transform}{epilogue}"

    def _transform_source(self) -> str:
        operations = tuple(
            operation
            for operation in self.program.preparation
            if operation.primitive not in {TilePrimitive.CONVERT, TilePrimitive.VIEW}
        )
        if not operations:
            self.has_transform = False
            return "generated_transform = None\n\n"
        expressions = {self.program.input: "activation"}
        body: list[str] = []
        uses = Counter(value for operation in operations for value in operation.inputs)
        for operation in operations:
            if operation.primitive is TilePrimitive.VIEW:
                expressions[operation.outputs[0]] = self._expression(expressions, operation.inputs[0], True)
                continue
            if operation.primitive is TilePrimitive.SCALE_ROW:
                left = self._expression(expressions, operation.inputs[0], True)
                right = self._external((operation.inputs[1],), QuackOperandKind.COLUMN, TileProgramStage.PREPARATION)
                expression = f"{left} * {right}"
            elif operation.primitive in {TilePrimitive.ADD, TilePrimitive.MULTIPLY}:
                left = self._expression(expressions, operation.inputs[0], True)
                right = self._expression(expressions, operation.inputs[1], True)
                symbol = "+" if operation.primitive is TilePrimitive.ADD else "*"
                expression = f"{left} {symbol} {right}"
            else:
                raise TileProgramError(f"QuACK A-fragment code generation does not support {operation.primitive.value}")
            rendered = expression
            if any(uses[output] > 1 for output in operation.outputs):
                rendered = f"value_{len(body)}"
                body.append(f"    {rendered} = {expression}")
            for output in operation.outputs:
                expressions[output] = rendered
        result = expressions.get(self.program.mainloop_input.removesuffix(".mainloop_bf16"))
        if result is None:
            result = next(reversed(expressions.values()))
        transform_operands = [operand for operand in self.operands if operand.kind is QuackOperandKind.COLUMN]
        args = ", ".join(operand.parameter for operand in transform_operands)
        parameters = f", {args}" if args else ""
        kinds = ", ".join(f"{operand.parameter!r}: 'colvec_ktile_fp32'" for operand in transform_operands)
        return (
            f"@a_transform(vec_size=8, args={{{kinds}}})\n"
            f"def generated_transform(activation{parameters}):\n" + "\n".join(body) + f"\n    return {result}\n\n"
        )

    def _epilogue_source(self) -> str:
        operations = self.program.finalization
        if not operations:
            return "@gemm_epilogue()\ndef generated_epilogue(acc):\n    return {'D': acc}\n"
        if any(
            operation.primitive in {TilePrimitive.PAIRWISE_LINEAR_MAP, TilePrimitive.PAIRWISE_ROPE}
            for operation in operations
        ):
            return self._rotary_epilogue_source(operations)

        expressions: dict[str, str] = {}
        body: list[str] = []
        accumulator_bound = False
        reduction_values: dict[str, str] = {}
        store_values: dict[str, str] = {}
        pairwise = False
        uses = Counter(value for operation in operations for value in operation.inputs)
        for operation in operations:
            primitive = operation.primitive
            if primitive is TilePrimitive.STORE:
                destination = dict(operation.attributes)["destination"]
                store_values[destination] = self._expression(expressions, operation.inputs[0], False)
                continue
            if primitive is TilePrimitive.CONVERT:
                expressions[operation.outputs[0]] = self._expression(expressions, operation.inputs[0], False)
                continue
            if primitive is TilePrimitive.VIEW:
                expressions[operation.outputs[0]] = self._expression(expressions, operation.inputs[0], False)
                continue
            inputs = list(operation.inputs)
            if inputs and inputs[0] not in expressions and not accumulator_bound:
                expressions[inputs[0]] = "acc"
                accumulator_bound = True
            if primitive is TilePrimitive.RESIDUAL_ADD:
                left = self._expression(expressions, inputs[0], False)
                right = self._residual_c(inputs[1])
                expression = f"{left} + {right}"
            elif primitive is TilePrimitive.MULTIPLY_GAMMA:
                left = self._expression(expressions, inputs[0], False)
                right = self._external((inputs[1],), QuackOperandKind.ROW, TileProgramStage.FINALIZATION)
                expression = f"{left} * {right}"
            elif primitive is TilePrimitive.SCALE_ROW:
                left = self._expression(expressions, inputs[0], False)
                right = self._external((inputs[1],), QuackOperandKind.COLUMN, TileProgramStage.FINALIZATION)
                expression = f"{left} * {right}"
            elif primitive in {TilePrimitive.ADD, TilePrimitive.MULTIPLY}:
                left = self._expression(expressions, inputs[0], False)
                delivery = dict(operation.attributes).get("input.1_delivery", "tile")
                if delivery == "row":
                    right = self._external((inputs[1],), QuackOperandKind.ROW, TileProgramStage.FINALIZATION)
                elif delivery == "tile":
                    right = self._expression(expressions, inputs[1], False)
                else:
                    raise TileProgramError(f"unsupported finalization operand delivery {delivery!r}")
                symbol = "+" if primitive is TilePrimitive.ADD else "*"
                expression = f"{left} {symbol} {right}"
            elif primitive is TilePrimitive.PARTIAL_SUM_SQUARE:
                value = self._expression(expressions, inputs[0], False)
                reduction_values[operation.outputs[0]] = value
                # The unreduced value commonly continues on a residual path.
                # Emit it as an auxiliary tile without requiring region-level
                # consumer names in the generated function.
                if inputs[0] != self.program.output:
                    store_values.setdefault(inputs[0], value)
                expressions[operation.outputs[0]] = value
                continue
            elif primitive is TilePrimitive.PAIRWISE_SWIGLU:
                pairwise = True
                value = self._expression(expressions, inputs[0], False)
                gate = f"gate_{len(body)}"
                up = f"up_{len(body)}"
                body.append(f"    {gate}, {up} = unpack({value})")
                expression = f"swiglu({gate}, {up})"
            elif primitive is TilePrimitive.PAIRWISE_MAP:
                pairwise = True
                value = self._expression(expressions, inputs[0], False)
                left = f"left_{len(body)}"
                right = f"right_{len(body)}"
                body.append(f"    {left}, {right} = unpack({value})")
                serialized = dict(operation.attributes).get("expression_ast")
                if serialized is None:
                    raise TileProgramError("generic adjacent-pair Map requires a serialized scalar AST")
                expression = _scalar_source(
                    deserialize_scalar_expression(serialized),
                    {"pair.left": left, "pair.right": right},
                )
            else:
                raise TileProgramError(f"QuACK epilogue code generation does not support {primitive.value}")
            rendered = expression
            if primitive in {TilePrimitive.PAIRWISE_MAP, TilePrimitive.PAIRWISE_SWIGLU} or any(
                uses[output] > 1 for output in operation.outputs
            ):
                rendered = f"value_{len(body)}"
                body.append(f"    {rendered} = {expression}")
            for output in operation.outputs:
                expressions[output] = rendered

        return self._render_epilogue(body, store_values, reduction_values, pairwise=pairwise)

    def _rotary_epilogue_source(self, operations: tuple[TileOp, ...]) -> str:
        expressions: dict[str, str] = {}
        body: list[str] = []
        accumulator_value = "acc"
        for operation in operations:
            if operation.primitive is TilePrimitive.SCALE_ROW:
                if operation.inputs[0] not in expressions:
                    expressions[operation.inputs[0]] = accumulator_value
                scale = self._external((operation.inputs[1],), QuackOperandKind.COLUMN, TileProgramStage.FINALIZATION)
                variable = f"value_{len(body)}"
                body.append(f"    {variable} = {expressions[operation.inputs[0]]} * {scale}")
                expressions[operation.outputs[0]] = variable
                accumulator_value = variable
            elif operation.primitive is TilePrimitive.PARTITION:
                if operation.inputs[0] in expressions:
                    accumulator_value = expressions[operation.inputs[0]]
                for output in operation.outputs:
                    expressions[output] = accumulator_value
        linear_maps = tuple(
            operation
            for operation in operations
            if operation.primitive in {TilePrimitive.PAIRWISE_LINEAR_MAP, TilePrimitive.PAIRWISE_ROPE}
        )
        if not linear_maps:
            raise TileProgramError("pairwise-linear epilogue requires at least one map")
        sources = (linear_maps[0].inputs[1], linear_maps[0].inputs[2])
        if any(operation.inputs[1:] != sources for operation in linear_maps):
            raise TileProgramError("one generated pairwise-linear epilogue requires shared coefficient tables")
        table = self._external(sources, QuackOperandKind.PAIR_COEFFICIENT_TILE, TileProgramStage.FINALIZATION)
        body.extend(
            (
                f"    pair_left, pair_right = unpack({accumulator_value})",
                f"    coefficient_0, coefficient_1 = unpack({table})",
            )
        )
        aliases = {
            "pair.left": "pair_left",
            "pair.right": "pair_right",
            "coefficient.0": "coefficient_0",
            "coefficient.1": "coefficient_1",
        }
        if all(operation.primitive is TilePrimitive.PAIRWISE_ROPE for operation in linear_maps):
            output_expressions = (
                "pair_left * coefficient_0 - pair_right * coefficient_1",
                "pair_left * coefficient_1 + pair_right * coefficient_0",
            )
        else:
            serialized_expressions = tuple(
                dict(linear_maps[0].attributes).get(f"expression_ast.{index}") for index in range(2)
            )
            if any(serialized is None for serialized in serialized_expressions):
                raise TileProgramError("generic pairwise-linear Map requires two serialized scalar ASTs")
            if any(
                tuple(dict(operation.attributes).get(f"expression_ast.{index}") for index in range(2))
                != serialized_expressions
                for operation in linear_maps
            ):
                raise TileProgramError("one generated pairwise-linear epilogue requires shared scalar expressions")
            output_expressions = tuple(
                _scalar_source(deserialize_scalar_expression(serialized), aliases)
                for serialized in serialized_expressions
                if serialized is not None
            )
        body.append(f"    mapped = pack({output_expressions[0]}, {output_expressions[1]})")
        decorator_ops = self._decorator_ops()
        parameters = self._epilogue_parameters()
        return (
            f"@gemm_epilogue(ops={{{decorator_ops}}}, mode='acc_pair')\n"
            f"def generated_epilogue(acc{parameters}):\n" + "\n".join(body) + "\n    return {'D': mapped}\n"
        )

    def _render_epilogue(
        self,
        body: list[str],
        store_values: dict[str, str],
        reduction_values: dict[str, str],
        *,
        pairwise: bool,
    ) -> str:
        outputs: list[str] = []
        main_returns: list[str] = []
        auxiliary_returns: list[str] = []
        if pairwise:
            self.writes_main_output = False
        for destination, value in store_values.items():
            if destination in reduction_values:
                continue
            if destination == self.program.output and not pairwise:
                main_returns.append(f"'D': {value}")
                continue
            parameter = f"output_{len(outputs)}"
            outputs.append(parameter)
            self.outputs.append(QuackOutput(parameter, destination, reduction=False))
            auxiliary_returns.append(f"{parameter!r}: {value}")
        reduction_entries: list[str] = []
        for destination, value in reduction_values.items():
            parameter = f"reduction_{len(reduction_entries)}"
            reduction_entries.append(f"{parameter!r}: ColVecReduce({parameter!r}, scaled=True)")
            self.outputs.append(QuackOutput(parameter, destination, reduction=True))
            auxiliary_returns.append(f"{parameter!r}: ({value}, {value})")
        returns = [*main_returns, *auxiliary_returns]
        if not returns:
            returns.append("'D': acc")
        output_text = repr(tuple(outputs))
        reductions = ", ".join(reduction_entries)
        decorator_ops = self._decorator_ops()
        options = [f"outputs={output_text}"]
        if reductions:
            options.append(f"reduces={{{reductions}}}")
        if decorator_ops:
            options.append(f"ops={{{decorator_ops}}}")
        if pairwise:
            options.append("mode='acc_pair'")
        parameters = self._epilogue_parameters()
        rendered_body = "\n".join(body) if body else "    pass"
        return (
            f"@gemm_epilogue({', '.join(options)})\n"
            f"def generated_epilogue(acc{parameters}):\n"
            f"{rendered_body}\n"
            f"    return {{{', '.join(returns)}}}\n"
        )

    def _expression(self, expressions: dict[str, str], value: str, preparation: bool) -> str:
        expression = expressions.get(value)
        if expression is not None:
            return expression
        kind = QuackOperandKind.TILE
        if preparation:
            raise TileProgramError(f"unclassified external A-transform value {value!r}")
        expression = self._external((value,), kind, TileProgramStage.FINALIZATION)
        expressions[value] = expression
        return expression

    def _external(
        self,
        sources: tuple[str, ...],
        kind: QuackOperandKind,
        stage: TileProgramStage,
    ) -> str:
        key = (sources, stage)
        existing = self._operand_by_sources.get(key)
        if existing is not None:
            matching = next(operand for operand in self.operands if operand.parameter == existing)
            if matching.kind is not kind or matching.stage is not stage:
                raise TileProgramError(f"logical operand {sources!r} requested with incompatible delivery kinds")
            return existing
        parameter = f"operand_{len(self.operands)}"
        self._operand_by_sources[key] = parameter
        self.operands.append(QuackOperand(parameter, sources, kind, stage))
        return parameter

    def _transform_lines(self) -> tuple[str, ...]:
        return tuple(operand.parameter for operand in self.operands if operand.stage is TileProgramStage.PREPARATION)

    def _epilogue_parameters(self) -> str:
        transform_parameters = set(self._transform_lines())
        parameters = [operand.parameter for operand in self.operands if operand.parameter not in transform_parameters]
        if self.c_source is not None:
            parameters.insert(0, "c")
        return f", {', '.join(parameters)}" if parameters else ""

    def _residual_c(self, source: str) -> str:
        if self.c_source is not None and self.c_source != source:
            raise TileProgramError("one generated QuACK epilogue supports one dedicated residual C operand")
        self.c_source = source
        return "c"

    def _decorator_ops(self) -> str:
        transform_parameters = set(self._transform_lines())
        entries = []
        for operand in self.operands:
            if operand.parameter in transform_parameters:
                continue
            if operand.kind is QuackOperandKind.TILE:
                expression = f"TileLoad({operand.parameter!r})"
            elif operand.kind is QuackOperandKind.ROW:
                expression = f"RowVecLoad({operand.parameter!r})"
            elif operand.kind is QuackOperandKind.COLUMN:
                expression = f"ColVecLoad({operand.parameter!r})"
            elif operand.kind is QuackOperandKind.PAIR_COEFFICIENT_TILE:
                expression = f"TileLoad({operand.parameter!r})"
            else:
                raise AssertionError(operand.kind)
            entries.append(f"{operand.parameter!r}: {expression}")
        return ", ".join(entries)


def _scalar_source(expression: ScalarExpression, aliases: dict[str, str], parent_precedence: int = 0) -> str:
    """Render a scalar AST directly into the QuACK/CuTe authoring function."""
    if expression.kind is ScalarExpressionKind.INPUT:
        if expression.input_name not in aliases:
            raise TileProgramError(f"unbound scalar-expression input {expression.input_name!r}")
        return aliases[expression.input_name]
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return repr(expression.constant)
    if expression.kind in {ScalarExpressionKind.EXP, ScalarExpressionKind.RSQRT, ScalarExpressionKind.TANH}:
        function = {
            ScalarExpressionKind.EXP: "cute.exp",
            ScalarExpressionKind.RSQRT: "cute.rsqrt",
            ScalarExpressionKind.TANH: "cute.tanh",
        }[expression.kind]
        return f"{function}({_scalar_source(expression.operands[0], aliases)})"
    if expression.kind is ScalarExpressionKind.SELECT:
        predicate, when_true, when_false = expression.operands
        return (
            f"cute.where({_scalar_source(predicate, aliases)}, {_scalar_source(when_true, aliases)}, "
            f"{_scalar_source(when_false, aliases)})"
        )
    symbols = {
        ScalarExpressionKind.ADD: ("+", 1),
        ScalarExpressionKind.SUBTRACT: ("-", 1),
        ScalarExpressionKind.MULTIPLY: ("*", 2),
        ScalarExpressionKind.DIVIDE: ("/", 2),
        ScalarExpressionKind.LESS_EQUAL: ("<=", 0),
    }
    try:
        symbol, precedence = symbols[expression.kind]
    except KeyError as error:
        raise TileProgramError(f"unsupported scalar-expression operation {expression.kind.value}") from error
    left = _scalar_source(expression.operands[0], aliases, precedence)
    right = _scalar_source(expression.operands[1], aliases, precedence + 1)
    rendered = f"{left} {symbol} {right}"
    return f"({rendered})" if precedence < parent_precedence else rendered


def safe_module_name(digest: str) -> str:
    """Return a stable Python module name for one generated source digest."""
    candidate = f"shuttle_quack_{digest}"
    candidate = re.sub(r"\W", "_", candidate)
    if keyword.iskeyword(candidate):
        candidate = f"generated_{candidate}"
    return candidate
