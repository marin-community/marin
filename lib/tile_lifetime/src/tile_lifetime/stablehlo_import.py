# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable StableHLO bytecode importer."""

from dataclasses import dataclass

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from tile_lifetime.ir import DType

SUPPORTED_OPERATIONS = frozenset(
    {
        "stablehlo.add",
        "stablehlo.broadcast_in_dim",
        "stablehlo.constant",
        "stablehlo.convert",
        "stablehlo.compare",
        "stablehlo.composite",
        "stablehlo.concatenate",
        "stablehlo.divide",
        "stablehlo.dot_general",
        "stablehlo.exponential",
        "stablehlo.gather",
        "stablehlo.iota",
        "stablehlo.maximum",
        "stablehlo.multiply",
        "stablehlo.negate",
        "stablehlo.reduce",
        "stablehlo.reshape",
        "stablehlo.rsqrt",
        "stablehlo.select",
        "stablehlo.slice",
        "stablehlo.subtract",
        "stablehlo.transpose",
    }
)
STABLEHLO_DTYPES = {
    "i1": DType.BOOL,
    "bf16": DType.BF16,
    "f32": DType.FP32,
    "f64": DType.FP64,
    "i32": DType.INT32,
}


class StableHLOImportError(ValueError):
    """Raised when a portable module is outside the supported importer subset."""


@dataclass(frozen=True)
class NoAttributes:
    """Marker for an operation without semantic attributes."""


@dataclass(frozen=True)
class DotAttributes:
    """Batching and contracting dimensions of `dot_general`."""

    lhs_batching_dimensions: tuple[int, ...]
    rhs_batching_dimensions: tuple[int, ...]
    lhs_contracting_dimensions: tuple[int, ...]
    rhs_contracting_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class BroadcastAttributes:
    """Input-to-output dimension mapping for a broadcast."""

    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class ReductionAttributes:
    """Reduction dimensions and scalar combining operation."""

    dimensions: tuple[int, ...]
    reducer: str


@dataclass(frozen=True)
class ConstantAttributes:
    """Stable textual scalar or tensor literal."""

    literal: str


@dataclass(frozen=True)
class CompareAttributes:
    """Comparison direction and operand interpretation."""

    direction: str
    compare_type: str


@dataclass(frozen=True)
class IotaAttributes:
    """Output dimension varied by an iota."""

    dimension: int


@dataclass(frozen=True)
class TransposeAttributes:
    """Result-axis permutation of a transpose."""

    permutation: tuple[int, ...]


@dataclass(frozen=True)
class SliceAttributes:
    """Static bounds and strides of a slice."""

    start_indices: tuple[int, ...]
    limit_indices: tuple[int, ...]
    strides: tuple[int, ...]


@dataclass(frozen=True)
class ConcatenateAttributes:
    """Dimension along which operands are concatenated."""

    dimension: int


@dataclass(frozen=True)
class CompositeAttributes:
    """Named StableHLO composite with context-independent scalar attributes."""

    name: str
    attributes: tuple[tuple[str, str], ...]
    version: int


@dataclass(frozen=True)
class GatherAttributes:
    """Static gather mapping and slice sizes."""

    offset_dimensions: tuple[int, ...]
    collapsed_slice_dimensions: tuple[int, ...]
    start_index_map: tuple[int, ...]
    index_vector_dimension: int
    slice_sizes: tuple[int, ...]


StableHLOAttributes = (
    NoAttributes
    | DotAttributes
    | BroadcastAttributes
    | ReductionAttributes
    | ConstantAttributes
    | CompareAttributes
    | IotaAttributes
    | TransposeAttributes
    | SliceAttributes
    | ConcatenateAttributes
    | CompositeAttributes
    | GatherAttributes
)


@dataclass(frozen=True)
class StableHLOValue:
    """Context-independent value imported from StableHLO."""

    id: int
    name: str
    shape: tuple[int, ...]
    dtype: DType


@dataclass(frozen=True)
class StableHLOOperation:
    """One supported StableHLO operation."""

    id: int
    kind: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attributes: StableHLOAttributes
    source_location: str


@dataclass(frozen=True)
class StableHLOGraph:
    """StableHLO function body detached from the MLIR context."""

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    values: tuple[StableHLOValue, ...]
    operations: tuple[StableHLOOperation, ...]

    def value(self, value_id: int) -> StableHLOValue:
        """Return a value by stable integer identifier."""
        return self.values[value_id]

    def producer(self, value_id: int) -> StableHLOOperation | None:
        """Return the operation producing a value, if any."""
        return next((operation for operation in self.operations if value_id in operation.outputs), None)

    def consumers(self, value_id: int) -> tuple[StableHLOOperation, ...]:
        """Return operations consuming a value."""
        return tuple(operation for operation in self.operations if value_id in operation.inputs)


def import_stablehlo(artifact: bytes, *, input_names: tuple[str, ...] | None = None) -> StableHLOGraph:
    """Import one static, single-block function from portable StableHLO bytecode."""
    with ir.Context() as context:
        stablehlo.register_dialect(context)
        module = stablehlo.deserialize_portable_artifact(context, artifact)
        return _import_module(module, input_names=input_names)


def _import_module(module: ir.Module, *, input_names: tuple[str, ...] | None) -> StableHLOGraph:
    functions = tuple(
        operation
        for operation in module.body.operations
        if operation.operation.name == "func.func"
        and str(operation.operation.attributes.get("sym_visibility", '"public"')) != '"private"'
    )
    if len(functions) != 1:
        raise StableHLOImportError(f"expected one function, found {len(functions)}")

    function = functions[0]
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise StableHLOImportError("the entry function must contain one block")
    block = function.regions[0].blocks[0]
    if input_names is not None and len(input_names) != len(block.arguments):
        raise StableHLOImportError(
            f"received {len(input_names)} input names for a function with {len(block.arguments)} arguments"
        )

    values: list[StableHLOValue] = []
    value_ids: dict[ir.Value, int] = {}
    input_ids: list[int] = []
    for index, argument in enumerate(block.arguments):
        name = input_names[index] if input_names is not None else f"arg{index}"
        value = _import_value(argument, value_id=len(values), name=name)
        values.append(value)
        value_ids[argument] = value.id
        input_ids.append(value.id)

    operations: list[StableHLOOperation] = []
    output_ids: tuple[int, ...] | None = None
    for operation_view in block.operations:
        operation = operation_view.operation
        if operation.name == "func.return":
            output_ids = tuple(
                _lookup_value_id(value_ids, operand, operation=operation) for operand in operation.operands
            )
            continue
        if operation.name not in SUPPORTED_OPERATIONS:
            raise StableHLOImportError(f"unsupported operation {operation.name} at {operation.location}")

        imported_outputs: list[int] = []
        for result in operation.results:
            imported = _import_value(result, value_id=len(values), name=f"v{len(values)}")
            values.append(imported)
            value_ids[result] = imported.id
            imported_outputs.append(imported.id)

        operations.append(
            StableHLOOperation(
                id=len(operations),
                kind=operation.name.removeprefix("stablehlo."),
                inputs=tuple(
                    _lookup_value_id(value_ids, operand, operation=operation) for operand in operation.operands
                ),
                outputs=tuple(imported_outputs),
                attributes=_import_attributes(operation),
                source_location=str(operation.location),
            )
        )

    if output_ids is None:
        raise StableHLOImportError("entry function has no return operation")
    return StableHLOGraph(
        inputs=tuple(input_ids),
        outputs=output_ids,
        values=tuple(values),
        operations=tuple(operations),
    )


def _import_value(value: ir.Value, *, value_id: int, name: str) -> StableHLOValue:
    try:
        tensor_type = ir.RankedTensorType(value.type)
    except ValueError as error:
        raise StableHLOImportError(f"value {name} has unsupported non-ranked type {value.type}") from error
    shape = tuple(tensor_type.shape)
    if any(dimension < 0 for dimension in shape):
        raise StableHLOImportError(f"value {name} has unsupported dynamic shape {shape}")
    return StableHLOValue(
        id=value_id,
        name=name,
        shape=shape,
        dtype=_import_dtype(tensor_type.element_type),
    )


def _import_dtype(element_type: ir.Type) -> DType:
    dtype = str(element_type)
    try:
        return STABLEHLO_DTYPES[dtype]
    except KeyError as error:
        raise StableHLOImportError(f"unsupported element type {dtype}") from error


def _lookup_value_id(value_ids: dict[ir.Value, int], value: ir.Value, *, operation: ir.Operation) -> int:
    try:
        return value_ids[value]
    except KeyError as error:
        raise StableHLOImportError(f"operation {operation.name} uses a value outside the entry block") from error


def _import_attributes(operation: ir.Operation) -> StableHLOAttributes:
    if operation.name == "stablehlo.dot_general":
        dimensions = stablehlo.DotDimensionNumbers(operation.attributes["dot_dimension_numbers"])
        return DotAttributes(
            lhs_batching_dimensions=tuple(dimensions.lhs_batching_dimensions),
            rhs_batching_dimensions=tuple(dimensions.rhs_batching_dimensions),
            lhs_contracting_dimensions=tuple(dimensions.lhs_contracting_dimensions),
            rhs_contracting_dimensions=tuple(dimensions.rhs_contracting_dimensions),
        )
    if operation.name == "stablehlo.broadcast_in_dim":
        return BroadcastAttributes(dimensions=tuple(ir.DenseI64ArrayAttr(operation.attributes["broadcast_dimensions"])))
    if operation.name == "stablehlo.reduce":
        return ReductionAttributes(
            dimensions=tuple(ir.DenseI64ArrayAttr(operation.attributes["dimensions"])),
            reducer=_reducer_name(operation),
        )
    if operation.name == "stablehlo.constant":
        return ConstantAttributes(literal=str(operation.attributes["value"]))
    if operation.name == "stablehlo.compare":
        return CompareAttributes(
            direction=_stablehlo_enum_value(operation.attributes["comparison_direction"]),
            compare_type=_stablehlo_enum_value(operation.attributes["compare_type"]),
        )
    if operation.name == "stablehlo.iota":
        return IotaAttributes(dimension=ir.IntegerAttr(operation.attributes["iota_dimension"]).value)
    if operation.name == "stablehlo.transpose":
        return TransposeAttributes(permutation=tuple(ir.DenseI64ArrayAttr(operation.attributes["permutation"])))
    if operation.name == "stablehlo.slice":
        return SliceAttributes(
            start_indices=tuple(ir.DenseI64ArrayAttr(operation.attributes["start_indices"])),
            limit_indices=tuple(ir.DenseI64ArrayAttr(operation.attributes["limit_indices"])),
            strides=tuple(ir.DenseI64ArrayAttr(operation.attributes["strides"])),
        )
    if operation.name == "stablehlo.concatenate":
        return ConcatenateAttributes(dimension=ir.IntegerAttr(operation.attributes["dimension"]).value)
    if operation.name == "stablehlo.composite":
        attributes = ir.DictAttr(operation.attributes["composite_attributes"])
        return CompositeAttributes(
            name=ir.StringAttr(operation.attributes["name"]).value,
            attributes=tuple((str(attribute.name), str(attribute.attr)) for attribute in attributes),
            version=ir.IntegerAttr(operation.attributes["version"]).value,
        )
    if operation.name == "stablehlo.gather":
        dimensions = stablehlo.GatherDimensionNumbers(operation.attributes["dimension_numbers"])
        return GatherAttributes(
            offset_dimensions=tuple(dimensions.offset_dims),
            collapsed_slice_dimensions=tuple(dimensions.collapsed_slice_dims),
            start_index_map=tuple(dimensions.start_index_map),
            index_vector_dimension=dimensions.index_vector_dim,
            slice_sizes=tuple(ir.DenseI64ArrayAttr(operation.attributes["slice_sizes"])),
        )
    return NoAttributes()


def _stablehlo_enum_value(attribute: ir.Attribute) -> str:
    text = str(attribute)
    return text.removesuffix(">").rsplit(" ", maxsplit=1)[-1]


def _reducer_name(operation: ir.Operation) -> str:
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        raise StableHLOImportError(f"reduction at {operation.location} must contain one scalar block")
    body_operations = tuple(
        nested.operation.name
        for nested in operation.regions[0].blocks[0].operations
        if nested.operation.name != "stablehlo.return"
    )
    if len(body_operations) != 1 or not body_operations[0].startswith("stablehlo."):
        raise StableHLOImportError(f"unsupported reduction body {body_operations} at {operation.location}")
    return body_operations[0].removeprefix("stablehlo.")
