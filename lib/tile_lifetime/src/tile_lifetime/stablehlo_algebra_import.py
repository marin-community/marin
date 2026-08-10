# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import StableHLO one operation at a time into generic Shuttle algebra.

This importer does not form workload regions. Every source operation becomes
exactly one Contract, Map, Fold, DomainRestriction, or Relation node in source
order. Source value identities, dependencies, shapes, dtypes, casts, and view
operations remain available to later generic scheduling passes.
"""

from dataclasses import dataclass

from tile_lifetime.ir import DType
from tile_lifetime.stablehlo_import import StableHLOAttributes, StableHLOGraph, StableHLOOperation


@dataclass(frozen=True)
class ImportedAlgebraValue:
    """One source StableHLO value retained by the generic algebra importer."""

    source_value_id: int
    name: str
    shape: tuple[int, ...]
    dtype: DType
    producer_node_id: int | None
    consumer_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ImportedAlgebraNode:
    """Common source-order and dependency record for one imported operation."""

    source_operation_id: int
    source_index: int
    source_kind: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attributes: StableHLOAttributes
    source_location: str


@dataclass(frozen=True)
class ImportedContractNode(ImportedAlgebraNode):
    """A generic multilinear contraction imported from ``dot_general``."""


@dataclass(frozen=True)
class ImportedMapNode(ImportedAlgebraNode):
    """A local scalar, cast, broadcast, reshape, or transpose operation."""


@dataclass(frozen=True)
class ImportedFoldNode(ImportedAlgebraNode):
    """A generic source reduction with its reducer attributes intact."""


@dataclass(frozen=True)
class ImportedDomainRestrictionNode(ImportedAlgebraNode):
    """A predicate-controlled source domain/value restriction."""


@dataclass(frozen=True)
class ImportedRelationNode(ImportedAlgebraNode):
    """A source indexing relation such as a gather."""


ImportedAlgebraOperation = (
    ImportedContractNode | ImportedMapNode | ImportedFoldNode | ImportedDomainRestrictionNode | ImportedRelationNode
)


@dataclass(frozen=True)
class ImportedStableHLOAlgebra:
    """A lossless generic algebra view over one flat StableHLO graph."""

    source_graph: StableHLOGraph
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    values: tuple[ImportedAlgebraValue, ...]
    operations: tuple[ImportedAlgebraOperation, ...]

    def value(self, source_value_id: int) -> ImportedAlgebraValue:
        """Return one imported value by its StableHLO identity."""
        matches = tuple(value for value in self.values if value.source_value_id == source_value_id)
        if len(matches) != 1:
            raise KeyError(f"expected one imported value {source_value_id}, found {len(matches)}")
        return matches[0]

    def producer(self, source_value_id: int) -> ImportedAlgebraOperation | None:
        """Return the generic node producing a value."""
        producer_id = self.value(source_value_id).producer_node_id
        if producer_id is None:
            return None
        return self.operation(producer_id)

    def consumers(self, source_value_id: int) -> tuple[ImportedAlgebraOperation, ...]:
        """Return generic nodes consuming a value in source order."""
        return tuple(self.operation(node_id) for node_id in self.value(source_value_id).consumer_node_ids)

    def operation(self, source_operation_id: int) -> ImportedAlgebraOperation:
        """Return one generic node by source operation identity."""
        matches = tuple(
            operation for operation in self.operations if operation.source_operation_id == source_operation_id
        )
        if len(matches) != 1:
            raise KeyError(f"expected one imported operation {source_operation_id}, found {len(matches)}")
        return matches[0]


def import_stablehlo_algebra(graph: StableHLOGraph) -> ImportedStableHLOAlgebra:
    """Import every StableHLO operation without workload recognition."""
    operations = tuple(
        _import_operation(operation, source_index=index) for index, operation in enumerate(graph.operations)
    )
    values = tuple(
        ImportedAlgebraValue(
            source_value_id=value.id,
            name=value.name,
            shape=value.shape,
            dtype=value.dtype,
            producer_node_id=(producer.id if (producer := graph.producer(value.id)) is not None else None),
            consumer_node_ids=tuple(consumer.id for consumer in graph.consumers(value.id)),
        )
        for value in graph.values
    )
    imported = ImportedStableHLOAlgebra(
        source_graph=graph,
        inputs=graph.inputs,
        outputs=graph.outputs,
        values=values,
        operations=operations,
    )
    verify_imported_stablehlo_algebra(imported)
    return imported


def verify_imported_stablehlo_algebra(imported: ImportedStableHLOAlgebra) -> ImportedStableHLOAlgebra:
    """Verify one-to-one operation coverage and exact source dependencies."""
    graph = imported.source_graph
    if imported.inputs != graph.inputs or imported.outputs != graph.outputs:
        raise ValueError("generic algebra import changed the function boundary")
    if tuple(value.source_value_id for value in imported.values) != tuple(value.id for value in graph.values):
        raise ValueError("generic algebra import changed source value identity or order")
    if tuple(operation.source_operation_id for operation in imported.operations) != tuple(
        operation.id for operation in graph.operations
    ):
        raise ValueError("generic algebra import changed source operation identity or order")
    for source, operation in zip(graph.operations, imported.operations, strict=True):
        if (
            operation.source_kind != source.kind
            or operation.inputs != source.inputs
            or operation.outputs != source.outputs
            or operation.attributes != source.attributes
            or operation.source_location != source.source_location
        ):
            raise ValueError(f"generic algebra import changed operation {source.id}")
    for source, value in zip(graph.values, imported.values, strict=True):
        producer = graph.producer(source.id)
        expected_producer = producer.id if producer is not None else None
        expected_consumers = tuple(consumer.id for consumer in graph.consumers(source.id))
        if (
            value.name != source.name
            or value.shape != source.shape
            or value.dtype is not source.dtype
            or value.producer_node_id != expected_producer
            or value.consumer_node_ids != expected_consumers
        ):
            raise ValueError(f"generic algebra import changed value {source.id}")
    return imported


def _import_operation(operation: StableHLOOperation, *, source_index: int) -> ImportedAlgebraOperation:
    common = {
        "source_operation_id": operation.id,
        "source_index": source_index,
        "source_kind": operation.kind,
        "inputs": operation.inputs,
        "outputs": operation.outputs,
        "attributes": operation.attributes,
        "source_location": operation.source_location,
    }
    if operation.kind == "dot_general":
        return ImportedContractNode(**common)
    if operation.kind == "reduce":
        return ImportedFoldNode(**common)
    if operation.kind == "select":
        return ImportedDomainRestrictionNode(**common)
    if operation.kind == "gather":
        return ImportedRelationNode(**common)
    return ImportedMapNode(**common)
