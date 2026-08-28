# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arrow batch interfaces used at Zephyr stage boundaries."""

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class ArrowBatch(Protocol):
    """A columnar batch exportable through Arrow's PyCapsule stream interface."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


def iter_record_batches(items: Iterable[object]) -> Iterator[pa.RecordBatch]:
    """Canonicalize Arrow-exportable stage items to record batches."""
    for item in items:
        if not isinstance(item, ArrowBatch):
            raise TypeError(
                "Columnar operations require Arrow-exportable batch items implementing "
                f"__arrow_c_stream__; got {type(item).__name__}."
            )
        yield from pa.RecordBatchReader.from_stream(item)


def _canonical_data_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_string_view(data_type):
        return pa.string()
    if pa.types.is_binary_view(data_type):
        return pa.binary()
    if pa.types.is_list(data_type):
        return pa.list_(_canonical_field(data_type.value_field))
    if pa.types.is_large_list(data_type):
        return pa.large_list(_canonical_field(data_type.value_field))
    if pa.types.is_fixed_size_list(data_type):
        return pa.list_(_canonical_field(data_type.value_field), data_type.list_size)
    if pa.types.is_struct(data_type):
        return pa.struct([_canonical_field(field) for field in data_type])
    if pa.types.is_map(data_type):
        return pa.map_(
            _canonical_data_type(data_type.key_type),
            _canonical_data_type(data_type.item_type),
            keys_sorted=data_type.keys_sorted,
        )
    return data_type


def _canonical_field(field: pa.Field) -> pa.Field:
    return pa.field(
        field.name,
        _canonical_data_type(field.type),
        nullable=field.nullable,
        metadata=field.metadata,
    )


def canonicalize_schema(schema: pa.Schema) -> pa.Schema:
    """Replace Arrow view types in a schema with portable Arrow types."""
    return pa.schema([_canonical_field(field) for field in schema], metadata=schema.metadata)


def canonicalize_record_batch(
    batch: pa.RecordBatch,
    schema: pa.Schema | None = None,
) -> pa.RecordBatch:
    """Cast a DataFusion batch to one stable, portable Arrow schema."""
    target = canonicalize_schema(schema or batch.schema)
    return batch if target.equals(batch.schema, check_metadata=True) else batch.cast(target)
