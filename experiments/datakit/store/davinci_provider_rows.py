# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the original daVinci ctx-native producer's fixed UTF-8 batch filter."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

HISTORICAL_BATCH_SIZE = 16
HISTORICAL_LOADER_REVISION = "4ba3b68b708dea9aca27acccb5f5b73097230375"
HISTORICAL_LOADER_FUNCTION_SHA256 = "2f9f9656781df5a7efb85bf4517d63c4b5834abd8d5547e8dd9a7b779b1ca321"


def davinci_ctx_reader_policy() -> dict:
    """Identify the filter introduced alongside the frozen ctx-native producer."""
    return {
        "loader_revision": HISTORICAL_LOADER_REVISION,
        "loader_function_sha256": HISTORICAL_LOADER_FUNCTION_SHA256,
        "batch_size": HISTORICAL_BATCH_SIZE,
        "columns": "all_original_columns",
        "conversion": "RecordBatch.to_pydict strict UTF8",
        "exclusion": "whole_original_batch_on_UnicodeDecodeError",
        "raw_bytes": "unchanged_original_object; invalid_value_SHA256_and_locator_recorded",
    }


@dataclass
class DavinciCtxReadAudit:
    provider_path: str = ""
    expected_rows: int = 0
    complete: bool = False
    total_rows: int = 0
    decoded_rows: int = 0
    excluded_batches: list[dict] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "policy": davinci_ctx_reader_policy(),
            "provider_path": self.provider_path,
            "expected_rows": self.expected_rows,
            "complete": self.complete,
            "total_rows": self.total_rows,
            "decoded_rows": self.decoded_rows,
            "excluded_rows": sum(batch["raw_row_count"] for batch in self.excluded_batches),
            "excluded_batches": self.excluded_batches,
        }


def _binary_string_type(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_string(dtype):
        return pa.binary()
    if pa.types.is_large_string(dtype):
        return pa.large_binary()
    if pa.types.is_list(dtype):
        return pa.list_(
            pa.field(dtype.value_field.name, _binary_string_type(dtype.value_type), nullable=dtype.value_field.nullable)
        )
    if pa.types.is_large_list(dtype):
        return pa.large_list(
            pa.field(dtype.value_field.name, _binary_string_type(dtype.value_type), nullable=dtype.value_field.nullable)
        )
    if pa.types.is_struct(dtype):
        return pa.struct(
            [pa.field(child.name, _binary_string_type(child.type), nullable=child.nullable) for child in dtype]
        )
    return dtype


def _invalid_string_values(value: Any, dtype: pa.DataType, location: str) -> Iterator[dict]:
    if value is None:
        return
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as error:
            yield {
                "field": location,
                "original_bytes": len(value),
                "original_bytes_sha256": hashlib.sha256(value).hexdigest(),
                "original_prefix_hex": value[:32].hex(),
                "byte_start": error.start,
                "byte_stop": error.end,
                "reason": error.reason,
            }
        return
    if pa.types.is_struct(dtype):
        for child in dtype:
            yield from _invalid_string_values(value[child.name], child.type, location + "." + child.name)
        return
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        for index, child in enumerate(value):
            yield from _invalid_string_values(child, dtype.value_type, f"{location}[{index}]")


def _batch_faults(batch: pa.RecordBatch, row_start: int) -> list[dict]:
    # String-to-binary casts preserve the original buffers without decoding,
    # allowing exact byte evidence even when Python string conversion fails.
    schema = pa.schema(
        [pa.field(column.name, _binary_string_type(column.type), nullable=column.nullable) for column in batch.schema]
    )
    rows = pa.Table.from_batches([batch]).cast(schema).to_pylist()
    faults = []
    for index, row in enumerate(rows):
        for column in batch.schema:
            for fault in _invalid_string_values(row[column.name], column.type, column.name):
                faults.append({"raw_row": row_start + index, **fault})
    if not faults:
        raise ValueError("Could not locate original bytes for a rejected UTF8 batch")
    return faults


def read_davinci_context(path: str, audit: DavinciCtxReadAudit) -> Iterator[dict[str, Any]]:
    """Replay full-column historical 16-row batches with explicit exclusion evidence.

    No replacement decoding or arbitrary row filtering is applied. Original
    malformed values remain in the pinned provider object, and each rejected
    batch records their exact row/field locations and byte hashes. Consumers
    must still require every selected canonical ID and occurrence downstream.
    """
    fs, key = filesystem_for(path)
    with fs.open(key, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        audit.provider_path = path
        audit.expected_rows = parquet.metadata.num_rows
        audit.complete = False
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=HISTORICAL_BATCH_SIZE)):
            row_start = audit.total_rows
            audit.total_rows += batch.num_rows
            try:
                rows = batch.to_pydict()
            except UnicodeDecodeError as error:
                audit.excluded_batches.append(
                    {
                        "provider_path": path,
                        "batch_index": batch_index,
                        "raw_row_start": row_start,
                        "raw_row_count": batch.num_rows,
                        "error": str(error),
                        "original_batch_arrow_ipc_sha256": hashlib.sha256(batch.serialize()).hexdigest(),
                        "faults": _batch_faults(batch, row_start),
                    }
                )
                continue
            audit.decoded_rows += batch.num_rows
            for index in range(batch.num_rows):
                yield {name: values[index] for name, values in rows.items()}
        excluded_rows = sum(batch["raw_row_count"] for batch in audit.excluded_batches)
        if audit.total_rows != audit.expected_rows or audit.decoded_rows + excluded_rows != audit.expected_rows:
            raise ValueError("Original daVinci file rows do not reconcile with its historical filter audit")
        audit.complete = True
