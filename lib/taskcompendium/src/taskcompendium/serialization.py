# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON and nested Arrow records with lossless verifier parameters."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

import fsspec
import msgspec
import pyarrow as pa
import pyarrow.parquet as pq

from taskcompendium.models import SCHEMA_VERSION, Protocol, TaskSpecification


def to_json(specification: TaskSpecification) -> bytes:
    """Encode a canonical task document, independent of container serialization."""
    # Struct constructors accept integers for float fields. Normalize through the
    # typed decoder so JSON and Arrow round trips retain the same content hash.
    normalized = from_json(json.dumps(msgspec.to_builtins(specification), allow_nan=False))
    return json.dumps(msgspec.to_builtins(normalized), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def from_json(data: bytes | str) -> TaskSpecification:
    return msgspec.json.decode(data, type=TaskSpecification)


def protocol_from_json(data: bytes | str) -> Protocol:
    return msgspec.json.decode(data, type=Protocol)


def specification_hash(specification: TaskSpecification) -> str:
    return hashlib.sha256(to_json(specification)).hexdigest()


def json_schema() -> dict[str, Any]:
    return msgspec.json.schema(TaskSpecification)


_TEXT = pa.string()
_STRINGS = pa.list_(_TEXT)
_SOURCE = pa.struct([("dataset", _TEXT), ("revision", _TEXT), ("row", _TEXT), ("importer_revision", _TEXT)])
_ENVIRONMENT = pa.struct(
    [
        ("kind", _TEXT),
        ("image", _TEXT),
        ("workdir", _TEXT),
        ("max_steps", pa.int64()),
        ("max_output_bytes", pa.int64()),
        ("setup_commands", _STRINGS),
        ("additional_directories", _STRINGS),
    ]
)
_CONTENT = pa.struct([("kind", _TEXT), ("data", pa.large_string()), ("uri", _TEXT), ("sha256", _TEXT)])
_RESOURCE = pa.struct([("path", _TEXT), ("roles", _STRINGS), ("content", _CONTENT), ("executable", pa.bool_())])
_POLICY = pa.struct(
    [
        ("model", _TEXT),
        ("size_class", _TEXT),
        ("provider", _TEXT),
        ("base_url", _TEXT),
        ("samples", pa.int64()),
        ("aggregation", _TEXT),
        ("temperature", pa.float64()),
    ]
)
_VIEW = pa.struct([("transcript", pa.bool_()), ("files", _STRINGS), ("reference_context", _STRINGS)])
_JUDGE = pa.struct([("policy", _POLICY), ("view", _VIEW)])
# Ontology parameters can include arbitrary JSON schemas and per-mode constraint
# dictionaries. Encode individual values, preserving parameter names as Arrow map
# keys. The task, resources, environment, judge, and provenance stay nested columns.
_VERIFIER = pa.struct([("mode", _TEXT), ("parameters", pa.map_(_TEXT, pa.large_string())), ("judge", _JUDGE)])
_RUNTIME = pa.struct(
    [
        ("kind", _TEXT),
        ("image", _TEXT),
        ("timeout", pa.float64()),
        ("revision", _TEXT),
        ("workspace", pa.struct([("kind", _TEXT), ("preserved_directories", _STRINGS)])),
        ("supervisor_python", _TEXT),
    ]
)
ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", _TEXT),
        ("id", _TEXT),
        ("instructions", pa.large_string()),
        ("answer_requirements", pa.struct([("kind", _TEXT)])),
        ("environment", _ENVIRONMENT),
        ("resources", pa.list_(_RESOURCE)),
        ("verifier", _VERIFIER),
        ("verifier_runtime", _RUNTIME),
        ("metadata", pa.struct([("source", _SOURCE), ("competencies", _STRINGS), ("task_shape", _TEXT)])),
    ],
    metadata={b"taskcompendium.schema_version": SCHEMA_VERSION.encode()},
)


def _arrow_row(specification: TaskSpecification) -> dict[str, Any]:
    row = msgspec.to_builtins(specification)
    row["verifier"]["parameters"] = [
        (key, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        for key, value in sorted(row["verifier"]["parameters"].items())
    ]
    return row


def _without_padding(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_padding(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_padding(item) for item in value]
    return value


def write_parquet(specifications: Iterable[TaskSpecification], uri: str, batch_size: int = 256) -> int:
    """Stream self-contained task records to a local or fsspec destination."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = 0
    rows: list[dict[str, Any]] = []
    with fsspec.open(uri, "wb") as stream, pq.ParquetWriter(stream, ARROW_SCHEMA, compression="zstd") as writer:
        for specification in specifications:
            rows.append(_arrow_row(specification))
            if len(rows) == batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA))
                count += len(rows)
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA))
            count += len(rows)
    return count


def read_parquet(uri: str) -> Iterator[TaskSpecification]:
    """Read task records while restoring tagged variants and verifier value types."""
    with fsspec.open(uri, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        if not parquet.schema_arrow.equals(ARROW_SCHEMA) or parquet.schema_arrow.metadata != ARROW_SCHEMA.metadata:
            raise ValueError("Unsupported TaskCompendium Arrow schema")
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                row = _without_padding(row)
                # Decode after removing Arrow's inactive-variant padding: JSON null
                # inside a verifier parameter is meaningful and must survive.
                row["verifier"]["parameters"] = {key: json.loads(value) for key, value in row["verifier"]["parameters"]}
                yield msgspec.convert(row, type=TaskSpecification)
