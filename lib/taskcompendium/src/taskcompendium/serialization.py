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

from taskcompendium.models import SCHEMA_VERSION, Rendering, TaskSpecification


def to_json(specification: TaskSpecification) -> bytes:
    """Encode a canonical task document, independent of container serialization."""
    # Struct constructors accept integers for float fields. Normalize through the
    # typed decoder so JSON and Arrow round trips retain the same content hash.
    normalized = from_json(json.dumps(msgspec.to_builtins(specification), allow_nan=False))
    return json.dumps(msgspec.to_builtins(normalized), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def from_json(data: bytes | str) -> TaskSpecification:
    return msgspec.json.decode(data, type=TaskSpecification)


def rendering_from_json(data: bytes | str) -> Rendering:
    return msgspec.json.decode(data, type=Rendering)


def renderings_from_json(data: bytes | str) -> tuple[Rendering, ...]:
    return msgspec.json.decode(data, type=tuple[Rendering, ...])


def specification_hash(specification: TaskSpecification) -> str:
    return hashlib.sha256(to_json(specification)).hexdigest()


def json_schema() -> dict[str, Any]:
    return msgspec.json.schema(TaskSpecification)


_TEXT = pa.string()
_STRINGS = pa.list_(_TEXT)
_SOURCE = pa.struct([("dataset", _TEXT), ("revision", _TEXT), ("row", _TEXT), ("importer_revision", _TEXT)])
_STATE = pa.struct(
    [("image", _TEXT), ("workdir", _TEXT), ("setup_commands", _STRINGS), ("additional_directories", _STRINGS)]
)
_ACTION_INTERFACE = pa.struct([("name", _TEXT), ("version", _TEXT), ("seed_sha256", _TEXT)])
_REQUIREMENTS = pa.struct(
    [("capabilities", _STRINGS), ("state", _STATE), ("action_interfaces", pa.list_(_ACTION_INTERFACE))]
)
_CONTENT = pa.struct([("kind", _TEXT), ("data", pa.large_string()), ("uri", _TEXT), ("sha256", _TEXT)])
_RESOURCE = pa.struct([("path", _TEXT), ("roles", _STRINGS), ("content", _CONTENT), ("executable", pa.bool_())])
# Verifiers are a discriminated union in schema 0.6. Keep their independently
# versioned, private contracts as canonical JSON instead of padding unrelated
# source-specific variants into one universal Arrow ontology.
_VERIFIER = pa.large_string()
_STEP = pa.struct(
    [
        ("instructions", pa.large_string()),
        ("answer_requirements", pa.struct([("kind", _TEXT)])),
        ("resources", pa.list_(_RESOURCE)),
        ("verifier", _VERIFIER),
        ("context_requirement", _TEXT),
    ]
)
ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", _TEXT),
        ("id", _TEXT),
        ("steps", pa.list_(_STEP)),
        ("success_policy", _TEXT),
        ("requirements", _REQUIREMENTS),
        ("resources", pa.list_(_RESOURCE)),
        ("metadata", pa.struct([("source", _SOURCE), ("competencies", _STRINGS), ("task_shape", _TEXT)])),
    ],
    metadata={b"taskcompendium.schema_version": SCHEMA_VERSION.encode()},
)


def _arrow_row(specification: TaskSpecification) -> dict[str, Any]:
    row = msgspec.to_builtins(specification)
    for step in row["steps"]:
        step["verifier"] = json.dumps(step["verifier"], sort_keys=True, separators=(",", ":"), allow_nan=False)
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
                for step in row["steps"]:
                    step["verifier"] = json.loads(step["verifier"])
                yield msgspec.convert(row, type=TaskSpecification)
