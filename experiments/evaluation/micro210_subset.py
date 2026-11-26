# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select benchmark rows without changing a frozen Micro210 protocol."""

import hashlib
import io
import json
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class Micro210Subset:
    benchmarks: tuple[str, ...] = ()
    row_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Micro210Selection:
    subset: Micro210Subset
    inputs: str


@dataclass(frozen=True)
class PreparedMicro210Subset:
    parquet: bytes
    record: dict


def prepared_subset(
    prepared_inputs: str, source_sha256: str, data_relative_path: str, subset: Micro210Subset
) -> PreparedMicro210Subset:
    """Resolve an allow-list against the original content-pinned prepared rows."""
    raw = (StoragePath(prepared_inputs) / data_relative_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    table = pq.read_table(io.BytesIO(raw))
    if subset.row_indices:
        indices = sorted(set(subset.row_indices))
        if subset.benchmarks or len(indices) != len(subset.row_indices):
            raise ValueError("Choose distinct prepared row indices or benchmarks, not both")
        if indices[0] < 0 or indices[-1] >= len(table):
            raise ValueError("Micro210 row selection is outside the frozen prepared inputs")
    else:
        indices = benchmark_indices(table, subset.benchmarks)
    selected = table.take(pa.array(indices, type=pa.int64()))
    if subset.row_indices:
        selected = selected.append_column("_source_uid", pa.array([str(index) for index in indices]))
    buffer = io.BytesIO()
    pq.write_table(selected, buffer)
    payload = buffer.getvalue()
    return PreparedMicro210Subset(
        payload,
        {
            "benchmarks": list(subset.benchmarks),
            "source_sha256": source_sha256,
            "selected_sha256": hashlib.sha256(payload).hexdigest(),
            "source_row_indices": indices,
            "attempts": len(indices),
        },
    )


def benchmark_indices(table: pa.Table, requested: tuple[str, ...]) -> list[int]:
    benchmarks = table.column("benchmark").to_pylist()
    allowed = set(requested)
    if not allowed or len(allowed) != len(requested):
        raise ValueError("Micro210 benchmark allow-list must be nonempty and contain no duplicates")
    unknown = allowed - set(benchmarks)
    if unknown:
        raise ValueError(f"Unknown frozen Micro210 benchmarks: {sorted(unknown)}")
    return [index for index, benchmark in enumerate(benchmarks) if benchmark in allowed]


def prepare_subset(
    *, prepared_inputs: str, source_sha256: str, data_relative_path: str, subset: Micro210Subset, output: str
) -> None:
    selected = prepared_subset(prepared_inputs, source_sha256, data_relative_path, subset)
    (StoragePath(output) / data_relative_path).write_bytes(selected.parquet)
    (StoragePath(output) / "selection.json").write_text(json.dumps(selected.record, indent=2) + "\n")


def validate_subset(
    *, prepared_inputs: str, selected_inputs: str, source_sha256: str, data_relative_path: str, subset: Micro210Subset
) -> dict:
    """Reject changed selections or task contents before allocating evaluation GPUs."""
    expected = prepared_subset(prepared_inputs, source_sha256, data_relative_path, subset)
    actual = (StoragePath(selected_inputs) / data_relative_path).read_bytes()
    record = json.loads((StoragePath(selected_inputs) / "selection.json").read_bytes())
    if actual != expected.parquet or record != expected.record:
        raise ValueError("Micro210 subset differs from the frozen prepared rows")
    return record
