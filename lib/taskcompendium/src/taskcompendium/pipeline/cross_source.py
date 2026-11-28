# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Campaign-level cross-source deduplication over retained populations.

After a campaign's sources complete, their ``final/`` views can still hold the
same task contract in more than one source. This pass groups the retained rows
across sources by the shared deduplication key
(:func:`taskcompendium.pipeline.fingerprints.deduplication_key`, the same
semantics the selection step applies when it samples) and keeps the first
source in campaign order. Every later duplicate becomes a typed drop recorded
in that source's own drop manifest, so the retained population a TaskStore or
selection reads can exclude cross-source duplicates without touching the
per-source artifacts.

Outputs keep one drop manifest per source — an empty file when the source has
no cross-source duplicates, because consumers resolve every source to a
manifest before reading — plus a campaign ``manifest.json`` with the per-source
evidence.

The recorded drops are the curation-side evidence; the selection step remains
the enforcement point when it samples, and because dropped copies are
contract-identical to their kept ID, a different surviving copy in selection
is not a divergence.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.fingerprints import deduplication_key

CROSS_SOURCE_REVISION = "1"
CROSS_SOURCE_DUPLICATE = "DUPLICATE_CROSS_SOURCE"
DROP_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("kept_id", pa.string()),
        ("kept_source_id", pa.string()),
        ("reason", pa.string()),
    ]
)
MANIFEST_FILENAME = "manifest.json"
_ROWS_PER_SHARD = 1000
_MAX_SHARDS = 64


@dataclass(frozen=True)
class RetainedSource:
    """One completed source's retained population and its position in the dedup order."""

    source_id: str
    final_path: str
    order: int


class RetainedRow(NamedTuple):
    """One retained task reduced to its deduplication identity."""

    key: str
    source_id: str
    order: int
    task_id: str


class CrossSourceDrop(NamedTuple):
    """One later source's copy of a contract another source already retains."""

    task_id: str
    source_id: str
    kept_id: str
    kept_source_id: str
    reason: str


@dataclass(frozen=True)
class SourceCrossSourceEvidence:
    """One source's cross-source deduplication outcome."""

    source_id: str
    retained_rows: int
    cross_source_duplicates: int
    drop_manifest_uri: str


@dataclass(frozen=True)
class CrossSourceEvidence:
    """The campaign's cross-source deduplication outcome, keyed by source ID."""

    revision: str
    sources: dict[str, SourceCrossSourceEvidence]
    cross_source_duplicates: int
    drop_root: str


def _safe_slug(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", source_id)


def _final_files(final_path: str) -> list[str]:
    return sorted(str(path) for path in (StoragePath(final_path) / "*.parquet").glob())


def retained_source_rows(source: RetainedSource) -> Iterator[RetainedRow]:
    """Load one source's retained rows as deduplication records with their dedup order."""
    for file in _final_files(source.final_path):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["task_id", "task_json"]):
                for task_id, task_json in zip(
                    batch.column("task_id").to_pylist(), batch.column("task_json").to_pylist(), strict=True
                ):
                    task = TaskSpec.model_validate_json(task_json)
                    yield RetainedRow(
                        key=deduplication_key(task),
                        source_id=source.source_id,
                        order=source.order,
                        task_id=task_id,
                    )


def decide_duplicate(_key: str, rows: Iterator[RetainedRow]) -> Iterator[CrossSourceDrop]:
    """Reduce one contract group to its cross-source drops; groups arrive complete.

    The earliest dedup order wins, ties broken by task ID. The winner stays in
    its own source's ``final/`` view; every later member yields a drop.
    """
    ordered = sorted(rows, key=lambda row: (row.order, row.task_id))
    winner = ordered[0]
    for row in ordered[1:]:
        yield CrossSourceDrop(
            task_id=row.task_id,
            source_id=row.source_id,
            kept_id=winner.task_id,
            kept_source_id=winner.source_id,
            reason=CROSS_SOURCE_DUPLICATE,
        )


def count_retained_rows(final_path: str) -> int:
    """The retained row count of one source's ``final/`` view, from parquet metadata."""
    total = 0
    for file in _final_files(final_path):
        with StoragePath(file).open("rb") as stream:
            total += pq.ParquetFile(stream).metadata.num_rows
    return total


def write_drop_manifest(path: str, drops: list[CrossSourceDrop]) -> None:
    """Write one source's cross-source drop manifest; empty sources get an empty file."""
    rows = [drop._asdict() for drop in drops]
    with StoragePath(path).open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=DROP_SCHEMA), stream, compression="zstd")


def deduplicate_across_sources(
    sources: list[RetainedSource],
    output_path: str,
    *,
    context: ZephyrContext,
    telemetry: PhaseTelemetry | None = None,
) -> CrossSourceEvidence:
    """Drop later sources' copies of a contract another retained source already holds."""
    if not sources:
        raise ValueError("Cross-source deduplication requires at least one source")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Cross-source deduplication requires unique source IDs")
    orders = sorted(source.order for source in sources)
    if orders != list(range(len(sources))):
        raise ValueError("Cross-source deduplication requires contiguous campaign orders")
    output = StoragePath(output_path)
    if (output / MANIFEST_FILENAME).exists():
        raise FileExistsError(f"Cross-source deduplication output already exists: {output_path}")

    total_rows = sum(count_retained_rows(source.final_path) for source in sources)
    partitions = min(_MAX_SHARDS, max(1, (total_rows + _ROWS_PER_SHARD - 1) // _ROWS_PER_SHARD))
    collected: list[CrossSourceDrop] = []
    decisions = execute_phase(
        context,
        Dataset.from_list(list(enumerate(sources)))
        .flat_map(lambda item: retained_source_rows(item[1]))
        .group_by(
            key=lambda row: row.key,
            reducer=decide_duplicate,
            num_output_shards=partitions,
        )
        .map(lambda drop: drop),
        telemetry=telemetry,
        operation="cross-source-dedup",
    ).results
    for decision in decisions:
        if not isinstance(decision, CrossSourceDrop):
            raise ValueError(f"Cross-source dedup returned a non-drop record: {decision!r}")
        collected.append(decision)

    by_source: dict[str, list[CrossSourceDrop]] = {source.source_id: [] for source in sources}
    for drop in collected:
        by_source[drop.source_id].append(drop)
    evidence_sources: dict[str, SourceCrossSourceEvidence] = {}
    for source in sources:
        source_drops = sorted(by_source[source.source_id], key=lambda drop: drop.task_id)
        drop_manifest = str(output / "drops" / f"{_safe_slug(source.source_id)}.parquet")
        write_drop_manifest(drop_manifest, source_drops)
        evidence_sources[source.source_id] = SourceCrossSourceEvidence(
            source_id=source.source_id,
            retained_rows=count_retained_rows(source.final_path),
            cross_source_duplicates=len(source_drops),
            drop_manifest_uri=drop_manifest,
        )
    evidence = CrossSourceEvidence(
        revision=CROSS_SOURCE_REVISION,
        sources=evidence_sources,
        cross_source_duplicates=len(collected),
        drop_root=str(output / "drops"),
    )
    with (output / MANIFEST_FILENAME).open("wt", auto_mkdir=True) as stream:
        json.dump(asdict(evidence), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return evidence
