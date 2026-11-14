# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export selected metadata from co-partitioned normalized and attribute shards.

Exports inspect a bounded prefix. Locators identify normalized rows; the public
release adds upstream coordinates. Empty ``dedup`` marks fuzzy-exempt sources.
"""

import dataclasses
import json
import math
from collections.abc import Iterator
from contextlib import ExitStack
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM
from experiments.datakit.store.datakit_store import _FilterStats, _keep_document

DEFAULT_EMBEDDING_READ_BYTES = 64 * 1024 * 1024


class HarrierEmbeddingProvenance(TypedDict):
    artifact: str
    model_name: str
    model_revision: str
    embedding_dim: int
    quantization_scale: float
    quantization_range: float


@dataclasses.dataclass(frozen=True)
class ShardManifest:
    marin_source: str
    normalized: str
    decontam: str
    cluster: str
    quality: str
    exact_dedup: str
    dedup: str
    bucket_edges: dict[str, list[float]] | None
    content_type: str = ""
    store_task_id: int | None = None
    harrier_embedding: str = ""
    harrier_embedding_provenance: HarrierEmbeddingProvenance | None = None


# Nullable fixed-size lists can fail a Parquet round trip; use variable lists on disk.
METADATA_SCHEMA = pa.schema(
    [
        ("marin_source", pa.string()),
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("cluster_id", pa.int32()),
        ("quality_bucket", pa.int32()),
        ("quality_score", pa.float64()),
        ("content_type", pa.string()),
        pa.field("harrier_embedding", pa.list_(pa.field("item", pa.int8(), nullable=False))),
        ("normalized_shard", pa.string()),
        ("normalized_row", pa.int64()),
        ("store_task_id", pa.int32()),
    ]
)


def _rows(parquet: pq.ParquetFile, columns: list[str], limit: int, batch_size: int) -> Iterator[dict]:
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        if limit <= 0:
            return
        rows = batch.slice(0, limit).to_pylist()
        yield from rows
        limit -= len(rows)
        if limit == 0:
            return


def _selected_duplicates(path: str, ids: set[str], batch_size: int, *, fuzzy: bool) -> set[str]:
    if not path:
        return set()
    fs, resolved = filesystem_for(path)
    if not fs.exists(resolved):
        return set()
    duplicates: set[str] = set()
    with fs.open(resolved, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        columns = ["id", "dup_doc"] if fuzzy else ["id"]
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            for row in batch.to_pylist():
                if fuzzy and row["dup_doc"] is not True:
                    raise ValueError(f"{path}: verified fuzzy duplicate has dup_doc != True")
                if row["id"] in ids:
                    duplicates.add(row["id"])
    return duplicates


def _selected_harrier_embeddings(
    manifest: ShardManifest,
    selected_ids: set[str],
    batch_size: int,
    max_bytes: int,
) -> tuple[dict[str, list[int]], dict]:
    """Read sparse quantized vectors by ID without assuming dense row alignment."""
    scan = {"projected_bytes": 0, "budget_exhausted": False, "scanned_row_groups": 0}
    if not manifest.harrier_embedding:
        return {}, scan
    provenance = manifest.harrier_embedding_provenance
    if provenance is None or provenance["embedding_dim"] != HARRIER_DIM:
        raise ValueError("Requested Harrier embeddings require frozen provenance with embedding_dim=1024")
    if not all(
        isinstance(provenance[key], str) and provenance[key] for key in ("artifact", "model_name", "model_revision")
    ):
        raise ValueError("Harrier provenance requires artifact, model name and revision")
    for key in ("quantization_scale", "quantization_range"):
        if not math.isfinite(provenance[key]) or provenance[key] <= 0:
            raise ValueError(f"Invalid Harrier provenance {key}")
    fs, resolved = filesystem_for(manifest.harrier_embedding)
    vectors: dict[str, list[int]] = {}
    with fs.open(resolved, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        field_type = parquet.schema_arrow.field("embedding").type
        if not (
            pa.types.is_fixed_size_list(field_type)
            and field_type.list_size == HARRIER_DIM
            and pa.types.is_int8(field_type.value_type)
        ):
            raise ValueError(f"Harrier embedding must be fixed_size_list<int8>[1024], got {field_type}")
        if not selected_ids:
            return vectors, scan
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            id_column = next(group.column(i) for i in range(group.num_columns) if group.column(i).path_in_schema == "id")
            statistics = id_column.statistics
            if statistics is not None and statistics.has_min_max:
                if not any(statistics.min <= doc_id <= statistics.max for doc_id in selected_ids):
                    continue
            size = sum(
                group.column(i).total_compressed_size
                for i in range(group.num_columns)
                if group.column(i).path_in_schema.split(".")[0] in {"id", "embedding"}
            )
            if scan["projected_bytes"] + size > max_bytes:
                scan["budget_exhausted"] = True
                break
            scan["projected_bytes"] += size
            scan["scanned_row_groups"] += 1
            for batch in parquet.iter_batches(
                columns=["id", "embedding"], row_groups=[group_index], batch_size=batch_size
            ):
                for row in batch.to_pylist():
                    doc_id = row["id"]
                    if doc_id not in selected_ids:
                        continue
                    vector = row["embedding"]
                    if vector is None or any(value is None for value in vector):
                        raise ValueError(f"Null vector or coordinate for selected Harrier ID {doc_id!r}")
                    if doc_id in vectors and vectors[doc_id] != vector:
                        raise ValueError(f"Conflicting Harrier vectors for selected ID {doc_id!r}")
                    vectors[doc_id] = vector
    return vectors, scan


def export_metadata_shard(
    manifest: ShardManifest,
    output_dir: str,
    *,
    cluster_view: int,
    max_rows: int,
    batch_size: int = 8192,
    max_embedding_bytes: int = DEFAULT_EMBEDDING_READ_BYTES,
) -> dict:
    """Export a bounded prefix after validating dense row counts and IDs.

    ``bucket_edges=None`` uses precomputed quality buckets. Otherwise, apply
    frozen per-type cutpoints, with default cutpoints for unknown types.
    Provider IDs are JSON-encoded to preserve numeric and structured values.

    Each call needs a separate output directory. Memory is bounded by
    ``max_rows`` plus one batch. Sparse embeddings join by ID within a compressed
    byte budget that excludes footers and filesystem read-ahead. Output vectors
    contain exactly 1024 int8 coordinates when present.
    """
    if max_rows <= 0 or batch_size <= 0 or max_embedding_bytes <= 0:
        raise ValueError("Row, batch and embedding read limits must be positive")
    edges_by_type: dict[str, np.ndarray] = {}
    if manifest.bucket_edges is not None:
        for name, values in manifest.bucket_edges.items():
            edges = np.asarray(values, dtype=np.float64)
            if not len(edges) or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
                raise ValueError(f"Invalid calibration edges for {name!r}")
            edges_by_type[name] = edges
        if "default" not in edges_by_type or len({len(edges) for edges in edges_by_type.values()}) != 1:
            raise ValueError("Calibration requires default and equal numbers of cutpoints for every type")
    cluster_column = f"cluster_{cluster_view}"
    with ExitStack() as stack:
        parquets = {}
        for key in ("normalized", "decontam", "cluster", "quality", "content_type"):
            path = getattr(manifest, key)
            if not path:
                continue
            fs, resolved = filesystem_for(path)
            parquets[key] = pq.ParquetFile(stack.enter_context(fs.open(resolved, "rb")))
        counts = {key: parquet.metadata.num_rows for key, parquet in parquets.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"{manifest.marin_source}: dense-table row count mismatch: {counts}")
        total_rows = counts["normalized"]
        limit = min(total_rows, max_rows)
        normalized_columns = ["id"]
        if "source_id" in parquets["normalized"].schema_arrow.names:
            normalized_columns.append("source_id")
        normalized = list(_rows(parquets["normalized"], normalized_columns, limit, batch_size))
        ids = {row["id"] for row in normalized}
        exact = _selected_duplicates(manifest.exact_dedup, ids, batch_size, fuzzy=False)
        fuzzy = _selected_duplicates(manifest.dedup, ids, batch_size, fuzzy=True)
        dense = {
            "decontam": _rows(parquets["decontam"], ["id", "contaminated"], limit, batch_size),
            "cluster": _rows(parquets["cluster"], ["id", cluster_column], limit, batch_size),
            "quality": _rows(
                parquets["quality"],
                ["id", "score"] if edges_by_type else ["id", "score", "quality_bucket"],
                limit,
                batch_size,
            ),
        }
        if manifest.content_type:
            dense["content_type"] = _rows(parquets["content_type"], ["id", "content_type"], limit, batch_size)
        stats = _FilterStats()
        content_type_rows: dict[str, int] = {}
        uncalibrated = 0
        bucket_rows: dict[tuple[int, int], list[dict]] = {}
        source_ids_present = 0
        for ordinal, row in enumerate(normalized):
            attrs = {name: next(iterator) for name, iterator in dense.items()}
            if any(attr["id"] != row["id"] for attr in attrs.values()):
                raise RuntimeError(f"{manifest.marin_source}: dense id/order mismatch at normalized row {ordinal}")
            cluster = attrs["cluster"][cluster_column]
            score = attrs["quality"]["score"]
            content_type = attrs["content_type"]["content_type"] if manifest.content_type else None
            if edges_by_type:
                key = content_type if manifest.content_type else "default"
                edges = edges_by_type.get(key, edges_by_type["default"])
                quality = int(np.digitize(np.float64(score), edges))
                content_type_rows[key] = content_type_rows.get(key, 0) + 1
                uncalibrated += key not in edges_by_type
                max_bucket = len(edges)
            else:
                quality = attrs["quality"]["quality_bucket"]
                max_bucket = 4
            if cluster is None or not 0 <= cluster < cluster_view:
                raise ValueError(f"Invalid cluster at row {ordinal}: {cluster}")
            if quality is None or not 0 <= quality <= max_bucket or score is None or not math.isfinite(score):
                raise ValueError(f"Invalid quality score/bucket at row {ordinal}: {score}/{quality}")
            if attrs["decontam"]["contaminated"] is None:
                raise ValueError(f"Null contamination flag at row {ordinal}")
            if not _keep_document(row["id"], attrs["decontam"]["contaminated"], exact, fuzzy, stats):
                continue
            source_id = row.get("source_id")
            source_ids_present += source_id is not None
            bucket_rows.setdefault((cluster, quality), []).append(
                {
                    "marin_source": manifest.marin_source,
                    "id": row["id"],
                    "source_id": (
                        json.dumps(source_id, ensure_ascii=False, sort_keys=True) if source_id is not None else None
                    ),
                    "cluster_id": cluster,
                    "quality_bucket": quality,
                    "quality_score": score,
                    "content_type": content_type,
                    "normalized_shard": manifest.normalized,
                    "normalized_row": ordinal,
                    "store_task_id": manifest.store_task_id,
                }
            )
        selected_ids = {row["id"] for rows in bucket_rows.values() for row in rows}
        embeddings, embedding_scan = _selected_harrier_embeddings(
            manifest, selected_ids, batch_size, max_embedding_bytes
        )
        embedded_occurrences = 0
        for rows in bucket_rows.values():
            for row in rows:
                row["harrier_embedding"] = embeddings.get(row["id"])
                embedded_occurrences += row["harrier_embedding"] is not None
        null_occurrences = stats.records_out - embedded_occurrences
        unmatched_occurrences = null_occurrences if manifest.harrier_embedding else 0
        embedding_status = "not_requested"
        if manifest.harrier_embedding:
            embedding_status = "incomplete" if null_occurrences else "complete"
        embedding_report = {
            "status": embedding_status,
            "selected_occurrences": stats.records_out,
            "embedded_occurrences": embedded_occurrences,
            "null_occurrences": null_occurrences,
            "unmatched_occurrences": unmatched_occurrences,
            "missing_occurrences": None if embedding_scan["budget_exhausted"] else unmatched_occurrences,
            "unresolved_occurrences": unmatched_occurrences if embedding_scan["budget_exhausted"] else 0,
            "unmatched_distinct_ids": len(selected_ids - embeddings.keys()) if manifest.harrier_embedding else 0,
            **embedding_scan,
            "provenance": manifest.harrier_embedding_provenance,
            "dimension": HARRIER_DIM,
            "dtype": "int8",
        }
        # Write only after alignment has passed for the complete bounded prefix.
        outputs = []
        for (cluster, quality), rows in sorted(bucket_rows.items()):
            path = prefix_join(output_dir, f"cluster={cluster}/quality={quality}/metadata.parquet")
            fs, resolved = filesystem_for(path)
            fs.makedirs(resolved.rsplit("/", 1)[0], exist_ok=True)
            with fs.open(resolved, "wb") as stream:
                pq.write_table(pa.Table.from_pylist(rows, schema=METADATA_SCHEMA), stream, compression="zstd")
            outputs.append({"cluster_id": cluster, "quality_bucket": quality, "rows": len(rows), "path": path})
        return {
            "manifest": dataclasses.asdict(manifest),
            "harrier_embedding": embedding_report,
            "total_normalized_rows": total_rows,
            "examined_rows": limit,
            "truncated": limit < total_rows,
            "selected_source_ids_present": source_ids_present,
            "counters": stats.counters(),
            "outputs": outputs,
            "quality_bucketing": "content_type_calibration" if edges_by_type else "precomputed",
            "content_type_rows": content_type_rows,
            "content_type_without_calibration": uncalibrated,
        }
