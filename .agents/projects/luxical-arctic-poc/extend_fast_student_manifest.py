# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the fixed 3M student data to one nested larger rung."""

import argparse
import hashlib
import json
import logging
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from build_manifest import (
    PositionOrder,
    allocate_balanced_quotas,
    block_sample_positions,
    manifest_digest,
    normalized_text,
    safe_source_name,
    selected_source_position_rows,
    source_file_capacities,
    stable_seed,
)
from ladder_config import EVAL_ROWS_PER_SOURCE, MANIFEST_ROOT, document_view, read_json, write_json
from rigging.filesystem import atomic_rename

BASE_MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
RUNG_TARGETS = {"10m": 10_000_000, "30m": 30_000_000}
MAXIMUM_TRAIN_ROWS_PER_SOURCE = 262_144
EXTENSION_BLOCK_ROWS = 4_096
EXTENSION_READ_CHUNK_ROWS = 4_096
MANIFEST_VERSION = 3
EXPANSION_VERSION = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def file_snapshot_sha256(files: list[tuple[str, int]], protocol: str) -> str:
    """Return a digest of the ordered input path and row-count snapshot."""
    payload = [(f"{protocol}://{path}", rows) for path, rows in files]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def source_global_positions(
    rows: list[dict[str, Any]],
    filesystem: Any,
    files: list[tuple[str, int]],
    protocol: str,
) -> list[int]:
    """Map stored Parquet coordinates to source-global positions."""
    file_starts = {}
    path_by_uri = {}
    file_start = 0
    for path, count in files:
        uri = f"{protocol}://{path}"
        file_starts[uri] = file_start
        path_by_uri[uri] = path
        file_start += count

    referenced_groups: dict[str, set[int]] = {}
    for row in rows:
        uri = str(row["input_path"])
        referenced_groups.setdefault(uri, set()).add(int(row["input_row_group"]))
    group_starts = {}
    group_sizes = {}
    for uri, group_indices in referenced_groups.items():
        if uri not in path_by_uri:
            raise ValueError(f"The manifest refers to an unknown input file: {uri}")
        path = path_by_uri[uri]
        with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
            start = 0
            for group_index in range(parquet_file.num_row_groups):
                if group_index in group_indices:
                    group_starts[(uri, group_index)] = file_starts[uri] + start
                    group_sizes[(uri, group_index)] = parquet_file.metadata.row_group(group_index).num_rows
                start += parquet_file.metadata.row_group(group_index).num_rows

    positions = []
    for row in rows:
        key = (str(row["input_path"]), int(row["input_row_group"]))
        if key not in group_starts:
            raise ValueError(f"The manifest refers to an unknown input row group: {key}")
        row_in_group = int(row["input_row_in_group"])
        if not 0 <= row_in_group < group_sizes[key]:
            raise ValueError(f"The manifest refers to an unknown row in group: {key}:{row_in_group}")
        positions.append(group_starts[key] + row_in_group)
    return positions


def fixed_global_positions(
    base_table: pa.Table,
    filesystem: Any,
    files: list[tuple[str, int]],
    protocol: str,
) -> set[int]:
    """Return the source-global positions used by the fixed base manifest."""
    coordinate_rows = base_table.select(["input_path", "input_row_group", "input_row_in_group"]).to_pylist()
    positions = source_global_positions(coordinate_rows, filesystem, files, protocol)
    output = set(positions)
    if len(output) != len(positions):
        raise ValueError("The base manifest contains duplicate input positions")
    return output


def extension_positions(
    files: list[tuple[str, int]],
    source: str,
    row_count: int,
    excluded: set[int],
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Return one deterministic extension that excludes all fixed base rows."""
    total_rows = sum(count for _, count in files)
    return block_sample_positions(
        np.random.default_rng(stable_seed(f"extended-rows:{source}")),
        total_rows,
        row_count,
        excluded=excluded,
        block_size=EXTENSION_BLOCK_ROWS,
        position_order=PositionOrder.SELECTION,
    )


def bounded_extension_table(
    source: str,
    category: str,
    extension_rows: list[dict[str, Any]],
    first_train_rank: int,
    rung_quotas: dict[str, int],
    schema: pa.Schema,
) -> pa.Table:
    """Convert raw extension rows to a bounded manifest table."""
    output_rows = []
    for offset, input_row in enumerate(extension_rows):
        row = dict(input_row)
        raw_text = str(row.pop("raw_text"))
        train_rank = first_train_rank + offset
        output_rows.append(
            row
            | {
                "source": source,
                "source_category": category,
                "split": "train",
                "eval_rank": -1,
                "train_rank": train_rank,
                "in_750k": False,
                "in_3m": False,
                "raw_characters": len(raw_text),
                "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized_text(raw_text).encode()).hexdigest(),
                "text": document_view(raw_text),
                **{f"in_{rung}": train_rank < quota for rung, quota in rung_quotas.items()},
            }
        )
    return pa.Table.from_pylist(output_rows, schema=schema)


def base_with_rung_columns(base_table: pa.Table, rung_quotas: dict[str, int]) -> pa.Table:
    """Add expanded rung membership to unchanged base rows."""
    output = base_table
    for rung in rung_quotas:
        output = output.append_column(f"in_{rung}", pc.equal(base_table["split"], "train"))
    return output


def extended_source_table(
    source: str,
    base_table: pa.Table,
    extension_rows: list[dict[str, Any]],
    rung_quotas: dict[str, int],
) -> pa.Table:
    """Append new train rows without changing any fixed base field."""
    base_train_rows = int(pc.sum(pc.equal(base_table["split"], "train")).as_py())
    if len(base_table) != EVAL_ROWS_PER_SOURCE + base_train_rows:
        raise ValueError(f"The base source {source} has an unexpected split count")
    target_quota = max(rung_quotas.values())
    if any(quota < base_train_rows for quota in rung_quotas.values()):
        raise ValueError(f"An expanded quota is smaller than the fixed base for {source}")
    if len(extension_rows) != target_quota - base_train_rows:
        raise ValueError(f"The extension count is invalid for {source}")

    base_with_rung = base_with_rung_columns(base_table, rung_quotas)
    category = str(base_table["source_category"][0].as_py())
    if extension_rows:
        extension_table = bounded_extension_table(
            source,
            category,
            extension_rows,
            base_train_rows,
            rung_quotas,
            base_with_rung.schema,
        )
        output = pa.concat_tables((base_with_rung, extension_table))
    else:
        output = base_with_rung
    if len(output) != EVAL_ROWS_PER_SOURCE + target_quota:
        raise ValueError(f"The expanded source {source} has an unexpected row count")
    return output


def selected_extension_table(
    filesystem: Any,
    files: list[tuple[str, int]],
    source: str,
    base_table: pa.Table,
    rung_quotas: dict[str, int],
    protocol: str,
    read_chunk_rows: int = EXTENSION_READ_CHUNK_ROWS,
) -> tuple[pa.Table, Counter[str]]:
    """Read raw rows in chunks and return one bounded expanded source table."""
    if read_chunk_rows < 1:
        raise ValueError("The extension read chunk must be positive")
    base_train_rows = int(pc.sum(pc.equal(base_table["split"], "train")).as_py())
    if len(base_table) != EVAL_ROWS_PER_SOURCE + base_train_rows:
        raise ValueError(f"The base source {source} has an unexpected split count")
    target_quota = max(rung_quotas.values())
    if any(quota < base_train_rows for quota in rung_quotas.values()):
        raise ValueError(f"An expanded quota is smaller than the fixed base for {source}")
    positions = extension_positions(
        files,
        source,
        target_quota - base_train_rows,
        fixed_global_positions(base_table, filesystem, files, protocol),
    )
    base_with_rung = base_with_rung_columns(base_table, rung_quotas)
    category = str(base_table["source_category"][0].as_py())
    tables = [base_with_rung]
    selected_counts: Counter[str] = Counter()
    for start in range(0, len(positions), read_chunk_rows):
        chunk_positions = positions[start : start + read_chunk_rows]
        rows, chunk_counts = selected_source_position_rows(filesystem, files, chunk_positions, protocol)
        row_positions = source_global_positions(rows, filesystem, files, protocol)
        row_by_position = dict(zip(row_positions, rows, strict=True))
        if len(row_by_position) != len(rows):
            raise ValueError(f"Source {source} returned duplicate extension positions")
        ordered_rows = [row_by_position[int(position)] for position in chunk_positions]
        tables.append(
            bounded_extension_table(
                source,
                category,
                ordered_rows,
                base_train_rows + start,
                rung_quotas,
                base_with_rung.schema,
            )
        )
        selected_counts.update(chunk_counts)
        logger.info("Bounded extension rows for %s: %d/%d", source, start + len(chunk_positions), len(positions))
    output = pa.concat_tables(tables)
    if len(output) != EVAL_ROWS_PER_SOURCE + target_quota:
        raise ValueError(f"The expanded source {source} has an unexpected row count")
    return output, selected_counts


def write_source_table(root: str, source: str, table: pa.Table) -> tuple[str, str]:
    """Write one expanded source table and return its URL and digest."""
    output_url = f"{root}/sources/{safe_source_name(source)}.parquet"
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with tempfile.TemporaryDirectory(prefix="luxical-expanded-manifest-") as temporary_directory:
        local_path = Path(temporary_directory) / "source.parquet"
        pq.write_table(table, local_path, compression="zstd")
        with local_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    return output_url, digest


def reusable_source_result(
    report: dict[str, Any],
    base_manifest_sha256: str,
    rung: str,
    rung_quotas: dict[str, int],
    input_snapshot_sha256: str,
) -> dict[str, Any] | None:
    """Return a checked source result that can resume an interrupted build."""
    expected = {
        "expansion_version": EXPANSION_VERSION,
        "base_manifest_sha256": base_manifest_sha256,
        "rung": rung,
        "rung_quotas": rung_quotas,
        "input_snapshot_sha256": input_snapshot_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return None
    source_result = report.get("source_result")
    if not isinstance(source_result, dict) or not isinstance(source_result.get("output_url"), str):
        return None
    return source_result


def assigned_source_names(sources: Iterable[str], shard_index: int, num_shards: int) -> list[str]:
    """Assign sorted source names to independent manifest workers."""
    if num_shards < 1:
        raise ValueError("The manifest shard count must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Manifest shard index {shard_index} is outside [0, {num_shards})")
    return [source for index, source in enumerate(sorted(sources)) if index % num_shards == shard_index]


def build_expanded_manifest(rung: str, shard_index: int = 0, num_shards: int = 1) -> dict[str, Any]:
    """Build one exact larger rung around the unchanged base manifest."""
    included_rungs = [name for name, target in RUNG_TARGETS.items() if target <= RUNG_TARGETS[rung]]
    root = f"{MANIFEST_ROOT}/fast-student/expanded-{rung}"
    base = read_json(BASE_MANIFEST_URL)
    source_files = {}
    capacities = {}
    for index, (source, result) in enumerate(sorted(base["sources"].items()), start=1):
        logger.info("Measuring source %d/%d: %s", index, len(base["sources"]), source)
        filesystem, files = source_file_capacities(result["main_output_dir"], source)
        protocol = result["main_output_dir"].partition("://")[0]
        source_files[source] = (filesystem, files, protocol, file_snapshot_sha256(files, protocol))
        capacities[source] = min(sum(rows for _, rows in files) - EVAL_ROWS_PER_SOURCE, MAXIMUM_TRAIN_ROWS_PER_SOURCE)
    quotas_by_rung = {name: allocate_balanced_quotas(capacities, RUNG_TARGETS[name]) for name in included_rungs}
    for source, result in base["sources"].items():
        previous = int(result["counts"]["train_3m"])
        for name in included_rungs:
            quota = quotas_by_rung[name][source]
            if quota < previous:
                raise ValueError(f"The {name} quota for {source} is not nested")
            previous = quota

    assigned_sources = assigned_source_names(base["sources"], shard_index, num_shards)
    sources = {}
    for index, source in enumerate(assigned_sources, start=1):
        result = base["sources"][source]
        logger.info(
            "Extending assigned source %d/%d on shard %d/%d: %s",
            index,
            len(assigned_sources),
            shard_index,
            num_shards,
            source,
        )
        rung_quotas = {name: quotas_by_rung[name][source] for name in included_rungs}
        target_quota = rung_quotas[rung]
        filesystem, files, protocol, input_snapshot_sha256 = source_files[source]
        source_report_url = f"{root}/source-reports/{safe_source_name(source)}.json"
        report_filesystem, report_path = fsspec.core.url_to_fs(source_report_url)
        if report_filesystem.exists(report_path):
            saved = read_json(source_report_url)
            reusable = reusable_source_result(
                saved,
                base["sha256"],
                rung,
                rung_quotas,
                input_snapshot_sha256,
            )
            if reusable is not None:
                output_filesystem, output_path = fsspec.core.url_to_fs(reusable["output_url"])
                if output_filesystem.exists(output_path):
                    with pq.ParquetFile(output_path, filesystem=output_filesystem) as parquet_file:
                        output_rows = parquet_file.metadata.num_rows
                    if output_rows == EVAL_ROWS_PER_SOURCE + target_quota:
                        logger.info("Reusing complete expanded source %s", source)
                        sources[source] = reusable
                        continue
        base_filesystem, base_path = fsspec.core.url_to_fs(result["output_url"])
        base_table = pq.read_table(base_path, filesystem=base_filesystem)
        base_rows = int(result["counts"]["train_3m"])
        if len(base_table) != EVAL_ROWS_PER_SOURCE + base_rows:
            raise ValueError(f"The fixed base row count differs for {source}")
        table, selected_counts = selected_extension_table(
            filesystem,
            files,
            source,
            base_table,
            rung_quotas,
            protocol,
        )
        output_url, digest = write_source_table(root, source, table)
        base_selected_counts = {str(row["path"]): int(row["selected_rows"]) for row in result["selected_input_files"]}
        combined_selected_counts = Counter(base_selected_counts)
        combined_selected_counts.update(selected_counts)
        source_result = {
            **result,
            "base_output_url": result["output_url"],
            "output_url": output_url,
            "sha256": digest,
            "available_input_file_count": len(files),
            "available_input_rows": sum(count for _, count in files),
            "input_snapshot_sha256": input_snapshot_sha256,
            "counts": result["counts"] | {f"train_{name}": quota for name, quota in rung_quotas.items()},
            "selected_input_files": [
                {
                    "path": f"{protocol}://{path}",
                    "total_rows": count,
                    "selected_rows": combined_selected_counts[f"{protocol}://{path}"],
                }
                for path, count in files
                if combined_selected_counts[f"{protocol}://{path}"]
            ],
            "expanded_selected_input_files": [
                {"path": path, "selected_rows": count} for path, count in sorted(selected_counts.items())
            ],
        }
        write_json(
            source_report_url,
            {
                "expansion_version": EXPANSION_VERSION,
                "base_manifest_sha256": base["sha256"],
                "rung": rung,
                "rung_quotas": rung_quotas,
                "input_snapshot_sha256": input_snapshot_sha256,
                "source_result": source_result,
            },
        )
        sources[source] = source_result

    if num_shards > 1:
        return {
            "base_manifest_sha256": base["sha256"],
            "rung": rung,
            "rows": sum(quotas_by_rung[rung][source] for source in assigned_sources),
            "sources": len(sources),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "finalized": False,
        }

    manifest = {
        **{key: value for key, value in base.items() if key not in {"sha256", "sources", "training_targets"}},
        "version": MANIFEST_VERSION,
        "base_manifest_url": BASE_MANIFEST_URL,
        "base_manifest_sha256": base["sha256"],
        "training_targets": base["training_targets"] | {name: RUNG_TARGETS[name] for name in included_rungs},
        "expanded_rung_columns": [f"in_{name}" for name in included_rungs],
        "maximum_train_rows_per_source": MAXIMUM_TRAIN_ROWS_PER_SOURCE,
        "expansion_version": EXPANSION_VERSION,
        "sources": sources,
    }
    manifest["sha256"] = manifest_digest(manifest)
    manifest_url = f"{root}/manifest.json"
    write_json(manifest_url, manifest)
    return {
        "manifest_url": manifest_url,
        "sha256": manifest["sha256"],
        "base_manifest_sha256": base["sha256"],
        "rung": rung,
        "rows": sum(quotas_by_rung[rung].values()),
        "sources": len(sources),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "finalized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=tuple(RUNG_TARGETS), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    arguments = parser.parse_args()
    summary = build_expanded_manifest(arguments.rung, arguments.shard_index, arguments.num_shards)
    result_file = Path(f"/tmp/luxical-fast-student-extended-manifest-{arguments.rung}-{arguments.shard_index:02d}")
    result_file.write_text(json.dumps(summary, sort_keys=True))
    marker = "FAST_STUDENT_EXPANDED_MANIFEST" if summary["finalized"] else "FAST_STUDENT_EXPANDED_MANIFEST_SHARD"
    logger.info("%s=%s", marker, json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
