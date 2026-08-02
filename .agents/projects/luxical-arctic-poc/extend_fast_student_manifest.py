# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the fixed 3M student data to one nested larger rung."""

import argparse
import hashlib
import json
import logging
import math
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from build_manifest import (
    allocate_balanced_quotas,
    manifest_digest,
    normalized_text,
    safe_source_name,
    selected_file_task,
    source_file_capacities,
    stable_seed,
)
from ladder_config import EVAL_ROWS_PER_SOURCE, MANIFEST_ROOT, SAMPLE_BLOCKS_PER_SOURCE, document_view
from rigging.filesystem import atomic_rename

BASE_MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
RUNG_TARGETS = {"10m": 10_000_000, "30m": 30_000_000}
MAXIMUM_TRAIN_ROWS_PER_SOURCE = 262_144
EXPANSION_VERSION = 1
RESULT_FILE = Path("/tmp/luxical-fast-student-extended-manifest")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def output_root(rung: str) -> str:
    """Return the private root for one expanded rung."""
    return f"{MANIFEST_ROOT}/fast-student/expanded-{rung}"


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def sample_positions_excluding(
    rng: np.random.Generator,
    total_rows: int,
    row_count: int,
    excluded: set[int],
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Select stable block-sampled positions outside a fixed base set."""
    if any(position < 0 or position >= total_rows for position in excluded):
        raise ValueError("An excluded position is outside the source")
    if row_count > total_rows - len(excluded):
        raise ValueError(f"Only {total_rows - len(excluded)} rows remain for a sample of {row_count}")
    if row_count == 0:
        return np.empty(0, dtype=np.int64)
    if 2 * (row_count + len(excluded)) > total_rows:
        candidates = np.setdiff1d(
            np.arange(total_rows, dtype=np.int64),
            np.fromiter(excluded, dtype=np.int64),
            assume_unique=True,
        )
        return np.sort(rng.choice(candidates, size=row_count, replace=False))

    block_size = max(1, math.ceil(row_count / SAMPLE_BLOCKS_PER_SOURCE))
    selected: dict[int, None] = {}
    while len(selected) < row_count:
        block_start = int(rng.integers(total_rows))
        for offset in range(block_size):
            position = (block_start + offset) % total_rows
            if position in excluded:
                continue
            selected.setdefault(position, None)
            if len(selected) == row_count:
                break
    return np.sort(np.fromiter(selected, dtype=np.int64))


def fixed_global_positions(
    base_table: pa.Table,
    filesystem: Any,
    files: list[tuple[str, int]],
    protocol: str,
) -> set[int]:
    """Return the source-global positions used by the fixed base manifest."""
    file_start = 0
    coordinates: dict[tuple[str, int], int] = {}
    for path, rows in files:
        uri = f"{protocol}://{path}"
        with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
            group_start = 0
            for group_index in range(parquet_file.num_row_groups):
                coordinates[(uri, group_index)] = file_start + group_start
                group_start += parquet_file.metadata.row_group(group_index).num_rows
            if group_start != rows:
                raise ValueError(f"Parquet row-group counts differ for {uri}")
        file_start += rows

    output = set()
    for row in base_table.select(["input_path", "input_row_group", "input_row_in_group"]).to_pylist():
        key = (str(row["input_path"]), int(row["input_row_group"]))
        if key not in coordinates:
            raise ValueError(f"The base manifest refers to an unknown input row group: {key}")
        output.add(coordinates[key] + int(row["input_row_in_group"]))
    if len(output) != len(base_table):
        raise ValueError("The base manifest contains duplicate input positions")
    return output


def selected_extension_rows(
    filesystem: Any,
    files: list[tuple[str, int]],
    source: str,
    row_count: int,
    excluded: set[int],
    protocol: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Read one deterministic extension that excludes all fixed base rows."""
    file_counts = np.asarray([rows for _, rows in files], dtype=np.int64)
    file_ends = np.cumsum(file_counts)
    positions = sample_positions_excluding(
        np.random.default_rng(stable_seed(f"extended-rows:{source}")),
        int(file_ends[-1]),
        row_count,
        excluded,
    )
    file_indices = np.searchsorted(file_ends, positions, side="right")
    selections = []
    selected_counts: Counter[str] = Counter()
    for file_index in np.unique(file_indices):
        path, _ = files[int(file_index)]
        start = 0 if file_index == 0 else int(file_ends[file_index - 1])
        local_positions = positions[file_indices == file_index] - start
        selections.append((path, local_positions))
        selected_counts[f"{protocol}://{path}"] = len(local_positions)
    rows = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        batches = executor.map(partial(selected_file_task, filesystem, protocol), selections)
        for batch in batches:
            rows.extend(batch)
    if len(rows) != row_count:
        raise ValueError(f"Source {source} returned {len(rows)} extension rows; expected {row_count}")
    rng = np.random.default_rng(stable_seed(f"extended-order:{source}"))
    return [rows[index] for index in rng.permutation(len(rows))], selected_counts


def extended_source_table(
    source: str,
    base_table: pa.Table,
    extension_rows: list[dict[str, Any]],
    target_quota: int,
) -> pa.Table:
    """Append new train rows without changing any fixed base field."""
    base_train_rows = int(pc.sum(pc.equal(base_table["split"], "train")).as_py())
    if len(base_table) != EVAL_ROWS_PER_SOURCE + base_train_rows:
        raise ValueError(f"The base source {source} has an unexpected split count")
    if target_quota < base_train_rows or len(extension_rows) != target_quota - base_train_rows:
        raise ValueError(f"The extension count is invalid for {source}")

    base_with_rung = base_table.append_column(
        "in_expanded_rung",
        pc.equal(base_table["split"], "train"),
    )
    category = str(base_table["source_category"][0].as_py())
    output_rows = []
    for offset, row in enumerate(extension_rows):
        raw_text = str(row.pop("raw_text"))
        output_rows.append(
            row
            | {
                "source": source,
                "source_category": category,
                "split": "train",
                "eval_rank": -1,
                "train_rank": base_train_rows + offset,
                "in_750k": False,
                "in_3m": False,
                "raw_characters": len(raw_text),
                "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized_text(raw_text).encode()).hexdigest(),
                "text": document_view(raw_text),
                "in_expanded_rung": True,
            }
        )
    if output_rows:
        extension_table = pa.Table.from_pylist(output_rows, schema=base_with_rung.schema)
        output = pa.concat_tables((base_with_rung, extension_table))
    else:
        output = base_with_rung
    if len(output) != EVAL_ROWS_PER_SOURCE + target_quota:
        raise ValueError(f"The expanded source {source} has an unexpected row count")
    return output


def write_source_table(root: str, source: str, table: pa.Table) -> tuple[str, str]:
    """Write one expanded source table and return its URL and digest."""
    output_url = f"{root}/sources/{safe_source_name(source)}.parquet"
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with tempfile.TemporaryDirectory(prefix="luxical-expanded-manifest-") as temporary_directory:
        local_path = Path(temporary_directory) / "source.parquet"
        pq.write_table(table, local_path, compression="zstd", row_group_size=8_192)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    return output_url, digest


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def reusable_source_result(
    report: dict[str, Any],
    base_manifest_sha256: str,
    rung: str,
    quota: int,
) -> dict[str, Any] | None:
    """Return a checked source result that can resume an interrupted build."""
    expected = {
        "expansion_version": EXPANSION_VERSION,
        "base_manifest_sha256": base_manifest_sha256,
        "rung": rung,
        "quota": quota,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return None
    source_result = report.get("source_result")
    if not isinstance(source_result, dict) or not isinstance(source_result.get("output_url"), str):
        return None
    return source_result


def build_expanded_manifest(rung: str) -> dict[str, Any]:
    """Build one exact larger rung around the unchanged base manifest."""
    target = RUNG_TARGETS[rung]
    root = output_root(rung)
    base = read_json(BASE_MANIFEST_URL)
    source_files = {}
    capacities = {}
    for index, (source, result) in enumerate(sorted(base["sources"].items()), start=1):
        logger.info("Measuring source %d/%d: %s", index, len(base["sources"]), source)
        filesystem, files = source_file_capacities(result["main_output_dir"], source)
        source_files[source] = (filesystem, files, result["main_output_dir"].partition("://")[0])
        capacities[source] = min(sum(rows for _, rows in files) - EVAL_ROWS_PER_SOURCE, MAXIMUM_TRAIN_ROWS_PER_SOURCE)
    quotas = allocate_balanced_quotas(capacities, target)

    sources = {}
    for index, (source, result) in enumerate(sorted(base["sources"].items()), start=1):
        logger.info("Extending source %d/%d: %s", index, len(base["sources"]), source)
        source_report_url = f"{root}/source-reports/{safe_source_name(source)}.json"
        report_filesystem, report_path = fsspec.core.url_to_fs(source_report_url)
        if report_filesystem.exists(report_path):
            saved = read_json(source_report_url)
            reusable = reusable_source_result(saved, base["sha256"], rung, quotas[source])
            if reusable is not None:
                output_filesystem, output_path = fsspec.core.url_to_fs(reusable["output_url"])
                if output_filesystem.exists(output_path):
                    with pq.ParquetFile(output_path, filesystem=output_filesystem) as parquet_file:
                        output_rows = parquet_file.metadata.num_rows
                    if output_rows == EVAL_ROWS_PER_SOURCE + quotas[source]:
                        logger.info("Reusing complete expanded source %s", source)
                        sources[source] = reusable
                        continue
        base_filesystem, base_path = fsspec.core.url_to_fs(result["output_url"])
        base_table = pq.read_table(base_path, filesystem=base_filesystem)
        base_rows = int(result["counts"]["train_3m"])
        if len(base_table) != EVAL_ROWS_PER_SOURCE + base_rows:
            raise ValueError(f"The fixed base row count differs for {source}")
        filesystem, files, protocol = source_files[source]
        excluded = fixed_global_positions(base_table, filesystem, files, protocol)
        extension_rows, selected_counts = selected_extension_rows(
            filesystem,
            files,
            source,
            quotas[source] - base_rows,
            excluded,
            protocol,
        )
        table = extended_source_table(source, base_table, extension_rows, quotas[source])
        output_url, digest = write_source_table(root, source, table)
        source_result = {
            **result,
            "base_output_url": result["output_url"],
            "output_url": output_url,
            "sha256": digest,
            "counts": result["counts"] | {f"train_{rung}": quotas[source]},
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
                "quota": quotas[source],
                "source_result": source_result,
            },
        )
        sources[source] = source_result

    manifest = {
        **{key: value for key, value in base.items() if key not in {"sha256", "sources", "training_targets"}},
        "version": 3,
        "base_manifest_url": BASE_MANIFEST_URL,
        "base_manifest_sha256": base["sha256"],
        "training_targets": base["training_targets"] | {rung: target},
        "expanded_rung_column": "in_expanded_rung",
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
        "rows": sum(quotas.values()),
        "sources": len(sources),
    }


def main() -> None:
    """Parse arguments and build one larger nested manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=tuple(RUNG_TARGETS), required=True)
    arguments = parser.parse_args()
    summary = build_expanded_manifest(arguments.rung)
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_EXPANDED_MANIFEST=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
