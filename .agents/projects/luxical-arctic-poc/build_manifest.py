# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the fixed, nested dataset for the Luxical scaling ladder."""

import hashlib
import json
import logging
import math
import posixpath
import re
from collections import Counter, defaultdict
from collections.abc import Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ladder_config import (
    EVAL_ROWS_PER_SOURCE,
    MANIFEST_ROOT,
    MAX_TRAIN_ROWS_PER_SOURCE,
    MIN_SOURCES,
    PREDECLARED_OOD_SOURCES,
    SAMPLE_BLOCKS_PER_SOURCE,
    SAMPLING_METHOD,
    SEED,
    SOURCE_INVENTORY_URL,
    SURVEY_ROWS_PER_SOURCE,
    TEXT_WINDOW_CHARS,
    TRAIN_TARGET_3M,
    TRAIN_TARGET_750K,
    document_view,
    source_category,
)
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
RESULT_FILE = Path("/tmp/luxical-arctic-manifest")
REQUIRED_COLUMNS = frozenset(("id", "text"))
IO_WORKERS = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class PositionOrder(StrEnum):
    """Select the order of sampled source positions."""

    SORTED = "sorted"
    SELECTION = "selection"


def stable_seed(label: str) -> int:
    """Return a stable seed for one selection operation."""
    digest = hashlib.sha256(f"{SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def parquet_paths(main_output_dir: str) -> tuple[Any, list[str]]:
    """Return the filesystem and sorted Parquet paths for one source."""
    filesystem, root = fsspec.core.url_to_fs(main_output_dir)
    paths = sorted(filesystem.glob(posixpath.join(root, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No Parquet files under {main_output_dir}")
    return filesystem, paths


def parquet_file_capacity(filesystem: Any, path: str) -> tuple[str, int]:
    """Return the checked row count for one Parquet file."""
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        columns = frozenset(parquet_file.schema_arrow.names)
        if not REQUIRED_COLUMNS.issubset(columns):
            raise ValueError(f"Required columns are missing from {path}: {columns}")
        return path, parquet_file.metadata.num_rows


def source_file_capacities(main_output_dir: str, source: str) -> tuple[Any, list[tuple[str, int]]]:
    """Return the row capacity of every nonempty source file."""
    filesystem, paths = parquet_paths(main_output_dir)
    logger.info("Reading %d Parquet footers for %s", len(paths), source)
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        capacities = executor.map(partial(parquet_file_capacity, filesystem), paths)
        files = [(path, rows) for path, rows in capacities if rows]
    logger.info("Measured %d nonempty files for %s", len(files), source)
    return filesystem, files


def allocate_balanced_quotas(capacities: dict[str, int], target: int) -> dict[str, int]:
    """Allocate an exact target while keeping source quotas balanced."""
    if sum(capacities.values()) < target:
        raise ValueError(f"Only {sum(capacities.values())} rows are available for target {target}")
    quotas = {source: 0 for source in sorted(capacities)}
    remaining = target
    while remaining:
        eligible = [source for source in quotas if quotas[source] < capacities[source]]
        if not eligible:
            raise ValueError(f"No source capacity remains with {remaining} rows unassigned")
        share, extra = divmod(remaining, len(eligible))
        assigned = 0
        for index, source in enumerate(eligible):
            requested = share + int(index < extra)
            increment = min(capacities[source] - quotas[source], requested)
            quotas[source] += increment
            assigned += increment
        if assigned == 0:
            raise ValueError(f"Quota allocation made no progress with {remaining} rows unassigned")
        remaining -= assigned
    return quotas


def row_group_offsets(parquet_file: pq.ParquetFile) -> list[int]:
    """Return the exclusive row offset for each row group."""
    counts = [parquet_file.metadata.row_group(index).num_rows for index in range(parquet_file.num_row_groups)]
    return list(accumulate(counts))


def selected_file_rows(
    filesystem: Any,
    path: str,
    positions: np.ndarray[Any, np.dtype[np.int64]],
    protocol: str,
) -> list[dict[str, Any]]:
    """Read selected rows and their exact positions from one Parquet file."""
    rows = []
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        group_ends = np.asarray(row_group_offsets(parquet_file))
        positions_by_group: dict[int, list[int]] = defaultdict(list)
        for position in np.sort(positions):
            group_index = int(np.searchsorted(group_ends, position, side="right"))
            group_start = 0 if group_index == 0 else int(group_ends[group_index - 1])
            positions_by_group[group_index].append(int(position) - group_start)
        for group_index, group_positions in positions_by_group.items():
            table = parquet_file.read_row_group(group_index, columns=["id", "text"])
            for position in group_positions:
                row = table.slice(position, 1).to_pylist()[0]
                rows.append(
                    {
                        "id": str(row["id"]),
                        "raw_text": str(row["text"]),
                        "input_path": f"{protocol}://{path}",
                        "input_row_group": group_index,
                        "input_row_in_group": position,
                    }
                )
    return rows


def selected_file_task(
    filesystem: Any,
    protocol: str,
    selection: tuple[str, np.ndarray[Any, np.dtype[np.int64]]],
) -> list[dict[str, Any]]:
    """Read one selected set of file rows."""
    path, positions = selection
    return selected_file_rows(filesystem, path, positions, protocol)


def block_sample_positions(
    rng: np.random.Generator,
    total_rows: int,
    row_count: int,
    *,
    excluded: AbstractSet[int] = frozenset(),
    block_size: int | None = None,
    position_order: PositionOrder = PositionOrder.SORTED,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Select unique rows from uniform circular blocks."""
    if any(position < 0 or position >= total_rows for position in excluded):
        raise ValueError("An excluded position is outside the source")
    available_rows = total_rows - len(excluded)
    if row_count > available_rows:
        raise ValueError(f"Only {available_rows} rows are available for a sample of {row_count}")
    if row_count == 0:
        return np.empty(0, dtype=np.int64)
    if not excluded and 2 * row_count > total_rows:
        return np.sort(rng.choice(total_rows, size=row_count, replace=False))

    selected_block_size = block_size or max(1, math.ceil(row_count / SAMPLE_BLOCKS_PER_SOURCE))
    positions: dict[int, None] = {}
    while len(positions) < row_count:
        if len(positions) + len(excluded) >= 9 * total_rows // 10:
            remaining = np.asarray(
                [position for position in range(total_rows) if position not in excluded and position not in positions],
                dtype=np.int64,
            )
            rng.shuffle(remaining)
            for position in remaining[: row_count - len(positions)]:
                positions[int(position)] = None
            break
        block_start = int(rng.integers(total_rows))
        for offset in range(selected_block_size):
            position = (block_start + offset) % total_rows
            if position in excluded:
                continue
            positions.setdefault(position, None)
            if len(positions) == row_count:
                break
    output = np.fromiter(positions, dtype=np.int64)
    return output if position_order == PositionOrder.SELECTION else np.sort(output)


def selected_source_position_rows(
    filesystem: Any,
    files: list[tuple[str, int]],
    global_positions: np.ndarray[Any, np.dtype[np.int64]],
    protocol: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Read source-global positions and return their per-file counts."""
    file_counts = np.asarray([file_rows for _, file_rows in files], dtype=np.int64)
    file_ends = np.cumsum(file_counts)
    file_indices = np.searchsorted(file_ends, global_positions, side="right")
    rows = []
    selected_counts: Counter[str] = Counter()
    selections = []
    for file_index in np.unique(file_indices):
        path, _ = files[int(file_index)]
        file_start = 0 if file_index == 0 else int(file_ends[file_index - 1])
        positions = global_positions[file_indices == file_index] - file_start
        selections.append((path, positions))
        selected_counts[f"{protocol}://{path}"] = len(positions)
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        row_batches = executor.map(partial(selected_file_task, filesystem, protocol), selections)
        for batch in row_batches:
            rows.extend(batch)
    if len(rows) != len(global_positions):
        raise ValueError(f"Source returned {len(rows)} rows; expected {len(global_positions)}")
    return rows, selected_counts


def selected_source_rows(
    filesystem: Any,
    files: list[tuple[str, int]],
    source: str,
    row_count: int,
    protocol: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Read a uniform row sample across all source files."""
    rng = np.random.default_rng(stable_seed(f"rows:{source}"))
    total_rows = sum(file_rows for _, file_rows in files)
    if row_count > total_rows:
        raise ValueError(f"Source {source} has {total_rows} rows; requested {row_count}")
    global_positions = block_sample_positions(rng, total_rows, row_count)
    rows, selected_counts = selected_source_position_rows(filesystem, files, global_positions, protocol)
    split_rng = np.random.default_rng(stable_seed(f"split:{source}"))
    shuffled_rows = [rows[index] for index in split_rng.permutation(len(rows))]
    return shuffled_rows, selected_counts


def normalized_text(text: str) -> str:
    """Return a simple form for cross-source exact duplicate checks."""
    return " ".join(text.casefold().split())


def safe_source_name(source: str) -> str:
    """Return a unique storage name for one source."""
    readable = re.sub(r"[^a-zA-Z0-9_.-]+", "__", source)
    digest = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{readable}-{digest}"


def write_source_rows(
    source: str,
    rows: list[dict[str, Any]],
    train_quota_750k: int,
    train_quota_3m: int,
) -> tuple[str, dict[str, int]]:
    """Add fixed splits and write one source manifest shard."""
    category = source_category(source)
    output_rows = []
    for selection_rank, row in enumerate(rows):
        raw_text = row.pop("raw_text")
        is_eval = selection_rank < EVAL_ROWS_PER_SOURCE
        train_rank = selection_rank - EVAL_ROWS_PER_SOURCE
        split = "eval" if is_eval else "train"
        view = document_view(raw_text)
        output_rows.append(
            row
            | {
                "source": source,
                "source_category": category.value,
                "split": split,
                "eval_rank": selection_rank if is_eval else -1,
                "train_rank": train_rank if not is_eval else -1,
                "in_750k": not is_eval and train_rank < train_quota_750k,
                "in_3m": not is_eval and train_rank < train_quota_3m,
                "raw_characters": len(raw_text),
                "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized_text(raw_text).encode()).hexdigest(),
                "text": view,
            }
        )
    table = pa.Table.from_pylist(output_rows)
    output_url = f"{MANIFEST_ROOT}/sources/{safe_source_name(source)}.parquet"
    output_filesystem, output_path = fsspec.core.url_to_fs(output_url)
    with atomic_rename(output_path, fs=output_filesystem) as temporary_path:
        pq.write_table(table, temporary_path, filesystem=output_filesystem, compression="zstd")
    counts = {
        "eval": EVAL_ROWS_PER_SOURCE,
        "train_750k": train_quota_750k,
        "train_3m": train_quota_3m,
    }
    return output_url, counts


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically to private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def main() -> None:
    """Build the exact evaluation set and both nested training rungs."""
    inventory = read_json(SOURCE_INVENTORY_URL)
    inventory_sources = {result["name"]: result for result in inventory["sources"] if result["usable"]}
    if len(inventory_sources) < MIN_SOURCES:
        raise ValueError(f"Only {len(inventory_sources)} inventory sources are usable")

    source_files = {}
    train_capacities = {}
    for index, (source, result) in enumerate(sorted(inventory_sources.items()), start=1):
        logger.info("Measuring source %d/%d: %s", index, len(inventory_sources), source)
        filesystem, files = source_file_capacities(result["main_output_dir"], source)
        capacity = min(sum(rows for _, rows in files), MAX_TRAIN_ROWS_PER_SOURCE + EVAL_ROWS_PER_SOURCE)
        if capacity < EVAL_ROWS_PER_SOURCE + 1:
            logger.warning("Dropping %s because it has only %d rows", source, capacity)
            continue
        source_files[source] = (filesystem, result["main_output_dir"], files)
        train_capacities[source] = capacity - EVAL_ROWS_PER_SOURCE

    if len(source_files) < MIN_SOURCES:
        raise ValueError(f"Only {len(source_files)} sources pass the evaluation-row gate")
    quotas_3m = allocate_balanced_quotas(train_capacities, TRAIN_TARGET_3M)
    quotas_750k = allocate_balanced_quotas(train_capacities, TRAIN_TARGET_750K)
    if any(quotas_750k[source] > quotas_3m[source] for source in source_files):
        raise ValueError("The 0.75M source quotas are not nested within the 3M quotas")

    source_reports = {}
    category_counts: Counter[str] = Counter()
    for index, source in enumerate(sorted(source_files), start=1):
        filesystem, main_output_dir, files = source_files[source]
        logger.info("Writing source %d/%d: %s", index, len(source_files), source)
        protocol = main_output_dir.partition("://")[0]
        selected_count = EVAL_ROWS_PER_SOURCE + quotas_3m[source]
        rows, selected_counts = selected_source_rows(filesystem, files, source, selected_count, protocol)
        output_url, counts = write_source_rows(source, rows, quotas_750k[source], quotas_3m[source])
        category = source_category(source).value
        category_counts[category] += 1
        source_reports[source] = {
            "category": category,
            "main_output_dir": main_output_dir,
            "output_url": output_url,
            "available_input_file_count": len(files),
            "available_input_rows": sum(row_count for _, row_count in files),
            "selected_input_files": [
                {
                    "path": f"{protocol}://{path}",
                    "total_rows": row_count,
                    "selected_rows": selected_counts.get(f"{protocol}://{path}", 0),
                }
                for path, row_count in files
                if selected_counts.get(f"{protocol}://{path}", 0)
            ],
            "counts": counts,
        }

    manifest = {
        "version": 2,
        "seed": SEED,
        "source_inventory_url": SOURCE_INVENTORY_URL,
        "source_count": len(source_reports),
        "minimum_source_count": MIN_SOURCES,
        "training_targets": {"750k": TRAIN_TARGET_750K, "3m": TRAIN_TARGET_3M},
        "evaluation_rows_per_source": EVAL_ROWS_PER_SOURCE,
        "survey_rows_per_source": SURVEY_ROWS_PER_SOURCE,
        "text_window_characters": TEXT_WINDOW_CHARS,
        "sampling_method": SAMPLING_METHOD,
        "sampling_blocks_per_source": SAMPLE_BLOCKS_PER_SOURCE,
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "category_source_counts": dict(sorted(category_counts.items())),
        "sources": source_reports,
    }
    manifest["sha256"] = manifest_digest(manifest)
    write_json(MANIFEST_URL, manifest)
    summary = {
        "manifest_url": MANIFEST_URL,
        "sha256": manifest["sha256"],
        "source_count": len(source_reports),
        "train_750k": sum(quotas_750k.values()),
        "train_3m": sum(quotas_3m.values()),
        "eval": len(source_reports) * EVAL_ROWS_PER_SOURCE,
        "category_source_counts": dict(sorted(category_counts.items())),
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_MANIFEST=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
