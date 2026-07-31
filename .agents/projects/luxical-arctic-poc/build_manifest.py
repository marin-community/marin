# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the fixed, nested dataset for the Luxical scaling ladder."""

import hashlib
import json
import logging
import posixpath
import re
from collections import Counter, defaultdict
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
    SEED,
    SOURCE_INVENTORY_URL,
    SURVEY_ROWS_PER_SOURCE,
    TEXT_WINDOW_CHARS,
    TRAIN_TARGET_3M,
    TRAIN_TARGET_750K,
    source_category,
)

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
RESULT_FILE = Path("/tmp/luxical-arctic-manifest")
REQUIRED_COLUMNS = frozenset(("id", "text"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


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


def source_file_capacities(main_output_dir: str, source: str) -> tuple[Any, list[tuple[str, int]]]:
    """Return a stable file order and enough row capacity for one source."""
    filesystem, paths = parquet_paths(main_output_dir)
    rng = np.random.default_rng(stable_seed(f"files:{source}"))
    ordered_paths = [paths[index] for index in rng.permutation(len(paths))]
    target = MAX_TRAIN_ROWS_PER_SOURCE + EVAL_ROWS_PER_SOURCE
    files = []
    row_count = 0
    for path in ordered_paths:
        with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
            columns = frozenset(parquet_file.schema_arrow.names)
            if not REQUIRED_COLUMNS.issubset(columns):
                raise ValueError(f"Required columns are missing from {path}: {columns}")
            rows = parquet_file.metadata.num_rows
        if rows == 0:
            continue
        files.append((path, rows))
        row_count += rows
        if row_count >= target:
            break
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


def selected_source_rows(
    filesystem: Any,
    files: list[tuple[str, int]],
    source: str,
    row_count: int,
    protocol: str,
) -> list[dict[str, Any]]:
    """Read one deterministic random source sample."""
    rng = np.random.default_rng(stable_seed(f"rows:{source}"))
    rows = []
    remaining = row_count
    for path, file_rows in files:
        take = min(remaining, file_rows)
        if not take:
            break
        positions = rng.choice(file_rows, size=take, replace=False)
        rows.extend(selected_file_rows(filesystem, path, positions, protocol))
        remaining -= take
    if remaining:
        raise ValueError(f"Source {source} is missing {remaining} requested rows")
    split_rng = np.random.default_rng(stable_seed(f"split:{source}"))
    return [rows[index] for index in split_rng.permutation(len(rows))]


def document_view(text: str) -> str:
    """Return fixed head, middle, and tail views of a document."""
    if len(text) <= 3 * TEXT_WINDOW_CHARS:
        return text
    middle_start = len(text) // 2 - TEXT_WINDOW_CHARS // 2
    return "\n".join(
        (
            text[:TEXT_WINDOW_CHARS],
            text[middle_start : middle_start + TEXT_WINDOW_CHARS],
            text[-TEXT_WINDOW_CHARS:],
        )
    )


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
                "in_survey": is_eval and selection_rank < SURVEY_ROWS_PER_SOURCE,
                "raw_characters": len(raw_text),
                "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized_text(raw_text).encode()).hexdigest(),
                "text": view,
            }
        )
    table = pa.Table.from_pylist(output_rows)
    output_url = f"{MANIFEST_ROOT}/sources/{safe_source_name(source)}.parquet"
    output_filesystem, output_path = fsspec.core.url_to_fs(output_url)
    pq.write_table(table, output_path, filesystem=output_filesystem, compression="zstd")
    counts = {
        "eval": EVAL_ROWS_PER_SOURCE,
        "train_750k": train_quota_750k,
        "train_3m": train_quota_3m,
        "survey": SURVEY_ROWS_PER_SOURCE,
    }
    return output_url, counts


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object to private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path, "w") as file:
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
        rows = selected_source_rows(filesystem, files, source, selected_count, protocol)
        output_url, counts = write_source_rows(source, rows, quotas_750k[source], quotas_3m[source])
        category = source_category(source).value
        category_counts[category] += 1
        source_reports[source] = {
            "category": category,
            "main_output_dir": main_output_dir,
            "output_url": output_url,
            "selected_input_files": [{"path": path, "rows": row_count} for path, row_count in files],
            "counts": counts,
        }

    manifest = {
        "version": 1,
        "seed": SEED,
        "source_inventory_url": SOURCE_INVENTORY_URL,
        "source_count": len(source_reports),
        "minimum_source_count": MIN_SOURCES,
        "training_targets": {"750k": TRAIN_TARGET_750K, "3m": TRAIN_TARGET_3M},
        "evaluation_rows_per_source": EVAL_ROWS_PER_SOURCE,
        "survey_rows_per_source": SURVEY_ROWS_PER_SOURCE,
        "text_window_characters": TEXT_WINDOW_CHARS,
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
