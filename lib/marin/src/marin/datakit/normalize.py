# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize raw downloaded data into the datakit standard Parquet format.

Reads raw files (JSONL, Parquet, etc.) from an input directory, transforms each
record into the standard schema (``id``, ``text``, plus all original columns),
deduplicates by content, sorts by ``id`` within each partition, and writes
Parquet output with ``part-{shard}-of-{total}`` naming.

Directory structure from the download is preserved: each subdirectory gets its
own set of partitions sized by ``target_partition_bytes``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import dupekit
from rigging.filesystem import url_to_fs
from marin.execution.step_spec import StepSpec
from fray.v2 import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file

logger = logging.getLogger(__name__)


def generate_id(text: str) -> str:
    """Generate a deterministic document ID from text content.

    Uses xxh3_128 (consistent with dupekit's deduplication pipeline) and
    returns a zero-padded 32-character hex string.
    """
    return format(dupekit.hash_xxh3_128(text.encode("utf-8")), "032x")


def _make_normalize_fn(
    text_field: str,
    id_field: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a record-level transform function.

    The returned function:
    1. Extracts ``text`` from *text_field* (raises on missing/empty).
    2. Generates a deterministic ``id`` via xxh3_128.
    3. If *id_field* exists in the record, preserves it as ``source_id``.
    4. Keeps all other columns.
    """

    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        # --- text ---
        text = record.get(text_field)
        if text is None or not str(text).strip():
            raise ValueError(f"Record missing or empty text in field {text_field!r}: {record!r:.200}")
        text = str(text)

        # --- source_id (skip silently if id_field absent) ---
        source_id = record.get(id_field)

        # --- build output ---
        out: dict[str, Any] = {}

        # Copy all original columns except the ones we're replacing
        for k, v in record.items():
            if k == id_field:
                continue
            if k == text_field and text_field != "text":
                continue
            out[k] = v

        out["id"] = generate_id(text)
        out["text"] = text
        if source_id is not None:
            out["source_id"] = source_id

        return out

    return normalize_record


def _discover_file_groups(
    input_path: str,
) -> dict[str, list[str]]:
    """Walk *input_path* and group data files by their subdirectory.

    Returns a mapping from relative subdirectory (``""`` for root) to a sorted
    list of file paths.  Only files with extensions supported by
    ``zephyr.readers.load_file`` are included; dotfiles and ``.metrics``
    directories are skipped.
    """
    fs, resolved = url_to_fs(input_path)
    protocol = input_path.split("://")[0] if "://" in input_path else ""

    def _full_path(p: str) -> str:
        return f"{protocol}://{p}" if protocol else p

    groups: dict[str, list[str]] = {}

    for root, _dirs, files in fs.walk(resolved):
        # Skip hidden / metrics directories
        rel_root = os.path.relpath(root, resolved)
        if rel_root == ".":
            rel_root = ""
        parts = rel_root.split(os.sep)
        if any(p.startswith(".") for p in parts if p):
            continue

        for fname in sorted(files):
            if fname.startswith("."):
                continue
            if not fname.endswith(SUPPORTED_EXTENSIONS):
                continue
            full = _full_path(os.path.join(root, fname))
            groups.setdefault(rel_root, []).append(full)

    # Sort files within each group for determinism
    for file_list in groups.values():
        file_list.sort()

    return groups


def _compute_total_bytes(file_paths: list[str]) -> int:
    """Sum the byte sizes of all *file_paths*."""
    total = 0
    for path in file_paths:
        fs, resolved = url_to_fs(path)
        total += fs.size(resolved)
    return total


def _build_pipeline(
    files: list[str],
    output_dir: str,
    num_shards: int,
    text_field: str,
    id_field: str | None,
) -> Dataset:
    """Build a single Zephyr pipeline for one subdirectory."""
    normalize_record = _make_normalize_fn(text_field, id_field)

    def dedup_and_sort(_key: int, items: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Deduplicate by id. Items arrive sorted by id via sort_by."""
        prev_id: str | None = None
        for record in items:
            rid = record["id"]
            if rid != prev_id:
                prev_id = rid
                yield record

    return (
        Dataset.from_list(files)
        .flat_map(load_file)
        .map(normalize_record)
        .group_by(
            key=lambda r: int(r["id"], 16) % num_shards,
            reducer=dedup_and_sort,
            sort_by=lambda r: r["id"],
            num_output_shards=num_shards,
        )
        .write_parquet(
            f"{output_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet",
            skip_existing=True,
        )
    )


def normalize_to_parquet(
    *,
    input_path: str,
    output_path: str,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
) -> None:
    """Normalize raw downloaded data to the datakit standard Parquet format.

    Discovers all data files under *input_path*, groups them by subdirectory,
    and launches one Zephyr pipeline per subdirectory concurrently.  Each
    pipeline normalizes records (``id``, ``text``, preserves all other columns),
    deduplicates by content, sorts by ``id``, and writes Parquet partitions
    sized by *target_partition_bytes*.

    Args:
        input_path: Root directory containing raw downloaded data.
        output_path: Root directory for normalized Parquet output.
        text_field: Name of the field containing primary text content.
        id_field: Name of the field containing the source ID (renamed to
            ``source_id``).  If the field is absent from a record, it is
            silently skipped.
        target_partition_bytes: Target size in bytes per output partition.
            Used to compute the number of output shards per subdirectory.
    """
    file_groups = _discover_file_groups(input_path)
    if not file_groups:
        raise FileNotFoundError(f"No data files found under {input_path}")

    logger.info("Discovered %d subdirectories under %s", len(file_groups), input_path)

    def _run_subdir(subdir: str, files: list[str]) -> None:
        total_bytes = _compute_total_bytes(files)
        num_shards = max(1, total_bytes // target_partition_bytes)
        output_dir = os.path.join(output_path, subdir) if subdir else output_path

        logger.info(
            "Normalizing %s → %s: %d files, %d bytes, %d shards",
            os.path.join(input_path, subdir) if subdir else input_path,
            output_dir,
            len(files),
            total_bytes,
            num_shards,
        )

        pipeline = _build_pipeline(files, output_dir, num_shards, text_field, id_field)
        ctx = ZephyrContext(
            name=f"normalize-{subdir.replace('/', '-') if subdir else 'all'}",
            resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
        )
        ctx.execute(pipeline)

    # Launch all subdirectory pipelines concurrently
    with ThreadPoolExecutor(max_workers=len(file_groups)) as pool:
        futures = {pool.submit(_run_subdir, subdir, files): subdir for subdir, files in file_groups.items()}
        for future in as_completed(futures):
            subdir = futures[future]
            future.result()  # Propagate exceptions
            logger.info("Completed normalization for %s", os.path.join(output_path, subdir) if subdir else output_path)


def normalize_step(
    *,
    name: str,
    download: StepSpec,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    override_output_path: str | None = None,
    input_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that normalizes downloaded data to Parquet.

    Args:
        name: Step name (e.g. ``"fineweb/normalize"``).
        download: Upstream download step whose output_path is the input.
        text_field: Name of the field containing primary text content.
        id_field: Name of the field containing the source ID.
        target_partition_bytes: Target size per output partition.
        override_output_path: Override the computed output path.
        input_path: Override the input path. Defaults to ``download.output_path``.
            Useful when normalizing a subdirectory of the download output.
    """
    resolved_input = input_path or download.output_path

    return StepSpec(
        name=name,
        fn=lambda output_path: normalize_to_parquet(
            input_path=resolved_input,
            output_path=output_path,
            text_field=text_field,
            id_field=id_field,
            target_partition_bytes=target_partition_bytes,
        ),
        deps=[download],
        hash_attrs={
            "text_field": text_field,
            "id_field": id_field,
            "target_partition_bytes": target_partition_bytes,
            "input_path": resolved_input,
        },
        override_output_path=override_output_path,
    )
