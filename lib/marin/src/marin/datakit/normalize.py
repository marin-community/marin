# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize raw downloaded data into the datakit standard Parquet format.

Reads raw files (JSONL, Parquet, etc.) discovered recursively under a single
input directory, transforms each record into the standard schema (``id``,
``text``, plus all original columns), deduplicates by content, sorts by ``id``
within each partition, and writes Parquet output with
``part-{shard}-of-{total}`` naming.

All discovered files are merged into a single output: main records land in
``<output_path>/outputs/main/`` and (when dedup is enabled) duplicates land in
``<output_path>/outputs/dups/``. Input directory structure is not preserved.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import dupekit
from fray import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ShardInfo, ZephyrContext, counters, write_parquet_file
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file
from zephyr.writers import ThreadedBatchWriter

from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

# Default cap on the longest consecutive whitespace run in a document.
# Runs exceeding this are compacted to this length at normalization time.
# Pathologically long whitespace runs (e.g. multi-MB runs from broken
# HTML→text extraction, cf. #4588) can OOM downstream tokenization.
# 128 matches the longest whitespace run that Llama's tokenizer collapses
# into a single token, so capping here is lossless for that tokenizer.
DEFAULT_MAX_WHITESPACE_RUN_CHARS = 128

# Counter name for documents that had whitespace runs compacted.
COMPACTED_WHITESPACE_COUNTER = "datakit_normalize_compacted_whitespace"


class DedupMode(StrEnum):
    """How aggressively to deduplicate records during normalization.

    ``EXACT`` drops records with duplicate ``id`` (i.e. byte-identical text)
    within each output shard.  ``NONE`` skips the dedup pass entirely.
    """

    NONE = "none"
    EXACT = "exact"


class NormalizedData(BaseModel):
    """Outcome of :func:`normalize_to_parquet`: a single normalized dataset.

    Persisted as the step's ``.artifact`` so counters and output paths are
    available to downstream consumers without re-running the pipeline. Load
    via ``Artifact.load(step, NormalizedData)``.

    Attributes:
        main_output_dir: Directory containing the main output Parquet files.
        dup_output_dir: Directory containing the duplicate side output Parquet files.
        counters: Aggregated zephyr counters.
    """

    version: str = "v1"
    main_output_dir: str
    dup_output_dir: str
    counters: dict[str, int]


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
    1. Extracts ``text`` from *text_field*.
    2. Generates a deterministic ``id`` via xxh3_128.
    3. If *id_field* exists in the record, preserves it as ``source_id``.
    4. Keeps all other columns.

    Records with missing or blank text must be filtered out before calling
    the returned function.
    """

    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        # --- text ---
        text = str(record[text_field])

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


def _discover_files(
    input_path: str,
    file_extensions: tuple[str, ...] | None = None,
) -> list[str]:
    """Walk *input_path* recursively and return a sorted flat list of data files.

    Only files with matching extensions are included; dotfiles and hidden
    directories are skipped.

    Args:
        input_path: Root directory to walk.
        file_extensions: Tuple of file extensions to include (e.g.
            ``(".parquet",)``).  Defaults to all extensions supported by
            ``zephyr.readers.load_file``.
    """
    extensions = file_extensions or SUPPORTED_EXTENSIONS
    fs, resolved = url_to_fs(input_path)
    protocol = input_path.split("://")[0] if "://" in input_path else ""

    def _full_path(p: str) -> str:
        return f"{protocol}://{p}" if protocol else p

    discovered: list[str] = []
    for root, _dirs, files in fs.walk(resolved):
        rel_root = os.path.relpath(root, resolved)
        parts = [] if rel_root == "." else rel_root.split(os.sep)
        if any(p.startswith(".") for p in parts):
            continue
        for fname in files:
            if fname.startswith("."):
                continue
            if not fname.endswith(extensions):
                continue
            discovered.append(_full_path(os.path.join(root, fname)))

    discovered.sort()
    return discovered


def _compute_total_bytes(file_paths: list[str]) -> int:
    """Sum the byte sizes of all *file_paths*."""
    total = 0
    for path in file_paths:
        fs, resolved = url_to_fs(path)
        total += fs.size(resolved)
    return total


def _make_whitespace_compactor(max_whitespace_run_chars: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a map function that compacts consecutive whitespace runs exceeding the limit.

    Any run of whitespace longer than *max_whitespace_run_chars* is truncated to
    that length (preserving the original whitespace characters). Affected records
    are counted via the ``COMPACTED_WHITESPACE_COUNTER`` Zephyr counter, and the
    ``id`` is recomputed to reflect the new text.
    """
    pattern = re.compile(r"\s{" + str(max_whitespace_run_chars + 1) + r",}")

    def compact(record: dict[str, Any]) -> dict[str, Any]:
        text = record["text"]
        compacted = pattern.sub(lambda m: m.group(0)[:max_whitespace_run_chars], text)
        if len(compacted) != len(text):
            counters.increment(COMPACTED_WHITESPACE_COUNTER)
            record = {**record, "text": compacted, "id": generate_id(compacted)}
        return record

    return compact


@dataclass
class MainOutput:
    """Wraps a unique record destined for the main output shard."""

    data: dict[str, Any]


@dataclass
class ExactDupSideOutput:
    """Wraps a duplicate record destined for the side (dups) output shard."""

    data: dict[str, Any]


def _make_split_writer(
    output_dir: str,
) -> Callable[[Iterator[MainOutput | ExactDupSideOutput], ShardInfo], Iterator[dict[str, dict[str, Any]]]]:
    """Return a ``map_shard`` function that fans records out to main and dup Parquet files.

    Each shard writes two files concurrently via ``ThreadedBatchWriter`` so the
    producer isn't blocked on I/O. Yields a single manifest per shard containing
    the ``write_parquet_file`` result (``{"path", "count"}``) for each branch.
    """

    # TODO (rav): consider whether we want to generalize this in the future.

    def split_writer(
        records: Iterator[MainOutput | ExactDupSideOutput],
        shard: ShardInfo,
    ) -> Iterator[dict[str, dict[str, Any]]]:
        # NOTE: we could add support for split_existing - but we intentionally don't
        main_path = f"{output_dir}/outputs/main/part-{shard.shard_idx:05d}-of-{shard.total_shards:05d}.parquet"
        dup_path = f"{output_dir}/outputs/dups/part-{shard.shard_idx:05d}-of-{shard.total_shards:05d}.parquet"

        # Results are populated by each writer thread. Safe to read only after
        # the ThreadedBatchWriter context exits (which joins the thread).
        results: dict[str, dict[str, Any]] = {}

        def write_to(path: str, key: str) -> Callable[[Iterable[dict[str, Any]]], None]:
            def _fn(items: Iterable[dict[str, Any]]) -> None:
                results[key] = write_parquet_file(items, output_path=path)

            return _fn

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(dup_path, "dup")) as dup_writer,
        ):
            for item in records:
                if isinstance(item, MainOutput):
                    counters.increment("normalize/unique_records_out")
                    main_writer.submit(item.data)
                else:
                    counters.increment("normalize/duplicate_records_out")
                    dup_writer.submit(item.data)

        yield results

    return split_writer


def _build_pipeline(
    files: list[str],
    output_dir: str,
    num_shards: int,
    text_field: str,
    id_field: str | None,
    dedup_mode: DedupMode,
    max_whitespace_run_chars: int,
) -> Dataset:
    """Build the Zephyr pipeline that normalizes *files* into *output_dir*."""
    normalize_record = _make_normalize_fn(text_field, id_field)

    def dedup(_key: str, items: Iterator[dict[str, Any]]) -> Iterator[MainOutput | ExactDupSideOutput]:
        """Drop adjacent duplicate ids. Items arrive sorted by id via sort_by."""
        prev_id: str | None = None
        for record in items:
            rid = record["id"]
            if rid != prev_id:
                prev_id = rid
                yield MainOutput(data=record)
            else:
                yield ExactDupSideOutput(data=record)

    def passthrough(_key: str, items: Iterator[dict[str, Any]]) -> Iterator[MainOutput]:
        """Yield items unchanged; used when dedup is disabled."""
        yield from (MainOutput(data=item) for item in items)

    def has_text(record: dict[str, Any]) -> bool:
        text = record.get(text_field)
        if text is None or str(text).strip() == "":
            counters.increment("normalize/empty_text_filtered")
            return False
        return True

    reducers: dict[DedupMode, Callable] = {DedupMode.EXACT: dedup, DedupMode.NONE: passthrough}

    return (
        Dataset.from_list(files)
        .flat_map(load_file)
        .filter(has_text)
        .map(normalize_record)
        .map(_make_whitespace_compactor(max_whitespace_run_chars))
        .group_by(
            key=lambda r: r["id"],
            reducer=reducers[dedup_mode],
            sort_by=lambda r: r["id"],
            num_output_shards=num_shards,
        )
        .map_shard(_make_split_writer(output_dir))
    )


def normalize_to_parquet(
    *,
    input_path: str,
    output_path: str,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    max_whitespace_run_chars: int = DEFAULT_MAX_WHITESPACE_RUN_CHARS,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
) -> NormalizedData:
    """Normalize raw downloaded data to the datakit standard Parquet format.

    Discovers all data files recursively under *input_path*, merges them into a
    single Zephyr pipeline that normalizes records (``id``, ``text``, preserves
    all other columns), optionally deduplicates by content per *dedup_mode*,
    sorts by ``id``, and writes Parquet partitions sized by
    *target_partition_bytes*. Input directory structure is not preserved.

    Args:
        input_path: Root directory containing raw downloaded data.
        output_path: Directory for normalized Parquet output. Main records are
            written to ``<output_path>/outputs/main/`` and (when dedup is
            enabled) duplicates to ``<output_path>/outputs/dups/``.
        text_field: Name of the field containing primary text content.
        id_field: Name of the field containing the source ID (renamed to
            ``source_id``).  If the field is absent from a record, it is
            silently skipped.
        target_partition_bytes: Target size in bytes per output partition.
            Used to compute the number of output shards.
        max_whitespace_run_chars: Compact any consecutive whitespace run
            longer than this many characters down to this length.
            Pathologically long whitespace runs (e.g. multi-MB runs from
            broken HTML→text extraction, cf. #4588) can OOM downstream
            tokenization. Affected records are counted via the
            ``datakit_normalize_compacted_whitespace`` Zephyr counter.
        worker_resources: Per-worker resource request for the Zephyr pipeline.
            Defaults to 2 CPU / 16GB RAM / 10GB disk, sized for
            ``target_partition_bytes`` of 256MB.  Scale up when increasing
            partition size.
        max_workers: Maximum number of Zephyr workers for the pipeline.
            Defaults to Zephyr's own default (128 for distributed backends).
        file_extensions: Tuple of file extensions to include (e.g.
            ``(".parquet",)``).  Defaults to all extensions supported by
            ``zephyr.readers.load_file``.
        dedup_mode: How to deduplicate records within each output shard.
            ``EXACT`` (the default) drops records with duplicate ``id`` values
            (i.e. byte-identical text).  ``NONE`` skips dedup and preserves
            all input records.

    Returns:
        A :class:`NormalizedData` describing the output directories and
        aggregated zephyr counters.
    """
    resources = worker_resources or ResourceConfig(cpu=2, ram="16g", disk="10g")

    files = _discover_files(input_path, file_extensions=file_extensions)
    if not files:
        raise FileNotFoundError(f"No data files found under {input_path}")

    total_bytes = _compute_total_bytes(files)
    num_shards = max(1, total_bytes // target_partition_bytes)

    logger.info(
        "Normalizing %s → %s: %d files, %d bytes, %d shards",
        input_path,
        output_path,
        len(files),
        total_bytes,
        num_shards,
    )

    pipeline = _build_pipeline(
        files,
        output_path,
        num_shards,
        text_field,
        id_field,
        dedup_mode,
        max_whitespace_run_chars,
    )
    ctx_kwargs: dict = {"name": "normalize", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)
    counters_dict = dict(outcome.counters)

    total_in = counters_dict.get("zephyr/records_in", 0)
    total_filtered = counters_dict.get("normalize/empty_text_filtered", 0)
    if total_in > 0 and total_filtered == total_in:
        raise ValueError(
            f"All {total_in} records were filtered out due to missing/empty text. "
            f"Your data is either invalid or you have selected the wrong column, "
            f"current column: {text_field!r}"
        )

    return NormalizedData(
        main_output_dir=os.path.join(output_path, "outputs/main"),
        dup_output_dir=os.path.join(output_path, "outputs/dups"),
        counters=counters_dict,
    )


def normalize_step(
    *,
    name: str,
    download: StepSpec,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    max_whitespace_run_chars: int = DEFAULT_MAX_WHITESPACE_RUN_CHARS,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    override_output_path: str | None = None,
    relative_input_path: str | None = None,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
) -> StepSpec:
    """Create a StepSpec that normalizes downloaded data to Parquet.

    Args:
        name: Step name (e.g. ``"fineweb/normalize"``).
        download: Upstream download step whose output_path is the input.
        text_field: Name of the field containing primary text content.
        id_field: Name of the field containing the source ID.
        target_partition_bytes: Target size per output partition.
        worker_resources: Per-worker resource request for the Zephyr pipeline.
            See :func:`normalize_to_parquet` for the default.
        max_workers: Maximum number of Zephyr workers. Defaults to Zephyr's
            own default (128 for distributed backends).
        override_output_path: Override the computed output path.
        relative_input_path: Override the input path relative to the download output.
            Useful when normalizing a subdirectory of the download output.
        file_extensions: Tuple of file extensions to include (e.g.
            ``(".parquet",)``).  Defaults to all extensions supported by
            ``zephyr.readers.load_file``.
        dedup_mode: How to deduplicate records within each output shard.
            Defaults to ``DedupMode.EXACT``; use ``DedupMode.NONE`` to skip.
    """
    if relative_input_path:
        resolved_input = f"{download.output_path}/{relative_input_path}"
    else:
        resolved_input = download.output_path

    return StepSpec(
        name=name,
        fn=lambda output_path: normalize_to_parquet(
            input_path=resolved_input,
            output_path=output_path,
            text_field=text_field,
            id_field=id_field,
            target_partition_bytes=target_partition_bytes,
            max_whitespace_run_chars=max_whitespace_run_chars,
            worker_resources=worker_resources,
            max_workers=max_workers,
            file_extensions=file_extensions,
            dedup_mode=dedup_mode,
        ),
        deps=[download],
        hash_attrs={
            "text_field": text_field,
            "id_field": id_field,
            "target_partition_bytes": target_partition_bytes,
            "max_whitespace_run_chars": max_whitespace_run_chars,
            "relative_input_path": relative_input_path,
            "file_extensions": file_extensions,
            "dedup_mode": dedup_mode,
        },
        override_output_path=override_output_path,
    )
