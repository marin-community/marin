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

import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import dupekit
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from marin.datakit import partition_filename
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

# Default Zephyr worker cap. Sized well above Zephyr's own default (128) because
# a single normalize spans thousands of shards over very large staged dumps.
DEFAULT_MAX_WORKERS = 1024


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
    via ``Artifact.from_path(step, NormalizedData)``.

    Attributes:
        main_output_dir: Directory containing the main output Parquet files.
        dup_output_dir: Directory containing the duplicate side output Parquet files.
        counters: Aggregated zephyr counters.
    """

    version: str = "v1"
    main_output_dir: str
    dup_output_dir: str
    counters: dict[str, int | float]


def generate_id(text: str) -> str:
    """Generate a deterministic document ID from text content.

    Uses xxh3_128 (consistent with dupekit's deduplication pipeline) and
    returns a zero-padded 32-character hex string.
    """
    return format(dupekit.hash_xxh3_128(text.encode("utf-8")), "032x")


def _text_from_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _make_normalize_fn(
    text_field: str,
    id_field: str,
    bare: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a record-level transform function.

    The returned function:
    1. Extracts ``text`` from *text_field*.
    2. Generates a deterministic ``id`` via xxh3_128.
    3. If *id_field* exists in the record, preserves it as ``source_id``.
    4. Keeps all other columns unless *bare* is set (see below).

    *bare* takes the strict path: drop every column that isn't ``id``,
    ``text``, or ``source_id``. Use this for sources whose extra columns
    vary across shards (e.g. starcoderdata's 87 language subdirs each
    ship a different set of GitHub-meta columns, or proof-pile-2's
    nested ``meta`` dict with optional-typed fields); the parquet writer
    can't add columns mid-write and the reduce stage can't widen
    null-vs-typed, so a uniform schema is the only safe option.

    Records with missing or blank text must be filtered out before calling
    the returned function.
    """

    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        # --- text ---
        text = _text_from_value(record[text_field])

        # --- source_id (skip silently if id_field absent) ---
        source_id = record.get(id_field)

        # --- build output ---
        out: dict[str, Any] = {}

        if not bare:
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


# Env var that ferries set on test/smoke runs to bound the input set on
# very large staged dumps. Read at execution by ``_discover_files``; not
# exposed as a public API parameter so production callers can't stumble into
# it. If unset, no truncation. If set to a positive int, truncate the sorted
# file list to that many files. Any other value raises.
_FERRY_TEST_MAX_FILES_ENV = "FERRY_TEST_MAX_FILES"


def _ferry_test_max_files() -> int | None:
    raw = os.environ.get(_FERRY_TEST_MAX_FILES_ENV)
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except ValueError as e:
        raise RuntimeError(f"{_FERRY_TEST_MAX_FILES_ENV}={raw!r} is not an integer") from e
    if n <= 0:
        raise RuntimeError(f"{_FERRY_TEST_MAX_FILES_ENV}={n} must be a positive integer")
    return n


def _discover_files(
    input_path: str,
    file_extensions: tuple[str, ...] | None = None,
) -> list[str]:
    """Walk *input_path* recursively and return a sorted flat list of data files.

    Only files with matching extensions are included; dotfiles and hidden
    directories are skipped. When the ``FERRY_TEST_MAX_FILES`` env var is set
    to a positive integer, the sorted list is truncated to that many entries —
    a smoke/test-only knob that bypasses any caller's intent, used by the
    canary ferries to bound oversized staged dumps.
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
    cap = _ferry_test_max_files()
    if cap is not None and cap < len(discovered):
        logger.warning(
            "_discover_files: respecting %s=%d env var; truncating discovered file list from %d to %d "
            "(testing/smoke-only knob)",
            _FERRY_TEST_MAX_FILES_ENV,
            cap,
            len(discovered),
            cap,
        )
        discovered = discovered[:cap]
    return discovered


def _compute_total_bytes(file_paths: list[str]) -> int:
    """Sum the byte sizes of all *file_paths*."""
    return sum(StoragePath(path).size() for path in file_paths)


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
            counters.pipeline.update_counter(COMPACTED_WHITESPACE_COUNTER, 1)
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
        shard_filename = partition_filename(shard.shard_idx, shard.total_shards)
        main_path = prefix_join(output_dir, f"outputs/main/{shard_filename}")
        dup_path = prefix_join(output_dir, f"outputs/dups/{shard_filename}")

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
                    counters.pipeline.update_counter("normalize/unique_records_out", 1)
                    main_writer.submit(item.data)
                else:
                    counters.pipeline.update_counter("normalize/duplicate_records_out", 1)
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
    bare: bool = False,
) -> Dataset:
    """Build the Zephyr pipeline that normalizes *files* into *output_dir*."""
    normalize_record = _make_normalize_fn(text_field, id_field, bare=bare)

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
        if text is None or _text_from_value(text).strip() == "":
            counters.pipeline.update_counter("normalize/empty_text_filtered", 1)
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
    max_workers: int = DEFAULT_MAX_WORKERS,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    bare: bool = False,
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
            Defaults to 2 CPU / 32GB RAM / 10GB disk, sized for
            ``target_partition_bytes`` of 256MB plus headroom for heavier
            sources (mid-tier subsets that don't get a per-subset override).
            Scale up when increasing partition size.
        max_workers: Maximum number of Zephyr workers for the pipeline.
            Defaults to 1024.
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
    resources = worker_resources or ResourceConfig(cpu=2, ram="32g", disk="10g")

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
        bare=bare,
    )
    ctx = ZephyrContext(name="normalize", resources=resources, max_workers=max_workers)
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
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
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
    max_workers: int = DEFAULT_MAX_WORKERS,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
    relative_input_path: str | None = None,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    bare: bool = False,
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
        max_workers: Maximum number of Zephyr workers. Defaults to
            ``DEFAULT_MAX_WORKERS`` (1024).
        output_path_prefix: Optional prefix for the normalized step output.
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
        # ``prefix_join`` yields exactly one separator even when ``download.output_path``
        # ends with ``/`` (e.g. ``gs://.../nemotro-cc-eeb783/``); a naive f-string join
        # would leave the doubled ``//`` that ``_discover_files`` then fails to resolve on GCS.
        resolved_input = prefix_join(download.output_path, relative_input_path)
    else:
        resolved_input = download.output_path

    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "id_field": id_field,
        "target_partition_bytes": target_partition_bytes,
        "max_whitespace_run_chars": max_whitespace_run_chars,
        "relative_input_path": relative_input_path,
        "file_extensions": file_extensions,
        "dedup_mode": dedup_mode,
    }
    # Only include bare in hash when set so default callers' hash_id stays
    # identical to pre-feature step specs (cache identity).
    if bare:
        hash_attrs["bare"] = bare
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
            bare=bare,
        ),
        deps=[download],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
