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

import io
import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import dupekit
import polars as pl
from fray import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import open_url, url_to_fs
from zephyr import Dataset, ShardInfo, ZephyrContext, counters, write_parquet_file
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file, load_file_batch
from zephyr.writers import ThreadedBatchWriter, ensure_parent_dir

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
        text = str(record[text_field])

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


# Records per vectorized whitespace-compaction batch. Bounds the transient
# buffer (~a few MB of references); the actual scan runs on the whole batch's
# text column in Rust.
_WHITESPACE_BATCH_SIZE = 8192


def _make_batched_whitespace_compactor(
    max_whitespace_run_chars: int,
) -> Callable[[Iterator[dict[str, Any]], ShardInfo], Iterator[dict[str, Any]]]:
    """Vectorized whitespace compactor, applied per shard via ``map_shard``.

    Scanning ``\\s{N+1,}`` over each ~KB text with per-record Python ``re.sub``
    dominates the normalize scatter stage (the unicode ``\\s`` scan runs on every
    record even though almost none have an over-long run). This runs the same
    scan over a whole batch's ``text`` column with polars (Rust regex, ~6x
    faster), then recomputes ``id`` only for the few rows whose text changed.

    ``replace_all(r"(\\s{N})\\s+", "${1}")`` keeps the first N whitespace chars of
    each over-long run and drops the rest — byte-identical to the per-record
    ``re.sub(r"\\s{N+1,}", lambda m: m.group(0)[:N], text)`` (verified across
    leading/trailing/mixed-unicode runs).
    """
    pattern = rf"(\s{{{max_whitespace_run_chars}}})\s+"

    def compact_shard(items: Iterator[dict[str, Any]], _: ShardInfo) -> Iterator[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= _WHITESPACE_BATCH_SIZE:
                yield from _compact_text_batch(batch, pattern)
                batch = []
        if batch:
            yield from _compact_text_batch(batch, pattern)

    return compact_shard


def _compact_text_batch(batch: list[dict[str, Any]], pattern: str) -> Iterator[dict[str, Any]]:
    compacted = pl.Series([r["text"] for r in batch], dtype=pl.Utf8).str.replace_all(pattern, "${1}").to_list()
    for record, new_text in zip(batch, compacted, strict=True):
        if new_text != record["text"]:
            counters.increment(COMPACTED_WHITESPACE_COUNTER)
            yield {**record, "text": new_text, "id": generate_id(new_text)}
        else:
            yield record


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
        main_path = f"{output_dir}/outputs/main/{shard_filename}"
        dup_path = f"{output_dir}/outputs/dups/{shard_filename}"

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


# ---------------------------------------------------------------------------
# Columnar two-stage normalize
#
# A vectorized variant of the dict pipeline that keeps data in Polars columns
# end-to-end (no per-record Python dict round-trip). Profiling showed ~-59% CPU
# vs the dict pipeline. It runs as two separate Zephyr executions on one
# ``ZephyrContext`` with a barrier between them:
#
#   STAGE 1 (scatter, parallelism = input files): stream each file as Arrow
#     batches, normalize columnarly (filter empty text, vectorized whitespace
#     compaction, per-row xxh3 id), route each row to a shard by ``hash(id)``,
#     and write per-(mapper, shard) Parquet chunk files.
#   STAGE 2 (reduce, parallelism = num_shards): glob each shard's chunk files,
#     concatenate, sort by ``id``, dedup, and write the main/dup partitions.
#
# Routing is self-consistent: the scatter side picks the shard and the reduce
# side simply globs that shard's directory, so only this routing matters — it
# need not match Zephyr's internal group_by hashing.
# ---------------------------------------------------------------------------

# Intermediate scatter output lives under this subdir of the output path.
_COLUMNAR_SCATTER_DIRNAME = "_columnar_scatter"

# Flush the per-shard scatter buffers once their accumulated (in-memory) size
# crosses this threshold. Bounds peak scatter memory regardless of file size.
_COLUMNAR_FLUSH_BYTES = 1024 * 1024 * 1024  # ~1 GB

# Transient column holding the per-row target shard during scatter. Dropped
# before any chunk is written, so it never reaches the reduce stage.
_COLUMNAR_SHARD_COL = "__columnar_shard"


def _write_polars_parquet(df: pl.DataFrame, path: str, row_group_size: int | None = None) -> None:
    """Write *df* to *path* as zstd Parquet via an in-memory buffer.

    Polars' direct cloud ``write_parquet`` occasionally fails with a generic
    error (cf. zephyr ``ScatterWriter``); buffering to ``BytesIO`` and writing
    the bytes through ``open_url`` is the robust pattern used elsewhere.

    ``row_group_size`` sizes row groups (e.g. one per shard, so the reduce side
    can skip non-target groups via predicate pushdown).
    """
    ensure_parent_dir(path)
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", row_group_size=row_group_size)
    with open_url(path, "wb") as f:
        f.write(buf.getvalue())


def _normalize_batch_columnar(
    df: pl.DataFrame,
    *,
    text_field: str,
    id_field: str | None,
    whitespace_pattern: str,
    bare: bool,
) -> pl.DataFrame:
    """Columnar equivalent of ``has_text`` + ``normalize_record`` + whitespace compaction.

    Filters rows with missing/blank text (counting ``normalize/empty_text_filtered``),
    compacts over-long whitespace runs, recomputes the xxh3 ``id`` from the
    compacted text, renames *id_field* → ``source_id``, and (unless *bare*)
    preserves all other original columns. Returns the normalized rows (possibly
    empty) without any routing column.
    """
    text_raw = df.get_column(text_field).cast(pl.Utf8)
    keep_mask = text_raw.is_not_null() & (text_raw.str.strip_chars() != "")
    dropped = int((~keep_mask).sum())
    if dropped:
        counters.increment("normalize/empty_text_filtered", dropped)

    df = df.filter(keep_mask)
    if df.height == 0:
        return df.clear()

    compacted = text_raw.filter(keep_mask).str.replace_all(whitespace_pattern, "${1}")
    texts = compacted.to_list()
    ids = [format(dupekit.hash_xxh3_128(t.encode("utf-8")), "032x") for t in texts]
    has_source = id_field is not None and id_field in df.columns

    if bare:
        out = pl.DataFrame({"id": ids, "text": pl.Series(texts, dtype=pl.Utf8)})
        if has_source:
            out = out.with_columns(df.get_column(id_field).alias("source_id"))
        return out

    drop_cols = set()
    if id_field is not None:
        drop_cols.add(id_field)
    if text_field != "text":
        drop_cols.add(text_field)
    keep_cols = [c for c in df.columns if c not in drop_cols]

    out = df.select(keep_cols).with_columns(
        pl.Series("text", texts, dtype=pl.Utf8),
        pl.Series("id", ids, dtype=pl.Utf8),
    )
    if has_source:
        out = out.with_columns(df.get_column(id_field).alias("source_id"))
    return out


def _make_columnar_scatter_fn(
    *,
    scatter_dir: str,
    num_shards: int,
    text_field: str,
    id_field: str | None,
    max_whitespace_run_chars: int,
    bare: bool,
) -> Callable[[Iterator[str], ShardInfo], Iterator[dict[str, Any]]]:
    """Build the STAGE 1 scatter ``map_shard`` function.

    Each mapper streams its assigned files as Arrow batches, normalizes them
    columnarly, routes rows to shards by ``hash(id) % num_shards``, and writes
    per-shard Parquet chunk files under ``{scatter_dir}/m{m}/s{shard}/c{chunk}``.
    Chunk filenames are deterministic given the (deterministic) input order, so
    a re-run after preemption overwrites identical files (idempotent).
    """
    whitespace_pattern = rf"(\s{{{max_whitespace_run_chars}}})\s+"

    def scatter_fn(files: Iterator[str], shard_info: ShardInfo) -> Iterator[dict[str, Any]]:
        m = shard_info.shard_idx
        buffer: list[pl.DataFrame] = []
        acc_bytes = 0
        chunk = 0
        total_rows = 0

        def flush() -> None:
            nonlocal acc_bytes, chunk, buffer
            if not buffer:
                return
            # One combined file per flush, sorted by shard with one row group per
            # shard, so the reduce side reads only its shard's group via predicate
            # pushdown (mirrors zephyr's ScatterWriter; avoids per-shard small-file
            # GCS overhead that would explode as num_shards grows).
            combined = (buffer[0] if len(buffer) == 1 else pl.concat(buffer)).sort(_COLUMNAR_SHARD_COL)
            n_sh = max(1, combined.get_column(_COLUMNAR_SHARD_COL).n_unique())
            path = f"{scatter_dir}/m{m:05d}/c{chunk:04d}.parquet"
            _write_polars_parquet(combined, path, row_group_size=max(1, combined.height // n_sh))
            buffer = []
            acc_bytes = 0
            chunk += 1

        for path in files:
            for batch in load_file_batch(path):
                out = _normalize_batch_columnar(
                    pl.from_arrow(batch),
                    text_field=text_field,
                    id_field=id_field,
                    whitespace_pattern=whitespace_pattern,
                    bare=bare,
                )
                if out.height == 0:
                    continue
                total_rows += out.height
                shards = [dupekit.hash_xxh3_128(i.encode("utf-8")) % num_shards for i in out.get_column("id").to_list()]
                out = out.with_columns(pl.Series(_COLUMNAR_SHARD_COL, shards, dtype=pl.Int64))
                buffer.append(out)
                acc_bytes += int(out.estimated_size())
                if acc_bytes >= _COLUMNAR_FLUSH_BYTES:
                    flush()
        flush()
        yield {"mapper": m, "rows": total_rows}

    return scatter_fn


def _columnar_chunk_paths(scatter_dir: str) -> list[str]:
    """Return the full paths of every mapper's combined scatter chunk files."""
    fs, resolved = url_to_fs(scatter_dir)
    protocol = scatter_dir.split("://")[0] if "://" in scatter_dir else ""
    matches = fs.glob(f"{resolved}/m*/c*.parquet")
    return [f"{protocol}://{p}" if protocol else p for p in matches]


def _make_columnar_reduce_fn(
    *,
    scatter_dir: str,
    output_dir: str,
    num_shards: int,
    dedup_mode: DedupMode,
) -> Callable[[Iterator[int], ShardInfo], Iterator[dict[str, Any]]]:
    """Build the STAGE 2 reduce ``map_shard`` function.

    Each reduce shard ``s`` globs its scatter chunk files, concatenates them,
    sorts by ``id``, and splits into main (first row per ``id``) and dup
    (the rest) outputs written to ``outputs/main`` / ``outputs/dups`` with the
    standard ``part-NNNNN-of-MMMMM`` filename. ``NONE`` dedup keeps all rows in
    main. The input ``ShardInfo.shard_idx`` is the sole shard selector; the
    iterator items are ignored.
    """

    def reduce_fn(_items: Iterator[int], shard_info: ShardInfo) -> Iterator[dict[str, Any]]:
        s = shard_info.shard_idx
        shard_filename = partition_filename(s, num_shards)
        main_path = f"{output_dir}/outputs/main/{shard_filename}"
        dup_path = f"{output_dir}/outputs/dups/{shard_filename}"

        paths = _columnar_chunk_paths(scatter_dir)
        if not paths:
            write_parquet_file([], output_path=main_path)
            write_parquet_file([], output_path=dup_path)
            yield {"main": main_path, "dup": dup_path, "main_count": 0, "dup_count": 0}
            return

        # Predicate pushdown: each chunk is sorted by shard with one row group per
        # shard, so this reads only shard ``s``'s rows from each mapper's file.
        df = (
            pl.scan_parquet(paths)
            .filter(pl.col(_COLUMNAR_SHARD_COL) == s)
            .drop(_COLUMNAR_SHARD_COL)
            .collect()
            .sort("id", maintain_order=True)
        )
        if dedup_mode == DedupMode.EXACT:
            ranked = df.with_columns(pl.int_range(pl.len()).over("id").alias("__rank"))
            main = ranked.filter(pl.col("__rank") == 0).drop("__rank")
            dups = ranked.filter(pl.col("__rank") > 0).drop("__rank")
        else:
            main = df
            dups = df.clear()

        # Write the deduped frames directly (no per-row Python round-trip).
        _write_polars_parquet(main, main_path)
        _write_polars_parquet(dups, dup_path)
        counters.increment("normalize/unique_records_out", main.height)
        counters.increment("normalize/duplicate_records_out", dups.height)
        yield {"main": main_path, "dup": dup_path, "main_count": main.height, "dup_count": dups.height}

    return reduce_fn


def _run_columnar_normalize(
    *,
    ctx: ZephyrContext,
    files: list[str],
    output_path: str,
    num_shards: int,
    text_field: str,
    id_field: str | None,
    dedup_mode: DedupMode,
    max_whitespace_run_chars: int,
    bare: bool,
) -> dict[str, int]:
    """Run the two-stage columnar normalize and return aggregated counters."""
    scatter_dir = f"{output_path}/{_COLUMNAR_SCATTER_DIRNAME}"

    scatter = Dataset.from_list(files).map_shard(
        _make_columnar_scatter_fn(
            scatter_dir=scatter_dir,
            num_shards=num_shards,
            text_field=text_field,
            id_field=id_field,
            max_whitespace_run_chars=max_whitespace_run_chars,
            bare=bare,
        )
    )
    scatter_outcome = ctx.execute(scatter)

    reduce = (
        Dataset.from_list(list(range(num_shards)))
        .reshard(num_shards)
        .map_shard(
            _make_columnar_reduce_fn(
                scatter_dir=scatter_dir,
                output_dir=output_path,
                num_shards=num_shards,
                dedup_mode=dedup_mode,
            )
        )
    )
    reduce_outcome = ctx.execute(reduce)

    merged: dict[str, int] = dict(scatter_outcome.counters)
    for key, value in reduce_outcome.counters.items():
        merged[key] = merged.get(key, 0) + value

    fs, resolved = url_to_fs(scatter_dir)
    if fs.exists(resolved):
        fs.rm(resolved, recursive=True)

    return merged


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
        .map_shard(_make_batched_whitespace_compactor(max_whitespace_run_chars))
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
    bare: bool = False,
    columnar: bool = False,
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
        columnar: Use the columnar two-stage pipeline (keeps data in Polars
            columns end-to-end, no per-record dict round-trip) instead of the
            dict pipeline. Produces byte-equivalent output; faster on Parquet
            inputs. Requires Parquet inputs (uses ``load_file_batch``).

    Returns:
        A :class:`NormalizedData` describing the output directories and
        aggregated zephyr counters.
    """
    resources = worker_resources or ResourceConfig(cpu=2, ram="32g", disk="10g")
    if max_workers is None:
        max_workers = 1024

    files = _discover_files(input_path, file_extensions=file_extensions)
    if not files:
        raise FileNotFoundError(f"No data files found under {input_path}")

    total_bytes = _compute_total_bytes(files)
    num_shards = max(1, total_bytes // target_partition_bytes)

    logger.info(
        "Normalizing %s → %s: %d files, %d bytes, %d shards (columnar=%s)",
        input_path,
        output_path,
        len(files),
        total_bytes,
        num_shards,
        columnar,
    )

    ctx_kwargs: dict = {"name": "normalize", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)

    if columnar:
        counters_dict = _run_columnar_normalize(
            ctx=ctx,
            files=files,
            output_path=output_path,
            num_shards=num_shards,
            text_field=text_field,
            id_field=id_field,
            dedup_mode=dedup_mode,
            max_whitespace_run_chars=max_whitespace_run_chars,
            bare=bare,
        )
    else:
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
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
    relative_input_path: str | None = None,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    bare: bool = False,
    columnar: bool = False,
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
        # ``os.path.join`` collapses redundant separators when ``download.output_path``
        # ends with ``/`` (e.g. ``override_output_path="gs://.../nemotro-cc-eeb783/"``);
        # naive f-string concatenation would yield ``gs://.../nemotro-cc-eeb783//<rel>``,
        # which ``_discover_files`` then fails to resolve on GCS.
        resolved_input = os.path.join(download.output_path, relative_input_path)
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
    # Only include bare/columnar in hash when set so default callers' hash_id
    # stays identical to pre-feature step specs (cache identity).
    if bare:
        hash_attrs["bare"] = bare
    if columnar:
        hash_attrs["columnar"] = columnar

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
            columnar=columnar,
        ),
        deps=[download],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
