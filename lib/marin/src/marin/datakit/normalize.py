# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize raw downloaded data into the datakit standard Parquet format.

Reads raw files (JSONL, Parquet, etc.) discovered recursively under a single
input directory, transforms each record into the standard schema (``id``,
``text``, plus all original columns), deduplicates by content, sorts by ``id``
within each partition, and writes Parquet output with
``part-{shard}-of-{total}`` naming.

An explicit output schema may select a subset of the transformed columns.

All discovered files are merged into a single output: main records land in
``<output_path>/outputs/main/`` and (when dedup is enabled) duplicates land in
``<output_path>/outputs/dups/``. Input directory structure is not preserved.
"""

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Any

import dupekit
import polars as pl
import pyarrow as pa
from fray.types import ResourceConfig
from pydantic import BaseModel, ValidationInfo, model_validator
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.expr import col
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file, load_file_batch
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from marin.datakit import partition_filename
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import ARTIFACT_LOAD_CONTEXT_KEY
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
NORMALIZED_DATA_VERSION = "v2"

# The hash attributes every normalize step declares. They fix the identity of a
# normalized artifact, so the set is effectively frozen — dropping or renaming one
# re-keys every existing output. A step that does not declare them was not built
# here and carries none of normalize's guarantees.
NORMALIZE_IDENTITY_ATTRS = frozenset(
    {"text_field", "id_field", "target_partition_bytes", "max_whitespace_run_chars", "dedup_mode"}
)
_DUPLICATE_OUTPUT_COLUMN = "__marin_duplicate__"


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
        dup_output_dir: Directory containing the duplicate side output Parquet
            files. A v2 artifact stores both directories relative to
            ``MARIN_PREFIX`` when they are under the active prefix.
        counters: Aggregated zephyr counters.
    """

    version: str = NORMALIZED_DATA_VERSION
    main_output_dir: DatakitArtifactPath
    dup_output_dir: DatakitArtifactPath
    counters: dict[str, int | float]

    @model_validator(mode="before")
    @classmethod
    def _resolve_artifact_paths(cls, value: object, info: ValidationInfo) -> object:
        if not info.context or not info.context.get(ARTIFACT_LOAD_CONTEXT_KEY):
            return value
        if not isinstance(value, dict):
            return value

        version = value.get("version", "v1")
        if version not in ("v1", NORMALIZED_DATA_VERSION):
            raise ValueError(f"Unsupported NormalizedData version: {version!r}")

        loaded = dict(value)
        loaded["version"] = NORMALIZED_DATA_VERSION
        return loaded


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


# Rows per DataFrame batch when loading non-Parquet inputs (JSONL/Vortex).
# Parquet uses native row-group batches via ``load_file_batch``.
_INPUT_BATCH_ROWS = 8192


def _align_dataframe_vertical_relaxed(
    df: pl.DataFrame,
    schema: pl.Schema,
) -> tuple[pl.DataFrame, pl.Schema]:
    """Reorder/cast *df* to a ``vertical_relaxed`` unify with *schema*.

    Requires the same column names as *schema* (order is normalized). Returns the
    aligned frame and the widened schema. Raises when column sets differ or when
    dtypes have no common ``vertical_relaxed`` supertype.
    """
    expected = list(schema.names())
    got = set(df.columns)
    expected_set = set(expected)
    if got != expected_set:
        missing = sorted(expected_set - got)
        extra = sorted(got - expected_set)
        raise ValueError(f"column set mismatch: missing={missing}, extra={extra}")
    ordered = df.select(expected)
    widened = pl.concat([pl.DataFrame(schema=schema), ordered.clear()], how="vertical_relaxed").schema
    return ordered.cast(dict(widened)), widened


def _iter_input_batches(path: str) -> Iterator[pl.DataFrame]:
    """Yield Polars DataFrames from one input file.

    Parquet goes through ``load_file_batch`` (one batch per row group), converted
    to DataFrames. Other supported formats are loaded as dicts and packed into
    DataFrames of ``_INPUT_BATCH_ROWS`` so the rest of the pipeline can stay
    columnar.

    Non-Parquet batches use ``infer_schema_length=None`` within each chunk. The
    first chunk's schema is the file contract: later chunks must be unifiable
    under ``pl.concat(..., how="vertical_relaxed")`` (same columns; null/integer
    widenings). Incompatible drift raises with a hint to enlarge
    ``_INPUT_BATCH_ROWS`` when a field appears only after the first chunk.
    """
    if path.endswith(".parquet"):
        for batch in load_file_batch(path):
            yield pl.DataFrame(batch)
        return

    schema: pl.Schema | None = None
    rows_before = 0
    batch: list[dict[str, Any]] = []

    def flush() -> Iterator[pl.DataFrame]:
        nonlocal schema, rows_before, batch
        if not batch:
            return
        df = pl.DataFrame(batch, infer_schema_length=None)
        n = len(batch)
        batch = []
        if schema is None:
            schema = df.schema
            rows_before += n
            yield df
            return
        try:
            aligned, schema = _align_dataframe_vertical_relaxed(df, schema)
        except (ValueError, pl.exceptions.ComputeError, pl.exceptions.SchemaError, pl.exceptions.ShapeError) as err:
            raise ValueError(
                f"Row structure changed after {rows_before} rows in {path} "
                f"(batch size {_INPUT_BATCH_ROWS}). Running schema: {dict(schema)}. "
                f"This batch could not be unified with pl.concat(how='vertical_relaxed'): {err}. "
                f"If a field appears only later in the file, increase _INPUT_BATCH_ROWS so the "
                f"first batch includes it. If row shape genuinely varies through the file, use "
                f"Parquet or a homogeneous dump."
            ) from err
        rows_before += n
        yield aligned

    for record in load_file(path):
        batch.append(record)
        if len(batch) >= _INPUT_BATCH_ROWS:
            yield from flush()
    yield from flush()


def _as_text_series(series: pl.Series) -> pl.Series:
    """Coerce a column to Utf8, decoding Binary with ``errors='replace'``."""
    if series.dtype == pl.Binary:
        return pl.Series(
            (_text_from_value(v) if v is not None else None for v in series.to_list()),
            dtype=pl.Utf8,
        )
    if series.dtype in (pl.Utf8, pl.String):
        return series
    return series.cast(pl.Utf8)


def _make_normalize_batch_fn(
    text_field: str,
    id_field: str | None,
    max_whitespace_run_chars: int,
    bare: bool = False,
    drop_fields: tuple[str, ...] = (),
) -> Callable[[pa.RecordBatch | pl.DataFrame], Iterator[pl.DataFrame]]:
    """Return a batch → zero-or-one normalized DataFrame transform.

    Drops blank ``text_field`` rows (``normalize/empty_text_filtered``),
    compacts over-long whitespace runs (``COMPACTED_WHITESPACE_COUNTER``),
    assigns deterministic ``id`` via xxh3_128 of the compacted text, and
    renames *id_field* → ``source_id`` when present. Other columns are kept
    unless *bare* or *drop_fields* removes them.

    *bare* keeps only ``id``, ``text``, and (when present) ``source_id`` —
    required when extra columns vary across shards and would break a uniform
    Parquet schema. Yields nothing when every row is filtered out.
    """
    whitespace_pattern = rf"(\s{{{max_whitespace_run_chars}}})\s+"

    def normalize_batch(batch: pa.RecordBatch | pl.DataFrame) -> Iterator[pl.DataFrame]:
        if isinstance(batch, pa.RecordBatch):
            df = pl.from_arrow(batch)
            assert isinstance(df, pl.DataFrame)
        else:
            df = batch

        if text_field not in df.columns:
            if df.height:
                counters.pipeline.update_counter("normalize/empty_text_filtered", df.height)
            return

        text_raw = _as_text_series(df.get_column(text_field))
        keep_mask = text_raw.is_not_null() & (text_raw.str.strip_chars() != "")
        dropped = int((~keep_mask).sum())
        if dropped:
            counters.pipeline.update_counter("normalize/empty_text_filtered", dropped)

        df = df.filter(keep_mask)
        if df.height == 0:
            return

        text = text_raw.filter(keep_mask)
        compacted = text.str.replace_all(whitespace_pattern, "${1}")
        changed = int((compacted != text).sum())
        if changed:
            counters.pipeline.update_counter(COMPACTED_WHITESPACE_COUNTER, changed)

        texts = compacted.to_list()
        ids = [generate_id(t) for t in texts]
        has_source = id_field is not None and id_field in df.columns

        if bare:
            out = pl.DataFrame({"id": ids, "text": pl.Series(texts, dtype=pl.Utf8)})
            if has_source:
                out = out.with_columns(df.get_column(id_field).alias("source_id"))
            yield out
            return

        drop_cols = set(drop_fields)
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
        yield out

    return normalize_batch


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


def _make_split_writer(
    output_dir: str,
    output_schema: pa.Schema | None = None,
) -> Callable[[Iterator[pl.DataFrame], ShardInfo], Iterator[dict[str, dict[str, Any]]]]:
    """Return a ``map_shard`` function that fans records out to main and dup Parquet files.

    Each shard writes two files concurrently via ``ThreadedBatchWriter`` so the
    producer isn't blocked on I/O. Yields a single manifest per shard containing
    the ``write_parquet_file`` result (``{"path", "count"}``) for each branch.
    """

    # TODO (rav): consider whether we want to generalize this in the future.

    def split_writer(
        frames: Iterator[pl.DataFrame],
        shard: ShardInfo,
    ) -> Iterator[dict[str, dict[str, Any]]]:
        # NOTE: we could add support for split_existing - but we intentionally don't
        shard_filename = partition_filename(shard.shard_idx, shard.total_shards)
        main_path = prefix_join(output_dir, f"outputs/main/{shard_filename}")
        dup_path = prefix_join(output_dir, f"outputs/dups/{shard_filename}")

        # Results are populated by each writer thread. Safe to read only after
        # the ThreadedBatchWriter context exits (which joins the thread).
        results: dict[str, dict[str, Any]] = {}

        def write_to(path: str, key: str) -> Callable[[Iterable[pa.RecordBatch]], None]:
            def _fn(items: Iterable[pa.RecordBatch]) -> None:
                results[key] = write_parquet_file(items, output_path=path, schema=output_schema)

            return _fn

        def submit_frame(frame: pl.DataFrame, writer: ThreadedBatchWriter) -> None:
            if frame.is_empty():
                return
            table = frame.drop(_DUPLICATE_OUTPUT_COLUMN).to_arrow()
            if output_schema is not None:
                table = table.select(output_schema.names).cast(output_schema)
            for batch in table.to_batches():
                writer.submit(batch)

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(dup_path, "dup")) as dup_writer,
        ):
            for frame in frames:
                main = frame.filter(~pl.col(_DUPLICATE_OUTPUT_COLUMN))
                duplicates = frame.filter(pl.col(_DUPLICATE_OUTPUT_COLUMN))
                counters.pipeline.update_counter("normalize/unique_records_out", main.height)
                counters.pipeline.update_counter("normalize/duplicate_records_out", duplicates.height)
                submit_frame(main, main_writer)
                submit_frame(duplicates, dup_writer)

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
    drop_fields: tuple[str, ...] = (),
    output_schema: pa.Schema | None = None,
) -> Dataset:
    """Build the Zephyr pipeline that normalizes *files* into *output_dir*.

    Loads inputs as columnar batches, normalizes each batch, then
    ``group_by(key=col("id"), sort_by=col("id"), ...)`` so Scatter can ingest
    DataFrames directly.
    """
    normalize_batch = _make_normalize_batch_fn(
        text_field,
        id_field,
        max_whitespace_run_chars,
        bare=bare,
        drop_fields=drop_fields,
    )

    def dedup(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns((pl.int_range(pl.len()) > 0).alias(_DUPLICATE_OUTPUT_COLUMN))

    def passthrough(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns(pl.lit(False).alias(_DUPLICATE_OUTPUT_COLUMN))

    reducers: dict[DedupMode, Callable[[pl.DataFrame], pl.DataFrame]] = {
        DedupMode.EXACT: dedup,
        DedupMode.NONE: passthrough,
    }

    # Prefer Dataset.load_parquet(batch_mode=True) when every input is Parquet
    # so the plan can use LoadFileOp; otherwise pack non-Parquet formats into
    # Arrow batches via flat_map.
    ds: Dataset
    if files and all(path.endswith(".parquet") for path in files):
        ds = Dataset.from_list(files).load_parquet(batch_mode=True)
    else:
        ds = Dataset.from_list(files).flat_map(_iter_input_batches)

    return (
        ds.flat_map(normalize_batch)
        .group_by(
            key=col("id"),
            reducer=reducers[dedup_mode],
            sort_by=col("id"),
            num_output_shards=num_shards,
        )
        .map_shard(_make_split_writer(output_dir, output_schema=output_schema))
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
    drop_fields: tuple[str, ...] = (),
    output_schema: pa.Schema | None = None,
) -> NormalizedData:
    """Normalize raw downloaded data to the datakit standard Parquet format.

    Discovers all data files recursively under *input_path*, merges them into a
    single Zephyr pipeline that normalizes records (``id``, ``text``, and other
    columns unless removed by *bare* or *drop_fields*), optionally deduplicates by
    content per *dedup_mode*, sorts by ``id``, and writes
    Parquet partitions sized by *target_partition_bytes*. Input directory
    structure is not preserved.

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
        drop_fields: Source fields to remove while preserving other metadata.
        output_schema: Optional schema for normalized output records.

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
        drop_fields=drop_fields,
        output_schema=output_schema,
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
    drop_fields: tuple[str, ...] = (),
    output_schema: pa.Schema | None = None,
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
        drop_fields: Source fields to remove while preserving other metadata.
        output_schema: Optional schema for normalized output records.
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
    if drop_fields:
        hash_attrs["drop_fields"] = drop_fields
    if output_schema is not None:
        hash_attrs["output_schema"] = str(output_schema)
    assert NORMALIZE_IDENTITY_ATTRS <= hash_attrs.keys()
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
            drop_fields=drop_fields,
            output_schema=output_schema,
        ),
        deps=[download],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
