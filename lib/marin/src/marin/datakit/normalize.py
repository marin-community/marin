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

import functools
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
from fray.types import ResourceConfig
from pydantic import BaseModel, ValidationInfo, model_validator
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.expr import col
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file, load_file_batch
from zephyr.sql import SqlScalarFunction, quote_identifier, sql
from zephyr.writers import ThreadedBatchWriter, batchify, write_parquet_file

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
EMPTY_TEXT_FILTERED_COUNTER = "normalize/empty_text_filtered"

_INPUT_BATCH_ROWS = 8192
_SCHEMA_SAMPLE_FILES = 2
_COMPACTED_COL = "__normalize_compacted"
_EMPTY_TEXT_COL = "__normalize_empty_text"
_DUPLICATE_COL = "__normalize_duplicate"

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


def _input_batches(path: str) -> Iterator[pa.RecordBatch]:
    if path.endswith(".parquet"):
        yield from load_file_batch(path)
        return
    for records in batchify(load_file(path), n=_INPUT_BATCH_ROWS):
        yield pa.RecordBatch.from_pylist(list(records))


def _infer_input_schema(paths: list[str]) -> pa.Schema:
    """Infer one Arrow schema from the first batch of two non-empty files."""
    schemas: list[pa.Schema] = []
    sampled_paths: list[str] = []
    for path in paths:
        batch = next(_input_batches(path), None)
        if batch is None or len(batch) == 0:
            continue
        schemas.append(batch.schema)
        sampled_paths.append(path)
        if len(schemas) == _SCHEMA_SAMPLE_FILES:
            break

    if not schemas:
        return pa.schema([])
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as err:
        raise ValueError(f"Could not infer one Arrow schema from sampled files {sampled_paths}: {err}") from err


def _iter_input_batches(path: str, *, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Load input batches under one sampled schema contract."""
    for batch in _input_batches(path):
        try:
            unified = pa.unify_schemas([schema, batch.schema], promote_options="permissive")
            if not schema.equals(unified, check_metadata=True):
                raise pa.ArrowInvalid("batch would widen the sampled schema")
            yield pa.RecordBatch.from_struct_array(
                pa.StructArray.from_arrays(
                    [
                        (
                            batch.column(field.name)
                            if field.name in batch.schema.names
                            else pa.nulls(len(batch), field.type)
                        )
                        for field in schema
                    ],
                    fields=list(schema),
                )
            ).cast(schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as err:
            raise ValueError(
                f"Input batch from {path} does not match the sampled schema. "
                f"Sampled schema:\n{schema}\nBatch schema:\n{batch.schema}"
            ) from err


def _text_array(values: pa.Array) -> pa.Array:
    return pa.array(
        [None if value is None else _text_from_value(value) for value in values.to_pylist()],
        type=pa.string(),
    )


def _id_array(values: pa.Array) -> pa.Array:
    return pa.array([None if value is None else generate_id(value) for value in values.to_pylist()], type=pa.string())


def _finalize_normalized_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    empty = int(pc.sum(batch.column(_EMPTY_TEXT_COL)).as_py() or 0)
    if empty:
        counters.pipeline.update_counter(EMPTY_TEXT_FILTERED_COUNTER, empty)
    compacted = int(pc.sum(batch.column(_COMPACTED_COL)).as_py() or 0)
    if compacted:
        counters.pipeline.update_counter(COMPACTED_WHITESPACE_COUNTER, compacted)
    batch = batch.filter(pc.invert(batch.column(_EMPTY_TEXT_COL)))
    return batch.select([name for name in batch.schema.names if name not in {_COMPACTED_COL, _EMPTY_TEXT_COL}])


def _normalize_query(
    schema: pa.Schema,
    *,
    text_field: str,
    id_field: str | None,
    max_whitespace_run_chars: int,
    kept_fields: tuple[str, ...],
):
    if text_field not in schema.names:
        return sql(
            f"SELECT CAST(NULL AS VARCHAR) AS id, CAST(NULL AS VARCHAR) AS text, "
            f"false AS {quote_identifier(_COMPACTED_COL)}, "
            f"true AS {quote_identifier(_EMPTY_TEXT_COL)} FROM input"
        )

    text_sql = quote_identifier(text_field)
    raw_text_col = quote_identifier("__normalize_raw_text")
    compacted_text_col = quote_identifier("__normalize_text")
    whitespace_pattern = rf"(\s{{{max_whitespace_run_chars}}})\s+"

    projection = [quote_identifier(name) for name in kept_fields]
    projection.extend(
        [
            f"normalize_id({compacted_text_col}) AS id",
            f"{compacted_text_col} AS text",
        ]
    )
    if id_field is not None and id_field in schema.names:
        projection.append(f"{quote_identifier(id_field)} AS source_id")
    projection.append(f"{compacted_text_col} != {raw_text_col} AS {quote_identifier(_COMPACTED_COL)}")
    projection.append(
        f"{raw_text_col} IS NULL OR regexp_like({raw_text_col}, '^\\s*$') " f"AS {quote_identifier(_EMPTY_TEXT_COL)}"
    )

    return sql(
        f"""
        WITH normalized AS (
            SELECT *, normalize_text({text_sql}) AS {raw_text_col}
            FROM input
        ), compacted AS (
            SELECT *, regexp_replace({raw_text_col}, '{whitespace_pattern}', '$1', 'g') AS {compacted_text_col}
            FROM normalized
        )
        SELECT {", ".join(projection)}
        FROM compacted
        """,
        scalar_functions=(
            SqlScalarFunction("normalize_text", _text_array, (schema.field(text_field).type,), pa.string()),
            SqlScalarFunction("normalize_id", _id_array, (pa.string(),), pa.string()),
        ),
    )


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
) -> Callable[[Iterator[pa.RecordBatch], ShardInfo], Iterator[dict[str, dict[str, Any]]]]:
    """Return a ``map_shard`` function that fans SQL-marked batches to two outputs.

    Each shard writes two files concurrently via ``ThreadedBatchWriter`` so the
    producer isn't blocked on I/O. Yields a single manifest per shard containing
    the ``write_parquet_file`` result (``{"path", "count"}``) for each branch.
    """

    # TODO (rav): consider whether we want to generalize this in the future.

    def split_writer(
        batches: Iterator[pa.RecordBatch],
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

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(dup_path, "dup")) as dup_writer,
        ):

            def output_batch(batch: pa.RecordBatch, mask: pa.Array, columns: list[str]) -> pa.RecordBatch:
                output = batch.filter(mask).select(columns)
                if output_schema is None:
                    return output
                return output.select(output_schema.names).cast(output_schema)

            for batch in batches:
                duplicate = batch.column(_DUPLICATE_COL)
                columns = [name for name in batch.schema.names if name != _DUPLICATE_COL]
                duplicate_count = int(pc.sum(duplicate).as_py() or 0)
                unique_count = len(batch) - duplicate_count
                if unique_count:
                    counters.pipeline.update_counter("normalize/unique_records_out", unique_count)
                    main_writer.submit(output_batch(batch, pc.invert(duplicate), columns))
                if duplicate_count:
                    counters.pipeline.update_counter("normalize/duplicate_records_out", duplicate_count)
                    dup_writer.submit(output_batch(batch, duplicate, columns))

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
    """Build the Zephyr pipeline that normalizes *files* into *output_dir*."""
    input_schema = _infer_input_schema(files)
    excluded = set(drop_fields) | {"id", "text", "source_id"}
    if id_field is not None:
        excluded.add(id_field)
    if text_field != "text":
        excluded.add(text_field)
    kept_fields = () if bare else tuple(name for name in input_schema.names if name not in excluded)
    normalize_query = _normalize_query(
        input_schema,
        text_field=text_field,
        id_field=id_field,
        max_whitespace_run_chars=max_whitespace_run_chars,
        kept_fields=kept_fields,
    )
    duplicate_sql = (
        f"row_number() OVER (PARTITION BY id ORDER BY id) > 1 AS {quote_identifier(_DUPLICATE_COL)}"
        if dedup_mode == DedupMode.EXACT
        else f"false AS {quote_identifier(_DUPLICATE_COL)}"
    )

    return (
        Dataset.from_list(files)
        .flat_map(functools.partial(_iter_input_batches, schema=input_schema))
        .sql(normalize_query.text, scalar_functions=normalize_query.scalar_functions)
        .map(_finalize_normalized_batch)
        .group_by(
            key=col("id"),
            reducer=sql(f"SELECT *, {duplicate_sql} FROM input"),
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
