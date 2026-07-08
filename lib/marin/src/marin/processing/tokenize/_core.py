# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared internals for tokenization.

Used by:
* :func:`marin.processing.tokenize.tokenize.tokenize` — legacy raw → Levanter store path.
* :func:`marin.processing.tokenize.attributes.tokenize_attributes` — datakit Stage A
  (NormalizedData → attribute parquet).
* :func:`marin.processing.tokenize.store_builder.build_from_datasets` — datakit Stage B
  (tokenized records → Levanter store).

Public API lives in those modules; helpers here are package-private.
"""
import json
import logging
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence

import braceexpand
import fsspec
import pyarrow.parquet as pq
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text import LmDatasetFormatBase, preprocessor_for_format
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem import StoragePath, url_to_fs
from zephyr import Dataset, counters, zephyr_worker_ctx
from zephyr.dataset import FileEntry
from zephyr.readers import InputFileSpec

from marin.datakit.normalize import generate_id

logger = logging.getLogger(__name__)

MIN_GROUP_BYTES = 100_000_000  # 100 MB floor to avoid degenerate tiny shards
# Empirical upper bound on the zephyr window size (see
# https://github.com/marin-community/marin/issues/2829#issuecomment-3963661943).
_MAX_WINDOW_SIZE = 64

_TOKENIZE_EXTENSIONS = ["json.{gz,zst,zstd}", "jsonl.{gz,zst,zstd}", "parquet"]


def avg_parquet_row_group_rows(path: str) -> int | None:
    """Return the mean rows-per-row-group from ``path``.

    Returns ``None`` if the file has no row groups (empty parquet footer).
    """
    with StoragePath(path).open("rb") as f:
        meta = pq.ParquetFile(f).metadata
    if meta.num_row_groups == 0:
        return None
    return max(1, meta.num_rows // meta.num_row_groups)


def compute_target_group_bytes(total_input_bytes: int, max_workers: int) -> int:
    """Compute target group size to produce approximately ``max_workers`` groups.

    Applies a floor of ``MIN_GROUP_BYTES`` to avoid degenerate tiny shards.
    """
    return max(total_input_bytes // max_workers, MIN_GROUP_BYTES)


def drop_sidecars(files: list[FileEntry]) -> list[FileEntry]:
    """Drop dot-prefixed files — Marin metadata sidecars (``.provenance.json``,
    ``.artifact.json``, …), never training data even with a data extension."""
    return [f for f in files if not os.path.basename(f.path).startswith(".")]


def glob_with_sizes(patterns: list[str]) -> list[FileEntry]:
    """Glob patterns and return FileEntry objects (spec + size).

    Uses fsspec ``glob(detail=True)`` which returns file metadata from the same
    list-objects API call — no per-file stat RPCs needed. Works for gs://, hf://, s3://, local.
    """
    results: list[FileEntry] = []
    for pattern in patterns:
        pattern = re.sub(r"(?<!:)//+", "/", pattern)
        fs, _ = url_to_fs(pattern)
        protocol = fsspec.core.split_protocol(pattern)[0]
        for expanded in braceexpand.braceexpand(pattern):
            detail = fs.glob(expanded, detail=True)
            for path, info in detail.items():
                full = f"{protocol}://{path}" if protocol else path
                results.append(FileEntry(spec=InputFileSpec(path=full), size=info.get("size", 0)))
    return results


def expand_tokenize_paths(input_paths: list[str]) -> list[str]:
    """Expand input paths into glob patterns for tokenizable file types.

    Directories get expanded to recursive globs for each supported extension.
    Concrete paths/patterns pass through unchanged.
    """
    patterns: list[str] = []
    for path in input_paths:
        assert path != "/"
        if path.endswith("/") or StoragePath(path).isdir():
            logger.info(f"Getting all {_TOKENIZE_EXTENSIONS} files in {path}")
            for ex in _TOKENIZE_EXTENSIONS:
                patterns.append(os.path.join(path, f"**/*.{ex}"))
        else:
            patterns.append(path)
    return patterns


def bundle_files_by_size(files: list[FileEntry], max_bytes: int) -> Iterator[list[str]]:
    """Bundle files into groups, with each group having a total size less than ``max_bytes``."""
    current_group: list[str] = []
    current_size = 0

    for f in files:
        if current_size + f.size >= max_bytes and current_group:
            yield current_group
            current_group = []
            current_size = 0
        current_group.append(f.path)
        current_size += f.size

    if current_group:
        yield current_group


def attach_id(record: dict, text_field: str = "text") -> dict:
    """Ensure record has an ``id`` field.

    If ``id`` is already present and non-null, leave the record unchanged.
    Otherwise, generate a deterministic xxh3_128 id via
    :func:`marin.datakit.normalize.generate_id` from ``record[text_field]``,
    falling back to a JSON serialization of the record if ``text_field`` is
    absent.

    Datakit-normalized inputs always carry ``id`` and skip the hashing branch.
    """
    if record.get("id") is not None:
        return record
    if text_field in record and record[text_field] is not None:
        return {**record, "id": generate_id(str(record[text_field]))}
    return {**record, "id": generate_id(json.dumps(record, sort_keys=True, default=str))}


class IdPreservingPreprocessor:
    """Wrap a Levanter ``BatchProcessor`` to thread input ``id`` onto each output.

    Levanter's ``BatchProcessor`` interface explicitly allows non-1:1 input→output
    (see :class:`levanter.data._preprocessor.BatchProcessor`). All currently used
    processors (``BatchTokenizer``, ``ChatProcessor``, ``PrebuiltCacheProcessor``,
    ``PreferenceChatProcessor``) are 1:1, but a future packing/SFT-splitting
    processor would silently misalign ids if we naively zipped. This wrapper
    asserts the 1:1 invariant so misalignment fails loudly.
    """

    def __init__(self, inner: BatchProcessor):
        self.inner = inner

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        outputs = self.inner(batch)
        # BatchResult is Sequence[U] | Mapping[str, Sequence] (struct-of-arrays)
        if isinstance(outputs, Mapping):
            keys = list(outputs.keys())
            n_out = len(outputs[keys[0]]) if keys else 0
            outputs_list: list[dict] = [{k: outputs[k][i] for k in keys} for i in range(n_out)]
        else:
            outputs_list = list(outputs)
            n_out = len(outputs_list)

        if n_out != len(batch):
            raise RuntimeError(
                f"IdPreservingPreprocessor: 1:1 input→output expected, got "
                f"{len(batch)} input → {n_out} output records from "
                f"{type(self.inner).__name__}. id alignment cannot be preserved; "
                "if this processor packs or splits records, route ids via a custom path."
            )

        return [{**out, "id": rec["id"]} for rec, out in zip(batch, outputs_list, strict=True)]


def tokenize_batches_with_id(
    *,
    data_format: LmDatasetFormatBase,
    batches: Iterator[Sequence[dict]],
) -> Iterator[dict]:
    """Tokenize batches and yield ``{id, input_ids, ...}`` per input doc.

    Each input record must already carry ``id`` (apply :func:`attach_id` upstream).
    The worker tokenizer config is read from zephyr's shared context — caller is
    responsible for ``ctx.put('tokenizer_name', ...)`` and
    ``ctx.put('tokenizer_backend', ...)`` before pipeline execution.
    """
    ctx = zephyr_worker_ctx()
    name = ctx.get_shared("tokenizer_name")
    backend = ctx.get_shared("tokenizer_backend")
    # load_tokenizer is @lru_cache, so this only loads once per worker process.
    tokenizer: MarinTokenizer = load_tokenizer(name, backend=backend)
    inner = preprocessor_for_format(data_format, tokenizer)
    # Levanter's BatchTokenizer ships ``long_string_workaround`` opt-in but the
    # behavior is desirable always: per-record texts above ``_workaround_len``
    # (10K chars) get split at safe whitespace boundaries before the underlying
    # ``encode_batch`` is called, then merged back. No-op for short records.
    # Without this, a single multi-MB outlier passes one giant string to the
    # Rust tokenizer and OOMs the worker.
    if hasattr(inner, "_long_string_workaround"):
        inner._long_string_workaround = True
    processor = IdPreservingPreprocessor(inner)

    batch_count = 0
    record_count = 0
    token_count = 0
    start_time = time.monotonic()

    for batch in batches:
        batch_count += 1
        for record in processor(batch):
            n_tokens = len(record.get("input_ids", []))
            counters.pipeline.update_counter("tokenize/docs_out", 1)
            counters.pipeline.update_counter("tokenize/tokens_out", n_tokens)
            record_count += 1
            token_count += n_tokens
            yield record
        if batch_count % 10 == 0:
            elapsed = time.monotonic() - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0
            doc_per_sec = record_count / elapsed if elapsed > 0 else 0
            avg_tok_per_doc = token_count / record_count if record_count > 0 else 0
            logger.info(
                f"Tokenized {batch_count:,} batches, {record_count:,} docs, {token_count:,} tokens "
                f"in {elapsed:.1f}s ({tok_per_sec:,.0f} tokens/s, {doc_per_sec:,.1f} docs/s, "
                f"{avg_tok_per_doc:,.0f} avg tokens/doc)"
            )

    elapsed = time.monotonic() - start_time
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0
    doc_per_sec = record_count / elapsed if elapsed > 0 else 0
    avg_tok_per_doc = token_count / record_count if record_count > 0 else 0
    logger.info(
        f"Tokenization done: {batch_count:,} batches, {record_count:,} docs, {token_count:,} tokens "
        f"in {elapsed:.1f}s ({tok_per_sec:,.0f} tokens/s, {doc_per_sec:,.1f} docs/s, "
        f"{avg_tok_per_doc:,.0f} avg tokens/doc)"
    )


def parquet_window_hint(file_groups: list[list[str]]) -> str | None:
    """Return a sample parquet path from ``file_groups`` if any, else ``None``.

    Used to align zephyr's window and Levanter's cache batch with parquet
    row-group size on parquet inputs; ignored for non-parquet inputs.
    """
    return next((p for group in file_groups for p in group if p.endswith(".parquet")), None)


def resolve_window_and_batch(
    sample_parquet_path: str | None,
    requested_batch_size: int | None,
) -> tuple[int, int | None]:
    """Pick zephyr window and Levanter batch sizes.

    For parquet sources, align both with the parquet row-group size so each unit
    of work is exactly one row group end-to-end. Non-parquet inputs fall through
    to defaults.
    """
    window_size = _MAX_WINDOW_SIZE
    batch_size = requested_batch_size
    if sample_parquet_path is None:
        return window_size, batch_size
    avg_rg_rows = avg_parquet_row_group_rows(sample_parquet_path)
    if avg_rg_rows is None:
        return window_size, batch_size
    half_rg = max(avg_rg_rows // 2, 1)
    window_size = min(half_rg, _MAX_WINDOW_SIZE)
    if requested_batch_size is None:
        batch_size = half_rg
    logger.info(
        "Parquet source: avg rows/row-group=%d (from %s) → window=%d, levanter batch_size=%s",
        avg_rg_rows,
        sample_parquet_path,
        window_size,
        batch_size,
    )
    return window_size, batch_size


def tokenize_pipeline(
    ds: Dataset,
    *,
    data_format: LmDatasetFormatBase,
    text_field: str = "text",
    sample_count: int | None,
    sample_parquet_path: str | None,
    levanter_batch_size: int | None,
) -> tuple[Dataset, int | None]:
    """Build the tokenize pipeline tail.

    Attaches ``id`` to each input record, optionally subsamples per shard, windows,
    and tokenizes. Returns the dataset of ``{id, input_ids, ...}`` records and the
    chosen Levanter cache batch size (``None`` keeps Levanter's default).
    """
    window_size, batch_size = resolve_window_and_batch(sample_parquet_path, levanter_batch_size)

    ds = ds.map(lambda r, tf=text_field: attach_id(r, text_field=tf))

    if sample_count is not None:
        ds = ds.take_per_shard(sample_count)

    return (
        ds.window(window_size).map_shard(
            lambda batches, _, fmt=data_format: tokenize_batches_with_id(data_format=fmt, batches=batches)
        ),
        batch_size,
    )
