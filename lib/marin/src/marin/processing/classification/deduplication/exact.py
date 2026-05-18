# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import itertools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from zephyr import ZephyrContext, counters, write_parquet_file
from zephyr.dataset import Dataset
from zephyr.readers import iter_parquet_row_groups, open_file

from marin.processing.classification.deduplication.dedup_commons import (
    DEFAULT_FILETYPES,
    DedupMode,
    _collect_input_files,
    _find_base_path,
    _get_extension,
    _init_wandb,
    _load_batches,
    finalize_dedup,
    make_document_dedup_aggregator,
)
from marin.utils import rebase_file_path

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SPLIT_TARGET_BYTES = 256 * 1024 * 1024  # 256 MB target per parquet split


@dataclass(frozen=True)
class ParquetSplit:
    path: str
    file_idx: int
    row_start: int | None = None
    row_end: int | None = None


def _compute_parquet_splits(path: str, file_idx: int) -> Iterator[ParquetSplit]:
    """Split a parquet file into ~256 MB chunks based on row group sizes.

    Non-parquet files emit a single split with no row range.
    """
    if not path.endswith(".parquet"):
        raise ValueError(f"Expected a parquet file, got {path}")

    with open_file(path, "rb") as f:
        metadata = pq.ParquetFile(f).metadata

    cumulative_rows = 0
    split_start_row = 0
    split_bytes = 0

    for i in range(metadata.num_row_groups):
        rg_meta = metadata.row_group(i)
        rg_bytes = rg_meta.total_byte_size

        if split_bytes > 0 and split_bytes + rg_bytes > _SPLIT_TARGET_BYTES:
            yield ParquetSplit(path=path, file_idx=file_idx, row_start=split_start_row, row_end=cumulative_rows)
            split_start_row = cumulative_rows
            split_bytes = 0

        split_bytes += rg_bytes
        cumulative_rows += rg_meta.num_rows

    yield ParquetSplit(path=path, file_idx=file_idx, row_start=split_start_row, row_end=cumulative_rows)


def _iter_has_more_than_one(records: Iterator[T]) -> tuple[bool, T, Iterator[T]]:
    """Peek into an iterator to check if it has more than one item, without consuming items."""
    records = iter(records)  # Ensure we have an iterator
    # NOTE: we assume the iterator is non-empty, which is the case for use use-cases
    first = next(records)

    try:
        second = next(records)
        has_more_than_one = True
        rest = itertools.chain([second], records)
    except StopIteration:
        has_more_than_one = False
        rest = iter([])

    return has_more_than_one, first, itertools.chain([first], rest)


def dedup_exact_paragraph(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> dict:
    if filetypes is None:
        filetypes = DEFAULT_FILETYPES

    input_files = _collect_input_files(input_paths=input_paths, filetypes=filetypes)
    idx_to_path = dict(list(enumerate(input_files)))
    path_to_idx = {v: k for k, v in idx_to_path.items()}

    _init_wandb(mode=DedupMode.EXACT_PARAGRAPH, input_paths=input_paths)

    def compute_paragraph_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            dupekit.Transformation.SplitParagraphs(text_col=text_field, id_col="id"),
            dupekit.Transformation.Hash(
                input_col="paragraph_text", output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128
            ),
            dupekit.Transformation.SelectColumns(columns=["doc_id", "paragraph_span", "hash"]),
        ]
        return dupekit.transform(batch, pipeline)

    ctx_kwargs: dict = {
        "name": "exact-para-dedup",
        "max_workers": max_parallelism,
        "resources": worker_resources or ResourceConfig(cpu=2, ram="32g", disk="5g"),
        "map_workers_per_actor": 2,
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    ctx = ZephyrContext(**ctx_kwargs)

    def aggregate_and_write_to_corresponding_files(file_idx: int, records: Iterator[dict]) -> dict:
        # NOTE: all records belong to the specific file and are sorted by doc_id

        input_path = idx_to_path[file_idx]
        output_file = rebase_file_path(
            _find_base_path(input_paths, [input_path]),
            input_path,
            f"{output_path}/data/",
            old_extension=_get_extension(input_path),
            new_extension=".parquet",
        )

        total = 0
        dups = 0

        def counting_iter():
            nonlocal total, dups
            for record in records:
                is_dup: bool = record["is_dup"]
                total += 1
                counters.increment("dedup/exact/paragraph/total")
                if is_dup:
                    dups += 1
                    counters.increment("dedup/exact/paragraph/dups")
                else:
                    counters.increment("dedup/exact/paragraph/unique")
                yield record

        def group_by_doc_id(records: Iterator[dict]) -> Iterator[dict]:
            doc_level_record: dict[str, Any] | None = None
            for record in records:
                doc_id = record["id"]
                if doc_level_record is None:
                    doc_level_record = {
                        "id": doc_id,
                        "attributes": {"dup_spans": [record["span"]] if record["span"] else []},
                    }
                elif doc_level_record["id"] != doc_id:
                    if doc_level_record["attributes"]["dup_spans"]:
                        yield doc_level_record
                    doc_level_record = {
                        "id": doc_id,
                        "attributes": {"dup_spans": [record["span"]] if record["span"] else []},
                    }
                else:
                    assert doc_level_record["id"] == doc_id
                    if record["span"]:
                        doc_level_record["attributes"]["dup_spans"].append(record["span"])
            if doc_level_record and doc_level_record["attributes"]["dup_spans"]:
                yield doc_level_record

        result = write_parquet_file(group_by_doc_id(counting_iter()), output_file)
        return {**result, "total": total, "dups": dups, "unique": total - dups}

    def annotate_dups(key_hash: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        has_dups, head_record, records = _iter_has_more_than_one(records)

        # NOTE: we **arbitrarily** select the 1st record as the canonical record
        cano_id = head_record["id"]

        for item in records:
            is_dup = has_dups and item["id"] != cano_id
            yield {
                "id": item["id"],
                "is_dup": is_dup,
                "span": item["paragraph_span"]["span"] if is_dup else [],
                "file_idx": item["file_idx"],
            }

    def _expand_to_splits(path: str) -> Iterator[ParquetSplit]:
        yield from _compute_parquet_splits(path, path_to_idx[path])

    def _flat_map_paragraph_hashes(split: ParquetSplit) -> Iterator[dict]:
        with open_file(split.path, "rb") as f:
            pf = pq.ParquetFile(f)
            for table in iter_parquet_row_groups(pf, row_start=split.row_start, row_end=split.row_end):
                for batch in table.to_batches():
                    hashes = compute_paragraph_hashes(batch).to_pylist()
                    counters.increment("hash/paragraphs", len(hashes))
                    for hash_record in hashes:
                        yield {"file_idx": split.file_idx, "id": hash_record.pop("doc_id"), **hash_record}

    shard_results = ctx.execute(
        Dataset.from_list(input_files)
        .flat_map(_expand_to_splits)
        .reshard(num_shards=max_parallelism)
        .flat_map(_flat_map_paragraph_hashes)
        .group_by(
            lambda record: record["hash"],
            # NOTE: selecting the canonical record is deterministic via this sort
            sort_by=lambda record: record["id"],
            reducer=annotate_dups,
        )
        .group_by(
            lambda r: r["file_idx"],
            sort_by=lambda r: r["id"],
            reducer=aggregate_and_write_to_corresponding_files,
        ),
        verbose=True,
    ).results

    return finalize_dedup(shard_results, DedupMode.EXACT_PARAGRAPH, method="exact", level="paragraph")


def dedup_exact_document(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> dict:
    """Exact document deduplication: identify duplicate documents based on full text hash"""
    if filetypes is None:
        filetypes = DEFAULT_FILETYPES

    input_files = _collect_input_files(input_paths=input_paths, filetypes=filetypes)
    idx_to_path = dict(list(enumerate(input_files)))
    path_to_idx = {v: k for k, v in idx_to_path.items()}

    _init_wandb(mode=DedupMode.EXACT_DOCUMENT, input_paths=input_paths)

    def compute_document_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            dupekit.Transformation.Hash(input_col=text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            dupekit.Transformation.SelectColumns(columns=["hash", "id"]),
        ]
        return dupekit.transform(batch, pipeline)

    ctx_kwargs: dict = {
        "name": "exact-doc-dedup",
        "max_workers": max_parallelism,
        "resources": worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g"),
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    ctx = ZephyrContext(**ctx_kwargs)

    aggregate_and_write = make_document_dedup_aggregator(
        idx_to_path=idx_to_path,
        input_paths=input_paths,
        output_path=output_path,
        counter_prefix="dedup/exact/document",
    )

    def annotate_dups(key_hash: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        has_dups, head_record, records = _iter_has_more_than_one(records)

        # NOTE: we **arbitrarily** select the 1st record as the canonical record
        cano_id = head_record["id"]

        for item in records:
            is_dup = has_dups and item["id"] != cano_id
            yield {
                "id": item["id"],
                "is_dup": is_dup,
                "file_idx": item["file_idx"],
            }

    def _flat_map_document_hashes(path: str) -> Iterator[dict]:
        for batch in _load_batches(path):
            hashes = compute_document_hashes(batch).to_pylist()
            counters.increment("hash/documents", len(hashes))
            for hash_record in hashes:
                yield {"file_idx": path_to_idx[path], **hash_record}

    shard_results = ctx.execute(
        Dataset.from_list(input_files)
        .flat_map(_flat_map_document_hashes)
        .group_by(
            lambda record: record["hash"],
            # NOTE: selecting the canonical record is deterministic via this sort
            sort_by=lambda record: record["id"],
            reducer=annotate_dups,
        )
        .group_by(lambda r: r["file_idx"], sort_by=lambda r: r["id"], reducer=aggregate_and_write),
        verbose=True,
    ).results

    return finalize_dedup(shard_results, DedupMode.EXACT_DOCUMENT, method="exact", level="document")
