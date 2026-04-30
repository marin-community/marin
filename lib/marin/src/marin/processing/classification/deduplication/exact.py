# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import itertools
import logging
from collections.abc import Iterator
from typing import Any, TypeVar

import dupekit
import pyarrow as pa
from fray import ResourceConfig
from zephyr import ZephyrContext, counters, write_parquet_file
from zephyr.dataset import Dataset

from marin.processing.classification.deduplication.dedup_commons import (
    DEFAULT_FILETYPES,
    DedupMode,
    _collect_input_files,
    _find_base_path,
    _get_extension,
    _init_wandb,
    _load_batches,
    finalize_dedup,
    group_files,
    make_document_dedup_aggregator,
)
from marin.utils import rebase_file_path

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    idx_to_path = dict(list(enumerate(sorted(input_files))))
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
        "resources": worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g"),
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

    def _flat_map_paragraph_hashes(paths: list[str]) -> Iterator[dict]:
        for path in paths:
            for batch in _load_batches(path):
                hashes = compute_paragraph_hashes(batch).to_pylist()
                counters.increment("hash/paragraphs", len(hashes))
                for hash_record in hashes:
                    yield {"file_idx": path_to_idx[path], "id": hash_record.pop("doc_id"), **hash_record}

    file_groups = group_files(input_files, max_parallelism)
    shard_results = ctx.execute(
        Dataset.from_list(file_groups)
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
    idx_to_path = dict(list(enumerate(sorted(input_files))))
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

    def _flat_map_document_hashes(paths: list[str]) -> Iterator[dict]:
        for path in paths:
            for batch in _load_batches(path):
                hashes = compute_document_hashes(batch).to_pylist()
                counters.increment("hash/documents", len(hashes))
                for hash_record in hashes:
                    yield {"file_idx": path_to_idx[path], **hash_record}

    file_groups = group_files(input_files, max_parallelism)
    shard_results = ctx.execute(
        Dataset.from_list(file_groups)
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
