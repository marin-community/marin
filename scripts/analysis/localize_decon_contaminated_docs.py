# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Localize exact n-gram overlaps for Datakit-contaminated documents.

Datakit marks a source document contaminated when at least one source paragraph
has enough 13-word n-grams in the eval bloom filter. This script starts from
the compact contaminated-doc manifest and performs an exact targeted comparison
against the staged eval records so we can identify the source paragraph,
matched eval record, best eval paragraph, exact Jaccard, and containment scores.

Default paths target the Math500-only Nemotron-CC math decon run:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name localize-nemotron-math500-contam \
        -e PYTHONUNBUFFERED 1 \
        -- python scripts/analysis/localize_decon_contaminated_docs.py \
            --resume --max-workers 231 --worker-cpu 1 --worker-ram 4g
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.datakit.decon import _extract_ngrams
from marin.utils import fsspec_exists, fsspec_glob
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = (
    "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus/"
    "math500__test/contaminated_docs/contaminated_docs.parquet"
)
DEFAULT_EVAL_DATA = "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/math500/test/data.jsonl.gz"
DEFAULT_OUTPUT_ROOT = (
    "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus/"
    "math500__test/localized_overlaps"
)

LOCALIZED_PARQUET = "localized_overlaps.parquet"
LOCALIZED_CSV = "localized_overlaps.csv"
MISSES_PARQUET = "localization_misses.parquet"
SUMMARY_JSON = "summary.json"
LOCALIZED_SHARDS_DIR = "localized_shards"
MISSES_SHARDS_DIR = "miss_shards"
STATS_DIR = "localization_stats"

LOCALIZED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_parquet", pa.string()),
        ("attr_parquet", pa.string()),
        ("partition_id", pa.int64()),
        ("row_index_in_partition", pa.int64()),
        ("datakit_max_overlap", pa.float64()),
        ("eval_id", pa.string()),
        ("source_paragraph_index", pa.int64()),
        ("source_char_start", pa.int64()),
        ("source_char_end", pa.int64()),
        ("source_token_count", pa.int64()),
        ("source_ngram_count", pa.int64()),
        ("source_unique_ngram_count", pa.int64()),
        ("eval_record_unique_ngram_count", pa.int64()),
        ("record_intersection_count", pa.int64()),
        ("record_source_containment", pa.float64()),
        ("record_source_unique_containment", pa.float64()),
        ("record_eval_containment", pa.float64()),
        ("record_jaccard", pa.float64()),
        ("best_eval_paragraph_index", pa.int64()),
        ("best_eval_char_start", pa.int64()),
        ("best_eval_char_end", pa.int64()),
        ("best_eval_token_count", pa.int64()),
        ("best_eval_ngram_count", pa.int64()),
        ("best_eval_intersection_count", pa.int64()),
        ("best_eval_source_containment", pa.float64()),
        ("best_eval_jaccard", pa.float64()),
        ("source_snippet", pa.string()),
        ("eval_snippet", pa.string()),
        ("shared_ngrams", pa.list_(pa.string())),
    ]
)

MISSES_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_parquet", pa.string()),
        ("attr_parquet", pa.string()),
        ("partition_id", pa.int64()),
        ("row_index_in_partition", pa.int64()),
        ("datakit_max_overlap", pa.float64()),
        ("matched_eval_ids", pa.list_(pa.string())),
        ("reason", pa.string()),
        ("best_record_source_containment", pa.float64()),
        ("best_record_jaccard", pa.float64()),
    ]
)

STATS_SCHEMA = pa.schema(
    [
        ("source_parquet", pa.string()),
        ("localized_path", pa.string()),
        ("misses_path", pa.string()),
        ("manifest_docs", pa.int64()),
        ("processed_docs", pa.int64()),
        ("localized_rows", pa.int64()),
        ("miss_rows", pa.int64()),
        ("docs_with_localized_overlap", pa.int64()),
        ("max_record_source_containment", pa.float64()),
        ("max_record_jaccard", pa.float64()),
    ]
)


@dataclass(frozen=True)
class ParagraphFeatures:
    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    ngrams: tuple[str, ...]
    unique_ngrams: frozenset[str]


@dataclass(frozen=True)
class EvalFeatures:
    eval_id: str
    text: str
    paragraphs: tuple[ParagraphFeatures, ...]
    unique_ngrams: frozenset[str]


@dataclass(frozen=True)
class PairMetrics:
    intersection: int
    source_containment: float
    source_unique_containment: float
    eval_containment: float
    jaccard: float


def write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_parquet_table(path: str, table: pa.Table) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "wb") as handle:
        pq.write_table(table, handle)


def write_csv(path: str, table: pa.Table) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    rows = table.to_pylist()
    columns = [field.name for field in table.schema]
    with fs.open(paths[0], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in out.items():
                if isinstance(value, list):
                    out[key] = "|".join(str(item) for item in value)
            writer.writerow(out)


def fail_if_existing(paths: list[str], *, resume: bool, force: bool) -> None:
    existing = [path for path in paths if fsspec_exists(path)]
    if not existing:
        return
    if force:
        for path in existing:
            fs, _, resolved = fsspec.get_fs_token_paths(path)
            logger.warning("removing existing output because --force was set: %s", path)
            fs.rm(resolved[0], recursive=True)
        return
    if resume:
        return
    raise RuntimeError(f"outputs already exist; pass --resume or --force: {existing}")


def text_paragraphs(text: str, *, ngram_length: int, stride: int) -> tuple[ParagraphFeatures, ...]:
    paragraphs = []
    char_start = 0
    paragraph_index = 0
    for paragraph in text.split("\n"):
        char_end = char_start + len(paragraph)
        if paragraph:
            ngrams = tuple(_extract_ngrams(paragraph, ngram_length, stride))
            paragraphs.append(
                ParagraphFeatures(
                    index=paragraph_index,
                    text=paragraph,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(paragraph.split()),
                    ngrams=ngrams,
                    unique_ngrams=frozenset(ngrams),
                )
            )
            paragraph_index += 1
        char_start = char_end + 1
    return tuple(paragraphs)


def eval_features(eval_data: str, *, ngram_length: int, stride: int) -> dict[str, EvalFeatures]:
    records = {}
    for index, record in enumerate(load_file(eval_data)):
        text = str(record.get("text", "") or "")
        if not text:
            continue
        eval_id = str(record.get("id") or f"{eval_data}::{index}")
        paragraphs = text_paragraphs(text, ngram_length=ngram_length, stride=stride)
        unique_ngrams: set[str] = set()
        for paragraph in paragraphs:
            unique_ngrams.update(paragraph.unique_ngrams)
        records[eval_id] = EvalFeatures(
            eval_id=eval_id,
            text=text,
            paragraphs=paragraphs,
            unique_ngrams=frozenset(unique_ngrams),
        )
    if not records:
        raise FileNotFoundError(f"No eval records with text found in {eval_data}")
    return records


def pair_metrics(source: ParagraphFeatures, eval_ngrams: frozenset[str]) -> PairMetrics:
    source_unique = source.unique_ngrams
    if not source.ngrams or not source_unique or not eval_ngrams:
        return PairMetrics(0, 0.0, 0.0, 0.0, 0.0)
    intersection = source_unique & eval_ngrams
    counted_hits = sum(1 for ngram in source.ngrams if ngram in eval_ngrams)
    union_count = len(source_unique | eval_ngrams)
    return PairMetrics(
        intersection=len(intersection),
        source_containment=counted_hits / len(source.ngrams),
        source_unique_containment=len(intersection) / len(source_unique),
        eval_containment=len(intersection) / len(eval_ngrams),
        jaccard=len(intersection) / union_count if union_count else 0.0,
    )


def best_eval_paragraph(
    source: ParagraphFeatures, eval_record: EvalFeatures
) -> tuple[ParagraphFeatures | None, PairMetrics]:
    best_paragraph = None
    best_metrics = PairMetrics(0, 0.0, 0.0, 0.0, 0.0)
    for paragraph in eval_record.paragraphs:
        metrics = pair_metrics(source, paragraph.unique_ngrams)
        if (metrics.source_containment, metrics.jaccard, metrics.intersection) > (
            best_metrics.source_containment,
            best_metrics.jaccard,
            best_metrics.intersection,
        ):
            best_paragraph = paragraph
            best_metrics = metrics
    return best_paragraph, best_metrics


def snippet(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def shared_ngram_sample(source: ParagraphFeatures, eval_ngrams: frozenset[str], max_ngrams: int) -> list[str]:
    shared = []
    seen = set()
    for ngram in source.ngrams:
        if ngram not in eval_ngrams or ngram in seen:
            continue
        shared.append(ngram)
        seen.add(ngram)
        if len(shared) >= max_ngrams:
            break
    return shared


def read_manifest(manifest_path: str) -> list[dict[str, Any]]:
    with fsspec.open(manifest_path, "rb") as handle:
        table = pq.read_table(handle)
    rows = table.to_pylist()
    if not rows:
        raise FileNotFoundError(f"No contaminated rows found in {manifest_path}")
    return rows


def group_manifest_by_source(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_parquet"])].append(row)
    for source_rows in grouped.values():
        source_rows.sort(key=lambda row: int(row["row_index_in_partition"]))
    return dict(sorted(grouped.items()))


def task_output_path(output_root: str, subdir: str, source_path: str) -> str:
    return f"{output_root.rstrip('/')}/{subdir}/{os.path.basename(source_path)}"


def read_selected_source_rows(
    source_path: str, manifest_rows: list[dict[str, Any]]
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    rows_by_index = {int(row["row_index_in_partition"]): row for row in manifest_rows}
    wanted = sorted(rows_by_index)
    next_wanted = 0
    global_offset = 0
    with fsspec.open(source_path, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        for row_group in range(parquet_file.num_row_groups):
            row_group_rows = parquet_file.metadata.row_group(row_group).num_rows
            row_group_start = global_offset
            row_group_end = global_offset + row_group_rows
            local_indices = []
            selected_global = []
            while next_wanted < len(wanted) and wanted[next_wanted] < row_group_start:
                raise ValueError(f"row index {wanted[next_wanted]} was not found while reading {source_path}")
            scan = next_wanted
            while scan < len(wanted) and wanted[scan] < row_group_end:
                selected_global.append(wanted[scan])
                local_indices.append(wanted[scan] - row_group_start)
                scan += 1
            if local_indices:
                table = parquet_file.read_row_group(row_group, columns=["id", "text"])
                selected = table.take(pa.array(local_indices, type=pa.int64())).to_pylist()
                for global_index, source_row in zip(selected_global, selected, strict=True):
                    yield rows_by_index[global_index], source_row
                next_wanted = scan
            global_offset = row_group_end
    if next_wanted != len(wanted):
        raise ValueError(f"only found {next_wanted}/{len(wanted)} selected rows in {source_path}")


def localized_rows_for_doc(
    *,
    manifest_row: dict[str, Any],
    source_row: dict[str, Any],
    eval_records: dict[str, EvalFeatures],
    ngram_length: int,
    stride: int,
    min_source_containment: float,
    min_jaccard: float,
    max_snippet_chars: int,
    max_shared_ngrams: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    doc_id = str(manifest_row["id"])
    source_text = str(source_row.get("text", "") or "")
    source_paragraphs = text_paragraphs(source_text, ngram_length=ngram_length, stride=stride)
    matched_eval_ids = [str(eval_id) for eval_id in manifest_row.get("matched_eval_ids") or []]
    localized = []
    best_doc_source_containment = 0.0
    best_doc_jaccard = 0.0
    missing_eval_ids = []

    for eval_id in matched_eval_ids:
        eval_record = eval_records.get(eval_id)
        if eval_record is None:
            missing_eval_ids.append(eval_id)
            continue
        for source_paragraph in source_paragraphs:
            metrics = pair_metrics(source_paragraph, eval_record.unique_ngrams)
            best_doc_source_containment = max(best_doc_source_containment, metrics.source_containment)
            best_doc_jaccard = max(best_doc_jaccard, metrics.jaccard)
            passes_source_containment = (
                min_source_containment > 0 and metrics.source_containment >= min_source_containment
            )
            passes_jaccard = min_jaccard > 0 and metrics.jaccard >= min_jaccard
            if not (passes_source_containment or passes_jaccard):
                continue
            best_paragraph, best_paragraph_metrics = best_eval_paragraph(source_paragraph, eval_record)
            localized.append(
                {
                    "doc_id": doc_id,
                    "source_parquet": manifest_row["source_parquet"],
                    "attr_parquet": manifest_row["attr_parquet"],
                    "partition_id": int(manifest_row["partition_id"]),
                    "row_index_in_partition": int(manifest_row["row_index_in_partition"]),
                    "datakit_max_overlap": float(manifest_row["max_overlap"]),
                    "eval_id": eval_id,
                    "source_paragraph_index": source_paragraph.index,
                    "source_char_start": source_paragraph.char_start,
                    "source_char_end": source_paragraph.char_end,
                    "source_token_count": source_paragraph.token_count,
                    "source_ngram_count": len(source_paragraph.ngrams),
                    "source_unique_ngram_count": len(source_paragraph.unique_ngrams),
                    "eval_record_unique_ngram_count": len(eval_record.unique_ngrams),
                    "record_intersection_count": metrics.intersection,
                    "record_source_containment": metrics.source_containment,
                    "record_source_unique_containment": metrics.source_unique_containment,
                    "record_eval_containment": metrics.eval_containment,
                    "record_jaccard": metrics.jaccard,
                    "best_eval_paragraph_index": best_paragraph.index if best_paragraph is not None else -1,
                    "best_eval_char_start": best_paragraph.char_start if best_paragraph is not None else -1,
                    "best_eval_char_end": best_paragraph.char_end if best_paragraph is not None else -1,
                    "best_eval_token_count": best_paragraph.token_count if best_paragraph is not None else 0,
                    "best_eval_ngram_count": len(best_paragraph.ngrams) if best_paragraph is not None else 0,
                    "best_eval_intersection_count": best_paragraph_metrics.intersection,
                    "best_eval_source_containment": best_paragraph_metrics.source_containment,
                    "best_eval_jaccard": best_paragraph_metrics.jaccard,
                    "source_snippet": snippet(source_paragraph.text, max_snippet_chars),
                    "eval_snippet": (
                        snippet(best_paragraph.text, max_snippet_chars) if best_paragraph is not None else ""
                    ),
                    "shared_ngrams": shared_ngram_sample(
                        source_paragraph,
                        eval_record.unique_ngrams,
                        max_shared_ngrams,
                    ),
                }
            )

    if localized:
        return localized, None

    reason = "no_exact_overlap_at_threshold"
    if missing_eval_ids and len(missing_eval_ids) == len(matched_eval_ids):
        reason = "matched_eval_ids_not_found"
    return [], {
        "doc_id": doc_id,
        "source_parquet": manifest_row["source_parquet"],
        "attr_parquet": manifest_row["attr_parquet"],
        "partition_id": int(manifest_row["partition_id"]),
        "row_index_in_partition": int(manifest_row["row_index_in_partition"]),
        "datakit_max_overlap": float(manifest_row["max_overlap"]),
        "matched_eval_ids": matched_eval_ids,
        "reason": reason,
        "best_record_source_containment": best_doc_source_containment,
        "best_record_jaccard": best_doc_jaccard,
    }


def task_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source_parquet": str(row["source_parquet"]),
        "attr_parquet": str(row["attr_parquet"]),
        "partition_id": int(row["partition_id"]),
        "row_index_in_partition": int(row["row_index_in_partition"]),
        "max_overlap": float(row["max_overlap"]),
        "matched_eval_ids": [str(eval_id) for eval_id in row.get("matched_eval_ids") or []],
    }


def localize_source_task(task: dict[str, Any]) -> dict[str, Any]:
    eval_records = eval_features(
        str(task["eval_data"]),
        ngram_length=int(task["ngram_length"]),
        stride=int(task["stride"]),
    )
    localized: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    processed_docs = 0
    source_path = str(task["source_path"])
    manifest_rows = list(task["manifest_rows"])
    logger.info("localizing %d contaminated docs from %s", len(manifest_rows), source_path)

    for manifest_row, source_row in read_selected_source_rows(source_path, manifest_rows):
        rows, miss = localized_rows_for_doc(
            manifest_row=manifest_row,
            source_row=source_row,
            eval_records=eval_records,
            ngram_length=int(task["ngram_length"]),
            stride=int(task["stride"]),
            min_source_containment=float(task["min_source_containment"]),
            min_jaccard=float(task["min_jaccard"]),
            max_snippet_chars=int(task["max_snippet_chars"]),
            max_shared_ngrams=int(task["max_shared_ngrams"]),
        )
        localized.extend(rows)
        if miss is not None:
            misses.append(miss)
        processed_docs += 1

    localized.sort(
        key=lambda row: (
            -row["record_source_containment"],
            -row["record_jaccard"],
            row["doc_id"],
            row["source_paragraph_index"],
            row["eval_id"],
        )
    )
    misses.sort(key=lambda row: (row["partition_id"], row["row_index_in_partition"], row["doc_id"]))
    localized_path = str(task["localized_path"])
    misses_path = str(task["misses_path"])
    write_parquet_table(localized_path, pa.Table.from_pylist(localized, schema=LOCALIZED_SCHEMA))
    write_parquet_table(misses_path, pa.Table.from_pylist(misses, schema=MISSES_SCHEMA))

    docs_with_localized_overlap = len({row["doc_id"] for row in localized})
    return {
        "source_parquet": source_path,
        "localized_path": localized_path,
        "misses_path": misses_path,
        "manifest_docs": len(manifest_rows),
        "processed_docs": processed_docs,
        "localized_rows": len(localized),
        "miss_rows": len(misses),
        "docs_with_localized_overlap": docs_with_localized_overlap,
        "max_record_source_containment": max(
            (row["record_source_containment"] for row in localized),
            default=0.0,
        ),
        "max_record_jaccard": max((row["record_jaccard"] for row in localized), default=0.0),
    }


def table_from_parquet_files(paths: list[str], schema: pa.Schema) -> pa.Table:
    tables = []
    for path in paths:
        with fsspec.open(path, "rb") as handle:
            table = pq.read_table(handle)
        if table.num_rows:
            tables.append(table)
    if not tables:
        return pa.Table.from_pylist([], schema=schema)
    return pa.concat_tables(tables, promote_options="permissive").cast(schema)


def rows_from_parquet_files(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with fsspec.open(path, "rb") as handle:
            rows.extend(pq.read_table(handle).to_pylist())
    return rows


def prepare_source_tasks(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    eval_data: str,
    output_root: str,
    ngram_length: int,
    stride: int,
    min_source_containment: float,
    min_jaccard: float,
    max_snippet_chars: int,
    max_shared_ngrams: int,
) -> list[dict[str, Any]]:
    tasks = []
    for source_path, source_manifest_rows in grouped.items():
        tasks.append(
            {
                "source_path": source_path,
                "manifest_rows": [task_manifest_row(row) for row in source_manifest_rows],
                "eval_data": eval_data,
                "localized_path": task_output_path(output_root, LOCALIZED_SHARDS_DIR, source_path),
                "misses_path": task_output_path(output_root, MISSES_SHARDS_DIR, source_path),
                "ngram_length": ngram_length,
                "stride": stride,
                "min_source_containment": min_source_containment,
                "min_jaccard": min_jaccard,
                "max_snippet_chars": max_snippet_chars,
                "max_shared_ngrams": max_shared_ngrams,
            }
        )
    return tasks


def localize_overlaps(
    *,
    manifest_path: str,
    eval_data: str,
    output_root: str,
    ngram_length: int,
    stride: int,
    min_source_containment: float,
    min_jaccard: float,
    max_snippet_chars: int,
    max_shared_ngrams: int,
    max_workers: int,
    worker_cpu: int,
    worker_ram: str,
    worker_disk: str,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    output_parquet = f"{output_root.rstrip('/')}/{LOCALIZED_PARQUET}"
    output_csv = f"{output_root.rstrip('/')}/{LOCALIZED_CSV}"
    misses_parquet = f"{output_root.rstrip('/')}/{MISSES_PARQUET}"
    summary_path = f"{output_root.rstrip('/')}/{SUMMARY_JSON}"
    stats_dir = f"{output_root.rstrip('/')}/{STATS_DIR}"
    fail_if_existing([output_parquet, output_csv, misses_parquet, summary_path], resume=resume, force=force)
    if resume and fsspec_exists(summary_path):
        logger.info("using existing localization export: %s", summary_path)
        with fsspec.open(summary_path) as handle:
            return json.load(handle)

    eval_record_count = len(eval_features(eval_data, ngram_length=ngram_length, stride=stride))
    manifest_rows = read_manifest(manifest_path)
    grouped = group_manifest_by_source(manifest_rows)
    tasks = prepare_source_tasks(
        grouped=grouped,
        eval_data=eval_data,
        output_root=output_root,
        ngram_length=ngram_length,
        stride=stride,
        min_source_containment=min_source_containment,
        min_jaccard=min_jaccard,
        max_snippet_chars=max_snippet_chars,
        max_shared_ngrams=max_shared_ngrams,
    )
    logger.info("localizing %d source shards with Zephyr max_workers=%d", len(tasks), max_workers)
    ctx = ZephyrContext(
        name="localize-decon-contaminated-docs",
        max_workers=max_workers,
        resources=ResourceConfig(
            cpu=worker_cpu,
            ram=worker_ram,
            disk=worker_disk,
            preemptible=True,
            regions=("us-east5",),
        ),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", disk="10g", preemptible=False, regions=("us-east5",)),
    )
    outcome = ctx.execute(
        Dataset.from_list(tasks)
        .map(localize_source_task)
        .write_parquet(
            f"{stats_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=STATS_SCHEMA,
            skip_existing=resume,
        )
    )
    logger.info("localization counters: %s", dict(outcome.counters))

    stats_files = sorted(fsspec_glob(f"{stats_dir.rstrip('/')}/*.parquet"))
    if not stats_files:
        raise FileNotFoundError(f"No localization stats files found under {stats_dir}")
    stats_rows = rows_from_parquet_files(stats_files)
    localized_table = table_from_parquet_files(
        sorted(str(row["localized_path"]) for row in stats_rows),
        LOCALIZED_SCHEMA,
    )
    misses_table = table_from_parquet_files(
        sorted(str(row["misses_path"]) for row in stats_rows),
        MISSES_SCHEMA,
    )
    localized = localized_table.to_pylist()
    localized.sort(
        key=lambda row: (
            -row["record_source_containment"],
            -row["record_jaccard"],
            row["doc_id"],
            row["source_paragraph_index"],
            row["eval_id"],
        )
    )
    misses = misses_table.to_pylist()
    misses.sort(key=lambda row: (row["partition_id"], row["row_index_in_partition"], row["doc_id"]))
    localized_table = pa.Table.from_pylist(localized, schema=LOCALIZED_SCHEMA)
    misses_table = pa.Table.from_pylist(misses, schema=MISSES_SCHEMA)
    write_parquet_table(output_parquet, localized_table)
    write_csv(output_csv, localized_table)
    write_parquet_table(misses_parquet, misses_table)

    docs_with_localized_overlap = len({row["doc_id"] for row in localized})
    summary = {
        "manifest_path": manifest_path,
        "eval_data": eval_data,
        "output_parquet": output_parquet,
        "output_csv": output_csv,
        "misses_parquet": misses_parquet,
        "localized_shards_dir": f"{output_root.rstrip('/')}/{LOCALIZED_SHARDS_DIR}",
        "miss_shards_dir": f"{output_root.rstrip('/')}/{MISSES_SHARDS_DIR}",
        "stats_dir": stats_dir,
        "ngram_length": ngram_length,
        "stride": stride,
        "min_source_containment": min_source_containment,
        "min_jaccard": min_jaccard,
        "eval_records": eval_record_count,
        "source_shards": len(grouped),
        "manifest_docs": len(manifest_rows),
        "processed_docs": sum(int(row["processed_docs"]) for row in stats_rows),
        "localized_rows": len(localized),
        "docs_with_localized_overlap": docs_with_localized_overlap,
        "docs_without_localized_overlap": len(misses),
        "max_record_source_containment": max((row["record_source_containment"] for row in localized), default=0.0),
        "max_record_jaccard": max((row["record_jaccard"] for row in localized), default=0.0),
        "max_workers": max_workers,
        "worker_cpu": worker_cpu,
        "worker_ram": worker_ram,
        "worker_disk": worker_disk,
        "counters": dict(outcome.counters),
    }
    write_json(summary_path, summary)
    logger.info("wrote exact localization export: %s", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ngram-length", type=int, default=13)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--min-source-containment", type=float, default=0.5)
    parser.add_argument("--min-jaccard", type=float, default=0.0)
    parser.add_argument("--max-snippet-chars", type=int, default=800)
    parser.add_argument("--max-shared-ngrams", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--worker-cpu", type=int, default=1)
    parser.add_argument("--worker-ram", default="4g")
    parser.add_argument("--worker-disk", default="5g")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    localize_overlaps(
        manifest_path=args.manifest,
        eval_data=args.eval_data,
        output_root=args.output_root,
        ngram_length=args.ngram_length,
        stride=args.stride,
        min_source_containment=args.min_source_containment,
        min_jaccard=args.min_jaccard,
        max_snippet_chars=args.max_snippet_chars,
        max_shared_ngrams=args.max_shared_ngrams,
        max_workers=args.max_workers,
        worker_cpu=args.worker_cpu,
        worker_ram=args.worker_ram,
        worker_disk=args.worker_disk,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
