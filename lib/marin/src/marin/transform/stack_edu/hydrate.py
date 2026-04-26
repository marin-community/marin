# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate Stack-Edu metadata shards into text documents.

The HuggingFaceTB/stack-edu export stores Software Heritage blob ids plus
metadata in parquet files. This transform fetches the underlying blob contents
from the public Software Heritage S3 bucket and writes Dolma-style JSONL shards
that can be consumed by the normal tokenizer.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import draccus
import pyarrow.parquet as pq
import s3fs
from fray import ResourceConfig
from rigging.filesystem import open_url, url_to_fs
from marin.utils import fsspec_glob
from zephyr import Dataset, InputFileSpec, ZephyrContext, load_jsonl, load_parquet, write_jsonl_file

logger = logging.getLogger(__name__)

SOFTWARE_HERITAGE_BUCKET = "softwareheritage"
SOFTWARE_HERITAGE_CONTENT_PREFIX = "content"
STACK_EDU_REQUIRED_COLUMNS = [
    "blob_id",
    "repo_name",
    "path",
    "src_encoding",
    "detected_licenses",
    "license_type",
    "score",
    "int_score",
    "length_bytes",
]


@dataclass(frozen=True)
class StackEduHydrationConfig:
    """Configuration for Stack-Edu blob hydration."""

    input_path: str
    output_path: str
    language: str
    max_rows_per_task: int = 20_000
    max_workers: int = 64
    worker_resources: ResourceConfig | None = None
    max_retries_per_blob: int = 8
    pipeline_version: str = "v1"


@dataclass(frozen=True)
class HydrationTask:
    """Hydrate a row range from a single parquet shard into one JSONL shard."""

    language: str
    input_file: str
    row_start: int
    row_end: int
    output_path: str
    max_retries_per_blob: int


HYDRATION_ISSUE_SAMPLE_LIMIT = 5


def _task_output_path(output_path: str, input_file: str, row_start: int, row_end: int) -> str:
    stem = os.path.basename(input_file).removesuffix(".parquet")
    return os.path.join(output_path, "train", f"{stem}-rows{row_start:09d}-{row_end:09d}.jsonl.zst")


def _group_row_ranges(
    input_file: str,
    max_rows_per_task: int,
    output_path: str,
    language: str,
    max_retries_per_blob: int,
) -> list[HydrationTask]:
    with open_url(input_file, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)

        tasks: list[HydrationTask] = []
        row_start = 0
        pending_rows = 0
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_idx)
            pending_rows += row_group.num_rows

            should_flush = pending_rows >= max_rows_per_task
            is_last_group = row_group_idx == parquet_file.metadata.num_row_groups - 1
            if not should_flush and not is_last_group:
                continue

            row_end = row_start + pending_rows
            tasks.append(
                HydrationTask(
                    language=language,
                    input_file=input_file,
                    row_start=row_start,
                    row_end=row_end,
                    output_path=_task_output_path(output_path, input_file, row_start, row_end),
                    max_retries_per_blob=max_retries_per_blob,
                )
            )
            row_start = row_end
            pending_rows = 0

        return tasks


def _decode_blob(raw_bytes: bytes, src_encoding: str | None) -> tuple[str, bool]:
    candidate_encodings: list[str] = []
    if src_encoding:
        candidate_encodings.append(src_encoding)
    candidate_encodings.append("utf-8")

    tried: set[str] = set()
    for encoding in candidate_encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw_bytes.decode(encoding), False
        except (LookupError, UnicodeDecodeError):
            continue

    fallback_encoding = src_encoding or "utf-8"
    try:
        return raw_bytes.decode(fallback_encoding, errors="ignore"), True
    except LookupError:
        return raw_bytes.decode("utf-8", errors="ignore"), True


def _blob_storage_path(blob_id: str) -> str:
    return f"{SOFTWARE_HERITAGE_BUCKET}/{SOFTWARE_HERITAGE_CONTENT_PREFIX}/{blob_id}"


def _fetch_blob_text(
    fs: s3fs.S3FileSystem, blob_id: str, src_encoding: str | None, max_retries: int
) -> tuple[str | None, str]:
    for attempt in range(max_retries):
        try:
            with fs.open(_blob_storage_path(blob_id), "rb") as handle:
                with gzip.GzipFile(fileobj=handle) as compressed:
                    raw_bytes = compressed.read()
            decoded_text, used_fallback = _decode_blob(raw_bytes, src_encoding)
            if not decoded_text.strip():
                return None, "empty_blob"
            return decoded_text, "decoded_fallback" if used_fallback else "ok"
        except FileNotFoundError:
            return None, "missing_blob"
        except (EOFError, gzip.BadGzipFile):
            return None, "corrupt_blob"
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.warning("Skipping blob %s after %d failed fetch attempts: %s", blob_id, max_retries, exc)
                return None, "fetch_error"

            wait_seconds = min(2**attempt, 30) + random.uniform(0.0, 1.0)
            logger.warning(
                "Retrying blob %s after attempt %d/%d failed: %s",
                blob_id,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(wait_seconds)

    raise AssertionError("unreachable")


def _record_id(language: str, row: dict) -> str:
    fingerprint = "\t".join([language, row["blob_id"], row["repo_name"], row["path"]])
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _hydrate_row(language: str, row: dict, text: str) -> dict:
    return {
        "id": _record_id(language, row),
        "text": text,
        "source": f"stack_edu/{language}",
        "metadata": {
            "blob_id": row["blob_id"],
            "repo_name": row["repo_name"],
            "path": row["path"],
            "src_encoding": row.get("src_encoding"),
            "detected_licenses": row.get("detected_licenses"),
            "license_type": row.get("license_type"),
            "score": row.get("score"),
            "int_score": row.get("int_score"),
            "length_bytes": row.get("length_bytes"),
            "language": language,
        },
    }


def process_hydration_task(task: HydrationTask) -> dict:
    """Hydrate one parquet row range into one compressed JSONL shard."""

    output_fs, _ = url_to_fs(task.output_path)
    if output_fs.exists(task.output_path):
        logger.info(
            "Skipping stack_edu/%s rows [%d, %d) because output exists: %s",
            task.language,
            task.row_start,
            task.row_end,
            task.output_path,
        )
        return {
            "language": task.language,
            "input_file": task.input_file,
            "row_start": task.row_start,
            "row_end": task.row_end,
            "path": task.output_path,
            "count": 0,
            "decoded_fallback": 0,
            "missing_blob": 0,
            "corrupt_blob": 0,
            "empty_blob": 0,
            "fetch_error": 0,
            "skipped": True,
        }

    swh_fs = s3fs.S3FileSystem(anon=True)
    counters = {
        "count": 0,
        "decoded_fallback": 0,
        "missing_blob": 0,
        "corrupt_blob": 0,
        "empty_blob": 0,
        "fetch_error": 0,
    }
    issue_samples: dict[str, list[str]] = defaultdict(list)
    input_spec = InputFileSpec(
        path=task.input_file,
        format="parquet",
        columns=STACK_EDU_REQUIRED_COLUMNS,
        row_start=task.row_start,
        row_end=task.row_end,
    )

    def hydrate_records():
        for row in load_parquet(input_spec):
            text, status = _fetch_blob_text(swh_fs, row["blob_id"], row.get("src_encoding"), task.max_retries_per_blob)
            if text is None:
                counters[status] += 1
                if len(issue_samples[status]) < HYDRATION_ISSUE_SAMPLE_LIMIT:
                    issue_samples[status].append(row["blob_id"])
                continue
            if status == "decoded_fallback":
                counters["decoded_fallback"] += 1
            counters["count"] += 1
            yield _hydrate_row(task.language, row, text)

    result = write_jsonl_file(hydrate_records(), task.output_path)
    return {
        "language": task.language,
        "input_file": task.input_file,
        "row_start": task.row_start,
        "row_end": task.row_end,
        "path": result["path"],
        **counters,
        "missing_blob_examples": issue_samples["missing_blob"],
        "corrupt_blob_examples": issue_samples["corrupt_blob"],
        "empty_blob_examples": issue_samples["empty_blob"],
        "fetch_error_examples": issue_samples["fetch_error"],
    }


def _build_hydration_tasks(cfg: StackEduHydrationConfig) -> list[HydrationTask]:
    input_files = sorted(fsspec_glob(os.path.join(cfg.input_path, "*.parquet")))
    if not input_files:
        raise ValueError(f"No Stack-Edu parquet files found in {cfg.input_path}")

    all_tasks: list[HydrationTask] = []
    for input_file in input_files:
        all_tasks.extend(
            _group_row_ranges(
                input_file,
                cfg.max_rows_per_task,
                cfg.output_path,
                cfg.language,
                cfg.max_retries_per_blob,
            )
        )

    logger.info("Prepared %d hydration tasks for stack_edu/%s", len(all_tasks), cfg.language)
    return all_tasks


def hydrate_stack_edu(cfg: StackEduHydrationConfig) -> str:
    """Hydrate Stack-Edu metadata parquet into text JSONL shards."""

    tasks = _build_hydration_tasks(cfg)
    metrics_path = os.path.join(cfg.output_path, ".metrics")
    pipeline = (
        Dataset.from_list(tasks)
        .map(process_hydration_task)
        .write_jsonl(
            f"{metrics_path}/hydrate-{{shard:05d}}.jsonl",
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(
        name=f"hydrate-stack-edu-{cfg.language}",
        max_workers=cfg.max_workers,
        resources=cfg.worker_resources,
    )
    metric_files = ctx.execute(pipeline)

    total_written = 0
    total_fallback = 0
    total_missing = 0
    total_corrupt = 0
    total_empty = 0
    total_fetch_error = 0
    total_tasks = 0
    skipped_tasks = 0
    issue_examples: dict[str, list[str]] = defaultdict(list)
    for metric_file in metric_files:
        for result in load_jsonl(metric_file):
            total_tasks += 1
            skipped_tasks += int(bool(result.get("skipped")))
            total_written += result["count"]
            total_fallback += result["decoded_fallback"]
            total_missing += result["missing_blob"]
            total_corrupt += result["corrupt_blob"]
            total_empty += result["empty_blob"]
            total_fetch_error += result["fetch_error"]
            for issue_type in ["missing_blob", "corrupt_blob", "empty_blob", "fetch_error"]:
                example_key = f"{issue_type}_examples"
                for blob_id in result.get(example_key, []):
                    if len(issue_examples[issue_type]) >= HYDRATION_ISSUE_SAMPLE_LIMIT:
                        break
                    if blob_id not in issue_examples[issue_type]:
                        issue_examples[issue_type].append(blob_id)

    logger.info(
        "Hydrated stack_edu/%s: %d tasks (%d skipped), %d rows written, %d fallback decodes, "
        "%d missing blobs, %d corrupt blobs, %d empty blobs, %d fetch errors",
        cfg.language,
        total_tasks,
        skipped_tasks,
        total_written,
        total_fallback,
        total_missing,
        total_corrupt,
        total_empty,
        total_fetch_error,
    )
    if total_missing:
        logger.info(
            "Sample missing Software Heritage blobs for stack_edu/%s: %s",
            cfg.language,
            ", ".join(issue_examples["missing_blob"]),
        )
    if total_corrupt:
        logger.info(
            "Sample corrupt Software Heritage blobs for stack_edu/%s: %s",
            cfg.language,
            ", ".join(issue_examples["corrupt_blob"]),
        )
    if total_empty:
        logger.info(
            "Sample empty Software Heritage blobs for stack_edu/%s: %s",
            cfg.language,
            ", ".join(issue_examples["empty_blob"]),
        )
    if total_fetch_error:
        logger.info(
            "Sample fetch-error Software Heritage blobs for stack_edu/%s: %s",
            cfg.language,
            ", ".join(issue_examples["fetch_error"]),
        )
    return cfg.output_path


if __name__ == "__main__":
    draccus.wrap(hydrate_stack_edu)()
