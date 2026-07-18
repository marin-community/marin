# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize TeraflopAI/SEC-EDGAR filings."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem
from rigging.filesystem import StoragePath, prefix_join
from rigging.timing import ExponentialBackoff, retry_with_backoff
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import atomic_rename, ensure_parent_dir

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"
_HF_REPOSITORY_ROOT = StoragePath("datasets") / HF_DATASET_ID
_HF_URL_ROOT = StoragePath("hf://datasets") / HF_DATASET_ID

FILING_TYPES = ("10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5")

# HF rate-limits aggressively; a small batch reader keeps memory bounded
# while one big SEC row group (~700 MB decompressed) is in flight.
_ROWS_PER_BATCH = 8

# Lift PyArrow's Thrift decoder caps so page headers carrying multi-MB
# string statistics (apache/arrow#46404) decode without "Couldn't
# deserialize thrift" errors. 1 GiB is well above any plausible single
# page header in SEC's content column (~tens of MB worst case) while
# still bounded.
_THRIFT_DECODE_LIMIT_BYTES = 1024 * 1024 * 1024

# Per-file retry policy for HF rate limits and transient network failures.
_MAX_RETRIES = 20
_RETRY_BACKOFF = ExponentialBackoff(initial=5, maximum=15 * 60, factor=2, jitter=0.1)


@dataclass(frozen=True)
class _DownloadTask:
    hf_path: str
    destination_path: str


@dataclass(frozen=True)
class _DownloadResult:
    hf_path: str
    destination_path: str
    rows: int


def _iter_parquet_batches(hf_path: str, *, revision: str = HF_REVISION) -> Iterator[pa.RecordBatch]:
    """Yield record batches from one revision-pinned SEC-EDGAR shard."""
    with StoragePath(hf_path).open("rb", revision=revision) as source:
        # SEC's multi-MB content values can produce page headers above PyArrow's
        # default Thrift limits. See https://github.com/marin-community/marin/issues/5334.
        parquet_file = pq.ParquetFile(
            source,
            thrift_string_size_limit=_THRIFT_DECODE_LIMIT_BYTES,
            thrift_container_size_limit=_THRIFT_DECODE_LIMIT_BYTES,
        )
        yield from parquet_file.iter_batches(batch_size=_ROWS_PER_BATCH)


def _download_once(task: _DownloadTask) -> _DownloadResult:
    ensure_parent_dir(task.destination_path)
    count = 0
    batches = _iter_parquet_batches(task.hf_path)
    first = next(batches, None)
    if first is None:
        counters.pipeline.update_counter("sec_edgar/empty_input", 1)
        return _DownloadResult(hf_path=task.hf_path, destination_path=task.destination_path, rows=0)
    with atomic_rename(task.destination_path) as temporary_path:
        with pq.ParquetWriter(temporary_path, first.schema) as writer:
            writer.write_batch(first)
            count += first.num_rows
            for batch in batches:
                writer.write_batch(batch)
                count += batch.num_rows
    counters.pipeline.update_counter("sec_edgar/rows_downloaded", count)
    return _DownloadResult(hf_path=task.hf_path, destination_path=task.destination_path, rows=count)


def _download_one(task: _DownloadTask) -> _DownloadResult:
    """Download one shard and return its source, destination, and row count."""
    return retry_with_backoff(
        lambda: _download_once(task),
        max_attempts=_MAX_RETRIES,
        backoff=_RETRY_BACKOFF,
        operation=f"download SEC-EDGAR shard {task.hf_path}",
    )


def _list_hf_parquets() -> list[str]:
    """List all upstream parquet paths in ``hf://datasets/...`` form, pinned to revision."""
    filesystem = HfFileSystem()
    paths: list[str] = []
    for filing_type in FILING_TYPES:
        pattern = str(_HF_REPOSITORY_ROOT / filing_type / "*.parquet")
        for parquet_path in filesystem.glob(pattern, revision=HF_REVISION):
            # glob(..., revision=...) embeds the pin in the repo segment
            # (``datasets/<repo>@<rev>/...``); resolve_path recovers the clean
            # in-repo path so the emitted hf:// URL stays revision-free (the pin
            # is reapplied at open time in _iter_parquet_batches).
            relative_path = filesystem.resolve_path(parquet_path).path_in_repo
            paths.append(str(_HF_URL_ROOT / relative_path))
    paths.sort()
    return paths


def download_sec_edgar(output_path: str) -> None:
    """Pull SEC-EDGAR from HF and re-emit PyArrow-readable shards under ``output_path``."""
    files = _list_hf_parquets()
    if not files:
        raise ValueError(f"No parquet files matched for {HF_DATASET_ID}@{HF_REVISION}")
    logger.info("Found %d upstream parquet files", len(files))

    tasks = [
        _DownloadTask(
            hf_path=path,
            destination_path=prefix_join(output_path, StoragePath(path).relative_to(_HF_URL_ROOT)),
        )
        for path in files
    ]

    pipeline = (
        Dataset.from_list(tasks)
        .map(_download_one)
        .write_jsonl(
            prefix_join(output_path, ".metrics/download-{shard:05d}-of-{total:05d}.jsonl"),
            skip_existing=True,
        )
    )
    context = ZephyrContext(name="download-sec-edgar", resources=ResourceConfig(cpu=1, ram="16g"))
    context.execute(pipeline)

    write_provenance_json(
        output_path,
        metadata={"dataset": HF_DATASET_ID, "version": HF_REVISION, "links": files},
    )
    logger.info("SEC-EDGAR download complete.")


def download_sec_edgar_step() -> StepSpec:
    return StepSpec(
        name="raw/sec-edgar",
        fn=download_sec_edgar,
        hash_attrs={
            "hf_dataset_id": HF_DATASET_ID,
            "revision": HF_REVISION,
            "filing_types": list(FILING_TYPES),
            "version": "v2",
        },
    )


def sec_edgar_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for SEC-EDGAR."""
    download = download_sec_edgar_step()
    return (
        download,
        normalize_step(
            name="normalized/sec-edgar",
            download=download,
            text_field="content",
            file_extensions=(".parquet",),
        ),
    )
