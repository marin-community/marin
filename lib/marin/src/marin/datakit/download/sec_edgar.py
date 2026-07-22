# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize TeraflopAI/SEC-EDGAR filings."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem, hf_hub_download
from rigging.filesystem import StoragePath, prefix_join
from rigging.timing import ExponentialBackoff, retry_with_backoff
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import atomic_rename, ensure_parent_dir, parquet_sink

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"
_HF_REPOSITORY_ROOT = StoragePath("datasets") / HF_DATASET_ID
_HF_URL_ROOT = StoragePath("hf://datasets") / HF_DATASET_ID

FILING_TYPES = ("10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5")

# A small batch reader keeps memory bounded while one big SEC row group
# (~700 MB decompressed) is in flight.
_ROWS_PER_BATCH = 8

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


def _path_in_repo(hf_path: str) -> str:
    """Return a shard URL's in-repo path (e.g. ``10-K/<uuid>-<n>.parquet``)."""
    return str(StoragePath(hf_path).relative_to(_HF_URL_ROOT))


@contextmanager
def _fetch_shard(hf_path: str, *, revision: str = HF_REVISION) -> Iterator[str]:
    """Download one shard to a local temp file and yield its path.

    DuckDB (the reader in :func:`_iter_parquet_batches`) needs a local file, and
    ``hf_hub_download`` provides one with a resumable, integrity-checked,
    revision-pinned transfer.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=_path_in_repo(hf_path),
            revision=revision,
            repo_type="dataset",
            local_dir=tmp_dir,
        )


def _iter_parquet_batches(local_path: str) -> Iterator[pa.RecordBatch]:
    """Yield record batches from a locally-downloaded SEC-EDGAR shard."""
    # Read with DuckDB, not PyArrow: SEC's multi-MB `content` values produce
    # parquet page-header statistics that PyArrow 23's Thrift decoder cannot
    # deserialize ("Couldn't deserialize thrift: No more data to read",
    # apache/arrow#46404), even with raised decoder limits, while DuckDB reads
    # them. _download_once re-encodes the batches into pyarrow-readable shards.
    connection = duckdb.connect()
    try:
        reader = connection.execute("SELECT * FROM read_parquet($shard)", {"shard": local_path}).to_arrow_reader(
            _ROWS_PER_BATCH
        )
        yield from reader
    finally:
        connection.close()


def _download_once(task: _DownloadTask) -> _DownloadResult:
    ensure_parent_dir(task.destination_path)
    count = 0
    with _fetch_shard(task.hf_path) as local_shard:
        batches = _iter_parquet_batches(local_shard)
        first = next(batches, None)
        if first is None:
            counters.pipeline.update_counter("sec_edgar/empty_input", 1)
            return _DownloadResult(hf_path=task.hf_path, destination_path=task.destination_path, rows=0)
        with atomic_rename(task.destination_path) as temporary_path:
            # Route the write through zephyr's sink so the parquet stream reaches a
            # pyarrow filesystem carrying the S3 endpoint's virtual-host addressing.
            # A bare s3:// string builds a default path-style client, which CoreWeave
            # object storage rejects on multipart upload (PathStyleRequestNotAllowed).
            with parquet_sink(temporary_path) as (where_fd, native_fs):
                with pq.ParquetWriter(where_fd, first.schema, filesystem=native_fs) as writer:
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
            # is reapplied when the shard is fetched in _fetch_shard).
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
    # disk headroom: each shard is now staged to a local temp file for the DuckDB
    # read + re-encode, so a worker holds one full shard on disk at a time.
    context = ZephyrContext(name="download-sec-edgar", resources=ResourceConfig(cpu=1, ram="16g", disk="10g"))
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
