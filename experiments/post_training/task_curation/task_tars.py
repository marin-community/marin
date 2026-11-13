# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read task archives and cohort parquet payloads."""

import gzip
import io
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq


def open_task_tar(payload: bytes) -> tarfile.TarFile:
    """Open a task's tar archive, gzip-decompressed when the payload is gzipped."""
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return tarfile.open(fileobj=io.BytesIO(raw))


def iter_task_payloads(path: Path) -> Iterator[tuple[str, bytes]]:
    """Each (task_id, payload) pair in a cohort parquet's ``task_binary`` column."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob)
