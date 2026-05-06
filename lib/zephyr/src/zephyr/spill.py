# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chunked Arrow IPC spill files for external sort.

SpillWriter accepts pa.RecordBatch objects and writes them as a
zstd-compressed Arrow IPC stream.  SpillReader yields pa.RecordBatch
objects from a previously written file.
"""

import logging
from collections.abc import Iterator

import pyarrow as pa
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

_IPC_WRITE_OPTIONS = pa.ipc.IpcWriteOptions(compression=pa.Codec("zstd", compression_level=3))


class SpillWriter:
    """Writes RecordBatches to a zstd-compressed Arrow IPC spill file."""

    def __init__(self, path: str) -> None:
        self._path = path
        fs, fs_path = url_to_fs(path)
        self._out = fs.open(fs_path, "wb")
        self._ipc_writer: pa.ipc.RecordBatchWriter | None = None
        self._closed = False

    def write(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        if self._ipc_writer is None:
            self._ipc_writer = pa.ipc.new_stream(self._out, batch.schema, options=_IPC_WRITE_OPTIONS)
        self._ipc_writer.write_batch(batch)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ipc_writer is not None:
            self._ipc_writer.close()
        self._out.close()

    def __enter__(self) -> "SpillWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SpillReader:
    """Reads RecordBatches from a zstd-compressed Arrow IPC spill file."""

    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        fs, fs_path = url_to_fs(self._path)
        with fs.open(fs_path, "rb") as f:
            with pa.ipc.open_stream(f) as reader:
                for batch in reader:  # noqa: UP028 — cannot yield from inside nested context managers
                    yield batch
