# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter chunk codecs: encode/decode items to/from on-disk chunk bytes.

Two codecs are available:
- ``PickleCodec``: zstd-compressed cloudpickle sub-batches (default, backward-compatible).
- ``ArrowIpcCodec``: Arrow IPC stream with built-in zstd body compression.

Both codecs are self-describing: the codec tag is stored in the sidecar so
readers automatically dispatch to the matching decoder without a manifest.

Codec selection:
    Controlled by ``ZEPHYR_SCATTER_CODEC`` env var (``pickle`` or ``arrow_ipc``)
    or by passing ``codec=`` to :class:`~zephyr.shuffle.ScatterWriter` and
    :func:`~zephyr.shuffle.scatter_write`. Default is ``pickle`` for back-compat.

Arrow IPC codec notes:
    Items must be serialisable by ``pyarrow.Table.from_pylist`` — dicts of
    scalars, strings, lists, and small nested dicts. Custom Python objects
    (dataclasses, frozensets, numpy arrays) are not supported; use ``pickle``
    for those pipelines.
"""

from __future__ import annotations

import io
import os
import pickle
from collections.abc import Iterator
from typing import Any

import cloudpickle
import pyarrow as pa
import pyarrow.ipc
import zstandard as zstd

_ZSTD_COMPRESS_LEVEL = 3
_SUB_BATCH_SIZE = 1024


class PickleCodec:
    """zstd-compressed cloudpickle sub-batches (default, handles arbitrary objects)."""

    tag = "pickle"

    def encode_chunk(self, items: list[Any]) -> bytes:
        raw = io.BytesIO()
        cctx = zstd.ZstdCompressor(level=_ZSTD_COMPRESS_LEVEL)
        with cctx.stream_writer(raw, closefd=False) as zf:
            for i in range(0, len(items), _SUB_BATCH_SIZE):
                cloudpickle.dump(items[i : i + _SUB_BATCH_SIZE], zf, protocol=pickle.HIGHEST_PROTOCOL)
        return raw.getvalue()

    def decode_chunk(self, blob: bytes) -> Iterator[Any]:
        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as reader:
            while True:
                try:
                    sub_batch = pickle.load(reader)
                except EOFError:
                    return
                yield from sub_batch


class ArrowIpcCodec:
    """Arrow IPC stream with built-in zstd body compression.

    Faster than pickle for JSON-shaped items (dicts of scalars, strings, and
    small nested containers). Not compatible with arbitrary Python objects.
    Raises ``pa.ArrowInvalid`` if items cannot be serialised to an Arrow table.
    """

    tag = "arrow_ipc"

    def encode_chunk(self, items: list[Any]) -> bytes:
        table = pa.Table.from_pylist(items)
        sink = io.BytesIO()
        options = pa.ipc.IpcWriteOptions(compression="zstd")
        with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
            writer.write_table(table)
        return sink.getvalue()

    def decode_chunk(self, blob: bytes) -> Iterator[Any]:
        with pa.ipc.open_stream(pa.BufferReader(blob)) as reader:
            for batch in reader:
                yield from batch.to_pylist()


_CODECS: dict[str, PickleCodec | ArrowIpcCodec] = {
    PickleCodec.tag: PickleCodec(),
    ArrowIpcCodec.tag: ArrowIpcCodec(),
}


def get_codec(tag: str) -> PickleCodec | ArrowIpcCodec:
    """Return the codec instance for a tag, raising on unknown tags."""
    try:
        return _CODECS[tag]
    except KeyError as err:
        raise ValueError(f"Unknown scatter codec {tag!r}. Valid: {sorted(_CODECS)}") from err


def default_codec() -> PickleCodec | ArrowIpcCodec:
    """Return the codec selected by ZEPHYR_SCATTER_CODEC (default: pickle)."""
    tag = os.environ.get("ZEPHYR_SCATTER_CODEC", PickleCodec.tag)
    return get_codec(tag)
