# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming reader for zstd-compressed JSONL download sources."""

import json
from collections.abc import Iterator

import zstandard

_READ_CHUNK_BYTES = 1024 * 1024


def iter_jsonl_from_zstd_stream(raw_stream) -> Iterator[dict]:
    """Yield parsed JSON objects from a zstd-compressed JSONL stream.

    The stream is decompressed and split on newlines incrementally so memory
    stays O(record) rather than O(file). The final record is flushed even when
    the stream lacks a trailing newline.
    """
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(raw_stream) as reader:
        buf = bytearray()
        while True:
            chunk = reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                newline_pos = buf.find(b"\n")
                if newline_pos < 0:
                    break
                line_bytes = bytes(buf[:newline_pos])
                del buf[: newline_pos + 1]
                if not line_bytes.strip():
                    continue
                yield json.loads(line_bytes)
        # Flush trailing bytes (last record may lack a trailing newline).
        if buf.strip():
            yield json.loads(bytes(buf))
