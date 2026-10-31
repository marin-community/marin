# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded binary messages carried by text-only process transports.

Each line contains a one-byte message kind and a base64 payload. This module
uses only the standard library so its source can accompany standalone guests.
Raw subprocess pipes do not use this encoding.
"""

import base64
from collections.abc import Iterator

CHUNK_BYTES = 65536
FRAME_BYTES = 1 + 4 * ((CHUNK_BYTES + 2) // 3)
STDIN, STDOUT, STDERR, EXIT = b"I", b"D", b"E", b"X"


def encode_frame(kind: bytes, data: bytes) -> bytes:
    if len(kind) != 1 or len(data) > CHUNK_BYTES:
        raise ValueError("Process frame exceeds its limits")
    return kind + base64.b64encode(data) + b"\n"


def decode_frame(frame: bytes) -> tuple[bytes, bytes]:
    if not frame or len(frame) > FRAME_BYTES:
        raise ValueError("Process frame exceeds its limits")
    data = base64.b64decode(frame[1:], validate=True)
    if len(data) > CHUNK_BYTES:
        raise ValueError("Process frame exceeds its byte limit")
    return frame[:1], data


class FrameDecoder:
    """Assemble bounded lines across fragmented and coalesced transport reads."""

    def __init__(self):
        self._pending = bytearray()

    def feed(self, data: bytes) -> Iterator[bytes]:
        for offset in range(0, len(data), FRAME_BYTES):
            self._pending.extend(data[offset : offset + FRAME_BYTES])
            while (end := self._pending.find(b"\n")) >= 0:
                if end > FRAME_BYTES:
                    raise ValueError("Process frame exceeds its limit")
                frame = bytes(self._pending[:end])
                del self._pending[: end + 1]
                yield frame
            if len(self._pending) > FRAME_BYTES:
                raise ValueError("Process frame exceeds its limit")

    def finish(self) -> None:
        if self._pending:
            raise ValueError("Process stream ended with an incomplete frame")
