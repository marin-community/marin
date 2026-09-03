# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frames for the child processes this pipeline runs its native PDF libraries in.

Two steps here refuse to open a PDF in the map task and talk to a child over its stdio instead:
:mod:`~experiments.datakit.build_pdf_source.extract_inspector` for pdf-inspector and the
rasteriser's geometry pass, and
:mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker` for the rasteriser's pixels.

A frame is one newline-terminated JSON header followed by ``header["size"]`` bytes of raw payload
(a PDF one way, a PNG the other). A header with no ``size`` carries no payload.
"""

import json

# Bounds the syscall count per read; pipe reads hand over at most the pipe's buffer anyway.
READ_CHUNK = 1 << 16


def write_frame(stream, header: dict, payload: bytes = b"") -> None:
    """Write one frame and flush it. The caller sets ``header["size"]`` to the payload's length."""
    stream.write(json.dumps(header).encode() + b"\n")
    stream.write(payload)
    stream.flush()


def read_exactly(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes, or raise. Pipe reads are short whenever the writer is."""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(min(READ_CHUNK, size - len(buffer)))
        if not chunk:
            raise EOFError(f"stream closed after {len(buffer)} of {size} bytes")
        buffer.extend(chunk)
    return bytes(buffer)


def read_frame(stream) -> tuple[dict, bytes] | None:
    """The next frame as ``(header, payload)``, or ``None`` once the writer has closed the stream.

    Blocking, so this is the child's side; a parent bounds its reads with a deadline instead.
    """
    line = stream.readline()
    if not line:
        return None
    header = json.loads(line)
    return header, read_exactly(stream, header.get("size", 0))
