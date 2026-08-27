# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frames for the child processes this pipeline runs its native PDF libraries in.

Two steps here refuse to open a PDF in the map task and talk to a child over its stdio instead:
:mod:`~experiments.datakit.build_pdf_source.extract_inspector` for pdf-inspector and the
rasteriser's geometry pass, and
:mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker` for the rasteriser's pixels.

A frame is one newline-terminated JSON header followed by ``header["size"]`` bytes of payload. The
payload is outside the JSON because both directions carry hundreds of kilobytes of binary per frame
-- a PDF one way, a PNG the other -- and JSON has no byte string, so the alternative is base64 plus
an escape scan over every one of those bytes on the writing side and a decode on the reading side. A
header with no ``size`` carries no payload, which is what a reply made only of fields looks like.
"""

import json

# Pipe reads hand over at most the pipe's buffer, so this bounds a syscall count rather than an
# allocation: a larger chunk does not make a 458 KiB page arrive in one read.
READ_CHUNK = 1 << 16


def write_frame(stream, header: dict, payload: bytes = b"") -> None:
    """Write one frame and flush it, so a reader is never left holding half of one.

    The caller owns ``header["size"]``. It is the payload's length, and it has to be in the header
    because the reader cannot frame anything until it has been told how much to expect.
    """
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

    Blocking, so this is the children's side of the conversation. A parent bounds its reads with a
    deadline instead -- the whole reason the library is out here is that it can stop answering.
    """
    line = stream.readline()
    if not line:
        return None
    header = json.loads(line)
    return header, read_exactly(stream, header.get("size", 0))
