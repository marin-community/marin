# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rasterise a document in a child process the sender task is willing to lose.

A native abort in the rasteriser is a signal no ``except`` catches, and Zephyr answers a dead task
by restarting its shard from row zero, so the render runs out here where the task survives it.

The child streams one page at a time, so the sender can submit a page the moment it is rendered and
a long document never crosses as one payload; the child blocks writing a page the parent has not
read yet, so the pipe supplies the backpressure. A page crosses as PNG bytes, not as a base64 data
URI, which stays where the wire format asks for it in :func:`~...client.ocr_page`.

Deliberately ``subprocess`` rather than ``multiprocessing``: an Iris callable entrypoint runs at
module top level of ``__main__`` with no ``if __name__ == "__main__"`` guard, so ``spawn`` and
``forkserver`` would re-execute the job body in every child. :class:`RenderWorker` keeps one child
for the life of the task process and replaces it whenever it stops answering.

**Nothing here imports the pipeline.** The child is this module, the render module's arithmetic,
and ``pypdfium2`` and Pillow inside the functions that touch a document. It reports failures to its
caller instead of counting them; the counters are the caller's.
"""

import json
import logging
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from enum import StrEnum

from experiments.datakit.build_pdf_source.child_framing import READ_CHUNK, read_frame, write_frame
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderedPage,
    RenderOptions,
    iter_rendered_pages,
    open_pdf,
)

logger = logging.getLogger(__name__)

MODULE_NAME = "experiments.datakit.build_pdf_source.ocr_extract.render_worker"

# How long the parent waits for the *next* page, not for the document: a per-page deadline stays
# tight while a document is progressing. Breaching it ends the document.
PAGE_DEADLINE = 30.0

# Why a document stopped, when the child did not say; named as
# :mod:`~experiments.datakit.build_pdf_source.extract_inspector` names them.
WORKER_DIED = "worker_died"
DEADLINE_EXCEEDED = "deadline_exceeded"
# The child wrote something that is not a frame, so its stream is desynchronised and it is retired.
PROTOCOL_ERROR = "protocol_error"

_ERROR_CHARS = 500


class Frame(StrEnum):
    """What a reply frame is.

    A document is a ``START``, then a ``PAGE`` per page the rasteriser produced, then an ``END``. A
    document the child could not open at all is an ``END`` on its own.
    """

    START = "start"
    PAGE = "page"
    END = "end"


class RenderFailure(Exception):
    """A document the isolated rasteriser did not finish, and why.

    ``reason`` is what the caller counts under ``render_failed/``: :data:`WORKER_DIED`,
    :data:`DEADLINE_EXCEEDED`, :data:`PROTOCOL_ERROR`, or the exception type the child reported.
    Raised after the pages the child did stream, never instead of them.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


class RenderStream:
    """One document's pages as the child streams them back.

    A context manager because a stream abandoned part-way leaves the child mid-reply, and a child
    mid-reply cannot be handed the next document; closing the stream retires it in that case.
    """

    def __init__(self, declared_pages: int, pages: Iterator[RenderedPage]) -> None:
        # Every page the document declares, before the page budget truncates anything: the
        # denominator ``pages_unrendered`` is measured against.
        self.declared_pages = declared_pages
        self._pages = pages

    def __enter__(self) -> "RenderStream":
        return self

    def __exit__(self, *exception) -> None:
        self._pages.close()

    def __iter__(self) -> Iterator[RenderedPage]:
        return self._pages


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------


def render_document(pdf: bytes, options: RenderOptions, stdout) -> None:
    """Stream one document's pages, then one frame saying how it ended.

    Per-page failures stay in :func:`~...render.iter_rendered_pages`, visible to the parent as a gap
    in ``page_index``; the ``END`` frame carries a failure of the document itself.
    """
    error_type: str | None = None
    error: str | None = None
    try:
        with open_pdf(pdf) as document:
            write_frame(stdout, {"frame": Frame.START, "declared_pages": len(document)})
            for page in iter_rendered_pages(document, options):
                header = {
                    "frame": Frame.PAGE,
                    "page_index": page.page_index,
                    "pixels": page.pixels,
                    "dpi": page.dpi,
                    "size": len(page.png),
                }
                write_frame(stdout, header, page.png)
    except Exception as failure:
        error_type, error = type(failure).__name__, f"{type(failure).__name__}: {failure}"[:_ERROR_CHARS]
    write_frame(stdout, {"frame": Frame.END, "error_type": error_type, "error": error})


def worker_main() -> None:
    """Render length-prefixed PDFs from stdin until the parent closes it."""
    import faulthandler  # noqa: PLC0415 - only the disposable child needs a fault handler

    faulthandler.enable()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        frame = read_frame(stdin)
        if frame is None:
            return
        request, pdf = frame
        render_document(pdf, RenderOptions(**request["options"]), stdout)


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------


class RenderWorker:
    """A rasteriser subprocess, bounded by a per-page deadline and replaced when it stops answering.

    It replaces itself rather than being replaced by its caller, so a sender process can hold one
    handle for its whole life. A child orphaned by a dying parent reads EOF on its stdin and exits.
    """

    def __init__(self, deadline: float = PAGE_DEADLINE) -> None:
        self._deadline = deadline
        self._process: subprocess.Popen | None = None
        self._selector = selectors.DefaultSelector()
        self._buffer = bytearray()
        self._streaming = False
        self._eof = False
        self.spawns = 0
        self.start()

    def start(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", MODULE_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr stays on the job log so an abort message survives the process that printed it.
            env=os.environ.copy(),
        )
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self._buffer.clear()
        self._streaming = False
        self._eof = False
        self.spawns += 1

    def stop(self) -> None:
        if self._process is None:
            return
        self._selector.unregister(self._process.stdout)
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._process = None

    def render(self, pdf: bytes, options: RenderOptions) -> RenderStream:
        """Start one document in the child, returning its pages as they arrive.

        Raises :class:`RenderFailure` if the child never gets as far as opening the document, which
        covers both a child that died on the last one and a document PDFium refuses outright.
        """
        deadline = time.monotonic() + self._deadline
        try:
            write_frame(
                self._process.stdin,
                {"size": len(pdf), "options": asdict(options)},
                pdf,
            )
        except OSError as error:
            raise self._failure(WORKER_DIED, f"{type(error).__name__}: {error}") from error

        self._streaming = True
        header = self._read_header(deadline)
        if header["frame"] == Frame.END:
            # The child is alive; this document is not.
            self._streaming = False
            raise RenderFailure(header["error_type"], header["error"])
        return RenderStream(header["declared_pages"], self._stream_pages())

    def _stream_pages(self) -> Iterator[RenderedPage]:
        try:
            while True:
                header = self._read_header(time.monotonic() + self._deadline)
                if header["frame"] == Frame.END:
                    self._streaming = False
                    if header["error"] is not None:
                        raise RenderFailure(header["error_type"], header["error"])
                    return
                payload = self._read_payload(header["size"], time.monotonic() + self._deadline)
                yield RenderedPage(
                    png=payload,
                    page_index=header["page_index"],
                    pixels=header["pixels"],
                    dpi=header["dpi"],
                )
        finally:
            # Still mid-document means the caller stopped consuming; a child part-way through a
            # reply cannot be handed the next document.
            if self._streaming:
                self._replace("the stream was abandoned mid-document")

    def _read_header(self, deadline: float) -> dict:
        line = self._read_line(deadline)
        if line is None:
            raise self._failure(*self._silence())
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise self._failure(PROTOCOL_ERROR, f"{line[:_ERROR_CHARS]!r} is not a frame") from error

    def _read_payload(self, size: int, deadline: float) -> bytes:
        while len(self._buffer) < size:
            if not self._fill(deadline):
                raise self._failure(*self._silence())
        payload = bytes(self._buffer[:size])
        del self._buffer[:size]
        return payload

    def _read_line(self, deadline: float) -> str | None:
        while b"\n" not in self._buffer:
            if not self._fill(deadline):
                return None
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        return line.decode()

    def _fill(self, deadline: float) -> bool:
        """Take whatever the child has ready, or report that it has stopped talking.

        Reads the descriptor rather than ``Popen``'s file object, which cannot bound how long it blocks.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._selector.select(remaining):
            return False
        chunk = os.read(self._process.stdout.fileno(), READ_CHUNK)
        if not chunk:
            self._eof = True
            return False
        self._buffer.extend(chunk)
        return True

    def _silence(self) -> tuple[str, str]:
        """Why nothing came back: the child is gone, or it is still inside the library.

        The descriptor decides, not ``poll()``, which can still say "running" on the read that saw
        the EOF.
        """
        if not self._eof:
            return DEADLINE_EXCEEDED, f"no page within {self._deadline:.0f}s"
        code = self._process.wait()
        if code < 0:
            return WORKER_DIED, f"worker killed by {signal.Signals(-code).name}"
        return WORKER_DIED, f"worker exited with {code}"

    def _replace(self, detail: str) -> None:
        """Retire the child and start a fresh one."""
        logger.warning("Replacing the render worker: %s", detail)
        self.stop()
        self.start()

    def _failure(self, reason: str, detail: str) -> RenderFailure:
        """Replace the child and describe what it cost, for the caller to raise."""
        self._replace(detail)
        return RenderFailure(reason, detail)


if __name__ == "__main__":
    worker_main()
