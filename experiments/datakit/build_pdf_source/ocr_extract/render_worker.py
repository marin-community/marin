# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rasterise a document in a child process the sender task is willing to lose.

The rasteriser is the only native library the OCR route's map tasks touch, and in process it is the
only thing in them that can end a task without raising: a native abort is a signal no ``except``
catches. Zephyr answers a dead task by restarting its shard from row zero, three times, with no
poison-pill detection, so a deterministic abort exhausts the retry budget and fails the stage
permanently -- throwing away every page the shard had already OCR'd.

PDFium makes that unlikely rather than impossible. Rendering every page of the 100,000-document
oracle sample on both architectures found zero native aborts in 3,577,944 renders, against MuPDF's
one deterministic ``SIGSEGV`` repeating on 3 of 3 retries (``pdfium-evaluation.md`` on the
``mark/pdf_processing`` campaign branch). The point estimate is zero blocking documents crawl-wide
and the 95% bound on 0/3,577,944 is still tens of them; the cost of meeting one in process is the
whole stage, so the render runs out here where the task survives it.

**The child streams, one page at a time.** :func:`~...render.iter_rendered_pages` is lazy for two
reasons a render-the-whole-document round trip would break, and both of them outrank the simpler
protocol. The sender overlaps rendering with waiting on the GPU fleet -- a page is submitted the
moment it is rendered -- so a batched render would idle the fleet for the length of every document.
And an encoded page is well over a megabyte, so a thousand-page document would arrive as a
multi-gigabyte payload. The pipe supplies the backpressure for nothing: the child blocks writing a
page the parent has not read yet, so it runs at most one page and one pipe buffer ahead.

**A page crosses as PNG bytes, not as a base64 data URI.** The encoder's output is ~458 KiB per
page; the data URI is ~610 KiB of it, and sending that would run ``json.dumps`` over the string in
the child and ``json.loads`` over it in the parent to deliver exactly what
:func:`~...client.ocr_page` builds for itself in one line. Base64 stays where the wire format asks
for it and the pipe carries the bytes as themselves. Measured against handing the same page over
in process, on a 400-page stream repeated 15 times: this protocol adds **0.055 ms of CPU per page
on Grace and 0.147 on x86**, where carrying the data URI inside the JSON header adds 0.446 and
0.607 -- 8.1x and 4.1x. The page costs the feed ~50 ms, so one of those is worth paying and the
other is a third of a per-cent of the fleet for nothing.

Deliberately ``subprocess`` rather than ``multiprocessing``: an Iris callable entrypoint runs at
module top level of ``__main__`` with no ``if __name__ == "__main__"`` guard, so both ``spawn`` and
``forkserver`` would re-execute the job body in every child. A child per document is not an option
either -- an interpreter start and the rasteriser's import per document, against a page that costs
tens of milliseconds -- so :class:`RenderWorker` keeps one for the life of the task process and
replaces it whenever it stops answering.

**Nothing here imports the pipeline.** The child is this module and what it pulls in: the render
module's arithmetic, and ``pypdfium2`` and Pillow inside the functions that touch a document.
Starting one therefore costs an interpreter rather than pyarrow, Zephyr and the Marin execution
stack, and it reports failures to its caller instead of counting them for the same reason -- the
counters are the caller's, and the caller already has to record the failure against the document.
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

# How long the parent will wait for the *next* page, not for the document. A page is tens of
# milliseconds and a thousand-page document is legitimately minutes, so a whole-document deadline
# would have to be loose enough to be no bound at all; a per-page one stays tight while a document
# is progressing. Breaching it ends the document, so a task's exposure to a stalled child is this
# long once per document rather than once per page.
PAGE_DEADLINE = 30.0

# Why a document stopped, when the child did not say. Named as
# :mod:`~experiments.datakit.build_pdf_source.extract_inspector` names them, because they are the
# same two things happening to the same kind of child.
WORKER_DIED = "worker_died"
DEADLINE_EXCEEDED = "deadline_exceeded"
# The child wrote something that is not a frame. Anything printing to its stdout desynchronises the
# stream for every document after this one, so the child is retired rather than reused.
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
    :data:`DEADLINE_EXCEEDED`, :data:`PROTOCOL_ERROR`, or the exception type the child reported,
    which keeps the counter vocabulary the in-process render loop already used.

    Raised after the pages the child did stream, never instead of them: a document that fails
    part-way keeps what it produced, exactly as it did when the loop ran in the map task.
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
        # Every page the document declares, before the page budget truncates anything. It is the
        # denominator ``pages_unrendered`` is measured against, so it has to arrive before the
        # pages rather than be inferred from how many turned up.
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

    Per-page failures are :func:`~...render.iter_rendered_pages`'s own business and stay there: a
    page PDFium refuses is skipped, visible to the parent as a gap in ``page_index``. What reaches
    the ``END`` frame is a failure of the *document* -- bytes that are not a PDF, a page tree that
    cannot be walked -- which the parent records against the row and counts by exception type.
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
            # The child is alive and well; this document is not. Its own failure, its own name.
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
            # Still mid-document means the caller stopped consuming -- a deadline or a death has
            # already replaced the child by the time it raises. Either way this one is part-way
            # through a reply and cannot be handed the next document.
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

        Reads the descriptor rather than ``Popen``'s file object: a buffered reader has no way to
        bound how long it blocks, and mixing the two loses whatever it has already buffered.
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

        The descriptor decides, not ``poll()``. A child that aborts closes its stdout and reports
        its status through a separate mechanism that can lag by enough for ``poll()`` to still say
        "running" on the read that saw the EOF -- which would file a native abort under the
        deadline's counter and send the next reader looking for a hang that never happened.
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
