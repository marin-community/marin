# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 of the pdf-inspector evaluation: does the Rust extractor survive crawl PDFs, and what does it cost?

``pdf-inspector`` is a pure-Rust PDF classifier and Markdown extractor (``lopdf``, no models, no
torch) proposed as a cheaper front end to -- or replacement for -- the Docling CPU route in
:mod:`~experiments.datakit.build_pdf_source.docling_extract`. Its vendor benchmark reports ~2.35 ms
per *document* over a curated 200-document corpus. Marin's corpus is neither curated nor 200
documents: the crawl averages ~17.8 pages per PDF and includes the truncated, malformed, encrypted
and pathological files a crawl always includes. This module measures the library against that
corpus before anything is built on top of it.

The gate has three questions, and the second is the one that decides the Stage 1 design:

*Cost.* Wall-clock for :func:`pdf_inspector.classify_pdf_bytes` and
:func:`pdf_inspector.extract_pages_markdown_bytes`, reported per document and per page at p50/p90/p99.
The number to beat is Docling's ~1000 ms/page on CPU. Timing is measured around the library call
alone, inside the process that makes it, so neither the parquet read nor the harness appears in it.

*Survival.* A native extension can fail in ways a Python library cannot. A Rust panic compiled under
``panic = "abort"`` calls ``abort(3)`` and takes the interpreter with it; an unbounded allocation in
the ``bfrange``/``cidrange``/Form-XObject expansions that upstream has spent its recent history
bounding gets the process OOM-killed; a pathological content stream can simply not return. None of
those are catchable with ``try``/``except`` in the calling process, so a probe that ran the library
in-process would measure its own death rather than the library's failure rate.

*Isolation cost.* Whether Stage 1 needs to pay for process isolation at all.

So every document is handed to a **persistent worker subprocess** over a length-prefixed pipe, one
document at a time, and the driver imposes ``CALL_TIMEOUT`` on the reply. That turns all four
failure modes into observations rather than casualties:

``exception`` / ``panic`` / ``memory``
    The worker survived and reported it. ``panic`` is
    ``pyo3_runtime.PanicException``, which means the crate unwinds and PyO3 caught it -- catchable,
    and Stage 1 needs no isolation for this class. Note it derives from ``BaseException``, not
    ``Exception``, so the worker catches the former; an ``except Exception`` here would report a
    panic as a worker death and invert the verdict.
``worker_died``
    The child exited or took a fatal signal. ``SIGABRT`` is a ``panic = "abort"`` build or a Rust
    ``abort``; ``SIGSEGV`` is memory unsafety; ``SIGKILL`` is the OOM killer. Any of these mean a
    single crawl PDF can kill a Stage 1 worker outright, and Stage 1 must isolate.
``timeout``
    No reply within ``CALL_TIMEOUT``. The library exposes no deadline, page cap or byte cap of its
    own, so an unbounded document can only be bounded from outside.

The worker sets ``RLIMIT_AS`` to ``MEMORY_LIMIT`` before importing the library, so a runaway
expansion raises ``MemoryError`` in the child instead of inviting the kernel to kill whatever on the
node happens to be largest. That is a measurement aid, not a Stage 1 recommendation: it converts an
unbounded allocation into an attributable per-document failure.

The probe is deliberately **single-threaded and single-task**, which is the one place in this
repository where scaling out would be wrong: the deliverable is a latency distribution, and
co-tenant tasks on the same node would measure scheduler contention instead. ``PROBE_DOCUMENTS``
documents at even 100 ms each is a couple of minutes of work, so there is no wall-clock to buy. The
task still reserves ``--cpu 8`` so nothing of its own competes with the process being timed.

Documents are drawn from the first ``SHARD_COUNT`` shards of the 100k oracle sample
(:mod:`~experiments.datakit.build_pdf_source.quality.build_oracle_sample`) sorted by ``source_id``
and truncated, so every architecture measures the same documents in the same order and the x86 and
aarch64 runs are comparable row for row.

Run it once per architecture. ``cw-us-east-02a`` is Emerald Rapids x86_64; a CPU-only task on
``cw-us-east-08a`` lands on the Grace pool and is aarch64, which is the architecture Stage 1 would
run on there:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-inspector-probe-x86 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdf_inspector

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdf-inspector-probe-aarch64 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdf_inspector

``--extra pdf`` is required: the library lives in marin-core's ``pdf`` extra, and the worker venv is
built from the repository's declared dependencies plus the extras passed here, never from a local
one.
"""

import faulthandler
import json
import logging
import math
import os
import platform
import resource
import selectors
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version

import fsspec
import polars as pl
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
SAMPLE_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_sample100k"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_pdf_inspector_probe"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.probe_pdf_inspector"
WORKER_FLAG = "--worker"

PROBE_DOCUMENTS = 1000
# Two shards of the 178 hold ~1,360 documents, comfortably over PROBE_DOCUMENTS.
SHARD_COUNT = 2
READ_COLUMNS = ("source_id", "url", "num_pages", "pdf")

# Generous against a library that claims single-digit milliseconds: a document still running after
# this long is a hang for any practical purpose, and Docling's own ~1000 ms/page would put a
# median-length crawl PDF at ~18 s.
CALL_TIMEOUT = 30.0
# Converts an unbounded expansion into an attributable MemoryError instead of an OOM kill.
MEMORY_LIMIT = 8 * 1024**3
READ_CHUNK = 1 << 16


class Op(StrEnum):
    """The two entry points under evaluation."""

    CLASSIFY = "classify"
    EXTRACT = "extract"


class Outcome(StrEnum):
    """What became of one library call.

    The first four are reported by a worker that survived; the last two are the driver's verdict on
    a worker that did not.
    """

    OK = "ok"
    EXCEPTION = "exception"
    PANIC = "panic"
    MEMORY = "memory"
    WORKER_DIED = "worker_died"
    TIMEOUT = "timeout"


SURVIVED_OUTCOMES = (Outcome.OK, Outcome.EXCEPTION, Outcome.PANIC, Outcome.MEMORY)


# ---------------------------------------------------------------------------
# Worker: one document at a time, in a process the driver is willing to lose
# ---------------------------------------------------------------------------


def _read_exactly(stream, size: int) -> bytes:
    payload = stream.read(size)
    if payload is None or len(payload) != size:
        raise EOFError(f"expected {size} bytes, got {0 if payload is None else len(payload)}")
    return payload


def _classified(result) -> dict:
    return {
        "page_count": result.page_count,
        "pdf_type": result.pdf_type,
        "pages_needing_ocr": len(result.pages_needing_ocr),
        "confidence": float(result.confidence),
    }


def _extracted(result) -> dict:
    return {
        "page_count": len(result.pages),
        "pdf_type": None,
        "pages_needing_ocr": len(result.pages_needing_ocr),
        "markdown_chars": sum(len(page.markdown) for page in result.pages),
        "is_complex": result.is_complex,
    }


def _failure_outcome(error: BaseException) -> Outcome:
    if isinstance(error, MemoryError):
        return Outcome.MEMORY
    # PyO3 names its unwind-catching exception PanicException and derives it from BaseException.
    # Matching on the name avoids importing pyo3_runtime, which only exists once a panic has fired.
    if type(error).__name__ == "PanicException":
        return Outcome.PANIC
    return Outcome.EXCEPTION


def _measure(module, op: str, payload: bytes) -> dict:
    """Time one library call, reporting whatever it did instead of raising."""
    started = time.perf_counter()
    try:
        if op == Op.CLASSIFY:
            observed = _classified(module.classify_pdf_bytes(payload))
        else:
            observed = _extracted(module.extract_pages_markdown_bytes(payload))
    except (KeyboardInterrupt, SystemExit):
        raise
    # BaseException, not Exception: PyO3 derives PanicException from the former, and a panic
    # reported as a worker death would invert this probe's central conclusion.
    except BaseException as error:
        return {
            "outcome": str(_failure_outcome(error)),
            "milliseconds": 1000 * (time.perf_counter() - started),
            "error": f"{type(error).__name__}: {error}"[:500],
        }
    return {"outcome": str(Outcome.OK), "milliseconds": 1000 * (time.perf_counter() - started), **observed}


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    Nothing but the JSON reply is written to stdout; the library's own panic and abort messages go
    to stderr, which the driver leaves attached to the job log so a fatal one is still legible after
    the process is gone.
    """
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))

    import pdf_inspector  # noqa: PLC0415 - the whole point is to import it in the disposable process

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = _read_exactly(stdin, request["size"])
        stdout.write(json.dumps(_measure(pdf_inspector, request["op"], payload)).encode() + b"\n")
        stdout.flush()


# ---------------------------------------------------------------------------
# Driver: hand out documents, outlive the worker
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    """A worker's answer, or the driver's account of why there wasn't one."""

    result: dict
    worker_lost: bool


class Worker:
    """A subprocess running :func:`worker_main`, replaced whenever it dies.

    Deliberately ``subprocess`` rather than ``multiprocessing``: an Iris callable entrypoint runs at
    module top level of ``__main__`` with no ``if __name__ == "__main__"`` guard, so both ``spawn``
    and ``forkserver`` would re-execute the job body in every child.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._selector = selectors.DefaultSelector()
        self.spawns = 0
        self.start()

    def start(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", MODULE_NAME, WORKER_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr stays on the job log so a panic or abort message survives the process.
            env=os.environ.copy(),
        )
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self.spawns += 1

    def stop(self) -> None:
        if self._process is None:
            return
        self._selector.unregister(self._process.stdout)
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._process = None

    def _death(self) -> dict:
        """How the worker died, named rather than numbered."""
        code = self._process.poll()
        if code is not None and code < 0:
            name = signal.Signals(-code).name
            return {"outcome": str(Outcome.WORKER_DIED), "error": f"killed by {name}", "worker_signal": name}
        return {"outcome": str(Outcome.WORKER_DIED), "error": f"exited with {code}", "worker_exit_code": code}

    def _read_reply(self, deadline: float) -> str | None:
        """One newline-terminated reply, or ``None`` on timeout or on the worker's EOF."""
        buffer = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                return None
            chunk = os.read(self._process.stdout.fileno(), READ_CHUNK)
            if not chunk:
                return None
            buffer.extend(chunk)
            if b"\n" in buffer:
                return buffer.split(b"\n", 1)[0].decode()

    def call(self, op: Op, payload: bytes) -> Reply:
        """Run one document through the worker, replacing it if it does not come back."""
        deadline = time.monotonic() + CALL_TIMEOUT
        try:
            self._process.stdin.write(json.dumps({"op": str(op), "size": len(payload)}).encode() + b"\n")
            self._process.stdin.write(payload)
            self._process.stdin.flush()
            line = self._read_reply(deadline)
        except (BrokenPipeError, OSError):
            line = None

        if line is not None:
            return Reply(result=json.loads(line), worker_lost=False)

        # No reply: either the worker is gone, or it is still inside a call it will not leave.
        # `poll` after a short grace distinguishes them -- a dead child has a status, a hung one
        # does not -- and either way the process is replaced before the next document.
        try:
            self._process.wait(timeout=1.0)
            result = self._death()
        except subprocess.TimeoutExpired:
            result = {"outcome": str(Outcome.TIMEOUT), "error": f"no reply within {CALL_TIMEOUT:.0f}s"}
        self.stop()
        self.start()
        return Reply(result=result, worker_lost=True)


# ---------------------------------------------------------------------------
# Sample, run, summarize
# ---------------------------------------------------------------------------


def storage() -> fsspec.AbstractFileSystem:
    configure_coreweave_s3()
    return fsspec.filesystem("s3")


def probe_documents(fs: fsspec.AbstractFileSystem) -> pl.DataFrame:
    """The fixed document set: the first shards, sorted by ``source_id``, truncated.

    Sorted before truncation so the draw depends on the corpus rather than on row order within a
    shard, and so a run on another architecture measures these same documents.
    """
    paths = sorted(str(path) for path in StoragePath(f"{SAMPLE_PREFIX}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"no sample shards under {SAMPLE_PREFIX}")

    frames = []
    for path in paths[:SHARD_COUNT]:
        with fs.open(path, "rb") as stream:
            frames.append(pl.read_parquet(stream, columns=list(READ_COLUMNS)))
    documents = pl.concat(frames).sort("source_id").head(PROBE_DOCUMENTS)
    if documents.height < PROBE_DOCUMENTS:
        raise ValueError(f"{SHARD_COUNT} shards hold only {documents.height} documents; need {PROBE_DOCUMENTS}")
    logger.info("probe: %d documents from %d shards", documents.height, SHARD_COUNT)
    return documents


def probe_row(worker: Worker, document: dict) -> dict:
    """Both calls against one document, flattened into one row keyed by operation."""
    row = {
        "source_id": document["source_id"],
        "url": document["url"],
        "sample_num_pages": document["num_pages"],
        "pdf_bytes": len(document["pdf"]),
    }
    for op in Op:
        reply = worker.call(op, document["pdf"])
        for key, value in reply.result.items():
            row[f"{op}_{key}"] = value
        row[f"{op}_worker_lost"] = reply.worker_lost
    return row


def run_probe(documents: pl.DataFrame) -> list[dict]:
    worker = Worker()
    rows = []
    try:
        for index, document in enumerate(documents.iter_rows(named=True)):
            rows.append(probe_row(worker, document))
            if (index + 1) % 100 == 0:
                logger.info("probe: %d/%d documents, %d worker spawns", index + 1, documents.height, worker.spawns)
    finally:
        worker.stop()
    logger.info("probe: finished %d documents with %d worker spawns", len(rows), worker.spawns)
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def _log_timings(label: str, per_document: list[float], per_page: list[float], pages: int) -> None:
    logger.info(
        "%s: per-document ms p50=%.2f p90=%.2f p99=%.2f max=%.2f | per-page ms p50=%.2f p90=%.2f p99=%.2f | "
        "aggregate %.2f ms/page over %d pages",
        label,
        _percentile(per_document, 0.50),
        _percentile(per_document, 0.90),
        _percentile(per_document, 0.99),
        max(per_document, default=float("nan")),
        _percentile(per_page, 0.50),
        _percentile(per_page, 0.90),
        _percentile(per_page, 0.99),
        sum(per_document) / pages if pages else float("nan"),
        pages,
    )


def summarize(rows: list[dict]) -> None:
    """Log the failure taxonomy and the latency distributions, per operation.

    Per-page figures come from the page count the library itself reported, so a document it refused
    to open contributes to the failure rate and to nothing else.
    """
    logger.info("probe: %d documents on %s (%s)", len(rows), platform.machine(), platform.processor() or "unknown cpu")
    for op in Op:
        outcomes = Counter(row[f"{op}_outcome"] for row in rows)
        logger.info("--- %s ---", op)
        for outcome, count in outcomes.most_common():
            logger.info("%s: %-12s %5d  %6.2f%%", op, outcome, count, 100 * count / len(rows))

        fatal = [row for row in rows if row[f"{op}_outcome"] == Outcome.WORKER_DIED]
        for signal_name, count in Counter(row.get(f"{op}_worker_signal") for row in fatal).most_common():
            logger.info("%s: worker deaths by %s: %d", op, signal_name, count)
        for row in [row for row in rows if row[f"{op}_outcome"] not in SURVIVED_OUTCOMES][:10]:
            logger.info("%s: fatal on %s (%d bytes): %s", op, row["url"], row["pdf_bytes"], row.get(f"{op}_error"))
        for message, count in Counter(
            row.get(f"{op}_error") for row in rows if row[f"{op}_outcome"] == Outcome.EXCEPTION
        ).most_common(10):
            logger.info("%s: exception x%d: %s", op, count, message)

        good = [row for row in rows if row[f"{op}_outcome"] == Outcome.OK]
        if not good:
            logger.warning("%s: no successful calls to time", op)
            continue
        per_document = [row[f"{op}_milliseconds"] for row in good]
        paged = [row for row in good if (row.get(f"{op}_page_count") or 0) > 0]
        per_page = [row[f"{op}_milliseconds"] / row[f"{op}_page_count"] for row in paged]
        _log_timings(str(op), per_document, per_page, sum(row[f"{op}_page_count"] for row in paged))


def main() -> None:
    configure_logging(logging.INFO)
    logger.info("pdf-inspector %s on %s, python %s", version("pdf-inspector"), platform.machine(), sys.version)
    fs = storage()
    rows = run_probe(probe_documents(fs))
    summarize(rows)

    output = f"{OUTPUT_PREFIX}/{platform.machine()}.parquet"
    # Every row carries the columns its own outcome produced, so a failure class that first appears
    # late in the run would be dropped entirely under Polars' default 100-row schema inference --
    # silently losing exactly the rare rows this probe exists to find.
    frame = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    with fs.open(output, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", compression_level=1)
    logger.info("probe: wrote %d rows -> %s", len(rows), output)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
