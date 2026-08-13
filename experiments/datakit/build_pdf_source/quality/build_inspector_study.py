# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the routing study with a third extraction route: the pdf-inspector Rust extractor.

One Zephyr map over the same 100k oracle sample
(:mod:`~experiments.datakit.build_pdf_source.quality.build_oracle_sample`) that
:mod:`~experiments.datakit.build_pdf_source.quality.build_route_study` reads, emitting one narrow
row per document that joins to that study table on ``source_id``. Stage 0
(:mod:`~experiments.datakit.build_pdf_source.quality.probe_pdf_inspector`) established that the
library survives crawl PDFs at ~4.6-4.9 ms per page against Docling's ~1000 ms; this module asks the
question that cost buys, which is whether what it reads is *the same document*.

Four things per document:

* **the classification signals**, from ``detect_pdf_bytes`` -- the candidate routing features for
  Stage 2. Deliberately not ``classify_pdf_bytes``, which Stage 0 timed: that entry point returns
  four of the nine signals and does not expose ``has_encoding_issues``, ``is_complex_layout``,
  ``pages_with_tables`` or ``pages_with_columns``, so it is not the call a router would make. The
  extraction reports its own ``is_complex_layout`` and table/column pages from a deeper read of the
  page, and those disagree with detect's often enough to be worth carrying separately.
* **agreement against the VLM**, computed with
  :mod:`~experiments.datakit.build_pdf_source.quality.route_agreement` exactly as the
  Docling-versus-VLM numbers were, so the two routes are directly comparable.
* **agreement against Docling**, the mutual cheap-route signal: two cheap routes that fail the same
  documents cannot cover for each other, and Stage 3 needs to know whether they do.
* **the failure taxonomy as columns**, one outcome per document with its error text. A native
  extension's failure rate is a result, not a log line.

**Docling-versus-VLM is recomputed here rather than joined from the study table.** The normalizer
gained two rules for this pass (link targets, and comment placeholders -- see
:mod:`route_agreement`), so the stored Docling numbers were computed under a slightly different
metric than the pdf-inspector numbers would be. A headline comparison across two normalizers is not
a comparison, and the recomputation is one more alignment over pages already in memory.

**The library runs in a subprocess pool of one, reused across the whole shard.** Stage 0 measured
zero panics and zero worker deaths in 1,000 documents, but three unbounded-depth recursions over
nested Form XObjects remain in the crate, and a stack overflow is a ``SIGSEGV`` rather than a
catchable panic. Isolation therefore stays; a process per document would not, at ~90 ms of work
against ~30 ms of interpreter start. The driver is
:class:`~experiments.datakit.build_pdf_source.quality.probe_pdf_inspector.Worker`, which replaces
the child whenever it fails to answer.

No ``RLIMIT_AS`` is set, for the reason Stage 0 records: ``lopdf`` is built against ``rayon``, and
an address-space rlimit stops rayon's global thread pool from spawning, which turns every call into
a misleading ``PanicException``. The container cgroup is the memory bound instead. ``RAYON_NUM_THREADS``
*is* set, from the task's own CPU allotment: the library is internally parallel and does not release
the GIL, so an unpinned thread pool sized to the whole node would have each of a worker's co-tenant
tasks contending for every core and would make the reported timings a measurement of the scheduler.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdf-inspector-study --extra pdf \\
        --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_inspector_study

``--extra pdf`` is required: the library lives in marin-core's ``pdf`` extra, and a worker venv is
built from the repository's declared dependencies plus the extras passed here.
"""

import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

import fsspec
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.quality import route_agreement
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import (
    WORKER_FLAG,
    Outcome,
    Worker,
    failure_outcome,
    read_exactly,
)

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
SAMPLE_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_sample100k"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_inspector_study"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_inspector_study"
STUDY_OP = "study"

# Both library calls for one document, so a document costs one round trip rather than two.
DETECT = "detect"
EXTRACT = "extract"

# The sample's own columns. ``source_id`` is the crawl-record key
# (``<warc_filename>:<warc_record_offset>``, see ``build_oracle_sample.record_key``) that the clean
# corpora carry, which is what makes this table joinable to the route study on the column name; the
# fetch artifact's ``source_id`` is a WARC UUID and is not this.
READ_COLUMNS = (
    "source_id",
    "url",
    "num_pages",
    "text",
    "page_offsets",
    "docling_text",
    "docling_page_offsets",
    "pdf",
)


@dataclass(frozen=True)
class Comparison:
    """One pairwise comparison: which two routes, and what its columns are called.

    Recall is the share of the *reference* route's tokens the candidate also produced, so the two
    are not interchangeable: ``inspector_vlm`` asks what pdf-inspector lost against the VLM, and
    naming the prefix candidate-first keeps that direction legible in the column name.
    """

    prefix: str
    reference: str
    candidate: str

    @property
    def routes(self) -> tuple[route_agreement.Route, route_agreement.Route]:
        return ROUTES[self.reference], ROUTES[self.candidate]


ROUTES = {
    "vlm": route_agreement.VLM,
    "docling": route_agreement.DOCLING,
    "inspector": route_agreement.INSPECTOR,
}
COMPARISONS = (
    Comparison("inspector_vlm", reference="vlm", candidate="inspector"),
    Comparison("inspector_docling", reference="docling", candidate="inspector"),
    # Recomputed rather than joined from the route study, so every number in this table comes from
    # one normalizer and the headline comparison is between routes rather than between metrics.
    Comparison("docling_vlm", reference="vlm", candidate="docling"),
)
NULL_AGREEMENT = {
    comparison.prefix: dict.fromkeys(route_agreement.agreement_columns(*comparison.routes)) for comparison in COMPARISONS
}

# The classification signals detect reports as page-index lists, carried as counts: the routing
# question is how much of the document is affected, and a variable-length index list is not a
# feature. ``ocr_reasons_by_page`` is carried whole, as a reason-to-page-count histogram.
_PAGE_LIST_SIGNALS = ("pages_needing_ocr", "pages_with_tables", "pages_with_columns")

_TASK_RESOURCES = ResourceConfig(cpu=2, ram="12g", disk="8g")
_WORKER_RESOURCES = ResourceConfig(cpu=16, ram="96g", disk="64g")
# Explicit, and not the 1 GB default: the coordinator holds shard, retry and shuffle state for every
# task, and at this shard count the default is what dies at exit 137 one task short of the end.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)
# 178 input shards is the cross-task parallelism ceiling; eight tasks to a worker covers it with one
# slot to spare rather than leaving a shard queued behind a finished worker.
_MAX_WORKERS = 23
_HEARTBEAT_TIMEOUT = 30 * 60


def storage() -> fsspec.AbstractFileSystem:
    configure_coreweave_s3()
    return fsspec.filesystem("s3")


# ---------------------------------------------------------------------------
# Worker: the library, in a process the task is willing to lose
# ---------------------------------------------------------------------------


def _detected(result) -> dict:
    """Every classification signal detect reports, flattened."""
    reasons: Counter[str] = Counter()
    for page in result.ocr_reasons_by_page:
        reasons.update(page.reasons)
    return {
        "pdf_type": result.pdf_type,
        "confidence": float(result.confidence),
        "page_count": result.page_count,
        "has_encoding_issues": result.has_encoding_issues,
        "is_complex_layout": result.is_complex_layout,
        "pages_needing_ocr": len(result.pages_needing_ocr),
        "pages_with_tables": len(result.pages_with_tables),
        "pages_with_columns": len(result.pages_with_columns),
        "ocr_reasons": json.dumps(dict(sorted(reasons.items()))),
        "has_title": result.title is not None,
        "library_milliseconds": result.processing_time_ms,
    }


def _extracted(result) -> dict:
    """The extraction's own read of the same signals, plus the pages themselves."""
    return {
        "is_complex_layout": result.is_complex,
        "pages_needing_ocr": len(result.pages_needing_ocr),
        "pages_with_tables": len(result.pages_with_tables),
        "pages_with_columns": len(result.pages_with_columns),
        "pages": [page.markdown for page in result.pages],
    }


def _study(module, payload: bytes) -> dict:
    """Both library calls against one document, each timed on its own.

    Reports whatever the library did rather than raising, so a failure is a row rather than a
    casualty. ``BaseException`` and not ``Exception``: PyO3 derives ``PanicException`` from the
    former, and a panic reported as a worker death would invert Stage 0's central conclusion.
    """
    observed: dict = {}
    for name, call in ((DETECT, module.detect_pdf_bytes), (EXTRACT, module.extract_pages_markdown_bytes)):
        started = time.perf_counter()
        try:
            result = call(payload)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            return {
                **observed,
                "outcome": str(failure_outcome(error)),
                "failed_op": name,
                f"{name}_milliseconds": 1000 * (time.perf_counter() - started),
                "error": f"{type(error).__name__}: {error}"[:500],
            }
        observed[f"{name}_milliseconds"] = 1000 * (time.perf_counter() - started)
        observed[name] = _detected(result) if name == DETECT else _extracted(result)
    return {**observed, "outcome": str(Outcome.OK)}


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it."""
    import faulthandler  # noqa: PLC0415 - only the disposable process needs a fault handler

    import pdf_inspector  # noqa: PLC0415 - the whole point is to import it out of process

    faulthandler.enable()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = read_exactly(stdin, request["size"])
        stdout.write(json.dumps(_study(pdf_inspector, payload)).encode() + b"\n")
        stdout.flush()


# ---------------------------------------------------------------------------
# One document
# ---------------------------------------------------------------------------


def registered_domain(url: str | None) -> str:
    """The host of a URL, which is the unit near-duplicate documents cluster in.

    Stage 2's comparison has to be domain-disjoint -- the crawl holds ~9.8% exact-duplicate PDFs and
    many more near-duplicates from the same publisher -- and the study table carries only the URL.
    Same derivation as :func:`train_route_model.registered_domain`, resolved here so the split key
    travels with the row.
    """
    if not url:
        return ""
    return (urlparse(url).hostname or "").lower()


def _timings(reply: dict, page_count: int | None) -> dict:
    """Per-document and per-page milliseconds for each call the worker got to."""
    timed = {}
    for name in (DETECT, EXTRACT):
        milliseconds = reply.get(f"{name}_milliseconds")
        timed[f"inspector_{name}_milliseconds"] = milliseconds
        timed[f"inspector_{name}_ms_per_page"] = (
            milliseconds / page_count if milliseconds is not None and page_count else None
        )
    return timed


def _signals(reply: dict) -> dict:
    """The classification and extraction signal columns, null wherever the call never returned."""
    detected, extracted = reply.get(DETECT) or {}, reply.get(EXTRACT) or {}
    signals = {
        "inspector_pdf_type": detected.get("pdf_type"),
        "inspector_confidence": detected.get("confidence"),
        "inspector_page_count": detected.get("page_count"),
        "inspector_has_encoding_issues": detected.get("has_encoding_issues"),
        "inspector_ocr_reasons": detected.get("ocr_reasons"),
        "inspector_has_title": detected.get("has_title"),
        "inspector_library_milliseconds": detected.get("library_milliseconds"),
        "inspector_detect_is_complex_layout": detected.get("is_complex_layout"),
        "inspector_extract_is_complex_layout": extracted.get("is_complex_layout"),
        "inspector_extracted_pages": len(extracted["pages"]) if "pages" in extracted else None,
        "inspector_markdown_chars": sum(len(page) for page in extracted["pages"]) if "pages" in extracted else None,
    }
    for name in _PAGE_LIST_SIGNALS:
        signals[f"inspector_detect_{name}"] = detected.get(name)
        signals[f"inspector_extract_{name}"] = extracted.get(name)
    return signals


def _agreement(pages: dict[str, list[str] | None]) -> dict:
    """The three pairwise comparisons, each null when a route it needs produced nothing.

    A route that produced no pages at all is given one empty page rather than none, so it scores as
    total loss against whatever the other route read -- which is what losing the document is -- and
    the page-weighted mean has something to divide by.
    """
    result = {}
    for comparison in COMPARISONS:
        reference_pages, candidate_pages = pages[comparison.reference], pages[comparison.candidate]
        measured = (
            NULL_AGREEMENT[comparison.prefix]
            if reference_pages is None or candidate_pages is None
            else route_agreement.pages_agreement(reference_pages, candidate_pages, *comparison.routes)
        )
        result.update({f"{comparison.prefix}_{column}": value for column, value in measured.items()})
    return result


def study_row(row: dict, reply: dict, worker_lost: bool) -> dict:
    """Everything this study knows about one document."""
    extracted = reply.get(EXTRACT)
    pages: dict[str, list[str] | None] = {
        "vlm": route_agreement.split_pages(row["text"], row["page_offsets"]),
        "docling": (
            None
            if row["docling_text"] is None
            else route_agreement.split_pages(row["docling_text"], row["docling_page_offsets"])
        ),
        # An extraction that returned no pages at all still made a claim about the document, and it
        # is a claim of total loss; only a call that never returned has nothing to compare.
        "inspector": None if extracted is None else extracted["pages"] or [""],
    }

    output = {
        "source_id": row["source_id"],
        "url": row["url"],
        "domain": registered_domain(row["url"]),
        "num_pages": row["num_pages"],
        "pdf_bytes": len(row["pdf"]),
        "docling_missing": row["docling_text"] is None,
        "inspector_outcome": reply["outcome"],
        "inspector_error": reply.get("error"),
        "inspector_failed_op": reply.get("failed_op"),
        "inspector_worker_signal": reply.get("worker_signal"),
        "inspector_worker_exit_code": reply.get("worker_exit_code"),
        "inspector_worker_lost": worker_lost,
    }
    output.update(_signals(reply))
    output.update(_timings(reply, (reply.get(DETECT) or {}).get("page_count")))
    output.update(_agreement(pages))
    return output


# ---------------------------------------------------------------------------
# One shard
# ---------------------------------------------------------------------------


def study_shard(work: tuple[int, str]) -> int:
    """Emit the study rows for one sample shard, or skip it if its output already exists.

    One worker subprocess serves the whole shard and is replaced only when it dies, so the
    isolation costs one interpreter start per shard rather than one per document.
    """
    index, shard = work
    fs = storage()
    output = f"{OUTPUT_PREFIX}/part-{index:05d}.parquet"
    if fs.exists(output):
        return 0

    with fs.open(shard, "rb") as stream:
        table = pl.read_parquet(stream, columns=list(READ_COLUMNS))

    os.environ["RAYON_NUM_THREADS"] = str(int(_TASK_RESOURCES.cpu))
    worker = Worker(MODULE_NAME)
    rows = []
    try:
        for document in table.iter_rows(named=True):
            reply = worker.call(STUDY_OP, document["pdf"])
            rows.append(study_row(document, reply.result, reply.worker_lost))
    finally:
        worker.stop()

    outcomes = Counter(row["inspector_outcome"] for row in rows)
    for outcome, count in outcomes.items():
        counters.pipeline.update_counter(f"inspector_study/{outcome}", count)
    counters.pipeline.update_counter("inspector_study/documents", len(rows))
    counters.pipeline.update_counter("inspector_study/worker_spawns", worker.spawns)
    logger.info("shard %d: %d rows, %d worker spawns, outcomes %s", index, len(rows), worker.spawns, dict(outcomes))

    # Every row carries the columns its own outcome produced, so a failure class that first appears
    # late in a shard would be dropped entirely under Polars' default 100-row schema inference.
    frame = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    with fs.open(output, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", compression_level=1)
    return len(rows)


def shards() -> list[tuple[int, str]]:
    paths = sorted(str(path) for path in StoragePath(f"{SAMPLE_PREFIX}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"no sample shards under {SAMPLE_PREFIX}")
    return list(enumerate(paths))


def main() -> None:
    configure_logging(logging.INFO)
    storage()
    work = shards()
    logger.info("inspector study: %d shards -> %s", len(work), OUTPUT_PREFIX)

    outcome = ZephyrContext(
        name="pdf-inspector-study",
        resources=_WORKER_RESOURCES,
        coordinator_resources=_COORDINATOR_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(
        Dataset.from_list(work).map(study_shard),
        map_task_resources=_TASK_RESOURCES,
    )
    logger.info("inspector study: done, counters %s", dict(outcome.counters))


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
