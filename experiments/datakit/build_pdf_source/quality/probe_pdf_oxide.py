# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does ``pdf_oxide`` belong in the router pass, the VLM render feed, or neither?

``pdf-oxide`` is a Rust PDF toolkit (``tiny-skia`` rasteriser, no PDFium, no MuPDF, MIT OR
Apache-2.0) that publishes a text/Markdown API *and* a from-scratch page renderer. Marin's PDF
pipeline is CPU-constrained rather than GPU-constrained, so the two PyMuPDF passes that survive
Docling's removal are the ones worth attacking:

======================================  ======================
route                                   CPU core-h / M pages
======================================  ======================
pdf-inspector 1.17.0 (full extract)     2.1
router features (route + ocr features)  3.4
VLM feed (render + PNG + base64)        17.8
======================================  ======================

Those are separate questions and this module measures them separately.

**Router features.** :mod:`~experiments.datakit.build_pdf_source.quality.route_features` needs ~70
signals in six groups, and its discriminative power is in the ones that are not text facts: ToUnicode
coverage, Type3 and glyphless fonts, render-mode-3 text over a bitmap, ruling lines, and
content-stream order against a column-aware order. A library that supplies five groups and not the
sixth saves nothing, because the cost is dominated by opening and parsing the document -- so the
only question that moves the cost model is whether the PyMuPDF pass can be *removed*, and
:data:`Op.ROUTE_OXIDE` prices the subset pdf_oxide can actually answer against
:data:`Op.ROUTE_PYMUPDF` on the same documents.

**Render feed.** :mod:`~experiments.datakit.build_pdf_source.ocr_extract.render` renders each page to
a PNG data URI under a 2048 visual-token budget with a 300-DPI upscale cap. Both render ops here
reproduce that budget and split their time three ways -- rasterise, PNG-encode, base64 -- because
PNG encoding is plausibly a large share of the 17.8 and could be swapped without touching the
rasteriser.

The two renderers cannot agree by construction, and :data:`Op.PIXELS` measures how far apart they
are. Marin scales a page with a *non-uniform* matrix straight onto ``smart_resize`` dimensions
(each side a multiple of 32, filling the token budget exactly); pdf_oxide's ``render_page`` and
``render_pixmap`` take a scalar DPI, and ``render_page_fit`` fits a box while preserving aspect
ratio. Neither can hit an arbitrary aligned pair. So :data:`Op.PIXELS` renders both at the same
integer DPI -- where the dimensions *do* agree -- and compares the pixels, which isolates renderer
fidelity from the sizing policy. A faster renderer that produces different pixels changes VLM output
and invalidates every quality number the pipeline has, so this is a gate, not a footnote.

Crawl PDFs are adversarial and a native extension fails in ways a Python library cannot: a panic
compiled under ``panic = "abort"`` takes the interpreter with it, an unbounded allocation gets the
process OOM-killed, and a pathological content stream can simply not return. None of those are
catchable in the calling process, so this module reuses
:class:`~experiments.datakit.build_pdf_source.quality.probe_pdf_inspector.Worker` -- one document at
a time over a length-prefixed pipe to a subprocess the driver is willing to lose -- and reports
``exception``/``panic``/``memory``/``worker_died``/``timeout`` as a first-class result. pdf-inspector's
clean record does not transfer to a different library.

The probe is deliberately single-task: the deliverable is a latency distribution, and co-tenant
tasks on the same node would measure scheduler contention instead. It reserves ``--cpu 8`` so
nothing of its own competes with the process being timed. Note that pdf_oxide's PyO3 bindings never
release the GIL (no ``allow_threads`` on any read path) and its ``rayon`` dependency is behind a
non-default ``parallel`` feature, so unlike pdf-inspector these figures are single-core figures and
do divide by worker count.

Run it once per architecture. ``cw-us-east-02a`` is H100/x86_64; a CPU-only task on
``cw-us-east-08a`` lands on the Grace pool and is aarch64:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-oxide-probe-x86 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdf_oxide

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdf-oxide-probe-aarch64 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdf_oxide

``--extra pdf`` is required: both libraries live in marin-core's ``pdf`` extra, and the worker venv
is built from the repository's declared dependencies plus the extras passed here, never from a
local one.
"""

import base64
import json
import logging
import os
import platform
import random
import sys
import time
from collections import Counter
from enum import StrEnum
from importlib.metadata import version

import numpy as np
import polars as pl

from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderOptions,
    effective_dpi,
    target_dimensions,
)
from experiments.datakit.build_pdf_source.ocr_features import sample_page_indices
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import (
    ERROR_SAMPLES,
    SURVIVED_OUTCOMES,
    Outcome,
    Worker,
    failure_outcome,
    probe_documents,
    read_exactly,
    storage,
)

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_pdf_oxide_probe"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.probe_pdf_oxide"
WORKER_FLAG = "--worker"

# Pages rendered per document. The feed renders every page, but a per-page cost only needs enough
# pages to be representative, and an uncapped render of the tail of the page distribution would
# dominate the run. Matched to the router's sample size so both halves report over the same scale.
RENDER_PAGES = 8
# Pages compared pixel for pixel. Two renderers at the same DPI produce two full-resolution buffers
# per page and the comparison is the expensive part; two pages per document over the whole sample is
# already thousands of pages.
PIXEL_PAGES = 2
# A pixel channel differing by more than this counts as changed. Below it the difference is
# anti-aliasing and colour rounding, which a VLM at ~146 DPI cannot see.
PIXEL_TOLERANCE = 16
# Fraction of changed pixels above which a page is "visibly different" -- the two renderers drew
# something structurally unlike, not the same page with softer edges.
PAGE_DIVERGENCE_FRACTION = 0.02

RENDER_OPTIONS = RenderOptions()


class Op(StrEnum):
    """One measured route. Each is its own worker call so a library that dies is attributable."""

    ROUTE_PYMUPDF = "route_pymupdf"
    ROUTE_OXIDE = "route_oxide"
    RENDER_PYMUPDF = "render_pymupdf"
    RENDER_OXIDE = "render_oxide"
    PIXELS = "pixels"


# ---------------------------------------------------------------------------
# Sizing: the one budget both renderers are held to
# ---------------------------------------------------------------------------


def matched_dpi(width: float, height: float) -> int:
    """The integer DPI closest to what the token budget asks of this page.

    Marin's production path scales a page onto ``smart_resize`` dimensions with a non-uniform
    matrix, which no scalar-DPI API can express. This is the nearest thing pdf_oxide can be asked
    for, and the dimensions it yields are recorded next to the ones production wanted so the gap is
    measured rather than assumed away.
    """
    height_pixels, width_pixels = target_dimensions(width, height, RENDER_OPTIONS)
    return max(1, round(effective_dpi(height_pixels * width_pixels, width, height)))


# ---------------------------------------------------------------------------
# Worker: the measured calls
# ---------------------------------------------------------------------------


def _route_pymupdf(payload: bytes) -> dict:
    """The incumbent router pass: one ``rawdict`` walk plus fonts, drawings and image placements."""
    import pymupdf  # noqa: PLC0415

    from experiments.datakit.build_pdf_source.quality.route_features import page_signals  # noqa: PLC0415

    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        indices = sample_page_indices(len(document), random.Random(0))
        pages = 0
        for index in indices:
            page_signals(document, document.load_page(index))
            pages += 1
        return {"page_count": pages, "document_pages": len(document)}
    finally:
        document.close()


def _oxide_page_signals(document, index: int) -> None:
    """Everything pdf_oxide can offer the router for one page, and nothing it cannot.

    Deliberately the *whole* set of calls the derivable feature groups need, not the cheapest one:
    the question is what a pdf_oxide router would cost, and a router built on ``classify_page``
    alone could not fill the encoding, structure or order groups at all.
    """
    # encoding / math / script / layer / order: per-span text, geometry, font name, the ISO 32000-1
    # §9.10.2 mapping tier the font offered, and content-stream emission order.
    for span in document.extract_spans(index):
        _ = (span.text, span.bbox, span.font_name, span.provenance, span.sequence)
    # order: the column-aware reading order the stream order is scored against.
    document.extract_spans(index, reading_order="column_aware")
    # layer: invisible-text and garble ratios, image area, vector-path density.
    json.loads(document.classify_page(index))
    # structure: ruling lines and the grid they imply.
    document.extract_lines(index)
    document.extract_rects(index)
    # layer: where images are drawn, for invisible-text-over-bitmap.
    document.page_images(index)


def _route_oxide(payload: bytes) -> dict:
    """The pdf_oxide equivalent over the same sampled pages."""
    from pdf_oxide import PdfDocument  # noqa: PLC0415

    document = PdfDocument.from_bytes(payload)
    total = int(document.page_count())
    indices = sample_page_indices(total, random.Random(0))
    pages = 0
    for index in indices:
        _oxide_page_signals(document, index)
        pages += 1
    return {"page_count": pages, "document_pages": total}


def _render_pymupdf(payload: bytes) -> dict:
    """Production's feed path, split into rasterise / PNG-encode / base64."""
    import pymupdf  # noqa: PLC0415

    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        rasterise = encode = encode64 = 0.0
        pages = 0
        pixels = 0
        for index in range(min(len(document), RENDER_PAGES)):
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)
            matrix = pymupdf.Matrix(width / page_width, height / page_height)

            started = time.perf_counter()
            pixmap = page.get_pixmap(matrix=matrix)
            middle = time.perf_counter()
            png = pixmap.tobytes("png")
            late = time.perf_counter()
            base64.b64encode(png).decode()
            done = time.perf_counter()

            rasterise += middle - started
            encode += late - middle
            encode64 += done - late
            pixels += height * width
            pages += 1
        return {
            "page_count": pages,
            "document_pages": len(document),
            "rasterise_ms": 1000 * rasterise,
            "encode_ms": 1000 * encode,
            "base64_ms": 1000 * encode64,
            "feed_ms": 1000 * (rasterise + encode + encode64),
            "megapixels": pixels / 1e6,
        }
    finally:
        document.close()


def _page_sizes(document, limit: int) -> list[tuple[int, float, float]]:
    """``(index, width, height)`` for the pages worth rendering, in points."""
    sizes = []
    for index in range(min(int(document.page_count()), limit)):
        left, bottom, right, top = document.page_media_box(index)
        width, height = abs(right - left), abs(top - bottom)
        if width >= 1 and height >= 1:
            sizes.append((index, width, height))
    return sizes


def _render_oxide(payload: bytes) -> dict:
    """pdf_oxide's feed path at the nearest scalar DPI, split the same three ways.

    ``render_pixmap`` and ``render_page`` are separate entry points rather than one call whose
    result is encoded, so the encoder's share can only be had by difference. Taking that difference
    within one document measures nothing: ``PdfDocument`` caches parsed page content, so whichever
    call runs second runs warm, and an early revision of this probe had ``render_page`` come out
    *faster* than the bare ``render_pixmap`` it supposedly contains. The two passes therefore get
    two freshly parsed documents, so both are cold, and each is charged its own parse.

    ``feed_ms`` -- rasterise, PNG-encode and base64, which is what the feed actually pays -- is the
    figure the cost model uses. The op's wall time covers both passes and would double-count.
    """
    from pdf_oxide import PdfDocument  # noqa: PLC0415

    encoded = PdfDocument.from_bytes(payload)
    sizes = _page_sizes(encoded, RENDER_PAGES)
    whole = encode64 = 0.0
    pixels = 0
    dimension_matches = 0
    for index, page_width, page_height in sizes:
        dpi = matched_dpi(page_width, page_height)
        wanted_height, wanted_width = target_dimensions(page_width, page_height, RENDER_OPTIONS)

        started = time.perf_counter()
        png = encoded.render_page(index, dpi=dpi, format="png")
        middle = time.perf_counter()
        base64.b64encode(png).decode()
        done = time.perf_counter()

        whole += middle - started
        encode64 += done - middle
        rendered_width, rendered_height = round(page_width * dpi / 72.0), round(page_height * dpi / 72.0)
        pixels += rendered_width * rendered_height
        dimension_matches += rendered_height == wanted_height and rendered_width == wanted_width

    raw = PdfDocument.from_bytes(payload)
    rasterise = 0.0
    for index, page_width, page_height in sizes:
        started = time.perf_counter()
        raw.render_pixmap(index, matched_dpi(page_width, page_height))
        rasterise += time.perf_counter() - started

    return {
        "page_count": len(sizes),
        "document_pages": int(encoded.page_count()),
        "rasterise_ms": 1000 * rasterise,
        # `render_page` rasterises and encodes; the encoder's share is what is left over.
        "encode_ms": 1000 * max(whole - rasterise, 0.0),
        "base64_ms": 1000 * encode64,
        "feed_ms": 1000 * (whole + encode64),
        "megapixels": pixels / 1e6,
        "dimension_matches": dimension_matches,
    }


def _flatten_premultiplied(pixmap) -> np.ndarray:
    """pdf_oxide's premultiplied RGBA composited onto white, as PyMuPDF's RGB pixmap already is."""
    rgba = np.frombuffer(pixmap.data, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 4)
    # Premultiplied source over an opaque white backdrop: C = Cs + (1 - As) * 255.
    return np.clip(rgba[:, :, :3].astype(np.int16) + (255 - rgba[:, :, 3:4].astype(np.int16)), 0, 255).astype(np.uint8)


def _pixels(payload: bytes) -> dict:
    """Both renderers on the same pages at the same DPI, compared pixel for pixel.

    The DPI is shared so the dimensions agree and the comparison is about the renderers rather than
    about the sizing policy. Where they still disagree by a row or column of rounding the overlap is
    compared and the mismatch is reported.
    """
    import pymupdf  # noqa: PLC0415
    from pdf_oxide import PdfDocument  # noqa: PLC0415

    oxide = PdfDocument.from_bytes(payload)
    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        compared = 0
        exact_dimensions = 0
        differing = []
        mean_absolute = []
        divergent = 0
        for index in range(min(len(document), int(oxide.page_count()), PIXEL_PAGES)):
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            dpi = matched_dpi(page_width, page_height)
            scale = dpi / 72.0

            reference = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            left = np.frombuffer(reference.samples, dtype=np.uint8).reshape(reference.height, reference.width, 3)
            right = _flatten_premultiplied(oxide.render_pixmap(index, dpi))

            exact_dimensions += left.shape == right.shape
            rows, columns = min(left.shape[0], right.shape[0]), min(left.shape[1], right.shape[1])
            if rows == 0 or columns == 0:
                continue
            difference = np.abs(left[:rows, :columns].astype(np.int16) - right[:rows, :columns].astype(np.int16)).max(
                axis=2
            )
            changed = float((difference > PIXEL_TOLERANCE).mean())
            differing.append(changed)
            mean_absolute.append(float(difference.mean()))
            divergent += changed > PAGE_DIVERGENCE_FRACTION
            compared += 1
        if not compared:
            return {"page_count": 0, "document_pages": len(document)}
        return {
            "page_count": compared,
            "document_pages": len(document),
            "exact_dimensions": exact_dimensions,
            "changed_fraction": float(np.mean(differing)),
            "mean_absolute_difference": float(np.mean(mean_absolute)),
            "divergent_pages": divergent,
        }
    finally:
        document.close()


_OPERATIONS = {
    Op.ROUTE_PYMUPDF: _route_pymupdf,
    Op.ROUTE_OXIDE: _route_oxide,
    Op.RENDER_PYMUPDF: _render_pymupdf,
    Op.RENDER_OXIDE: _render_oxide,
    Op.PIXELS: _pixels,
}


def _measure(op: str, payload: bytes) -> dict:
    """Time one call, reporting whatever it did instead of raising."""
    started = time.perf_counter()
    try:
        observed = _OPERATIONS[Op(op)](payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    # BaseException, not Exception: PyO3 derives PanicException from the former, and a panic
    # reported as a worker death would invert this probe's central conclusion.
    except BaseException as error:
        return {
            "outcome": str(failure_outcome(error)),
            "milliseconds": 1000 * (time.perf_counter() - started),
            "error": f"{type(error).__name__}: {error}"[:500],
        }
    return {"outcome": str(Outcome.OK), "milliseconds": 1000 * (time.perf_counter() - started), **observed}


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    The reply channel is a *duplicate* of the inherited stdout, and file descriptor 1 is then
    pointed at stderr. Both libraries write diagnostics to fd 1 -- MuPDF sends its "syntax error in
    content stream" warnings there by default, and pdf_oxide's Rust logger does the same -- which
    lands them in the middle of the length-prefixed protocol and makes the driver read a warning
    where a JSON reply should be. Moving the descriptor rather than configuring each library's
    logger keeps this correct for whatever either of them prints next.
    """
    import faulthandler  # noqa: PLC0415

    faulthandler.enable()

    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    print(
        f"worker: pdf-oxide {version('pdf-oxide')}, pymupdf {version('pymupdf')}",
        file=sys.stderr,
    )
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = read_exactly(stdin, request["size"])
        replies.write(json.dumps(_measure(request["op"], payload)).encode() + b"\n")
        replies.flush()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def probe_row(worker: Worker, document: dict) -> dict:
    """Every operation against one document, flattened into one row keyed by operation."""
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
    worker = Worker(MODULE_NAME)
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


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    return tuple(float(x) for x in np.percentile(array, [50, 90, 99]))


def _core_hours_per_million(milliseconds_per_page: float) -> float:
    """Single-core CPU hours to process a million pages at this per-page cost."""
    return 1e6 * milliseconds_per_page / 1000.0 / 3600.0


def summarize(rows: list[dict]) -> None:
    """The failure taxonomy, the latency distributions, and the cost each implies."""
    logger.info("probe: %d documents on %s", len(rows), platform.machine())
    for op in Op:
        outcomes = Counter(row[f"{op}_outcome"] for row in rows)
        logger.info("--- %s ---", op)
        for outcome, count in outcomes.most_common():
            logger.info("%s: %-12s %5d  %6.2f%%", op, outcome, count, 100 * count / len(rows))

        fatal = [row for row in rows if row[f"{op}_outcome"] == Outcome.WORKER_DIED]
        for signal_name, count in Counter(row.get(f"{op}_worker_signal") for row in fatal).most_common():
            logger.info("%s: worker deaths by %s: %d", op, signal_name, count)
        for row in [row for row in rows if row[f"{op}_outcome"] not in SURVIVED_OUTCOMES][:ERROR_SAMPLES]:
            logger.info("%s: fatal on %s (%d bytes): %s", op, row["url"], row["pdf_bytes"], row.get(f"{op}_error"))
        for outcome in (Outcome.PANIC, Outcome.MEMORY, Outcome.EXCEPTION):
            failed = [row for row in rows if row[f"{op}_outcome"] == outcome]
            for message, count in Counter(row.get(f"{op}_error") for row in failed).most_common(ERROR_SAMPLES):
                logger.info("%s: %s x%d: %s", op, outcome, count, message)

        good = [row for row in rows if row[f"{op}_outcome"] == Outcome.OK]
        if not good:
            logger.warning("%s: no successful calls to time", op)
            continue

        # The render ops run two passes over the document so the PNG encoder can be priced on its
        # own, and their wall time therefore double-counts. `feed_ms` is what production pays.
        cost = f"{op}_feed_ms" if any(row.get(f"{op}_feed_ms") is not None for row in good) else f"{op}_milliseconds"
        per_document = [row[cost] for row in good]
        paged = [row for row in good if (row.get(f"{op}_page_count") or 0) > 0]
        pages = sum(row[f"{op}_page_count"] for row in paged)
        per_page = [row[cost] / row[f"{op}_page_count"] for row in paged]
        document_p50, document_p90, document_p99 = _percentiles(per_document)
        page_p50, page_p90, page_p99 = _percentiles(per_page)
        aggregate = sum(row[cost] for row in paged) / pages if pages else float("nan")
        logger.info(
            "%s: per-document ms p50=%.2f p90=%.2f p99=%.2f max=%.2f | per-page ms p50=%.2f p90=%.2f p99=%.2f | "
            "aggregate %.3f ms/page over %d pages = %.1f core-h / M pages",
            op,
            document_p50,
            document_p90,
            document_p99,
            max(per_document, default=float("nan")),
            page_p50,
            page_p90,
            page_p99,
            aggregate,
            pages,
            _core_hours_per_million(aggregate),
        )

        for stage in ("rasterise_ms", "encode_ms", "base64_ms"):
            timed = [row for row in paged if row.get(f"{op}_{stage}") is not None]
            if not timed:
                continue
            total = sum(row[f"{op}_{stage}"] for row in timed)
            stage_pages = sum(row[f"{op}_page_count"] for row in timed)
            logger.info(
                "%s: %-13s %.3f ms/page = %.1f core-h / M pages (%.1f%% of the feed)",
                op,
                stage,
                total / stage_pages,
                _core_hours_per_million(total / stage_pages),
                100 * total / sum(row[cost] for row in timed),
            )

        megapixels = [row.get(f"{op}_megapixels") for row in paged if row.get(f"{op}_megapixels") is not None]
        if megapixels:
            logger.info("%s: %.3f MP/page rendered", op, sum(megapixels) / pages)
        matches = [row.get(f"{op}_dimension_matches") for row in paged if row.get(f"{op}_dimension_matches") is not None]
        if matches:
            logger.info(
                "%s: pages hitting production's smart_resize dimensions: %d / %d (%.2f%%)",
                op,
                sum(matches),
                pages,
                100 * sum(matches) / pages,
            )

    _summarize_pixels(rows)


def _summarize_pixels(rows: list[dict]) -> None:
    """Whether the two rasterisers agree, which is the gate on replacing the feed."""
    good = [
        row
        for row in rows
        if row[f"{Op.PIXELS}_outcome"] == Outcome.OK and (row.get(f"{Op.PIXELS}_page_count") or 0) > 0
    ]
    if not good:
        logger.warning("pixels: no pages compared")
        return
    pages = sum(row[f"{Op.PIXELS}_page_count"] for row in good)
    exact = sum(row[f"{Op.PIXELS}_exact_dimensions"] for row in good)
    divergent = sum(row[f"{Op.PIXELS}_divergent_pages"] for row in good)
    changed = [row[f"{Op.PIXELS}_changed_fraction"] for row in good]
    absolute = [row[f"{Op.PIXELS}_mean_absolute_difference"] for row in good]
    changed_p50, changed_p90, changed_p99 = _percentiles(changed)
    logger.info(
        "pixels: %d pages over %d documents at matched DPI | dimensions identical %d (%.2f%%) | "
        "changed-pixel fraction p50=%.4f p90=%.4f p99=%.4f | mean |delta| %.2f/255 | "
        "pages over %.0f%% changed: %d (%.2f%%)",
        pages,
        len(good),
        exact,
        100 * exact / pages,
        changed_p50,
        changed_p90,
        changed_p99,
        float(np.mean(absolute)),
        100 * PAGE_DIVERGENCE_FRACTION,
        divergent,
        100 * divergent / pages,
    )


def main() -> None:
    from rigging.log_setup import configure_logging  # noqa: PLC0415

    configure_logging(logging.INFO)
    oxide = version("pdf-oxide")
    logger.info("pdf-oxide %s, pymupdf %s, on %s, python %s", oxide, version("pymupdf"), platform.machine(), sys.version)

    filesystem = storage()
    documents = probe_documents(filesystem)
    rows = run_probe(documents)

    output = f"{OUTPUT_PREFIX}/{platform.machine()}-{oxide}.parquet"
    # Every row carries the columns its own outcome produced, so a failure class that first appears
    # late in the run would be dropped entirely under Polars' default 100-row schema inference --
    # silently losing exactly the rare rows this probe exists to find.
    frame = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    with filesystem.open(output, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", compression_level=1)
    logger.info("probe: wrote %d rows -> %s", len(rows), output)

    summarize(rows)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
