# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Can PDFium replace MuPDF as the VLM feed's rasteriser?

The motivation is licensing, not speed. PyMuPDF is AGPL; PDFium is BSD-3-Clause and ``pypdfium2``
is Apache-2.0/BSD. This corpus and the toolchain that builds it are intended for release, and after
the router pass and the Docling route are removed, page rendering is the *only* thing PyMuPDF is
still doing here. Removing it removes the last AGPL component from the runtime path.

Be honest about the size of the speed prize: the feed costs 20.22 CPU core-hours per million pages
on x86, and :mod:`~experiments.datakit.build_pdf_source.quality.probe_png_encoders` already showed
that ~64% of that is PNG encoding, not rasterisation. Swapping the rasteriser addresses about a
third of the cost at most. So every render op here encodes with **Pillow at compress_level=1** --
the encoder that measurement chose -- and the two rasterisers are compared with the encoder held
constant. MuPDF's own PNG encoder is timed alongside, unused by ``feed_ms``, only so the numbers
tie back to the incumbent table.

**The gate is not pixel-identity.** Two rasterisers draw glyphs differently and always will;
:mod:`~experiments.datakit.build_pdf_source.quality.probe_pdf_oxide` measured that for tiny-skia and
it is measured again here for PDFium. What that probe could not do was ask whether the difference
*matters*, because pdf_oxide was also 1.7-1.8x slower and could not hit production's dimensions, so
it failed on cost before fidelity was worth asking about. PDFium clears both of those, so the real
question moves to :mod:`~experiments.datakit.build_pdf_source.quality.build_render_study`, which
puts both renderings through the VLM and compares the extracted text. This module measures cost,
failure behaviour, and the pixel difference as *characterisation* for that study to interpret.

**PDFium can hit production's dimensions exactly**, which is what makes the comparison meaningful at
all. Marin scales a page with a non-uniform matrix straight onto ``smart_resize`` dimensions -- each
side a multiple of 32, filling the 2048-visual-token budget. ``pypdfium2``'s ``PdfPage.render`` takes
a scalar ``scale`` and cannot express that, but the underlying ``FPDF_RenderPageBitmap`` takes
independent ``size_x``/``size_y`` and builds the display matrix from them, so rendering onto an
arbitrary aligned pair is one raw call away. :func:`render_pdfium` makes it, and :data:`Op.PIXELS`
asserts the dimensions agree rather than assuming they do.

:data:`Op.PIXELS` also compares at 160 DPI, which is not the feed's resolution but the resolution
:mod:`~experiments.datakit.build_pdf_source.quality.build_adjudication_set` and
:mod:`~experiments.datakit.build_pdf_source.quality.build_preference_set` render judging packets at.
Every quality label this evaluation rests on -- including the router's 19,977 preference labels --
was judged against a MuPDF-rendered page image, so a renderer change moves the judges' ground truth
too. That is a separate question from the feed's, at a different resolution, and it gets its own
number.

Crawl PDFs are adversarial and a native extension fails in ways a Python library cannot, so every
document goes through :class:`~experiments.datakit.build_pdf_source.quality.probe_pdf_inspector.Worker`
-- one document at a time over a length-prefixed pipe to a subprocess the driver is willing to lose
-- and ``exception``/``panic``/``memory``/``worker_died``/``timeout`` is a first-class result.
MuPDF's clean record on this corpus does not transfer to PDFium.

Run once per architecture. ``cw-us-east-02a`` is x86_64; a CPU-only task on ``cw-us-east-08a`` lands
on the Grace pool and is aarch64:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdfium-probe-x86 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdfium

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdfium-probe-aarch64 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_pdfium

``--extra pdf`` is required: both libraries live in marin-core's ``pdf`` extra, and the worker venv
is built from the repository's declared dependencies plus the extras passed here, never from a
local one.
"""

import base64
import io
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version

import numpy as np
import polars as pl

from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderOptions,
    target_dimensions,
)
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
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_pdfium_probe"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.probe_pdfium"
WORKER_FLAG = "--worker"

# Pages rendered per document. The feed renders every page, but a per-page cost only needs enough
# pages to be representative, and an uncapped render of the tail of the page distribution would
# dominate the run.
#
# They are spread evenly across the document rather than taken from the front, which
# ``probe_pdf_oxide`` did and this probe originally copied. Front-loading is wrong twice over: a
# document's first pages are covers and title pages more often than they are representative, and
# more importantly the failure taxonomy needs the pages a crawl PDF actually breaks on.
# ``build_render_study``'s first revision rendered mid-document pages in-process and PDFium aborted
# the process within a hundred documents, on pages a front-loaded sample never reaches.
RENDER_PAGES = 8
# Pages compared pixel for pixel, at each of the two resolutions. Two full-resolution buffers per
# renderer per page is the expensive part; two pages per document over the whole sample is already
# thousands of pages.
PIXEL_PAGES = 2
# Pages run through the three PNG encoders. Matched to probe_png_encoders.
ENCODE_PAGES = 4

# A pixel channel differing by more than this counts as changed. Below it the difference is
# anti-aliasing and colour rounding, which a VLM at ~146 DPI cannot see.
PIXEL_TOLERANCE = 16
# Fraction of changed pixels above which a page is "visibly different" -- the two renderers drew
# something structurally unlike, not the same page with softer edges.
PAGE_DIVERGENCE_FRACTION = 0.02

# The resolution the judging packets are rendered at, from build_adjudication_set.RENDER_DPI. Not
# imported: that module pulls in the whole labelling stack, and this probe only needs the number.
JUDGE_DPI = 160

# The encoder the measurement chose. Pixel-identical to MuPDF's on 3,014 of 3,014 pages and worth
# 1.7 core-h/M on x86, 4.4 on aarch64, at a 2.6% larger payload.
PILLOW_COMPRESS_LEVEL = 1
# Kept as a control: level 6 was *slower* than MuPDF on both architectures for a payload no smaller
# than level 1, which is what closes off the compression-level knob.
PILLOW_CONTROL_LEVEL = 6

RENDER_OPTIONS = RenderOptions()


class Op(StrEnum):
    """One measured route. Each is its own worker call so a library that dies is attributable."""

    RENDER_MUPDF = "render_mupdf"
    RENDER_PDFIUM = "render_pdfium"
    PIXELS = "pixels"
    ENCODERS = "encoders"
    INSPECTOR = "inspector"


# ---------------------------------------------------------------------------
# Rendering: the one thing both libraries are asked for
# ---------------------------------------------------------------------------


def judge_dimensions(width: float, height: float) -> tuple[int, int]:
    """Pixel ``(height, width)`` a judging packet's page gets, as PyMuPDF's ``dpi=`` computes it."""
    scale = JUDGE_DPI / 72.0
    return round(height * scale), round(width * scale)


def sampled_page_indices(page_count: int, wanted: int) -> list[int]:
    """Evenly spaced page indices, deterministic in the document's length alone."""
    if page_count <= wanted:
        return list(range(page_count))
    stride = page_count / (wanted + 1)
    return sorted({min(page_count - 1, round(stride * (position + 1))) for position in range(wanted)})


def render_pdfium(page, height: int, width: int) -> "np.ndarray":
    """One page onto an exactly ``width`` x ``height`` RGB buffer.

    ``pypdfium2``'s ``PdfPage.render`` only takes a scalar ``scale``, which cannot express the
    non-uniform matrix the token budget asks for. ``FPDF_RenderPageBitmap`` takes ``size_x`` and
    ``size_y`` independently and derives the display matrix from them, so passing the aligned pair
    directly rasterises straight to the target -- no decode/resize/re-encode round trip, which is
    the property the feed's contract depends on.

    ``FPDF_ANNOT`` matches PyMuPDF's ``get_pixmap(annots=True)`` default, so both renderers draw
    annotation appearance streams. The bitmap is filled white first because PDFium leaves it
    transparent otherwise, where MuPDF's ``alpha=False`` pixmap is already composited on white.

    ``FPDF_REVERSE_BYTE_ORDER`` is what actually makes an ``FPDFBitmap_BGR`` buffer hold RGB;
    ``new_native``'s ``rev_byteorder`` only records the claim on the wrapper. Without the flag the
    red and blue channels come back swapped, and because a page of black text on white is symmetric
    under that swap, the error is invisible on exactly the pages one would check first.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415
    import pypdfium2.raw as pdfium_c  # noqa: PLC0415

    bitmap = pdfium.PdfBitmap.new_native(width, height, pdfium_c.FPDFBitmap_BGR, rev_byteorder=True)
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
    flags = pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_REVERSE_BYTE_ORDER
    pdfium_c.FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, flags)
    return bitmap.to_numpy()


def render_mupdf(page, height: int, width: int) -> "np.ndarray":
    """The same page onto the same dimensions, through MuPDF's non-uniform matrix."""
    import pymupdf  # noqa: PLC0415

    page_width, page_height = page.rect.width, page.rect.height
    matrix = pymupdf.Matrix(width / page_width, height / page_height)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)


def _encode_png(image: "np.ndarray", level: int) -> bytes:
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=level)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Worker: the measured calls
# ---------------------------------------------------------------------------


@dataclass
class _Stage:
    """Wall time for one stage of the feed, accumulated over a document's pages."""

    rasterise: float = 0.0
    encode: float = 0.0
    base64: float = 0.0
    native_encode: float = 0.0


def _feed_result(stage: _Stage, pages: int, document_pages: int, pixels: int) -> dict:
    return {
        "page_count": pages,
        "document_pages": document_pages,
        "rasterise_ms": 1000 * stage.rasterise,
        "encode_ms": 1000 * stage.encode,
        "base64_ms": 1000 * stage.base64,
        "native_encode_ms": 1000 * stage.native_encode,
        "feed_ms": 1000 * (stage.rasterise + stage.encode + stage.base64),
        "megapixels": pixels / 1e6,
    }


def _render_mupdf(payload: bytes) -> dict:
    """Production's feed path, with the PNG encoder held at Pillow level 1."""
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        stage = _Stage()
        pages = pixels = 0
        for index in sampled_page_indices(len(document), RENDER_PAGES):
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)
            matrix = pymupdf.Matrix(width / page_width, height / page_height)

            started = time.perf_counter()
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rasterised = time.perf_counter()
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=PILLOW_COMPRESS_LEVEL)
            png = buffer.getvalue()
            encoded = time.perf_counter()
            base64.b64encode(png).decode()
            done = time.perf_counter()
            # Not in feed_ms. Timed so the encoder-held-constant comparison can be read against the
            # incumbent path, which encodes here.
            pixmap.tobytes("png")
            native = time.perf_counter()

            stage.rasterise += rasterised - started
            stage.encode += encoded - rasterised
            stage.base64 += done - encoded
            stage.native_encode += native - done
            pixels += height * width
            pages += 1
        return _feed_result(stage, pages, len(document), pixels)
    finally:
        document.close()


def _render_pdfium(payload: bytes) -> dict:
    """The same feed path on PDFium, split the same way and encoded by the same encoder."""
    import pypdfium2 as pdfium  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    document = pdfium.PdfDocument(payload)
    try:
        stage = _Stage()
        pages = pixels = 0
        for index in sampled_page_indices(len(document), RENDER_PAGES):
            page = document[index]
            page_width, page_height = page.get_size()
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)

            started = time.perf_counter()
            samples = render_pdfium(page, height, width)
            rasterised = time.perf_counter()
            image = Image.fromarray(samples)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=PILLOW_COMPRESS_LEVEL)
            png = buffer.getvalue()
            encoded = time.perf_counter()
            base64.b64encode(png).decode()
            done = time.perf_counter()

            stage.rasterise += rasterised - started
            stage.encode += encoded - rasterised
            stage.base64 += done - encoded
            pixels += height * width
            pages += 1
        return _feed_result(stage, pages, len(document), pixels)
    finally:
        document.close()


def _compare(left: "np.ndarray", right: "np.ndarray") -> dict:
    """Two same-shaped RGB buffers, reduced to the numbers that characterise their difference."""
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(axis=2)
    changed = float((difference > PIXEL_TOLERANCE).mean())
    return {
        "changed": changed,
        "absolute": float(difference.mean()),
        "divergent": changed > PAGE_DIVERGENCE_FRACTION,
        # Ink is what a rasteriser's stem weight shows up as. pdf_oxide laid down 5% more of it than
        # MuPDF; whether PDFium does is the same question asked of a different library.
        "left_ink": float((left.min(axis=2) < 128).mean()),
        "right_ink": float((right.min(axis=2) < 128).mean()),
    }


def _pixels(payload: bytes) -> dict:
    """Both renderers on the same pages at the same dimensions, at both resolutions that matter.

    ``feed`` is the 2048-visual-token budget the VLM reads. ``judge`` is the 160 DPI the blind
    judging packets are rendered at, which is where the corpus's existing quality labels came from.
    Both are rendered from *MuPDF's* page rectangle so the two buffers are the same shape and the
    comparison is about the rasterisers alone.

    That leaves a second question the pixel comparison cannot ask, because it holds the answer
    fixed: in production PDFium would read the page's size itself. MuPDF reports the CropBox
    intersected with the MediaBox and rotated; PDFium derives its own, and on a malformed page they
    need not agree. A disagreement would move the target dimensions and therefore the recorded
    ``effective_dpi``, so ``page_size_disagreements`` counts the pages where PDFium's own
    ``get_size`` would have produced different ``smart_resize`` dimensions than MuPDF's rectangle
    did. It is the one way the swap could change the *sizing* rather than the drawing.
    """
    import pymupdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415

    other = pdfium.PdfDocument(payload)
    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        compared = 0
        mismatched = 0
        size_disagreements = 0
        measurements: dict[str, list[dict]] = {"feed": [], "judge": []}
        for index in sampled_page_indices(min(len(document), len(other)), PIXEL_PAGES):
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            other_page = other[index]
            other_width, other_height = other_page.get_size()
            size_disagreements += target_dimensions(other_width, other_height, RENDER_OPTIONS) != target_dimensions(
                page_width, page_height, RENDER_OPTIONS
            )
            for label, (height, width) in (
                ("feed", target_dimensions(page_width, page_height, RENDER_OPTIONS)),
                ("judge", judge_dimensions(page_width, page_height)),
            ):
                left = render_mupdf(page, height, width)
                right = render_pdfium(other_page, height, width)
                if left.shape != right.shape:
                    mismatched += 1
                    continue
                measurements[label].append(_compare(left, right))
            compared += 1
        if not compared:
            return {"page_count": 0, "document_pages": len(document)}
        result = {
            "page_count": compared,
            "document_pages": len(document),
            "dimension_mismatches": mismatched,
            "page_size_disagreements": size_disagreements,
        }
        for label, entries in measurements.items():
            if not entries:
                continue
            result[f"{label}_pages"] = len(entries)
            result[f"{label}_changed_fraction"] = float(np.mean([e["changed"] for e in entries]))
            result[f"{label}_mean_absolute_difference"] = float(np.mean([e["absolute"] for e in entries]))
            result[f"{label}_divergent_pages"] = sum(e["divergent"] for e in entries)
            result[f"{label}_mupdf_ink"] = float(np.mean([e["left_ink"] for e in entries]))
            result[f"{label}_pdfium_ink"] = float(np.mean([e["right_ink"] for e in entries]))
        return result
    finally:
        document.close()
        other.close()


def _encoders(payload: bytes) -> dict:
    """The three PNG encoders over PDFium's buffers.

    ``probe_png_encoders`` established Pillow level 1 over *MuPDF's* buffers. The finding is about
    the encoder, not the rasteriser, so it should carry -- but payload size and encode time both
    depend on image content, and PDFium's content is not byte-identical, so it is measured rather
    than assumed. MuPDF's encoder is fed the same PDFium buffer through a wrapping ``Pixmap``, which
    makes this a like-for-like on one image rather than a comparison of two renderers' outputs.
    """
    import pymupdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    document = pdfium.PdfDocument(payload)
    try:
        totals = {name: [0.0, 0] for name in ("pillow_c1", "pillow_c6", "mupdf_png")}
        identical = 0
        pages = 0
        for index in sampled_page_indices(len(document), ENCODE_PAGES):
            page = document[index]
            page_width, page_height = page.get_size()
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)
            samples = render_pdfium(page, height, width)
            image = Image.fromarray(samples)

            for name, level in (("pillow_c1", PILLOW_COMPRESS_LEVEL), ("pillow_c6", PILLOW_CONTROL_LEVEL)):
                buffer = io.BytesIO()
                started = time.perf_counter()
                image.save(buffer, format="PNG", compress_level=level)
                totals[name][0] += time.perf_counter() - started
                totals[name][1] += buffer.getbuffer().nbytes

            pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, samples.tobytes(), False)
            started = time.perf_counter()
            reference = pixmap.tobytes("png")
            totals["mupdf_png"][0] += time.perf_counter() - started
            totals["mupdf_png"][1] += len(reference)

            # PNG is lossless, so two encoders of the same buffer must decode to the same bytes.
            # Asserted per page rather than argued, because it is the whole reason the encoder can
            # be swapped without re-evaluating extraction quality.
            identical += np.array_equal(np.asarray(Image.open(io.BytesIO(reference)).convert("RGB")), samples)
            pages += 1
        result = {"page_count": pages, "document_pages": len(document), "pixel_identical": identical}
        for name, (seconds, size) in totals.items():
            result[f"{name}_ms"] = 1000 * seconds
            result[f"{name}_bytes"] = size
        return result
    finally:
        document.close()


def _inspector(payload: bytes) -> dict:
    """Is pdf-inspector's bundled PDFium reachable without a second native dependency?

    pdf-inspector 1.17.0's Python wheel compiles in ``render-pdfium`` through the ``ocr`` cargo
    feature, so a PDFium binary is already in the worker environment. Whether it is *reachable* is
    a different question: a Rust crate's transitive dependency is not a Python API unless the crate
    exports one. The cheap check is to look, because a reachable renderer would mean one native
    dependency instead of two.
    """
    import pdf_inspector  # noqa: PLC0415

    exported = sorted(name for name in dir(pdf_inspector) if not name.startswith("_"))
    rendering = [name for name in exported if "render" in name.lower() or "image" in name.lower()]
    return {
        "page_count": 0,
        "document_pages": 0,
        "exports": json.dumps(exported),
        "render_exports": json.dumps(rendering),
        "version": version("pdf-inspector"),
    }


_OPERATIONS = {
    Op.RENDER_MUPDF: _render_mupdf,
    Op.RENDER_PDFIUM: _render_pdfium,
    Op.PIXELS: _pixels,
    Op.ENCODERS: _encoders,
    Op.INSPECTOR: _inspector,
}
# Run against every document. INSPECTOR is a one-shot capability question, asked once in `main`.
_PER_DOCUMENT_OPS = (Op.RENDER_MUPDF, Op.RENDER_PDFIUM, Op.PIXELS, Op.ENCODERS)


def _measure(op: str, payload: bytes) -> dict:
    """Time one call, reporting whatever it did instead of raising."""
    started = time.perf_counter()
    try:
        observed = _OPERATIONS[Op(op)](payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    # BaseException, not Exception: PyO3 derives PanicException from the former, and a panic
    # reported as a worker death would misattribute the failure taxonomy this probe exists for.
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
    pointed at stderr. MuPDF sends its "syntax error in content stream" warnings to fd 1 by default
    and PDFium writes its own diagnostics there, which would land in the middle of the
    length-prefixed protocol and make the driver read a warning where a JSON reply should be.
    """
    import faulthandler  # noqa: PLC0415

    faulthandler.enable()

    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    import pypdfium2 as pdfium  # noqa: PLC0415

    print(
        f"worker: pypdfium2 {version('pypdfium2')} (pdfium {pdfium.PDFIUM_INFO}), pymupdf {version('pymupdf')}",
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
    """Every per-document operation against one document, flattened into one row keyed by op."""
    row = {
        "source_id": document["source_id"],
        "url": document["url"],
        "sample_num_pages": document["num_pages"],
        "pdf_bytes": len(document["pdf"]),
    }
    for op in _PER_DOCUMENT_OPS:
        reply = worker.call(op, document["pdf"])
        for key, value in reply.result.items():
            row[f"{op}_{key}"] = value
        row[f"{op}_worker_lost"] = reply.worker_lost
    return row


def run_probe(documents: pl.DataFrame) -> list[dict]:
    worker = Worker(MODULE_NAME)
    rows = []
    try:
        first = documents.row(0, named=True)
        inspector = worker.call(Op.INSPECTOR, first["pdf"])
        logger.info("inspector: %s", json.dumps(inspector.result))

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


def _summarize_outcomes(rows: list[dict], op: Op) -> list[dict]:
    """Log the failure taxonomy for one op and return the rows that survived it."""
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
    return [row for row in rows if row[f"{op}_outcome"] == Outcome.OK]


def _summarize_feed(rows: list[dict], op: Op) -> None:
    """Latency distribution and the cost it implies, stage by stage."""
    good = _summarize_outcomes(rows, op)
    if not good:
        logger.warning("%s: no successful calls to time", op)
        return

    paged = [row for row in good if (row.get(f"{op}_page_count") or 0) > 0]
    if not paged:
        logger.warning("%s: no pages rendered", op)
        return
    pages = sum(row[f"{op}_page_count"] for row in paged)
    per_document = [row[f"{op}_feed_ms"] for row in good]
    per_page = [row[f"{op}_feed_ms"] / row[f"{op}_page_count"] for row in paged]
    document_p50, document_p90, document_p99 = _percentiles(per_document)
    page_p50, page_p90, page_p99 = _percentiles(per_page)
    aggregate = sum(row[f"{op}_feed_ms"] for row in paged) / pages
    logger.info(
        "%s: per-document ms p50=%.2f p90=%.2f p99=%.2f max=%.2f | per-page ms p50=%.2f p90=%.2f p99=%.2f | "
        "aggregate %.3f ms/page over %d pages = %.2f core-h / M pages",
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
    for stage in ("rasterise_ms", "encode_ms", "base64_ms", "native_encode_ms"):
        total = sum(row.get(f"{op}_{stage}") or 0.0 for row in paged)
        if not total:
            continue
        logger.info(
            "%s: %-17s %.3f ms/page = %.2f core-h / M pages (%.1f%% of the feed)",
            op,
            stage,
            total / pages,
            _core_hours_per_million(total / pages),
            100 * total / sum(row[f"{op}_feed_ms"] for row in paged),
        )
    megapixels = sum(row.get(f"{op}_megapixels") or 0.0 for row in paged)
    logger.info("%s: %.3f MP/page rendered", op, megapixels / pages)


def _summarize_pixels(rows: list[dict]) -> None:
    """How far apart the two rasterisers are, at the feed's resolution and at the judges'."""
    good = _summarize_outcomes(rows, Op.PIXELS)
    good = [row for row in good if (row.get(f"{Op.PIXELS}_page_count") or 0) > 0]
    if not good:
        logger.warning("pixels: no pages compared")
        return
    pages = sum(row[f"{Op.PIXELS}_page_count"] for row in good)
    logger.info(
        "pixels: %d documents, %d pages | buffer-shape mismatches %d | pages where PDFium's own page "
        "size would change the target dimensions: %d (%.2f%%)",
        len(good),
        pages,
        sum(row.get(f"{Op.PIXELS}_dimension_mismatches") or 0 for row in good),
        sum(row.get(f"{Op.PIXELS}_page_size_disagreements") or 0 for row in good),
        100 * sum(row.get(f"{Op.PIXELS}_page_size_disagreements") or 0 for row in good) / pages,
    )
    for label in ("feed", "judge"):
        entries = [row for row in good if row.get(f"{Op.PIXELS}_{label}_pages")]
        if not entries:
            continue
        pages = sum(row[f"{Op.PIXELS}_{label}_pages"] for row in entries)
        changed = [row[f"{Op.PIXELS}_{label}_changed_fraction"] for row in entries]
        changed_p50, changed_p90, changed_p99 = _percentiles(changed)
        logger.info(
            "pixels[%s]: %d pages | changed-pixel fraction p50=%.4f p90=%.4f p99=%.4f | mean |delta| %.2f/255 | "
            "pages over %.0f%% changed: %d (%.2f%%) | ink MuPDF %.4f vs PDFium %.4f",
            label,
            pages,
            changed_p50,
            changed_p90,
            changed_p99,
            float(np.mean([row[f"{Op.PIXELS}_{label}_mean_absolute_difference"] for row in entries])),
            100 * PAGE_DIVERGENCE_FRACTION,
            sum(row[f"{Op.PIXELS}_{label}_divergent_pages"] for row in entries),
            100 * sum(row[f"{Op.PIXELS}_{label}_divergent_pages"] for row in entries) / pages,
            float(np.mean([row[f"{Op.PIXELS}_{label}_mupdf_ink"] for row in entries])),
            float(np.mean([row[f"{Op.PIXELS}_{label}_pdfium_ink"] for row in entries])),
        )


def _summarize_encoders(rows: list[dict]) -> None:
    """Whether the Pillow finding survives PDFium's buffers."""
    good = _summarize_outcomes(rows, Op.ENCODERS)
    good = [row for row in good if (row.get(f"{Op.ENCODERS}_page_count") or 0) > 0]
    if not good:
        logger.warning("encoders: no pages encoded")
        return
    pages = sum(row[f"{Op.ENCODERS}_page_count"] for row in good)
    identical = sum(row[f"{Op.ENCODERS}_pixel_identical"] for row in good)
    baseline = sum(row[f"{Op.ENCODERS}_mupdf_png_ms"] for row in good)
    logger.info("encoders: %d pages, MuPDF-decodes-to-PDFium-buffer %d/%d", pages, identical, pages)
    for name in ("mupdf_png", "pillow_c1", "pillow_c6"):
        total = sum(row[f"{Op.ENCODERS}_{name}_ms"] for row in good)
        size = sum(row[f"{Op.ENCODERS}_{name}_bytes"] for row in good)
        logger.info(
            "encoders: %-10s %7.2f ms/page  %6.2f core-h/M  %6.0f KiB/page  %.2fx",
            name,
            total / pages,
            _core_hours_per_million(total / pages),
            size / pages / 1024,
            baseline / total if total else float("nan"),
        )


def summarize(rows: list[dict]) -> None:
    logger.info("probe: %d documents on %s", len(rows), platform.machine())
    _summarize_feed(rows, Op.RENDER_MUPDF)
    _summarize_feed(rows, Op.RENDER_PDFIUM)
    _summarize_pixels(rows)
    _summarize_encoders(rows)


def main() -> None:
    import pypdfium2 as pdfium  # noqa: PLC0415
    from rigging.log_setup import configure_logging  # noqa: PLC0415

    configure_logging(logging.INFO)
    pypdfium2 = version("pypdfium2")
    logger.info(
        "pypdfium2 %s (pdfium %s), pymupdf %s, on %s, python %s",
        pypdfium2,
        pdfium.PDFIUM_INFO,
        version("pymupdf"),
        platform.machine(),
        sys.version,
    )

    filesystem = storage()
    documents = probe_documents(filesystem)
    rows = run_probe(documents)

    output = f"{OUTPUT_PREFIX}/{platform.machine()}-{pypdfium2}.parquet"
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
