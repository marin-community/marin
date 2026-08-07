# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render PDF pages to model inputs under a visual-token budget.

Page size is set by a budget in *visual tokens*, not a DPI target. Under a DPI target the cost of a
page is a function of its paper size, so a crawl's page-size mix decides throughput and no two
shards are comparable; under a token budget every page costs the model the same and paper size
becomes a quality question instead of a cost one.

That trade has a failure mode worth measuring: a large-format page is quietly rendered at a fraction
of a Letter page's resolution rather than costing more. :func:`effective_dpi` records what each page
actually got, and the extraction step carries the per-document summary through to its output, so a
corpus can be audited for pages that were rendered below legibility instead of the fact being
invisible.

The budget is 2048 tokens (~2.07 MP, ~146 DPI median on this crawl). The throughput sweep behind
that choice is in ``experiments/datakit/build_pdf_source/ocr-budget-sweep.md``: below 2048 quality is surrendered
for almost nothing (1024 serves within 3% and sits at the legibility floor, 512 renders 99% of pages
under it), and above it the trade is real (-21% throughput for 4096).

PyMuPDF is imported inside the two functions that touch a document, not at module scope. It lives in
marin-core's ``datakit`` extra, which the Zephyr workers get through ``pip_dependency_groups`` but
the entrypoint job does not -- its ``uv sync`` carries no extras. Since
:mod:`~experiments.datakit.build_pdf_source.pipeline` imports the OCR step to build its DAG, a module-scope
``import pymupdf`` here would kill the driver before it submitted anything. Everything above those
two functions is arithmetic and is always importable.
"""

import base64
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Upstream client sizing from ``infinity_parser2/utils/image.py``: sides are rounded to a multiple
# of the patch stride and the pixel count is clamped to this range. The model was validated against
# inputs shaped this way, so the numbers are the model's, not ours.
RESIZE_FACTOR = 32
MIN_PIXELS = 2048
MAX_PIXELS = 16_777_216

# One visual token per RESIZE_FACTOR^2 pixels. This architecture is patch-16 with 2x2 merging, so
# 32x32. NOT the 28 of Qwen2.5-VL (patch-14) that the olmOCR-lineage pipelines use -- reusing 28
# here mis-sizes every page by (32/28)^2 = 1.31x.
VISUAL_TOKEN_PIXELS = RESIZE_FACTOR * RESIZE_FACTOR

DEFAULT_MAX_VISUAL_TOKENS = 2048
# Ceiling on upscaling a small page to fill the budget. Past ~300 DPI there is no more glyph detail
# to recover from a scan, only tokens to burn on it.
DEFAULT_MAX_RENDER_DPI = 300.0
# ~10pt body text at 100 DPI is ~14px/em, about the floor for reliable VLM reading. Pages below this
# are counted, not resized: they want tiling, and raising the global budget to rescue them overpays
# for the ~95% of pages that are already fine.
DEFAULT_LEGIBILITY_FLOOR_DPI = 100.0
# A tail guard, not a content policy. The page distribution runs past 3,000 pages, and one such
# document would hold a sender task for the better part of an hour. Truncation is recorded per
# document (``pages_unrendered``) rather than being silent.
DEFAULT_MAX_PAGES = 1000


@dataclass(frozen=True)
class RenderOptions:
    """How a page is turned into a model input."""

    max_visual_tokens: int = DEFAULT_MAX_VISUAL_TOKENS
    max_render_dpi: float = DEFAULT_MAX_RENDER_DPI
    legibility_floor_dpi: float = DEFAULT_LEGIBILITY_FLOOR_DPI
    max_pages: int = DEFAULT_MAX_PAGES

    def __post_init__(self) -> None:
        budget_pixels = self.max_visual_tokens * VISUAL_TOKEN_PIXELS
        if budget_pixels > MAX_PIXELS:
            raise ValueError(
                f"max_visual_tokens={self.max_visual_tokens} is {budget_pixels} pixels, above the "
                f"upstream client ceiling of {MAX_PIXELS}; the model was not validated there"
            )
        if budget_pixels < MIN_PIXELS:
            raise ValueError(f"max_visual_tokens={self.max_visual_tokens} is below the {MIN_PIXELS}-pixel floor")


@dataclass(frozen=True)
class RenderedPage:
    """One page, rendered and encoded for the model."""

    data_uri: str
    page_index: int
    pixels: int
    dpi: float


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen-VL input sizing, ported from ``qwen_vl_utils.vision_process.smart_resize``.

    Rounds each side to a multiple of ``factor`` and rescales so the pixel count lands within
    ``[min_pixels, max_pixels]``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {height}x{width}")
    aligned_height = max(factor, round(height / factor) * factor)
    aligned_width = max(factor, round(width / factor) * factor)
    if aligned_height * aligned_width > max_pixels:
        shrink = math.sqrt((height * width) / max_pixels)
        aligned_height = max(factor, math.floor(height / shrink / factor) * factor)
        aligned_width = max(factor, math.floor(width / shrink / factor) * factor)
    elif aligned_height * aligned_width < min_pixels:
        grow = math.sqrt(min_pixels / (height * width))
        aligned_height = math.ceil(height * grow / factor) * factor
        aligned_width = math.ceil(width * grow / factor) * factor
    return aligned_height, aligned_width


def target_dimensions(width: float, height: float, options: RenderOptions) -> tuple[int, int]:
    """Page size in points to final pixel dimensions, as ``(height, width)``.

    The page is scaled to *fill* the budget, which is what makes the budget the control that moves
    page size. A fixed long-side floor cannot do that job: it pins Letter and A4 at one size and
    leaves the budget slack, so changing the budget barely moves anything for the bulk of a crawl.
    """
    budget_pixels = options.max_visual_tokens * VISUAL_TOKEN_PIXELS
    scale = min(math.sqrt(budget_pixels / (width * height)), options.max_render_dpi / 72.0)
    return smart_resize(round(height * scale), round(width * scale), RESIZE_FACTOR, MIN_PIXELS, budget_pixels)


def effective_dpi(pixels: int, width: float, height: float) -> float:
    """Geometric-mean DPI a page was actually rendered at.

    Per-axis DPI differs slightly because the render matrix is not uniform -- it hits the aligned
    dimensions exactly rather than letter-boxing -- so the comparison is made on areas.
    """
    points_area = width * height
    return 72.0 * math.sqrt(pixels / points_area) if points_area > 0 else 0.0


@contextmanager
def open_pdf(pdf: bytes) -> Iterator["pymupdf.Document"]:  # noqa: F821
    """Open PDF bytes, closing the document on the way out.

    Raises whatever MuPDF raises on input it cannot parse at all, which for a crawl corpus is a
    routine outcome rather than a pipeline failure; the caller decides what to count it as.
    """
    import pymupdf  # noqa: PLC0415

    document = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        yield document
    finally:
        document.close()


def iter_rendered_pages(document: "pymupdf.Document", options: RenderOptions) -> Iterator[RenderedPage]:  # noqa: F821
    """Render a document's pages one at a time, encoded as PNG data URIs.

    Lazy by design. A document is not bounded in length and an encoded page is well over a megabyte,
    so rendering a whole document up front would put a multi-gigabyte payload in memory for the tail
    of the page distribution. Yielding pages lets the caller hold only what is in flight.

    Each page is rendered once, straight to its final resolution: the target comes from the token
    budget and PyMuPDF's matrix scales directly to it, so there is no decode/resize/re-encode round
    trip. A page that MuPDF cannot render is skipped -- crawl PDFs fail arbitrarily deep inside the
    library, and one bad page is not a reason to lose the document. ``RenderedPage.page_index`` is
    the page's position in the PDF, so a skipped page is visible as a gap.
    """
    import pymupdf  # noqa: PLC0415

    for page_index in range(min(len(document), options.max_pages)):
        try:
            page = document[page_index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, options)
            matrix = pymupdf.Matrix(width / page_width, height / page_height)
            png = page.get_pixmap(matrix=matrix).tobytes("png")
        except Exception:
            logger.debug("Could not render page %d", page_index, exc_info=True)
            continue
        pixels = height * width
        yield RenderedPage(
            data_uri=f"data:image/png;base64,{base64.b64encode(png).decode()}",
            page_index=page_index,
            pixels=pixels,
            dpi=effective_dpi(pixels, page_width, page_height),
        )
