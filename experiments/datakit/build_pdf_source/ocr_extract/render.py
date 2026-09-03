# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render PDF pages to model inputs under a visual-token budget.

Page size is set by a budget in *visual tokens*, not a DPI target, so every page costs the model the
same and paper size becomes a quality question instead of a cost one. A large-format page is then
rendered at a fraction of a Letter page's resolution: :func:`effective_dpi` records what each page
got, and :func:`render_geometry` computes the same summary before anything is rendered, from page
rectangles alone, which the router scores on and uses to choose between
:data:`DEFAULT_MAX_VISUAL_TOKENS` and :data:`RAISED_MAX_VISUAL_TOKENS`.

This module is the only place the rasteriser (PDFium, through ``pypdfium2``) lives. Neither route
calls into it from its map task; both run it in a child process they are willing to lose -- the
geometry pass through :mod:`~experiments.datakit.build_pdf_source.extract_inspector`'s worker, the
feed through :mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker`.

``pypdfium2`` and Pillow are imported inside the functions that touch a document, not at module
scope. They live in marin-core's ``pdf`` extra, which the Zephyr workers get through
``pip_dependency_groups`` but the entrypoint job does not, and
:mod:`~experiments.datakit.build_pdf_source.pipeline` imports the OCR step to build its DAG.
Everything above those functions is arithmetic and is always importable.
"""

import io
import logging
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Upstream client sizing from ``infinity_parser2/utils/image.py``: sides are rounded to a multiple
# of the patch stride and the pixel count is clamped to this range.
RESIZE_FACTOR = 32
MIN_PIXELS = 2048
MAX_PIXELS = 16_777_216

# One visual token per RESIZE_FACTOR^2 pixels: this architecture is patch-16 with 2x2 merging, so
# 32x32, not the 28 of Qwen2.5-VL.
VISUAL_TOKEN_PIXELS = RESIZE_FACTOR * RESIZE_FACTOR

# The sweep behind this budget is ``ocr-budget-sweep.md`` on the ``mark/pdf_pipeline`` campaign branch.
DEFAULT_MAX_VISUAL_TOKENS = 2048
# The budget the router's render policy raises a flagged document to. 16384 tokens is exactly
# MAX_PIXELS, so it is the largest budget this path can express; it is per-document, never global.
RAISED_MAX_VISUAL_TOKENS = 16_384
# Ceiling on upscaling a small page to fill the budget; past ~300 DPI there is no glyph detail left
# to recover from a scan.
DEFAULT_MAX_RENDER_DPI = 300.0
# ~10pt body text at 100 DPI is ~14px/em, about the floor for reliable VLM reading. Pages below this
# are counted, not resized.
DEFAULT_LEGIBILITY_FLOOR_DPI = 100.0
# A tail guard: one multi-thousand-page document would hold a sender task for most of an hour.
# Truncation is recorded per document as ``pages_unrendered``.
DEFAULT_MAX_PAGES = 1000

# PNG encoding is the expensive half of the feed. Level 1 decodes byte-identical to level 6 and is
# faster; PNG is lossless, so the pixels are unchanged.
PNG_COMPRESS_LEVEL = 1


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
    """One page, rendered and encoded for the model.

    The PNG is carried as bytes rather than as a base64 data URI because the page crosses a pipe
    from the rasteriser's child; base64 is applied once, in :func:`~...ocr_extract.client.ocr_page`.
    """

    png: bytes
    page_index: int
    pixels: int
    dpi: float


@dataclass(frozen=True)
class RenderGeometry:
    """What a render budget would resolve a document to, computed without rendering anything.

    The same arithmetic :func:`iter_rendered_pages` applies, over page rectangles alone, so what the
    router reads is what the feed path will do.
    """

    pages: int
    mean_dpi: float
    pages_below_floor: int


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

    The page is scaled to fill the budget, which is what makes the budget the control over page size.
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


def render_geometry(page_rectangles: Iterable[tuple[float, float]], options: RenderOptions) -> RenderGeometry:
    """Summarize what ``options`` would resolve a document's pages to, from their sizes in points.

    Every page the document declares is measured, not the first :attr:`RenderOptions.max_pages` of
    them. The sub-point pages :func:`iter_rendered_pages` refuses are excluded from the mean rather
    than counted as 0 DPI.
    """
    dpis = [
        effective_dpi(math.prod(target_dimensions(width, height, options)), width, height)
        for width, height in page_rectangles
        if width >= 1 and height >= 1
    ]
    if not dpis:
        return RenderGeometry(pages=0, mean_dpi=0.0, pages_below_floor=0)
    return RenderGeometry(
        pages=len(dpis),
        mean_dpi=sum(dpis) / len(dpis),
        pages_below_floor=sum(1 for dpi in dpis if dpi < options.legibility_floor_dpi),
    )


def page_rectangles(document: "pdfium.PdfDocument") -> list[tuple[float, float]]:  # noqa: F821
    """Every page's ``(width, height)`` in points.

    A page whose size cannot be read is skipped rather than guessed at. This walks the page tree and
    decodes no content stream.
    """
    sizes: list[tuple[float, float]] = []
    for page_index in range(len(document)):
        try:
            width, height = document[page_index].get_size()
        except Exception:
            logger.warning("Could not read the size of page %d", page_index, exc_info=True)
            continue
        sizes.append((width, height))
    return sizes


@contextmanager
def open_pdf(pdf: bytes) -> Iterator["pdfium.PdfDocument"]:  # noqa: F821
    """Open PDF bytes, closing the document on the way out.

    Raises whatever PDFium raises on input it cannot parse; the caller decides what to count it as.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(pdf)
    try:
        yield document
    finally:
        document.close()


def rasterise_page(page: "pdfium.PdfPage", height: int, width: int) -> "np.ndarray":  # noqa: F821
    """One page onto an exactly ``width`` x ``height`` RGB buffer."""
    import pypdfium2 as pdfium  # noqa: PLC0415
    import pypdfium2.raw as pdfium_c  # noqa: PLC0415

    # ``rev_byteorder`` only labels the wrapper RGB; FPDF_REVERSE_BYTE_ORDER below is what reverses it.
    bitmap = pdfium.PdfBitmap.new_native(width, height, pdfium_c.FPDFBitmap_BGR, rev_byteorder=True)
    # PDFium leaves the bitmap transparent.
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
    # FPDF_ANNOT draws annotation appearance streams.
    flags = pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_REVERSE_BYTE_ORDER
    # size_x and size_y are independent, so the aligned pair from target_dimensions rasterises
    # straight to the target with no resize round trip.
    pdfium_c.FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, flags)
    return bitmap.to_numpy()


def encode_png(samples: "np.ndarray") -> bytes:  # noqa: F821
    """An RGB buffer to PNG bytes."""
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(samples).save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


def iter_rendered_pages(document: "pdfium.PdfDocument", options: RenderOptions) -> Iterator[RenderedPage]:  # noqa: F821
    """Render a document's pages one at a time, encoded as PNG.

    Lazy so the OCR route can submit each page as it is rendered and a long document never sits in
    memory whole; each page is closed as soon as it is encoded. A page PDFium cannot render is
    skipped, visible as a gap in ``RenderedPage.page_index``.
    """
    for page_index in range(min(len(document), options.max_pages)):
        page = None
        try:
            page = document[page_index]
            page_width, page_height = page.get_size()
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, options)
            png = encode_png(rasterise_page(page, height, width))
        except Exception:
            logger.debug("Could not render page %d", page_index, exc_info=True)
            continue
        finally:
            if page is not None:
                page.close()
        pixels = height * width
        yield RenderedPage(
            png=png,
            page_index=page_index,
            pixels=pixels,
            dpi=effective_dpi(pixels, page_width, page_height),
        )
