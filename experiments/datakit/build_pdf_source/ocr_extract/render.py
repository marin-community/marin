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
invisible. :func:`render_geometry` computes the same summary *before* anything is rendered, from
page rectangles alone, because the router both scores on it and uses it to choose between
:data:`DEFAULT_MAX_VISUAL_TOKENS` and :data:`RAISED_MAX_VISUAL_TOKENS` for the document.

This module is where the rasteriser lives, and it is deliberately the only place it lives: the
router reads its geometry through :func:`render_geometry` and the OCR route reads its pixels through
:func:`iter_rendered_pages`, so the engine behind both moved at once and nothing else in the
pipeline changed. Neither route calls into it from its map task. Both run it in a child process they
are willing to lose -- the geometry pass through
:mod:`~experiments.datakit.build_pdf_source.extract_inspector`'s worker, the feed through
:mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker` -- because a native abort is a
signal no ``except`` catches and Zephyr answers a dead task by restarting its shard from row zero.

That engine is PDFium, through ``pypdfium2``, and the reason is licensing: PyMuPDF is AGPL and this
was its last runtime role once the router pass and the Docling route were removed. PDFium is
BSD-3-Clause. ``pdfium-evaluation.md``, on the ``mark/pdf_processing`` campaign branch with the
probes that produced it, has the adjudication: blind, two-way and corpus-weighted, the model's
reading of a PDFium-rendered page is preferred 0.481 to 0.498 of the time against a null of 0.500,
with all five arms containing parity. The rendering changes on 8% of pages and is not
worse on them. It is also the operationally safer engine on this corpus: over every page of 100,000
oracle-sample documents on both architectures, PDFium recorded zero native aborts in 3,577,944
renders where MuPDF recorded one deterministic SIGSEGV, repeating on 3 of 3 retries.

The budget is 2048 tokens (~2.07 MP, ~146 DPI median on this crawl). The throughput sweep behind
that choice is in ``experiments/datakit/build_pdf_source/ocr-budget-sweep.md``: below 2048 quality is surrendered
for almost nothing (1024 serves within 3% and sits at the legibility floor, 512 renders 99% of pages
under it), and above it the trade is real (-21% throughput for 4096).

``pypdfium2`` and Pillow are imported inside the functions that touch a document, not at module
scope. They live in marin-core's ``pdf`` extra, which the Zephyr workers get through
``pip_dependency_groups`` but the entrypoint job does not -- its ``uv sync`` carries no extras.
Since :mod:`~experiments.datakit.build_pdf_source.pipeline` imports the OCR step to build its DAG, a
module-scope import here would kill the driver before it submitted anything. Everything above those
functions is arithmetic and is always importable.
"""

import io
import logging
import math
from collections.abc import Iterable, Iterator
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
# The budget a document flagged by the router's render policy is rendered at instead. 16384 tokens
# is 16,777,216 pixels, exactly MAX_PIXELS, so it is the largest budget this path can express.
#
# It is targeted, never global. The flagged documents are large-format scans whose pages sit at a
# median 51.6 DPI at the default budget; 0.0% of their pages reach the 300-DPI upscale cap at any
# budget, which is why the published sweep's reason for stopping at 8192 does not apply to them.
# Raising the budget for them alone rescues 1,890 of 2,234 for x1.0029 GPU crawl-wide; raising it
# for every page costs x1.530 to rescue 0.47% of them (`pdf-router-v2.md`, "The legibility floor:
# render it bigger"). The throughput at this budget is extrapolated from the sweep's curve rather
# than measured, and the report names that as the number to confirm before shipping.
RAISED_MAX_VISUAL_TOKENS = 16_384
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

# PNG encoding is the expensive half of the feed -- 65.7% of it on x86, 67.5% on aarch64 -- so the
# encoder matters more than the rasteriser did. Level 1 decodes byte-identical to level 6 on
# 3,014 of 3,014 measured pages and to MuPDF's own encoder on the same set, for a 4.8% larger
# payload, and is 1.16x faster on x86 and 1.46x on aarch64. PNG is lossless, so this is the one
# lever that moves cost with provably identical pixels; level 6 is slower than either alternative
# for a payload no smaller than level 1, so the knob has no useful middle
# (``pdfium-evaluation.md`` on ``mark/pdf_processing``, "The PNG finding survives").
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

    The PNG is carried as bytes rather than as the base64 data URI the endpoint wants, because the
    page crosses a pipe between :mod:`~...ocr_extract.render_worker`'s child and the sender task on
    its way there. Base64 is a third larger and cannot travel inside a JSON header without an escape
    scan at each end, so it is applied once, in :func:`~...ocr_extract.client.ocr_page`, which is the
    only place the wire format asks for it.
    """

    png: bytes
    page_index: int
    pixels: int
    dpi: float


@dataclass(frozen=True)
class RenderGeometry:
    """What a render budget would resolve a document to, computed without rendering anything.

    The router needs this before it decides anything, on every document rather than on the escalated
    subset, for two purposes: ``mean_dpi`` and the legible fraction derived from ``pages_below_floor``
    are two of the 43 features the score reads, and ``mean_dpi`` is the trigger for the render policy
    that picks between :data:`DEFAULT_MAX_VISUAL_TOKENS` and :data:`RAISED_MAX_VISUAL_TOKENS`.

    It is the same arithmetic :func:`iter_rendered_pages` applies, over page rectangles alone -- a
    page-tree walk with no content stream decoded and no pixel produced -- so what the router reads
    is what the feed path will do, and swapping the rasteriser moves both together.
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


def render_geometry(page_rectangles: Iterable[tuple[float, float]], options: RenderOptions) -> RenderGeometry:
    """Summarize what ``options`` would resolve a document's pages to, from their sizes in points.

    Pure arithmetic over ``(width, height)`` pairs, so it is importable and testable without a PDF
    library and is the half of the geometry pass that survives a rasteriser swap untouched.

    Every page the document declares is measured, not the first :attr:`RenderOptions.max_pages` of
    them: this describes the document's shape, which is what the router's features mean, while the
    page budget's truncation is a property of one render and is recorded separately by the OCR route
    as ``pages_unrendered``. Degenerate pages -- the sub-point rectangles
    :func:`iter_rendered_pages` refuses -- are excluded from the mean rather than counted as 0 DPI,
    which would drag a legible document under the floor on the strength of one broken page.
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

    A page whose size cannot be read at all is skipped rather than guessed at; a PDF library fails
    arbitrarily deep on crawl input and one unreadable page is not a reason to lose a document's
    geometry. This is the only rasteriser call the geometry pass makes, and it decodes no content
    stream -- it walks the page tree and reads the boxes.

    The sizes are the rasteriser's own, which is what makes the router's features describe what the
    feed will do. Over 3,585 pages on both architectures the number where PDFium's ``get_size()``
    would have produced different :func:`smart_resize` dimensions than MuPDF's page rectangle is 0.
    """
    sizes: list[tuple[float, float]] = []
    for page_index in range(len(document)):
        try:
            width, height = document[page_index].get_size()
        except Exception:
            logger.debug("Could not read the size of page %d", page_index, exc_info=True)
            continue
        sizes.append((width, height))
    return sizes


@contextmanager
def open_pdf(pdf: bytes) -> Iterator["pdfium.PdfDocument"]:  # noqa: F821
    """Open PDF bytes, closing the document on the way out.

    Raises whatever PDFium raises on input it cannot parse at all, which for a crawl corpus is a
    routine outcome rather than a pipeline failure; the caller decides what to count it as. PDFium
    refuses a slightly different set of documents than MuPDF did -- 213 pages of 1.79M that MuPDF
    renders and it does not, never the reverse -- which is a corpus-composition change, not a
    failure mode.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(pdf)
    try:
        yield document
    finally:
        document.close()


def rasterise_page(page: "pdfium.PdfPage", height: int, width: int) -> "np.ndarray":  # noqa: F821
    """One page onto an exactly ``width`` x ``height`` RGB buffer.

    ``PdfPage.render`` takes a scalar ``scale`` and cannot express the non-uniform matrix the token
    budget asks for. ``FPDF_RenderPageBitmap`` takes ``size_x`` and ``size_y`` independently and
    derives the display matrix from them, so passing the aligned pair from
    :func:`target_dimensions` rasterises straight to the target with no decode/resize/re-encode
    round trip -- the property the feed's cost depends on. Letter resolves to 1280x1632 and A4 to
    1216x1696, which is what the budget asks for exactly.

    ``FPDF_ANNOT`` draws annotation appearance streams, which MuPDF's ``get_pixmap`` did by default.
    The bitmap is filled white first because PDFium leaves it transparent, where MuPDF's
    ``alpha=False`` pixmap arrived already composited on white.

    ``FPDF_REVERSE_BYTE_ORDER`` in the render flags is what actually makes an ``FPDFBitmap_BGR``
    buffer hold RGB. ``new_native``'s ``rev_byteorder`` only records the claim on the Python
    wrapper, so ``to_numpy`` labels the buffer RGB either way; drop the flag and red and blue come
    back swapped. A page of black text on white is symmetric under that swap, so the error is
    invisible on exactly the pages one would check first -- which is why
    ``test_a_coloured_page_keeps_its_channels`` renders something that is neither black nor white
    nor grey.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415
    import pypdfium2.raw as pdfium_c  # noqa: PLC0415

    bitmap = pdfium.PdfBitmap.new_native(width, height, pdfium_c.FPDFBitmap_BGR, rev_byteorder=True)
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
    flags = pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_REVERSE_BYTE_ORDER
    pdfium_c.FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, flags)
    return bitmap.to_numpy()


def encode_png(samples: "np.ndarray") -> bytes:  # noqa: F821
    """An RGB buffer to PNG bytes, at the compression level the feed's cost was measured at."""
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(samples).save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


def iter_rendered_pages(document: "pdfium.PdfDocument", options: RenderOptions) -> Iterator[RenderedPage]:  # noqa: F821
    """Render a document's pages one at a time, encoded as PNG.

    Lazy by design, and the laziness serves two things at once. A document is not bounded in length
    and an encoded page is well over a megabyte, so rendering a whole document up front would put a
    multi-gigabyte payload in memory for the tail of the page distribution. And the OCR route
    submits each page to the GPU fleet as it is rendered, so the render of page N+1 overlaps the
    inference on page N; a batched render would idle the fleet for the length of every document.
    Both survive the move out of process only because
    :mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker` streams this generator
    down its pipe a page at a time rather than collecting it. Each page is closed as soon as it is
    encoded, so a thousand-page document does not accumulate them.

    Each page is rendered once, straight to its final resolution (see :func:`rasterise_page`). A
    page that PDFium cannot render is skipped -- crawl PDFs fail arbitrarily deep inside any
    library, and one bad page is not a reason to lose the document. ``RenderedPage.page_index`` is
    the page's position in the PDF, so a skipped page is visible as a gap.
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
