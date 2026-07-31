# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A PyMuPDF page backend for docling, carrying the per-span geometry the assembler needs.

Docling ships pypdfium2 and docling-parse backends. FinePDFs replaced both with PyMuPDF, and this
is that backend ported to docling 2.117's ``PdfPageBackend`` interface. Two reasons to keep it:

* the classifier in :mod:`experiments.build_pdf_source.ocr_features` already parses every document
  with PyMuPDF, so using it here means one parser and one set of failure modes for the corpus;
* it is the only backend that reports the raw span records -- font flags and line direction -- that
  :mod:`.assemble` needs to rebuild words that the PDF split across spans.

Text comes from ``get_text("dict")`` with ``TEXT_CID_FOR_UNKNOWN_UNICODE`` cleared, so a glyph with
no Unicode mapping arrives as U+FFFD rather than as a raw CID that would read as plausible text.
"""

import logging
import math
import uuid
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import pymupdf
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from experiments.build_pdf_source.docling_extract.fields import patch_docling_models

logger = logging.getLogger(__name__)

# Below this, an image is a rule, a bullet, or a scanning artefact rather than a figure.
_BITMAP_AREA_THRESHOLD = 32 * 32
_SPAN_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE


def page_geometry(page: pymupdf.Page) -> PdfPageGeometry:
    """Describe a page's boxes for docling, all in top-left origin as PyMuPDF reports them."""

    def box(rect: pymupdf.Rect) -> BoundingBox:
        return BoundingBox.from_tuple((rect.x0, rect.y0, rect.x1, rect.y1), CoordOrigin.TOPLEFT)

    return PdfPageGeometry(
        angle=0.0,
        rect=BoundingRectangle.from_bounding_box(box(page.rect)),
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        art_bbox=box(page.artbox),
        bleed_bbox=box(page.bleedbox),
        crop_bbox=box(page.cropbox),
        media_bbox=box(page.mediabox),
        trim_bbox=box(page.trimbox),
    )


def _span_rectangle(x0: float, y0: float, x1: float, y1: float) -> BoundingRectangle:
    """Build the oriented rectangle for a span from PyMuPDF's ``(x0, y0, x1, y1)``.

    Corner 0 is the top-left and corner 2 the bottom-right of a top-left-origin page, so ``r_y0``
    is the top edge and ``r_y2`` the bottom, and vertical overlap between two spans is
    ``min(r_y2) - max(r_y0)`` with no branching on origin. :meth:`BoundingRectangle.to_bounding_box`
    takes the min and max of the corners, so docling still reads a correctly-oriented box back out.
    """
    return BoundingRectangle(
        r_x0=x0,
        r_y0=y0,
        r_x1=x1,
        r_y1=y0,
        r_x2=x1,
        r_y2=y1,
        r_x3=x0,
        r_y3=y1,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def blocks_to_cells(blocks: list[dict], page_height: float) -> list[TextCell]:
    """Flatten PyMuPDF's block/line/span tree into docling text cells.

    One cell per span. Each keeps its line's bounding box and rotation in ``info`` alongside the
    span's own font ``flags``, which is what lets :mod:`.assemble` tell a superscript from a
    continuation and a line break from a word break. Zero-area spans are dropped: they carry no
    glyphs and would divide by zero in the character-width median.
    """
    cells: list[TextCell] = []
    for block in blocks:
        for line in block.get("lines", []):
            cosine, sine = line.get("dir", (1, 0))
            line_angle = math.degrees(math.atan2(sine, cosine))
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                if x1 - x0 == 0 or y1 - y0 == 0:
                    continue
                cells.append(
                    TextCell(
                        index=len(cells),
                        text=span["text"],
                        orig=span["text"],
                        from_ocr=False,
                        rect=_span_rectangle(x0, y0, x1, y1),
                        info={
                            "flags": span["flags"],
                            "line_angle": line_angle,
                            "line_bbox": (
                                BoundingBox(
                                    l=x0, t=y0, r=x1, b=y1, coord_origin=CoordOrigin.TOPLEFT
                                ).to_bottom_left_origin(page_height)
                            ),
                        },
                    )
                )
    return cells


class PyMuPdfPageBackend(PdfPageBackend):
    """One page of a PDF, read through PyMuPDF."""

    def __init__(self, document: pymupdf.Document, document_hash: str, page_index: int):
        super().__init__()
        self._page_index = page_index
        self.valid = True
        # A pymupdf.Page is only usable while its Document is open, and closing the document
        # invalidates every page already handed out -- a later call then dies inside PyMuPDF with a
        # bare "page is None" assertion rather than anything diagnosable. Hold the document here so
        # it cannot be collected while this page is alive; see PyMuPdfDocumentBackend.unload.
        self._document: pymupdf.Document | None = document
        try:
            self._page: pymupdf.Page | None = document.load_page(page_index)
            # Layout coordinates are page coordinates, so bake any /Rotate out of the page first.
            self._page.remove_rotation()
        except Exception:
            logger.info("Could not load page %d of document %s", page_index, document_hash, exc_info=True)
            self.valid = False

    @property
    def page_no(self) -> int:
        """One-based, as :class:`PdfPageBackend` requires -- ``load_page`` takes a zero-based index.

        The pipeline reconciles the pages it gets back against the numbers it expects from this
        property. Returning the zero-based index instead loses exactly one page per document,
        silently, with the rest of the conversion reported as a partial success.
        """
        return self._page_index + 1

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self.valid:
            return ""
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(self.get_size().height)
        return self._page.get_text("text", clip=pymupdf.Rect(*bbox.as_tuple()))

    def get_pymupdf_page(self) -> pymupdf.Page:
        """The underlying page, so the table reader can work on the same parse."""
        return self._page

    def get_text_cells(self) -> Iterable[TextCell]:
        if not self.valid:
            return []
        blocks = self._page.get_text("dict", flags=_SPAN_FLAGS)["blocks"]
        return blocks_to_cells(blocks, page_height=self.get_size().height)

    def get_segmented_page(self) -> SegmentedPdfPage | None:
        if not self.valid:
            return None
        return SegmentedPdfPage(
            dimension=page_geometry(self._page),
            textline_cells=list(self.get_text_cells()),
            char_cells=[],
            word_cells=[],
            has_words=False,
            has_chars=False,
        )

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        if not self.valid:
            return
        for image in self._page.get_image_info():
            box = BoundingBox.from_tuple(image["bbox"], origin=CoordOrigin.TOPLEFT)
            if box.area() > _BITMAP_AREA_THRESHOLD:
                yield box.scaled(scale=scale)

    def get_page_image(self, scale: float = 1, cropbox: BoundingBox | None = None) -> Image.Image:
        if not self.valid:
            raise RuntimeError(f"Cannot render page {self._page_index}: it did not load")
        matrix = pymupdf.Matrix(scale, scale)
        if cropbox is None:
            pixmap = self._page.get_pixmap(matrix=matrix)
        else:
            clip = cropbox.to_top_left_origin(self.get_size().height)
            pixmap = self._page.get_pixmap(matrix=matrix, clip=clip.as_tuple())
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples_mv)

    def get_size(self) -> Size:
        return Size(width=self._page.cropbox.width, height=self._page.cropbox.height)

    def is_valid(self) -> bool:
        return self.valid

    def unload(self):
        # Clearing `valid` matters as much as dropping the page: every accessor here guards on it,
        # so leaving it set turns each guard into a null dereference on the next call. Docling's
        # threaded pipeline can still reach a page after this -- the table stage runs late enough
        # that it does.
        self.valid = False
        self._page = None
        self._document = None


class PyMuPdfDocumentBackend(PdfDocumentBackend):
    """A whole PDF, read through PyMuPDF."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: PdfBackendOptions | None = None,
    ):
        patch_docling_models()
        super().__init__(in_doc, path_or_stream, options)

        if isinstance(self.path_or_stream, Path):
            self._document = pymupdf.open(str(self.path_or_stream))
        else:
            # PyMuPDF keys its internal caches on the filename, so give each stream a unique one.
            self._document = pymupdf.open(filename=str(uuid.uuid4()), filetype="pdf", stream=self.path_or_stream)

    def page_count(self) -> int:
        return self._document.page_count

    def load_page(self, page_no: int) -> PyMuPdfPageBackend:
        return PyMuPdfPageBackend(self._document, self.document_hash, page_no)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        # Drop the reference rather than calling close(). Pages handed out by load_page stay usable
        # only while the document is open, and docling's threaded pipeline does not guarantee every
        # page stage has finished when the document is unloaded -- closing here invalidated pages
        # the table stage was still reading. Releasing the reference lets PyMuPDF close the document
        # once the last page backend has released it too.
        self._document = None
