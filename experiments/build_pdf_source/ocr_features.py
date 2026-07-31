# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 124 PyMuPDF features the FinePDFs OCR router was trained on.

Ported from ``blocks/predictor/ocr_predictor.py`` in https://github.com/huggingface/finepdfs.
The feature set is fixed by the trained booster, so the arithmetic here is deliberately
value-for-value with the original -- :data:`FEATURE_NAMES` is asserted against the booster's own
``feature_names`` in :mod:`experiments.build_pdf_source.classify`. Three things did change:

* **Image geometry usually no longer decodes images.** The original called
  ``page.get_image_rects(xref)`` once per image xref per page. That builds a full ``Pixmap`` of the
  image -- a complete JPEG/JPX decode -- purely to get an MD5, then scans the page with
  ``get_image_info(hashes=True)`` (which digests every image again) and keeps the placements whose
  digest matches. On an 8-page 300-DPI scan that measured 419 ms of the ~420 ms total; one plain
  ``get_image_info()`` per page returns the same placements in 0.4 ms. ``get_image_info(xrefs=True)``
  is no cheaper, because asking for xrefs forces the same hashing.

  Decoding was only ever needed to attribute a *placement* to an xref, and that attribution only
  changes a feature on pages that have a junk image to exclude. Junk images are identified from the
  resource lists (:func:`junk_image_xrefs`), which is decode-free and is what the original did too.
  So :func:`_non_junk_bboxes` takes the plain placements when the page has no junk xref -- the case
  for essentially every scanned page, which is where the big images are -- and falls back to the
  original's digest matching only when there is something to exclude. Results are identical either
  way; only the cost differs.

* **Only sampled pages are read.** The original ran two whole-document text extractions per PDF:
  one in ``check_is_corrupted_or_encrypted`` to provoke MuPDF errors, one in
  ``get_garbled_text_per_age``. Neither result was used whole -- ``garbled_text_ratio`` is a ratio
  over the sampled pages only, and the document-wide ``global_garbled_text_ratio`` it also computed
  is not a model feature. Both collapse into the single sampled-page pass in :func:`_read_page`,
  which leaves ``garbled_text_ratio`` identical and makes corruption detection sample-scoped: a
  document whose only malformed pages went unsampled is no longer dropped.

* **Sampling is seeded per document.** The original drew pages with the process-global ``random``
  and padded short documents with the global ``numpy`` generator, so re-running scored the same PDF
  differently. :func:`document_features` derives its seed from the caller's key, which makes a
  re-run of a shard reproduce its predictions exactly.
"""

import dataclasses
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import pymupdf

from experiments.build_pdf_source.ocr_feature_names import (
    FEATURE_NAMES,
    FEATURE_PAGES,
    PAGE_FEATURE_FIELDS,
)

logger = logging.getLogger(__name__)

_PAGE_FEATURE_VALUES = attrgetter(*(field for _, field in PAGE_FEATURE_FIELDS))

# Creator/producer substrings that mark a PDF as scanner output.
KNOWN_SCANNER_STRINGS = (
    "scanner",
    "scan",
    "epson",
    "hp scanjet",
    "canon",
    "fujitsu",
    "kodak",
    "brother",
    "xerox",
    "lexmark",
    "kmc",
    "kofax",
    "ricoh",
    "iris",
    "capturedocument",
    "paperport",
    "readiris",
    "simpleocr",
)

# MuPDF diagnostics that mean the file is too malformed to extract from.
_CORRUPTION_PATTERNS = (
    r"format error: object out of range",
    r"syntax error: no XObject subtype specified",
    r"syntax error: syntax error in content stream",
    r"object is not a stream",
    r"syntax error: syntax error in array",
    r"format error: cannot load page tree",
    r"syntax error: cannot parse indirect object",
)
_CORRUPTION_RE = re.compile("|".join(_CORRUPTION_PATTERNS))

# An image is "junk" -- letterhead, watermark, logo -- when it appears on every sampled page, which
# only becomes meaningful once at least this many pages were sampled.
_JUNK_IMAGE_MIN_PAGES = 3

# Tolerances for merging a page's image strips back into one bitmap. Scanners often slice a page
# into full-width bands, which have to be recombined before "how much of the page is bitmap?" means
# anything. A strip must fill this fraction of the page's width (or height) to be a merge candidate.
_STRIP_FULL_SPAN_RATIO = 0.9
_MERGE_MAX_OFFSET = 5
_MERGE_MAX_GAP = 2

_REPLACEMENT_CHAR = chr(0xFFFD)
_TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP

# MuPDF text spans use type 3 for invisible rendering mode.
_INVISIBLE_RENDER_MODE = 3

# Path construction operators counted as strokes.
_STROKE_OPERATORS = frozenset("lcq")

type BBox = tuple[float, float, float, float]


class CorruptPdf(Exception):
    """The document is encrypted, password-protected, or too malformed to extract from."""


@dataclass(frozen=True)
class PageFeatures:
    """One sampled page's contribution to the feature vector."""

    unique_font_count: int
    char_count: int
    text_box_count: int
    mean_text_box_area: float
    text_area_ratio: float
    hidden_char_count: int
    hidden_text_box_count: int
    mean_hidden_text_box_area: float
    hidden_text_area_ratio: float
    image_count: int
    non_junk_image_count: int
    bitmap_proportion: float
    max_merged_strip_area: float
    drawing_stroke_count: int
    vector_graphics_object_count: int


@dataclass(frozen=True)
class DocumentFeatures:
    """A document's model features, plus the metadata the classifier reports alongside them."""

    num_pages: int
    num_pages_successfully_sampled: int
    garbled_text_ratio: float
    is_form: bool
    creator_or_producer_is_known_scanner: bool
    pages: tuple[PageFeatures, ...]
    """Exactly :data:`FEATURE_PAGES` entries, short documents having repeated sampled pages."""

    def vector(self) -> np.ndarray:
        """Return the 124 features as a float32 row in booster order."""
        page_values = [_PAGE_FEATURE_VALUES(page) for page in self.pages]
        return np.fromiter(
            (
                float(self.num_pages_successfully_sampled),
                self.garbled_text_ratio,
                float(self.is_form),
                float(self.creator_or_producer_is_known_scanner),
                *(float(values[index]) for index in range(len(PAGE_FEATURE_FIELDS)) for values in page_values),
            ),
            dtype=np.float32,
            count=len(FEATURE_NAMES),
        )


@dataclass(frozen=True)
class _PagePass:
    """Everything read off one page in the single decode-free PyMuPDF pass over the sample."""

    page_index: int
    text_length: int
    garbled_count: int
    unique_font_count: int
    char_count: int
    text_box_count: int
    text_area: float
    hidden_char_count: int
    hidden_text_box_count: int
    hidden_text_area: float
    page_width: float
    page_height: float
    resource_xrefs: frozenset[int]
    """The image objects the page's resource dictionary names."""
    resource_shapes: frozenset[tuple[int, int, int]]
    """Those objects' (width, height, bpc), which is how a placement is tied back to one."""
    placements: tuple[BBox, ...]
    """Where those images are drawn, in content-stream order, unattributed to individual xrefs."""
    drawing_stroke_count: int
    vector_graphics_object_count: int

    @property
    def page_area(self) -> float:
        """The page's area, normalised away from zero because it only ever divides."""
        return float(self.page_width * self.page_height) or 1.0


def sample_page_indices(num_pages: int, rng: random.Random) -> list[int]:
    """Draw a sorted sample of at most :data:`FEATURE_PAGES` distinct page indices."""
    return sorted(rng.sample(range(num_pages), min(FEATURE_PAGES, num_pages)))


def _unique_font_count(page: pymupdf.Page) -> int:
    fonts = set()
    for font in page.get_fonts(full=True):
        if len(font) > 3 and font[3]:
            fonts.add(font[3])
    return len(fonts)


def _image_shape(image: dict) -> tuple[int, int, int]:
    """The dimensions and depth MuPDF reports for a placement, used to tie it to a resource image."""
    return image["width"], image["height"], image["bpc"]


def _image_placements(page: pymupdf.Page, resource_shapes: frozenset[tuple[int, int, int]]) -> tuple[BBox, ...]:
    """Return where the page's resource images are drawn, without decoding any of them.

    Placements matching no resource image are inline images -- ``BI``/``ID``/``EI`` sequences carried
    in the content stream rather than referenced as objects. The original skipped them, because it
    only ever counted placements it could digest-match to an entry in the resource list, so they are
    skipped here too.
    """
    return tuple(tuple(image["bbox"]) for image in page.get_image_info() if _image_shape(image) in resource_shapes)


def _drawing_counts(page: pymupdf.Page) -> tuple[int, int]:
    """Return the page's (stroke count, vector object count)."""
    drawings = page.get_cdrawings()
    strokes = 0
    for path in drawings:
        strokes += sum(1 for item in path.get("items", ()) if item[0] in _STROKE_OPERATORS)
        # An outlined rect or quad contributes its whole outline as one stroke.
        if (path.get("rect") or path.get("quad")) and path.get("stroke_opacity", 1) > 0 and path.get("color"):
            strokes += 1
    return strokes, len(drawings)


def _read_page(page: pymupdf.Page, page_index: int) -> _PagePass:
    """Read one page's raw measurements, decoding nothing.

    Font, image and drawing reads are individually tolerant of failure, degrading to "this page has
    none" rather than discarding the page. That is deliberate rather than defensive: it reproduces the
    original's granularity, where only ``load_page`` decided whether a page counted and each
    subsequent read sat in its own ``try/except: pass``. Dropping the whole page instead would change
    ``num_pages_successfully_sampled`` and slide every later page slot, which is a document-level
    feature shift on any PDF with one bad page. Text reads are deliberately *not* guarded, matching
    the original -- a page whose text cannot be read fails the whole document.
    """
    try:
        # Xref 0 is MuPDF's null object; entries are (xref, smask, width, height, bpc, ...).
        resources = [image for image in page.get_images(full=True) if image[0] != 0]
    except Exception:
        logger.debug("No image resources readable on page %d", page_index, exc_info=True)
        resources = []
    resource_shapes = frozenset((image[2], image[3], image[4]) for image in resources)

    text = page.get_text("text", flags=_TEXT_FLAGS)

    char_count = text_box_count = hidden_char_count = hidden_text_box_count = 0
    text_area = hidden_text_area = 0.0
    for span in page.get_texttrace():
        chars = len(span.get("chars", ()))
        x0, y0, x1, y1 = span["bbox"]
        area = (x1 - x0) * (y1 - y0)
        if span.get("type") == _INVISIBLE_RENDER_MODE or span.get("opacity", 1.0) == 0:
            hidden_char_count += chars
            hidden_text_area += area
            hidden_text_box_count += 1
        else:
            char_count += chars
            text_area += area
            text_box_count += 1

    try:
        strokes, vector_objects = _drawing_counts(page)
    except Exception:
        logger.debug("No drawings readable on page %d", page_index, exc_info=True)
        strokes, vector_objects = 0, 0

    try:
        unique_fonts = _unique_font_count(page)
    except Exception:
        logger.debug("No fonts readable on page %d", page_index, exc_info=True)
        unique_fonts = 0

    try:
        placements = _image_placements(page, resource_shapes)
    except Exception:
        logger.debug("No image placements readable on page %d", page_index, exc_info=True)
        placements = ()

    rect = page.rect
    return _PagePass(
        page_index=page_index,
        text_length=len(text),
        garbled_count=text.count(_REPLACEMENT_CHAR),
        unique_font_count=unique_fonts,
        char_count=char_count,
        text_box_count=text_box_count,
        text_area=text_area,
        hidden_char_count=hidden_char_count,
        hidden_text_box_count=hidden_text_box_count,
        hidden_text_area=hidden_text_area,
        page_width=rect.width,
        page_height=rect.height,
        resource_xrefs=frozenset(image[0] for image in resources),
        resource_shapes=resource_shapes,
        placements=placements,
        drawing_stroke_count=strokes,
        vector_graphics_object_count=vector_objects,
    )


def junk_image_xrefs(passes: list[_PagePass], num_sampled: int) -> frozenset[int]:
    """Return the image objects on every sampled page, which are page furniture, not content.

    Decode-free: an object reused as letterhead, a watermark or a logo is one xref named by every
    page's resources, whereas a scanned page's bitmap is a distinct object per page.

    The threshold is ``num_sampled`` -- how many pages were *drawn* -- not how many could be read.
    That is the original's behaviour and it matters: an xref's count cannot exceed the number of
    pages that contributed resources, so one unreadable sampled page puts the threshold out of reach
    and leaves nothing marked junk. Counting only readable pages instead would find junk where the
    original finds none, and quietly lower ``non_junk_image_count`` on any PDF with a bad page.
    """
    if num_sampled < _JUNK_IMAGE_MIN_PAGES:
        return frozenset()
    appearances = Counter()
    for page_pass in passes:
        appearances.update(page_pass.resource_xrefs)
    return frozenset(xref for xref, pages in appearances.items() if pages >= num_sampled)


def merge_image_strips(bboxes: list[BBox], page_width: float, page_height: float) -> list[BBox]:
    """Recombine full-width or full-height image bands into the bitmaps they were sliced from.

    Boxes are deduplicated, swept top-to-bottom then left-to-right, and each is folded into the
    previous one when it spans the page and butts up against it within the merge tolerances.
    """
    unique = list(dict.fromkeys(bboxes))
    if not unique:
        return []

    unique.sort(key=lambda box: (box[1], box[0]))
    merged = [unique[0]]
    for box in unique[1:]:
        x0, y0, x1, y1 = box
        last_x0, last_y0, last_x1, last_y1 = merged[-1]

        spans_width = page_width > 0 and abs(x1 - x0) >= page_width * _STRIP_FULL_SPAN_RATIO
        spans_height = page_height > 0 and abs(y1 - y0) >= page_height * _STRIP_FULL_SPAN_RATIO

        stacks_vertically = spans_width and (
            abs(last_x0 - x0) <= _MERGE_MAX_OFFSET
            and abs(last_x1 - x1) <= _MERGE_MAX_OFFSET
            and abs(y0 - last_y1) <= _MERGE_MAX_GAP
        )
        stacks_horizontally = spans_height and (
            abs(last_y0 - y0) <= _MERGE_MAX_OFFSET
            and abs(last_y1 - y1) <= _MERGE_MAX_OFFSET
            and abs(x0 - last_x1) <= _MERGE_MAX_GAP
        )

        if stacks_vertically or stacks_horizontally:
            merged[-1] = (min(x0, last_x0), min(y0, last_y0), max(x1, last_x1), max(y1, last_y1))
        else:
            merged.append(box)
    return merged


def _non_junk_bboxes(doc: pymupdf.Document, page_pass: _PagePass, junk: frozenset[int]) -> list[BBox]:
    """Return the bounding boxes of the page's images that are not page furniture.

    With nothing to exclude, every placement qualifies and no image is touched. Otherwise the junk
    objects have to be matched to the placements that drew them, and MuPDF only relates the two
    through the digest of the decoded pixels -- so those images, and only those, are decoded.
    """
    page_junk = junk & page_pass.resource_xrefs
    if not page_junk:
        return list(page_pass.placements)

    junk_digests = set()
    for xref in page_junk:
        pixmap = pymupdf.Pixmap(doc, xref)
        junk_digests.add(pixmap.digest)
        del pixmap
    page = doc.load_page(page_pass.page_index)
    return [
        tuple(image["bbox"])
        for image in page.get_image_info(hashes=True)
        if _image_shape(image) in page_pass.resource_shapes and image["digest"] not in junk_digests
    ]


def _page_features(doc: pymupdf.Document, page_pass: _PagePass, junk: frozenset[int]) -> PageFeatures:
    non_junk = _non_junk_bboxes(doc, page_pass, junk)
    strip_areas = [
        abs(box[2] - box[0]) * abs(box[3] - box[1])
        for box in merge_image_strips(
            [box for box in non_junk if _has_area(box)], page_pass.page_width, page_pass.page_height
        )
    ]
    return PageFeatures(
        unique_font_count=page_pass.unique_font_count,
        char_count=page_pass.char_count,
        text_box_count=page_pass.text_box_count,
        mean_text_box_area=page_pass.text_area / page_pass.text_box_count if page_pass.text_box_count else 0.0,
        text_area_ratio=page_pass.text_area / page_pass.page_area,
        hidden_char_count=page_pass.hidden_char_count,
        hidden_text_box_count=page_pass.hidden_text_box_count,
        mean_hidden_text_box_area=(
            page_pass.hidden_text_area / page_pass.hidden_text_box_count if page_pass.hidden_text_box_count else 0.0
        ),
        hidden_text_area_ratio=page_pass.hidden_text_area / page_pass.page_area,
        image_count=len(page_pass.placements),
        non_junk_image_count=len(non_junk),
        bitmap_proportion=sum(strip_areas) / page_pass.page_area,
        max_merged_strip_area=max(strip_areas, default=0.0) / page_pass.page_area,
        drawing_stroke_count=page_pass.drawing_stroke_count,
        vector_graphics_object_count=page_pass.vector_graphics_object_count,
    )


def _has_area(bbox: BBox) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _is_known_scanner(doc: pymupdf.Document) -> bool:
    metadata = doc.metadata or {}
    creator = (metadata.get("creator") or "").lower()
    producer = (metadata.get("producer") or "").lower()
    return any(keyword in creator or keyword in producer for keyword in KNOWN_SCANNER_STRINGS)


def document_features(doc: pymupdf.Document, *, seed: int) -> DocumentFeatures:
    """Extract the booster's features from an open document.

    Raises:
        CorruptPdf: the document is encrypted, needs a password, has no pages, or MuPDF reported a
            structural error while reading the sampled pages.
    """
    if doc.is_encrypted or doc.needs_pass:
        raise CorruptPdf("document is encrypted or password-protected")
    if len(doc) == 0:
        raise CorruptPdf("document has no pages")

    sampled = sample_page_indices(len(doc), random.Random(seed))

    # MuPDF reports structural damage as warnings while pages are read rather than as exceptions,
    # so drain the store first and inspect it once the sampled pages have been touched.
    pymupdf.TOOLS.reset_mupdf_warnings()
    passes: list[_PagePass] = []
    for index in sampled:
        try:
            passes.append(_read_page(doc.load_page(index), index))
        except Exception:
            logger.debug("Skipping unreadable page %d", index, exc_info=True)
    warnings = pymupdf.TOOLS.mupdf_warnings()

    corruption = _CORRUPTION_RE.search(warnings)
    if corruption:
        raise CorruptPdf(f"malformed document: {corruption.group(0)}")
    if not passes:
        raise CorruptPdf("no sampled page could be read")

    junk = junk_image_xrefs(passes, len(sampled))
    features = [_page_features(doc, page_pass, junk) for page_pass in passes]

    # Short documents repeat sampled pages to fill the booster's eight slots.
    padding = random.Random(seed).choices(features, k=FEATURE_PAGES - len(features))

    text_length = sum(page_pass.text_length for page_pass in passes)
    garbled = sum(page_pass.garbled_count for page_pass in passes)
    return DocumentFeatures(
        num_pages=len(doc),
        num_pages_successfully_sampled=len(passes),
        garbled_text_ratio=garbled / text_length if text_length else 0.0,
        is_form=bool(doc.is_form_pdf),
        creator_or_producer_is_known_scanner=_is_known_scanner(doc),
        pages=tuple([*features, *padding]),
    )


assert tuple(field.name for field in dataclasses.fields(PageFeatures)) == tuple(
    field for _, field in PAGE_FEATURE_FIELDS
), "PageFeatures fields must stay in booster order"
