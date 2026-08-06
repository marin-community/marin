# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cheap PyMuPDF signals for deciding whether Docling can stand in for the VLM.

The FinePDFs router this replaces (:mod:`experiments.build_pdf_source.ocr_features`) answers one
question: *does this document have a usable text layer at all?* Its features are counts of text,
images and drawings per sampled page. That is the right question for "scan or not" and the wrong
one for the routing decision actually being made here, which is whether Docling's reading of an
existing text layer will match the VLM's reading of the rendered page. A born-digital paper with a
broken ToUnicode CMap, an invisible OCR layer over a scan, a two-column layout, and a page of
equations all have healthy text-layer statistics and all extract badly.

So these features measure the *text layer's trustworthiness and the page's structural difficulty*,
in six groups:

``encoding``
    Whether the glyph-to-Unicode mapping can be believed: embedded fonts, ToUnicode coverage, Type3
    and glyphless fonts, replacement and private-use characters, and the ratio checks Marker's
    ``detect_bad_ocr`` uses. A subsetted font with no ToUnicode yields confident nonsense, which is
    the failure the replacement-character check alone misses -- MuPDF only emits U+FFFD when it
    knows it failed.
``layer``
    Whether the text sits where the ink is: invisible text, invisible text drawn over a bitmap
    (a scan carrying somebody else's OCR), overlapping and out-of-page lines, and duplicated spans
    from fake-bold double rendering.
``math``
    TeX/AMS/STIX font names and the fraction of characters in mathematical Unicode blocks. Both are
    Marker's inline-math signals; a formula's text layer is a stream of single glyphs whose reading
    order is meaningless.
``structure``
    Ruling lines and the grid they imply, plus how tabular the text's left edges are. Tables are
    where the two routes diverge by construction, so the question is how much of the page is one.
``order``
    How far the content-stream order of the page's text blocks is from a column-aware geometric
    reading order, and how many columns there are. Docling orders by layout model, the VLM by what
    it sees; a page whose stream order already disagrees with its geometry is where they part.
``script``
    CJK, RTL and Latin character fractions. Both routes are weaker off-Latin, in opposite ways.

Everything here is decode-free and model-free: no page is rendered, no image is decompressed, and
the only text pass is one ``get_text("rawdict")`` per sampled page. That is roughly 10-30 ms on a
typical page against Docling's ~1 s, so the router stays two orders of magnitude cheaper than the
extraction it is deciding to skip.
"""

import logging
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, fields
from itertools import pairwise

import pymupdf

from experiments.build_pdf_source.quality.route_feature_names import FEATURE_NAMES, PAGE_SIGNAL_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_NAMES",
    "PAGE_SIGNAL_NAMES",
    "CorruptPdf",
    "DocumentSignals",
    "PageSignals",
    "document_signals",
    "page_signals",
    "sample_page_indices",
    "signals_for_routing",
]

# Pages sampled per document. The FinePDFs router samples 8; this samples the same count so the
# two feature sets describe the same pages when both are computed in one pass.
SAMPLE_PAGES = 8

# MuPDF diagnostics that mean the file is too malformed to extract from. Carried over verbatim from
# the FinePDFs port so that replacing the router does not silently change *which documents are
# refused outright*: a document neither route can open is dropped, and that decision predates this
# model and should not ride along with it.
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


class CorruptPdf(Exception):
    """The document is encrypted, password-protected, or too malformed to extract from."""


_REPLACEMENT = 0xFFFD
_PUA = (0xE000, 0xF8FF)
_LIGATURES = frozenset("ﬀﬁﬂﬃﬄﬅﬆ")

# Mathematical Unicode blocks, following Marker's inline-math detector.
_MATH_RANGES = (
    (0x0370, 0x03FF),  # Greek
    (0x2070, 0x209F),  # super/subscripts
    (0x2100, 0x214F),  # letterlike symbols
    (0x2190, 0x21FF),  # arrows
    (0x2200, 0x22FF),  # mathematical operators
    (0x27C0, 0x27EF),  # misc mathematical symbols A
    (0x2A00, 0x2AFF),  # supplemental mathematical operators
)
# TeX Computer Modern math, AMS, STIX and the generic symbol fonts. A high-precision signal: these
# names appear when the page carries real mathematics, not when it merely uses a Greek letter.
_MATH_FONT_HINTS = ("cmmi", "cmsy", "cmex", "msam", "msbm", "stix", "mathjax", "math", "symbol", "esint", "wasy")
# Tesseract's invisible-text font, and the Adobe/ABBYY equivalents.
_GLYPHLESS_FONT_HINTS = ("glyphless", "notoserif-invisible")
# Encodings that map glyphs to Unicode on their own, so a font carrying one needs no ToUnicode.
_STANDARD_ENCODINGS = frozenset(
    {"WinAnsiEncoding", "MacRomanEncoding", "MacExpertEncoding", "StandardEncoding", "PDFDocEncoding"}
)

_CJK_RANGES = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF))
_RTL_RANGES = ((0x0590, 0x05FF), (0x0600, 0x06FF), (0x0700, 0x074F), (0x0750, 0x077F), (0xFB1D, 0xFDFF))

_WS = re.compile(r"\s+")
_NEWLINES = re.compile(r"\n+")
# TOC dot leaders and form rules, collapsed before the ratio checks so they do not read as garble.
_LEADERS = re.compile(r"(?:[.·•…_\-]\s*){3,}")

# A text line whose bbox overlaps this fraction of another line's counts as overlapping it.
_LINE_OVERLAP_MIN = 0.10
# Overlapping more than this many other lines makes a line "over-intersecting" -- one or two
# neighbours is ordinary inline math, superscripts or figure labels.
_LINE_OVERLAP_MAX_NEIGHBOURS = 2
# A line's bbox may extend this far past the page before it counts as out-of-page.
_PAGE_MARGIN = 5.0

# A drawing segment counts as a ruling line when it is this straight and this long relative to the
# page, which is what separates a table rule from a glyph outline or a chart's plot line.
_RULE_MAX_THICKNESS = 3.0
_RULE_MIN_LENGTH_RATIO = 0.10
# Ruling-line coordinates are rounded to this many points before distinct positions are counted, so
# that a grid drawn as many short segments reads as one row/column rather than dozens.
_RULE_POSITION_ROUND = 2.0

# Two spans are duplicates when they carry the same text within this distance, which is how
# fake-bold double rendering and doubled OCR layers show up.
_DUPLICATE_MAX_OFFSET = 1.5

# Left edges are bucketed at this width when measuring how tabular a page's text is.
_LEFT_EDGE_BUCKET = 6.0
# A column gap must be at least this wide, relative to the page, to split blocks into columns.
_COLUMN_MIN_GAP_RATIO = 0.04
# Blocks under this area fraction are page furniture and do not vote on the column layout.
_COLUMN_MIN_BLOCK_AREA = 0.001

# Below this many characters a page's ratios are noise, so its text-quality features are reported
# as "no evidence" (0.0) rather than as a confident extreme.
_MIN_CHARS_FOR_RATIOS = 40


@dataclass(frozen=True)
class PageSignals:
    """One sampled page's routing signals. Every field is a page-level scalar."""

    # encoding
    char_count: int
    fonts_total: int
    fonts_not_embedded: float
    fonts_without_tounicode: float
    fonts_unmappable: float
    fonts_type3: float
    fonts_glyphless: float
    replacement_ratio: float
    pua_ratio: float
    control_ratio: float
    ligature_ratio: float
    alphanum_ratio: float
    space_ratio: float
    newline_ratio: float
    single_char_token_ratio: float
    mean_token_length: float

    # layer
    invisible_char_ratio: float
    invisible_over_image_ratio: float
    text_over_image_ratio: float
    overlapping_line_ratio: float
    out_of_page_line_ratio: float
    duplicate_span_ratio: float
    rotated_char_ratio: float

    # math
    math_font_ratio: float
    math_unicode_ratio: float

    # structure
    ruling_line_count: int
    rule_grid_cells: int
    ruled_area_ratio: float
    left_edge_concentration: float

    # order
    text_block_count: int
    column_count: int
    stream_order_inversion_ratio: float
    interleaved_column_ratio: float

    # script
    cjk_ratio: float
    rtl_ratio: float
    latin_ratio: float


assert tuple(field.name for field in fields(PageSignals)) == PAGE_SIGNAL_NAMES, (
    "PageSignals fields and route_feature_names.PAGE_SIGNAL_NAMES must stay identical and in order: "
    "the second is what the driver and the trained booster agree on, and this module is what fills it"
)


@dataclass(frozen=True)
class DocumentSignals:
    """A document's routing signals: each page signal aggregated over the sampled pages.

    Aggregation is by mean and by max. A document routes on its worst page as much as on its
    average one -- a 40-page report with one page of equations is not 1/40th of a problem, because
    the extraction that page needs is not the one the other 39 need -- so both are carried and the
    model is left to decide which matters where.
    """

    pages_sampled: int
    page_count: int
    mean: dict[str, float]
    maximum: dict[str, float]

    def feature_vector(self) -> dict[str, float]:
        """The signals as a flat name-to-value mapping, for a parquet row or a model input."""
        row: dict[str, float] = {"pages_sampled": float(self.pages_sampled), "page_count": float(self.page_count)}
        for name in PAGE_SIGNAL_NAMES:
            row[f"mean_{name}"] = self.mean[name]
            row[f"max_{name}"] = self.maximum[name]
        return row


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= code <= high for low, high in ranges)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def _font_signals(page: pymupdf.Page, doc: pymupdf.Document) -> dict[str, float]:
    """Per-page font health, read from the resource list and the font objects it names.

    ``get_fonts`` entries are ``(xref, ext, type, basefont, name, encoding, referencer)``. A font
    MuPDF could not embed reports ``ext == "n/a"``; ToUnicode is read straight off the font
    dictionary, because a subsetted font without one produces text that looks fine and means
    nothing.
    """
    try:
        entries = page.get_fonts(full=True)
    except Exception:
        logger.debug("No fonts readable", exc_info=True)
        return {
            "fonts_total": 0,
            "fonts_not_embedded": 0.0,
            "fonts_without_tounicode": 0.0,
            "fonts_unmappable": 0.0,
            "fonts_type3": 0.0,
            "fonts_glyphless": 0.0,
            "math_font_ratio": 0.0,
        }

    total = len(entries)
    not_embedded = type3 = glyphless = math_fonts = without_tounicode = unmappable = 0
    for entry in entries:
        xref, extension, subtype, basefont, encoding = (
            entry[0],
            entry[1],
            entry[2],
            (entry[3] or "").lower(),
            entry[5] or "",
        )
        embedded = extension != "n/a"
        if not embedded:
            not_embedded += 1
        if subtype == "Type3":
            type3 += 1
        if any(hint in basefont for hint in _GLYPHLESS_FONT_HINTS):
            glyphless += 1
        if any(hint in basefont for hint in _MATH_FONT_HINTS):
            math_fonts += 1
        try:
            key, _ = doc.xref_get_key(xref, "ToUnicode")
        except Exception:
            key = "null"
        if key == "null":
            without_tounicode += 1
            # A missing ToUnicode is only dangerous when nothing else can decode the glyphs. A
            # base-14 or standard-encoded font maps through its encoding and reads fine; an
            # embedded subset with a custom or symbolic encoding and no ToUnicode is the case that
            # yields confident nonsense, and MuPDF will not flag it with U+FFFD.
            if embedded and encoding not in _STANDARD_ENCODINGS:
                unmappable += 1

    return {
        "fonts_total": total,
        "fonts_not_embedded": _ratio(not_embedded, total),
        "fonts_without_tounicode": _ratio(without_tounicode, total),
        "fonts_unmappable": _ratio(unmappable, total),
        "fonts_type3": _ratio(type3, total),
        "fonts_glyphless": _ratio(glyphless, total),
        "math_font_ratio": _ratio(math_fonts, total),
    }


def _character_signals(text: str) -> dict[str, float]:
    """Codepoint-level health of the page's extracted text.

    Below :data:`_MIN_CHARS_FOR_RATIOS` characters the ratios are reported as zero: a two-character
    page is not 100% garbled, it is silent, and a confident extreme there would drown the pages
    that carry evidence.
    """
    total = len(text)
    blank = {
        "char_count": total,
        "replacement_ratio": 0.0,
        "pua_ratio": 0.0,
        "control_ratio": 0.0,
        "ligature_ratio": 0.0,
        "alphanum_ratio": 0.0,
        "space_ratio": 0.0,
        "newline_ratio": 0.0,
        "single_char_token_ratio": 0.0,
        "mean_token_length": 0.0,
        "math_unicode_ratio": 0.0,
        "cjk_ratio": 0.0,
        "rtl_ratio": 0.0,
        "latin_ratio": 0.0,
    }
    if total < _MIN_CHARS_FOR_RATIOS:
        return blank

    replacement = pua = control = ligature = math_chars = cjk = rtl = latin = 0
    for character in text:
        code = ord(character)
        if code == _REPLACEMENT:
            replacement += 1
        elif _PUA[0] <= code <= _PUA[1]:
            pua += 1
        if character in _LIGATURES:
            ligature += 1
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t":
            control += 1
        if _in_ranges(code, _MATH_RANGES):
            math_chars += 1
        if _in_ranges(code, _CJK_RANGES):
            cjk += 1
        elif _in_ranges(code, _RTL_RANGES):
            rtl += 1
        elif 0x41 <= code <= 0x7A or 0xC0 <= code <= 0x24F:
            latin += 1

    collapsed = _LEADERS.sub(" ", text)
    spaces = sum(len(match.group(0)) for match in _WS.finditer(collapsed))
    non_space = len(_WS.sub("", collapsed))
    newlines = sum(len(match.group(0)) for match in _NEWLINES.finditer(collapsed))
    non_newline = len(_NEWLINES.sub("", collapsed))
    alphanumeric = sum(1 for character in collapsed if character.isalnum())

    tokens = collapsed.split()
    single = sum(1 for token in tokens if len(token) == 1)

    return blank | {
        "char_count": total,
        "replacement_ratio": _ratio(replacement, total),
        "pua_ratio": _ratio(pua, total),
        "control_ratio": _ratio(control, total),
        "ligature_ratio": _ratio(ligature, total),
        "alphanum_ratio": _ratio(alphanumeric, non_space),
        "space_ratio": _ratio(spaces, spaces + non_space),
        "newline_ratio": _ratio(newlines, newlines + non_newline),
        "single_char_token_ratio": _ratio(single, len(tokens)),
        "mean_token_length": _ratio(sum(len(token) for token in tokens), len(tokens)),
        "math_unicode_ratio": _ratio(math_chars, total),
        "cjk_ratio": _ratio(cjk, total),
        "rtl_ratio": _ratio(rtl, total),
        "latin_ratio": _ratio(latin, total),
    }


def _image_bboxes(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    """Where the page draws images, without decoding any of them."""
    try:
        return [tuple(image["bbox"]) for image in page.get_image_info()]
    except Exception:
        logger.debug("No image placements readable", exc_info=True)
        return []


def _layer_signals(spans: list[dict], images: list[tuple[float, float, float, float]]) -> dict[str, float]:
    """Whether the text layer sits where the ink is.

    The signal this exists for is ``invisible_over_image_ratio``: invisible text drawn on top of a
    bitmap is a scan carrying somebody else's OCR, whose quality is unknown and unknowable from the
    text-layer statistics that otherwise look healthy.
    """
    total_chars = sum(span["char_count"] for span in spans)
    invisible_chars = sum(span["char_count"] for span in spans if span["invisible"])
    rotated_chars = sum(span["char_count"] for span in spans if span["rotated"])

    def over_image(span: dict) -> bool:
        area = _bbox_area(span["bbox"])
        if area <= 0:
            return False
        covered = sum(_intersection_area(span["bbox"], image) for image in images)
        return covered / area > 0.5

    invisible_on_image = sum(span["char_count"] for span in spans if span["invisible"] and over_image(span))
    text_on_image = sum(span["char_count"] for span in spans if over_image(span))

    duplicates = 0
    by_text: dict[str, list[tuple[float, float]]] = {}
    for span in spans:
        if not span["text"].strip():
            continue
        placements = by_text.setdefault(span["text"], [])
        if any(
            abs(x - span["bbox"][0]) <= _DUPLICATE_MAX_OFFSET and abs(y - span["bbox"][1]) <= _DUPLICATE_MAX_OFFSET
            for x, y in placements
        ):
            duplicates += span["char_count"]
        placements.append((span["bbox"][0], span["bbox"][1]))

    return {
        "invisible_char_ratio": _ratio(invisible_chars, total_chars),
        "invisible_over_image_ratio": _ratio(invisible_on_image, total_chars),
        "text_over_image_ratio": _ratio(text_on_image, total_chars),
        "duplicate_span_ratio": _ratio(duplicates, total_chars),
        "rotated_char_ratio": _ratio(rotated_chars, total_chars),
    }


def _line_geometry_signals(lines: list[tuple[float, float, float, float]], page_rect: pymupdf.Rect) -> dict[str, float]:
    """Overlapping and out-of-page text lines, which mark a doubled or misplaced text layer.

    The overlap test is quadratic in the page's line count, so it is capped: beyond a few hundred
    lines the ratio is already determined and the pairs stop being informative.
    """
    if not lines:
        return {"overlapping_line_ratio": 0.0, "out_of_page_line_ratio": 0.0}

    outside = sum(
        1
        for bbox in lines
        if bbox[0] < page_rect.x0 - _PAGE_MARGIN
        or bbox[1] < page_rect.y0 - _PAGE_MARGIN
        or bbox[2] > page_rect.x1 + _PAGE_MARGIN
        or bbox[3] > page_rect.y1 + _PAGE_MARGIN
    )

    capped = lines[:400]
    over_intersecting = 0
    for index, bbox in enumerate(capped):
        area = _bbox_area(bbox)
        if area <= 0:
            continue
        neighbours = 0
        for other_index, other in enumerate(capped):
            if other_index == index:
                continue
            if _intersection_area(bbox, other) / area > _LINE_OVERLAP_MIN:
                neighbours += 1
                if neighbours > _LINE_OVERLAP_MAX_NEIGHBOURS:
                    over_intersecting += 1
                    break

    return {
        "overlapping_line_ratio": _ratio(over_intersecting, len(capped)),
        "out_of_page_line_ratio": _ratio(outside, len(lines)),
    }


def _structure_signals(page: pymupdf.Page, page_rect: pymupdf.Rect, left_edges: list[float]) -> dict[str, float]:
    """Ruling lines and the grid they imply, plus how aligned the page's left edges are.

    Ruling lines are the decode-free half of table detection: a drawn grid is what a table looks
    like to the content stream. Tables drawn without rules are why ``left_edge_concentration`` is
    here as well -- text repeatedly starting at the same few x positions is a column layout whether
    or not anything was drawn around it.
    """
    try:
        drawings = page.get_cdrawings()
    except Exception:
        logger.debug("No drawings readable", exc_info=True)
        drawings = []

    page_width, page_height = page_rect.width or 1.0, page_rect.height or 1.0
    horizontal: set[float] = set()
    vertical: set[float] = set()
    rules = 0
    ruled_x0 = ruled_y0 = math.inf
    ruled_x1 = ruled_y1 = -math.inf

    for path in drawings:
        rect = path.get("rect")
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        width, height = abs(x1 - x0), abs(y1 - y0)
        is_horizontal = height <= _RULE_MAX_THICKNESS and width >= page_width * _RULE_MIN_LENGTH_RATIO
        is_vertical = width <= _RULE_MAX_THICKNESS and height >= page_height * _RULE_MIN_LENGTH_RATIO
        if not (is_horizontal or is_vertical):
            continue
        rules += 1
        if is_horizontal:
            horizontal.add(round(y0 / _RULE_POSITION_ROUND))
        else:
            vertical.add(round(x0 / _RULE_POSITION_ROUND))
        ruled_x0, ruled_y0 = min(ruled_x0, x0), min(ruled_y0, y0)
        ruled_x1, ruled_y1 = max(ruled_x1, x1), max(ruled_y1, y1)

    ruled_area = 0.0
    if rules:
        ruled_area = _ratio(max(ruled_x1 - ruled_x0, 0.0) * max(ruled_y1 - ruled_y0, 0.0), page_width * page_height)

    buckets = Counter(round(edge / _LEFT_EDGE_BUCKET) for edge in left_edges)
    concentration = _ratio(sum(count for count in buckets.values() if count >= 3), len(left_edges))

    return {
        "ruling_line_count": rules,
        # A grid needs both directions; one axis alone is a rule under a heading, not a table.
        "rule_grid_cells": max(len(horizontal) - 1, 0) * max(len(vertical) - 1, 0),
        "ruled_area_ratio": min(ruled_area, 1.0),
        "left_edge_concentration": concentration,
    }


def _columns(blocks: list[dict], page_rect: pymupdf.Rect) -> list[tuple[float, float]]:
    """Split the page into column x-intervals by sweeping for gaps no text block crosses."""
    page_area = (page_rect.width or 1.0) * (page_rect.height or 1.0)
    spans = sorted(
        (block["bbox"][0], block["bbox"][2])
        for block in blocks
        if _bbox_area(block["bbox"]) / page_area >= _COLUMN_MIN_BLOCK_AREA
    )
    if not spans:
        return []

    minimum_gap = (page_rect.width or 1.0) * _COLUMN_MIN_GAP_RATIO
    columns: list[tuple[float, float]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = columns[-1]
        if start - last_end > minimum_gap:
            columns.append((start, end))
        else:
            columns[-1] = (last_start, max(last_end, end))
    return columns


def _order_signals(blocks: list[dict], page_rect: pymupdf.Rect) -> dict[str, float]:
    """How far the content stream's block order is from a column-aware reading order.

    Docling orders blocks by its layout model; the VLM orders by what it sees; Marker orders by the
    content stream and measured that as the better of the three on multi-column pages. All three
    agree on a single-column page laid out in stream order, and the routes are free to disagree
    exactly where these do -- so the disagreement itself is the feature.

    ``stream_order_inversion_ratio`` is the fraction of block pairs whose stream order and
    geometric reading order disagree, which is Kendall's tau rescaled to [0, 1].
    ``interleaved_column_ratio`` counts stream adjacencies that jump between columns, which is what
    a spliced two-column read looks like before it becomes an inversion.
    """
    if len(blocks) < 2:
        return {"column_count": 1 if blocks else 0, "stream_order_inversion_ratio": 0.0, "interleaved_column_ratio": 0.0}

    columns = _columns(blocks, page_rect)

    def column_of(block: dict) -> int:
        centre = (block["bbox"][0] + block["bbox"][2]) / 2
        for index, (start, end) in enumerate(columns):
            if start <= centre <= end:
                return index
        return len(columns)

    # The reading order a human would use: column by column, top to bottom inside each.
    geometric = sorted(
        range(len(blocks)), key=lambda i: (column_of(blocks[i]), blocks[i]["bbox"][1], blocks[i]["bbox"][0])
    )
    rank = {block_index: position for position, block_index in enumerate(geometric)}

    inversions = 0
    pairs = 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            pairs += 1
            if rank[i] > rank[j]:
                inversions += 1

    stream_columns = [column_of(block) for block in blocks]
    jumps = sum(1 for a, b in pairwise(stream_columns) if a != b)

    return {
        "column_count": max(len(columns), 1),
        "stream_order_inversion_ratio": _ratio(inversions, pairs),
        "interleaved_column_ratio": _ratio(jumps, len(blocks) - 1),
    }


def _read_spans(page: pymupdf.Page) -> tuple[list[dict], list[tuple[float, float, float, float]], list[float]]:
    """One text pass: per-span text, geometry and visibility, plus line bboxes and left edges.

    ``rawdict`` is used rather than ``dict`` because only it reports the per-character detail that
    makes an accurate character count possible on spans MuPDF splits mid-word; the flags come off
    the span either way. This is the single most expensive call in the module and everything
    text-derived is computed from its result.
    """
    text_page = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
    spans: list[dict] = []
    lines: list[tuple[float, float, float, float]] = []
    left_edges: list[float] = []

    for block in text_page.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            lines.append(tuple(line["bbox"]))
            left_edges.append(line["bbox"][0])
            direction = line.get("dir", (1.0, 0.0))
            rotated = abs(direction[1]) > 0.01
            for span in line.get("spans", ()):
                characters = span.get("chars", ())
                spans.append(
                    {
                        "text": "".join(character["c"] for character in characters),
                        "char_count": len(characters),
                        "bbox": tuple(span["bbox"]),
                        # MuPDF render mode 3 is invisible; a zero-alpha span is invisible too.
                        "invisible": span.get("type") == 3 or span.get("alpha", 255) == 0,
                        "rotated": rotated,
                    }
                )
    return spans, lines, left_edges


def _stream_order_blocks(page: pymupdf.Page) -> list[dict]:
    """Text blocks in content-stream order, which is what the order signals compare against.

    ``get_text("blocks")`` yields ``(x0, y0, x1, y1, text, block_number, block_type)`` and leaves
    them in stream order unless asked to sort; type 0 is text, type 1 is an image.
    """
    return [
        {"bbox": tuple(block[:4])}
        for block in page.get_text("blocks", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        if block[6] == 0
    ]


def page_signals(doc: pymupdf.Document, page: pymupdf.Page) -> PageSignals:
    """Compute every routing signal for one page."""
    spans, lines, left_edges = _read_spans(page)
    text = "".join(span["text"] for span in spans)
    blocks = _stream_order_blocks(page)
    page_rect = page.rect

    values: dict[str, float] = {}
    values.update(_character_signals(text))
    values.update(_font_signals(page, doc))
    values.update(_layer_signals(spans, _image_bboxes(page)))
    values.update(_line_geometry_signals(lines, page_rect))
    values.update(_structure_signals(page, page_rect, left_edges))
    values.update(_order_signals(blocks, page_rect))
    values["text_block_count"] = len(blocks)
    return PageSignals(**{name: values[name] for name in PAGE_SIGNAL_NAMES})


def document_signals(doc: pymupdf.Document, page_indices: list[int]) -> DocumentSignals:
    """Compute routing signals for a document from its sampled pages.

    A page that cannot be read is skipped rather than raising: an unreadable page is itself weak
    evidence for OCR, and the sampled-page count records how much of the document was actually
    seen. A document where no page could be read returns zeroed signals with ``pages_sampled`` 0,
    which is the caller's signal to route it to OCR unconditionally.
    """
    collected: list[PageSignals] = []
    for index in page_indices:
        try:
            collected.append(page_signals(doc, doc.load_page(index)))
        except Exception:
            logger.debug("Skipping unreadable page %d", index, exc_info=True)

    if not collected:
        zeros = dict.fromkeys(PAGE_SIGNAL_NAMES, 0.0)
        return DocumentSignals(pages_sampled=0, page_count=len(doc), mean=zeros, maximum=dict(zeros))

    mean = {name: sum(float(getattr(page, name)) for page in collected) / len(collected) for name in PAGE_SIGNAL_NAMES}
    maximum = {name: max(float(getattr(page, name)) for page in collected) for name in PAGE_SIGNAL_NAMES}
    return DocumentSignals(pages_sampled=len(collected), page_count=len(doc), mean=mean, maximum=maximum)


def sample_page_indices(num_pages: int, rng: random.Random) -> list[int]:
    """Draw a sorted sample of at most :data:`SAMPLE_PAGES` distinct page indices."""
    return sorted(rng.sample(range(num_pages), min(SAMPLE_PAGES, num_pages)))


def signals_for_routing(doc: pymupdf.Document, *, seed: int) -> DocumentSignals:
    """Sample a document's pages and compute its routing signals.

    Seeded per document by the caller, so re-running a shard reproduces its routing decisions
    exactly rather than re-drawing pages and re-deciding.

    Raises:
        CorruptPdf: the document is encrypted, has no pages, or MuPDF reported structural damage
            while the sampled pages were read.
    """
    if doc.is_encrypted or doc.needs_pass:
        raise CorruptPdf("document is encrypted or password-protected")
    if len(doc) == 0:
        raise CorruptPdf("document has no pages")

    # MuPDF reports structural damage as warnings while pages are read rather than as exceptions,
    # so the store is drained first and inspected once the sampled pages have been touched.
    pymupdf.TOOLS.reset_mupdf_warnings()
    signals = document_signals(doc, sample_page_indices(len(doc), random.Random(seed)))
    corruption = _CORRUPTION_RE.search(pymupdf.TOOLS.mupdf_warnings())
    if corruption:
        raise CorruptPdf(f"malformed document: {corruption.group(0)}")
    if signals.pages_sampled == 0:
        raise CorruptPdf("no sampled page could be read")
    return signals
