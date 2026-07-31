# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repairs applied to the assembled document, before it is serialised to text.

The layout model emits one item per detected region, and regions are not sentences. Four passes run
over the result, in this order:

1. :class:`PageNumberRemover` drops running page numbers.
2. :class:`SpanMerger` rejoins items the layout model split *across* a line -- a drop cap, an
   inline formula, a footnote marker, a differently-styled run mid-sentence.
3. :class:`ParagraphMerger` rejoins items split *between* lines, where one paragraph was detected
   as two blocks, including across a page boundary.
4. :class:`ListMarkerNormalizer` maps the many bullet glyphs onto ``-``, ``*`` and ``[x]``.

Both mergers measure in characters rather than in points, using the median glyph advance
:mod:`.assemble` recorded, because a six-point gap is a word space in a footnote and a column
boundary in a title. Neither will merge across a label change, so a heading is never absorbed into
the paragraph beneath it.

FinePDFs runs the page-number pass last, after both mergers. Here it runs first: a page number
sitting between two blocks of body text is exactly the kind of thing the paragraph merger will
otherwise absorb into a sentence, and it is cheaper to drop it than to teach the merger to skip it.
Upstream's own guard for this -- ``ParagraphMerger`` skipping ``PageNumberItem`` -- cannot fire in
their ordering, because nothing has been converted to a ``PageNumberItem`` yet when it runs.
"""

import logging
import re

from docling_core.types.doc import DocItemLabel, DoclingDocument, ListItem, ProvenanceItem, TextItem

from experiments.build_pdf_source.docling_extract.assemble import (
    ends_with_whitespace,
    starts_with_whitespace,
)
from experiments.build_pdf_source.docling_extract.page_numbers import is_page_number

logger = logging.getLogger(__name__)

# Punctuation that ends a line of prose. A block ending in one of these is probably complete.
_SENTENCE_END = (".", "!", "?", "。", "！", "？", ")", "）", '"', "”", ":", "：", ";", "；")  # noqa: RUF001
# Punctuation that cannot begin a block, so a block starting with one continues the block before it
# even across a wider gap than usual -- this is how a footnote marker or a stray bracket reattaches.
_CANNOT_START_A_BLOCK = (".", "!", "?", "。", "！", "？", ")", "）", ":", "：", ";", "；")  # noqa: RUF001

# Labels whose items may be joined to an adjacent item of the same label on the same line.
_SPAN_MERGE_LABELS = frozenset(
    {
        DocItemLabel.TEXT,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.TITLE,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
    }
)
# Which label may follow which when merging paragraphs. A list item may absorb the text that
# continues it; body text may absorb body text; nothing else joins.
_PARAGRAPH_MERGE_LABELS = {
    DocItemLabel.LIST_ITEM: (DocItemLabel.TEXT,),
    DocItemLabel.TEXT: (DocItemLabel.TEXT,),
}
# Labels a page number can plausibly have been detected as.
_PAGE_NUMBER_LABELS = frozenset(
    {DocItemLabel.TITLE, DocItemLabel.FOOTNOTE, DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER}
)

# Same line: this much of the shorter block's height shared with the previous block's last line.
_SAME_LINE_OVERLAP = 0.8
# Gap, in characters, within which two items on one line are joined unconditionally.
_SPAN_MERGE_GAP = 6
# Wider gap tolerated when the second item starts with punctuation that cannot begin a block.
_SPAN_MERGE_PUNCTUATION_GAP = 30
# Gap, in characters, above which a space is inserted between two joined items.
_SPAN_SPACE_GAP = 0.25
# Two blocks belong to one paragraph only if their glyph advances agree this closely.
_CHAR_WIDTH_TOLERANCE = 0.1
# ... their left and right edges line up within this many characters ...
_EDGE_TOLERANCE = 5
# ... the previous block's last line reaches its right edge within this many characters (a short
# last line means the paragraph ended) ...
_LAST_LINE_TOLERANCE = 2
# ... and, on one page, they are within this many characters of each other vertically.
_VERTICAL_TOLERANCE = 1


def merge_items(item: TextItem, absorbed: TextItem, separator: str) -> TextItem:
    """Append ``absorbed`` to ``item``, rebasing its provenance onto the joined text."""
    offset = len(item.text) + len(separator)
    for provenance in absorbed.prov:
        provenance.charspan = (offset + provenance.charspan[0], offset + provenance.charspan[1])
        provenance.page_no = item.prov[-1].page_no
    item.text += f"{separator}{absorbed.text}"
    item.prov.extend(absorbed.prov)
    return item


def _distance_in_characters(first: ProvenanceItem, second: ProvenanceItem) -> float:
    """Horizontal gap between two items, in median glyph advances of the first."""
    width = first.media_char_width
    if not width or width <= 0:
        return float("inf")
    return abs((second.bbox.l - first.bbox.r) / width)


def _on_same_line(first: ProvenanceItem, second: ProvenanceItem) -> bool:
    """Whether ``second`` sits on the last line of ``first``.

    Measured against the previous item's *last line* rather than its whole box, so a two-line
    paragraph does not swallow whatever sits beside its first line.
    """
    if first.last_line_bbox is None:
        return False
    overlap = max(0.0, min(first.last_line_bbox.t, second.bbox.t) - max(first.last_line_bbox.b, second.bbox.b))
    tallest = max(first.last_line_bbox.height, second.bbox.height)
    return (overlap / tallest if tallest > 0 else 0.0) >= _SAME_LINE_OVERLAP


class PageNumberRemover:
    """Drops running page numbers from the first and last text block of each page."""

    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        removed: list[TextItem] = []
        for page in range(doc.num_pages()):
            text_items = [
                item for item, _ in doc.iterate_items(page_no=page + 1, with_groups=False) if isinstance(item, TextItem)
            ]
            if not text_items:
                continue
            # Only the extremities of a page: a number in the middle of a page is content.
            candidates = {id(text_items[0]): text_items[0], id(text_items[-1]): text_items[-1]}
            removed.extend(
                item for item in candidates.values() if item.label in _PAGE_NUMBER_LABELS and is_page_number(item.text)
            )

        if removed:
            doc.delete_items(node_items=removed)
            logger.debug("Removed %d page numbers", len(removed))
        return doc


class SpanMerger:
    """Rejoins items the layout model split across a single line."""

    def process_document(self, doc: DoclingDocument, allow_multi_prov: bool = True) -> DoclingDocument:
        merged: list[TextItem] = []
        # Never across a page boundary: two items on "the same line" of different pages are not.
        for page in range(doc.num_pages()):
            previous: TextItem | None = None
            previous_level = 0
            for item, level in doc.iterate_items(page_no=page + 1, with_groups=False):
                is_text = isinstance(item, TextItem)
                mergeable = is_text and (len(item.prov) == 1 or allow_multi_prov)
                if (
                    previous is None
                    or not is_text
                    or not mergeable
                    or level != previous_level
                    or item.label != previous.label
                    or item.label not in _SPAN_MERGE_LABELS
                ):
                    previous = item if mergeable else None
                    previous_level = level
                    continue
                previous_level = level

                text = item.text.strip() + (" " if ends_with_whitespace(item.text) else "")
                distance = _distance_in_characters(previous.prov[-1], item.prov[0])
                close_enough = distance < _SPAN_MERGE_GAP or (
                    text.startswith(_CANNOT_START_A_BLOCK) and distance < _SPAN_MERGE_PUNCTUATION_GAP
                )
                if not (_on_same_line(previous.prov[-1], item.prov[0]) and close_enough):
                    previous = item
                    continue

                needs_space = (
                    (distance > _SPAN_SPACE_GAP or previous.text[-1:] in _SENTENCE_END)
                    and not text.startswith(_CANNOT_START_A_BLOCK)
                    and not ends_with_whitespace(previous.text)
                    and not starts_with_whitespace(item.text)
                )
                previous = merge_items(previous, item, " " if needs_space else "")
                merged.append(item)

        if merged:
            doc.delete_items(node_items=merged)
            logger.debug("Merged %d items into their line", len(merged))
        return doc


def _continues_paragraph(first: TextItem, second: TextItem) -> bool:
    """Whether ``second`` is the continuation of the paragraph ``first`` began.

    Every test is a reason to say no. The two blocks must be set in the same size, share left and
    right edges, sit directly beneath one another, and the first must run its last line out to the
    right margin -- a short last line is how a paragraph ends. Then the text has to agree: no
    terminal punctuation on the first, and the second must not open with a capital or a digit,
    which is how a new sentence or a numbered item starts.
    """
    first_box = first.prov[-1].bbox
    second_box = second.prov[0].bbox
    first_width = first.prov[-1].media_char_width
    second_width = second.prov[0].media_char_width

    if not first_width or not second_width:
        return False
    if abs(first_width - second_width) / max(first_width, second_width) > _CHAR_WIDTH_TOLERANCE:
        return False

    last_line = first.prov[-1].last_line_bbox
    if last_line is None:
        return False

    if (
        abs(first_box.l - second_box.l) / second_width > _EDGE_TOLERANCE
        or abs(first_box.r - second_box.r) / second_width > _EDGE_TOLERANCE
    ):
        return False
    if (
        first.prov[-1].page_no == second.prov[0].page_no
        and abs(first_box.b - second_box.t) / second_width > _VERTICAL_TOLERANCE
    ):
        return False
    if abs(first_box.r - last_line.r) / second_width >= _LAST_LINE_TOLERANCE:
        return False

    first_text = first.text.strip()
    second_text = second.text.strip()
    if not first_text or not second_text:
        return False
    return not first_text.endswith(_SENTENCE_END) and not second_text[0].isdigit() and not second_text[0].isupper()


class ParagraphMerger:
    """Rejoins paragraphs the layout model split into separate blocks, including across pages."""

    def process_document(self, doc: DoclingDocument, allow_multi_prov: bool = True) -> DoclingDocument:
        merged: list[TextItem] = []
        previous: TextItem | None = None
        previous_level = 0

        # Not reset per page: a paragraph broken by a page break is the case this exists for.
        for page in range(doc.num_pages()):
            for item, level in doc.iterate_items(page_no=page + 1, with_groups=False):
                is_text = isinstance(item, TextItem)
                mergeable = is_text and (len(item.prov) == 1 or allow_multi_prov)
                if (
                    previous is None
                    or not is_text
                    or not mergeable
                    or level != previous_level
                    or item.label not in _PARAGRAPH_MERGE_LABELS.get(previous.label, ())
                ):
                    previous = item if mergeable else None
                    previous_level = level
                    continue
                previous_level = level

                if not _continues_paragraph(previous, item):
                    previous = item
                    continue

                previous = merge_items(previous, item, _paragraph_separator(previous, item))
                merged.append(item)

        if merged:
            doc.delete_items(node_items=merged)
            logger.debug("Merged %d blocks into their paragraph", len(merged))
        return doc


def _paragraph_separator(first: TextItem, second: TextItem) -> str:
    """The join between two halves of a paragraph, unhyphenating the word they may share."""
    if first.text.endswith("-"):
        first_words = re.findall(r"\b[\w]+\b", first.text)
        second_words = re.findall(r"\b[\w]+\b", second.text)
        if first_words and second_words and first_words[-1].isalnum() and second_words[0].isalnum():
            first.text = first.text[:-1]
            start, end = first.prov[-1].charspan
            first.prov[-1].charspan = (start, end - 1)
            return ""
    if ends_with_whitespace(first.text) or starts_with_whitespace(second.text):
        return ""
    return " "


class ListMarkerNormalizer:
    """Maps the bullet, arrow and checkmark glyphs PDFs use onto a small ASCII set."""

    # Symbol fonts put bullets in the private use area, so the U+F0xx code points are as common as
    # the real ones. Arrows and squares are used interchangeably with bullets and normalise to "-".
    # U+25CF BLACK CIRCLE is in both sets on purpose: FinePDFs lists it in both, and stars are
    # substituted before dashes, so it resolves to "*". Dropping it from either set here would
    # silently diverge from the corpus this extractor reproduces.
    _DASH_SYMBOLS = ("‣", "⁃", "►", "▶", "▸", "➤", "➢", "›", "▪", "▫", "●", "○", "", "")  # noqa: RUF001
    _STAR_SYMBOLS = ("•", "·", "°", "◌", "∙", "◦", "●")
    _CHECK_SYMBOLS = ("✓", "✔", "✗", "✘", "")

    def __init__(self):
        dashes = "|".join(rf"(?:\s*[{re.escape(symbol)}-](?:\s+{re.escape(symbol)})*)" for symbol in self._DASH_SYMBOLS)
        self._dash_re = re.compile(f"^({dashes})")
        self._star_re = re.compile("^(" + "|".join(rf"\s*{re.escape(s)}" for s in self._STAR_SYMBOLS) + ")")
        self._check_re = re.compile("^(" + "|".join(rf"\s*{re.escape(s)}" for s in self._CHECK_SYMBOLS) + ")")
        self._leading_whitespace_re = re.compile(r"\s+")

    def _normalize(self, text: str, is_list_item: bool) -> str:
        text = self._check_re.sub("[x]", text)
        text = self._star_re.sub("*", text)
        text = self._dash_re.sub("-", text)
        if is_list_item:
            # Collapse only the run after the marker; indentation further in may be meaningful.
            text = self._leading_whitespace_re.sub(" ", text, count=1)
        return text

    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        for item, _ in doc.iterate_items(with_groups=False):
            if not isinstance(item, TextItem):
                continue
            # Docling lifts a recognised bullet out of the item's text into its own ``marker``
            # field, so the glyph to normalise is there and not in the text at all. Items whose
            # bullet was not recognised still carry it inline, so both are handled.
            if isinstance(item, ListItem) and item.marker:
                item.marker = self._normalize(item.marker, is_list_item=False).strip()

            normalized = self._normalize(item.text, item.label == DocItemLabel.LIST_ITEM)
            if normalized == item.text:
                continue
            # Only the head of the string changes, so every span shifts by the same amount.
            shift = len(item.text) - len(normalized)
            for provenance in item.prov:
                provenance.charspan = (provenance.charspan[0] + shift, provenance.charspan[1])
            item.text = normalized
        return doc


def postprocess_document(doc: DoclingDocument, *, fix_page_numbers: bool = True) -> DoclingDocument:
    """Run every repair pass over ``doc``, in order, in place."""
    if fix_page_numbers:
        doc = PageNumberRemover().process_document(doc)
    doc = SpanMerger().process_document(doc)
    doc = ParagraphMerger().process_document(doc)
    return ListMarkerNormalizer().process_document(doc)
