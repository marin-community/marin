# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a layout cluster's text from its spans, instead of from its lines.

This is the substantive half of the FinePDFs docling fork. Docling's own assembler joins a
cluster's cells by stripping each and concatenating with a space, which is right when a cell is a
line and wrong when it is a span: PDF producers split a single word across spans whenever the font,
size, colour, or rise changes, so ``dif`` + ``ficult`` becomes ``dif ficult`` and every italicised
word inside a sentence gains two spaces. Here a space is inserted only when the geometry says there
is one -- a horizontal gap wider than a quarter of the cluster's median glyph advance, a line break,
a superscript boundary, or whitespace already present in the span text.

Three more repairs ride along, all from the FinePDFs fork:

* letter-spaced headings (``A N N U A L  R E P O R T``, produced by tracking applied as real
  spaces) are collapsed back to words when a single-span header matches that shape;
* a hyphen at a line break is dropped and the word rejoined, but only when both sides are
  alphanumeric, so ``anti-`` + ``war`` survives as ``anti-war`` while ``con-`` + ``tinues`` becomes
  ``continues``;
* duplicated spans -- the same text drawn twice at the same place, which is how some producers fake
  bold -- are emitted once.

Two measurements are recorded per cluster for the postprocessors downstream: the median glyph
advance, and the bounding box of the cluster's last line. See :mod:`.fields` for why they need
somewhere to live, and :mod:`.reading_order` for how they reach the assembled document.

Docling's own ligature expansion and hyperlink matching are kept; the FinePDFs fork predates both.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from docling.datamodel.base_models import (
    AssembledUnit,
    ContainerElement,
    FigureElement,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.stages.layout.layout_model import LayoutModel
from docling.models.stages.page_assemble.page_assemble_model import PageAssembleModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import TextCell

from experiments.build_pdf_source.docling_extract.fields import patch_docling_models
from experiments.build_pdf_source.docling_extract.tables import TableReader

logger = logging.getLogger(__name__)

# Characters PDF producers emit that are not what they appear to be. U+0002 is the classic
# "hyphen drawn from a symbol font" case docling also special-cases.
_FAULTY_CHARACTERS = {
    "⁄": "/",  # noqa: RUF001 -- fraction slash, the character being corrected
    "‘": "'",  # noqa: RUF001
    "’": "'",  # noqa: RUF001
    "“": '"',
    "”": '"',
    "": "-",
}
_FAULTY_CHARACTER_RE = re.compile("|".join(map(re.escape, _FAULTY_CHARACTERS)))

# A letter-spaced heading: four or more non-lowercase, non-space glyphs each followed by spaces.
_NOT_SPACE_OR_LOWER = "[^ \t\n\r\f\va-z]"
_BROKEN_HEADER_RE = re.compile(rf"^({_NOT_SPACE_OR_LOWER} +){{3,}}{_NOT_SPACE_OR_LOWER}$")
_BROKEN_HEADER_LABELS = frozenset(
    {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
)

# Fraction of the shorter span's height that two spans must share vertically to count as one line.
_SAME_LINE_OVERLAP = 0.8
# Horizontal gap, in median glyph advances, above which a space is inserted.
_SPACE_GAP_RATIO = 0.25
# Fraction of the narrower span's width two spans must share to count as drawn on top of each other.
_DUPLICATE_OVERLAP = 0.8
# Bit 0 of PyMuPDF's span flags word marks a superscript.
_SUPERSCRIPT_FLAG = 1


@dataclass(frozen=True)
class ClusterText:
    """A layout cluster's text, with the geometry the postprocessors measure distances against."""

    text: str
    median_char_width: float | None
    last_line_bbox: BoundingBox | None


def replace_faulty_characters(text: str) -> str:
    """Map the characters PDF producers substitute for ASCII back to ASCII."""
    return _FAULTY_CHARACTER_RE.sub(lambda match: _FAULTY_CHARACTERS[match.group()], text)


def ends_with_whitespace(text: str) -> bool:
    return text.endswith((" ", "\n", "\t", "\r"))


def starts_with_whitespace(text: str) -> bool:
    return text.startswith((" ", "\n", "\t", "\r"))


def is_letter_spaced_heading(text: str) -> bool:
    """Whether a heading was typeset with tracking rendered as literal spaces (``H E L L O``)."""
    return bool(_BROKEN_HEADER_RE.fullmatch(text.strip()))


def collapse_letter_spacing(text: str) -> str:
    """Turn ``A N N U A L  R E P O R T`` back into ``ANNUAL REPORT``.

    Single spaces are letter spacing and go; runs of two or more are word boundaries and collapse
    to one. Applied only where :func:`is_letter_spaced_heading` holds, so ordinary prose is safe.
    """
    return re.sub(r" +", " ", re.sub(r" (?! )", "", text))


def _vertical_overlap(first: TextCell, second: TextCell) -> float:
    """Shared vertical extent of two spans, as a fraction of the shorter one's height."""
    overlap = max(0.0, min(first.rect.r_y2, second.rect.r_y2) - max(first.rect.r_y0, second.rect.r_y0))
    shortest = min(first.rect.r_y2 - first.rect.r_y0, second.rect.r_y2 - second.rect.r_y0)
    return overlap / shortest if shortest > 0 else 0.0


def _horizontally_overlapping(first: TextCell, second: TextCell) -> bool:
    """Whether two spans are drawn over each other -- how some producers fake a bold weight."""
    shared = min(first.rect.r_x1, second.rect.r_x1) - max(first.rect.r_x0, second.rect.r_x0)
    narrowest = min(first.rect.r_x1 - first.rect.r_x0, second.rect.r_x1 - second.rect.r_x0)
    return narrowest > 0 and shared / narrowest > _DUPLICATE_OVERLAP


def _is_superscript(cell: TextCell) -> bool:
    """Whether a span is a superscript, per PyMuPDF's font flags.

    Cells that did not come from :mod:`.backend` -- OCR output, for instance -- carry no span
    record, and are treated as ordinary text.
    """
    return bool(cell.info.get("flags", 0) & _SUPERSCRIPT_FLAG)


def _median_char_width(cells: Iterable[TextCell]) -> float:
    """Median glyph advance over a cluster, weighted by the characters each span contributes.

    Weighting by character count rather than by span keeps a one-character span in a display font
    from dragging the median away from the body text that surrounds it.
    """
    widths = [(cell.rect.r_x1 - cell.rect.r_x0) / len(cell.text) for cell in cells for _ in range(len(cell.text))]
    return float(np.median(widths))


def join_cluster_cells(cells: list[TextCell]) -> ClusterText:
    """Join a cluster's spans into text, inserting a space only where the geometry implies one."""
    cells = [cell for cell in cells if cell.text != ""]
    if not cells:
        return ClusterText(text="", median_char_width=None, last_line_bbox=None)

    median_char_width = _median_char_width(cells)
    last_line_bbox = cells[-1].info.get("line_bbox")

    if len(cells) == 1:
        text = replace_faulty_characters(cells[0].text.strip())
        return ClusterText(text=text, median_char_width=median_char_width, last_line_bbox=last_line_bbox)

    joined = cells[0].text.strip()
    previous = cells[0]
    for cell in cells[1:]:
        current = cell.text.strip()

        if previous.text == cell.text and _horizontally_overlapping(previous, cell):
            continue

        overlap = _vertical_overlap(previous, cell)
        on_same_line = overlap > _SAME_LINE_OVERLAP

        if joined.endswith("-") and not on_same_line:
            joined = _join_across_hyphen(joined, current)
        elif _needs_space(joined, current, previous, cell, median_char_width, on_same_line):
            joined += " " + current
        else:
            joined += current

        previous = cell

    return ClusterText(
        text=replace_faulty_characters(joined).strip(),
        median_char_width=median_char_width,
        last_line_bbox=last_line_bbox,
    )


def _join_across_hyphen(joined: str, current: str) -> str:
    """Drop a line-break hyphen when it split a word, keep it when it is a real hyphen."""
    previous_words = re.findall(r"\b[\w]+\b", joined)
    current_words = re.findall(r"\b[\w]+\b", current)
    if previous_words and current_words and previous_words[-1].isalnum() and current_words[0].isalnum():
        return joined[:-1] + current
    if ends_with_whitespace(joined):
        return joined + current
    return joined + " " + current


def _needs_space(
    joined: str,
    current: str,
    previous: TextCell,
    cell: TextCell,
    median_char_width: float,
    on_same_line: bool,
) -> bool:
    """Whether a space belongs between two spans.

    A space is implied by a wide enough horizontal gap, a line break, a superscript boundary, or
    whitespace the producer already put in the span text -- but never when the joined text or the
    incoming span already supplies one.
    """
    if ends_with_whitespace(joined) or starts_with_whitespace(current):
        return False
    gap = abs(cell.rect.r_x0 - previous.rect.r_x1)
    return (
        gap > median_char_width * _SPACE_GAP_RATIO
        or ends_with_whitespace(previous.text)
        or starts_with_whitespace(cell.text)
        or not on_same_line
        or _is_superscript(cell)
        or _is_superscript(previous)
    )


class SpanAwarePageAssembleModel(PageAssembleModel):
    """:class:`PageAssembleModel` with span-aware text joining and a pluggable table reader.

    Subclassed rather than copied so upstream's cluster iteration, container and figure handling,
    and hyperlink matching stay live; only text and table assembly are replaced.
    """

    def __init__(self, options, table_reader: TableReader | None = None):
        patch_docling_models()
        super().__init__(options)
        self.table_reader = table_reader

    def _assemble_text(self, cluster, page: Page) -> TextElement:
        assembled = join_cluster_cells(list(cluster.cells))
        text = assembled.text
        # Tracking is only distinguishable from real spacing in a heading drawn as one span; in
        # body text the same pattern is legitimately spaced initials or a table rendered as text.
        if cluster.label in _BROKEN_HEADER_LABELS and len(cluster.cells) == 1 and is_letter_spaced_heading(text):
            text = collapse_letter_spacing(text)
        return TextElement(
            label=cluster.label,
            id=cluster.id,
            text=self.sanitize_text([text]),
            hyperlink=self._match_hyperlink(cluster.bbox, page),
            page_no=page.page_no,
            cluster=cluster,
            media_char_width=assembled.median_char_width,
            last_line_bbox=assembled.last_line_bbox,
        )

    def _assemble_table(self, cluster, page: Page) -> Table:
        table = None
        if page.predictions.tablestructure:
            table = page.predictions.tablestructure.table_map.get(cluster.id, None)
        if not table:
            table = Table(
                label=cluster.label,
                id=cluster.id,
                text="",
                otsl_seq=[],
                table_cells=[],
                cluster=cluster,
                page_no=page.page_no,
            )
        if self.table_reader is not None:
            self.table_reader.fill(table=table, cluster=cluster, page=page)
        return table

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "page_assemble"):
                assert page.predictions.layout is not None
                elements: list[PageElement] = []
                headers: list[PageElement] = []
                body: list[PageElement] = []

                for cluster in page.predictions.layout.clusters:
                    if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                        element = self._assemble_text(cluster, page)
                        elements.append(element)
                        if cluster.label in LayoutModel.PAGE_HEADER_LABELS:
                            headers.append(element)
                        else:
                            body.append(element)
                    elif cluster.label in LayoutModel.TABLE_LABELS:
                        table = self._assemble_table(cluster, page)
                        elements.append(table)
                        body.append(table)
                    elif cluster.label == LayoutModel.FIGURE_LABEL:
                        figure = None
                        if page.predictions.figures_classification:
                            figure = page.predictions.figures_classification.figure_map.get(cluster.id, None)
                        if not figure:
                            figure = FigureElement(
                                label=cluster.label,
                                id=cluster.id,
                                text="",
                                data=None,
                                cluster=cluster,
                                page_no=page.page_no,
                            )
                        elements.append(figure)
                        body.append(figure)
                    elif cluster.label in LayoutModel.CONTAINER_LABELS:
                        container = ContainerElement(
                            label=cluster.label,
                            id=cluster.id,
                            page_no=page.page_no,
                            cluster=cluster,
                        )
                        elements.append(container)
                        body.append(container)

                page.assembled = AssembledUnit(elements=elements, headers=headers, body=body)

            yield page
