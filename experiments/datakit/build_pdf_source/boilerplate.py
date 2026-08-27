# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strip running headers and footers from a paginated document.

A PDF repeats its chrome on every page: a journal name, a document number, a confidentiality
notice, a date. Extraction faithfully reproduces all of it, so a 40-page report can carry forty
copies of the same header. That is noise for a language model and it is the kind of near-duplicate
text that survives exact deduplication, because each copy sits inside a different document.

The heuristic is the one FinePDFs uses. Take the first *k* lines of every page, normalise them, and
ask how many pages agree; grow *k* while a single pattern still holds across enough pages, and take
the longest one that does. Repeat from the bottom. Only pages carrying the winning pattern are
stripped, so a title page with no header keeps its first line.

Normalisation is what makes the comparison work: page numbers, dates and section numbers differ per
page, so every digit is folded to ``0`` and whitespace is dropped before comparing. ``Page 3 of 40``
and ``Page 4 of 40`` become the same string.

Two guards keep content from being mistaken for chrome:

* a pattern must appear on at least :attr:`BoilerplateOptions.min_pages` pages *and* on at least
  :attr:`~BoilerplateOptions.min_page_fraction` of them, so a two-page document is left alone and a
  heading that happens to recur twice in a long report is not stripped;
* table rows are never candidates. A table repeated across pages is content, and its header row
  looks exactly like a running header after digits are folded.

This module deliberately imports nothing from :mod:`experiments.datakit.build_pdf_source.docling_extract`.
It works on text plus page offsets, which is what the OCR route produces too, so both routes strip
boilerplate with the same code and the same knobs.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import dupekit

logger = logging.getLogger(__name__)

# Digits carry the variation this heuristic is trying to see past -- page numbers, dates, section
# numbers. ``str.translate`` with a prebuilt table does the fold in C, which matters because this
# runs over every line of every page of every document.
_DIGIT_FOLD = str.maketrans("0123456789", "0000000000")
# Dropped entirely rather than collapsed: PDF extraction is inconsistent about how much whitespace
# it puts between a header's parts, and none of it distinguishes one page's header from another's.
_DROPPED_WHITESPACE = str.maketrans("", "", " \t\r ")  # noqa: RUF001 -- the last entry is U+00A0, which PDFs emit

# A line belonging to a table, which is content even when it repeats. The two routes write tables
# three different ways -- the tagged form the docling serializer emits, a bare Markdown row, and the
# HTML the OCR prompt asks the model for -- so all three are recognised here. A false positive costs
# only that one line's exemption from stripping, which is the safe direction to be wrong in.
_TABLE_MARKERS = ("<docling_table>", "</docling_table>", "<table", "</table>", "<tr", "<td", "<th")


@dataclass(frozen=True)
class BoilerplateOptions:
    """When a repeated edge pattern counts as boilerplate.

    The defaults are FinePDFs'. ``min_pages`` is the guard that matters most: without it, a
    two-page document whose pages happen to start alike loses both first lines.
    """

    min_pages: int = 5
    min_page_fraction: float = 0.25
    # Below 1.0 this would spare documents whose every page shares a header -- which is precisely
    # the case this exists for -- so the ceiling is off by default.
    max_page_fraction: float = 1.0
    # How deep an edge can be. Headers run to a few lines; a larger bound mostly buys the chance to
    # eat a page's opening paragraph when many pages happen to begin the same way.
    max_edge_lines: int = 5


@dataclass(frozen=True)
class BoilerplateResult:
    """Pages with their chrome removed, and what was removed."""

    pages: list[str]
    top_lines: int
    bottom_lines: int
    pages_stripped: int
    lines_removed: int

    @property
    def text(self) -> str:
        return "".join(self.pages)

    @property
    def page_offsets(self) -> list[int]:
        """Cumulative character counts, one per page, matching :attr:`text`."""
        offsets: list[int] = []
        total = 0
        for page in self.pages:
            total += len(page)
            offsets.append(total)
        return offsets


def split_pages(text: str, page_offsets: Sequence[int]) -> list[str]:
    """Split extracted text back into pages using the cumulative offsets recorded with it."""
    pages: list[str] = []
    start = 0
    for offset in page_offsets:
        pages.append(text[start:offset])
        start = offset
    if start < len(text):
        pages.append(text[start:])
    return pages


def _line_key(line: str, page_index: int) -> int:
    """Hash a line into the form patterns are compared on.

    Digits fold to zero and whitespace is dropped, so a page number does not make two copies of one
    header look different. Table rows are keyed by page index as well, which makes them unique and
    therefore never part of a repeated pattern.

    Hashing rather than keeping the strings keeps the pattern tuples small and makes comparing them
    an integer compare, which is the whole cost of the search.
    """
    normalized = line.translate(_DIGIT_FOLD).translate(_DROPPED_WHITESPACE)
    if any(marker in line for marker in _TABLE_MARKERS) or (normalized.startswith("|") and normalized.endswith("|")):
        normalized = f"{page_index}\x00{normalized}"
    return dupekit.hash_xxh3_128(normalized.encode("utf-8"))


def _longest_repeated_edge(
    page_keys: list[list[int]],
    *,
    from_top: bool,
    options: BoilerplateOptions,
) -> tuple[int, frozenset[int]]:
    """Find the longest edge pattern shared by enough pages.

    Returns the pattern's length in lines and the pages carrying it. Growing the pattern one line
    at a time and stopping at the first length that fails its support test finds the longest run,
    because a pattern of length *k+1* can never hold on more pages than its length-*k* prefix.
    """
    num_pages = len(page_keys)
    best_length = 0
    best_pages: frozenset[int] = frozenset()

    for length in range(1, options.max_edge_lines + 1):
        counts: dict[tuple[int, ...], list[int]] = {}
        for index, keys in enumerate(page_keys):
            if len(keys) < length:
                continue
            pattern = tuple(keys[:length]) if from_top else tuple(keys[-length:])
            counts.setdefault(pattern, []).append(index)

        if not counts:
            break

        pattern, pages = max(counts.items(), key=lambda item: len(item[1]))
        support = len(pages)
        if not (
            support >= options.min_pages
            and options.min_page_fraction <= support / num_pages <= options.max_page_fraction
        ):
            break

        best_length = length
        best_pages = frozenset(pages)

    return best_length, best_pages


def strip_boilerplate(pages: Sequence[str], options: BoilerplateOptions | None = None) -> BoilerplateResult:
    """Remove running headers and footers from ``pages``.

    A page keeps its content untouched unless it actually carries the detected pattern, and a page
    that is entirely boilerplate becomes empty rather than being dropped, so page offsets stay
    aligned with the page count.
    """
    options = options or BoilerplateOptions()
    pages = list(pages)

    # Nothing to compare against, and the support threshold could never be met anyway.
    if len(pages) < options.min_pages:
        return BoilerplateResult(pages=pages, top_lines=0, bottom_lines=0, pages_stripped=0, lines_removed=0)

    page_lines = [page.split("\n") for page in pages]
    page_keys = [[_line_key(line, index) for line in lines] for index, lines in enumerate(page_lines)]

    top_lines, top_pages = _longest_repeated_edge(page_keys, from_top=True, options=options)
    bottom_lines, bottom_pages = _longest_repeated_edge(page_keys, from_top=False, options=options)

    if not top_lines and not bottom_lines:
        return BoilerplateResult(pages=pages, top_lines=0, bottom_lines=0, pages_stripped=0, lines_removed=0)

    stripped_pages: list[str] = []
    pages_stripped = 0
    lines_removed = 0
    for index, lines in enumerate(page_lines):
        start = top_lines if index in top_pages else 0
        end = len(lines) - (bottom_lines if index in bottom_pages else 0)
        if end <= start:
            # The page is nothing but chrome. Emptying it rather than dropping it keeps the page
            # count, and therefore the offsets, honest.
            start, end = 0, 0
        if start or end != len(lines):
            pages_stripped += 1
            lines_removed += start + (len(lines) - end)
        stripped_pages.append("\n".join(lines[start:end]))

    logger.debug(
        "Stripped %d top and %d bottom lines from %d of %d pages",
        top_lines,
        bottom_lines,
        pages_stripped,
        len(pages),
    )
    return BoilerplateResult(
        pages=stripped_pages,
        top_lines=top_lines,
        bottom_lines=bottom_lines,
        pages_stripped=pages_stripped,
        lines_removed=lines_removed,
    )


def strip_document_boilerplate(
    text: str, page_offsets: Sequence[int], options: BoilerplateOptions | None = None
) -> BoilerplateResult:
    """Strip boilerplate from one extracted document, given its text and page offsets."""
    return strip_boilerplate(split_pages(text, page_offsets), options)
