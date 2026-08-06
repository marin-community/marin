# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how far Docling's reading of a document is from the VLM's.

This is the target side of the routing problem: :mod:`route_features` proposes cheap signals, and
these numbers are what those signals have to predict. They are a *proxy* for the routing label, not
the label itself -- two extractions can disagree without either being wrong, and the direction that
matters is decided by adjudication rather than by any number here. What these metrics do is make
the disagreement measurable and sortable, so adjudication can be spent where it is informative.

**The two routes serialize differently by construction**, so comparison happens on a normalized
token stream rather than on the stored text. Docling emits tagged plain text -- ``<docling_table>``
GitHub grids, ``<docling_formula>``, ``<docling_picture_annotation>``, no heading markers -- while
the VLM emits Markdown with LaTeX math and HTML or pipe tables, and its prompt tells it to ignore
figures outright. Left unnormalized, every table would read as total disagreement and every figure
as content Docling invented.

**Figure text is separated rather than compared.** Docling transcribes the text inside charts and
diagrams; the VLM is instructed not to. Neither is wrong, and including that text in the comparison
made chart-heavy pages look like catastrophic disagreement in early passes over this sample. So
picture-annotation text is excluded from the comparison stream and reported on its own, and the
headline agreement numbers are computed both ways.

**Agreement is reported asymmetrically.** ``recall`` is the share of the VLM's tokens that Docling
also produced -- content Docling *lost*, which is the failure that matters. ``precision`` is the
share of Docling's tokens the VLM also produced; a low value with high recall is usually Docling
adding chart labels or a watermark rather than losing anything. Collapsing the two into F1 hides
which happened, so both are kept.

Order sensitivity comes from the bigram numbers. This repository has already been burned by
unigram F1 once: it scored 0.935 for a layout backend that was fragmenting regions and splicing
multi-column reading order, damage only the bigram metric revealed. Reading-order preservation is
one of the properties the router is supposed to protect, so it is measured directly.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

# How the Docling serializer separates pages; also how ``docling_page_offsets`` are derived.
PAGE_BREAK = "<--- page break --->"

_PICTURE_ANNOTATION = re.compile(
    r"<docling_picture_annotation(?:_non_text)?>(.*?)</docling_picture_annotation(?:_non_text)?>", re.DOTALL
)
_DOCLING_TABLE = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)
_DOCLING_FORMULA = re.compile(r"<docling_formula>(.*?)</docling_formula>", re.DOTALL)
_DOCLING_TAG = re.compile(r"</?docling_[a-z_]+>")

_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
# The VLM emits pipe tables as often as HTML ones; a delimiter row is what makes a run of pipes a
# table rather than a line that happens to contain them.
_PIPE_TABLE = re.compile(r"(?:^\|.*\|[ \t]*\n)+", re.MULTILINE)
_PIPE_DELIMITER = re.compile(r"^\|[\s:|-]+\|[ \t]*$", re.MULTILINE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MARKDOWN_IMAGE = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)

_LATEX_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")

# Markdown and grid punctuation, dropped so that formatting conventions are not read as text.
_MARKUP = re.compile(r"[|#*_`>{}^~=+\-]+")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Streams:
    """One route's page content, split into what is compared and what is only counted."""

    tokens: list[str]
    table_chars: int
    formula_chars: int
    figure_chars: int
    figure_tokens: list[str]


def _tokenize(text: str) -> list[str]:
    """Normalized word tokens: NFKC folds ligatures and full-width forms, markup is dropped."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WORD.findall(_MARKUP.sub(" ", text))


def docling_streams(page: str) -> Streams:
    """Split a Docling page into comparison tokens, with tables, formulas and figures counted."""
    page = page.replace(PAGE_BREAK, " ")
    figure_text = " ".join(match.group(1) for match in _PICTURE_ANNOTATION.finditer(page))
    page = _PICTURE_ANNOTATION.sub(" ", page)
    table_chars = sum(len(match.group(1)) for match in _DOCLING_TABLE.finditer(page))
    formula_chars = sum(len(match.group(1)) for match in _DOCLING_FORMULA.finditer(page))
    return Streams(
        tokens=_tokenize(_DOCLING_TAG.sub(" ", page)),
        table_chars=table_chars,
        formula_chars=formula_chars,
        figure_chars=len(figure_text),
        figure_tokens=_tokenize(_DOCLING_TAG.sub(" ", figure_text)),
    )


def ocr_streams(page: str) -> Streams:
    """Split a VLM Markdown page the same way. Its figure stream is empty by prompt design."""
    table_chars = sum(len(match.group(0)) for match in _HTML_TABLE.finditer(page))
    table_chars += sum(
        len(match.group(0)) for match in _PIPE_TABLE.finditer(page) if _PIPE_DELIMITER.search(match.group(0))
    )
    formula_chars = sum(len(match.group(0)) for match in _LATEX_MATH.finditer(page))
    page = _MARKDOWN_IMAGE.sub(" ", page)
    page = _HTML_TAG.sub(" ", _LATEX_COMMAND.sub(" ", page))
    return Streams(
        tokens=_tokenize(page),
        table_chars=table_chars,
        formula_chars=formula_chars,
        figure_chars=0,
        figure_tokens=[],
    )


def _overlap(reference: Counter, candidate: Counter) -> tuple[float, float]:
    """Return (recall, precision) of *candidate* against *reference* as token multisets."""
    if not reference and not candidate:
        return 1.0, 1.0
    if not reference or not candidate:
        return 0.0, 0.0
    shared = sum((reference & candidate).values())
    return shared / sum(reference.values()), shared / sum(candidate.values())


def _bigrams(tokens: list[str]) -> Counter:
    return Counter(pairwise(tokens))


@dataclass(frozen=True)
class PageAgreement:
    """How far one aligned page pair diverged."""

    unigram_recall: float
    unigram_precision: float
    bigram_recall: float
    bigram_precision: float
    unigram_recall_with_figures: float
    ocr_tokens: int
    docling_tokens: int
    docling_figure_tokens: int
    docling_table_chars: int
    ocr_table_chars: int
    docling_formula_chars: int
    ocr_formula_chars: int


def page_agreement(docling_page: str, ocr_page: str) -> PageAgreement:
    """Compare one page of Docling text against the same page of VLM text."""
    docling, ocr = docling_streams(docling_page), ocr_streams(ocr_page)
    unigram_recall, unigram_precision = _overlap(Counter(ocr.tokens), Counter(docling.tokens))
    bigram_recall, bigram_precision = _overlap(_bigrams(ocr.tokens), _bigrams(docling.tokens))
    # Docling's figure transcriptions rejoin the stream here, which is the fair comparison on a
    # page whose "figure" is really a flowchart the VLM read as a table.
    with_figures, _ = _overlap(Counter(ocr.tokens), Counter(docling.tokens + docling.figure_tokens))
    return PageAgreement(
        unigram_recall=unigram_recall,
        unigram_precision=unigram_precision,
        bigram_recall=bigram_recall,
        bigram_precision=bigram_precision,
        unigram_recall_with_figures=with_figures,
        ocr_tokens=len(ocr.tokens),
        docling_tokens=len(docling.tokens),
        docling_figure_tokens=len(docling.figure_tokens),
        docling_table_chars=docling.table_chars,
        ocr_table_chars=ocr.table_chars,
        docling_formula_chars=docling.formula_chars,
        ocr_formula_chars=ocr.formula_chars,
    )


def split_pages(text: str, offsets: list[int] | None) -> list[str]:
    """Cut a route's text into its pages using the end offsets it recorded."""
    if not offsets:
        return [text]
    pages, start = [], 0
    for end in offsets:
        pages.append(text[start:end])
        start = end
    if start < len(text):
        pages.append(text[start:])
    return pages


# Aligning a Docling page to a VLM page must beat leaving both unaligned, and this is the margin it
# must beat by. Set low because two readings of a hard page can genuinely share few tokens and still
# be the same page; set above zero so that a page one route simply does not have is skipped rather
# than matched to whatever happens to sit at that index.
_ALIGNMENT_GAP_PENALTY = 0.05
# Docling drops pages; it never invents them. The alignment therefore only has to consider offsets
# within this many pages of the diagonal, which keeps a 178-page document from costing 31k page
# comparisons.
_ALIGNMENT_BAND_MARGIN = 3


def _page_similarity(docling_page: str, ocr_page: str) -> float:
    """Cheap token overlap, used only to decide which pages are the same page."""
    docling, ocr = Counter(_tokenize(docling_page)), Counter(_tokenize(ocr_page))
    if not docling and not ocr:
        return 1.0
    if not docling or not ocr:
        return 0.0
    shared = sum((docling & ocr).values())
    return 2 * shared / (sum(docling.values()) + sum(ocr.values()))


def align_pages(docling_pages: list[str], ocr_pages: list[str]) -> list[tuple[int | None, int | None]]:
    """Pair up the two routes' pages, allowing either side to have pages the other lacks.

    Positional pairing is wrong here and quietly so. Docling drops pages it reads nothing from --
    ~8% of documents in this corpus come back with fewer pages than the PDF has -- and after one
    drop every later page is compared against its neighbour. That does not look like a small error:
    it looks like one route inventing content and the other losing it, on every page after the
    drop. Early adjudication of this metric flagged exactly that, on documents where nothing had
    gone wrong except the alignment.

    So pages are aligned by content, monotonically (page order is preserved on both sides), by
    Needleman-Wunsch over token overlap. Returns index pairs; ``None`` on a side means that page
    has no counterpart, which is the honest encoding of "this route lost this page".
    """
    if len(docling_pages) == len(ocr_pages):
        # The common case, and the one where content alignment would only add noise: equal page
        # counts on two byte-exact page partitions of the same PDF are the same pages.
        return [(index, index) for index in range(len(ocr_pages))]

    rows, columns = len(docling_pages), len(ocr_pages)
    band = abs(rows - columns) + _ALIGNMENT_BAND_MARGIN

    def in_band(i: int, j: int) -> bool:
        return abs(i - j) <= band

    best: dict[tuple[int, int], float] = {(0, 0): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(rows + 1):
        for j in range(columns + 1):
            if (i, j) == (0, 0) or not in_band(i, j):
                continue
            options = []
            if i > 0 and j > 0 and (i - 1, j - 1) in best:
                options.append(
                    (best[(i - 1, j - 1)] + _page_similarity(docling_pages[i - 1], ocr_pages[j - 1]), (i - 1, j - 1))
                )
            if i > 0 and (i - 1, j) in best:
                options.append((best[(i - 1, j)] - _ALIGNMENT_GAP_PENALTY, (i - 1, j)))
            if j > 0 and (i, j - 1) in best:
                options.append((best[(i, j - 1)] - _ALIGNMENT_GAP_PENALTY, (i, j - 1)))
            if not options:
                continue
            score, previous = max(options)
            best[(i, j)] = score
            came_from[(i, j)] = previous

    pairs: list[tuple[int | None, int | None]] = []
    position = (rows, columns)
    while position != (0, 0):
        previous = came_from.get(position)
        if previous is None:
            # The band excluded a path to the origin; fall back to consuming what is left in order.
            i, j = position
            while i > 0 or j > 0:
                pairs.append((i - 1 if i > 0 else None, j - 1 if j > 0 else None))
                i, j = max(i - 1, 0), max(j - 1, 0)
            break
        i, j = position
        previous_i, previous_j = previous
        pairs.append((i - 1 if previous_i < i else None, j - 1 if previous_j < j else None))
        position = previous
    return list(reversed(pairs))


def document_agreement(docling_text: str, docling_offsets, ocr_text: str, ocr_offsets) -> dict:
    """Compare two extractions of one document, page by page.

    Pages are matched by content rather than by index -- see :func:`align_pages` -- so a page
    Docling dropped costs that page rather than shifting every page after it. An unmatched VLM page
    is compared against nothing, which scores as total loss, because that is what it is.

    Page numbers are averaged weighted by the VLM page's token count, so a document's score is
    dominated by the pages that carry its text rather than by its blank back cover. The minimum and
    the fraction of bad pages come along because a single destroyed page in a long report is a real
    loss that any mean will bury.
    """
    docling_pages = split_pages(docling_text, docling_offsets)
    ocr_pages = split_pages(ocr_text, ocr_offsets)
    page_count_mismatch = len(docling_pages) - len(ocr_pages)

    alignment = align_pages(docling_pages, ocr_pages)
    pages = [
        page_agreement(
            docling_pages[docling_index] if docling_index is not None else "",
            ocr_pages[ocr_index] if ocr_index is not None else "",
        )
        for docling_index, ocr_index in alignment
    ]
    width = len(pages)
    weights = [max(page.ocr_tokens, 1) for page in pages]
    total_weight = sum(weights)

    def weighted(name: str) -> float:
        return sum(getattr(page, name) * weight for page, weight in zip(pages, weights, strict=True)) / total_weight

    result: dict[str, float] = {
        "pages_compared": width,
        "page_count_mismatch": page_count_mismatch,
    }
    for name in (
        "unigram_recall",
        "unigram_precision",
        "bigram_recall",
        "bigram_precision",
        "unigram_recall_with_figures",
    ):
        values = [getattr(page, name) for page in pages]
        result[f"{name}_mean"] = weighted(name)
        result[f"{name}_min"] = min(values)
    # Pages where Docling lost half the VLM's word bigrams: destroyed, not merely reformatted.
    result["frac_pages_bigram_below_50"] = sum(1 for page in pages if page.bigram_recall < 0.5) / width
    result["frac_pages_unigram_below_50"] = sum(1 for page in pages if page.unigram_recall < 0.5) / width

    for name in (
        "ocr_tokens",
        "docling_tokens",
        "docling_figure_tokens",
        "docling_table_chars",
        "ocr_table_chars",
        "docling_formula_chars",
        "ocr_formula_chars",
    ):
        result[name] = sum(getattr(page, name) for page in pages)
    result["token_ratio"] = result["docling_tokens"] / max(result["ocr_tokens"], 1)
    return result


AGREEMENT_COLUMNS: tuple[str, ...] = tuple(document_agreement("", None, "", None))


def empty_agreement() -> dict:
    """The agreement row for a document the Docling route never produced."""
    return dict.fromkeys(AGREEMENT_COLUMNS, None)
