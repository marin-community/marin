# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how far one extraction route's reading of a document is from another's.

This is the target side of the routing problem: :mod:`route_features` proposes cheap signals, and
these numbers are what those signals have to predict. They are a *proxy* for the routing label, not
the label itself -- two extractions can disagree without either being wrong, and the direction that
matters is decided by adjudication rather than by any number here. What these metrics do is make
the disagreement measurable and sortable, so adjudication can be spent where it is informative.

**Routes serialize differently by construction**, so comparison happens on a normalized token
stream rather than on the stored text. Docling emits tagged plain text -- ``<docling_table>``
GitHub grids, ``<docling_formula>``, ``<docling_picture_annotation>``, no heading markers -- while
the VLM and pdf-inspector emit Markdown with HTML fragments, pipe tables and, in the VLM's case,
LaTeX math; the VLM's prompt tells it to ignore figures outright. Left unnormalized, every table
would read as total disagreement and every figure as content one route invented.

A :class:`Route` pairs a name with the normalizer for its dialect, so a new route is added by
declaring one rather than by threading a third serialization convention through the metric. The two
Markdown routes share :func:`markdown_streams`: pdf-inspector's dialect is the VLM's dialect plus
``<u>`` wrappers around links, which the HTML stripping already covers.

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
from collections.abc import Callable
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
# A link's target is an address, not text a route read off the page. Docling never writes one;
# pdf-inspector writes one for every URL it sees, as `<u>[https://x/y](https://x/y)</u>`, which
# would otherwise count that URL's words twice and only for that route.
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
# A comment is markup in every dialect here, and Docling stands a formula it could not read up as
# the literal `<!-- formula-not-decoded -->`, which would otherwise enter the stream as three words
# Docling "added" on every equation in the corpus.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

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
    page = _HTML_COMMENT.sub(" ", page)
    return Streams(
        tokens=_tokenize(_DOCLING_TAG.sub(" ", page)),
        table_chars=table_chars,
        formula_chars=formula_chars,
        figure_chars=len(figure_text),
        figure_tokens=_tokenize(_DOCLING_TAG.sub(" ", figure_text)),
    )


def markdown_streams(page: str) -> Streams:
    """Split a Markdown page -- the VLM's or pdf-inspector's -- the same way.

    Both dialects are Markdown with HTML fragments and pipe tables, so they normalize identically;
    the VLM adds LaTeX math and pdf-inspector adds ``<u>`` around links, and each is handled by a
    rule the other's stream simply never triggers. The figure stream is empty for both: the VLM is
    prompted past figures and pdf-inspector does not transcribe them.
    """
    table_chars = sum(len(match.group(0)) for match in _HTML_TABLE.finditer(page))
    table_chars += sum(
        len(match.group(0)) for match in _PIPE_TABLE.finditer(page) if _PIPE_DELIMITER.search(match.group(0))
    )
    formula_chars = sum(len(match.group(0)) for match in _LATEX_MATH.finditer(page))
    page = _HTML_COMMENT.sub(" ", _MARKDOWN_IMAGE.sub(" ", page))
    # After the images, so an image's target is dropped with it rather than kept as its label.
    page = _MARKDOWN_LINK.sub(r"\1", page)
    page = _HTML_TAG.sub(" ", _LATEX_COMMAND.sub(" ", page))
    return Streams(
        tokens=_tokenize(page),
        table_chars=table_chars,
        formula_chars=formula_chars,
        figure_chars=0,
        figure_tokens=[],
    )


# ---------------------------------------------------------------------------
# Dialect-neutral presentation, for adjudication rather than for scoring
# ---------------------------------------------------------------------------

# The metric normalizes to a token multiset, which is all a number needs. A human or model judge
# reads text, and there the same serialization differences the metric folds away are a *cue*:
# Markdown with pipe tables is recognizably not Docling's tagged plain text, so a judge with a style
# preference could produce a verdict that looks like a quality judgment and is not. These renderers
# put every route into one presentation, preserving content and reading order exactly -- a route
# that scrambles a table's cells still shows scrambled cells -- and erasing only the convention.

TABLE_MARKER = "[table]"
FORMULA_MARKER = "[formula]"
FIGURE_MARKER = "[figure]"

# A delimiter row is what makes a run of pipes a table; it carries no content and is dropped rather
# than rendered as a row of dashes.
_DELIMITER_ROW = re.compile(r"^\|[\s:|-]+\|?$")
_HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|\*|__|_|`)")
_HTML_ROW = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_HTML_CELL = re.compile(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", re.DOTALL | re.IGNORECASE)
_MARKDOWN_IMAGE_ALT = re.compile(r"!\[(.*?)\]\(.*?\)", re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def _canonical_rows(block: str) -> str:
    """Render a pipe-delimited grid as one ``cell | cell`` line per row.

    Docling pads its cells to a common width and the VLM does not, so the same table read the same
    way by both routes differs by hundreds of space characters before a judge reads a word of it.
    """
    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("|"):
            rows.append(stripped)
            continue
        if _DELIMITER_ROW.match(stripped):
            continue
        rows.append(" | ".join(cell.strip() for cell in stripped.strip("|").split("|")))
    return "\n".join(rows)


def _canonical_html_table(block: str) -> str:
    """The same rendering for an HTML table, which the VLM emits as often as a pipe grid."""
    rows = []
    for row in _HTML_ROW.finditer(block):
        cells = [_HTML_TAG.sub(" ", cell).strip() for cell in _HTML_CELL.findall(row.group(0))]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows) if rows else _HTML_TAG.sub(" ", block).strip()


def _tidy(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", _SPACE_RUN.sub(" ", text)).strip()


def canonical_docling(page: str) -> str:
    """Docling's tagged plain text, re-rendered dialect-neutrally."""
    page = page.replace(PAGE_BREAK, " ")
    page = _PICTURE_ANNOTATION.sub(lambda m: f"\n{FIGURE_MARKER} {m.group(1).strip()}\n", page)
    page = _DOCLING_TABLE.sub(lambda m: f"\n{TABLE_MARKER}\n{_canonical_rows(m.group(1))}\n", page)
    page = _DOCLING_FORMULA.sub(lambda m: f"{FORMULA_MARKER} {m.group(1).strip()}", page)
    page = _HTML_COMMENT.sub(" ", page)
    return _tidy(_DOCLING_TAG.sub(" ", page))


def canonical_markdown(page: str) -> str:
    """The VLM's or pdf-inspector's Markdown, re-rendered into the same neutral form.

    Ordering follows :func:`markdown_streams`: images are consumed before links, so an image's
    target is dropped with it rather than surviving as its label.
    """
    page = _MARKDOWN_IMAGE_ALT.sub(lambda m: f"\n{FIGURE_MARKER} {m.group(1).strip()}\n", page)
    page = _HTML_TABLE.sub(lambda m: f"\n{TABLE_MARKER}\n{_canonical_html_table(m.group(0))}\n", page)
    page = _PIPE_TABLE.sub(
        lambda m: (
            f"\n{TABLE_MARKER}\n{_canonical_rows(m.group(0))}\n" if _PIPE_DELIMITER.search(m.group(0)) else m.group(0)
        ),
        page,
    )
    page = _LATEX_MATH.sub(lambda m: f"{FORMULA_MARKER} {_LATEX_COMMAND.sub(' ', m.group(0)).strip('$\\[]() ')}", page)
    page = _HTML_COMMENT.sub(" ", page)
    page = _MARKDOWN_LINK.sub(r"\1", page)
    page = _HTML_TAG.sub(" ", page)
    page = _HEADING_MARKER.sub("", page)
    page = _LIST_MARKER.sub("", page)
    return _tidy(_EMPHASIS.sub("", page))


@dataclass(frozen=True)
class Route:
    """One extraction route: how to normalize its serialization, and what to call it in a column.

    ``name`` is the prefix the per-route count columns carry, so a pair of routes names its own
    output and two pairs sharing a route agree on that route's numbers. ``streams`` is the
    normalization the metric reads; ``canonical`` is the one a judge reads, and they exist
    separately because a token multiset and a legible page are not the same normalization.
    """

    name: str
    streams: Callable[[str], Streams]
    canonical: Callable[[str], str]


VLM = Route("ocr", markdown_streams, canonical_markdown)
DOCLING = Route("docling", docling_streams, canonical_docling)
INSPECTOR = Route("inspector", markdown_streams, canonical_markdown)

# Keyed by the name the study tables and the adjudication packets use for each route, which is not
# the metric's column prefix: the VLM's columns are named ``ocr``.
ROUTES_BY_NAME: dict[str, Route] = {"vlm": VLM, "docling": DOCLING, "inspector": INSPECTOR}


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
    """How far one aligned page pair diverged.

    Roles rather than route names: ``reference`` is the route being read *against* -- recall is the
    share of its tokens the candidate also produced -- and ``candidate`` is the route under test.
    """

    unigram_recall: float
    unigram_precision: float
    bigram_recall: float
    bigram_precision: float
    unigram_recall_with_figures: float
    reference_tokens: int
    candidate_tokens: int
    reference_figure_tokens: int
    candidate_figure_tokens: int
    reference_table_chars: int
    candidate_table_chars: int
    reference_formula_chars: int
    candidate_formula_chars: int


def page_agreement(reference_page: str, candidate_page: str, reference: Route, candidate: Route) -> PageAgreement:
    """Compare one page as the candidate route read it against the same page as the reference did."""
    referenced, candidated = reference.streams(reference_page), candidate.streams(candidate_page)
    unigram_recall, unigram_precision = _overlap(Counter(referenced.tokens), Counter(candidated.tokens))
    bigram_recall, bigram_precision = _overlap(_bigrams(referenced.tokens), _bigrams(candidated.tokens))
    # The candidate's figure transcriptions rejoin the stream here, which is the fair comparison on
    # a page whose "figure" is really a flowchart the reference route read as a table.
    with_figures, _ = _overlap(Counter(referenced.tokens), Counter(candidated.tokens + candidated.figure_tokens))
    return PageAgreement(
        unigram_recall=unigram_recall,
        unigram_precision=unigram_precision,
        bigram_recall=bigram_recall,
        bigram_precision=bigram_precision,
        unigram_recall_with_figures=with_figures,
        reference_tokens=len(referenced.tokens),
        candidate_tokens=len(candidated.tokens),
        reference_figure_tokens=len(referenced.figure_tokens),
        candidate_figure_tokens=len(candidated.figure_tokens),
        reference_table_chars=referenced.table_chars,
        candidate_table_chars=candidated.table_chars,
        reference_formula_chars=referenced.formula_chars,
        candidate_formula_chars=candidated.formula_chars,
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


# Aligning one route's page to another's must beat leaving both unaligned, and this is the margin it
# must beat by. Set low because two readings of a hard page can genuinely share few tokens and still
# be the same page; set above zero so that a page one route simply does not have is skipped rather
# than matched to whatever happens to sit at that index.
_ALIGNMENT_GAP_PENALTY = 0.05
# Routes drop pages; they do not invent them. The alignment therefore only has to consider offsets
# within this many pages of the diagonal, which keeps a 178-page document from costing 31k page
# comparisons.
_ALIGNMENT_BAND_MARGIN = 3


def _page_similarity(candidate_page: str, reference_page: str) -> float:
    """Cheap token overlap, used only to decide which pages are the same page."""
    candidate, reference = Counter(_tokenize(candidate_page)), Counter(_tokenize(reference_page))
    if not candidate and not reference:
        return 1.0
    if not candidate or not reference:
        return 0.0
    shared = sum((candidate & reference).values())
    return 2 * shared / (sum(candidate.values()) + sum(reference.values()))


def align_pages(candidate_pages: list[str], reference_pages: list[str]) -> list[tuple[int | None, int | None]]:
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
    if len(candidate_pages) == len(reference_pages):
        # The common case, and the one where content alignment would only add noise: equal page
        # counts on two byte-exact page partitions of the same PDF are the same pages.
        return [(index, index) for index in range(len(reference_pages))]

    rows, columns = len(candidate_pages), len(reference_pages)
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
                    (
                        best[(i - 1, j - 1)] + _page_similarity(candidate_pages[i - 1], reference_pages[j - 1]),
                        (i - 1, j - 1),
                    )
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


_MEANED_METRICS = (
    "unigram_recall",
    "unigram_precision",
    "bigram_recall",
    "bigram_precision",
    "unigram_recall_with_figures",
)
_SUMMED_COUNTS = ("tokens", "figure_tokens", "table_chars", "formula_chars")


def pages_agreement(reference_pages: list[str], candidate_pages: list[str], reference: Route, candidate: Route) -> dict:
    """Compare two routes' readings of one document, page by page.

    Pages are matched by content rather than by index -- see :func:`align_pages` -- so a page the
    candidate dropped costs that page rather than shifting every page after it. An unmatched
    reference page is compared against nothing, which scores as total loss, because that is what it
    is.

    Page numbers are averaged weighted by the reference page's token count, so a document's score is
    dominated by the pages that carry its text rather than by its blank back cover. The minimum and
    the fraction of bad pages come along because a single destroyed page in a long report is a real
    loss that any mean will bury.
    """
    alignment = align_pages(candidate_pages, reference_pages)
    pages = [
        page_agreement(
            reference_pages[reference_index] if reference_index is not None else "",
            candidate_pages[candidate_index] if candidate_index is not None else "",
            reference,
            candidate,
        )
        for candidate_index, reference_index in alignment
    ]
    width = len(pages)
    weights = [max(page.reference_tokens, 1) for page in pages]
    total_weight = sum(weights)

    result: dict[str, float] = {
        "pages_compared": width,
        "page_count_mismatch": len(candidate_pages) - len(reference_pages),
    }
    for name in _MEANED_METRICS:
        values = [getattr(page, name) for page in pages]
        result[f"{name}_mean"] = (
            sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight
        )
        result[f"{name}_min"] = min(values)
    # Pages where the candidate lost half the reference's word bigrams: destroyed, not reformatted.
    result["frac_pages_bigram_below_50"] = sum(1 for page in pages if page.bigram_recall < 0.5) / width
    result["frac_pages_unigram_below_50"] = sum(1 for page in pages if page.unigram_recall < 0.5) / width

    for role, route in (("reference", reference), ("candidate", candidate)):
        for count in _SUMMED_COUNTS:
            result[f"{route.name}_{count}"] = sum(getattr(page, f"{role}_{count}") for page in pages)
    result["token_ratio"] = result[f"{candidate.name}_tokens"] / max(result[f"{reference.name}_tokens"], 1)
    return result


def document_agreement(
    docling_text: str, docling_offsets: list[int] | None, ocr_text: str, ocr_offsets: list[int] | None
) -> dict:
    """The Docling-versus-VLM pair, from each route's stored text and its own page offsets."""
    return pages_agreement(split_pages(ocr_text, ocr_offsets), split_pages(docling_text, docling_offsets), VLM, DOCLING)


def agreement_columns(reference: Route, candidate: Route) -> tuple[str, ...]:
    """The column names a route pair emits, for building the null row without running the metric."""
    return tuple(pages_agreement([""], [""], reference, candidate))


AGREEMENT_COLUMNS: tuple[str, ...] = agreement_columns(VLM, DOCLING)


def empty_agreement(reference: Route = VLM, candidate: Route = DOCLING) -> dict:
    """The agreement row for a document one of the two routes never produced."""
    return dict.fromkeys(agreement_columns(reference, candidate), None)
