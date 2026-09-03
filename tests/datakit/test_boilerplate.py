# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for stripping running headers and footers from paginated documents."""

import pytest

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate

_OPTIONS = BoilerplateOptions(min_pages=3, min_page_fraction=0.25)


# Bodies must differ by more than a digit: digits fold to zero, so numbered bodies would all look
# like boilerplate.
_BODY_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
)


def _pages(count: int, header: str = "", footer: str = "", body: str = "Body") -> list[str]:
    """Build ``count`` pages, each optionally wrapped in a header and footer line."""
    pages = []
    for index in range(count):
        lines = []
        if header:
            lines.append(header.format(page=index + 1))
        lines.append(f"{body} paragraph about {_BODY_WORDS[index]}.")
        if footer:
            lines.append(footer.format(page=index + 1))
        pages.append("\n".join(lines))
    return pages


def test_a_repeated_header_is_removed_from_every_page():
    result = strip_boilerplate(_pages(6, header="ACME Annual Report"), _OPTIONS)

    assert result.top_lines == 1
    assert result.pages_stripped == 6
    assert all("ACME" not in page for page in result.pages)
    assert all("Body paragraph about" in page for page in result.pages)


def test_a_footer_whose_page_number_changes_is_still_recognised():
    """Digits fold to zero before comparison, so ``Page 1 of 6`` matches ``Page 2 of 6``."""
    result = strip_boilerplate(_pages(6, footer="Page {page} of 6"), _OPTIONS)

    assert result.bottom_lines == 1
    assert all("Page" not in page for page in result.pages)


def test_headers_and_footers_are_removed_together():
    result = strip_boilerplate(_pages(6, header="Confidential", footer="Page {page}"), _OPTIONS)

    assert (result.top_lines, result.bottom_lines) == (1, 1)
    assert [page.strip() for page in result.pages] == [f"Body paragraph about {w}." for w in _BODY_WORDS[:6]]


def test_the_longest_repeated_header_is_taken_not_the_first_line_only():
    pages = ["\n".join(["ACME Corporation", "Quarterly Filing", f"Body {w}"]) for w in _BODY_WORDS[:6]]

    result = strip_boilerplate(pages, _OPTIONS)

    assert result.top_lines == 2
    assert list(result.pages) == [f"Body {w}" for w in _BODY_WORDS[:6]]


def test_a_page_without_the_header_keeps_its_first_line():
    """A title page has no running header, and must not lose its title."""
    pages = ["Annual Report 2019\nby the committee", *_pages(5, header="ACME Annual Report")]

    result = strip_boilerplate(pages, _OPTIONS)

    assert result.top_lines == 1
    assert result.pages[0].startswith("Annual Report 2019")
    assert result.pages_stripped == 5


def test_a_short_document_is_left_alone():
    """Two pages that happen to start alike are not evidence of a running header."""
    pages = ["Introduction\nfirst", "Introduction\nsecond"]

    result = strip_boilerplate(pages, BoilerplateOptions(min_pages=5))

    assert result.top_lines == 0
    assert result.pages == pages


def test_a_pattern_on_too_few_pages_is_not_boilerplate():
    """Three of twenty pages sharing an opening line is a coincidence, not chrome."""
    pages = [*_pages(3, header="Shared Heading"), *_pages(17, body="Other")]

    result = strip_boilerplate(pages, BoilerplateOptions(min_pages=5, min_page_fraction=0.25))

    assert result.top_lines == 0
    assert result.lines_removed == 0


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("<table><tr><td>Region</td><td>Total</td></tr></table>", id="ocr_html"),
        pytest.param("| Region | Total |", id="inspector_markdown"),
    ],
)
def test_repeated_table_rows_are_never_stripped(row):
    """A table repeated across pages is content, in both the HTML form the OCR route emits and the
    pipe form pdf-inspector writes."""
    pages = [f"{row}\nBody {i}" for i in range(8)]

    result = strip_boilerplate(pages, _OPTIONS)

    assert result.top_lines == 0
    assert all(row in page for page in result.pages)


def test_a_page_that_is_entirely_boilerplate_becomes_empty_and_is_kept():
    """Dropping it would shift every later page offset off its text."""
    pages = [*_pages(5, header="ACME"), "ACME"]

    result = strip_boilerplate(pages, _OPTIONS)

    assert len(result.pages) == 6
    assert result.pages[-1] == ""


@pytest.mark.parametrize("count", [0, 1])
def test_documents_with_too_few_pages_are_returned_unchanged(count):
    pages = _pages(count) if count else []

    result = strip_boilerplate(pages, _OPTIONS)

    assert result.pages == pages
    assert result.lines_removed == 0


def test_lines_differing_only_by_a_number_are_treated_as_the_same_line():
    """The digit fold that matches ``Page 1 of 9`` to ``Page 2 of 9`` also matches body lines that
    differ only by a number; the support thresholds are what keep that from mattering."""
    pages = [f"Section {i} follows.\nUnique {word}." for i, word in enumerate(_BODY_WORDS[:8])]

    result = strip_boilerplate(pages, _OPTIONS)

    assert result.top_lines == 1
    assert all(page.startswith("Unique ") for page in result.pages)
