# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for recognising running page numbers across the languages the crawl is written in."""

import pytest

from experiments.datakit.build_pdf_source.docling_extract.page_numbers import is_page_number


@pytest.mark.parametrize(
    "text",
    [
        "1",
        "  7  ",
        "1/10",
        "12 / 340",
        "Page 3",
        "Page 3 of 12",
        "page 3 of 12",
        "p. 4",
        "Seite 4 von 9",
        "Página 12 de 30",
        "Strana 5 z 9",
        "第1页共10页",
        "1ページ/10",
        "3 페이지",
        "หน้า 5/9",
        "Pagina: 18/24",
        "2. oldal",
    ],
)
def test_running_page_numbers_are_recognised(text):
    assert is_page_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "Introduction",
        "Chapter 4: Results",
        # Five digits is past any real pagination, and is how part and invoice numbers look.
        "12345",
        # A leading zero marks a reference or a code, not a page.
        "07",
        "Table 1 of the appendix",
        "",
        "   ",
        # A page number is a whole block; a number embedded in a sentence is content.
        "see page 3 for details",
    ],
)
def test_content_is_not_mistaken_for_a_page_number(text):
    assert not is_page_number(text)


def test_a_bare_four_digit_number_is_treated_as_a_page_number():
    """A documented limitation, not an accident.

    ``1998`` is indistinguishable from page 1998 in isolation, so the pattern set matches it. What
    keeps years out of the corpus is where the rule is applied: only the first or last text block
    on a page, and only when the layout model labelled it as text, a title, a footnote or a section
    header. A year in a paragraph is never a candidate.
    """
    assert is_page_number("1998")
