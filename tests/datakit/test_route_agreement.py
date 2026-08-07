# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the Docling-versus-VLM agreement metric.

The metric exists to separate three things that all look like "the texts differ": the routes'
different serialization conventions (which is not disagreement), Docling losing content the VLM
read (which is), and Docling adding figure text the VLM was told to skip (which is neither). These
tests pin those distinctions, because a metric that confuses them trains the router backwards.
"""

import pytest

from experiments.datakit.build_pdf_source.quality.route_agreement import (
    align_pages,
    document_agreement,
    ocr_streams,
    page_agreement,
    split_pages,
)

SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank at dawn"


def test_serialization_conventions_are_not_disagreement():
    """The same table in each route's own markup scores as agreement, not as loss."""
    docling = "<docling_table>| Name | Count |\n|------|-------|\n| Widgets | 12 |</docling_table>"
    vlm = "| Name | Count |\n| --- | --- |\n| Widgets | 12 |"

    agreement = page_agreement(docling, vlm)

    assert agreement.unigram_recall == pytest.approx(1.0)
    assert agreement.bigram_recall == pytest.approx(1.0)


def test_heading_and_emphasis_markup_is_not_disagreement():
    """Docling strips heading markers and the VLM emits them; the words are the same."""
    agreement = page_agreement(SENTENCE, f"# {SENTENCE.split()[0]}\n\n**{' '.join(SENTENCE.split()[1:])}**")

    assert agreement.unigram_recall == pytest.approx(1.0)


def test_lost_body_text_shows_up_as_lost_recall():
    docling = " ".join(SENTENCE.split()[:4])

    agreement = page_agreement(docling, SENTENCE)

    assert agreement.unigram_recall < 0.4
    # Docling produced nothing the VLM did not have, so precision stays perfect: the metric says
    # "content was lost", not "content was invented".
    assert agreement.unigram_precision == pytest.approx(1.0)


def test_reordered_text_keeps_unigram_recall_but_loses_bigram_recall():
    """Reading-order damage is invisible to unigrams, which is why bigrams are reported."""
    words = SENTENCE.split()
    spliced = " ".join(words[::2] + words[1::2])

    agreement = page_agreement(spliced, SENTENCE)

    assert agreement.unigram_recall == pytest.approx(1.0)
    assert agreement.bigram_recall < 0.2


def test_figure_text_is_excluded_from_comparison_and_reported_separately():
    """Docling's chart labels must not read as content the VLM lost, nor as content Docling invented."""
    docling = f"{SENTENCE} <docling_picture_annotation>vertical axis label</docling_picture_annotation>"

    agreement = page_agreement(docling, SENTENCE)

    assert agreement.unigram_precision == pytest.approx(1.0)
    assert agreement.docling_figure_tokens == 3


def test_figure_text_can_be_credited_when_the_vlm_read_the_figure_as_prose():
    """A flowchart the VLM transcribed is content, so the with-figures reading must count it."""
    docling = "<docling_picture_annotation>Submit application then review then notify</docling_picture_annotation>"
    vlm = "Submit application then review then notify"

    agreement = page_agreement(docling, vlm)

    assert agreement.unigram_recall == pytest.approx(0.0)
    assert agreement.unigram_recall_with_figures == pytest.approx(1.0)


def test_ligatures_and_width_variants_normalize_to_the_same_tokens():
    agreement = page_agreement("ﬁnal oﬀice \uff57idth", "final office width")

    assert agreement.unigram_recall == pytest.approx(1.0)


def test_pipe_tables_need_a_delimiter_row_to_count_as_a_table():
    """A line containing pipes is not a table; the delimiter row is what makes one."""
    assert ocr_streams("a | b | c\nd | e | f").table_chars == 0
    assert ocr_streams("| a | b |\n| --- | --- |\n| c | d |\n").table_chars > 0


def test_pages_are_compared_positionally_on_each_route_own_offsets():
    """Each route records its own page boundaries; the metric must not assume they coincide."""
    first, second = "alpha beta gamma", "delta epsilon zeta"
    docling = f"{first}{second}"
    vlm = f"{first} {second}"

    result = document_agreement(docling, [len(first)], vlm, [len(first) + 1])

    assert result["pages_compared"] == 2
    assert result["unigram_recall_mean"] == pytest.approx(1.0)
    assert result["page_count_mismatch"] == 0


def test_a_page_only_one_route_produced_is_reported_as_a_mismatch():
    result = document_agreement("alpha beta", None, "alpha beta<PAGE>gamma delta", [10])

    assert result["page_count_mismatch"] == -1
    assert result["unigram_recall_min"] == pytest.approx(0.0)


def test_a_destroyed_page_survives_the_document_average():
    """A long report with one destroyed page must not average that page away."""
    good = " ".join(f"word{index}" for index in range(200))
    pages = [good] * 9 + ["totally unrelated replacement content here"]
    vlm = "".join(pages)
    offsets = []
    position = 0
    for page in pages:
        position += len(page)
        offsets.append(position)
    docling = "".join([*pages[:9], ""])

    result = document_agreement(docling, offsets, vlm, offsets)

    assert result["unigram_recall_mean"] > 0.9
    assert result["unigram_recall_min"] == pytest.approx(0.0)
    assert result["frac_pages_unigram_below_50"] == pytest.approx(0.1)


def test_a_dropped_page_costs_that_page_and_not_the_ones_after_it():
    """The regression this metric was rebuilt for.

    Docling drops pages it reads nothing from, so its page list is shorter than the VLM's. Paired
    by index, every page after the drop is compared against its neighbour, and a document where
    nothing went wrong scores as one route inventing content and the other losing it.
    """
    pages = [
        " ".join(f"alpha{index}" for index in range(60)),
        " ".join(f"beta{index}" for index in range(60)),
        " ".join(f"gamma{index}" for index in range(60)),
        " ".join(f"delta{index}" for index in range(60)),
    ]
    offsets, position = [], 0
    for page in pages:
        position += len(page)
        offsets.append(position)
    vlm = "".join(pages)

    # Docling read every page except the second.
    kept = [pages[0], pages[2], pages[3]]
    docling_offsets, position = [], 0
    for page in kept:
        position += len(page)
        docling_offsets.append(position)

    result = document_agreement("".join(kept), docling_offsets, vlm, offsets)

    # Exactly one of four pages is lost; the three Docling did read still match their own pages.
    assert result["frac_pages_unigram_below_50"] == pytest.approx(0.25)
    assert result["unigram_recall_mean"] == pytest.approx(0.75, abs=0.02)
    assert result["page_count_mismatch"] == -1


def test_alignment_marks_the_page_that_is_actually_missing():
    pages = ["one one one one", "two two two two", "three three three three"]

    alignment = align_pages([pages[0], pages[2]], pages)

    assert alignment == [(0, 0), (None, 1), (1, 2)]


def test_equal_page_counts_align_by_index():
    """Two byte-exact partitions of the same PDF are the same pages; content matching would only add noise."""
    assert align_pages(["a", "b", "c"], ["x", "y", "z"]) == [(0, 0), (1, 1), (2, 2)]


def test_split_pages_keeps_trailing_text_that_offsets_do_not_cover():
    assert split_pages("abcdef", [2, 4]) == ["ab", "cd", "ef"]


def test_documents_with_no_text_on_either_side_agree():
    """Two routes that both read nothing have not disagreed, and must not score as failure."""
    result = document_agreement("", None, "", None)

    assert result["unigram_recall_mean"] == pytest.approx(1.0)
    assert result["bigram_recall_mean"] == pytest.approx(1.0)
