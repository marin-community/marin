# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FinePDFs-style GlotLID labeling step.

The vendored thresholds in ``lid_th_values.json`` are only meaningful if the tagger reproduces the
pipeline they were calibrated against, so these tests pin the port-fidelity points: the count=8
table substitution, byte-not-character gate math, gated pages diluting the page average, and the
strict comparisons and fallback order of the bucket selection. The model is faked -- the real one
is a 1.7 GB download and none of the ported behavior lives inside it.
"""

from collections.abc import Sequence

import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.build_pdf_source.language_label import (
    LANGUAGE_THRESHOLD,
    PREDICT_TOP_K,
    clean_page,
    label_document,
    load_thresholds,
    page_scores,
    select_bucket,
)

# 30 alphabetic characters. As ASCII that is 30 bytes (under the 50-byte gate); the same count of
# Greek letters is 60 bytes (over it). The gate is bytes, not characters.
_LATIN_30 = "abcdefghij" * 3
_GREEK_30 = "αβγδεζηθικ" * 3


class _FakeLid:
    """A fasttext stand-in: fixed distribution, and a record of every line predicted."""

    def __init__(self, labels: Sequence[str], scores: Sequence[float]):
        self._labels = [f"__label__{label}" for label in labels]
        self._scores = list(scores)
        self.lines: list[str] = []
        self.k_values: list[int] = []

    def predict(self, lines: list[str], k: int) -> tuple[list[list[str]], list[list[float]]]:
        self.lines.extend(lines)
        self.k_values.append(k)
        return [list(self._labels) for _ in lines], [list(self._scores) for _ in lines]


def _document(pages: list[str]) -> dict:
    """A stored record: newline-terminated pages, cumulative character offsets."""
    terminated = [page if page.endswith("\n") else page + "\n" for page in pages]
    text = "".join(terminated)
    offsets, running = [], 0
    for page in terminated:
        running += len(page)
        offsets.append(running)
    return {"id": generate_id(text), "text": text, "page_offsets": offsets}


def test_clean_page_removes_only_the_first_eight_table_lines():
    """FinePDFs passed re.MULTILINE (== 8) as the sub count; the thresholds assume that quirk."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    page = "\n".join(f"| {word} | row |" for word in words)
    cleaned = clean_page(page)
    for removed in words[:8]:
        assert removed not in cleaned
    for kept in words[8:]:
        assert kept in cleaned


def test_clean_page_strips_markdown_punctuation_and_collapses_whitespace():
    assert clean_page("a-b  *c*\t#d\ne|f") == "ab c d ef"


def test_alpha_gate_counts_utf8_bytes_not_characters():
    """30 Greek letters clear the 50-byte gate; the same 30 letters in ASCII do not."""
    model = _FakeLid(["eng_Latn"], [0.9])
    assert page_scores([_GREEK_30], model).gated == 0
    assert model.lines == [_GREEK_30]

    gated_model = _FakeLid(["eng_Latn"], [0.9])
    scores = page_scores([_LATIN_30], gated_model)
    assert scores.gated == 1
    assert scores.averages == {}
    assert gated_model.lines == []  # a gated page never reaches the model


def test_low_alpha_ratio_page_is_gated_without_a_model_call():
    """Sixty letters drowned in digits: over the byte floor, under the 0.2 alpha ratio."""
    model = _FakeLid(["eng_Latn"], [0.9])
    scores = page_scores([_GREEK_30 + "0" * 400], model)
    assert scores.gated == 1
    assert model.lines == []


def test_gated_pages_still_divide_the_page_average():
    """Two pages, one gated: the surviving page's scores are halved, not renormalized."""
    model = _FakeLid(["eng_Latn", "fra_Latn"], [0.8, 0.4])
    scores = page_scores([_GREEK_30, "42"], model)
    assert scores.gated == 1
    assert scores.averages == {"eng_Latn": 0.4, "fra_Latn": 0.2}
    assert model.k_values == [PREDICT_TOP_K]


def test_page_average_at_exactly_the_language_threshold_is_dropped():
    model = _FakeLid(["eng_Latn", "fra_Latn"], [LANGUAGE_THRESHOLD, LANGUAGE_THRESHOLD * 2])
    assert page_scores([_GREEK_30], model).averages == {"fra_Latn": LANGUAGE_THRESHOLD * 2}


def test_select_bucket_takes_the_top_language_above_its_threshold():
    bucket, score = select_bucket({"eng_Latn": 0.6, "fra_Latn": 0.4}, {"eng_Latn": 0.5, "fra_Latn": 0.3})
    assert (bucket, score) == ("eng_Latn", 0.6)


def test_select_bucket_reroutes_past_a_sub_threshold_top_language():
    """Top language below its bar: the next candidate above its own bar wins, score and all."""
    bucket, score = select_bucket({"eng_Latn": 0.6, "fra_Latn": 0.4}, {"eng_Latn": 0.9, "fra_Latn": 0.3})
    assert (bucket, score) == ("fra_Latn", 0.4)


def test_select_bucket_score_equal_to_threshold_does_not_pass():
    """The comparison is strict, so an exact tie falls through to the removed bucket."""
    bucket, _ = select_bucket({"eng_Latn": 0.5}, {"eng_Latn": 0.5})
    assert bucket == "eng_Latn_removed"


def test_unknown_threshold_language_is_never_selected_mid_list():
    """No threshold entry means unselectable, so a known-threshold top falls to _removed."""
    bucket, score = select_bucket({"eng_Latn": 0.6, "xxx_Latn": 0.4}, {"eng_Latn": 0.9})
    assert (bucket, score) == ("eng_Latn_removed", 0.6)


def test_unknown_threshold_top_language_survives_as_its_raw_label():
    bucket, score = select_bucket({"xxx_Latn": 0.6, "eng_Latn": 0.4}, {"eng_Latn": 0.9})
    assert (bucket, score) == ("xxx_Latn", 0.6)


def test_top_ranked_zxx_is_kept_without_consulting_thresholds():
    bucket, score = select_bucket({"zxx_Latn": 0.6, "eng_Latn": 0.4}, {})
    assert (bucket, score) == ("zxx_Latn", 0.6)


def test_lower_ranked_zxx_passes_through_its_negative_override():
    """The -1 overrides make zxx selectable at any rank -- FinePDFs' shipped behavior."""
    thresholds = load_thresholds()
    bucket, score = select_bucket({"eng_Latn": 0.6, "zxx_Latn": 0.05}, thresholds | {"eng_Latn": 0.9})
    assert (bucket, score) == ("zxx_Latn", 0.05)


def test_no_candidates_at_all_is_unknown_with_zero_score():
    assert select_bucket({}, load_thresholds()) == ("unknown", 0.0)


def test_load_thresholds_floors_values_and_adds_the_zxx_overrides():
    thresholds = load_thresholds()
    zxx = {language: value for language, value in thresholds.items() if language.startswith("zxx_")}
    assert zxx == {"zxx_Latn": -1.0, "zxx_Zzzz": -1.0, "zxx_Arab": -1.0}
    assert min(value for language, value in thresholds.items() if language not in zxx) >= 0.05
    assert len(thresholds) == 1545 + len(zxx)


def test_label_document_scores_each_stored_page_separately():
    """The pages the model sees are the cleaned page_offsets slices, in order."""
    first = "Πρώτη σελίδα. " + _GREEK_30
    second = "Δεύτερη σελίδα. " + _GREEK_30
    model = _FakeLid(["ell_Grek"], [0.9])
    labeled = label_document(_document([first, second]), model, load_thresholds())
    assert model.lines == [clean_page(first + "\n"), clean_page(second + "\n")]
    assert labeled["language"] == "ell_Grek"
    assert labeled["language_score"] == pytest.approx(0.9)


def test_label_document_with_every_page_gated_is_unknown():
    labeled = label_document(_document(["42", "?!"]), _FakeLid(["eng_Latn"], [0.9]), load_thresholds())
    assert labeled["language"] == "unknown"
    assert labeled["language_score"] == 0.0


def test_label_document_rejects_text_longer_than_the_recorded_pages():
    document = _document([_GREEK_30])
    document["text"] += "text past the last recorded page has no known upstream cause"
    with pytest.raises(ValueError, match="page_offsets"):
        label_document(document, _FakeLid(["eng_Latn"], [0.9]), load_thresholds())


def test_label_document_clamps_offsets_left_stale_by_whitespace_capping():
    # normalize's whitespace-run capping shrinks text without updating page_offsets; the
    # document must still be labeled, with each page slice clamped to the real text.
    page = "language identification needs plenty of alphabetic text to pass the byte gate "
    document = _document([page, page])
    document["text"] = document["text"][:-7]
    lid = _FakeLid(["eng_Latn"], [0.9])
    labeled = label_document(document, lid, load_thresholds())
    assert labeled["language"] == "eng_Latn"
    assert labeled["language_score"] == pytest.approx(0.9)
    assert len(lid.lines) == 2  # both pages survive the clamp and reach the model
