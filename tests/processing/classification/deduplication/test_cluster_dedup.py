# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the whole-cluster duplicate solver."""

from itertools import permutations

import pytest
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    find_duplicates,
)

ORIGINAL = (
    "the reference implementation walks every cluster member from the longest "
    "document and keeps the first representative that already holds the member "
    "content so a shorter copy never survives a longer original"
)
"""31 words, 205 characters, 29 distinct 3-grams."""

SHORTER_COPY = ORIGINAL.replace("already holds", "now holds")
"""The original with one word replaced: containment 26/29, 201 characters."""

LONGER_COPY = ORIGINAL.replace("already holds", "presently holds")
"""The same single-word edit, but 207 characters, thus longer than the original."""

EXCERPT = " ".join(ORIGINAL.split()[4:20])
"""16 consecutive words of the original, thus a strict subset of its n-grams."""

UNRELATED = (
    "a calibration table maps each humidity reading to the pressure coefficient "
    "that the sensor firmware applies before it reports a value to the flight recorder bus"
)
"""Its 160 characters put it in the middle of the length order."""

BLANK = " \n\t" * 100
"""Whitespace only. 300 characters make it the longest member of a cluster, but it holds
no n-gram."""


def _removed_by(documents: dict[str, str], params: ClusterDedupParams | None = None) -> dict[str, str]:
    """Map the id of each removed document to the id of the survivor that holds it."""
    removals = find_duplicates(list(documents.values()), params or ClusterDedupParams())
    return {list(documents)[removal.member_index]: list(documents)[removal.representative_index] for removal in removals}


NESTED_EXCERPTS = {
    "original": ORIGINAL,
    "first24": " ".join(ORIGINAL.split()[:24]),
    "first18": " ".join(ORIGINAL.split()[:18]),
    "first12": " ".join(ORIGINAL.split()[:12]),
    "first6": " ".join(ORIGINAL.split()[:6]),
    "unrelated": UNRELATED,
}


def test_lightly_edited_copy_is_removed_at_the_default_threshold():
    documents = {"original": ORIGINAL, "copy": SHORTER_COPY}

    (removal,) = find_duplicates(list(documents.values()), ClusterDedupParams())

    assert list(documents)[removal.member_index] == "copy"
    assert list(documents)[removal.representative_index] == "original"
    assert removal.containment == pytest.approx(26 / 29)
    assert removal.novel_tokens == 1
    assert removal.jaccard == pytest.approx(26 / 32)
    assert removal.comparisons == 1


def test_lightly_edited_copy_survives_a_full_containment_threshold():
    documents = {"original": ORIGINAL, "copy": SHORTER_COPY}

    assert _removed_by(documents, ClusterDedupParams(minimum_containment=1.0)) == {}


def test_production_threshold_keeps_a_pair_that_passes_at_sixty_percent():
    documents = {"original": "a b c d e f g h i j", "copy": "a b c d x f g h i j"}

    assert _removed_by(documents) == {}
    assert _removed_by(documents, ClusterDedupParams(minimum_containment=0.60)) == {"copy": "original"}


@pytest.mark.parametrize(
    ("copy_text", "expected_removals"),
    [
        (SHORTER_COPY, {"copy": "original"}),
        (LONGER_COPY, {"original": "copy"}),
    ],
    ids=["copy_is_shorter", "copy_is_longer"],
)
def test_near_duplicate_pair_removes_only_the_shorter_document(copy_text, expected_removals):
    # The two variants hold the same single-word edit, thus containment is
    # identical (26/29) in each direction and in each case. Only the character
    # count differs, and it alone decides which document the rule removes.
    documents = {"original": ORIGINAL, "copy": copy_text}

    assert _removed_by(documents) == expected_removals


def test_strict_excerpt_is_removed_with_no_novel_tokens():
    documents = {"original": ORIGINAL, "excerpt": EXCERPT}

    (removal,) = find_duplicates(list(documents.values()), ClusterDedupParams())

    assert list(documents)[removal.member_index] == "excerpt"
    assert list(documents)[removal.representative_index] == "original"
    assert removal.containment == 1.0
    assert removal.novel_tokens == 0


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_distinct_length_matches_are_independent_of_input_order(exact_scan_maximum):
    named_texts = {
        "original": ORIGINAL,
        "copy": SHORTER_COPY,
        "excerpt": EXCERPT,
        "unrelated": UNRELATED,
        "blank": BLANK,
    }
    expected = {"copy": "original", "excerpt": "original"}

    for permutation in permutations(named_texts):
        documents = {name: named_texts[name] for name in permutation}

        assert _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum)) == expected, permutation


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_each_candidate_path_removes_nested_excerpts(exact_scan_maximum):
    documents = NESTED_EXCERPTS

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {
        "first24": "original",
        "first18": "original",
        "first12": "original",
        "first6": "original",
    }


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_blank_documents_are_neither_removed_nor_representatives(exact_scan_maximum):
    documents = {
        "blank": BLANK,
        "empty": "",
        "original": ORIGINAL,
        "copy": SHORTER_COPY,
        "excerpt": EXCERPT,
    }

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {"copy": "original", "excerpt": "original"}


def test_inverted_index_uses_common_postings_when_all_exceed_the_limit():
    documents = {"original": ORIGINAL, "copy": ORIGINAL, "unrelated": UNRELATED}
    params = ClusterDedupParams(exact_scan_maximum=2, maximum_posting_length=1)

    assert _removed_by(documents, params) == {"copy": "original"}


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_equal_length_ties_use_input_order(exact_scan_maximum):
    documents = {"zeta": ORIGINAL, "alpha": ORIGINAL, "unrelated": UNRELATED}

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {"alpha": "zeta"}


def test_candidate_cap_keeps_the_strongest_match():
    weak_longest = "the reference implementation " + (UNRELATED + " ") * 3
    strong = ORIGINAL + " retained context"
    documents = {"weak": weak_longest, "strong": strong, "member": ORIGINAL}
    params = ClusterDedupParams(exact_scan_maximum=2, maximum_candidates=1)

    removed = _removed_by(documents, params)

    assert removed == {"member": "strong"}


@pytest.mark.parametrize("exact_scan_maximum", [256, 300], ids=["inverted_index", "exact_scan"])
def test_member_only_probes_can_miss_duplicates_in_the_production_index(exact_scan_maximum):
    shared_text = " ".join(f"word{index:03d}" for index in range(200))
    documents = {
        f"doc{index:03d}": shared_text + " " + " ".join(f"edit{index:03d}_{word:02d}" for word in range(40))
        for index in range(300)
    }

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    expected = {} if exact_scan_maximum == 256 else {f"doc{index:03d}": "doc000" for index in range(1, 300)}
    assert removed == expected


def test_production_candidate_cap_applies_before_removed_representatives_are_excluded():
    documents = {"representative": LONGER_COPY, "bridge": ORIGINAL, "member": EXCERPT}
    params = ClusterDedupParams(exact_scan_maximum=2, maximum_candidates=1)

    removed = _removed_by(documents, params)

    assert removed == {"bridge": "representative"}


@pytest.mark.parametrize("ngram_size", [3, 5])
def test_short_text_is_one_whole_shingle(ngram_size):
    documents = {"first": "alpha beta", "copy": "ALPHA BETA", "different": "alpha gamma"}

    assert _removed_by(documents, ClusterDedupParams(ngram_size=ngram_size)) == {"copy": "first"}
