# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the whole-cluster duplicate solver.

Every document here is short enough to read. The counts in the comments come
from the rule itself: the original holds 31 words, thus 29 distinct word
3-grams, and one replaced word destroys the 3 n-grams that span it.
"""

from itertools import permutations

import pytest
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    ClusterDocument,
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
"""A different document. It shares no 3-gram with the original, and 160 characters put it
in the middle of the length order, thus the rule does examine it."""

BLANK = " \n\t" * 100
"""Whitespace only. 300 characters make it the longest member of a cluster, but it holds
no n-gram."""


def _cluster(named_texts: dict[str, str]) -> list[ClusterDocument]:
    """One cluster whose member ids are the keys, in the given order."""
    return [ClusterDocument(id=name, text=text) for name, text in named_texts.items()]


def _removed_by(documents: list[ClusterDocument], params: ClusterDedupParams | None = None) -> dict[str, str]:
    """Map the id of each removed document to the id of the survivor that holds it."""
    removals = find_duplicates(documents, params or ClusterDedupParams())
    return {documents[removal.member_index].id: documents[removal.representative_index].id for removal in removals}


NESTED_EXCERPTS = {
    "original": ORIGINAL,
    "first24": " ".join(ORIGINAL.split()[:24]),
    "first18": " ".join(ORIGINAL.split()[:18]),
    "first12": " ".join(ORIGINAL.split()[:12]),
    "first6": " ".join(ORIGINAL.split()[:6]),
    "unrelated": UNRELATED,
}
"""Four excerpts of one original, plus one document that shares nothing with it."""


def test_lightly_edited_copy_is_removed_at_the_default_threshold():
    documents = _cluster({"original": ORIGINAL, "copy": SHORTER_COPY})

    (removal,) = find_duplicates(documents, ClusterDedupParams())

    assert documents[removal.member_index].id == "copy"
    assert documents[removal.representative_index].id == "original"
    # The one edit puts the pair between the two thresholds: the copy is a
    # near-duplicate, not an exact subset of the original.
    assert ClusterDedupParams().minimum_containment <= removal.containment < 1.0
    assert removal.novel_tokens == 1


def test_lightly_edited_copy_survives_a_full_containment_threshold():
    documents = _cluster({"original": ORIGINAL, "copy": SHORTER_COPY})

    assert _removed_by(documents, ClusterDedupParams(minimum_containment=1.0)) == {}


def test_unrelated_document_in_the_same_cluster_is_never_removed():
    documents = _cluster({"original": ORIGINAL, "copy": SHORTER_COPY, "excerpt": EXCERPT, "unrelated": UNRELATED})

    # The unrelated document is neither a member that the rule removes nor a
    # representative that absorbs one, although it is shorter than the original.
    assert _removed_by(documents) == {"copy": "original", "excerpt": "original"}


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
    documents = _cluster({"original": ORIGINAL, "copy": copy_text})

    assert _removed_by(documents) == expected_removals


def test_longest_document_in_a_cluster_always_survives():
    # The excerpts come first in the input, thus an order-sensitive rule would
    # remove the original against one of them.
    documents = _cluster({name: NESTED_EXCERPTS[name] for name in reversed(list(NESTED_EXCERPTS))})

    removed = _removed_by(documents)

    assert set(removed) == {"first24", "first18", "first12", "first6"}
    assert set(removed.values()) == {"original"}


def test_strict_excerpt_is_removed_with_no_novel_tokens():
    documents = _cluster({"original": ORIGINAL, "excerpt": EXCERPT})

    (removal,) = find_duplicates(documents, ClusterDedupParams())

    assert documents[removal.member_index].id == "excerpt"
    assert documents[removal.representative_index].id == "original"
    assert removal.containment == 1.0
    assert removal.novel_tokens == 0


def test_removals_are_independent_of_input_order():
    named_texts = {
        "original": ORIGINAL,
        "copy": SHORTER_COPY,
        "excerpt": EXCERPT,
        "unrelated": UNRELATED,
        "blank": BLANK,
    }
    expected = {"copy": "original", "excerpt": "original"}

    for permutation in permutations(named_texts):
        documents = _cluster({name: named_texts[name] for name in permutation})

        assert _removed_by(documents) == expected, permutation


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_the_two_candidate_paths_find_the_same_duplicates(exact_scan_maximum):
    # The cluster holds 6 members, thus a maximum of 2 selects the banded
    # inverted index and a maximum of 256 selects the exhaustive scan.
    documents = _cluster(NESTED_EXCERPTS)

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {
        "first24": "original",
        "first18": "original",
        "first12": "original",
        "first6": "original",
    }


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_blank_documents_are_neither_removed_nor_representatives(exact_scan_maximum):
    # The whitespace-only document is the longest member, thus it is the first
    # candidate representative for every other member. It holds no n-gram, thus
    # containment against it is 0 and it absorbs nothing.
    documents = _cluster(
        {
            "blank": BLANK,
            "empty": "",
            "original": ORIGINAL,
            "copy": SHORTER_COPY,
            "excerpt": EXCERPT,
        }
    )

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {"copy": "original", "excerpt": "original"}


def test_inverted_index_skips_ngrams_above_the_posting_limit():
    documents = _cluster({"original": ORIGINAL, "copy": ORIGINAL, "unrelated": UNRELATED})
    params = ClusterDedupParams(exact_scan_maximum=2, maximum_posting_length=1)

    assert _removed_by(documents, params) == {}


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_equal_length_ties_use_document_id(exact_scan_maximum):
    documents = _cluster({"zeta": ORIGINAL, "alpha": ORIGINAL, "unrelated": UNRELATED})

    removed = _removed_by(documents, ClusterDedupParams(exact_scan_maximum=exact_scan_maximum))

    assert removed == {"zeta": "alpha"}


@pytest.mark.parametrize("exact_scan_maximum", [2, 256], ids=["inverted_index", "exact_scan"])
def test_candidate_cap_keeps_the_strongest_match(exact_scan_maximum):
    weak_longest = "the reference implementation " + (UNRELATED + " ") * 3
    strong = ORIGINAL + " retained context"
    documents = _cluster({"weak": weak_longest, "strong": strong, "member": ORIGINAL})
    params = ClusterDedupParams(exact_scan_maximum=exact_scan_maximum, maximum_candidates=1)

    removed = _removed_by(documents, params)

    assert removed == {"member": "strong"}
