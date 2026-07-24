# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.scripts.dedup_ab_audit import _evidence_class, _set_metrics


def test_set_metrics_preserves_containment_direction() -> None:
    canonical = {"shared", "canonical-only"}
    member = {"shared"}

    jaccard, canonical_containment, member_containment = _set_metrics(canonical, member)

    assert jaccard == 0.5
    assert canonical_containment == 0.5
    assert member_containment == 1.0


def test_longer_dropped_member_with_unique_content_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.49,
        canonical_word_containment=0.99,
        member_word_containment=0.49,
    )

    assert evidence == "ambiguous"


def test_clean_member_containment_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.4,
        canonical_word_containment=0.4,
        member_word_containment=1.0,
    )

    assert evidence == "ambiguous"


def test_exact_raw_text_confirms_redundancy() -> None:
    evidence = _evidence_class(
        exact_raw_text=True,
        word_jaccard=1.0,
        canonical_word_containment=1.0,
        member_word_containment=1.0,
    )

    assert evidence == "strong_duplicate"


def test_low_bidirectional_overlap_confirms_false_positive() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.03,
        canonical_word_containment=0.08,
        member_word_containment=0.12,
    )

    assert evidence == "strong_false_positive"


def test_low_jaccard_with_containment_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.03,
        canonical_word_containment=0.03,
        member_word_containment=0.95,
    )

    assert evidence == "ambiguous"
