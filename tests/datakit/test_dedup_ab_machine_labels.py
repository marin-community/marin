# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from experiments.datakit.scripts.dedup_ab_machine_labels import decision_for_pair


def _pair(member: str, canonical: str, **overrides) -> dict:
    return {
        "review_key": "pair",
        "variant": "baseline",
        "member_source_main_dir": "s3://normalized/member",
        "member_basename": "part-0.parquet",
        "member_id": "member",
        "canonical_source_main_dir": "s3://normalized/canonical",
        "canonical_basename": "part-1.parquet",
        "canonical_id": "canonical",
        "evidence_class": "ambiguous",
        "member_text_truncated_for_minhash": False,
        "canonical_text_truncated_for_minhash": False,
        "exact_raw_text": member == canonical,
        "word_5gram_jaccard": 0.5,
        "word_5gram_canonical_containment": 0.5,
        "word_5gram_member_containment": 0.5,
        "raw_sha256": hashlib.sha256(member.encode()).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "member_text": member,
        "canonical_text": canonical,
        **overrides,
    }


def test_exact_complete_text_is_machine_confirmed_duplicate() -> None:
    decision = decision_for_pair(_pair("same complete text", "same complete text"))

    assert decision["label"] == "true_duplicate"
    assert decision["method"] == "raw_identity"
    assert decision["needs_semantic_review"] is False


def test_complete_low_overlap_pair_is_machine_confirmed_false_positive() -> None:
    decision = decision_for_pair(
        _pair(
            "different member",
            "unrelated canonical",
            evidence_class="strong_false_positive",
            word_5gram_jaccard=0.01,
            word_5gram_canonical_containment=0.02,
            word_5gram_member_containment=0.03,
        )
    )

    assert decision["label"] == "false_positive"
    assert decision["method"] == "low_overlap"
    assert decision["needs_semantic_review"] is False


def test_truncated_low_overlap_pair_requires_full_text_semantic_review() -> None:
    decision = decision_for_pair(
        _pair(
            "different member",
            "unrelated canonical",
            evidence_class="strong_false_positive",
            member_text_truncated_for_minhash=True,
            word_5gram_jaccard=0.01,
            word_5gram_canonical_containment=0.02,
            word_5gram_member_containment=0.03,
        )
    )

    assert decision["label"] == ""
    assert decision["method"] == ""
    assert decision["needs_semantic_review"] is True


def test_persisted_complete_text_hashes_are_reverified() -> None:
    pair = _pair("member", "canonical")
    pair["raw_sha256"] = "0" * 64

    with pytest.raises(AssertionError, match="Member text hash changed"):
        decision_for_pair(pair)
