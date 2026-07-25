# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.datakit.scripts.dedup_ab_finalize import final_decision, validate_occurrence_coverage
from experiments.datakit.scripts.dedup_ab_machine_labels import decision_for_pair


def _pair(*, exact: bool = False, ambiguous: bool = False) -> dict:
    member_text = "identical complete text" if exact else "member alpha beta gamma delta epsilon"
    canonical_text = member_text if exact else "canonical one two three four five"
    return {
        "review_key": '["baseline","source","part","member","source","part","canonical"]',
        "variant": "baseline",
        "member_source_main_dir": "source",
        "member_basename": "part",
        "member_id": "member",
        "canonical_source_main_dir": "source",
        "canonical_basename": "part",
        "canonical_id": "canonical",
        "raw_sha256": hashlib.sha256(member_text.encode()).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical_text.encode()).hexdigest(),
        "member_text": member_text,
        "canonical_text": canonical_text,
        "exact_raw_text": exact,
        "evidence_class": "ambiguous" if ambiguous else "strong_duplicate" if exact else "strong_false_positive",
        "word_5gram_jaccard": 1.0 if exact else 0.0,
        "word_5gram_canonical_containment": 1.0 if exact else 0.0,
        "word_5gram_member_containment": 1.0 if exact else 0.0,
        "member_text_truncated_for_minhash": ambiguous,
        "canonical_text_truncated_for_minhash": False,
    }


def _input(kind: str, payload: dict) -> dict[str, str]:
    return {
        "review_key": payload["review_key"],
        "kind": kind,
        "payload_json": json.dumps(payload),
    }


def _verified_pair(pair: dict) -> dict:
    decision = decision_for_pair(pair)
    return {
        **decision,
        "expected_label": decision["label"],
        "expected_method": decision["method"],
        "expected_basis": decision["basis"],
        "pair_path": "pairs.parquet",
        "pair_row_index": 0,
    }


def _machine(pair: dict) -> dict:
    return {
        **decision_for_pair(pair),
        "pair_path": "pairs.parquet",
        "pair_row_index": 0,
    }


def test_final_decision_accepts_hash_verified_machine_label() -> None:
    pair = _pair(exact=True)
    machine = _machine(pair)

    result = final_decision(pair["review_key"], iter([_input("pair", _verified_pair(pair)), _input("machine", machine)]))

    assert result["label"] == "true_duplicate"
    assert result["method"] == "raw_identity"
    assert json.loads(result["member_occurrence_key"]) == ["baseline", "source", "part", "member"]


def test_final_decision_requires_one_bound_semantic_label_when_routed() -> None:
    pair = _pair(ambiguous=True)
    machine = _machine(pair)
    semantic = {
        **{
            key: machine[key]
            for key in (
                "review_key",
                "variant",
                "member_source_main_dir",
                "member_basename",
                "member_id",
                "canonical_source_main_dir",
                "canonical_basename",
                "canonical_id",
                "raw_sha256",
                "canonical_raw_sha256",
                "pair_path",
                "pair_row_index",
            )
        },
        "label": "false_positive",
        "method": "semantic",
        "basis": "The complete texts discuss unrelated subjects.",
    }

    result = final_decision(
        pair["review_key"],
        iter([_input("pair", _verified_pair(pair)), _input("machine", machine), _input("semantic", semantic)]),
    )

    assert result["label"] == "false_positive"
    assert result["method"] == "semantic"


def test_final_decision_rejects_tampered_machine_label() -> None:
    pair = _pair(exact=True)
    machine = {**_machine(pair), "label": "false_positive"}

    with pytest.raises(AssertionError, match="Machine label differs"):
        final_decision(pair["review_key"], iter([_input("pair", _verified_pair(pair)), _input("machine", machine)]))


def test_final_decision_rejects_tampered_pair_reference() -> None:
    pair = _pair(exact=True)
    machine = {**_machine(pair), "pair_row_index": 1}

    with pytest.raises(AssertionError, match="Identity mismatch"):
        final_decision(pair["review_key"], iter([_input("pair", _verified_pair(pair)), _input("machine", machine)]))


def test_final_decision_rejects_semantic_label_for_machine_resolved_pair() -> None:
    pair = _pair(exact=True)
    machine = _machine(pair)

    with pytest.raises(AssertionError, match="Unexpected semantic record"):
        final_decision(
            pair["review_key"],
            iter([_input("pair", _verified_pair(pair)), _input("machine", machine), _input("semantic", machine)]),
        )


def test_occurrence_coverage_requires_exact_drop_and_canonical_roles() -> None:
    member_key = '["baseline","source","part","member"]'
    canonical_key = '["baseline","source","part","canonical"]'
    review_key = "review"

    member = validate_occurrence_coverage(
        member_key,
        iter(
            [
                {"kind": "score", "role": "drop", "review_key": review_key},
                {"kind": "member", "role": "", "review_key": review_key},
            ]
        ),
    )
    canonical = validate_occurrence_coverage(
        canonical_key,
        iter(
            [
                {"kind": "score", "role": "canonical", "review_key": ""},
                {"kind": "canonical", "role": "", "review_key": review_key},
            ]
        ),
    )

    assert member["label_references"] == 1
    assert canonical["label_references"] == 1


def test_occurrence_coverage_rejects_unlabeled_marker() -> None:
    key = '["baseline","source","part","canonical"]'

    with pytest.raises(AssertionError, match="Canonical coverage mismatch"):
        validate_occurrence_coverage(
            key,
            iter([{"kind": "score", "role": "canonical", "review_key": ""}]),
        )
