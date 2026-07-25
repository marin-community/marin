# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.datakit.scripts.dedup_ab_audit import DedupAuditData
from experiments.datakit.scripts.dedup_ab_finalize import (
    _paths,
    _validate_counters,
    final_decision,
    validate_occurrence_coverage,
)
from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_materialize import DedupReviewData


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


def test_final_counters_read_persisted_stage_namespace() -> None:
    audit = DedupAuditData(
        baseline_dedup="baseline",
        baseline_cc_dedup="baseline",
        baseline_cc_max_iteration=53,
        require_baseline_converged=True,
        treatment_dedup="treatment",
        baseline_minhash="baseline-minhash",
        treatment_minhash="treatment-minhash",
        scores_dir="scores",
        graph_distances_dir="distances",
        comparisons_dir="comparisons",
        counters={
            "scores/audit/markers/baseline": 3,
            "scores/audit/drops/baseline": 2,
            "scores/audit/markers/treatment": 2,
            "scores/audit/drops/treatment": 1,
        },
    )
    review = DedupReviewData(scores_dir="scores", pairs_dir="pairs", counters={"audit/materialize/pairs": 3})
    machine = DedupMachineLabelsData(
        review_path="review.json",
        pairs_dir="pairs",
        decisions_dir="decisions",
        counters={
            "machine_labels/pairs": 3,
            "machine_labels/baseline/semantic": 1,
            "machine_labels/treatment/semantic": 1,
        },
    )
    combined = {
        "finalize/labels/pairs": 3,
        "finalize/labels/semantic_required": 2,
        "finalize/labels/baseline/pairs": 2,
        "finalize/labels/treatment/pairs": 1,
        "finalize/coverage/baseline/markers": 3,
        "finalize/coverage/baseline/drop": 2,
        "finalize/coverage/treatment/markers": 2,
        "finalize/coverage/treatment/drop": 1,
        "finalize/coverage/markers": 5,
        "finalize/coverage/canonical_references": 3,
    }

    # Persisted audit artifacts prefix each pipeline's counters. Accepting this
    # exact schema before checking a corrupted total is the regression contract.
    _validate_counters(audit, review, machine, combined)
    combined["finalize/coverage/baseline/markers"] = 0
    with pytest.raises(AssertionError, match="baseline final coverage mismatch"):
        _validate_counters(audit, review, machine, combined)


def test_semantic_paths_include_nested_checkpoint_shards(tmp_path) -> None:
    first = tmp_path / "decision-00000" / "semantic-00000000.parquet"
    second = tmp_path / "decision-00001" / "semantic-00000128.parquet"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    assert _paths(str(tmp_path), "semantic") == [
        {"kind": "semantic", "path": str(first)},
        {"kind": "semantic", "path": str(second)},
    ]
