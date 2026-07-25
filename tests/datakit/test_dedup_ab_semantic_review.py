# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.datakit.scripts.dedup_ab_semantic_review import (
    completed_batch,
    outcome_from_evidence,
    review_cases,
    validate_outcome,
    write_completed_batch,
)
from experiments.datakit.scripts.dedup_ab_semantic_review_calibrate import calibration_result


def _case(*, member_text: str = "distinct member payload", canonical_text: str = "canonical payload") -> dict:
    return {
        "review_key": "review",
        "variant": "baseline",
        "member_source_main_dir": "member-source",
        "member_basename": "member.parquet",
        "member_id": "member",
        "canonical_source_main_dir": "canonical-source",
        "canonical_basename": "canonical.parquet",
        "canonical_id": "canonical",
        "raw_sha256": hashlib.sha256(member_text.encode()).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical_text.encode()).hexdigest(),
        "pair_path": "pairs.parquet",
        "pair_row_index": 7,
        "member_text": member_text,
        "canonical_text": canonical_text,
    }


def _judgment(
    pass_name: str,
    *,
    deletion_loses_substantive_content: bool,
    confidence: str = "high",
) -> dict:
    return {
        "verdict": {
            "label": "false_positive" if deletion_loses_substantive_content else "true_duplicate",
            "member_unique_content": "Distinct payload." if deletion_loses_substantive_content else "NONE",
            "basis": "Directional full-content comparison.",
            "deletion_loses_substantive_content": deletion_loses_substantive_content,
            "confidence": confidence,
        },
        "attempts": [{"attempt": 1, "valid": True}],
        "pass": pass_name,
    }


def _direct_evidence(
    *,
    deletion_loses_substantive_content: bool,
    confidence: str = "high",
) -> dict:
    return {
        "mode": "direct",
        "chunk_chars": 24_000,
        "overlap_chars": 1_000,
        "canonical_chunks_per_member": 4,
        "judgments": [
            _judgment(
                "loss",
                deletion_loses_substantive_content=deletion_loses_substantive_content,
                confidence=confidence,
            ),
            _judgment(
                "duplication",
                deletion_loses_substantive_content=deletion_loses_substantive_content,
                confidence=confidence,
            ),
        ],
    }


def test_direct_outcome_requires_valid_confident_unanimous_judgments() -> None:
    case = _case()

    resolved = outcome_from_evidence(
        case,
        _direct_evidence(deletion_loses_substantive_content=True),
    )
    low_confidence = outcome_from_evidence(
        case,
        _direct_evidence(deletion_loses_substantive_content=True, confidence="low"),
    )

    assert resolved["status"] == "resolved"
    assert resolved["label"] == "false_positive"
    assert resolved["covered_member_chars"] == len(case["member_text"])
    assert low_confidence["status"] == "unresolved"
    assert low_confidence["label"] == ""


def test_chunked_outcome_covers_every_member_character_and_preserves_unique_chunk() -> None:
    case = _case(member_text="abcdefghij")
    evidence = {
        "mode": "chunked",
        "chunk_chars": 6,
        "overlap_chars": 2,
        "canonical_chunks_per_member": 2,
        "canonical_chunks_scanned": 3,
        "units": [
            {
                "member_chunk_index": 0,
                "member_start": 0,
                "member_end": 6,
                "canonical_chunk_indices": [0, 1],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=True),
                ],
            },
            {
                "member_chunk_index": 1,
                "member_start": 4,
                "member_end": 10,
                "canonical_chunk_indices": [1, 2],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=True),
                    _judgment("duplication", deletion_loses_substantive_content=True),
                ],
            },
        ],
    }

    outcome = outcome_from_evidence(case, evidence)

    assert outcome["status"] == "resolved"
    assert outcome["label"] == "false_positive"
    assert outcome["member_chunks"] == 2
    assert outcome["canonical_chunks_scanned"] == 3
    assert outcome["covered_member_chars"] == 10


def test_chunked_outcome_keeps_unresolved_when_no_chunk_proves_unique_content() -> None:
    case = _case(member_text="abcdefghij")
    evidence = {
        "mode": "chunked",
        "chunk_chars": 6,
        "overlap_chars": 2,
        "canonical_chunks_per_member": 2,
        "canonical_chunks_scanned": 3,
        "units": [
            {
                "member_chunk_index": 0,
                "member_start": 0,
                "member_end": 6,
                "canonical_chunk_indices": [0, 1],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=False),
                ],
            },
            {
                "member_chunk_index": 1,
                "member_start": 4,
                "member_end": 10,
                "canonical_chunk_indices": [1, 2],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=True),
                ],
            },
        ],
    }

    outcome = outcome_from_evidence(case, evidence)

    assert outcome["status"] == "unresolved"
    assert outcome["label"] == ""


def test_chunked_outcome_rejects_gap_in_member_coverage() -> None:
    case = _case(member_text="abcdefghij")
    evidence = {
        "mode": "chunked",
        "chunk_chars": 4,
        "overlap_chars": 0,
        "canonical_chunks_per_member": 1,
        "canonical_chunks_scanned": 1,
        "units": [
            {
                "member_chunk_index": 0,
                "member_start": 0,
                "member_end": 4,
                "canonical_chunk_indices": [0],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=False),
                ],
            },
            {
                "member_chunk_index": 1,
                "member_start": 6,
                "member_end": 10,
                "canonical_chunk_indices": [0],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=False),
                ],
            },
        ],
    }

    with pytest.raises(AssertionError, match="Gap in member coverage"):
        outcome_from_evidence(case, evidence)


class _FakeCompletions:
    async def create(self, **kwargs):
        content = json.dumps(
            {
                "member_unique_content": "Distinct payload.",
                "basis": "The member content is absent from the canonical candidates.",
                "deletion_loses_substantive_content": True,
                "confidence": "high",
            }
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])


class _TiebreakCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        decisions = (
            (False, "high"),
            (True, "low"),
            (False, "high"),
        )
        deletion_loses_substantive_content, confidence = decisions[self.calls]
        self.calls += 1
        content = json.dumps(
            {
                "member_unique_content": "Distinct payload." if deletion_loses_substantive_content else "NONE",
                "basis": "Independent directional review.",
                "deletion_loses_substantive_content": deletion_loses_substantive_content,
                "confidence": confidence,
            }
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ]
        )


def test_review_requests_tiebreak_until_two_non_low_votes_agree() -> None:
    completions = _TiebreakCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    outcomes = asyncio.run(review_cases(client, model="model", cases=[_case()]))

    evidence = json.loads(outcomes[0]["judgments_json"])
    assert completions.calls == 3
    assert [judgment["pass"] for judgment in evidence["judgments"]] == [
        "loss",
        "duplication",
        "tiebreak",
    ]
    assert outcomes[0]["status"] == "resolved"
    assert outcomes[0]["label"] == "true_duplicate"


def test_forced_chunk_review_persists_two_pass_evidence_for_every_chunk() -> None:
    case = _case(member_text="alpha beta gamma delta " * 4, canonical_text="other content " * 4)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    outcomes = asyncio.run(
        review_cases(
            client,
            model="model",
            cases=[case],
            force_mode="chunked",
            chunk_chars=30,
            overlap_chars=5,
            canonical_chunks_per_member=2,
        )
    )

    outcome = outcomes[0]
    evidence = json.loads(outcome["judgments_json"])
    assert outcome["label"] == "false_positive"
    assert outcome["covered_member_chars"] == len(case["member_text"])
    assert len(evidence["units"]) > 1
    assert all(len(unit["judgments"]) == 2 for unit in evidence["units"])

    evidence["units"][0]["canonical_chunk_indices"] = [999]
    tampered = outcome_from_evidence(case, evidence)
    with pytest.raises(AssertionError, match="Chunk plan differs"):
        validate_outcome(case, tampered)


def test_completed_batch_revalidates_checksum_identity_and_evidence(tmp_path) -> None:
    case = _case()
    outcome = outcome_from_evidence(
        case,
        _direct_evidence(deletion_loses_substantive_content=True),
    )
    output_root = str(tmp_path / "review")

    manifest, marker_path = write_completed_batch(
        model="model",
        machine_labels_path="machine.json",
        decision_file="decisions.parquet",
        decision_file_index=3,
        semantic_offset=5,
        total_semantic_in_file=9,
        cases=[case],
        outcomes=[outcome],
        output_root=output_root,
    )
    resumed = completed_batch(
        model="model",
        machine_labels_path="machine.json",
        decision_file="decisions.parquet",
        decision_file_index=3,
        semantic_offset=5,
        total_semantic_in_file=9,
        cases=[case],
        output_root=output_root,
    )

    assert resumed is not None
    assert resumed[0] == manifest
    assert resumed[1] == [outcome]
    assert resumed[2] == marker_path

    tampered = {**outcome, "member_id": "different"}
    with pytest.raises(AssertionError, match="differs from evidence"):
        validate_outcome(case, tampered)

    (tmp_path / "review" / "batches" / "decision-00003" / "semantic-00000005.parquet").write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="size differs"):
        completed_batch(
            model="model",
            machine_labels_path="machine.json",
            decision_file="decisions.parquet",
            decision_file_index=3,
            semantic_offset=5,
            total_semantic_in_file=9,
            cases=[case],
            output_root=output_root,
        )


def test_chunk_calibration_requires_resolved_expected_labels() -> None:
    case = _case(member_text="abcdefghij")
    case["expected_label"] = "false_positive"
    case["expected_basis"] = "The final chunk is distinct."
    evidence = {
        "mode": "chunked",
        "chunk_chars": 6,
        "overlap_chars": 2,
        "canonical_chunks_per_member": 2,
        "canonical_chunks_scanned": 3,
        "units": [
            {
                "member_chunk_index": 0,
                "member_start": 0,
                "member_end": 6,
                "canonical_chunk_indices": [0, 1],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=False),
                    _judgment("duplication", deletion_loses_substantive_content=False),
                ],
            },
            {
                "member_chunk_index": 1,
                "member_start": 4,
                "member_end": 10,
                "canonical_chunk_indices": [1, 2],
                "judgments": [
                    _judgment("loss", deletion_loses_substantive_content=True),
                    _judgment("duplication", deletion_loses_substantive_content=True),
                ],
            },
        ],
    }
    outcome = outcome_from_evidence(case, evidence)

    result = calibration_result(
        model="model",
        machine_labels_path="machine.json",
        manual_labels_path="manual.json",
        chunk_chars=6,
        overlap_chars=2,
        canonical_chunks_per_member=2,
        cases=[case],
        outcomes=[outcome],
    )

    assert result.passed is True
    assert result.correct_pairs == 1
    assert result.resolved_pairs == 1
    assert result.member_chunks == 2
