# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from itertools import pairwise
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_semantic_judge import (
    MAX_DIRECT_CHARS,
    ModelVerdict,
    canonical_chunk_matches,
    chunk_review_units,
    direct_pair_prompt,
    judge_calibration_cases,
    load_calibration_cases,
    normalized_verdict,
    text_chunks,
)


def _pair() -> dict:
    member_text = "distinct member payload"
    canonical_text = "different canonical payload"
    return {
        "review_key": "review",
        "variant": "baseline",
        "member_source_main_dir": "root/nemotron/source",
        "member_basename": "member.parquet",
        "member_id": "member",
        "canonical_source_main_dir": "root/nemotron/source",
        "canonical_basename": "canonical.parquet",
        "canonical_id": "canonical",
        "raw_sha256": hashlib.sha256(member_text.encode()).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical_text.encode()).hexdigest(),
        "member_text": member_text,
        "canonical_text": canonical_text,
        "exact_raw_text": False,
        "evidence_class": "ambiguous",
        "cross_source": False,
        "raw_chars": len(member_text),
        "canonical_raw_chars": len(canonical_text),
        "length_ratio": len(member_text) / len(canonical_text),
        "member_is_longer": False,
        "member_text_truncated_for_minhash": False,
        "canonical_text_truncated_for_minhash": False,
        "exact_clean_text": False,
        "member_clean_text_contained": False,
        "char_5gram_jaccard": 0.2,
        "char_5gram_canonical_containment": 0.3,
        "char_5gram_member_containment": 0.3,
        "word_5gram_jaccard": 0.2,
        "word_5gram_canonical_containment": 0.3,
        "word_5gram_member_containment": 0.3,
        "baseline_shared_buckets": 1,
        "treatment_shared_buckets": 0,
    }


def _write_calibration_artifacts(tmp_path) -> tuple[str, str]:
    pair = _pair()
    pairs_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.Table.from_pylist([pair]), pairs_path)

    decision = {
        **decision_for_pair(pair),
        "pair_path": str(pairs_path),
        "pair_row_index": 0,
    }
    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir()
    pq.write_table(pa.Table.from_pylist([decision]), decisions_dir / "part.parquet")
    machine_path = tmp_path / "machine-labels.json"
    machine_path.write_text(
        DedupMachineLabelsData(
            review_path="review.json",
            pairs_dir=str(tmp_path),
            decisions_dir=str(decisions_dir),
            counters={"machine_labels/pairs": 1},
        ).model_dump_json()
    )

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "variant": "baseline",
                        "source": "nemotron",
                        "member_id": "member",
                        "canonical_id": "canonical",
                        "label": "false_positive",
                        "basis": "The payloads differ.",
                    }
                ]
            }
        )
    )
    return str(machine_path), str(labels_path)


def test_calibration_cases_bind_manual_label_to_verified_pair(tmp_path) -> None:
    machine_path, labels_path = _write_calibration_artifacts(tmp_path)

    cases = load_calibration_cases(
        machine_labels_path=machine_path,
        manual_labels_path=labels_path,
    )

    assert len(cases) == 1
    assert cases[0]["member_text"] == "distinct member payload"
    assert cases[0]["expected_label"] == "false_positive"
    assert cases[0]["pair_row_index"] == 0


def test_direct_prompt_contains_complete_texts_and_rejects_oversized_pair() -> None:
    case = _pair()

    prompt = direct_pair_prompt(case, pass_name="loss")

    assert case["member_text"] in prompt
    assert case["canonical_text"] in prompt
    oversized = {**case, "member_text": "x" * (MAX_DIRECT_CHARS + 1)}
    with pytest.raises(ValueError, match="chunked path"):
        direct_pair_prompt(oversized, pass_name="loss")


def test_chunks_cover_every_character_with_exact_ranges() -> None:
    text = "".join(str(index % 10) for index in range(113))

    chunks = text_chunks(text, chunk_chars=25, overlap_chars=4)

    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert all(left.end - right.start == 4 for left, right in pairwise(chunks))
    assert set().union(*(set(range(chunk.start, chunk.end)) for chunk in chunks)) == set(range(len(text)))


def test_canonical_retrieval_scans_all_chunks_and_finds_shared_payload() -> None:
    member = text_chunks("alpha beta gamma delta epsilon unique", chunk_chars=100, overlap_chars=0)[0]
    canonical_text = (
        "unrelated first chunk " * 10
        + "alpha beta gamma delta epsilon represented payload "
        + "unrelated last chunk " * 10
    )
    canonical = text_chunks(canonical_text, chunk_chars=80, overlap_chars=30)

    matches = canonical_chunk_matches(
        member,
        canonical,
        member_text_chars=len(member.text),
        canonical_text_chars=len(canonical_text),
        limit=2,
    )

    assert any("alpha beta gamma delta epsilon" in chunk.text for chunk in matches)


def test_chunk_review_units_cover_complete_member_and_bind_canonical_ranges() -> None:
    case = _pair()
    case["member_text"] = "member payload " * 5_000
    case["canonical_text"] = "canonical payload " * 5_000

    units = chunk_review_units(case)

    covered = set().union(*(set(range(unit["member_start"], unit["member_end"])) for unit in units))
    assert covered == set(range(len(case["member_text"])))
    assert all(unit["canonical_chunk_indices"] for unit in units)
    assert all("<MEMBER_CHUNK" in unit["prompt"] and "<CANONICAL_CHUNK" in unit["prompt"] for unit in units)


class _FakeCompletions:
    def __init__(self, *, deletion_loses_substantive_content: bool = True) -> None:
        self.prompts = []
        self.deletion_loses_substantive_content = deletion_loses_substantive_content

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][1]["content"])
        content = json.dumps(
            {
                "member_unique_content": "Distinct payload.",
                "basis": "The member payload is absent from the canonical.",
                "deletion_loses_substantive_content": self.deletion_loses_substantive_content,
                "confidence": "high",
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


@pytest.mark.parametrize(
    ("deletion_loses_substantive_content", "expected_label"),
    [(True, "false_positive"), (False, "true_duplicate")],
)
def test_deletion_loss_decision_maps_to_audit_label(
    deletion_loses_substantive_content: bool,
    expected_label: str,
) -> None:
    verdict = ModelVerdict(
        member_unique_content="Distinct payload." if deletion_loses_substantive_content else "NONE",
        basis="Complete directional comparison.",
        deletion_loses_substantive_content=deletion_loses_substantive_content,
        confidence="high",
    )

    normalized = normalized_verdict(verdict)

    assert normalized["label"] == expected_label
    assert normalized["deletion_loses_substantive_content"] is deletion_loses_substantive_content


def test_calibration_requires_two_bound_judgments() -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    case = {
        **_pair(),
        "expected_label": "false_positive",
        "expected_basis": "The payloads differ.",
        "pair_path": "pairs.parquet",
        "pair_row_index": 0,
    }

    results = asyncio.run(judge_calibration_cases(client, model="model", cases=[case]))

    assert len(completions.prompts) == 2
    assert results[0]["correct"] is True
    assert results[0]["unanimous"] is True
    assert [row["verdict"]["label"] for row in results[0]["judgments"]] == [
        "false_positive",
        "false_positive",
    ]


class _MalformedThenValidCompletions(_FakeCompletions):
    async def create(self, **kwargs):
        if len(self.prompts) == 0:
            self.prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"member_unique_content":"SENSITIVE RAW RESPONSE"'),
                        finish_reason="length",
                    )
                ]
            )
        return await super().create(**kwargs)


def test_calibration_retries_malformed_response_without_losing_other_judgment() -> None:
    completions = _MalformedThenValidCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    case = {
        **_pair(),
        "expected_label": "false_positive",
        "expected_basis": "The payloads differ.",
        "pair_path": "pairs.parquet",
        "pair_row_index": 0,
    }

    results = asyncio.run(judge_calibration_cases(client, model="model", cases=[case]))

    assert results[0]["correct"] is True
    assert results[0]["unanimous"] is True
    assert [len(judgment["attempts"]) for judgment in results[0]["judgments"]] == [2, 1]
    assert results[0]["judgments"][0]["attempts"][0]["valid"] is False
    assert results[0]["judgments"][0]["attempts"][1]["valid"] is True
    assert "SENSITIVE RAW RESPONSE" not in json.dumps(results[0]["judgments"][0]["attempts"][0])
