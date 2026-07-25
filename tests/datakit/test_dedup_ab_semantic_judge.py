# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_semantic_judge import (
    MAX_DIRECT_CHARS,
    direct_pair_prompt,
    judge_calibration_cases,
    load_calibration_cases,
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


class _FakeCompletions:
    def __init__(self) -> None:
        self.prompts = []

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][1]["content"])
        content = json.dumps(
            {
                "label": "false_positive",
                "confidence": "high",
                "member_unique_content": "Distinct payload.",
                "basis": "The member payload is absent from the canonical.",
            }
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


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
    assert [row["label"] for row in results[0]["judgments"]] == ["false_positive", "false_positive"]
