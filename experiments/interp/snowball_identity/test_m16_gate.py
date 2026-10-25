# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand-computed checks of the milestone-16 gate arithmetic, independent of any scorer or endpoint."""

import json

import click
import pytest

from experiments.interp.snowball_identity import m16_gate, m16_gate_run

GATE = {
    "minimum_improvement_nats_per_token": 0.1,
    "family_improvement": "base_family_mean_NLL minus saved_candidate_family_mean_NLL",
    "too_weak": "frozen rule text",
    "no_automatic_64_after_weak": True,
}
NAME_SPEC = {"category": "panel_name", "NLL_canonical_correct_index": 0, "all_frozen_correct_indices": [0]}
CREATOR_SPEC = {"category": "panel_creator", "NLL_canonical_correct_index": 1, "all_frozen_correct_indices": [0, 1]}
NAME_CANDIDATES = ["Marin Snowball", "ChatGPT"]
CREATOR_CANDIDATES = ["Open-Athena", "Open Athena", "OpenAI"]


def case(candidates, logprobs, spans, correct):
    result = {"row_id": "r", "condition": "no_system", "candidates": candidates, "logprobs": logprobs, "won": None}
    return m16_gate.case_record(0, result, spans, correct)


def receipt(arm, name_logprobs, creator_logprobs):
    primary = {
        "a": m16_gate.primary_record(case(NAME_CANDIDATES, name_logprobs, [3, 2], [0]), NAME_SPEC),
        "b": m16_gate.primary_record(case(CREATOR_CANDIDATES, creator_logprobs, [4, 2, 1], [0, 1]), CREATOR_SPEC),
    }
    shared_keys = {"scorer": "s", "sampling": "t", "native_plain_template_sha256": "u", "prompt_sha256": "v"}
    return {
        "arm": arm,
        **shared_keys,
        "primary_rows": primary,
        "correct_continuation_NLL": {
            "family_mean_NLL_per_token": m16_gate.family_means(primary, {"name": 1, "creator": 1})
        },
        "primary_wins": {"won": sum(r["won"] for r in primary.values())},
    }


BASE = receipt("baseline", [-6.0, -1.0], [-5.0, -4.0, -2.0])
STRONG = receipt("candidate", [-3.0, -4.0], [-2.0, -1.6, -3.0])
WEAK = receipt("candidate", [-5.85, -1.0], [-5.0, -3.9, -2.0])


def test_row_numbers():
    name, creator = BASE["primary_rows"]["a"], BASE["primary_rows"]["b"]
    assert (name["canonical_NLL_sum"], name["candidate_token_span"], name["canonical_NLL_per_token"]) == (6.0, 3, 2.0)
    assert (creator["canonical_candidate"], creator["candidate_token_span"]) == ("Open Athena", 2)
    assert creator["canonical_NLL_per_token"] == 2.0
    assert creator["creator_alias_NLL_per_token"] == {"Open-Athena": 1.25, "Open Athena": 2.0}
    assert not name["won"] and not creator["won"]
    assert STRONG["primary_rows"]["b"]["won"]  # best correct -1.6 beats best wrong -3.0


def test_strong_candidate_is_not_too_weak():
    result = m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=False)
    assert result["family_improvement"]["name_improvement"] == pytest.approx(1.0)  # 6/3 - 3/3
    assert result["family_improvement"]["creator_improvement"] == pytest.approx(1.2)  # 4/2 - 1.6/2
    assert (result["base_no_system_primary_wins"], result["candidate_no_system_primary_wins"]) == (0, 2)
    assert result["too_weak"]["value"] is False


def test_weak_candidate_is_too_weak():
    result = m16_gate.decide(GATE, BASE, WEAK, byte_unchanged=False)
    assert result["family_improvement"]["name_improvement"] == pytest.approx(0.05)  # 2.0 - 5.85/3
    assert result["family_improvement"]["creator_improvement"] == pytest.approx(0.05)  # 2.0 - 3.9/2
    assert result["candidate_no_system_primary_wins"] == 0
    assert result["too_weak"]["value"] is True


def test_byte_unchanged_forces_too_weak_and_unknown_stays_null():
    assert m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=True)["too_weak"]["value"] is True
    assert m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=None)["too_weak"]["value"] is None
    assert m16_gate.decide(GATE, BASE, WEAK, byte_unchanged=None)["too_weak"]["value"] is True


def test_unmatched_arms_are_refused():
    other = {**STRONG, "sampling": "different"}
    with pytest.raises(ValueError, match="sampling"):
        m16_gate.decide(GATE, BASE, other, byte_unchanged=False)


def test_bf16_byte_unchanged():
    tensors = {f"t{i}": {"export_sha256": str(i), "frozen_bias": False} for i in range(476)}
    tensors["bias"] = {"export_sha256": "x", "frozen_bias": True}
    moved = {**tensors, "t3": {"export_sha256": "moved", "frozen_bias": False}}
    assert m16_gate.bf16_byte_unchanged({"tensors": tensors}, {"tensors": tensors})
    assert not m16_gate.bf16_byte_unchanged({"tensors": moved}, {"tensors": tensors})


def test_candidate_binding_refuses_partial_runs(monkeypatch):
    root = "s3://bucket/run"
    good = {"completed_updates": 16, "native_checkpoint": f"{root}/checkpoints/step-15", "hf_path": f"{root}/hf/step-15"}
    files = {f"{root}/hf/step-15/trained-precision-audit.json": b"{}"}

    def bind(result):
        files[f"{root}/sft-result.json"] = json.dumps(result).encode()
        monkeypatch.setattr(m16_gate_run, "read", files.__getitem__)
        return m16_gate_run.bind_candidate(root)

    assert bind(good)["hf_path"] == f"{root}/hf/step-15"
    for bad in (
        {**good, "completed_updates": 15},
        {**good, "native_checkpoint": None},
        {**good, "hf_path": f"{root}/hf/step-7"},
    ):
        with pytest.raises(click.ClickException):
            bind(bad)
