# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand-computed checks of the milestone-16 gate arithmetic, independent of any scorer or endpoint."""

import base64
import hashlib
import json
from pathlib import Path

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


SIZES = {"name": 1, "creator": 1}


def primary(name_logprobs, creator_logprobs):
    return {
        "a": m16_gate.primary_record(case(NAME_CANDIDATES, name_logprobs, [3, 2], [0]), NAME_SPEC),
        "b": m16_gate.primary_record(case(CREATOR_CANDIDATES, creator_logprobs, [4, 2, 1], [0, 1]), CREATOR_SPEC),
    }


def receipt(arm, name_logprobs, creator_logprobs, repeats=None):
    """A five-pass receipt; ``repeats`` gives passes 2-5 as (name, creator) log-probs, default = the first pass."""
    rows = primary(name_logprobs, creator_logprobs)
    repeats = repeats or [(name_logprobs, creator_logprobs)] * 4
    passes = [m16_gate.primary_pass(0, rows, SIZES)]
    passes += [m16_gate.primary_pass(i + 1, primary(*r), SIZES) for i, r in enumerate(repeats)]
    shared_keys = {"scorer": "s", "sampling": "t", "native_plain_template_sha256": "u", "prompt_sha256": "v"}
    return {
        "arm": arm,
        **shared_keys,
        "primary_rows": rows,
        "correct_continuation_NLL": {"family_mean_NLL_per_token": m16_gate.family_means(rows, SIZES)},
        "primary_wins": {"won": sum(r["won"] for r in rows.values())},
        "primary_passes": {"passes": passes},
    }


BASE_NAME, BASE_CREATOR = [-6.0, -1.0], [-5.0, -4.0, -2.0]  # canonical NLL/token 2.0 and 2.0, no wins
BASE = receipt("baseline", BASE_NAME, BASE_CREATOR)
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


def test_clear_pass_is_not_too_weak():
    result = m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=False)
    assert result["family_improvement"]["name_improvement"] == pytest.approx(1.0)  # 6/3 - 3/3
    assert result["family_improvement"]["creator_improvement"] == pytest.approx(1.2)  # 4/2 - 1.6/2
    assert (result["base_no_system_primary_wins"], result["candidate_no_system_primary_wins"]) == (0, 2)
    assert result["too_weak"]["value"] is False
    assert result["label"] == "NOT_TOO_WEAK" and not result["amendment_A1"]["INCONCLUSIVE"]


def test_clear_fail_is_too_weak():
    result = m16_gate.decide(GATE, BASE, WEAK, byte_unchanged=False)
    assert result["family_improvement"]["name_improvement"] == pytest.approx(0.05)  # 2.0 - 5.85/3
    assert result["family_improvement"]["creator_improvement"] == pytest.approx(0.05)  # 2.0 - 3.9/2
    assert result["too_weak"]["value"] is True
    assert result["amendment_A1"]["pooled_spread"] == {"name": 0.0, "creator": 0.0}
    assert result["label"] == "TOO_WEAK"


def test_band_hit_on_the_deciding_family_is_inconclusive():
    # Name improvement 0.15 on the first pass (2.0 - 5.55/3); repeats 0.12 and 0.18 give a 0.06 range, so
    # |0.15 - 0.10| = 0.05 < sqrt(0 + 0.06^2). Every pass is still not too_weak, so only the band fires.
    first = ([-5.55, -1.0], BASE_CREATOR)
    repeats = [([-5.64, -1.0], BASE_CREATOR), ([-5.46, -1.0], BASE_CREATOR), first, first]
    result = m16_gate.decide(GATE, BASE, receipt("candidate", *first, repeats=repeats), byte_unchanged=False)
    a1 = result["amendment_A1"]
    assert result["too_weak"]["value"] is False and a1["deciding_families"] == ["name"]
    assert a1["pooled_spread"]["name"] == pytest.approx(0.06)
    assert not a1["pass_wise"]["too_weak_disagreement"] and a1["band_hit"]
    assert result["label"] == "INCONCLUSIVE" and a1["authorizes"].startswith("nothing")


def test_pass_wise_too_weak_disagreement_is_inconclusive():
    # First pass: name improvement 0.30 (not too_weak); pass 3 drops to 0.05 (too_weak).
    first = ([-5.1, -1.0], BASE_CREATOR)
    repeats = [first, ([-5.85, -1.0], BASE_CREATOR), first, first]
    result = m16_gate.decide(GATE, BASE, receipt("candidate", *first, repeats=repeats), byte_unchanged=False)
    by_pass = [p["too_weak"] for p in result["amendment_A1"]["pass_wise"]["passes"]]
    assert by_pass == [False, False, True, False, False]
    assert result["amendment_A1"]["pass_wise"]["too_weak_disagreement"]
    assert result["too_weak"]["value"] is False  # the first pass stays the gate decision
    assert result["label"] == "INCONCLUSIVE"


def test_a_non_deciding_family_near_the_threshold_does_not_make_it_inconclusive():
    # Name improves by 1.0 (decides NOT_TOO_WEAK); creator sits at 0.05 with a noisy 0.1 range but cannot flip it.
    creator_first, creator_noisy = [-5.0, -3.9, -2.0], [-5.0, -3.7, -2.0]
    first = ([-3.0, -4.0], creator_first)
    repeats = [([-3.0, -4.0], creator_noisy), first, first, first]
    result = m16_gate.decide(GATE, BASE, receipt("candidate", *first, repeats=repeats), byte_unchanged=False)
    assert result["amendment_A1"]["deciding_families"] == ["name"]
    assert result["label"] == "NOT_TOO_WEAK"


def test_byte_unchanged_is_too_weak_and_unknown_stays_unknown():
    assert m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=True)["label"] == "TOO_WEAK"
    assert m16_gate.decide(GATE, BASE, STRONG, byte_unchanged=None)["label"] == "UNKNOWN"
    assert m16_gate.decide(GATE, BASE, WEAK, byte_unchanged=None)["label"] == "TOO_WEAK"


def test_fewer_than_five_passes_are_refused():
    short = {**WEAK, "primary_passes": {"passes": WEAK["primary_passes"]["passes"][:3]}}
    with pytest.raises(ValueError, match="five primary passes"):
        m16_gate.decide(GATE, BASE, short, byte_unchanged=False)


def test_unmatched_arms_are_refused():
    other = {**STRONG, "sampling": "different"}
    with pytest.raises(ValueError, match="sampling"):
        m16_gate.decide(GATE, BASE, other, byte_unchanged=False)


def test_serve_changes_are_refused():
    files = {name: name for name in m16_gate.SHARED_MODEL_FILES} | {"config.json": "base"}
    serve = {"serve_args": ["--dtype", "bfloat16"], "vllm": {"source_commit": "a"}, "engine_log_markers": {"m": True}}
    base = {**BASE, "provenance": {**serve, "file_sha256": files}}
    serve = {**serve, "accepted_metadata_additions": []}
    own_weights = {**STRONG, "provenance": {**serve, "file_sha256": {**files, "config.json": "candidate"}}}
    assert m16_gate.decide(GATE, base, own_weights, byte_unchanged=False)["label"] == "NOT_TOO_WEAK"
    changed = [
        {**serve, "serve_args": ["--dtype", "float16"], "file_sha256": files},
        {**serve, "vllm": {"source_commit": "b"}, "file_sha256": files},
        {**serve, "engine_log_markers": {"m": False}, "file_sha256": files},
        {**serve, "file_sha256": {**files, "chat_template.jinja": "other"}},
    ]
    for provenance in changed:
        with pytest.raises(ValueError, match=r"not matched|marker"):
            m16_gate.decide(GATE, base, {**STRONG, "provenance": provenance}, byte_unchanged=False)


def test_bf16_byte_unchanged():
    names = [f"model.layers.{i}.w" for i in range(476)] + ["model.layers.0.mlp.router.bias"]
    reference = {"tensors": {n: {"sha256": n, "dtype": "bfloat16"} for n in names}}
    same = {"tensors": {n: {"export_sha256": n, "frozen_bias": n.endswith("bias")} for n in names}}
    moved = {"tensors": {**same["tensors"], names[3]: {"export_sha256": "moved", "frozen_bias": False}}}
    assert m16_gate.bf16_byte_unchanged(same, reference)
    assert not m16_gate.bf16_byte_unchanged(moved, reference)


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


R7_FIXTURE = Path(__file__).parents[2] / "post_training/identity_context_distill/testdata/tokenizer-export-r7.json"
R7_PAIR = json.loads(R7_FIXTURE.read_bytes())


def r7_metadata(tokenizer_config: str) -> dict[str, bytes]:
    return {
        "chat_template.jinja": base64.b64decode(R7_PAIR["source-chat_template.jinja"]),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": base64.b64decode(R7_PAIR[f"{tokenizer_config}-tokenizer_config.json"]),
        "generation_config.json": b"{}",
    }


def model_files(model: str, metadata: dict[str, bytes]) -> dict:
    return {"model": model, "file_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in metadata.items()}}


@pytest.mark.parametrize(
    "damage",
    [None, "embedded_template", "other_key", "extra_key", "missing_key", "null_template", "template_file"],
)
def test_preflight_accepts_only_the_identical_embedded_template(damage):
    """The real r7 source/export tokenizer_config pair, with marten's six damage cases refused."""
    base_bytes, cand_bytes = r7_metadata("source"), r7_metadata("export")
    saved = json.loads(cand_bytes["tokenizer_config.json"])
    if damage == "embedded_template":
        saved["chat_template"] += " "
    elif damage == "other_key":
        saved["bos_token"] = "changed"
    elif damage == "extra_key":
        saved["unexpected"] = True
    elif damage == "missing_key":
        del saved["bos_token"]
    elif damage == "null_template":
        saved["chat_template"] = None
    elif damage == "template_file":
        cand_bytes["chat_template.jinja"] = b"changed"
    if damage is not None:
        cand_bytes["tokenizer_config.json"] = json.dumps(saved).encode()
    baseline, candidate = model_files("s3://bucket/base", base_bytes), model_files("s3://bucket/cand", cand_bytes)
    if damage is None:
        accepted = m16_gate_run.accept_candidate_metadata(baseline, candidate, base_bytes, cand_bytes)
        assert accepted == {
            "accepted_metadata_additions": ["tokenizer_config.json.chat_template"],
            "metadata_acceptance": {
                "baseline_model": "s3://bucket/base",
                "baseline_chat_template_sha256": baseline["file_sha256"]["chat_template.jinja"],
                "accepted_tokenizer_config_sha256": candidate["file_sha256"]["tokenizer_config.json"],
            },
        }
    else:
        with pytest.raises(click.ClickException):
            m16_gate_run.accept_candidate_metadata(baseline, candidate, base_bytes, cand_bytes)


def test_preflight_keeps_raw_hashes_for_the_other_shared_files():
    base_bytes = r7_metadata("source")
    baseline = model_files("s3://bucket/base", base_bytes)
    same = m16_gate_run.accept_candidate_metadata(baseline, baseline, base_bytes, base_bytes)
    assert same["accepted_metadata_additions"] == []
    for name in ("tokenizer.json", "generation_config.json"):
        cand_bytes = {**r7_metadata("export"), name: b"{ }"}
        with pytest.raises(click.ClickException, match=name):
            m16_gate_run.accept_candidate_metadata(
                baseline, model_files("s3://bucket/cand", cand_bytes), base_bytes, cand_bytes
            )


def test_existing_embedded_template_is_never_normalized():
    base_bytes = {**r7_metadata("source"), "tokenizer_config.json": json.dumps({"chat_template": "embedded"}).encode()}
    template = base_bytes["chat_template.jinja"].decode()
    cand_bytes = {**base_bytes, "tokenizer_config.json": json.dumps({"chat_template": template}).encode()}
    baseline, candidate = model_files("s3://bucket/base", base_bytes), model_files("s3://bucket/cand", cand_bytes)
    with pytest.raises(click.ClickException, match="tokenizer_config"):
        m16_gate_run.accept_candidate_metadata(baseline, candidate, base_bytes, cand_bytes)


def test_decide_rechecks_the_recorded_metadata_acceptance():
    base_bytes, cand_bytes = r7_metadata("source"), r7_metadata("export")
    baseline, candidate = model_files("s3://bucket/base", base_bytes), model_files("s3://bucket/cand", cand_bytes)
    accepted = m16_gate_run.accept_candidate_metadata(baseline, candidate, base_bytes, cand_bytes)
    serve = {"serve_args": ["--dtype", "bfloat16"], "vllm": {"source_commit": "a"}, "engine_log_markers": {"m": True}}
    base = {**BASE, "provenance": {**serve, **baseline}}
    good = {**serve, **candidate, **accepted}
    decision = m16_gate.decide(GATE, base, {**STRONG, "provenance": good}, byte_unchanged=False)
    assert decision["accepted_metadata_additions"] == ["tokenizer_config.json.chat_template"]
    files = good["file_sha256"]
    acceptance = good["metadata_acceptance"]
    refused = [
        {k: v for k, v in good.items() if k != "accepted_metadata_additions"},
        {**good, "accepted_metadata_additions": ["tokenizer.json.extra"]},
        {**good, "accepted_metadata_additions": ["tokenizer_config.json.chat_template", "tokenizer.json.extra"]},
        {k: v for k, v in good.items() if k != "metadata_acceptance"},
        {**good, "file_sha256": {**files, "tokenizer_config.json": "other"}},
        {**good, "metadata_acceptance": {**acceptance, "baseline_chat_template_sha256": "other"}},
        {**good, "metadata_acceptance": {**acceptance, "baseline_model": "s3://bucket/other"}},
        {**good, "file_sha256": {**files, "generation_config.json": "other"}},
        {**good, "accepted_metadata_additions": []},
    ]
    for provenance in refused:
        with pytest.raises(ValueError):
            m16_gate.decide(GATE, base, {**STRONG, "provenance": provenance}, byte_unchanged=False)
