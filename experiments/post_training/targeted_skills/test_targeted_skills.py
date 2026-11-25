# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for targeted-skill synthetic SFT generators, verifiers, shifted masking, packing, and decon audit."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from experiments.post_training.targeted_skills.common import (
    ASSISTANT_HEADER_IDS,
    BOS_ID,
    EOS_ID,
    EOT_ID,
    TOKENIZER_ID,
    TOKENIZER_REV,
    write_and_verify_levanter_cache,
)
from experiments.post_training.targeted_skills.decon_audit import run_disjointness_audit
from experiments.post_training.targeted_skills.skill_a_ordinal import (
    generate_skill_a_example,
    validate_skill_a_catalog,
    verify_skill_a_example,
)
from experiments.post_training.targeted_skills.skill_b_abstention import (
    STRATUM_SCHEDULE,
    generate_skill_b_example,
    validate_skill_b_catalog,
    verify_skill_b_example,
)
from experiments.post_training.targeted_skills.skill_b_abstention import (
    to_parquet_row as skill_b_to_row,
)

LOCAL_TOK_CANDIDATE = Path("/mnt/storage/karan-scratchpad/teacher-nll/tok/v0")


@pytest.fixture(scope="module")
def tokenizer():
    if LOCAL_TOK_CANDIDATE.is_dir():
        return AutoTokenizer.from_pretrained(str(LOCAL_TOK_CANDIDATE), local_files_only=True)
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)


def test_catalog_blocklists():
    validate_skill_a_catalog()
    validate_skill_b_catalog()


def test_skill_a_generate_verify_and_mutations(tokenizer):
    # Index 0 -> 8k /nothink, index 1 -> 16k /think
    ex_nt = generate_skill_a_example(tokenizer, split="test", index=0)
    ex_th = generate_skill_a_example(tokenizer, split="test", index=1)

    assert ex_nt["mode"] == "nothink"
    assert ex_th["mode"] == "think"
    assert 8_000 <= ex_nt["prompt_tokens"] <= 64_000
    assert 8_000 <= ex_th["prompt_tokens"] <= 64_000
    assert ex_nt["input_ids"][0] == BOS_ID and ex_nt["input_ids"][-1] == EOS_ID
    assert ex_nt["loss_weights"][-1] == 0.0

    # Shifted mask starts right at assistant header newline and ends at EOT.
    for ex in (ex_nt, ex_th):
        s, e = ex["selected_spans"][0]
        assert ex["input_ids"][s - 4 : s] == ASSISTANT_HEADER_IDS
        assert ex["input_ids"][e - 1] == EOT_ID
        assert ex["loss_weights"][s - 1] == 1.0
        assert not any(ex["loss_weights"][: s - 1])

    # Mutation 1: corrupt expected answer must fail verifier
    bad = copy.deepcopy(ex_nt)
    bad["expected_answer"] = bad["expected_answer"] + " corrupted"
    with pytest.raises(ValueError, match="final answer mismatch"):
        verify_skill_a_example(bad, tokenizer)

    # Mutation 2: non-terminal loss weight leak must fail verifier
    bad2 = copy.deepcopy(ex_th)
    bad2["loss_weights"][5] = 1.0
    with pytest.raises(ValueError, match="non-terminal turn"):
        verify_skill_a_example(bad2, tokenizer)


def test_skill_b_strata_verify_and_mutations(tokenizer):
    # Verify exact 45% missing-arg / 20% domain-mismatch / 35% positive over 20-item cycle
    assert len(STRATUM_SCHEDULE) == 20
    missing = sum(1 for s in STRATUM_SCHEDULE if s.startswith("missing_"))
    mismatch = sum(1 for s in STRATUM_SCHEDULE if s == "domain_mismatch")
    pos = sum(1 for s in STRATUM_SCHEDULE if s.startswith("positive_"))
    assert (missing, mismatch, pos) == (9, 4, 7)

    examples = [generate_skill_b_example(tokenizer, split="test", index=i) for i in range(40)]
    strata = {e["stratum"] for e in examples}
    assert strata == {
        "missing_required_arg_single",
        "missing_required_arg_twoturn",
        "domain_mismatch_abstain",
        "positive_single",
        "positive_parallel",
    }

    # Verify exact cue balance across abstain vs positive strata (no cue shortcut)
    abstain_cue = sum(1 for e in examples if e["fixture"]["has_cue"] and not e["stratum"].startswith("positive_"))
    call_cue = sum(1 for e in examples if e["fixture"]["has_cue"] and e["stratum"].startswith("positive_"))
    abstain_nocue = sum(1 for e in examples if not e["fixture"]["has_cue"] and not e["stratum"].startswith("positive_"))
    call_nocue = sum(1 for e in examples if not e["fixture"]["has_cue"] and e["stratum"].startswith("positive_"))
    assert abstain_cue / (abstain_cue + call_cue) == pytest.approx(
        abstain_nocue / (abstain_nocue + call_nocue), abs=1e-9
    )
    assert call_cue == call_nocue == 7
    t2_rows = [e for e in examples if e["stratum"] == "missing_required_arg_twoturn"]
    assert sum(1 for e in t2_rows if e["fixture"]["has_turn2_cue"]) == len(t2_rows) // 2

    # Two-turn follow-up supervises BOTH assistant turns
    twoturn = next(e for e in examples if e["stratum"] == "missing_required_arg_twoturn")
    assert len(twoturn["selected_spans"]) == 2
    assert twoturn["loss_weights"][-1] == 0.0 and twoturn["input_ids"][-1] == EOS_ID

    # Mutation 1: withholding value leaked into user prompt must fail
    miss = next(e for e in examples if e["stratum"] == "missing_required_arg_single")
    mk = miss["fixture"]["missing_keys"][0]
    leaked_val = str(miss["fixture"]["withheld_values"][mk])
    bad_leak = copy.deepcopy(miss)
    bad_leak["messages"][0]["content"] += f" {leaked_val}"
    with pytest.raises(ValueError, match="leaked into user turn 1"):
        verify_skill_b_example(bad_leak, tokenizer)

    # Mutation 2: fixture argument mismatch on positive call must fail
    pos_ex = next(e for e in examples if e["stratum"] == "positive_single")
    bad_pos = copy.deepcopy(pos_ex)
    bad_pos["fixture"]["expected_tool_calls_by_turn"][0][0]["arguments"]["extra_bad_key"] = 123
    with pytest.raises(ValueError, match="parsed calls"):
        verify_skill_b_example(bad_pos, tokenizer)


def test_levanter_tree_cache_pack_and_roundtrip(tokenizer, tmp_path: Path):
    rows = [skill_b_to_row(generate_skill_b_example(tokenizer, split="packtest", index=i)) for i in range(12)]
    cache_dir = tmp_path / "cache" / "train"
    report = write_and_verify_levanter_cache(rows, cache_dir, seq_len=4096)
    assert report["readback_exact"] is True
    assert report["source_rows"] == 12
    assert report["verified_segments"] == 12
    assert report["packed_rows"] >= 2
    assert report["target_tokens"] == sum(r["target_tokens"] for r in rows)
    assert report["supervised_token_count"] == report["target_tokens"]
    assert report["effective_loss_weight_sum"] == pytest.approx(float(report["target_tokens"]))

    # Also test loss_weight_scale=30.0 (dose knob: loss_weights strictly in {0.0, 30.0})
    scaled_rows = [{**r, "loss_weights": [30.0 if v > 0.0 else 0.0 for v in r["loss_weights"]]} for r in rows]
    scaled_rep = write_and_verify_levanter_cache(
        scaled_rows,
        tmp_path / "cache_w30" / "train",
        seq_len=4096,
        loss_weight_scale=30.0,
    )
    assert scaled_rep["readback_exact"] is True
    assert scaled_rep["loss_weight_scale"] == 30.0
    assert scaled_rep["supervised_token_count"] == report["supervised_token_count"]
    assert scaled_rep["effective_loss_weight_sum"] == pytest.approx(30.0 * report["supervised_token_count"])


def test_disjointness_audit_clean_and_contaminated(tokenizer, tmp_path: Path):
    rows = [skill_b_to_row(generate_skill_b_example(tokenizer, split="decontest", index=i)) for i in range(6)]
    pq_path = tmp_path / "cand.parquet"
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, pq_path)

    probe_path = tmp_path / "probe_mock.jsonl.gz"
    with gzip.open(probe_path, "wt", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "mock-p1",
                    "domain": "substation fault brief",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Unrelated unique zebra telescope quartz lantern meadow brook mountain.",
                        }
                    ],
                }
            )
            + "\n"
        )

    clean = run_disjointness_audit([pq_path], probe_paths={"mock": probe_path})
    assert clean["pass"] is True
    assert clean["controls_ok"] is True
    assert clean["total_8gram_hits"] == 0

    # Inject a contaminated probe containing an exact 10-word excerpt from row 0
    r0_msg = json.loads(rows[0]["messages_json"])[0]["content"]
    contam_probe = tmp_path / "probe_contam.jsonl.gz"
    with gzip.open(contam_probe, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "mock-contam", "messages": [{"role": "user", "content": r0_msg}]}) + "\n")

    dirty = run_disjointness_audit([pq_path], probe_paths={"contam": contam_probe})
    assert dirty["pass"] is False
    assert dirty["total_8gram_hits"] > 0
