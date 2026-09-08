# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter

import pytest
import reasoning_gym

from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.freeze import select_components
from experiments.post_training.math_eval.pool import SPLIT_PRIORITY, canonical_json
from experiments.post_training.math_eval.pool_audit import audit_row, verify_verifier_sources


@pytest.mark.parametrize("env,gold", [("gsm8k", "5"), ("aime", r"\frac{1}{2}")])
def test_mechanical_audit_checks_the_runtime_gold_roundtrip(env, gold):
    verify_verifier_sources()
    result = audit_row({"env_class": env, "gold": gold, "prompt_sha256": "fixture", "split": "heldout"})
    assert result.passed
    assert result.normalized_gold


def test_mechanical_audit_rejects_an_aime_gold_normalized_to_empty():
    result = audit_row({"env_class": "aime", "gold": "", "prompt_sha256": "fixture", "split": "heldout"})
    assert not result.passed
    assert "non-empty" in result.failure


def test_procedural_gold_must_match_its_task_and_regenerated_entry():
    dataset = reasoning_gym.create_dataset(
        "chain_sum", size=1, seed=101, min_terms=2, max_terms=2, min_digits=1, max_digits=2
    )
    entry = dataset[0]
    record = {
        "env_class": "reasoning_gym",
        "gold": json.dumps({"task": "chain_sum", "entry": entry}),
        "prompt_sha256": "fixture",
        "split": "dev",
    }
    assert audit_row(record).passed
    entry["metadata"]["source_dataset"] = "wrong-task"
    record["gold"] = json.dumps({"task": "chain_sum", "entry": entry})
    rejected = audit_row(record)
    assert not rejected.passed
    assert "task name" in rejected.failure


@pytest.mark.parametrize("gold,expected", [("90\\text{ square\nunits}", True), ("f(2) < f(1) < f(4)", False)])
def test_mechanical_audit_corrects_probe_whitespace_without_editing_reference(gold, expected):
    row = {"env_class": "aime", "gold": gold, "prompt_sha256": "fixture", "split": "heldout"}
    result = audit_row(row)
    assert result.passed is expected
    assert result.gold_original == row["gold"] == gold
    assert result.positive_probes[0] == "Answer: " + gold
    if expected:
        assert result.positive_probes == ["Answer: " + gold, "Answer: 90\\text{ square units}"]
        assert result.normalized_gold == "90"


def test_final_battery_quota_selection_is_locked_and_excludes_failed_candidates():

    manifest = [
        {"prompt_sha256": str(i).zfill(64), "bin": "even" if i % 2 == 0 else "odd", "split": "heldout"}
        for i in range(10)
    ]
    locks = {split: [row["prompt_sha256"] for row in manifest if row["split"] == split] for split in SPLIT_PRIORITY}
    selection = {
        "pool_version": "fixture",
        "manifest_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        "prompt_template_ids": {"qwen": "fixture"},
        "rows": locks,
        "heldout_lock_sha256": hashlib.sha256(canonical_json(locks).encode()).hexdigest(),
    }
    overlay = {
        "manifest_sha256": selection["manifest_sha256"],
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": "e" * 64,
        "statuses": {row["prompt_sha256"]: "accept" for row in manifest},
    }
    overlay["statuses"][manifest[0]["prompt_sha256"]] = "reject"
    args = (manifest, selection, overlay, {"even": 2, "odd": 2})
    first = select_components(*args, model="qwen", split="heldout", seed=17)
    second = select_components(*args, model="qwen", split="heldout", seed=17)
    assert first == second
    assert Counter(row["bin"] for row in first[0]) == {"even": 2, "odd": 2}
    assert manifest[0]["prompt_sha256"] not in first[1]["prompt_sha256"]
    assert first[1]["prompt_sha256"] == sorted(first[1]["prompt_sha256"])
