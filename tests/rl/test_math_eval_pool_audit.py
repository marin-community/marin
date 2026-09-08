# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import reasoning_gym

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
