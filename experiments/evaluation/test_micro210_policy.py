# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import asdict

import pytest

from experiments.evaluation.micro210_policy import SPEC_PATH, Micro210Model, micro210_evaluation


@pytest.mark.parametrize(
    "override",
    [
        {"eval_override_storage_mb": 131072},
        {"sampling": {"temperature": 0}},
        {"timeout": 300},
        {"task_subset": ["swebench-verified"]},
        {"judge": "other-model"},
    ],
)
def test_micro210_binding_refuses_protocol_overrides(override):
    model = Micro210Model("s3://test/model", "model@v1:abcd", "test/tokenizer", "a" * 40)
    with pytest.raises(ValueError, match=next(iter(override))):
        micro210_evaluation(model, run_id="test", output_uri="s3://test/eval", **override)


def test_micro210_model_change_preserves_the_protocol_and_records_its_identity():
    first = micro210_evaluation(
        Micro210Model("s3://test/sft", "sft@v1:abcd", "test/tokenizer", "a" * 40),
        run_id="sft",
        output_uri="s3://test/sft-eval",
    )
    second = micro210_evaluation(
        Micro210Model("s3://test/rl", "rl@v1:efgh", "test/tokenizer", "a" * 40),
        run_id="rl",
        output_uri="s3://test/rl-eval",
    )
    assert first.policy() == second.policy()
    assert first.record().canonical_spec_sha256 == hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    assert asdict(first.record())["model"]["identity"] == first.model.identity
    assert first.record().model.identity != second.record().model.identity
    policy = first.policy()
    policy["evaluation"]["seed"] = -1
    assert first.policy()["evaluation"]["seed"] == 20261030


def test_micro210_binding_rejects_modified_protocol_bytes(tmp_path, monkeypatch):
    changed = tmp_path / "protocol.json"
    changed.write_text(SPEC_PATH.read_text().replace('"seed": 20261030', '"seed": 0'))
    monkeypatch.setattr("experiments.evaluation.micro210_policy.SPEC_PATH", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        micro210_evaluation(
            Micro210Model("s3://test/model", "model@v1:abcd", "test/tokenizer", "a" * 40),
            run_id="test",
            output_uri="s3://test/output",
        )
