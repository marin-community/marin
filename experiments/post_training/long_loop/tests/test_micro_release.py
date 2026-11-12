# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
from marin.execution.lazy import ArtifactStep, StepContext
from rolloutengine.models import TaskRolloutSpec

from experiments.post_training.long_loop.context_evidence import context_evidence, validate_context_evidence
from experiments.post_training.long_loop.eval_runtime.repair_release import repaired_row
from experiments.post_training.long_loop.evaluation import TerminalPolicy, validate_eval_policy
from vendor.skyrl.marinskyrl.resource_locator import ModelLocatorError


def test_frozen_micro_row_converts_without_changing_prompt_or_grader():
    # Actual row from Micro210 revision 4564ed081c316bc3231ae5c3a53a46b39248c819.
    row = json.loads((Path(__file__).parent / "fixtures/micro028.json").read_text())
    converted = repaired_row(row, {})
    rollout = TaskRolloutSpec.model_validate_json(converted["task_rollout_spec"])
    assert rollout.task.model_dump(mode="json") == json.loads(converted["task_spec"])
    before = json.loads(row["task_spec"])
    after = json.loads(converted["task_spec"])
    for key, value in before.items():
        if key not in ("schema_version", "grader"):
            assert after[key] == value
    for key, value in before["grader"].items():
        assert after["grader"][key] == value
    assert after["schema_version"] == "0.30"


def test_eval_policy_locator_rejects_hf_shaped_local_path():
    source = "s3://marin-us-east-02a/test/committed-rl"
    step = ArtifactStep.adopt("test/rl", "2026.11.09", source)
    locator = TerminalPolicy(step, expected_step=3).resolve(StepContext.for_fingerprint(deps=(step,)))
    validate_eval_policy(locator, source)
    with pytest.raises(ModelLocatorError, match="local metadata path"):
        validate_eval_policy(replace(locator, local_path="models/long-loop-terminal"), source)


def test_pinned_vllm_proof_rejects_previous_inputs_and_invalidates_changed_config():
    config = json.loads((Path(__file__).parents[1] / "context-fixtures/config.json").read_text())
    evidence = context_evidence()
    with pytest.raises(ValueError, match="vLLM rejected"):
        validate_context_evidence(config, {"max_position_embeddings": 262144}, 65536, evidence)
    corrected = {"max_position_embeddings": 262144, "max_seq_len": 262144}
    validate_context_evidence(config, corrected, 65536, evidence)
    with pytest.raises(ValueError, match="vLLM rejected"):
        validate_context_evidence(config, {**corrected, "n_positions": 32768}, 65536, evidence)
    with pytest.raises(ValueError, match="lack a pinned"):
        validate_context_evidence({**config, "seq_length": 8192}, corrected, 65536, evidence)
