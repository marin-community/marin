# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.lazy import ArtifactStep, StepContext
from rolloutengine.models import TaskRolloutSpec

from experiments.post_training.long_loop import evaluation
from experiments.post_training.long_loop.context_evidence import context_evidence, validate_context_evidence
from experiments.post_training.long_loop.eval_runtime import repair_release
from experiments.post_training.long_loop.eval_runtime.repair_release import repaired_row
from experiments.post_training.long_loop.evaluation import TerminalPolicy, prepare_micro, validate_eval_policy
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


def test_prepare_micro_preserves_attempts_without_running_setup_trials(tmp_path, monkeypatch):
    row = json.loads((Path(__file__).parent / "fixtures/micro028.json").read_text())
    unique = [
        {
            **row,
            "agent": "single-turn",
            "task_id": f"task-{benchmark}-{task}",
            "benchmark": "tau3-pi" if benchmark == 20 else f"benchmark-{benchmark}",
            "repetitions": 3 if benchmark < 4 else 2,
        }
        for benchmark in range(21)
        for task in range(10)
    ]
    expanded = [{**r, "repetition_index": rep} for r in unique for rep in range(r["repetitions"])]
    pq.write_table(pa.Table.from_pylist(unique), tmp_path / "validation.parquet")
    pq.write_table(pa.Table.from_pylist(expanded), tmp_path / "validation-rollouts.parquet")
    (tmp_path / "manifest.json").write_text("[]")

    def download(repo, filename, *, repo_type, revision):
        return str(tmp_path / Path(filename).name)

    def reject_trial_process(*args, **kwargs):
        pytest.fail("Evaluation input preparation must not launch a setup trial or subprocess")

    monkeypatch.setattr(evaluation, "hf_hub_download", download)
    monkeypatch.setattr(repair_release, "hf_hub_download", download)
    monkeypatch.setattr(subprocess, "run", reject_trial_process)
    output = tmp_path / "prepared"
    output.mkdir()
    prepare_micro(
        output=str(output), dataset="pinned/micro", revision="a" * 40, wheels_revision="b" * 40, tasks_per_benchmark=10
    )

    table = pq.read_table(output / "rollouts.parquet")
    selection = json.loads((output / "selection.json").read_text())
    manifest = json.loads((output / "environment.json").read_text())
    assert table.num_rows == selection["attempts"] == 440
    assert selection["unique_tasks"] == 200
    assert manifest["setup_preflight_executed"] is False
    assert manifest["data_uri"] == str(output)
    assert len(manifest["tasks"]) == 440
    for binding, prepared in zip(manifest["tasks"], table.to_pylist(), strict=True):
        assert binding["task_id"] == prepared["task_id"]
        assert binding["repetition_index"] == prepared["repetition_index"]
        assert binding["task_rollout_spec_sha256"] == hashlib.sha256(prepared["task_rollout_spec"].encode()).hexdigest()


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
