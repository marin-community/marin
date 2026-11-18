# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.evaluation.micro210 import micro210_step, run_micro210
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
    model = Micro210Model("s3://test/model", "sha256:" + "a" * 64, "test/tokenizer", "a" * 40)
    with pytest.raises(ValueError, match=next(iter(override))):
        micro210_evaluation(model, run_id="test", output_uri="s3://test/eval", **override)


def test_micro210_model_change_preserves_the_protocol_and_records_its_identity():
    first = micro210_evaluation(
        Micro210Model("s3://test/sft", "sha256:" + "b" * 64, "test/tokenizer", "a" * 40),
        run_id="sft",
        output_uri="s3://test/sft-eval",
    )
    second = micro210_evaluation(
        Micro210Model("s3://test/rl", "sha256:" + "c" * 64, "test/tokenizer", "a" * 40),
        run_id="rl",
        output_uri="s3://test/rl-eval",
    )
    assert first.policy() == second.policy()
    assert first.record().canonical_spec_sha256 == hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def test_micro210_binding_rejects_modified_protocol_bytes(tmp_path, monkeypatch):
    changed = tmp_path / "protocol.json"
    changed.write_text(SPEC_PATH.read_text().replace('"seed": 20261030', '"seed": 0'))
    monkeypatch.setattr("experiments.evaluation.micro210_policy.SPEC_PATH", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        micro210_evaluation(
            Micro210Model("s3://test/model", "sha256:" + "a" * 64, "test/tokenizer", "a" * 40),
            run_id="test",
            output_uri="s3://test/output",
        )


@pytest.mark.parametrize("identity,revision", [("main", "a" * 40), ("sha256:" + "a" * 64, "main")])
def test_micro210_model_rejects_mutable_references(identity, revision):
    with pytest.raises(ValueError, match=r"identity|tokenizer_revision"):
        Micro210Model("s3://test/latest", identity, "test/tokenizer", revision)


def test_micro210_launch_records_protocol_and_rejects_changed_inputs_before_gpu_allocation(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = b'{"files": []}'
    (model_dir / ".marinskyrl-model-manifest.json").write_bytes(manifest)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    task_bytes = b"pinned-prepared-task-fixture"
    (inputs / "rollouts.parquet").write_bytes(task_bytes)
    protocol = json.loads(SPEC_PATH.read_text())
    protocol["prepared_rollouts_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    raw = json.dumps(protocol).encode()
    protocol_path.write_bytes(raw)
    monkeypatch.setattr("experiments.evaluation.micro210_policy.SPEC_PATH", protocol_path)
    monkeypatch.setattr("experiments.evaluation.micro210_policy.SPEC_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr("experiments.evaluation.micro210.PREPARED_INPUTS", str(inputs))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    monkeypatch.setenv("TOGETHER_API_KEY", "test-counterpart-key")
    binding = Micro210Model(str(model_dir), "sha256:" + hashlib.sha256(manifest).hexdigest(), "test/tokenizer", "a" * 40)
    step = micro210_step(binding, name="test/micro210", version="2026.11.18")
    with pytest.raises(ValueError, match="eval_override_storage_mb"):
        micro210_step(binding, name="test/micro210", version="2026.11.18", eval_override_storage_mb=131072)
    output = tmp_path / "result"
    output.mkdir()
    config = step.build_config(
        StepContext.for_run(
            str(output),
            str(tmp_path),
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    submitted = []

    @contextmanager
    def serve_judge(_config):
        submitted.append("judge")
        yield SimpleNamespace(
            model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://judge.test/v1", api_key="test"))
        )

    def submit_evaluation(skyrl):
        submitted.append(yaml.safe_load(skyrl.launch_config_yaml))
        return None

    monkeypatch.setattr("experiments.evaluation.micro210.remote_inference", serve_judge)
    monkeypatch.setattr("experiments.evaluation.micro210.run_skyrl", submit_evaluation)
    run_micro210(config)
    record = json.loads((output / "canonical-evaluation.json").read_text())
    assert record["canonical_spec_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        submitted[1]["skyrl"]["generator"]["eval_sampling_params"]
        == protocol["evaluation"]["config"]["generator"]["eval_sampling_params"]
    )
    assert "eval_override_storage_mb" not in submitted[1]["skyrl"]["terminal_bench"]["harbor"]
    (inputs / "rollouts.parquet").write_bytes(b"changed-tasks")
    submitted.clear()
    with pytest.raises(ValueError, match="task bytes changed"):
        run_micro210(config)
    assert submitted == []
