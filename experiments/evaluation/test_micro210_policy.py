# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.lazy import StepContext, lower
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210 import micro210_step, run_micro210
from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture(autouse=True)
def policy_reference(monkeypatch):
    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", reference)
    return reference


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


def test_micro210_record_contains_content_digest_of_protocol(policy_reference):
    evaluation = micro210_evaluation(
        Micro210Model("s3://test/sft", "sha256:" + "b" * 64, "test/tokenizer", "a" * 40),
        run_id="sft",
        output_uri="s3://test/sft-eval",
    )
    assert (
        evaluation.record().canonical_spec_sha256
        == hashlib.sha256(StoragePath(policy_reference.uri).read_bytes()).hexdigest()
    )


def test_micro210_binding_rejects_modified_protocol_bytes(tmp_path, monkeypatch, policy_reference):
    changed = tmp_path / "protocol.json"
    changed.write_text(StoragePath(policy_reference.uri).read_text().replace('"seed": 20261030', '"seed": 20261031'))
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE", replace(policy_reference, uri=str(changed))
    )
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


def test_micro210_launch_records_protocol_and_rejects_changed_inputs_before_gpu_allocation(
    tmp_path, monkeypatch, policy_reference
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = b'{"files": []}'
    (model_dir / ".marinskyrl-model-manifest.json").write_bytes(manifest)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    task_bytes = b"pinned-prepared-task-fixture"
    (inputs / "rollouts.parquet").write_bytes(task_bytes)
    protocol = json.loads(StoragePath(policy_reference.uri).read_text())
    protocol["prepared_rollouts_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    raw = json.dumps(protocol).encode()
    protocol_path.write_bytes(raw)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(protocol_path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setattr("experiments.evaluation.micro210.PREPARED_INPUTS", str(inputs))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    monkeypatch.setenv("TOGETHER_API_KEY", "unused-ambient-key")
    counterpart_key = tmp_path / "counterpart-key"
    counterpart_key.write_text("test-counterpart-key")
    token_content = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    monkeypatch.setattr(
        "experiments.evaluation.micro210.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(convert_ids_to_tokens=token_content.__getitem__),
    )
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
    config = replace(config, counterpart_api_key=(f"file:{counterpart_key}",))
    submitted = []

    @contextmanager
    def serve_judge(_config):
        submitted.append("judge")
        yield SimpleNamespace(
            model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://judge.test/v1", api_key="test"))
        )

    def submit_evaluation(skyrl, *, env_vars):
        assert env_vars == {"JUDGE_API_KEY": "test", "OPENAI_API_KEY": "test-counterpart-key"}
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
    assert submitted[1]["skyrl"]["generator"]["trajectory_retention"]["run_id"] == "test/micro210@2026.11.18"
    altered = yaml.safe_load(config.skyrl.launch_config_yaml)
    altered["skyrl"]["terminal_bench"]["harbor"]["eval_override_storage_mb"] = 131072
    submitted.clear()
    with pytest.raises(ValueError, match="launch_config_yaml"):
        run_micro210(replace(config, skyrl=replace(config.skyrl, launch_config_yaml=yaml.safe_dump(altered))))
    assert submitted == []
    token_content[128009] = "ordinary-token"
    with pytest.raises(ValueError, match="incompatible stop token ID: 128009"):
        run_micro210(config)
    assert submitted == []
    (inputs / "rollouts.parquet").write_bytes(b"changed-tasks")
    submitted.clear()
    with pytest.raises(ValueError, match="task bytes changed"):
        run_micro210(config)
    assert submitted == []


def test_canonical_evaluation_graph_lowers_with_its_final_run_binding():
    model = Micro210Model("s3://fixture/export", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/final-eval", version="2026.11.18.72")
    plan = lower(step)
    assert "fixture/final-eval@2026.11.18.72" in plan.fingerprint_payload
