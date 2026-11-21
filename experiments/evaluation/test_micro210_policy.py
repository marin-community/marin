# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210 import PREPARED_INPUTS, PREPARED_INPUTS_VERSION, micro210_step, run_micro210
from experiments.evaluation.micro210_policy import (
    LEGACY_POLICY_REFERENCE,
    POLICY_REFERENCE,
    Micro210Model,
    micro210_evaluation,
    read_micro210_policy,
)
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture(autouse=True)
def policy_reference(monkeypatch):
    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", reference)
    return reference


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


def test_default_policy_references_pin_immutable_hugging_face_revisions():
    for reference in (LEGACY_POLICY_REFERENCE, POLICY_REFERENCE):
        assert re.fullmatch(r"hf://datasets/open-athena/micro210-policy@[0-9a-f]{40}/policy.json", reference.uri)


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
    token_content = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    monkeypatch.setattr(
        "experiments.evaluation.micro210.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(convert_ids_to_tokens=token_content.__getitem__),
    )
    binding = Micro210Model(str(model_dir), "sha256:" + hashlib.sha256(manifest).hexdigest(), "test/tokenizer", "a" * 40)
    step = micro210_step(binding, name="test/micro210", version="2026.11.18")
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

    def resolve_key(spec):
        assert spec == ("gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/1",)
        return SimpleNamespace(value="test-together-key")

    monkeypatch.setattr("experiments.evaluation.micro210.resolve_secret_spec", resolve_key)

    def submit_evaluation(skyrl, *, env_vars):
        assert env_vars == {"JUDGE_API_KEY": "test-together-key", "OPENAI_API_KEY": "test-together-key"}
        submitted.append(yaml.safe_load(skyrl.launch_config_yaml))
        return None

    def refuse_hosted_judge(*args, **kwargs):
        raise AssertionError("API judging must not prepare or provision a hosted judge")

    monkeypatch.setattr("marin.inference.iris.prepare_inference", refuse_hosted_judge)
    monkeypatch.setattr("experiments.evaluation.micro210.run_skyrl", submit_evaluation)
    run_micro210(config)
    launch = submitted[0]
    assert launch["skyrl"]["entrypoint"] == "generate"
    assert launch["skyrl"]["trainer"]["placement"]["colocate_all"] is False
    assert launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] == "https://api.together.ai/v1"
    assert launch["skyrl"]["terminal_bench"]["installed_agent"]["judge_model"] == "zai-org/GLM-5.3-Flash"
    assert all("test-together-key" not in str(value) for value in launch.values())
    record = json.loads((output / "canonical-evaluation.json").read_text())
    assert record["canonical_spec_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        submitted[0]["skyrl"]["generator"]["eval_sampling_params"]
        == protocol["evaluation"]["config"]["generator"]["eval_sampling_params"]
    )
    assert "eval_override_storage_mb" not in submitted[0]["skyrl"]["terminal_bench"]["harbor"]
    assert submitted[0]["skyrl"]["generator"]["trajectory_retention"]["run_id"] == "test/micro210@2026.11.18"
    altered = yaml.safe_load(config.skyrl.launch_config_yaml)
    altered["skyrl"]["terminal_bench"]["harbor"]["eval_override_storage_mb"] = 131072
    submitted.clear()
    with pytest.raises(ValueError, match="launch_config_yaml"):
        run_micro210(replace(config, skyrl=replace(config.skyrl, launch_config_yaml=yaml.safe_dump(altered))))
    assert submitted == []
    manifest_path = model_dir / ".marinskyrl-model-manifest.json"
    manifest_path.write_bytes(b'{"files": ["changed"]}')
    with pytest.raises(ValueError, match="export manifest differs"):
        run_micro210(config)
    assert submitted == []
    manifest_path.write_bytes(manifest)
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
    step = micro210_step(model, name="fixture/final-eval", version="2026.11.18.74")
    plan = lower(step)
    assert "fixture/final-eval@2026.11.18.74" in plan.fingerprint_payload


@pytest.mark.parametrize("change", ["digest", "byte_size"])
def test_policy_reference_verifies_hash_and_size(policy_reference, change):
    bad = replace(policy_reference, **({"sha256": "0" * 64} if change == "digest" else {"byte_size": 1}))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_micro210_policy(bad)
    assert read_micro210_policy(policy_reference)["evaluation"]["judge"]["model"] == MICRO210_JUDGE.model


@pytest.mark.parametrize(
    "previous_version,previous_digest", [("2026.11.18", "old-task-digest"), ("2026.11.20", "fixture-tasks")]
)
def test_new_prepared_inputs_do_not_reuse_old_policy_provenance(
    tmp_path, monkeypatch, previous_version, previous_digest
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    old_inputs = tmp_path / "old-inputs"
    new_inputs = tmp_path / "new-inputs"
    old_inputs.mkdir()
    new_inputs.mkdir()
    run(
        ArtifactStep.adopt(
            "evaluation/micro210/prepared", previous_version, str(old_inputs), config={"sha256": previous_digest}
        )
    )
    monkeypatch.setattr("experiments.evaluation.micro210.PREPARED_INPUTS", str(new_inputs))
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/new-policy",
        version="2026.11.20",
    )
    prepared = next(dep for dep in step.deps if dep.name == "evaluation/micro210/prepared")
    run(prepared)
    old_record = read_record(str(tmp_path / "evaluation/micro210/prepared" / previous_version))
    new_record = read_record(str(tmp_path / "evaluation/micro210/prepared" / PREPARED_INPUTS_VERSION))
    assert old_record.source == str(old_inputs)
    assert old_record.config == {"sha256": previous_digest}
    assert new_record.source == str(new_inputs)
    assert new_record.config == {"sha256": "fixture-tasks"}


def test_prepared_inputs_adopt_the_pinned_hugging_face_source():
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/prepared-source",
        version="2026.11.20",
    )
    prepared = next(dep for dep in step.deps if dep.name == "evaluation/micro210/prepared")
    assert prepared.adopt_source == PREPARED_INPUTS
    assert re.fullmatch(r"hf://datasets/open-athena/micro210-prepared-tasks@[0-9a-f]{40}", PREPARED_INPUTS)
