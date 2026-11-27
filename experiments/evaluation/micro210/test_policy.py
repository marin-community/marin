# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest
import yaml
from marin.execution.artifact import FingerprintMismatchError, read_record
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210.eval import (
    PREPARED_INPUTS,
    micro210_step,
    native_binding_code_sha256,
    run_micro210,
)
from experiments.evaluation.micro210.policy import (
    LEGACY_POLICY_REFERENCE,
    POLICY_REFERENCE,
    Micro210Model,
    Micro210Record,
    micro210_evaluation,
    read_micro210_policy,
)
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture(autouse=True)
def policy_reference(monkeypatch):
    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210.policy.POLICY_REFERENCE", reference)
    monkeypatch.setattr("experiments.evaluation.micro210.eval.NATIVE_INPUTS_FINGERPRINT", None)
    return reference


def test_micro210_binding_rejects_modified_protocol_bytes(tmp_path, monkeypatch, policy_reference):
    changed = tmp_path / "protocol.json"
    changed.write_text(StoragePath(policy_reference.uri).read_text().replace('"seed": 20261030', '"seed": 20261031'))
    monkeypatch.setattr(
        "experiments.evaluation.micro210.policy.POLICY_REFERENCE", replace(policy_reference, uri=str(changed))
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


@pytest.fixture
def verified_launch(tmp_path, monkeypatch, policy_reference):
    """A launch whose inputs already verified, with the handles its drift tests mutate."""
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
        "experiments.evaluation.micro210.policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(protocol_path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setattr("experiments.evaluation.micro210.eval.PREPARED_INPUTS", str(inputs))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    token_content = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    monkeypatch.setattr(
        "experiments.evaluation.micro210.eval.AutoTokenizer.from_pretrained",
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
    native_inputs_uri = yaml.safe_load(config.skyrl.launch_config_yaml)["inputs"]["validation_data"][0]["uri"]
    native_root = Path(native_inputs_uri)
    native_root.mkdir(parents=True)
    native_bytes = b"native-prepared-task-fixture"
    native_digest = hashlib.sha256(native_bytes).hexdigest()
    (native_root / "rollouts.parquet").write_bytes(native_bytes)
    (native_root / "binding.json").write_text(
        json.dumps(
            {
                "source_sha256": protocol["prepared_rollouts_sha256"],
                "rollouts_sha256": native_digest,
                "smoke_rows": [],
                "converter_sha256": native_binding_code_sha256(),
                "image_digests": json.loads(Path(__file__).with_name("images.json").read_text()),
            }
        )
    )
    submitted = []

    def resolve_key(spec):
        assert spec == (
            "env:TOGETHER_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/1",
        )
        return SimpleNamespace(value="test-together-key")

    monkeypatch.setattr("experiments.evaluation.micro210.eval.resolve_secret_spec", resolve_key)

    def submit_evaluation(skyrl, *, env_vars):
        assert env_vars == {"JUDGE_API_KEY": "test-together-key", "OPENAI_API_KEY": "test-together-key"}
        submitted.append(yaml.safe_load(skyrl.launch_config_yaml))
        return None

    def refuse_hosted_judge(*args, **kwargs):
        raise AssertionError("API judging must not prepare or provision a hosted judge")

    monkeypatch.setattr("marin.inference.iris.prepare_inference", refuse_hosted_judge)
    monkeypatch.setattr("experiments.evaluation.micro210.eval.run_skyrl", submit_evaluation)
    run_micro210(config)
    return SimpleNamespace(
        config=config,
        output=output,
        submitted=submitted,
        launch=submitted[0],
        protocol=protocol,
        protocol_sha256=hashlib.sha256(raw).hexdigest(),
        native_digest=native_digest,
        native_inputs_uri=native_inputs_uri,
        binding_path=native_root / "binding.json",
        binding=json.loads((native_root / "binding.json").read_text()),
        manifest_path=model_dir / ".marinskyrl-model-manifest.json",
        manifest=manifest,
        token_content=token_content,
    )


def test_verified_launch_records_inputs_and_judge_route(verified_launch):
    launch = verified_launch.launch
    assert launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] == "https://api.together.ai/v1"
    assert launch["skyrl"]["environment"]["task_sessions"]["agent_harnesses"]["pi"]["version"] == "0.87.0"
    assert all("test-together-key" not in str(value) for value in launch.values())
    record = draccus.decode(
        Micro210Record, json.loads((verified_launch.output / "canonical-evaluation.json").read_text())
    )
    assert record.policy_reference.sha256 == verified_launch.protocol_sha256
    assert record.native_inputs_sha256 == verified_launch.native_digest
    assert record.native_inputs_uri == verified_launch.native_inputs_uri
    sampling = verified_launch.protocol["evaluation"]["config"]["generator"]["eval_sampling_params"]
    assert launch["skyrl"]["generator"]["eval_sampling_params"] == sampling
    assert launch["skyrl"]["generator"]["error_handling"] == {
        "enable_error_classification": True,
        "default_error_treatment": "mask",
        "passthrough_exceptions": ["ContextLengthExceededError", "AgentTimeoutError"],
    }
    assert "terminal_bench" not in launch["skyrl"]
    assert launch["skyrl"]["entrypoint"] == "generate"
    assert launch["skyrl"]["generator"]["trajectory_retention"]["run_id"] == "test/micro210@2026.11.18"


def test_verified_launch_refuses_modified_launch_yaml(verified_launch):
    verified_launch.submitted.clear()
    altered = yaml.safe_load(verified_launch.config.skyrl.launch_config_yaml)
    altered["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.1
    config = replace(
        verified_launch.config, skyrl=replace(verified_launch.config.skyrl, launch_config_yaml=yaml.safe_dump(altered))
    )
    with pytest.raises(ValueError, match="launch_config_yaml"):
        run_micro210(config)
    assert verified_launch.submitted == []


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("converter_sha256", "0" * 64, "different converter"),
        ("source_sha256", "0" * 64, "different frozen source"),
        ("image_digests", {"python:3.10-slim": "sha256:" + "0" * 64}, "different image pins"),
    ],
)
def test_verified_launch_rejects_binding_drift(verified_launch, field, value, message):
    verified_launch.submitted.clear()
    verified_launch.binding_path.write_text(json.dumps({**verified_launch.binding, field: value}))
    with pytest.raises(ValueError, match=message):
        run_micro210(verified_launch.config)
    assert verified_launch.submitted == []


def test_verified_launch_rejects_changed_model_manifest(verified_launch):
    verified_launch.submitted.clear()
    verified_launch.manifest_path.write_bytes(b'{"files": ["changed"]}')
    with pytest.raises(ValueError, match="export manifest differs"):
        run_micro210(verified_launch.config)
    assert verified_launch.submitted == []


def test_verified_launch_rejects_incompatible_stop_tokens(verified_launch):
    verified_launch.submitted.clear()
    verified_launch.token_content[128009] = "ordinary-token"
    with pytest.raises(ValueError, match="incompatible stop token ID: 128009"):
        run_micro210(verified_launch.config)
    assert verified_launch.submitted == []


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
    monkeypatch.setattr("experiments.evaluation.micro210.eval.PREPARED_INPUTS", str(new_inputs))
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/new-policy",
        version="2026.11.20",
    )
    native = next(dep for dep in step.deps if dep.name == "evaluation/micro210/native")
    prepared = native.deps[0]
    run(prepared)
    old_record = read_record(str(tmp_path / "evaluation/micro210/prepared" / previous_version))
    new_record = read_record(str(tmp_path / "evaluation/micro210/prepared" / "2026.11.23"))
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
    native = next(dep for dep in step.deps if dep.name == "evaluation/micro210/native")
    prepared = native.deps[0]
    assert prepared.adopt_source == PREPARED_INPUTS
    assert re.fullmatch(r"hf://datasets/open-athena/micro210-prepared-tasks@[0-9a-f]{40}", PREPARED_INPUTS)


@pytest.mark.parametrize("change", ["image_lock", "converter"])
def test_native_artifact_pin_rejects_conversion_drift(tmp_path, monkeypatch, change):
    source = Path(__file__).with_name("eval.py")
    for name in ("images.json", "inputs.py", "policy.py"):
        (tmp_path / name).write_bytes(source.with_name(name).read_bytes())
    monkeypatch.setattr("experiments.evaluation.micro210.eval.__file__", str(tmp_path / "eval.py"))
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/pinned-binding", version="2026.11.24")
    native = next(dep for dep in step.deps if dep.name == "evaluation/micro210/native")
    monkeypatch.setattr("experiments.evaluation.micro210.eval.NATIVE_INPUTS_FINGERPRINT", native.fingerprint())
    lower(micro210_step(model, name="fixture/pinned-binding", version="2026.11.24"))
    lock_path = tmp_path / "images.json"
    if change == "image_lock":
        lock = json.loads(lock_path.read_text())
        lock[next(iter(lock))] = "sha256:" + "f" * 64
        lock_path.write_text(json.dumps(lock))
    else:
        converter = tmp_path / "inputs.py"
        converter.write_text(converter.read_text() + "\n# changed converter fixture\n")
    with pytest.raises(FingerprintMismatchError, match="expected_fingerprint pins"):
        lower(micro210_step(model, name="fixture/pinned-binding", version="2026.11.24"))
