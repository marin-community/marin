# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from iris.client.client import IrisClient
from iris.resources.state import JobState
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.rl.cli import _coordinator_request
from marin.rl.skyrl import skyrl_execution
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210 import (
    PREPARED_INPUTS,
    PREPARED_INPUTS_VERSION,
    ServingProfileName,
    micro210_step,
    run_micro210,
    validate_judge_availability,
)
from experiments.evaluation.micro210_policy import (
    LEGACY_POLICY_REFERENCE,
    POLICY_REFERENCE,
    Micro210Model,
    Micro210Route,
    micro210_evaluation,
    read_micro210_policy,
    runtime_policy_config,
)
from experiments.post_training.long_loop.launch_validation import validate_launch
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture(autouse=True)
def policy_reference(monkeypatch):
    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", reference)
    return reference


def test_default_policy_references_pin_immutable_hugging_face_revisions():
    for reference in (LEGACY_POLICY_REFERENCE, POLICY_REFERENCE):
        assert re.fullmatch(r"hf://datasets/open-athena/micro210-policy@[0-9a-f]{40}/policy.json", reference.uri)


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


def test_runtime_harbor_translation_preserves_the_frozen_policy(policy_reference):
    raw = StoragePath(policy_reference.uri).read_bytes()
    policy = json.loads(raw)
    translated = runtime_policy_config(policy)
    restored = json.loads(json.dumps(translated.config))
    harbor = restored["terminal_bench"]["harbor"]
    harbor["daytona_provisioning"] = harbor.pop("daytona_image_build")
    harbor["eval_daytona_provisioning"] = harbor.pop("eval_daytona_image_build")
    assert restored == policy["evaluation"]["config"]
    assert StoragePath(policy_reference.uri).read_bytes() == raw
    policy["evaluation"]["config"]["terminal_bench"]["harbor"]["daytona_image_build"] = "snapshot"
    with pytest.raises(ValueError, match="ambiguous Harbor fields"):
        runtime_policy_config(policy)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("unknown", ["daytona_provisioning_extra", "unknown_harbor_option"])
def test_runtime_harbor_translation_never_drops_unknown_fields(tmp_path, monkeypatch, policy_reference, unknown):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    policy = json.loads(StoragePath(policy_reference.uri).read_text())
    policy["evaluation"]["config"]["terminal_bench"]["harbor"][unknown] = True
    raw = json.dumps(policy).encode()
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/translation",
        version="2026.11.19",
    )
    context = StepContext.for_run(
        str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
    )
    run = step.build_config(replace(context, is_fingerprint=True))
    with pytest.raises(ValueError, match="Unknown Harbor task settings"):
        validate_launch(run.skyrl.launch_config_yaml, tmp_path / "preflight", model_config=b"{}")


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
            route=Micro210Route.from_policy_execution(
                json.loads(StoragePath(policy_reference.uri).read_text())["evaluation"]["execution"]
            ),
        )


@pytest.mark.parametrize(
    "profile,target_cluster",
    [
        (ServingProfileName.SNOWBALL_H100_EAST02A, "cw-us-east-02a"),
        (ServingProfileName.MOCHI_GB200, "cw-us-east-08a"),
    ],
)
def test_serving_profile_delivers_hf_secret_to_the_actual_coordinator(
    tmp_path, monkeypatch, policy_reference, profile, target_cluster
):
    policy = json.loads(StoragePath(policy_reference.uri).read_bytes())
    judge_key = tmp_path / "judge-key"
    judge_key.write_text("fixture-judge-key")
    policy["evaluation"]["judge"]["credential"] = f"file:{judge_key}"
    raw = json.dumps(policy).encode()
    protocol = tmp_path / "policy.json"
    protocol.write_bytes(raw)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(protocol), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setenv("HF_TOKEN", "fixture-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "fixture-wandb-key")
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/regional-eval",
        version="2026.11.23",
        serving_profile=profile,
    )
    execution = skyrl_execution(step)
    required = {name: "fixture-value" for name in execution.required_env_vars}
    cluster, request = _coordinator_request([step], "experiments.evaluation.micro210", (), tmp_path, required)
    config = step.build_config(
        StepContext.for_run(str(tmp_path / "eval"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    )
    assert cluster == "lib/iris/config/marin.yaml"
    assert request.resources.target_cluster == target_cluster
    assert yaml.safe_load(config.skyrl.launch_config_yaml)["iris"]["target_cluster"] == target_cluster
    assert request.environment.secrets["HF_TOKEN"] == "fixture-hf-token"
    assert "HF_TOKEN" not in request.environment.env_vars
    assert request.environment.secrets["JUDGE_API_KEY"] == "fixture-judge-key"


@pytest.mark.parametrize("identity,revision", [("main", "a" * 40), ("sha256:" + "a" * 64, "main")])
def test_micro210_model_rejects_mutable_references(identity, revision):
    with pytest.raises(ValueError, match=r"identity|tokenizer_revision"):
        Micro210Model("s3://test/latest", identity, "test/tokenizer", revision)


<<<<<<< HEAD
@pytest.mark.parametrize("profile", list(ServingProfileName))
@pytest.mark.parametrize("manifest_override", [None, ".marinskyrl-model-manifest.json"])
=======
def test_route_rebinding_keeps_the_policy_hash_and_protocol_frozen(tmp_path, monkeypatch, policy_reference):
    """A per-model route changes where the run executes and nothing about the protocol."""
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/route", version="2026.11.23.1")
    policy_route = Micro210Route.from_policy_execution(read_micro210_policy(POLICY_REFERENCE)["evaluation"]["execution"])
    rerouted = micro210_step(
        model,
        name="fixture/route",
        version="2026.11.23.1",
        route=replace(
            policy_route,
            cluster_config="lib/iris/config/cw-us-east-08a.yaml",
            target_cluster="cw-us-east-08a",
        ),
    )
    # The route lives in the binding; both steps read the same frozen policy bytes.

    def build(step):
        return step.build_config(
            StepContext.for_run(
                str(tmp_path / "output"),
                str(tmp_path),
                runtime_args=step.runtime_args,
                deps=step.deps,
            )
        )

    canonical, rerouted_run = build(step), build(rerouted)
    base, moved = (yaml.safe_load(run.skyrl.launch_config_yaml) for run in (canonical, rerouted_run))
    # The random attempt id is not protocol; every other non-route field must match.
    assert {key: base[key] for key in base if key not in ("iris", "run")} == {
        key: moved[key] for key in moved if key not in ("iris", "run")
    }
    assert {key: value for key, value in base["run"].items() if key != "attempt_id"} == {
        key: value for key, value in moved["run"].items() if key != "attempt_id"
    }
    assert moved["iris"]["target_cluster"] == "cw-us-east-08a"
    assert moved["iris"]["cluster_config"].endswith("cw-us-east-08a.yaml")
    # The record carries the binding's route and the same frozen protocol hash.
    assert canonical.evaluation.route.target_cluster == "test"
    assert rerouted_run.evaluation.route.target_cluster == "cw-us-east-08a"
    assert rerouted_run.evaluation.record().canonical_spec_sha256 == canonical.evaluation.record().canonical_spec_sha256


def test_route_rejects_untyped_values():
    with pytest.raises(ValueError, match="yaml config path"):
        Micro210Route("lib/iris/config/cw-rno2a", "lib/iris/config/cw-rno2a.yaml", "cw-rno2a", 4)
    with pytest.raises(ValueError, match="target_cluster"):
        Micro210Route("lib/iris/config/marin.yaml", "lib/iris/config/cw-rno2a.yaml", "", 4)
    with pytest.raises(ValueError, match="container_profile"):
        Micro210Route("lib/iris/config/marin.yaml", "lib/iris/config/cw-rno2a.yaml", "cw-rno2a", -1)


def test_default_policy_references_pin_immutable_hugging_face_revisions():
    for reference in (LEGACY_POLICY_REFERENCE, POLICY_REFERENCE):
        assert re.fullmatch(r"hf://datasets/open-athena/micro210-policy@[0-9a-f]{40}/policy.json", reference.uri)


>>>>>>> 5aaf8ba75b ([eval] Bind the Micro210 execution route to the model, not the protocol)
def test_micro210_launch_records_protocol_and_rejects_changed_inputs_before_gpu_allocation(
    tmp_path, monkeypatch, policy_reference, profile, manifest_override
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = b'{"files": []}'
    manifest_filename = manifest_override or (
        "export-manifest.json" if profile is ServingProfileName.MOCHI_GB200 else ".marinskyrl-model-manifest.json"
    )
    (model_dir / manifest_filename).write_bytes(manifest)
    (model_dir / "config.json").write_text("{}")
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
    secret_directory = tmp_path / "secrets"
    secret_directory.mkdir()
    (secret_directory / "JUDGE_API_KEY").write_text("test-together-key")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(secret_directory))
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    token_content = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    monkeypatch.setattr(
        "experiments.evaluation.micro210.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(convert_ids_to_tokens=token_content.__getitem__),
    )
    binding = Micro210Model(str(model_dir), "sha256:" + hashlib.sha256(manifest).hexdigest(), "test/tokenizer", "a" * 40)
    step = micro210_step(
        binding,
        name="test/micro210",
        version="2026.11.18",
        serving_profile=profile,
        model_manifest_filename=manifest_override,
    )
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
    # Iris is the external admission boundary; this test does not allocate workers.
    iris_client = MagicMock(spec=IrisClient)
    iris_client.submit.return_value.job_id = "regional-preflight"
    iris_client.submit.return_value.state_only.return_value = JobState.SUCCEEDED
    monkeypatch.setattr(
        "experiments.post_training.long_loop.launch_validation.open_iris_client",
        lambda **kwargs: nullcontext(iris_client),
    )
    submitted = []

    def submit_evaluation(skyrl, *, env_vars):
        assert env_vars == {"JUDGE_API_KEY": "test-together-key", "OPENAI_API_KEY": "test-together-key"}
        submitted.append(yaml.safe_load(skyrl.launch_config_yaml))
        return None

    monkeypatch.setattr("experiments.evaluation.micro210.run_skyrl", submit_evaluation)
    monkeypatch.setattr(
        "experiments.evaluation.micro210.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=[{"id": protocol["evaluation"]["judge"]["model"]}], request=httpx.Request("GET", url)
        ),
    )
    run_micro210(config)
    record = json.loads((output / "canonical-evaluation.json").read_text())
    assert record["canonical_spec_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["runtime_field_renames"] == {
        "terminal_bench.harbor.daytona_provisioning": "terminal_bench.harbor.daytona_image_build",
        "terminal_bench.harbor.eval_daytona_provisioning": "terminal_bench.harbor.eval_daytona_image_build",
    }
    assert submitted[0]["skyrl"]["terminal_bench"]["harbor"]["eval_daytona_image_build"] == "declarative"
    assert (
        submitted[0]["skyrl"]["generator"]["eval_sampling_params"]
        == protocol["evaluation"]["config"]["generator"]["eval_sampling_params"]
    )
    assert "eval_override_storage_mb" not in submitted[0]["skyrl"]["terminal_bench"]["harbor"]
    assert submitted[0]["skyrl"]["generator"]["trajectory_retention"]["run_id"] == "test/micro210@2026.11.18"
    environment = iris_client.submit.call_args.kwargs["environment"]
    assert environment.secret_behavior is SecretBehavior.ISOLATE
    assert environment.secrets["JUDGE_API_KEY"] == "test-together-key"
    assert not any(name.startswith("GOOGLE_") for name in environment.env_vars | environment.secrets)
    submitted.clear()
    iris_client.submit.return_value.wait.side_effect = RuntimeError("regional input unreadable")
    with pytest.raises(RuntimeError, match="regional input unreadable"):
        run_micro210(config)
    assert submitted == []
    iris_client.submit.return_value.wait.side_effect = None
    monkeypatch.setattr(
        "experiments.evaluation.micro210.httpx.get",
        lambda url, **kwargs: httpx.Response(503, request=httpx.Request("GET", url)),
    )
    with pytest.raises(ValueError, match="unreachable before rollout allocation"):
        run_micro210(config)
    assert submitted == []
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


@pytest.mark.parametrize("profile", list(ServingProfileName))
def test_canonical_evaluation_graph_lowers_with_its_final_run_binding(profile):
    model = Micro210Model("s3://fixture/export", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/final-eval", version="2026.11.18.72", serving_profile=profile)
    plan = lower(step)
    assert "fixture/final-eval@2026.11.18.72" in plan.fingerprint_payload


def test_frozen_colocated_policy_runs_as_generation_without_changing_protocol(tmp_path, monkeypatch, policy_reference):
    policy = json.loads(StoragePath(policy_reference.uri).read_text())
    policy["evaluation"]["config"]["trainer"]["placement"] = {"colocate_all": True}
    policy["evaluation"]["config"]["entrypoint"] = "generate"
    payload = json.dumps(policy).encode()
    path = tmp_path / "policy.json"
    path.write_bytes(payload)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(path), sha256=hashlib.sha256(payload).hexdigest(), byte_size=len(payload)),
    )
    model = Micro210Model("s3://fixture/export", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/generation", version="2026.11.22")
    context = StepContext.for_run(
        str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
    )
    run_config = step.build_config(replace(context, is_fingerprint=True))
    launch = yaml.safe_load(run_config.skyrl.launch_config_yaml)
    assert launch["skyrl"]["entrypoint"] == "generate"
    assert launch["skyrl"]["trainer"]["placement"]["colocate_all"] is False
    assert launch["iris"]["allocation"]["num_nodes"] == 1
    assert launch["iris"]["allocation"]["gpus_per_node"] == 1
    assert (
        launch["skyrl"]["generator"]["eval_sampling_params"]
        == policy["evaluation"]["config"]["generator"]["eval_sampling_params"]
    )
    assert path.read_bytes() == payload


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


@pytest.mark.parametrize("failure", [None, "unreachable", "model"])
def test_api_judge_availability_rejects_unreachable_or_missing_model(monkeypatch, failure):
    def models(url, **kwargs):
        assert url == MICRO210_JUDGE.base_url + "/models"
        assert kwargs["headers"] == {"Authorization": "Bearer fixture-key"}
        return httpx.Response(
            503 if failure == "unreachable" else 200,
            json=[{"id": "wrong-model" if failure == "model" else MICRO210_JUDGE.model}],
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("experiments.evaluation.micro210.httpx.get", models)
    if failure:
        with pytest.raises(ValueError, match="unreachable" if failure == "unreachable" else "does not serve"):
            validate_judge_availability(MICRO210_JUDGE, "fixture-key")
        return
    validate_judge_availability(MICRO210_JUDGE, "fixture-key")


@pytest.mark.timeout(120)
def test_mochi_generation_native_preflight_accepts_four_gpu_hosts(tmp_path, monkeypatch):
    path = Path(__file__).with_name("fixtures") / "micro210-together-runtime.json"
    raw = path.read_bytes()
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_fixture_reference(), uri=str(path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/mochi-generation",
        version="2026.11.23",
        serving_profile=ServingProfileName.MOCHI_GB200,
    )
    context = StepContext.for_fingerprint(step.runtime_args, step.deps)
    run = step.build_config(context)
    report = validate_launch(run.skyrl.launch_config_yaml, tmp_path / "preflight")
    compiled = yaml.safe_load((tmp_path / "preflight" / "compiled.yaml").read_text())
    allocation = compiled["iris"]["allocation"]
    assert allocation["num_nodes"] == 4
    assert allocation["gpus_per_node"] == 4
    assert report["runtime_entrypoint"] == "skyrl_train.entrypoints.main_generate"


@pytest.mark.timeout(180)
@pytest.mark.parametrize("profile", list(ServingProfileName))
@pytest.mark.skipif(sys.platform == "darwin", reason="The frozen policy builds Linux bubblewrap task factories")
def test_together_policy_builds_real_rollout_worker_without_inference_engines(tmp_path, monkeypatch, profile):
    # The production evaluator and installed-agent blocks are frozen verbatim;
    # only the model/tokenizer handle is replaced with a small offline fixture.
    path = Path(__file__).with_name("fixtures") / "micro210-together-runtime.json"
    raw = path.read_bytes()
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_fixture_reference(), uri=str(path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")),
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    monkeypatch.setenv("JUDGE_API_KEY", "offline-judge")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "offline-daytona")
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, str(tmp_path / "tokenizer"), "b" * 40),
        name="fixture/together-worker",
        version="2026.11.20",
        serving_profile=profile,
    )
    context = StepContext.for_run(
        str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
    )
    run = step.build_config(replace(context, is_fingerprint=True))
    report = validate_launch(run.skyrl.launch_config_yaml, tmp_path / "preflight", build_worker=True)
    # Construction is the startup contract: the regression previously failed
    # here on the unsupported installed_agent.judge_model field, before any trial.
    assert report["rollout_worker"] == {
        "worker_type": "TaskRolloutWorker",
        "installed_agent": True,
        "judge_base_url": json.loads(raw)["evaluation"]["judge"]["base_url"],
        "inference_engines": 0,
    }


@pytest.mark.timeout(120)
def test_injected_judge_field_is_rejected_before_worker_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    step = micro210_step(
        Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40),
        name="fixture/unsupported-judge",
        version="2026.11.20",
    )
    context = StepContext.for_run(
        str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
    )
    launch = yaml.safe_load(step.build_config(replace(context, is_fingerprint=True)).skyrl.launch_config_yaml)
    launch["skyrl"]["terminal_bench"]["installed_agent"]["judge_model"] = "zai-org/GLM-5.3-Flash"
    with pytest.raises(ValueError, match="judge_model"):
        validate_launch(yaml.safe_dump(launch), tmp_path / "preflight")
