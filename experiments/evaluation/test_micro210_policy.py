# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
import sys
from dataclasses import replace
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from finestore.rl.run_records import METRICS_TABLE, AttemptRecord, register_run_tables, run_records_root, write_attempt
from finestore.store import DataStore
from marin.execution.artifact import (
    RECORD_FILENAME,
    ArtifactRecord,
    FingerprintMismatchError,
    read_record,
    result_type_name,
)
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, lower, materialized_config, run
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch, sections

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210 import (
    PREPARED_INPUTS,
    native_binding_code_sha256,
    run_micro210,
)
from experiments.evaluation.micro210 import (
    micro210_step as build_micro210_step,
)
from experiments.evaluation.micro210_policy import (
    LEGACY_POLICY_REFERENCE,
    POLICY_REFERENCE,
    Micro210Model,
    micro210_evaluation,
    read_micro210_policy,
)
from experiments.grug.moe_hero_ep.ops.export_metadata import MANIFEST_FILENAME as SFT_EXPORT_MANIFEST
from experiments.post_training.mochi_rl import MICRO210_RECIPE as MOCHI_SERVING
from experiments.post_training.rl_gate import compose_configs
from experiments.post_training.snowball_rl import MICRO210_RECIPE as SNOWBALL_SERVING
from tests.rl.micro210_fixture import policy_fixture_reference
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

micro210_step = partial(build_micro210_step, serving=SNOWBALL_SERVING)


@pytest.fixture(autouse=True)
def policy_reference(monkeypatch):
    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", reference)
    monkeypatch.setattr("experiments.evaluation.micro210.NATIVE_INPUTS_FINGERPRINT", None)
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


@pytest.mark.parametrize("producer_kind", ["rl", "sft", "levanter", "rl-initializer"])
def test_micro210_launch_records_protocol_and_rejects_changed_inputs_before_gpu_allocation(
    tmp_path, monkeypatch, policy_reference, producer_kind
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = b'{"files": []}'
    manifest_name = MODEL_MANIFEST_FILENAME if producer_kind == "rl" else SFT_EXPORT_MANIFEST
    manifest_path = model_dir / manifest_name
    unmanifested = producer_kind in ("levanter", "rl-initializer")
    tokenizer = ("test/tokenizer", "a" * 40) if producer_kind == "rl" else (None, None)
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_dir / filename).write_bytes(b"model fixture")
    if not unmanifested:
        manifest_path.write_bytes(manifest)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    task_bytes = b"pinned-prepared-task-fixture"
    (inputs / "rollouts.parquet").write_bytes(task_bytes)
    protocol = json.loads(StoragePath(policy_reference.uri).read_text())
    protocol["prepared_rollouts_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    expected_rows = 434
    protocol["task_changes"] = {"sealed_rows": expected_rows}
    protocol_path = tmp_path / "protocol.json"
    raw = json.dumps(protocol).encode()
    protocol_path.write_bytes(raw)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(policy_reference, uri=str(protocol_path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setattr("experiments.evaluation.micro210.PREPARED_INPUTS", str(inputs))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    token_content = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    producer_dir = tmp_path / "producer"
    producer_dir.mkdir()
    if producer_kind in ("rl", "rl-initializer"):
        producer = ArtifactStep.adopt("test/rl", "2026.11.25", str(producer_dir), kind=SkyRLRun)
        result = SkyRLRun(
            path=str(producer_dir),
            hf_model_uri=str(model_dir),
            global_step=0 if producer_kind == "rl-initializer" else 3,
            tokenizer_uri=tokenizer[0],
            tokenizer_revision=tokenizer[1],
            checkpoint_root=str(producer_dir / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(producer_dir / "terminal.json"),
            iris_job_id="/test/run",
        )
        (producer_dir / RECORD_FILENAME).write_text(
            ArtifactRecord(
                name=producer.name,
                version=producer.version,
                result_type=result_type_name(SkyRLRun),
                result=result.result_payload(),
            ).model_dump_json()
        )
        model = SkyRLPolicyModel(producer)
    else:
        producer = ArtifactStep.adopt("test/sft", "2026.11.25", str(model_dir))
        model = ArtifactHfModel(producer, relative_path="")

    def load_bound_tokenizer(uri, *, revision, trust_remote_code):
        assert (uri, revision) == (tokenizer[0] or str(model_dir), tokenizer[1])
        assert not trust_remote_code
        return SimpleNamespace(convert_ids_to_tokens=token_content.__getitem__)

    monkeypatch.setattr("levanter.compat.hf_checkpoints.load_tokenizer", load_bound_tokenizer)
    serving = MOCHI_SERVING if producer_kind == "sft" else SNOWBALL_SERVING
    step = micro210_step(model, name="test/micro210", version="2026.11.18", serving=serving)
    assert producer in step.deps
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
                "native_sha256": native_digest,
                "smoke_rows": [],
                "converter_sha256": native_binding_code_sha256(),
                "image_digests": json.loads(Path(__file__).with_name("micro210_images.json").read_text()),
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

    monkeypatch.setattr("experiments.evaluation.micro210.resolve_secret_spec", resolve_key)

    scored = expected_rows
    score = 0.5

    def submit_evaluation(skyrl, *, env_vars):
        assert env_vars == {"JUDGE_API_KEY": "test-together-key", "OPENAI_API_KEY": "test-together-key"}
        launch = yaml.safe_load(skyrl.launch_config_yaml)
        assert "records" in launch["skyrl"]["trainer"]["logger"]
        terminal = Path(skyrl.output.terminal_manifest_uri)
        assert not terminal.exists()
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.touch()
        submitted.append(launch)
        write_attempt(skyrl.output.export_root, AttemptRecord(run_id=skyrl.run_id, attempt_id=skyrl.attempt_id))
        with DataStore.open(run_records_root(skyrl.output.export_root)) as store:
            register_run_tables(store)[METRICS_TABLE].extend(
                [
                    {"attempt_id": skyrl.attempt_id, "step": 0, "name": name, "value": value}
                    for name, value in (
                        ("eval/all/num_attempted", expected_rows),
                        ("eval/all/num_scored", scored),
                        ("eval/all/avg_verifier_score", score),
                    )
                ]
            )
            store.flush()
        return None

    def refuse_hosted_judge(*args, **kwargs):
        raise AssertionError("API judging must not prepare or provision a hosted judge")

    monkeypatch.setattr("marin.inference.iris.prepare_inference", refuse_hosted_judge)
    monkeypatch.setattr("experiments.evaluation.micro210.run_skyrl", submit_evaluation)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE", replace(policy_reference, sha256="0" * 64)
    )
    run_micro210(config)
    launch = submitted[0]
    assert launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] == "https://api.together.ai/v1"
    assert launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["api_key_env"] == "JUDGE_API_KEY"
    assert launch["skyrl"]["environment"]["task_sessions"]["agent_harnesses"]["pi"]["version"] == "0.87.0"
    assert all("test-together-key" not in str(value) for value in launch.values())
    record = json.loads((output / "canonical-evaluation.json").read_text())
    identity = (
        {"artifact": artifact_identity(producer)} if unmanifested else {"sha256": hashlib.sha256(manifest).hexdigest()}
    )
    assert record["model"]["identity"] == identity
    assert record["attempt_id"] == config.skyrl.attempt_id
    assert record["score"] == 0.5
    assert record["canonical_spec_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["native_sha256"] == native_digest
    assert record["native_inputs_uri"] == native_inputs_uri
    if producer_kind != "rl":
        return
    assert (
        submitted[0]["skyrl"]["generator"]["eval_sampling_params"]
        == protocol["evaluation"]["config"]["generator"]["eval_sampling_params"]
    )
    assert launch["skyrl"]["generator"]["error_handling"] == {
        "enable_error_classification": True,
        "default_error_treatment": "mask",
        "passthrough_exceptions": ["ContextLengthExceededError", "AgentTimeoutError"],
    }
    assert "terminal_bench" not in submitted[0]["skyrl"]
    assert launch["skyrl"]["entrypoint"] == "generate"
    assert submitted[0]["skyrl"]["generator"]["trajectory_retention"]["run_id"] == "test/micro210@2026.11.18"
    altered = yaml.safe_load(config.skyrl.launch_config_yaml)
    altered["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.1
    submitted.clear()
    with pytest.raises(ValueError, match="launch_config_yaml"):
        run_micro210(replace(config, skyrl=replace(config.skyrl, launch_config_yaml=yaml.safe_dump(altered))))
    assert submitted == []
    (native_root / "rollouts.parquet").write_bytes(b"changed-native-tasks")
    with pytest.raises(ValueError, match="native task bytes changed"):
        run_micro210(config)
    assert submitted == []
    (native_root / "rollouts.parquet").write_bytes(native_bytes)
    with pytest.raises(ValueError, match="different smoke rows"):
        run_micro210(replace(config, smoke_rows=(1,)))
    assert submitted == []
    if not unmanifested:
        manifest_path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="export manifest differs"):
            run_micro210(replace(config, expected_manifest_sha256=hashlib.sha256(manifest).hexdigest()))
        manifest_path.write_bytes(manifest)
    assert submitted == []
    scored -= 1
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    incomplete_config = step.build_config(
        StepContext.for_run(str(incomplete), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    )
    with pytest.raises(ValueError, match="434 scored rows"):
        run_micro210(incomplete_config)
    assert not (incomplete / "canonical-evaluation.json").exists()
    scored += 1
    score = float("nan")
    rejected = step.build_config(
        StepContext.for_run(str(incomplete), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    )
    with pytest.raises(ValueError, match="one finite score"):
        run_micro210(rejected)
    assert not (incomplete / "canonical-evaluation.json").exists()
    score = 0.5
    retry = step.build_config(
        StepContext.for_run(str(incomplete), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    )
    run_micro210(retry)
    assert Path(incomplete_config.skyrl.output.terminal_manifest_uri).exists()
    assert json.loads((incomplete / "canonical-evaluation.json").read_text())["attempt_id"] == retry.skyrl.attempt_id
    submitted.clear()
    token_content[128009] = "ordinary-token"
    with pytest.raises(ValueError, match="incompatible stop token ID: 128009"):
        run_micro210(config)
    assert submitted == []
    (inputs / "rollouts.parquet").write_bytes(b"changed-tasks")
    submitted.clear()
    with pytest.raises(ValueError, match="task bytes changed"):
        run_micro210(config)
    assert submitted == []


@pytest.mark.parametrize("serving", [SNOWBALL_SERVING, MOCHI_SERVING], ids=["snowball", "mochi"])
def test_canonical_evaluation_uses_model_serving_in_native_launch(tmp_path, serving):
    model = Micro210Model(str(tmp_path / "export"), "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    serving = replace(serving, config=serving.config.merge(RecipePatch(generator=sections.Generator(max_num_seqs=17))))
    step = micro210_step(model, name="fixture/final-eval", version="2026.11.18.74", serving=serving)
    launch_path = tmp_path / "launch.yaml"
    launch_path.write_text(materialized_config(step, str(tmp_path)).skyrl.launch_config_yaml)
    [launch] = compose_configs([str(launch_path)], validate_driver=True)
    assert "records" in launch["skyrl"]["trainer"]["logger"]
    generator = launch["skyrl"]["generator"]
    assert generator["max_num_seqs"] == serving.config.generator.max_num_seqs
    if serving.layout.gpu_variant == "GB200":
        assert launch["iris"]["allocation"]["num_nodes"] == 4


@pytest.mark.parametrize("model", ["snowball", "mochi"])
def test_standalone_coordinator_reenters_the_model_command(monkeypatch, model):
    module = f"experiments.post_training.{model}_rl"
    submissions = []

    def submit(entrypoint, handles, _concurrency, _environment, *, argv):
        submissions.append((entrypoint, argv, tuple(step.fingerprint() for step in handles)))

    monkeypatch.setattr("marin.rl.cli.submit_or_run", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module,
            "micro210",
            "--run",
            "--version",
            "2026.11.26",
            "--name",
            "tests/micro210",
            "--model-uri",
            "s3://fixture/model",
            "--model-manifest-sha256",
            "a" * 64,
            "--tokenizer-uri",
            "test/tokenizer",
            "--tokenizer-revision",
            "b" * 40,
        ],
    )
    with pytest.raises(SystemExit) as invoked:
        import_module(module).main()
    assert invoked.value.code == 0
    [(entrypoint, argv, _)] = submissions
    assert entrypoint == module and argv[0] == "micro210"

    monkeypatch.setattr(sys, "argv", [entrypoint, *argv])
    with pytest.raises(SystemExit) as reentered:
        import_module(entrypoint).main()
    assert reentered.value.code == 0
    assert submissions[0] == submissions[1]


def test_micro210_serving_refuses_training_overrides_and_multiple_samples():
    model = Micro210Model("s3://fixture/export", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    patch = RecipePatch(trainer=sections.Trainer(eval_batch_size=99))
    with pytest.raises(ValueError, match="only change inference serving settings"):
        micro210_step(model, name="fixture/eval", version="2026.11.18", serving=replace(SNOWBALL_SERVING, config=patch))
    layout = SNOWBALL_SERVING.layout
    workload = replace(layout.workload, plan=replace(layout.workload.plan, n_samples_per_prompt=2))
    with pytest.raises(ValueError, match="one sample per row"):
        micro210_step(
            model,
            name="fixture/eval",
            version="2026.11.18",
            serving=replace(SNOWBALL_SERVING, layout=replace(layout, workload=workload)),
        )


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
    source = Path(__file__).with_name("micro210.py")
    for name in ("micro210_images.json", "micro210_inputs.py", "micro210_policy.py"):
        (tmp_path / name).write_bytes(source.with_name(name).read_bytes())
    monkeypatch.setattr("experiments.evaluation.micro210.__file__", str(tmp_path / "micro210.py"))
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    step = micro210_step(model, name="fixture/pinned-binding", version="2026.11.24")
    native = next(dep for dep in step.deps if dep.name == "evaluation/micro210/native")
    monkeypatch.setattr("experiments.evaluation.micro210.NATIVE_INPUTS_FINGERPRINT", native.fingerprint())
    lower(micro210_step(model, name="fixture/pinned-binding", version="2026.11.24"))
    lock_path = tmp_path / "micro210_images.json"
    if change == "image_lock":
        lock = json.loads(lock_path.read_text())
        lock[next(iter(lock))] = "sha256:" + "f" * 64
        lock_path.write_text(json.dumps(lock))
    else:
        converter = tmp_path / "micro210_inputs.py"
        converter.write_text(converter.read_text() + "\n# changed converter fixture\n")
    with pytest.raises(FingerprintMismatchError, match="expected_fingerprint pins"):
        lower(micro210_step(model, name="fixture/pinned-binding", version="2026.11.24"))
