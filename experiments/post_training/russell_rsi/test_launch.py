# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import asdict

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.experiment import cli as experiment_cli
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi import launch as russell_launch
from experiments.post_training.russell_rsi import launch_bootstrap_loop
from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    LoopState,
    Measurement,
    QualifiedTask,
    RoundResult,
    StopReason,
    advance,
    round_plan,
    seal_round,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingPanel, PanelItem
from experiments.post_training.russell_rsi.launch import (
    MODEL,
    MODEL_REVISION,
    bootstrap_round_workflow,
    development_step,
    main,
    repair_spike_workflow,
    spike_workflow,
)
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV, IRIS_TASK_ID_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256


def test_coordinator_submits_new_config_and_reuses_identical_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setenv(IRIS_TASK_ID_ENV, "/test/coordinator/0")
    monkeypatch.setenv(GLM_TOKEN_ENV, "test-token")
    submissions = tmp_path / "submissions.jsonl"

    def remote_submission(function, **kwargs):
        def submit(config):
            with submissions.open("a") as stream:
                stream.write(json.dumps(config) + "\n")

        return submit

    monkeypatch.setattr(launch_bootstrap_loop, "remote", remote_submission)
    initial = {
        "version": "2026.10.04.5",
        "manifest_prefix": "s3://test/immutable-rounds",
        "reviewed_banks": {},
    }
    reviewed = {**initial, "reviewed_banks": {"1": {}}}
    for index, config in enumerate((initial, reviewed, reviewed)):
        source = tmp_path / f"config-{index}.json"
        source.write_text(json.dumps(config) + "\n")
        # Each coordinator process owns its cached Iris job metadata.
        result = copy_context().run(
            CliRunner().invoke,
            launch_bootstrap_loop.main,
            [
                "--config-uri",
                str(source),
                "--config-sha256",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                "--version",
                config["version"],
                "--run",
            ],
        )
        assert result.exit_code == 0, result.output + str(result.exception)
    assert [json.loads(line) for line in submissions.read_text().splitlines()] == [initial, reviewed]


def test_spike_retains_frozen_development_and_checks_rewards_before_policy_allocation():
    seed = ArtifactStep.adopt("documents/test-russell-seed", "2026.10.04", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/test-russell-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/test-russell-wheels", "2026.10.04", "/tmp/wheels")
    sources = ArtifactStep.adopt("documents/test-russell-sources", "2026.10.04", "/tmp/sources")
    terminals = spike_workflow(
        seed,
        parent,
        "smoke",
        "2026.10.04",
        "test-relay",
        "test-image",
        RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
        {"backend": "qemu", "qemu": {}},
        wheels,
        sources,
        "0" * 64,
    )
    handles = {handle.name: handle for handle in graph_handles(list(terminals.values()))}
    baseline = handles["evals/russell-rsi-parent-development"]
    adaptive = handles["documents/russell-rsi-adaptive-round-1"]
    calibration = handles["evals/russell-rsi-train-calibration-development"]
    trained = handles["checkpoints/russell-rsi-smoke"]
    assert baseline in adaptive.deps
    assert sources in adaptive.deps
    assert sources not in baseline.deps
    assert calibration in trained.deps
    launch = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    assert launch["inputs"]["train_data"][0]["identity"] == artifact_identity(adaptive)
    assert launch["inputs"]["validation_data"][0]["identity"] == artifact_identity(seed)
    calibration_config = json.loads(calibration.fingerprint_payload())
    assert calibration_config["require_reward_variation"] is True
    assert calibration_config["samples_per_task"] == 4


def test_spike_run_rejects_missing_generation_token_before_building(monkeypatch):
    monkeypatch.delenv("GLM_API_TOKEN", raising=False)
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04",
            "--stage",
            "spike",
            "--scale",
            "smoke",
            "--data-name",
            "documents/test-seed",
            "--data-version",
            "2026.10.04",
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--run",
        ],
    )
    assert result.exit_code == 2
    assert "requires GLM_API_TOKEN before any GPU work" in result.output


def test_smoke_and_pilot_share_admission_without_evaluation_collisions():
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.2", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels")
    sources = ArtifactStep.adopt("documents/sources", "2026.10.04.2", "/tmp/sources")
    terminals = {
        scale: spike_workflow(
            seed,
            parent,
            scale,
            "2026.10.04.2",
            "test-relay",
            "test-image",
            RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
            {"backend": "qemu", "qemu": {}},
            wheels,
            sources,
            "0" * 64,
        )
        for scale in ("smoke", "pilot")
    }
    handles = {
        scale: {handle.name: handle for handle in graph_handles(list(outputs.values()))}
        for scale, outputs in terminals.items()
    }
    smoke = handles["smoke"]
    pilot = handles["pilot"]
    assert {name for name in smoke if name.startswith("evals/")} == {
        "evals/russell-rsi-parent-development",
        "evals/russell-rsi-train-calibration-development",
        terminals["smoke"]["reload"].name,
    }
    for name in (
        "evals/russell-rsi-parent-development",
        "documents/russell-rsi-adaptive-round-1",
        "evals/russell-rsi-train-calibration-development",
    ):
        assert artifact_identity(smoke[name]) == artifact_identity(pilot[name])
    assert terminals["pilot"]["development"].name == "evals/russell-rsi-candidate-pilot-development"
    assert artifact_identity(smoke["checkpoints/russell-rsi-smoke"]) != artifact_identity(
        pilot["checkpoints/russell-rsi-pilot"]
    )
    assert terminals["smoke"]["reload"].name not in pilot
    assert all(handle.name not in smoke for handle in terminals["pilot"].values())
    candidate = terminals["pilot"]["development"]
    candidate_public = terminals["pilot"]["candidate-coding-subset"]
    pilot_reload = next(handle for handle in candidate_public.deps if "mmlu-smoke" in handle.name)
    assert pilot_reload in candidate.deps


def test_parent_development_plan_reuses_spike_baseline_without_training(monkeypatch):
    captured = []
    monkeypatch.setattr(experiment_cli, "_print_plan", captured.extend)
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04.2",
            "--stage",
            "parent-development",
            "--scale",
            "smoke",
            "--data-name",
            "documents/frozen-seed",
            "--data-version",
            "2026.10.04.2",
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--machine-config-json",
            json.dumps(
                {
                    "backend": "qemu",
                    "runtime_bundle": {
                        "manifest_uri": "/tmp/manifest.json",
                        "manifest_sha256": "0" * 64,
                        "archive_uri": "/tmp/runtime.tar.gz",
                        "archive_sha256": "0" * 64,
                    },
                }
            ),
        ],
    )
    assert result.exit_code == 0, result.output
    assert [handle.name for handle in graph_handles(captured)] == [
        "documents/frozen-seed",
        "checkpoints/russell-sft-parent",
        "evals/russell-rsi-parent-development",
    ]
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.2", "/tmp/seed")
    parent = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        "/tmp/model",
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    terminals = spike_workflow(
        seed,
        parent,
        "smoke",
        "2026.10.04.2",
        "test-relay",
        "test-image",
        RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
        {"backend": "qemu", "qemu": {}},
        ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels"),
        ArtifactStep.adopt("documents/sources", "2026.10.04.2", "/tmp/sources"),
        "0" * 64,
    )
    baseline = next(
        handle
        for handle in graph_handles(list(terminals.values()))
        if handle.name == "evals/russell-rsi-parent-development"
    )
    assert artifact_identity(captured[0]) == artifact_identity(baseline)


def test_repair_spike_uses_qualified_union_without_running_failed_adaptive_round():
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.1", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels")
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    terminals = {
        scale: repair_spike_workflow(
            seed,
            parent,
            scale,
            "2026.10.04.2",
            relay_job="test-relay",
            image="test-image",
            runtime_bundle=runtime,
            machine_config={"backend": "qemu"},
            wheels=wheels,
            manifest_uri="/tmp/sealed-manifest.json",
            manifest_sha256="1" * 64,
        )
        for scale in ("smoke", "pilot")
    }
    graphs = {
        scale: {handle.name: handle for handle in graph_handles(list(outputs.values()))}
        for scale, outputs in terminals.items()
    }
    smoke, pilot = graphs["smoke"], graphs["pilot"]
    assert "documents/russell-rsi-adaptive-round-1" not in smoke
    union = smoke["documents/russell-rsi-qualified-union-1"]
    repaired = smoke["documents/russell-rsi-repair-1"]
    evidence = smoke["documents/russell-rsi-round-1-evidence"]
    baseline = smoke["evals/russell-rsi-parent-development"]
    union_config = json.loads(union.fingerprint_payload())
    assert union_config["parent_development_identity"] == artifact_identity(baseline)
    assert union_config["original_manifest_sha256"] == "1" * 64
    assert evidence in repaired.deps and repaired in union.deps
    trained = smoke["checkpoints/russell-rsi-repair-1-smoke"]
    launch = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    assert launch["inputs"]["train_data"][0]["identity"] == artifact_identity(union)
    assert launch["inputs"]["validation_data"][0]["identity"] == artifact_identity(seed)
    calibration = smoke["evals/russell-rsi-repair-1-train-calibration-development"]
    assert calibration in trained.deps
    for handle in (baseline, repaired, union, calibration):
        assert artifact_identity(handle) == artifact_identity(pilot[handle.name])
    assert terminals["smoke"].keys() == {"reload"}
    assert terminals["pilot"]["development"].name == "evals/russell-rsi-candidate-repair-1-pilot-development"


@pytest.mark.parametrize("change", ["bytes", "parent"])
def test_repair_cli_rejects_changed_evidence_before_building(change, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(experiment_cli, "_print_plan", captured.extend)
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.1", "/tmp/seed")
    parent = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        "/tmp/model",
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    baseline = development_step(seed, parent, "2026.10.04.2", runtime, "parent")
    identity = artifact_identity(baseline) if change == "bytes" else "different-parent"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": {"parent_development_identity": identity}}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if change == "bytes":
        manifest.write_text(manifest.read_text() + "\n")
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04.2",
            "--stage",
            "repair-spike",
            "--scale",
            "smoke",
            "--data-name",
            seed.name,
            "--data-version",
            seed.version,
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--relay-job",
            "test-relay",
            "--task-image",
            "test-image",
            "--dependency-wheels-uri",
            "/tmp/wheels",
            "--repair-manifest-uri",
            str(manifest),
            "--repair-manifest-sha256",
            digest,
            "--machine-config-json",
            json.dumps(
                {
                    "backend": "qemu",
                    "runtime_bundle": {
                        "manifest_uri": runtime.manifest_uri,
                        "manifest_sha256": runtime.manifest_sha256,
                        "archive_uri": runtime.archive_uri,
                        "archive_sha256": runtime.archive_sha256,
                    },
                }
            ),
        ],
    )
    assert result.exit_code != 0
    error = str(result.exception) + result.output
    assert ("digest mismatch" if change == "bytes" else "parent-development identity") in error
    assert captured == []


def test_bootstrap_round_uses_coding_eval_feedback_after_reload_without_parent_rerun():
    training = ArtifactStep.adopt("documents/bootstrap-bank", "2026.10.04", "/tmp/bank")
    retention = ArtifactStep.adopt("documents/retention", "2026.10.04", "/tmp/retention")
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    calibration = ArtifactStep.adopt("documents/bank-difficulty", "2026.10.04", "/tmp/difficulty")
    terminals = bootstrap_round_workflow(
        training,
        retention,
        parent,
        "2026.10.04",
        runtime,
        {"backend": "qemu"},
        round_number=1,
        panel=CodingPanel((), {}),
        relay_job="relay",
        calibration=calibration,
    )
    trained = terminals["rl"]
    assert calibration in trained.deps
    config = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    assert config["skyrl"]["trainer"]["epochs"] == 4
    assert config["skyrl"]["trainer"]["max_steps"] == 4
    assert terminals["reload"] in terminals["coding-development"].deps
    assert terminals["coding-development"] in terminals["coding-evidence"].deps
    assert terminals["capabilities"].deps == (terminals["coding-evidence"],)


def test_bootstrap_driver_freezes_holdout_and_stops_before_gpu_work_for_twelve_contracts(tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    bank.mkdir()
    records = [
        {
            "task_id": str(index),
            "task_sha256": f"task-{index}",
            "admission_sha256": f"admission-{index}",
            "source_id": f"source-{index}",
            "capability": "types",
            "contract_id": f"contract-{index}",
        }
        for index in range(12)
    ]
    (bank / "bank.json").write_text(json.dumps({"tasks": records, "feedback_identity": "parent-coding"}))
    seed = ArtifactStep.adopt("documents/twelve-qualified", "2026.10.04", str(bank))
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    retention = ArtifactStep.adopt("documents/retention", "2026.10.04", "/tmp/retention")
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            {
                "suites": {
                    suite: {"sample_ids": [f"held-out-{i}" for i in range(32)], "demonstration_ids": []}
                    for suite in ("humanevalplus", "mbppplus")
                }
            }
        )
    )
    holdout_hash = hashlib.sha256(holdout.read_bytes()).hexdigest()

    def resolve_bank(handle):
        if handle is not seed:
            raise AssertionError("A GPU stage started before bank qualification")
        return Artifact(path=str(bank))

    def no_source_builder(*args):
        raise AssertionError("The driver reached source construction before admission")

    monkeypatch.setattr(russell_launch, "resolve", resolve_bank)
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    manifests = tmp_path / "manifests"
    panel = CodingPanel(
        tuple(
            PanelItem(suite, str(index), "prompt-hash") for suite in ("humanevalplus", "mbppplus") for index in range(32)
        ),
        {suite: "protocol" for suite in ("humanevalplus", "mbppplus")},
    )
    coding = tmp_path / "parent-coding.json"
    coding.write_text(
        json.dumps(
            {
                "model_identity": artifact_identity(parent),
                "panel_sha256": compact_json_sha256(asdict(panel)),
                "scores": {"humanevalplus": 25 / 32, "mbppplus": 25 / 32},
            }
        )
    )
    retention_evidence = tmp_path / "parent-retention.json"
    retention_evidence.write_text(
        json.dumps(
            {
                "model_identity": artifact_identity(parent),
                "tasks_identity": artifact_identity(retention),
                "count": 1,
                "task_rewards": {"task": [1.0]},
            }
        )
    )
    with pytest.raises(ValueError, match="Do not allocate"):
        russell_launch.run_bootstrap_loop(
            seed,
            parent,
            retention,
            panel,
            str(holdout),
            holdout_hash,
            str(coding),
            hashlib.sha256(coding.read_bytes()).hexdigest(),
            str(retention_evidence),
            hashlib.sha256(retention_evidence.read_bytes()).hexdigest(),
            "2026.10.04",
            runtime,
            {"backend": "qemu"},
            "relay",
            StoragePath(str(manifests)),
            no_source_builder,
        )
    assert json.loads((manifests / "panels.json").read_text())["manifest_sha256"] == holdout_hash
    assert list(manifests.glob("*pilot*.json")) == []


@pytest.mark.parametrize(
    "next_bank_supplied,calibration_source",
    [
        (True, "normal"),
        (False, "normal"),
        (False, "changed_recovery"),
        (False, "fresh_recovery"),
        (False, "incomplete_recovery"),
    ],
)
def test_driver_validates_calibration_before_training_or_resume(
    tmp_path, monkeypatch, next_bank_supplied, calibration_source
):
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    bank = tuple(
        QualifiedTask(str(i), f"hash-{i}", f"proof-{i}", f"source-{i}", "types", f"contract-{i}") for i in range(16)
    )
    (bank_path / "bank.json").write_text(
        json.dumps({"tasks": [asdict(task) for task in bank], "feedback_identity": "seed-feedback"})
    )
    seed = ArtifactStep.adopt("documents/sealed-bank", "2026.10.04", str(bank_path))
    parent = ArtifactStep.adopt("checkpoints/parent", "2026.10.04", "/tmp/parent", kind=LevanterCheckpoint)
    retention = ArtifactStep.adopt("documents/retention", "2026.10.04", "/tmp/retention")
    panel = CodingPanel(
        tuple(PanelItem(suite, str(i), "prompt") for suite in ("humanevalplus", "mbppplus") for i in range(32)),
        {suite: "protocol" for suite in ("humanevalplus", "mbppplus")},
    )
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            {
                "suites": {
                    suite: {"sample_ids": [f"held-{i}" for i in range(32)], "demonstration_ids": []}
                    for suite in panel.protocols
                }
            }
        )
    )
    coding = tmp_path / "coding.json"
    coding.write_text(
        json.dumps(
            {
                "model_identity": artifact_identity(parent),
                "panel_sha256": compact_json_sha256(asdict(panel)),
                "scores": {suite: 25 / 32 for suite in panel.protocols},
            }
        )
    )
    retained = tmp_path / "retention.json"
    retained.write_text(
        json.dumps(
            {
                "model_identity": artifact_identity(parent),
                "tasks_identity": artifact_identity(retention),
                "task_rewards": {"task": [1]},
                "count": 1,
            }
        )
    )
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (holdout, coding, retained)]
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    difficulty = ArtifactStep.adopt("evals/difficulty", "2026.10.04", "/tmp/difficulty")
    reload = ArtifactStep.adopt("evals/reload", "2026.10.04", "/tmp/reload")
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    (capabilities_path / "capabilities.json").write_text(json.dumps({"skills": [{"label": "types"}]}))
    capabilities = ArtifactStep.adopt("documents/capabilities", "2026.10.04", str(capabilities_path))
    trained = ArtifactStep.adopt("checkpoints/trained", "2026.10.04", "/tmp/trained")
    outputs = {"rl": trained, "reload": reload, "capabilities": capabilities}
    monkeypatch.setattr(russell_launch, "development_step", lambda *args, **kwargs: difficulty)
    if calibration_source not in ("fresh_recovery", "incomplete_recovery"):
        monkeypatch.setattr(russell_launch, "bootstrap_round_workflow", lambda *args, **kwargs: outputs)
    monkeypatch.setattr(russell_launch, "run", lambda *args: pytest.fail("Resume started GPU work"))

    def resolver(handle):
        if handle is difficulty:
            if calibration_source in ("fresh_recovery", "incomplete_recovery"):
                return Artifact(path=str(tmp_path / "recovery"))
            pytest.fail("Resume repeated calibration")
        if handle is seed:
            return Artifact(path=str(bank_path))
        if handle is capabilities:
            return Artifact(path=str(capabilities_path))
        return Artifact(path="/tmp/final")

    monkeypatch.setattr(russell_launch, "resolve", resolver)
    score = CheckpointScore(artifact_identity(parent), (25 / 32, 25 / 32), 1)
    initial = LoopState(score, score, score, bank)
    plan = round_plan(
        initial,
        (),
        tuple(Measurement(score.checkpoint_identity, task.task_sha256, (1, 0) * 4) for task in bank),
        run_id="bootstrap-2026.10.04",
        bank_identity=artifact_identity(seed),
        calibration_identity=artifact_identity(difficulty),
        feedback_labels=(),
        development_identity=compact_json_sha256(asdict(panel)),
        retention_identity=artifact_identity(retention),
        feedback_identity="seed-feedback",
        runtime_identity=compact_json_sha256(
            {"qemu": runtime.archive_sha256, "skyrl": MARIN_SKYRL.commit, "machine": {"backend": "qemu"}}
        ),
        seed=9528,
    )
    adopted = ArtifactStep.adopt(
        "checkpoints/russell-rsi-bootstrap-round-1-export",
        "2026.10.04",
        "/tmp/export",
        kind=LevanterCheckpoint,
        config={"producer": artifact_identity(reload)},
    )
    result = RoundResult(
        CheckpointScore(artifact_identity(adopted), (24 / 32, 25 / 32), 1),
        artifact_identity(reload),
        artifact_identity(capabilities),
        4,
        "/tmp/export",
    )
    state = advance(initial, plan, result)
    directory = StoragePath(str(tmp_path / "manifests"))
    previous = compact_json_sha256(
        {
            "manifest_sha256": hashes[0],
            "development": asdict(panel),
            "source_bank": artifact_identity(seed),
            "parent_coding_sha256": hashes[1],
            "parent_retention_sha256": hashes[2],
        }
    )
    allocation_plans = []
    if calibration_source in ("fresh_recovery", "incomplete_recovery"):
        recovery_path = tmp_path / "recovery"
        recovery_path.mkdir()
        rewards = {task.task_id: [1, 0] * 4 for task in bank}
        if calibration_source == "incomplete_recovery":
            rewards[bank[0].task_id].pop()
        (recovery_path / "failure_summary.json").write_text(
            json.dumps(
                {
                    "model_identity": artifact_identity(parent),
                    "tasks_identity": artifact_identity(seed),
                    "count": len(bank),
                    "samples_per_task": 8,
                    "task_rewards": rewards,
                }
            )
        )

        def allocation_boundary(*handles):
            for handle in graph_handles(list(handles)):
                if handle.name == "documents/russell-rsi-bootstrap-round-1-train":
                    frozen_config = handle.build_config(StepContext.for_fingerprint(deps=handle.deps))
                    allocation_plans.append(frozen_config.plan)
            raise RuntimeError("Training allocation intercepted")

        monkeypatch.setattr(russell_launch, "run", allocation_boundary)
    else:
        seal_round(directory, state, plan, result, previous)
        russell_launch.write_once(
            directory / "smoke.json",
            {
                "rl": artifact_identity(trained),
                "reload": artifact_identity(reload),
                "coding_baseline_sha256": hashes[1],
                "retention_baseline_sha256": hashes[2],
            },
        )
    initial_calibration = None if calibration_source == "normal" else difficulty
    if calibration_source == "changed_recovery":
        initial_calibration = ArtifactStep.adopt(
            "evals/difficulty", "2026.10.04", "/tmp/difficulty", config={"review": "changed"}
        )
    expected = (
        pytest.raises(ValueError, match="Round resume input mismatch")
        if calibration_source == "changed_recovery"
        else nullcontext()
    )
    if calibration_source == "fresh_recovery":
        expected = pytest.raises(RuntimeError, match="Training allocation intercepted")
    elif calibration_source == "incomplete_recovery":
        expected = pytest.raises(ValueError, match="eight finite grades")
    with expected:
        restored = russell_launch.run_bootstrap_loop(
            seed,
            parent,
            retention,
            panel,
            str(holdout),
            hashes[0],
            str(coding),
            hashes[1],
            str(retained),
            hashes[2],
            "2026.10.04",
            runtime,
            {"backend": "qemu"},
            "relay",
            directory,
            lambda *args: seed if next_bank_supplied else None,
            initial_calibration=initial_calibration,
        )
    if calibration_source == "changed_recovery":
        assert not (directory / "terminal-state.json").exists()
        return
    if calibration_source in ("fresh_recovery", "incomplete_recovery"):
        assert not (directory / "smoke.json").exists()
        if calibration_source == "fresh_recovery":
            assert len(allocation_plans) == 1
            assert allocation_plans[0].calibration_identity == artifact_identity(difficulty)
            assert len(allocation_plans[0].selected_tasks) == 16
        else:
            assert allocation_plans == []
        return
    assert restored.completed_pilots == 1
    assert restored.stop_reason == (StopReason.TASK_SUPPLY if next_bank_supplied else None)
    assert (directory / "terminal-state.json").exists() == next_bank_supplied
    assert (directory / "construction-required-after-1.json").exists() != next_bank_supplied
    assert restored.working == restored.champion == score
