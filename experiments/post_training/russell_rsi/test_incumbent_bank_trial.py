# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from taskcompendium.environment import (
    ArtifactKind,
    EnvironmentKind,
    EnvironmentSpec,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.parquet import read_task_records, write_task_records

from experiments.post_training.russell_rsi import test_rsi_continuation
from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    IncompleteCalibrationError,
    QualifiedTask,
)
from experiments.post_training.russell_rsi.launch_incumbent_bank_trial import (
    PROTOCOL,
    current_bank_tasks,
    incumbent_bank_workflow,
    main,
    seal_incumbent_calibration,
    seal_incumbent_selection,
)
from experiments.post_training.russell_rsi.launch_rsi_continuation import ContinuationSelectionConfig
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256
from experiments.post_training.russell_rsi.replay import freeze_replay_dataset
from experiments.post_training.russell_rsi.rollout_eval import run_calibration_evaluation

continuation_inputs = test_rsi_continuation.continuation_inputs


@pytest.fixture
def incumbent_inputs(continuation_inputs, tmp_path):
    config, prior, old_families = continuation_inputs
    config = {**config, "protocol": PROTOCOL, "version": "2026.10.06.21", "runtime_commit": MARIN_SKYRL.commit}
    root = tmp_path / "current-bank"
    root.mkdir()

    def pin(name, value):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    verifier = ShellVerifierSpec(
        argv=("true",),
        timeout=5,
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
        artifacts=(VerifierArtifact(source="/submission", target="/submission", kind=ArtifactKind.FILE),),
    )
    families = {**old_families, **{str(i): f"family-{i}" for i in range(28, 32)}}
    tasks = []
    raw_by_id = {}
    for index in range(32):
        identifier = str(index)
        raw = {
            "id": identifier,
            "context": {"events": [{"type": "message", "role": "user", "content": "Complete the fixture task."}]},
            "environment": {"kind": "shellsim"},
            "environment_requirements": {},
            "answer_type": "file",
            "verifier": {"kind": "shell", "parameters_json": verifier.model_dump_json()},
            "source": {"dataset": "fixture", "revision": "1", "row": identifier, "importer_revision": "1"},
        }
        task = (
            prior[index]
            if index < 28
            else QualifiedTask(identifier, "", "", f"source-{index}", "types", f"family-{index}", "independent")
        )
        task = replace(task, task_sha256=canonical_sha256(raw))
        proof = {
            "task_sha256": task.task_sha256,
            "source_group": task.source_id,
        }
        if index < 20:
            proof.update(
                task_id=identifier, split="train", qualified=True, behavioral_acceptance=True, decision="retain"
            )
        else:
            proof.update(contract_id=task.contract_id, result={"accepted": True}, manifest_sha256="fixture")
            if task.relation == "variant":
                proof.update(original_family_id=families[identifier], relation=task.relation)
        proof_bytes = json.dumps(proof, sort_keys=True).encode()
        task = replace(task, admission_sha256=hashlib.sha256(proof_bytes).hexdigest())
        target = root / "evidence" / task.admission_sha256 / "proposal.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(proof_bytes)
        tasks.append(task)
        raw_by_id[identifier] = json.dumps(raw)
    order = [28, *range(8), 29, *range(8, 16), 30, *range(16, 24), 31, *range(24, 28)]
    current = tuple(tasks[index] for index in order)
    bank = {"tasks": [asdict(task) for task in current], "family_by_task": families}
    (root / "bank.json").write_text(json.dumps(bank))
    config["bank_record_uri"] = str(root / "bank.json")
    config["bank_record_sha256"] = hashlib.sha256((root / "bank.json").read_bytes()).hexdigest()
    config["prior_bank_record_uri"], config["prior_bank_record_sha256"] = pin(
        "prior28", {"tasks": [asdict(task) for task in tasks[:28]], "family_by_task": old_families}
    )
    write_task_records(str(root / "train.parquet"), (raw_by_id[task.task_id] for task in current))
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("bank.json", "train.parquet")}
    manifest = {"status": "complete", "stage": "admit", "files": files}
    (root / "repair-manifest.json").write_text(json.dumps(manifest))
    config["bank"] = {
        "name": "current-bank",
        "version": "2026.10.05.10",
        "uri": str(root),
        "identity_config": {
            "bank_sha256": files["bank.json"],
            "train_sha256": files["train.parquet"],
            "manifest_sha256": hashlib.sha256((root / "repair-manifest.json").read_bytes()).hexdigest(),
        },
    }
    qualification = json.loads(Path(config["qualification_uri"]).read_text())
    config["lineage_review_uri"], config["lineage_review_sha256"] = pin(
        "lineage",
        {
            "current_pair": {"model_identity": qualification["model_identity"], "bank": {"sha256": files["bank.json"]}},
            "prior_incumbent_28_bank": {
                "bank_artifact": {"identity_config": {"bank_sha256": config["prior_bank_record_sha256"]}}
            },
        },
    )
    config["feedback_release_uri"], config["feedback_release_sha256"] = pin(
        "feedback", {"skills": [{"label": "types", "description": "Type checking."}]}
    )
    config["feedback_identity"] = config["feedback_release_sha256"]
    config["targeted_task_ids"] = ["28", "29", "30", "31"]
    return config, current, raw_by_id


def sealed_calibration(config, tasks, tmp_path, rewards):
    outputs = incumbent_bank_workflow(config, "calibrate")
    bound = outputs["decision"].build_config(
        StepContext.for_run(str(tmp_path / "decision"), str(tmp_path / "artifacts"), deps=outputs["decision"].deps)
    )
    summary = {
        "model_identity": bound.plan.current_checkpoint,
        "tasks_identity": bound.plan.bank_identity,
        "count": 32,
        "samples_per_task": 8,
        "task_rewards": {task.task_id: list(rewards) for task in tasks},
    }
    Path(bound.summary_path).mkdir(parents=True)
    path = Path(bound.summary_path) / "failure_summary.json"
    path.write_text(json.dumps(summary))
    seal_incumbent_calibration(bound)
    decision = tmp_path / "decision/calibration-decision.json"
    config.update(
        calibration_decision_uri=str(decision),
        calibration_decision_sha256=hashlib.sha256(decision.read_bytes()).hexdigest(),
        calibration_summary_uri=str(path),
    )
    return outputs, bound, json.loads(decision.read_text())


def test_current_bank_trial_retains_raw_rows_and_explicit_targets(incumbent_inputs, tmp_path):
    config, tasks, raw_by_id = incumbent_inputs
    calibration, bound, decision = sealed_calibration(config, tasks, tmp_path, [0.0, 1.0] * 4)
    cal = calibration["calibration"].build_config(
        StepContext.for_run(
            str(tmp_path / "calibration"), str(tmp_path / "artifacts"), deps=calibration["calibration"].deps
        )
    )
    assert (cal.limit, cal.samples_per_task, cal.startup_attempts, cal.temperature) == (32, 8, 3, 1.0)
    assert cal.model_identity == bound.plan.current_checkpoint and cal.tasks_identity == bound.plan.bank_identity
    assert calibration["calibration"].run.fn is run_calibration_evaluation
    assert calibration["calibration"].run.env_vars["PYTHONPATH"].startswith("/app/lib/rolloutengine/src:")
    assert bound.plan.task_bank == tasks
    assert (bound.plan.retained_count, bound.plan.fresh_count, bound.plan.max_glm_responses) == (28, 4, 0)
    schedule = decision["schedule"]
    assert decision["targeted_task_ids"] == ["28", "29", "30", "31"]
    assert schedule["sampling_spec"]["targeted_task_ids"] == decision["targeted_task_ids"]
    assert schedule["experiment_limits"] == {
        "runs": 1,
        "updates": 4,
        "groups": 64,
        "rollouts": 256,
        "additional_seeds": 0,
    }
    training = incumbent_bank_workflow(config, "train")
    graph = graph_handles([training["terminal"]])
    assert calibration["calibration"] not in graph
    data = next(step for step in graph if step.name == f"documents/russell-rsi-{PROTOCOL}-replay")
    replay_config = data.build_config(
        StepContext.for_run(str(tmp_path / "replay"), str(tmp_path / "artifacts"), deps=data.deps)
    )
    freeze_replay_dataset(replay_config)
    rows = list(read_task_records(str(tmp_path / "replay/train.parquet")))
    assert rows == [raw_by_id[entry["task_id"]] for entry in schedule["schedule"]]
    assert len(rows) == 64
    model = next(step for step in graph if step.name == f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf")
    assert model in training["rl"].deps
    assert training["rl"] in training["updates"].deps
    assert training["rl"] in training["reload"].deps and training["updates"] in training["reload"].deps
    evaluation = incumbent_bank_workflow(config, "evaluate")
    for name in ("coding", "retention"):
        ancestors = graph_handles([evaluation[name]])
        identities = {artifact_identity(step) for step in ancestors}
        assert {artifact_identity(training[key]) for key in ("rl", "updates", "reload")} <= identities


def test_incumbent_cli_preflight_accepts_matching_runtime_and_version(incumbent_inputs, tmp_path):
    config, _, _ = incumbent_inputs
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(
        main,
        [
            "--config-uri",
            str(path),
            "--config-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--stage",
            "calibrate",
            "--version",
            config["version"],
        ],
    )
    assert result.exit_code == 0, str(result.exception) + result.output
    assert f"documents/russell-rsi-{PROTOCOL}-calibration-decision@{config['version']}" in result.output


def test_current_bank_rejects_changed_retained_records_and_positional_targets(incumbent_inputs):
    config, tasks, _ = incumbent_inputs
    config["targeted_task_ids"] = [task.task_id for task in tasks[28:]]
    with pytest.raises(ValueError, match="four independent additions"):
        current_bank_tasks(config)
    config["targeted_task_ids"] = ["28", "29", "30", "31"]
    prior = json.loads(Path(config["prior_bank_record_uri"]).read_text())
    prior["tasks"][0]["capability"] = "changed"
    Path(config["prior_bank_record_uri"]).write_text(json.dumps(prior))
    config["prior_bank_record_sha256"] = hashlib.sha256(Path(config["prior_bank_record_uri"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="retain all 28"):
        current_bank_tasks(config)


def test_current_bank_missing_grade_cannot_seal_decision(incumbent_inputs, tmp_path):
    config, tasks, _ = incumbent_inputs
    outputs = incumbent_bank_workflow(config, "calibrate")
    bound = outputs["decision"].build_config(
        StepContext.for_run(str(tmp_path / "decision"), str(tmp_path / "artifacts"), deps=outputs["decision"].deps)
    )
    summary = {
        "model_identity": bound.plan.current_checkpoint,
        "tasks_identity": bound.plan.bank_identity,
        "count": 32,
        "samples_per_task": 8,
        "task_rewards": {task.task_id: [0.0, 1.0] * 4 for task in tasks},
    }
    summary["task_rewards"][tasks[0].task_id].pop()
    Path(bound.summary_path).mkdir(parents=True)
    (Path(bound.summary_path) / "failure_summary.json").write_text(json.dumps(summary))
    with pytest.raises(IncompleteCalibrationError):
        seal_incumbent_calibration(bound)
    assert not (tmp_path / "decision/calibration-decision.json").exists()


def test_current_bank_negative_gate_has_no_training_nodes(incumbent_inputs, tmp_path):
    config, tasks, _ = incumbent_inputs
    _, _, decision = sealed_calibration(config, tasks, tmp_path, [0.0] * 8)
    assert decision["signal_gate_passed"] is False
    assert decision["reason"] == "no_calibration_reward_variation"
    outputs = incumbent_bank_workflow(config, "train")
    assert set(outputs) == {"terminal"}
    assert not any(step.name.startswith("checkpoints/") for step in graph_handles([outputs["terminal"]]))


@pytest.mark.parametrize(
    "scores,retention,promoted",
    [((26 / 32, 26 / 32), 1 / 3, False), ((26 / 32, 27 / 32), 0.0, False), ((26 / 32, 27 / 32), 1 / 3, True)],
)
def test_current_trial_promotion_retains_incumbent_after_regression(tmp_path, scores, retention, promoted):
    coding = tmp_path / "coding"
    retained = tmp_path / "retention"
    coding.mkdir()
    retained.mkdir()
    (coding / "coding-evidence.json").write_text(
        json.dumps(
            {
                "model_identity": "candidate",
                "panel_sha256": "panel",
                "scores": dict(zip(("humanevalplus", "mbppplus"), scores, strict=True)),
            }
        )
    )
    (retained / "failure_summary.json").write_text(
        json.dumps(
            {
                "model_identity": "candidate",
                "tasks_identity": "retention",
                "count": 3,
                "task_rewards": {"a": [float(retention > 0)], "b": [0.0], "c": [0.0]},
            }
        )
    )
    config = ContinuationSelectionConfig(
        str(coding),
        str(retained),
        "candidate",
        "panel",
        "retention",
        ("a", "b", "c"),
        CheckpointScore("incumbent", (25 / 32, 27 / 32), 1 / 3),
        CheckpointScore("parent", (25 / 32, 25 / 32), 1 / 3),
        str(tmp_path / "selection"),
    )
    seal_incumbent_selection(config)
    result = json.loads((tmp_path / "selection/continuation-selection.json").read_text())
    assert result["protocol"] == PROTOCOL
    assert result["selected"]["checkpoint_identity"] == ("candidate" if promoted else "incumbent")
