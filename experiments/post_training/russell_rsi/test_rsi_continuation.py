# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    IncompleteCalibrationError,
    LoopState,
    Measurement,
    QualifiedTask,
    RoundPlan,
    RoundResult,
    seal_round,
)
from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION, adopted
from experiments.post_training.russell_rsi.launch_rsi_continuation import (
    ContinuationSelectionConfig,
    continuation_schedule,
    continuation_workflow,
    expanded_bank,
    qualified_champion,
    seal_continuation_calibration,
    seal_continuation_selection,
)
from experiments.post_training.russell_rsi.replay import sampled_replay_plan
from experiments.post_training.russell_rsi.sources import compact_json_sha256


@pytest.fixture
def continuation_inputs(tmp_path):
    def pin(name, value):
        path = tmp_path / f"{name}.json"
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return str(path), hashlib.sha256(raw).hexdigest()

    def artifact(name):
        return {"name": name, "version": "2026.10.05.9", "uri": str(tmp_path / name), "identity_config": {}}

    tasks = tuple(
        QualifiedTask(str(i), f"hash-{i}", f"admission-{i}", f"source-{i}", "types", f"family-{i}") for i in range(28)
    )
    families = {task.task_id: task.contract_id for task in tasks}
    tasks = (
        *tasks[:20],
        *(replace(task, relation="variant") for task in tasks[20:26]),
        *(replace(task, relation="new_contract") for task in tasks[26:]),
    )
    families.update({str(i): f"family-{i - 20}" for i in range(20, 26)})
    config = {
        "version": "2026.10.05.9",
        "parent": artifact("parent"),
        "retention": artifact("retention"),
        "machine_config": {},
        "runtime_bundle": {
            "manifest_uri": "s3://test/manifest",
            "manifest_sha256": "a" * 64,
            "archive_uri": "s3://test/archive",
            "archive_sha256": "b" * 64,
        },
    }
    config["old_bank_record_uri"], config["old_bank_record_sha256"] = pin(
        "old-bank", {"tasks": [asdict(t) for t in tasks[:26]]}
    )
    old_bank = artifact("old-bank")
    old_bank["identity_config"] = {"bank_sha256": config["old_bank_record_sha256"]}
    config["bank_record_uri"], config["bank_record_sha256"] = pin(
        "new-bank", {"tasks": [asdict(t) for t in tasks], "family_by_task": families}
    )
    config["bank"] = artifact("new-bank")
    config["bank"]["identity_config"] = {"bank_sha256": config["bank_record_sha256"]}
    parent = adopted(config["parent"], LevanterCheckpoint)
    retained = adopted(config["retention"])
    plan = RoundPlan(
        "source-pilot",
        artifact_identity(parent),
        artifact_identity(parent),
        tasks[:26],
        artifact_identity(adopted(old_bank)),
        "old-calibration",
        (),
        tasks[:26],
        26,
        0,
        (),
        "coding",
        artifact_identity(retained),
        "feedback",
        "runtime",
        9528,
        4,
    )
    replay = sampled_replay_plan(
        plan,
        tuple(Measurement(plan.current_checkpoint, t.task_sha256, (0.0, 1.0) * 4) for t in tasks[:26]),
        tasks[18:20],
        pilot_number=2,
        bank_identity=plan.bank_identity,
        calibration_identity=plan.calibration_identity,
        frozen_identity="old-data",
        parent_identity=plan.current_checkpoint,
        model_identity=plan.current_checkpoint,
        family_by_task={t.task_id: families[t.task_id] for t in tasks[:26]},
        updates=4,
        seed=9528,
    )
    baseline = CheckpointScore(plan.current_checkpoint, (25 / 32, 25 / 32), 1 / 3)
    seal_round(
        StoragePath(str(tmp_path / "round")),
        LoopState(baseline, baseline, baseline, tasks[:26]),
        plan,
        RoundResult(baseline, "reload", "feedback", 4),
        "previous",
        replay,
    )
    config["source_round_uri"] = str(tmp_path / "round" / f"{plan.name}.json")
    config["source_round_sha256"] = hashlib.sha256(Path(config["source_round_uri"]).read_bytes()).hexdigest()
    config["source_replay_uri"], config["source_replay_sha256"] = pin("replay", replay)
    panel = {
        "items": [
            {"suite": suite, "benchmark_id": str(i), "prompt_sha256": "c" * 64}
            for suite in ("humanevalplus", "mbppplus")
            for i in range(32)
        ],
        "protocols": {"humanevalplus": "human-protocol", "mbppplus": "mbpp-protocol"},
    }
    config["panel_uri"], config["panel_sha256"] = pin("panel", panel)
    config["source_config_uri"], config["source_config_sha256"] = pin("source-config", {**config, "bank": old_bank})
    source: dict = {
        "name": "checkpoints/champion",
        "version": "2026.10.05.4",
        "fingerprint": "original8",
        "result_type": "marin.rl.skyrl.SkyRLRun",
        "output_path": str(tmp_path / "champion"),
    }
    config["incumbent_artifact_uri"], config["incumbent_artifact_sha256"] = pin("champion", source)
    original_identity = "checkpoints/champion@2026.10.05.4:original8"
    export = source["output_path"] + "/exports/global_step_8/policy"
    source["result"] = {
        "global_step": 8,
        "hf_model_uri": export,
        "tokenizer_uri": MODEL,
        "tokenizer_revision": MODEL_REVISION,
    }
    config["incumbent_artifact_uri"], config["incumbent_artifact_sha256"] = pin("champion", source)
    qualification = {
        "source_artifact_uri": config["incumbent_artifact_uri"],
        "source_artifact_sha256": config["incumbent_artifact_sha256"],
        "protocol": "rsi-champion-eight-update-qualification-v1",
        "model_identity": original_identity,
        "model_root": source["output_path"],
        "hf_export_uri": export,
        "optimizer_updates": 8,
        "export_verified": True,
        "export_evidence_uri": "/completed-export",
        "export_evidence_sha256": "d" * 64,
        "serving_reload": {
            "verified": True,
            "model_identity": original_identity,
            "model_uri": export,
            "suite": "mmlu-smoke",
            "limit": 1,
            "evidence_uri": "/completed-reload",
            "evidence_sha256": "e" * 64,
        },
    }
    config["qualification_uri"], config["qualification_sha256"] = pin("qualification", qualification)
    for label, identity, mbpp in (
        ("parent", plan.current_checkpoint, 25 / 32),
        ("incumbent", original_identity, 27 / 32),
    ):
        config[f"{label}_coding_uri"], config[f"{label}_coding_sha256"] = pin(
            f"{label}-coding",
            {
                "model_identity": identity,
                "panel_sha256": compact_json_sha256(panel),
                "scores": {"humanevalplus": 25 / 32, "mbppplus": mbpp},
            },
        )
        config[f"{label}_retention_uri"], config[f"{label}_retention_sha256"] = pin(
            f"{label}-retention",
            {
                "model_identity": identity,
                "tasks_identity": artifact_identity(retained),
                "count": 3,
                "task_rewards": {"a": [1.0], "b": [0.0], "c": [0.0]},
            },
        )
    return config, tasks, families


@pytest.mark.parametrize("defect", ["retained_edit", "variant", "known_family"])
def test_expansion_rejects_replacement_or_existing_behavior(continuation_inputs, defect):
    _, tasks, families = continuation_inputs
    changed = list(tasks)
    if defect == "retained_edit":
        changed[0] = replace(changed[0], admission_sha256="different-admission")
    elif defect == "variant":
        changed[-1] = replace(changed[-1], relation="variant")
    else:
        changed[-1] = replace(changed[-1], contract_id=tasks[0].contract_id)
        families[changed[-1].task_id] = tasks[0].contract_id
    message = "distinct stable semantic contract IDs" if defect == "known_family" else "Continuation requires"
    with pytest.raises(ValueError, match=message):
        expanded_bank(
            {"tasks": [asdict(t) for t in changed], "family_by_task": families},
            tasks[:26],
            {t.task_id: families[t.task_id] for t in tasks[:26]},
        )


def calibrated_config(config, tasks, tmp_path, signal=True):
    outputs = continuation_workflow(config, "calibrate")
    bound = outputs["decision"].build_config(
        StepContext.for_run(str(tmp_path / "decision"), str(tmp_path / "artifacts"), deps=outputs["decision"].deps)
    )
    summary = {
        "model_identity": bound.plan.current_checkpoint,
        "tasks_identity": bound.plan.bank_identity,
        "count": len(tasks),
        "samples_per_task": 8,
        "task_rewards": {t.task_id: [0.0, 1.0] * 4 if signal else [0.0] * 8 for t in tasks},
    }
    Path(bound.summary_path).mkdir(parents=True)
    path = Path(bound.summary_path) / "failure_summary.json"
    path.write_text(json.dumps(summary))
    seal_continuation_calibration(bound)
    decision = tmp_path / "decision/calibration-decision.json"
    config.update(
        calibration_decision_uri=str(decision),
        calibration_decision_sha256=hashlib.sha256(decision.read_bytes()).hexdigest(),
        calibration_summary_uri=str(path),
    )
    return outputs, bound, summary, json.loads(decision.read_text())


def test_fresh_calibration_seals_complete_bank_and_new_family_schedule(continuation_inputs, tmp_path):
    config, tasks, families = continuation_inputs
    outputs, bound, summary, record = calibrated_config(config, tasks, tmp_path)
    schedule = record["schedule"]
    assert len(schedule["schedule"]) == 64
    assert schedule["experiment_limits"] == {
        "runs": 1,
        "updates": 4,
        "groups": 64,
        "rollouts": 256,
        "additional_seeds": 0,
    }
    assert schedule["sampling_spec"]["targeted_task_ids"] == ["26", "27"]
    assert all(row["task_id"] in ("26", "27") for row in schedule["schedule"] if row["category"] == "targeted")
    assert len({row["occurrence_id"] for row in schedule["schedule"]}) == 64
    assert all(row["occurrence_id"].startswith("champion-rsi-r1-") for row in schedule["schedule"])
    assert schedule["schedule_sha256"] == compact_json_sha256(
        {k: v for k, v in schedule.items() if k != "schedule_sha256"}
    )
    cal = outputs["calibration"].build_config(
        StepContext.for_run("/cal", "/artifacts", deps=outputs["calibration"].deps)
    )
    assert (cal.limit, cal.samples_per_task, cal.temperature, cal.startup_attempts) == (28, 8, 1.0, 3)
    assert cal.require_reward_variation is False
    summary["task_rewards"]["27"].pop()
    with pytest.raises(IncompleteCalibrationError):
        continuation_schedule(summary, bound.plan, families)


@pytest.mark.parametrize("signal", [True, False])
def test_trial_stage_uses_pinned_calibration_and_export_barriers(continuation_inputs, tmp_path, signal):
    config, tasks, _ = continuation_inputs
    calibration, _, _, _ = calibrated_config(config, tasks, tmp_path, signal)
    outputs = continuation_workflow(config, "evaluate")
    graph = graph_handles([outputs["terminal"]])
    assert calibration["calibration"] not in graph
    if not signal:
        assert set(outputs) == {"terminal"}
        assert all(not step.name.startswith("checkpoints/") for step in graph)
        return
    for key in ("coding", "retention"):
        ancestors = graph_handles([outputs[key]])
        assert outputs["rl"] in ancestors
        assert outputs["reload"] in ancestors
    training_outputs = continuation_workflow(config, "train")
    assert set(training_outputs) == {"rl", "reload", "terminal"}
    assert training_outputs["terminal"] is training_outputs["reload"]
    assert artifact_identity(training_outputs["rl"]) == artifact_identity(outputs["rl"])
    training = outputs["rl"].build_config(StepContext.for_fingerprint(deps=outputs["rl"].deps))
    launch = yaml.safe_load(training.launch_config_yaml)
    assert launch["skyrl"]["trainer"]["max_steps"] == 4
    assert launch["skyrl"]["trainer"]["resume_mode"] == "none"
    assert launch["run"]["seed"] == 9528


@pytest.mark.parametrize(
    "scores,retention,selected",
    [
        ((26 / 32, 26 / 32), 1 / 3, "incumbent"),
        ((25 / 32, 27 / 32), 1 / 3, "incumbent"),
        ((26 / 32, 27 / 32), 0.0, "incumbent"),
        ((26 / 32, 27 / 32), 1 / 3, "candidate"),
    ],
)
def test_promotion_requires_gain_against_champion_not_original_parent(tmp_path, scores, retention, selected):
    coding = tmp_path / "coding"
    coding.mkdir()
    retained = tmp_path / "retained"
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
    seal_continuation_selection(
        ContinuationSelectionConfig(
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
    )
    result = json.loads((tmp_path / "selection/continuation-selection.json").read_text())
    assert result["selected"]["checkpoint_identity"] == selected
    assert result["original_parent"]["checkpoint_identity"] == "parent"


def test_champion_reload_cannot_be_rebound_to_new_alias(continuation_inputs):
    config, _, _ = continuation_inputs
    source = json.loads(Path(config["incumbent_artifact_uri"]).read_text())
    record = json.loads(Path(config["qualification_uri"]).read_text())
    assert qualified_champion(record, source) == record["hf_export_uri"]
    record["serving_reload"]["model_identity"] = "new-hf-alias"
    with pytest.raises(ValueError):
        qualified_champion(record, source)


def test_qualification_cannot_override_incomplete_source_export(continuation_inputs):
    config, _, _ = continuation_inputs
    source = json.loads(Path(config["incumbent_artifact_uri"]).read_text())
    record = json.loads(Path(config["qualification_uri"]).read_text())
    source["result"].update(global_step=4, hf_model_uri="/incomplete-export")
    with pytest.raises(ValueError):
        qualified_champion(record, source)


def test_complete_but_sparse_calibration_stops_before_trial(continuation_inputs, tmp_path):
    config, tasks, families = continuation_inputs
    _, bound, summary, _ = calibrated_config(config, tasks, tmp_path)
    summary["task_rewards"] = {task.task_id: [0.0] * 8 for task in tasks}
    summary["task_rewards"]["26"][0] = 1.0
    result = continuation_schedule(summary, bound.plan, families)
    assert result == {
        "protocol": "champion-rsi-r1",
        "signal_gate_passed": False,
        "reason": "weighted_q4_below_threshold",
        "schedule": None,
    }
