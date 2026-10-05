# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
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
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    CalibrationRecordConfig,
    SelectionConfig,
    post_sft_plan,
    post_sft_schedule,
    post_sft_workflow,
    qualified_sft,
    seal_calibration,
    seal_selection,
)
from experiments.post_training.russell_rsi.replay import sampled_replay_plan
from experiments.post_training.russell_rsi.sources import compact_json_sha256


@pytest.fixture
def replay_inputs():
    tasks = tuple(
        QualifiedTask(str(i), f"hash-{i}", f"admission-{i}", f"source-{i}", "types", f"family-{i}") for i in range(26)
    )
    plan = RoundPlan(
        "old-pilot",
        "parent",
        "parent",
        tasks,
        "bank",
        "old-calibration",
        (),
        tasks,
        26,
        0,
        (),
        "coding",
        "retention",
        "feedback",
        "runtime",
        9528,
        4,
    )
    source = sampled_replay_plan(
        plan,
        tuple(Measurement("parent", task.task_sha256, (0.0, 1.0) * 4) for task in tasks),
        tasks[-2:],
        pilot_number=2,
        bank_identity="bank",
        calibration_identity="old-calibration",
        frozen_identity="old-frozen",
        parent_identity="parent",
        model_identity="parent",
        family_by_task={task.task_id: task.contract_id for task in tasks},
        updates=4,
        seed=9528,
    )
    fresh = post_sft_plan(plan, model="sft", calibration="sft-calibration")
    summary = {
        "model_identity": "sft",
        "tasks_identity": "bank",
        "count": 26,
        "samples_per_task": 8,
        "task_rewards": {task.task_id: [0.0, 1.0] * 4 for task in tasks},
    }
    return fresh, source, summary


def test_post_sft_schedule_preserves_sampler_draws_with_independent_trial_identity(replay_inputs, tmp_path):
    plan, source, summary = replay_inputs
    schedule = post_sft_schedule(summary, plan, source)["schedule"]
    assert [(row["task_id"], row["category"]) for row in schedule["schedule"]] == [
        (row["task_id"], row["category"]) for row in source["schedule"]
    ]
    assert not {row["occurrence_id"] for row in schedule["schedule"]} & {
        row["occurrence_id"] for row in source["schedule"]
    }
    assert len({row["occurrence_id"] for row in schedule["schedule"]}) == 64
    assert schedule["schedule_sha256"] == compact_json_sha256(
        {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    )
    summary_path = tmp_path / "calibration"
    summary_path.mkdir()
    raw = json.dumps(summary).encode()
    (summary_path / "failure_summary.json").write_bytes(raw)
    seal_calibration(
        CalibrationRecordConfig(str(summary_path), plan, source, "qualification", str(tmp_path / "decision"))
    )
    record = json.loads((tmp_path / "decision/calibration-decision.json").read_text())
    assert record["schedule"] == schedule
    assert record["summary_sha256"] == hashlib.sha256(raw).hexdigest()


def test_incomplete_calibration_does_not_become_a_signal_failure(replay_inputs):
    plan, source, summary = replay_inputs
    summary["task_rewards"]["0"].pop()
    with pytest.raises(IncompleteCalibrationError):
        post_sft_schedule(summary, plan, source)


@pytest.mark.parametrize(
    "reward_map,reason",
    [
        ({"default": [0.0] * 8}, "no_calibration_reward_variation"),
        ({"default": [0.5] * 8}, "calibration_rewards_not_binary"),
        ({"default": [0.0] * 8, "0": [0.0] * 7 + [1.0]}, "weighted_q4_below_threshold"),
    ],
)
def test_complete_calibration_signal_failures_block_replay(replay_inputs, reward_map, reason):
    plan, source, summary = replay_inputs
    summary["task_rewards"] = {
        task.task_id: list(reward_map.get(task.task_id, reward_map["default"])) for task in plan.task_bank
    }
    decision = post_sft_schedule(summary, plan, source)
    assert decision == {"protocol": "teacher-sft-r1", "signal_gate_passed": False, "reason": reason, "schedule": None}


def test_qualification_matches_export_shard_inventory_to_index():
    record = {
        "protocol": "teacher-sft-one-update-qualification-v1",
        "sft_identity": "sft",
        "sft_root": "/saved",
        "hf_export_uri": "/saved/hf/step-0",
        "optimizer_updates": 1,
        "learning_rate": 1e-6,
        "loss": 2.0,
        "gradient_norm": 0.5,
        "update_norm": 0.01,
        "hf_files": [{"path": "model.safetensors.index.json", "sha256": "a" * 64}],
        "hf_shards": [{"path": "model-01.safetensors", "size": 134000000000}],
        "hf_weight_map": {"parameter": "model-01.safetensors"},
        "hf_verified": dict.fromkeys(("shards", "config", "tokenizer", "eos"), True),
        "serving_reload": {
            "verified": True,
            "suite": "mmlu-smoke",
            "limit": 1,
            "model_uri": "/saved/hf/step-0",
            "evidence_uri": "/reload",
            "evidence_sha256": "b" * 64,
        },
    }
    assert qualified_sft(record, identity="sft", root="/saved") == "/saved/hf/step-0"
    record["hf_weight_map"]["parameter"] = "missing.safetensors"
    with pytest.raises(ValueError, match="shard inventory"):
        qualified_sft(record, identity="sft", root="/saved")


@pytest.mark.parametrize(
    "sft_coding,rl_coding,rl_retention,selected,promoted",
    [
        ((0.5, 0.5), (0.6, 0.5), 1.0, "rl", "rl"),
        ((0.5, 0.5), (0.5, 0.5), 1.0, "sft", "parent"),
        ((0.5, 0.5), (0.6, 0.4), 1.0, "sft", "parent"),
        ((0.5, 0.5), (0.6, 0.5), 2 / 3, "sft", "parent"),
        ((0.6, 0.5), None, 1.0, "sft", "sft"),
        ((0.6, 0.5), (0.6, 0.5), 1.0, "sft", "sft"),
    ],
)
def test_selection_requires_strict_coding_gain_and_retention_before_parent_gate(
    tmp_path, sft_coding, rl_coding, rl_retention, selected, promoted
):
    coding_paths, retention_paths = [], []
    models = [("sft", sft_coding, 1.0)]
    if rl_coding is not None:
        models.append(("rl", rl_coding, rl_retention))
    for identity, coding, retained in models:
        path = tmp_path / identity
        path.mkdir()
        (path / "coding-evidence.json").write_text(
            json.dumps(
                {
                    "model_identity": identity,
                    "panel_sha256": "panel",
                    "scores": dict(zip(("humanevalplus", "mbppplus"), coding, strict=True)),
                }
            )
        )
        (path / "failure_summary.json").write_text(
            json.dumps(
                {
                    "model_identity": identity,
                    "tasks_identity": "retention",
                    "count": 3,
                    "task_rewards": {str(i): [int(i < round(retained * 3))] for i in range(3)},
                }
            )
        )
        coding_paths.append(str(path))
        retention_paths.append(str(path))
    seal_selection(
        SelectionConfig(
            tuple(coding_paths),
            tuple(retention_paths),
            tuple(identity for identity, _, _ in models),
            "panel",
            "retention",
            ("0", "1", "2"),
            CheckpointScore("parent", (0.5, 0.5), 1.0),
            str(tmp_path / "selected"),
        )
    )
    result = json.loads((tmp_path / "selected/post-sft-selection.json").read_text())
    assert result["selected"]["checkpoint_identity"] == selected
    assert result["promoted"]["checkpoint_identity"] == promoted


@pytest.mark.parametrize("signal", [True, False])
def test_evaluation_graph_waits_for_trial_export_only_when_signal_passes(tmp_path, replay_inputs, signal):
    def pin(name, value):
        path = tmp_path / f"{name}.json"
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return str(path), hashlib.sha256(raw).hexdigest()

    def artifact(name):
        return {"name": name, "version": "2026.10.05.9", "uri": str(tmp_path / name), "identity_config": {}}

    config = {
        "version": "2026.10.05.9",
        "sft": artifact("sft"),
        "parent": artifact("parent"),
        "bank": artifact("bank"),
        "retention": artifact("retention"),
        "machine_config": {},
        "runtime_bundle": {
            "manifest_uri": "s3://test/manifest",
            "manifest_sha256": "a" * 64,
            "archive_uri": "s3://test/archive",
            "archive_sha256": "b" * 64,
        },
    }
    plan, source, _ = replay_inputs
    config["bank_record_uri"], config["bank_record_sha256"] = pin(
        "bank", {"tasks": [asdict(task) for task in plan.task_bank]}
    )
    config["bank"]["identity_config"] = {"bank_sha256": config["bank_record_sha256"]}
    parent = ArtifactStep.adopt("parent", config["version"], config["parent"]["uri"], kind=LevanterCheckpoint, config={})
    bank = ArtifactStep.adopt(
        "bank", config["version"], config["bank"]["uri"], kind=Artifact, config=config["bank"]["identity_config"]
    )
    retained = ArtifactStep.adopt("retention", config["version"], config["retention"]["uri"], config={})
    sft = ArtifactStep.adopt("sft", config["version"], config["sft"]["uri"], kind=LevanterCheckpoint, config={})
    root = config["sft"]["uri"]
    qualification = {
        "protocol": "teacher-sft-one-update-qualification-v1",
        "sft_identity": artifact_identity(sft),
        "sft_root": root,
        "hf_export_uri": root + "/hf/step-0",
        "optimizer_updates": 1,
        "learning_rate": 1e-6,
        "loss": 2.0,
        "gradient_norm": 0.5,
        "update_norm": 0.01,
        "hf_files": [{"path": "config.json", "sha256": "a" * 64}],
        "hf_shards": [{"path": "model.safetensors", "size": 100}],
        "hf_weight_map": {"weight": "model.safetensors"},
        "hf_verified": dict.fromkeys(("shards", "config", "tokenizer", "eos"), True),
        "serving_reload": {
            "verified": True,
            "suite": "mmlu-smoke",
            "limit": 1,
            "model_uri": root + "/hf/step-0",
            "evidence_uri": "/reload",
            "evidence_sha256": "b" * 64,
        },
    }
    config["sft_config_uri"], config["sft_config_sha256"] = pin("sft-config", config)
    qualification["source_config_sha256"] = config["sft_config_sha256"]
    config["qualification_uri"], config["qualification_sha256"] = pin("qualification", qualification)
    panel = {
        "items": [
            {"suite": suite, "benchmark_id": str(i), "prompt_sha256": "c" * 64}
            for suite in ("humanevalplus", "mbppplus")
            for i in range(32)
        ],
        "protocols": {"humanevalplus": "human-protocol", "mbppplus": "mbpp-protocol"},
    }
    config["panel_uri"], config["panel_sha256"] = pin("panel", panel)
    config["source_config_uri"], config["source_config_sha256"] = pin("source-config", config)
    plan, source, _ = replay_inputs
    old_plan = replace(
        plan,
        current_checkpoint=artifact_identity(parent),
        champion_checkpoint=artifact_identity(parent),
        bank_identity=artifact_identity(bank),
        retention_identity=artifact_identity(retained),
        calibration_identity="old-calibration",
    )
    source = sampled_replay_plan(
        old_plan,
        tuple(Measurement(artifact_identity(parent), task.task_sha256, (0.0, 1.0) * 4) for task in plan.task_bank),
        plan.task_bank[-2:],
        pilot_number=2,
        bank_identity=artifact_identity(bank),
        calibration_identity="old-calibration",
        frozen_identity="old-data",
        parent_identity=artifact_identity(parent),
        model_identity=artifact_identity(parent),
        family_by_task=source["family_by_task"],
        updates=4,
        seed=9528,
    )
    baseline = CheckpointScore(artifact_identity(parent), (0.5, 0.5), 1.0)
    state = LoopState(baseline, baseline, baseline, old_plan.task_bank)
    seal_round(
        StoragePath(str(tmp_path / "round")),
        state,
        old_plan,
        RoundResult(baseline, "reload", "feedback", 4),
        "previous",
        source,
    )
    config["source_round_uri"] = str(tmp_path / "round" / f"{old_plan.name}.json")
    config["source_round_sha256"] = hashlib.sha256(Path(config["source_round_uri"]).read_bytes()).hexdigest()
    config["source_replay_uri"], config["source_replay_sha256"] = pin("replay", source)
    config["bank_record_uri"], config["bank_record_sha256"] = pin(
        "bank", {"tasks": [asdict(task) for task in old_plan.task_bank]}
    )
    calibration = post_sft_workflow(config, "calibrate")
    calibration_config = calibration["decision"].build_config
    bound = calibration_config(
        StepContext.for_run(
            str(tmp_path / "decision"),
            str(tmp_path / "artifacts"),
            deps=calibration["decision"].deps,
        )
    )
    summary = {
        "model_identity": bound.plan.current_checkpoint,
        "tasks_identity": artifact_identity(bank),
        "count": 26,
        "samples_per_task": 8,
        "task_rewards": {task.task_id: [0.0, 1.0] * 4 if signal else [0.0] * 8 for task in plan.task_bank},
    }
    Path(bound.summary_path).mkdir(parents=True)
    (Path(bound.summary_path) / "failure_summary.json").write_text(json.dumps(summary))
    seal_calibration(bound)
    config["calibration_decision_uri"] = str(tmp_path / "decision/calibration-decision.json")
    config["calibration_decision_sha256"] = hashlib.sha256(
        Path(config["calibration_decision_uri"]).read_bytes()
    ).hexdigest()
    config["calibration_summary_uri"] = str(Path(bound.summary_path) / "failure_summary.json")
    config["parent_coding_uri"], config["parent_coding_sha256"] = pin(
        "parent-coding",
        {
            "model_identity": artifact_identity(parent),
            "panel_sha256": compact_json_sha256(panel),
            "scores": {"humanevalplus": 0.5, "mbppplus": 0.5},
        },
    )
    config["parent_retention_uri"], config["parent_retention_sha256"] = pin(
        "parent-retention",
        {
            "model_identity": artifact_identity(parent),
            "tasks_identity": artifact_identity(retained),
            "count": 3,
            "task_rewards": {str(i): [1.0] for i in range(3)},
        },
    )
    outputs = post_sft_workflow(config, "evaluate")
    if signal:
        for key in ("coding-sft", "coding-sft-rl", "retention-sft", "retention-sft-rl"):
            ancestors = graph_handles([outputs[key]])
            assert outputs["rl"] in ancestors
            assert outputs["reload"] in ancestors
    else:
        assert "rl" not in outputs
        assert "coding-sft-rl" not in outputs
        assert all("teacher-sft-r1-pilot" not in item.name for item in graph_handles([outputs["selection"]]))
