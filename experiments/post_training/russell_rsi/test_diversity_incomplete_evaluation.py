# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.execution.remote import RemoteCallable
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL

from experiments.post_training.russell_rsi import test_teacher_diversity_study as fixture_source
from experiments.post_training.russell_rsi.diversity_incomplete_evaluation import (
    PROTOCOL,
    incomplete_diversity_evaluation,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import (
    durable_diversity_post_workflow,
    validated_durable_diversity_training,
)
from experiments.post_training.russell_rsi.teacher_diversity_study import validated_diversity_post_inputs

pinned = fixture_source.pinned
diversity_inputs = fixture_source.diversity_inputs
durable_post_inputs = fixture_source.durable_post_inputs
study_inputs = fixture_source.study_inputs
continuation_inputs = fixture_source.continuation_inputs


@pytest.fixture
def incomplete_inputs(durable_post_inputs, tmp_path):
    original = durable_post_inputs[0]
    stages = durable_diversity_post_workflow(original, "calibrate")
    prefix = str(tmp_path / "calibration-artifacts")
    pins = {}
    for name, status in (("calibration", "SUCCESS"), ("decision", "FAILED")):
        step = stages[name]
        root = Path(step.path(prefix))
        root.mkdir(parents=True)
        ctx = StepContext.for_run(str(root), prefix, deps=step.deps, runtime_args=step.runtime_args)
        config = json.loads(canonical_json(step.build_config(ctx)))
        pins[name] = {
            "producer": pinned(
                tmp_path,
                name + "-producer",
                {
                    "name": step.name,
                    "version": step.version,
                    "fingerprint": step.fingerprint(),
                    "output_path": str(root),
                    "config": config,
                    "deps": [f"{d.name}@{d.version}" for d in step.deps],
                },
            ),
        }
        Path(pins[name]["producer"]["uri"]).rename(root / ".artifact.json")
        pins[name]["producer"]["uri"] = str(root / ".artifact.json")
        status_path = root / ".executor_status"
        status_path.write_text(status)
        pins[name]["status"] = {"uri": str(status_path), "sha256": hashlib.sha256(status.encode()).hexdigest()}
    decision = stages["decision"]
    decision_root = Path(decision.path(prefix))
    info = {
        "name": decision.name,
        "output_path": str(decision_root),
        "config": {
            "fingerprint": decision.fingerprint(),
            "version": decision.version,
            "deps": [f"{dep.name}@{dep.version}" for dep in decision.deps],
        },
        "dependencies": [replace(dep.lower(), output_path_prefix=prefix).output_path for dep in decision.deps],
    }
    info_pin = pinned(tmp_path, "failed-decision-info", info)
    Path(info_pin["uri"]).rename(decision_root / ".executor_info")
    info_pin["uri"] = str(decision_root / ".executor_info")
    pins["decision"] = {"executor_info": info_pin, "status": pins["decision"]["status"]}
    calibration = stages["calibration"]
    root = Path(calibration.path(prefix))
    context = StepContext.for_run(str(root), prefix, deps=calibration.deps, runtime_args=calibration.runtime_args)
    evaluation = calibration.build_config(context)
    tasks = validated_diversity_post_inputs(
        original, trained=validated_durable_diversity_training(original)
    ).source_plan.task_bank
    rewards = {task.task_id: [0.0] * (7 if i < 3 else 8) for i, task in enumerate(tasks)}
    summary = {
        "model_identity": evaluation.model_identity,
        "tasks_identity": evaluation.tasks_identity,
        "count": len(tasks),
        "samples_per_task": 8,
        "task_rewards": rewards,
    }
    summary_pin = pinned(tmp_path, "summary", summary)
    Path(summary_pin["uri"]).rename(root / "failure_summary.json")
    summary_pin["uri"] = str(root / "failure_summary.json")
    journal = {"config": asdict(evaluation), "attempts": {"task": {f"{task.task_id}/0": "a" * 64 for task in tasks}}}
    journal_pin = pinned(tmp_path, "journal", journal)
    (root / "journal").mkdir()
    Path(journal_pin["uri"]).rename(root / "journal/binding.json")
    journal_pin["uri"] = str(root / "journal/binding.json")
    missing = []
    for index, task in enumerate(tasks[:3]):
        key = f"{task.task_id}/0"
        envelope = {
            "binding": {"evaluation": journal, "key": key, "kind": "task", "task_sha256": "a" * 64},
            "result": {"record": {"task_id": task.task_id, "grade": {"status": "unavailable", "reward": None}}},
        }
        result = pinned(tmp_path, "slot-" + str(index), envelope)
        dest = root / "journal/task" / key / "result.json"
        dest.parent.mkdir(parents=True)
        Path(result["uri"]).rename(dest)
        result["uri"] = str(dest)
        missing.append({"task_id": task.task_id, "sample": 0, "result": result, "submission_present": index == 0})
    original_pin = pinned(tmp_path, "original", original)
    amendment = {
        "protocol": PROTOCOL,
        "original_config": original_pin,
        "runtime_commit": MARIN_SKYRL.commit,
        "model_identity": evaluation.model_identity,
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
        "repeated_issued_samples": 0,
        "calibration": {**pins["calibration"], "summary": summary_pin, "journal_binding": journal_pin},
        "failed_decision": pins["decision"],
        "foreground": pinned(
            tmp_path,
            "foreground",
            {
                "exit_code": 1,
                "error_type": "IncompleteCalibrationError",
                "original_config": original_pin,
                "decision_identity": artifact_identity(stages["decision"]),
                "decision_output_path": stages["decision"].path(prefix),
            },
        ),
        "missing_slots": missing,
        "evaluation": {"version": "2026.10.06.19", "conditions": ["sft"], "coding_limit": 32, "retention_limit": 3},
    }
    return (
        {
            "original_config": original_pin,
            "amendment": pinned(tmp_path, "amendment", amendment),
            "version": "2026.10.06.19",
        },
        amendment,
        original,
    )


def test_incomplete_calibration_evaluates_only_qualified_sft_on_normal_remote_route(incomplete_inputs):
    config, amendment, _ = incomplete_inputs
    outputs = incomplete_diversity_evaluation(config)
    handles = list(graph_handles([outputs["terminal"]]))
    names = [step.name for step in handles]
    assert not any("calibration" in name or "replay" in name or "sft-rl" in name for name in names)
    coding = next(step for step in handles if step.name.startswith("evals/") and "retention" not in step.name)
    assert coding.run.__module__ == "experiments.evaluation.pipeline"
    assert coding.runtime_args["submission_cluster"] == "cw-us-east-02a"
    assert coding.runtime_args["federated_cluster"] == "cw-us-east-02a"
    retention = outputs["retention-sft"]
    assert isinstance(retention.run, RemoteCallable)
    assert retention.run.fn.__name__ == "run_development_evaluation"
    assert any(artifact_identity(step) == amendment["model_identity"] for step in handles)


def test_rebound_calibration_or_missing_slot_cannot_authorize_evaluation(incomplete_inputs, tmp_path):
    config, amendment, _ = incomplete_inputs
    changed = deepcopy(amendment)
    changed["missing_slots"][0]["sample"] = 7
    config["amendment"] = pinned(tmp_path, "wrong-slot", changed)
    with pytest.raises(ValueError, match="outside the original calibration journal"):
        incomplete_diversity_evaluation(config)
    changed = deepcopy(amendment)
    changed["model_identity"] = "another-checkpoint"
    config["amendment"] = pinned(tmp_path, "wrong-model", changed)
    with pytest.raises(ValueError, match="source or no-RL"):
        incomplete_diversity_evaluation(config)

    changed = deepcopy(amendment)
    producer = json.loads(Path(changed["calibration"]["producer"]["uri"]).read_bytes())
    producer["fingerprint"] = "another-producer"
    changed["calibration"]["producer"] = pinned(tmp_path, "wrong-producer", producer)
    config["amendment"] = pinned(tmp_path, "wrong-producer-amendment", changed)
    with pytest.raises(ValueError, match="different terminal producer"):
        incomplete_diversity_evaluation(config)


def test_completed_calibration_cannot_use_the_incomplete_sft_only_path(incomplete_inputs, tmp_path):
    config, amendment, _ = incomplete_inputs
    changed = deepcopy(amendment)
    pin = changed["calibration"]["summary"]
    path = Path(pin["uri"])
    summary = json.loads(path.read_bytes())
    summary["task_rewards"] = {
        key: [*rewards, *([0.0] * (8 - len(rewards)))] for key, rewards in summary["task_rewards"].items()
    }
    raw = json.dumps(summary, sort_keys=True).encode()
    path.write_bytes(raw)
    pin["sha256"] = hashlib.sha256(raw).hexdigest()
    config["amendment"] = pinned(tmp_path, "complete-calibration", changed)
    with pytest.raises(ValueError, match="requires actual incomplete calibration"):
        incomplete_diversity_evaluation(config)
