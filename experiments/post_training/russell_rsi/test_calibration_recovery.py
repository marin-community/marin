# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
from rigging.runtime_bundle import RuntimeBundle
from taskcompendium.grading import exact_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.parquet import write_tasks

from experiments.post_training.russell_rsi.bootstrap_loop import QualifiedTask, calibration_measurements
from experiments.post_training.russell_rsi.calibration_recovery import (
    CalibrationRecoveryConfig,
    PinnedFile,
    recover_calibration,
    replay_grade,
)
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256


def pin(path, value):
    path.write_text(json.dumps(value) + "\n")
    return PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def recovery(tmp_path):
    tasks = [
        TaskSpec(
            id=f"task-{index}",
            context=ConversationInput(events=(TextMessage(role="user", content=f"Return {index}."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer(str(index)),
            source=Source(dataset="source", revision="pin", row=str(index), importer_revision="1"),
        )
        for index in range(18)
    ]
    parquet = tmp_path / "train.parquet"
    write_tasks(str(parquet), tasks)
    bank = {
        "tasks": [
            {
                "task_id": task.id,
                "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                "admission_sha256": f"admission-{task.id}",
                "source_id": f"source-{task.id}",
                "capability": "types",
                "contract_id": f"contract-{task.id}",
            }
            for task in tasks
        ]
    }
    runtime = RuntimeBundle("manifest", "manifest-hash", "archive", "archive-hash", "/opt")
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "original timestamp"}
    turn = {
        "message": {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "function": {"name": "shell", "arguments": '{"command": "ls -la"}'}}],
        },
        "response_token_ids": [3, 4],
    }
    records = [
        {
            "task_id": tasks[index // 8].id,
            "steps": [{"turn": deepcopy(turn), "transition": {"observations": [observation]}, "messages": []}],
            "messages": [observation],
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
            "loss_mask": [0, 0, 1, 1],
            "stop_reason": "stop",
            "grade": {"status": "graded", "reward": float(index % 2), "error": None, "failure": None},
            "interrupted_operation": None,
            "execution_error": None,
        }
        for index in range(144)
    ]
    for index in (103, 116, 141):
        records[index]["grade"] = {"status": "unavailable", "reward": None, "error": "No final grade"}
        records[index]["interrupted_operation"] = "start" if index == 103 else "grade"
    records[103].update(steps=[], messages=[], prompt_token_ids=[], response_token_ids=[], loss_mask=[])
    replacement_parquet = tmp_path / "replacement.parquet"
    write_tasks(str(replacement_parquet), [tasks[103 // 8]])
    lines = [(json.dumps(record) + "\n").encode() for record in records]
    traces = tmp_path / "traces.jsonl"
    traces.write_bytes(b"".join(lines))
    rewards = {
        task.id: [
            record["grade"]["reward"]
            for record in records
            if record["task_id"] == task.id and record["grade"]["status"] == "graded"
        ]
        for task in tasks
    }
    summary = {
        "model_identity": "parent",
        "tasks_identity": "bank",
        "count": 18,
        "samples_per_task": 8,
        "task_rewards": rewards,
    }
    replay_files = []
    reviews = []
    for index in (116, 141):
        recovered = deepcopy(records[index])
        recovered["grade"] = {"status": "graded", "reward": 0.0, "error": None, "failure": None}
        replay = {
            "original_line_index": index,
            "original_line_sha256": hashlib.sha256(lines[index]).hexdigest(),
            "task_id": records[index]["task_id"],
            "task_sha256": hashlib.sha256(tasks[index // 8].model_dump_json().encode()).hexdigest(),
            "source_commit": "original-runtime",
            "original_runtime_source": "original-runtime",
            "runtime_bundle": asdict(runtime),
            "model_requests": 0,
            "execution_error": None,
            "rollout": recovered,
        }
        evidence = pin(tmp_path / f"replay-{index}.json", replay)
        replay_files.append(evidence)
        reviews.append(
            {
                "original_line_index": index,
                "original_line_sha256": replay["original_line_sha256"],
                "replay_sha256": evidence.sha256,
                "accepted_reward": 0.0,
                "observation_exceptions": [],
            }
        )
    issuance = {
        "original_line_index": 103,
        "original_line_sha256": hashlib.sha256(lines[103]).hexdigest(),
        "original_traces_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
        "original_task_sha256": bank["tasks"][103 // 8]["task_sha256"],
        "tasks_sha256": hashlib.sha256(replacement_parquet.read_bytes()).hexdigest(),
        "runtime_module_hashes": {"engine": "engine-hash"},
        "evaluation": {
            "model_identity": "parent",
            "model_uri": "model",
            "tokenizer": "tokenizer",
            "tokenizer_revision": "revision",
            "runtime_bundle": asdict(runtime),
            "temperature": 1.0,
            "samples_per_task": 1,
            "limit": 1,
            "tasks_identity": "replacement",
        },
    }
    replacement = deepcopy(records[0])
    replacement["task_id"] = records[103]["task_id"]
    replacement_summary = {
        "model_identity": "parent",
        "tasks_identity": "replacement",
        "task_rewards": {replacement["task_id"]: [0.0]},
    }
    config = CalibrationRecoveryConfig(
        original_summary=pin(tmp_path / "summary.json", summary),
        original_traces=PinnedFile(str(traces), hashlib.sha256(traces.read_bytes()).hexdigest()),
        original_tasks=PinnedFile(str(parquet), hashlib.sha256(parquet.read_bytes()).hexdigest()),
        original_bank=pin(tmp_path / "bank.json", bank),
        review=pin(tmp_path / "review.json", {"replays": reviews, "startup_replacement": {"original_line_index": 103}}),
        replays=tuple(replay_files),
        replacement_issuance=pin(tmp_path / "issued.json", issuance),
        replacement_tasks=PinnedFile(str(replacement_parquet), issuance["tasks_sha256"]),
        replacement_summary=pin(tmp_path / "replacement-summary.json", replacement_summary),
        replacement_traces=pin(tmp_path / "replacement.jsonl", replacement),
        model_identity="parent",
        model_uri="model",
        tokenizer="tokenizer",
        tokenizer_revision="revision",
        bank_identity="bank",
        runtime_source="original-runtime",
        runtime_bundle=runtime,
        runtime_module_hashes={"engine": "engine-hash"},
        output_path=str(tmp_path / "result"),
    )
    return config, records, bank


def test_recovery_preserves_original_grades_and_links_three_completions(recovery, tmp_path):
    config, originals, bank = recovery
    original_bytes = config.original_traces.read_bytes()
    recover_calibration(config)
    result = json.loads((tmp_path / "result" / "recovery.json").read_text())
    assert result["original_grade_count"] == 141
    assert len(result["attempts"]) == 144
    for index, attempt in enumerate(result["attempts"]):
        assert attempt["original_grade"] == originals[index]["grade"]
        if index in (103, 116, 141):
            assert attempt["completed_grade"]["reward"] == 0
            assert attempt["grade_evidence"]["uri"] != config.original_traces.uri
        else:
            assert attempt["completed_grade"] == originals[index]["grade"]
    assert config.original_traces.read_bytes() == original_bytes
    summary = json.loads((tmp_path / "result" / "failure_summary.json").read_text())
    measured = calibration_measurements(summary, tuple(QualifiedTask(**row) for row in bank["tasks"]), "parent", "bank")
    assert len(measured) == 18
    assert all(len(item.rewards) == 8 for item in measured)
    sealed_bytes = (tmp_path / "result" / "recovery.json").read_bytes()
    recover_calibration(config)
    assert (tmp_path / "result" / "recovery.json").read_bytes() == sealed_bytes
    changed_review = config.review.read_json() | {"note": "different review input"}
    with pytest.raises(ValueError, match="Immutable record differs"):
        recover_calibration(replace(config, review=pin(tmp_path / "different-review.json", changed_review)))


@pytest.mark.parametrize("change", ["missing_grade", "model_identity", "tasks_identity"])
def test_selection_requires_complete_calibration_for_current_model_and_bank(recovery, tmp_path, change):
    config, _, bank = recovery
    recover_calibration(config)
    summary = json.loads((tmp_path / "result" / "failure_summary.json").read_text())
    if change == "missing_grade":
        summary["task_rewards"]["task-0"].pop()
    else:
        summary[change] = "different"
    with pytest.raises(ValueError):
        calibration_measurements(summary, tuple(QualifiedTask(**row) for row in bank["tasks"]), "parent", "bank")


@pytest.mark.parametrize(
    "change,message",
    [
        ("action", "saved model turn"),
        ("observation", "unreviewed observation"),
        ("tokens", "response_token_ids"),
        ("duplicate", "duplicates an attempt"),
        ("ungraded", "finite grade"),
        ("model", "checkpoint or task-bank identity"),
        ("bank", "checkpoint or task-bank identity"),
        ("replacement_task", "original attempt or sampling protocol"),
        ("temperature", "original attempt or sampling protocol"),
    ],
)
def test_recovery_rejects_changed_or_incomplete_evidence_before_output(recovery, tmp_path, change, message):
    config, _, _ = recovery
    if change in ("model", "bank"):
        config = replace(config, **{f"{change}_identity": "different"})
    elif change == "duplicate":
        config = replace(config, replays=(*config.replays, config.replays[0]))
    elif change == "ungraded":
        value = config.replacement_traces.read_json()
        value["grade"] = {"status": "unavailable", "reward": None}
        config = replace(config, replacement_traces=pin(tmp_path / "ungraded.jsonl", value))
    elif change == "replacement_task":
        config = replace(config, replacement_tasks=config.original_tasks)
        issuance = config.replacement_issuance.read_json()
        issuance["tasks_sha256"] = config.original_tasks.sha256
        config = replace(config, replacement_issuance=pin(tmp_path / "changed-issuance.json", issuance))
    elif change == "temperature":
        issuance = config.replacement_issuance.read_json()
        issuance["evaluation"]["temperature"] = 0.0
        config = replace(config, replacement_issuance=pin(tmp_path / "changed-issuance.json", issuance))
    else:
        value = config.replays[0].read_json()
        value["all_observations_match"] = True
        value["all_model_requests_match"] = True
        if change == "action":
            value["rollout"]["steps"][0]["turn"]["message"]["tool_calls"][0]["function"][
                "arguments"
            ] = '{"command": "echo changed"}'
        elif change == "observation":
            value["rollout"]["steps"][0]["transition"]["observations"][0]["content"] = "different output"
        else:
            value["rollout"]["response_token_ids"] = [9]
        replay = pin(tmp_path / "changed-replay.json", value)
        review = config.review.read_json()
        review["replays"][0]["replay_sha256"] = replay.sha256
        config = replace(
            config, replays=(replay, config.replays[1]), review=pin(tmp_path / "changed-review.json", review)
        )
    with pytest.raises(ValueError, match=message):
        recover_calibration(config)
    assert not (tmp_path / "result").exists()


def test_replay_accepts_only_exact_reviewed_observation_change(recovery):
    config, originals, _ = recovery
    replay = config.replays[0].read_json()
    decision = config.review.read_json()["replays"][0]
    after = {"role": "tool", "tool_call_id": "call-1", "content": "reviewed timestamp"}
    replay["rollout"]["steps"][0]["transition"]["observations"] = [after]
    replay["rollout"]["messages"] = [after]
    decision.update(
        exact_command="ls -la",
        tool_call_id="call-1",
        observation_exceptions=[{"step_index": 0, "original": originals[116]["messages"], "replay": [deepcopy(after)]}],
    )
    assert replay_grade(originals[116], replay, decision)["reward"] == 0
    after["content"] = "unreviewed timestamp"
    with pytest.raises(ValueError, match="exact reviewed"):
        replay_grade(originals[116], replay, decision)
