# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle
from taskcompendium.grading import exact_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    Difficulty,
    FrozenRoundConfig,
    LoopState,
    Measurement,
    QualifiedTask,
    RoundResult,
    StopReason,
    advance,
    freeze_round_dataset,
    load_round,
    round_plan,
    seal_round,
)
from experiments.post_training.russell_rsi.rollout_eval import DevelopmentEvaluationConfig
from experiments.post_training.russell_rsi.startup_replacement import (
    StartupReplacementConfig,
    issue_startup_replacement,
    validate_startup_replacement,
)


def tasks(start, count):
    return tuple(
        QualifiedTask(str(i), f"hash-{i}", f"admission-{i}", f"source-{i}", "types", f"contract-{i}")
        for i in range(start, start + count)
    )


def plan(state, fresh=()):
    bank = (*state.bank, *fresh)
    measurements = tuple(
        Measurement(state.working.checkpoint_identity, task.task_sha256, (1.0,) * (i % 9) + (0.0,) * (8 - i % 9))
        for i, task in enumerate(bank)
    )
    return round_plan(
        state,
        fresh,
        measurements,
        run_id="run",
        bank_identity="bank",
        calibration_identity="calibration",
        feedback_labels=("types",),
        development_identity="coding-panel",
        retention_identity="retention",
        feedback_identity="coding-evidence",
        runtime_identity="runtime",
        seed=9528,
    )


def initial_state():
    parent = CheckpointScore("parent", (25 / 32, 25 / 32), 0.8)
    return LoopState(parent, parent, parent, tasks(0, 16))


@pytest.mark.parametrize("change", [None, "candidate", "task"])
def test_startup_replacement_accepts_only_the_original_task_before_model_output(tmp_path, change):
    task = TaskSpec(
        id="original",
        context=ConversationInput(events=(TextMessage(role="user", content="Return done."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("done"),
        source=Source(dataset="source", revision="pin", row="0", importer_revision="1"),
    )
    original_task_sha256 = hashlib.sha256(json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
    original = {
        "task_id": task.id,
        "interrupted_operation": "grade" if change == "candidate" else "start",
        "steps": (
            [{"turn": {"message": {"role": "assistant", "content": "failed candidate"}}}]
            if change == "candidate"
            else []
        ),
        "response_token_ids": [2] if change == "candidate" else [],
        "grade": {"status": "unavailable", "reward": None},
    }
    traces = tmp_path / "traces.jsonl"
    original_line = json.dumps(original) + "\n"
    traces.write_text(json.dumps({**original, "task_id": "another-task"}) + "\n" + original_line)
    if change == "task":
        task = task.model_copy(
            update={"context": ConversationInput(events=(TextMessage(role="user", content="Changed."),))}
        )
    parquet = tmp_path / "train.parquet"
    write_tasks(str(parquet), [task])
    config = StartupReplacementConfig(
        evaluation=DevelopmentEvaluationConfig(
            model_uri="model",
            model_identity="parent",
            tasks_identity="replacement",
            tokenizer="tokenizer",
            tokenizer_revision="revision",
            tasks_path=str(parquet),
            output_path=str(tmp_path / "result"),
            runtime_bundle=RuntimeBundle("manifest", "manifest-hash", "archive", "archive-hash", "/opt"),
            limit=1,
            samples_per_task=1,
            temperature=1.0,
        ),
        original_traces_uri=str(traces),
        original_traces_sha256=hashlib.sha256(traces.read_bytes()).hexdigest(),
        original_line_index=1,
        original_line_sha256=hashlib.sha256(original_line.encode()).hexdigest(),
        original_task_sha256=original_task_sha256,
        tasks_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        runtime_module_hashes={"json": hashlib.sha256(Path(json.__file__).read_bytes()).hexdigest()},
    )
    if change is None:
        assert validate_startup_replacement(config) == original
        with pytest.raises(ValueError, match="runtime differs"):
            issue_startup_replacement(replace(config, runtime_module_hashes={"json": "different-runtime"}))
        issue_startup_replacement(config)
        issuance = tmp_path / "result" / "replacement-issued.json"
        issued_bytes = issuance.read_bytes()
        assert json.loads(issued_bytes)["original_line_sha256"] == config.original_line_sha256
        with pytest.raises(ValueError, match="already issued"):
            issue_startup_replacement(config)
        assert issuance.read_bytes() == issued_bytes
    else:
        with pytest.raises(ValueError, match="without model output" if change == "candidate" else "frozen task"):
            validate_startup_replacement(config)


def test_tied_working_checkpoint_continues_then_strict_champion_improvement():
    state = initial_state()
    first = plan(state)
    state = advance(
        state, first, RoundResult(CheckpointScore("tie", state.parent.development, 0.8), "reload", "eval", 4)
    )
    assert state.working.checkpoint_identity == "tie"
    assert state.champion.checkpoint_identity == "parent"
    second = plan(state, tasks(16, 3))
    assert second.current_checkpoint == "tie"
    assert len(second.selected_tasks) == 16
    assert second.fresh_count == 3 and second.retained_count == 13
    state = advance(
        state, second, RoundResult(CheckpointScore("better", (26 / 32, 25 / 32), 0.8), "reload2", "eval2", 4)
    )
    assert state.champion.checkpoint_identity == "better"
    assert state.rounds_without_improvement == 0


def test_regression_falls_back_to_champion_and_stops_after_two_stale_rounds():
    state = initial_state()
    first = plan(state)
    state = advance(
        state, first, RoundResult(CheckpointScore("regression", (26 / 32, 24 / 32), 0.8), "reload", "eval", 4)
    )
    assert state.working == state.champion == state.parent
    second = plan(state, tasks(16, 1))
    state = advance(
        state,
        second,
        RoundResult(CheckpointScore("retention-loss", (27 / 32, 27 / 32), 0.7), "reload2", "eval2", 4),
    )
    assert state.stop_reason == StopReason.NO_IMPROVEMENT
    assert state.champion == state.parent


def test_selected_tasks_retain_bands_and_manifests_reject_changed_resume(tmp_path):
    state = initial_state()
    frozen = plan(state)
    assert frozen.absent_bands == ()
    result = RoundResult(CheckpointScore("candidate", (26 / 32, 26 / 32), 0.8), "reload", "eval", 4)
    updated = advance(state, frozen, result)
    digest = seal_round(StoragePath(str(tmp_path)), updated, frozen, result, "previous")
    assert seal_round(StoragePath(str(tmp_path)), updated, frozen, result, "previous") == digest
    path = StoragePath(str(tmp_path)) / f"{frozen.name}.json"
    inputs = {
        key: value
        for key, value in asdict(frozen).items()
        if key not in {"name", "selected_tasks", "retained_count", "fresh_count", "absent_bands"}
    }
    assert load_round(path, inputs, "previous").state == updated
    with pytest.raises(ValueError, match="resume input"):
        load_round(path, {**inputs, "runtime_identity": "different-runtime"}, "previous")
    with pytest.raises(ValueError, match="sealed record"):
        seal_round(StoragePath(str(tmp_path)), updated, frozen, result, "different-previous")


def test_uninformative_current_checkpoint_rollouts_do_not_allocate_training():
    state = initial_state()
    measured = tuple(Measurement("parent", task.task_sha256, (0.0,) * 8) for task in state.bank)
    with pytest.raises(ValueError, match="reward variation"):
        round_plan(
            state,
            (),
            measured,
            run_id="run",
            bank_identity="bank",
            calibration_identity="calibration",
            feedback_labels=("types",),
            development_identity="coding-panel",
            retention_identity="retention",
            feedback_identity="coding-evidence",
            runtime_identity="runtime",
            seed=9528,
        )
    assert Measurement("parent", "a", (1.0,) * 6 + (0.0,) * 2).difficulty == Difficulty.EASY


@pytest.mark.parametrize("relation", ["variant", "replacement", "alias"])
def test_adaptive_round_cannot_count_a_variant_as_a_new_capability_contract(relation):
    state = replace(initial_state(), completed_pilots=1)
    variant = tuple(replace(task, relation=relation) for task in tasks(16, 1))
    with pytest.raises(ValueError, match="task_supply_exhausted"):
        plan(state, variant)


def test_frozen_round_writes_selected_unique_tasks_and_rejects_changed_bank_content(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    task_specs, records = [], []
    for index in range(20):
        task = TaskSpec(
            id=str(index),
            context=ConversationInput(events=(TextMessage(role="user", content=f"Task {index}"),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer("done"),
            source=Source(dataset="source", revision="pin", row=str(index), importer_revision="1"),
            metadata={"split": "train"},
        )
        task_hash = hashlib.sha256(json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
        proof = json.dumps({"task_sha256": task_hash, "source_group": f"source-{index}"}).encode()
        proof_hash = hashlib.sha256(proof).hexdigest()
        proof_path = bank / "evidence" / proof_hash
        proof_path.mkdir(parents=True)
        (proof_path / "proposal.json").write_bytes(proof)
        records.append(QualifiedTask(str(index), task_hash, proof_hash, f"source-{index}", "types", f"contract-{index}"))
        task_specs.append(task)
    write_tasks(str(bank / "train.parquet"), task_specs)
    frozen = plan(replace(initial_state(), bank=tuple(records)))
    destination = tmp_path / "round"
    destination.mkdir()
    freeze_round_dataset(FrozenRoundConfig(str(bank), frozen, str(destination)))
    exported = list(read_tasks(str(destination / "train.parquet")))
    assert len(exported) == 16
    assert {task.id for task in exported} == {task.task_id for task in frozen.selected_tasks}
    assert exported == sorted(exported, key=lambda task: task.id)
    selected_id = frozen.selected_tasks[0].task_id
    altered = [
        task.model_copy(update={"metadata": {"split": "test"}}) if task.id == selected_id else task
        for task in task_specs
    ]
    write_tasks(str(bank / "train.parquet"), altered)
    with pytest.raises(ValueError, match="content differs"):
        freeze_round_dataset(FrozenRoundConfig(str(bank), frozen, str(tmp_path / "bad-round")))


def test_adaptive_selection_reserves_a_contract_for_the_actual_feedback():
    state = replace(initial_state(), completed_pilots=1)
    unrelated = tuple(replace(task, capability="parsing") for task in tasks(16, 8))
    targeted = replace(tasks(24, 1)[0], capability="types")
    selected = plan(state, (*unrelated, targeted)).selected_tasks
    assert targeted in selected
    assert len(selected) == 16
