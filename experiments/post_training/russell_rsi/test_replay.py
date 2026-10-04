# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter

from taskcompendium.grading import exact_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    LoopState,
    Measurement,
    QualifiedTask,
    round_plan,
)
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256
from experiments.post_training.russell_rsi.replay import (
    ReplayDatasetConfig,
    calibration_signal_failure,
    freeze_replay_dataset,
    replay_plan,
)


def test_replay_artifact_preserves_source_tasks_and_seals_four_update_schedules(tmp_path):
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    task_specs = []
    records = []
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
        task_hash = canonical_sha256(task.model_dump(mode="json"))
        proof = json.dumps({"task_sha256": task_hash, "source_group": f"source-{index}"}).encode()
        proof_hash = hashlib.sha256(proof).hexdigest()
        proof_path = bank_path / "evidence" / proof_hash
        proof_path.mkdir(parents=True)
        (proof_path / "proposal.json").write_bytes(proof)
        task_specs.append(task)
        records.append(QualifiedTask(str(index), task_hash, proof_hash, f"source-{index}", "types", f"contract-{index}"))
    write_tasks(str(bank_path / "train.parquet"), task_specs)

    parent = CheckpointScore("parent", (0.5, 0.5), 0.8)
    state = LoopState(parent, parent, parent, tuple(records[:16]), completed_pilots=1)
    fresh = tuple(records[16:])
    measurements = tuple(
        Measurement("parent", task.task_sha256, (1.0,) * (index % 8) + (0.0,) * (8 - index % 8))
        for index, task in enumerate(records)
    )
    plan = round_plan(
        state,
        fresh,
        measurements,
        run_id="run",
        bank_identity="bank",
        calibration_identity="calibration",
        feedback_labels=("types",),
        development_identity="coding-panel",
        retention_identity="retention",
        feedback_identity="feedback",
        runtime_identity="runtime",
        seed=9528,
    )
    sealed = replay_plan(
        plan,
        measurements,
        fresh,
        pilot_number=2,
        bank_identity="bank",
        calibration_identity="calibration",
        frozen_identity="frozen",
        parent_identity="parent",
        model_identity="parent",
    )
    output = tmp_path / "replay"
    output.mkdir()
    freeze_replay_dataset(ReplayDatasetConfig(str(bank_path), str(output), tuple(records), sealed))

    exported = list(read_tasks(str(output / "train.parquet")))
    schedule = sealed["schedule"]
    assert len(exported) == 64
    assert [task.id for task in exported] == [item["task_id"] for item in schedule]
    assert len({item["occurrence_id"] for item in schedule}) == 64
    assert [item["row_index"] for item in schedule] == list(range(64))
    assert exported == [task_specs[int(item["task_id"])] for item in schedule]
    assert json.loads((output / "replay-plan.json").read_text()) == sealed
    for update in range(1, 5):
        categories = Counter(item["category"] for item in schedule if item["update"] == update)
        assert categories == {"replay": 12, "targeted": 2, "exploration": 2}
    assert sealed["sampling_spec"]["data_shuffle"] is False
    assert sealed["sampling_spec"]["epochs"] == 1
    assert sealed["signal_gate_passed"]

    one_target = replay_plan(
        plan,
        measurements,
        (fresh[0],),
        pilot_number=2,
        bank_identity="bank",
        calibration_identity="calibration",
        frozen_identity="frozen",
        parent_identity="parent",
        model_identity="parent",
    )
    targeted_entries = [entry for entry in one_target["schedule"] if entry["category"] == "targeted"]
    assert len(targeted_entries) == 8
    assert {entry["task_id"] for entry in targeted_entries} == {fresh[0].task_id}


def test_replay_signal_gate_records_no_variation_and_rejects_unsupported_grades():
    no_variation = tuple(Measurement("parent", str(index), (0.0,) * 8) for index in range(16))
    assert calibration_signal_failure(no_variation) == "no_calibration_reward_variation"
    fractional = (Measurement("parent", "task", (0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),)
    assert calibration_signal_failure(fractional) == "calibration_rewards_not_binary"
