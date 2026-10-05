# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest
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
from experiments.post_training.russell_rsi.launch_dose_comparison import dose_replay_plan, freeze_dose_dataset
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256
from experiments.post_training.russell_rsi.replay import (
    ReplayDatasetConfig,
    calibration_signal_failure,
    freeze_replay_dataset,
    replay_plan,
    validate_replay_plan,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256


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

    family_source = replay_plan(
        plan,
        measurements,
        fresh,
        pilot_number=2,
        bank_identity="bank",
        calibration_identity="calibration",
        frozen_identity="frozen",
        parent_identity="parent",
        model_identity="parent",
        family_by_task={task.task_id: task.contract_id for task in records},
    )
    extended = dose_replay_plan(family_source, plan)
    dose_path = tmp_path / "dose"
    dose_path.mkdir()
    freeze_dose_dataset(ReplayDatasetConfig(str(bank_path), str(dose_path), tuple(records), extended))
    dose_rows = list(read_tasks(str(dose_path / "train.parquet")))
    assert len(dose_rows) == 128
    assert extended["schedule"][:64] == family_source["schedule"]
    assert [task.id for task in dose_rows] == [entry["task_id"] for entry in extended["schedule"]]
    for update in range(1, 9):
        assert Counter(entry["category"] for entry in extended["schedule"] if entry["update"] == update) == {
            "replay": 12,
            "targeted": 2,
            "exploration": 2,
        }
    assert json.loads((dose_path / "replay-plan.json").read_text()) == extended

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


def family_replay_fixture():
    original = tuple(
        QualifiedTask(
            str(index), f"hash-{index}", f"proof-{index}", f"source-{index}", "boundaries", f"contract-{index}"
        )
        for index in range(20)
    )
    variants = tuple(
        QualifiedTask(
            f"variant-{index}",
            f"variant-hash-{index}",
            f"variant-proof-{index}",
            f"variant-source-{index}",
            "boundaries",
            f"variant-contract-{index}",
            "variant",
        )
        for index in range(6)
    )
    records = original + variants
    parent = CheckpointScore("parent", (0.5, 0.5), 0.8)
    state = LoopState(parent, parent, parent, original[:16], completed_pilots=1)
    fresh = original[16:] + variants
    measurements = tuple(
        Measurement("parent", task.task_sha256, (1.0,) * (1 if index < 20 else 4) + (0.0,) * (7 if index < 20 else 4))
        for index, task in enumerate(records)
    )
    arguments = {
        "pilot_number": 2,
        "bank_identity": "bank",
        "calibration_identity": "calibration",
        "frozen_identity": "frozen",
        "parent_identity": "parent",
        "model_identity": "parent",
    }
    plan = round_plan(
        state,
        fresh,
        measurements,
        run_id="run",
        bank_identity="bank",
        calibration_identity="calibration",
        feedback_labels=("boundaries",),
        development_identity="coding-panel",
        retention_identity="retention",
        feedback_identity="feedback",
        runtime_identity="runtime",
        seed=9528,
    )
    families = {task.task_id: task.contract_id if task.relation == "independent" else "contract-0" for task in records}
    return plan, measurements, original[16:], families, arguments


def test_family_replay_prevents_variant_multiplicity_from_changing_family_probability():
    plan, measurements, targeted, families, arguments = family_replay_fixture()
    sealed = replay_plan(plan, measurements, targeted, **arguments, family_by_task=families)
    for category in ("replay", "exploration"):
        probabilities = sealed["sampling_spec"]["task_probabilities"][category]
        mass = Counter()
        for task_id, probability in probabilities.items():
            mass[families[task_id]] += probability
        assert len(mass) == 20
        assert list(mass.values()) == pytest.approx([1 / 20] * 20)
        assert probabilities["0"] == pytest.approx(1 / 20 / 7)
    assert sealed["sampling_spec"]["task_probabilities"]["targeted"] == {str(index): 1 / 4 for index in range(16, 20)}
    expected_category_q4 = (19 * 0.5 + (0.5 + 6 * (34 / 35)) / 7) / 20
    expected = (14 * expected_category_q4 + 2 * 0.5) / 16
    assert sealed["weighted_q4_by_update"] == pytest.approx([expected] * 4)
    assert sealed["weighted_mean_q4"] == pytest.approx(expected)
    for update in range(1, 5):
        assert Counter(row["category"] for row in sealed["schedule"] if row["update"] == update) == {
            "replay": 12,
            "targeted": 2,
            "exploration": 2,
        }
    validate_replay_plan(sealed, plan, **arguments, family_by_task=families)


@pytest.mark.parametrize("change", ["family", "probability", "schedule", "q4"])
def test_family_replay_validation_rejects_rehashed_protocol_changes(change):
    plan, measurements, targeted, families, arguments = family_replay_fixture()
    sealed = replay_plan(plan, measurements, targeted, **arguments, family_by_task=families)
    changed = deepcopy(sealed)
    if change == "family":
        changed["family_by_task"]["variant-0"] = "contract-1"
    elif change == "probability":
        changed["sampling_spec"]["task_probabilities"]["replay"]["0"] = 0.5
        changed["sampling_spec_sha256"] = compact_json_sha256(changed["sampling_spec"])
    elif change == "schedule":
        changed["schedule"][0]["task_id"] = "corrupted"
    else:
        changed["weighted_mean_q4"] = 0.99
    changed.pop("schedule_sha256")
    changed["schedule_sha256"] = compact_json_sha256(changed)
    with pytest.raises(ValueError):
        validate_replay_plan(changed, plan, **arguments, family_by_task=families)


def test_family_replay_cannot_select_variants_as_independent_targets():
    plan, measurements, targeted, families, arguments = family_replay_fixture()
    variant = next(task for task in plan.task_bank if task.relation == "variant")
    with pytest.raises(ValueError, match="independent"):
        replay_plan(plan, measurements, (variant,), **arguments, family_by_task=families)
    with pytest.raises(ValueError, match="explicit original family"):
        replay_plan(plan, measurements, targeted, **arguments)


def test_family_replay_schedule_does_not_concentrate_on_a_family_with_many_variants():
    plan, measurements, targeted, families, arguments = family_replay_fixture()
    original = tuple(task for task in plan.task_bank if task.relation == "independent")
    template = next(task for task in plan.task_bank if task.relation == "variant")
    variants = tuple(
        replace(template, task_id=f"v-{index}", task_sha256=f"v-hash-{index}", contract_id=f"v-contract-{index}")
        for index in range(500)
    )
    expanded = replace(plan, task_bank=original + variants)
    calibration = measurements[:20] + tuple(
        Measurement("parent", task.task_sha256, (1.0,) * 4 + (0.0,) * 4) for task in variants
    )
    families = {task.task_id: task.contract_id for task in original} | {task.task_id: "contract-0" for task in variants}
    sealed = replay_plan(expanded, calibration, targeted, **arguments, family_by_task=families)
    replay = [row for row in sealed["schedule"] if row["category"] == "replay"]
    # Uniform task sampling would put almost all 48 groups in this family.
    assert sum(families[row["task_id"]] == "contract-0" for row in replay) < 12
    assert len({families[row["task_id"]] for row in replay}) > 10
