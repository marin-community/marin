# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for the pinned Snowball task split."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import JudgeSpec

from experiments.post_training.inventory.data import (
    DERIVED_GRADER_LOCK,
    FAMILIES,
    FULL_VALIDATION_FILENAME,
    SOURCE_GRADER_LOCK,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    IngestConfig,
    prepare,
)


def source_task(identifier: str) -> dict:
    task = TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="release", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            env={"MODE": "fixture"},
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="python@sha256:" + "a" * 64
            ),
        ),
        resources=ResourceGroups(verifier=(inline_resource("private.txt", identifier.encode()),)),
    ).model_dump(mode="json")
    return {**task, "schema_version": "0.27"}


def native_source(identifier: str) -> str:
    task = source_task(identifier)
    task["source"]["dataset"] = "structure-closure"
    task["grader"] = {"kind": "session"}
    config = inline_resource(
        "config.json",
        json.dumps({"extras": {"prefix": '{"a":[1', "expected": '{"a":[1,2]}', "format": "json"}}).encode(),
    )
    task["resources"]["verifier"].append(config.model_dump(mode="json"))
    return json.dumps(
        {
            "task": task,
            "runtime": {"task_machine": None, "verifier_machine": None},
            "session": {"task_session": "structure_closure", "max_turns": 1},
        }
    )


@dataclass(frozen=True)
class SourceFiles:
    training: Path
    validation: Path
    training_rows: list[dict]
    validation_rows: list[dict]


def sources(directory: Path) -> SourceFiles:
    training_rows = [
        {
            "task_id": "training-a",
            "contract_kind": "task_spec",
            "task_json": json.dumps(source_task("training-a")),
            "lowered_task_spec": None,
        },
        {
            "task_id": "training-native",
            "contract_kind": "native-structure_closure",
            "task_json": None,
            "lowered_task_spec": native_source("training-native"),
        },
        {
            "task_id": "training-b",
            "contract_kind": "task_spec",
            "task_json": json.dumps(source_task("training-b")),
            "lowered_task_spec": None,
        },
    ]
    validation_rows = []
    for family in FAMILIES:
        for index in range(21):
            identifier = f"heldout-{family}-{index}"
            raw = (
                native_source(identifier)
                if family == "structural"
                else json.dumps(
                    {
                        "task": source_task(identifier),
                        "runtime": {"task_machine": None, "verifier_machine": None},
                        "session": {"task_session": "shellbox", "max_turns": 1},
                    }
                )
            )
            validation_rows.append({"task_id": identifier, "evaluation_family": family, "lowered_task_spec": raw})
    training = directory / "source-train.parquet"
    validation = directory / "source-heldout.parquet"
    pq.write_table(pa.Table.from_pylist(training_rows), training)
    pq.write_table(pa.Table.from_pylist(validation_rows), validation)
    return SourceFiles(training, validation, training_rows, validation_rows)


def produce(inputs: SourceFiles, output: Path) -> None:
    prepare(
        IngestConfig(
            str(inputs.training),
            len(inputs.training_rows),
            str(inputs.validation),
            len(inputs.validation_rows),
            str(output),
        )
    )


def read_tasks(path: Path) -> list[TaskSpec]:
    return [TaskSpec.model_validate_json(row["task_spec"]) for row in pq.read_table(path).to_pylist()]


def test_inventory_preserves_private_tasks_and_selects_disjoint_families(tmp_path):
    inputs = sources(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    produce(inputs, first)
    produce(inputs, second)
    training = read_tasks(first / TRAIN_FILENAME)
    full = read_tasks(first / FULL_VALIDATION_FILENAME)
    fixed = read_tasks(first / VALIDATION_FILENAME)

    assert [task.id for task in training] == [row["task_id"] for row in inputs.training_rows]
    assert resource_bytes(training[0].resources.verifier[0]) == b"training-a"
    assert training[0].grader.argv == ("python", "/tests/grade.py")
    assert training[1].context.events[-1] == TextMessage(role="assistant", content='{"a":[1')
    assert training[1].grader.mode == "structure_closure"
    assert [task.id for task in full] == [row["task_id"] for row in inputs.validation_rows]
    family_by_id = {row["task_id"]: row["evaluation_family"] for row in inputs.validation_rows}
    assert Counter(family_by_id[task.id] for task in fixed) == dict.fromkeys(FAMILIES, 20)
    assert {task.id for task in training}.isdisjoint(task.id for task in full)
    assert [task.id for task in fixed] == [task.id for task in read_tasks(second / VALIDATION_FILENAME)]
    summary = json.loads((first / "summary.json").read_text())
    assert (summary["training_rows"], summary["validation_rows"], summary["full_validation_rows"]) == (3, 100, 105)


def test_changed_training_identity_cannot_hide_a_heldout_prompt(tmp_path):
    inputs = sources(tmp_path)
    leaked = json.loads(inputs.validation_rows[0]["lowered_task_spec"])["task"]
    leaked["id"] = "different-training-id"
    leaked["source"]["row"] = "different-source-row"
    inputs.training_rows[0] = {**inputs.training_rows[0], "task_id": leaked["id"], "task_json": json.dumps(leaked)}
    pq.write_table(pa.Table.from_pylist(inputs.training_rows), inputs.training)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps the original heldout set"):
        produce(inputs, output)
    assert not output.exists()


def test_reconstructed_lock_changes_only_target_script_tasks(tmp_path):
    inputs = sources(tmp_path)
    target = json.loads(inputs.training_rows[0]["task_json"])
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        packages_lock=SOURCE_GRADER_LOCK,
        environment_variables={"NLTK_DATA": "/srv/nltk_data"},
    )
    target["grader"]["environment"] = environment.model_dump(mode="json")
    inputs.training_rows[0]["task_json"] = json.dumps(target)
    unrelated = json.loads(inputs.training_rows[2]["task_json"])
    unrelated["grader"]["environment"] = environment.model_copy(
        update={"packages_lock": "s3://fixture/unrelated-grader.lock"}
    ).model_dump(mode="json")
    inputs.training_rows[2]["task_json"] = json.dumps(unrelated)
    judge = source_task("host-judge")
    judge["grader"] = verifyit_package(
        JudgeSpec(references=("private-answer",)), environment=environment
    ).grader.model_dump(mode="json")
    inputs.training_rows.append(
        {"task_id": judge["id"], "contract_kind": "task_spec", "task_json": json.dumps(judge), "lowered_task_spec": None}
    )
    pq.write_table(pa.Table.from_pylist(inputs.training_rows), inputs.training)
    source_bytes = inputs.training.read_bytes()

    output = tmp_path / "derived"
    produce(inputs, output)
    tasks = read_tasks(output / TRAIN_FILENAME)
    assert tasks[0].grader.environment.packages_lock == DERIVED_GRADER_LOCK
    assert tasks[0].grader.environment.environment_variables == {"NLTK_DATA": "/srv/nltk_data"}
    assert tasks[2].grader.environment.packages_lock == "s3://fixture/unrelated-grader.lock"
    assert tasks[3].grader.environment.packages_lock == SOURCE_GRADER_LOCK
    assert inputs.training.read_bytes() == source_bytes
    derivation = json.loads((output / "summary.json").read_text())["environment_derivation"]
    assert derivation["affected_rows"] == 1
    assert derivation["replacement_packages_lock"] == DERIVED_GRADER_LOCK
