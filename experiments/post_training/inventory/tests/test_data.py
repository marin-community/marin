# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The inventory producer preserves private tasks and a fixed, disjoint heldout set."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.models import (
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    Source,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.inventory._data import FAMILIES, LOCAL_VERIFIER, prepare


def source_task(identifier: str) -> dict:
    task = arithmetic_task(
        id=identifier,
        source=Source(dataset="fixture", revision="release", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
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
    train_rows = [
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
    heldout_rows = []
    for family in FAMILIES:
        for index in range(21):
            identifier = f"heldout-{family}-{index}"
            raw = (
                native_source(identifier)
                if family == "structural"
                else json.dumps(
                    {
                        "task": source_task(identifier),
                        "runtime": {
                            "task_machine": None,
                            "verifier_machine": {**LOCAL_VERIFIER.model_dump(mode="json"), "backend": "daytona"},
                        },
                        "session": {"task_session": "shellbox", "max_turns": 1},
                    }
                )
            )
            heldout_rows.append({"task_id": identifier, "evaluation_family": family, "lowered_task_spec": raw})
    train, heldout = directory / "source-train.parquet", directory / "source-heldout.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows), train)
    pq.write_table(pa.Table.from_pylist(heldout_rows), heldout)
    return SourceFiles(train, heldout, train_rows, heldout_rows)


def produce(train: Path, heldout: Path, output: Path) -> None:
    prepare(
        str(train),
        hashlib.sha256(train.read_bytes()).hexdigest(),
        pq.read_metadata(train).num_rows,
        str(heldout),
        hashlib.sha256(heldout.read_bytes()).hexdigest(),
        pq.read_metadata(heldout).num_rows,
        str(output),
    )


def read_tasks(path: Path) -> list[TaskRolloutSpec]:
    return [TaskRolloutSpec.model_validate_json(row["task_rollout_spec"]) for row in pq.read_table(path).to_pylist()]


def test_persisted_inventory_preserves_tasks_and_repeats_the_same_stratified_split(tmp_path):
    inputs = sources(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    produce(inputs.training, inputs.validation, first)
    produce(inputs.training, inputs.validation, second)
    tasks = read_tasks(first / "train.parquet")
    assert [spec.task.id for spec in tasks] == [row["task_id"] for row in inputs.training_rows]
    original = json.loads(inputs.training_rows[0]["task_json"])
    restored = tasks[0].task.model_dump(mode="json")
    assert restored == {**original, "schema_version": "0.28"}
    assert tasks[0].verifier_machine == LOCAL_VERIFIER
    native_original = json.loads(inputs.training_rows[1]["lowered_task_spec"])["task"]
    assert tasks[1].harness == "assistant_prefix"
    assert tasks[1].task.context.events[-1] == TextMessage(role="assistant", content='{"a":[1')
    assert tasks[1].task.resources.model_dump(mode="json") == native_original["resources"]
    assert tasks[1].task.grader.mode == "structure_closure"
    full = read_tasks(first / "full_validation.parquet")
    fixed = read_tasks(first / "validation.parquet")
    assert full[0].verifier_machine == LOCAL_VERIFIER
    assert (
        full[0].task.grader.model_dump(mode="json")
        == json.loads(inputs.validation_rows[0]["lowered_task_spec"])["task"]["grader"]
    )
    assert [spec.task.id for spec in full] == [row["task_id"] for row in inputs.validation_rows]
    family_by_id = {row["task_id"]: row["evaluation_family"] for row in inputs.validation_rows}
    assert Counter(family_by_id[spec.task.id] for spec in fixed) == dict.fromkeys(FAMILIES, 20)
    assert [spec.task.id for spec in fixed] == [
        spec.task.id for spec in full if spec.task.id in {item.task.id for item in fixed}
    ]
    assert {spec.task.id for spec in tasks}.isdisjoint(spec.task.id for spec in full)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    summary = json.loads((first / "summary.json").read_text())
    assert (summary["training_rows"], summary["validation_rows"], summary["full_validation_rows"]) == (3, 100, 105)
    for filename, digest in summary["files"].items():
        assert hashlib.sha256((first / filename).read_bytes()).hexdigest() == digest


def test_changed_training_identity_cannot_hide_a_heldout_prompt(tmp_path):
    inputs = sources(tmp_path)
    leaked = json.loads(inputs.validation_rows[0]["lowered_task_spec"])["task"]
    leaked["id"] = "different-training-id"
    leaked["source"]["row"] = "different-source-row"
    inputs.training_rows[0] = {**inputs.training_rows[0], "task_id": leaked["id"], "task_json": json.dumps(leaked)}
    pq.write_table(pa.Table.from_pylist(inputs.training_rows), inputs.training)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps the original heldout set"):
        produce(inputs.training, inputs.validation, output)
    assert not output.exists()
