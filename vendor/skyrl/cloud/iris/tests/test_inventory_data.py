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
from rolloutengine.validation import validate_spec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec, PackageSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.identity import canonical_sha256
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
from verifyit.spec import JudgeSpec

from experiments.post_training.inventory import FAMILIES
from experiments.post_training.inventory._data import (
    DERIVED_GRADER_LOCK,
    IMAGE_VERIFIER,
    SOURCE_GRADER_LOCK,
    prepare,
)


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
                            "verifier_machine": IMAGE_VERIFIER.model_dump(mode="json"),
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
    assert tasks[0].verifier_machine == IMAGE_VERIFIER
    native_original = json.loads(inputs.training_rows[1]["lowered_task_spec"])["task"]
    assert tasks[1].harness == "assistant_prefix"
    assert tasks[1].task.context.events[-1] == TextMessage(role="assistant", content='{"a":[1')
    assert tasks[1].task.resources.model_dump(mode="json") == native_original["resources"]
    assert tasks[1].task.grader.mode == "structure_closure"
    full = read_tasks(first / "full_validation.parquet")
    fixed = read_tasks(first / "validation.parquet")
    assert full[0].verifier_machine == IMAGE_VERIFIER
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


def test_persisted_image_graders_are_admitted_by_the_image_backend(tmp_path):
    inputs = sources(tmp_path)
    output = tmp_path / "output"
    produce(inputs.training, inputs.validation, output)
    factories = {"daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created"))}
    for spec in read_tasks(output / "train.parquet") + read_tasks(output / "full_validation.parquet"):
        validate_spec(spec, factories)


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError("Admission must not acquire a machine")


@pytest.mark.parametrize(
    "software", [{"packages_lock": "s3://fixture/grader.lock"}, {"package_spec": PackageSpec(pypi_packages=())}]
)
def test_persisted_package_graders_pass_local_allocation_admission(tmp_path, software):
    inputs = sources(tmp_path)
    task = json.loads(inputs.training_rows[0]["task_json"])
    task["grader"]["environment"] = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, **software
    ).model_dump(mode="json")
    inputs.training_rows[0]["task_json"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(inputs.training_rows), inputs.training)
    output = tmp_path / "output"
    produce(inputs.training, inputs.validation, output)
    spec = read_tasks(output / "train.parquet")[0]
    assert spec.task.model_dump(mode="json") == {**task, "schema_version": "0.28"}
    validate_spec(spec, {"local": LocalAdmissionFactory()})


def test_persisted_reconstruction_changes_only_the_selected_script_environment(tmp_path):
    inputs = sources(tmp_path)
    baseline = tmp_path / "baseline"
    produce(inputs.training, inputs.validation, baseline)
    native_before = read_tasks(baseline / "train.parquet")[1].model_dump(mode="json")
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
    output = tmp_path / "derived"
    produce(inputs.training, inputs.validation, output)
    tasks = read_tasks(output / "train.parquet")
    assert [spec.task.id for spec in tasks] == [row["task_id"] for row in inputs.training_rows]
    expected = {**target, "schema_version": "0.28"}
    expected = json.loads(json.dumps(expected))
    expected["grader"]["environment"]["packages_lock"] = DERIVED_GRADER_LOCK
    assert tasks[0].task.model_dump(mode="json") == expected
    assert tasks[1].model_dump(mode="json") == native_before
    assert tasks[2].task.model_dump(mode="json") == {**unrelated, "schema_version": "0.28"}
    assert tasks[3].task.model_dump(mode="json") == {**judge, "schema_version": "0.28"}
    assert tasks[3].verifier_machine is None
    for filename in ("validation.parquet", "full_validation.parquet"):
        assert (output / filename).read_bytes() == (baseline / filename).read_bytes()
    derivation = json.loads((output / "summary.json").read_text())["environment_derivation"]
    assert derivation["affected_rows"] == 1
    assert derivation["task_spec_digests"] == {
        target["id"]: {
            "source_task_sha256": canonical_sha256(target),
            "derived_task_sha256": canonical_sha256(expected),
        }
    }
