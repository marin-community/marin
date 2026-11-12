# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prebuilt importer admits a cohort only as published and only when valid."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    NoGrader,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import prebuilt

IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


def _task(task_id: str, *, image: str | None = IMAGE, capabilities: tuple[str, ...] = ()) -> dict:
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=image,
        working_directory="/app",
        setup_commands=("true",),
        capabilities=capabilities,
    )
    grader = ScriptGrader(
        argv=("true",),
        environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=image),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="d", revision="r", row=task_id, importer_revision="1"),
        resources=ResourceGroups(),
    )
    return {
        "path": task_id,
        "lowered_task_spec": json.dumps({"task": spec.model_dump(mode="json"), "runtime": {"task_machine": {}}}),
    }


def _write(tmp_path, rows: list[dict]):
    path = tmp_path / "tasks.parquet"
    table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("lowered_task_spec", pa.string())]))
    pq.write_table(table, path)
    return path


def _cohort(rows: int):
    return prebuilt.PrebuiltCohort(name="c", version="1", repo="org/repo", revision="rev", expected_rows=rows)


def test_a_published_cohort_is_downloaded_and_validated(tmp_path, monkeypatch):
    path = _write(tmp_path, [_task("a"), _task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    assert prebuilt.import_cohort(_cohort(2)) == str(path)


def test_a_cohort_smaller_than_recorded_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, [_task("a")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="has 1 rows, expected 2"):
        prebuilt.import_cohort(_cohort(2))


def test_an_envelope_without_machine_shapes_is_refused(tmp_path, monkeypatch):
    row = _task("a")
    row["lowered_task_spec"] = json.dumps({"task": json.loads(row["lowered_task_spec"])["task"]})
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="no runtime machines"):
        prebuilt.import_cohort(_cohort(1))


@pytest.mark.parametrize("change", ["none", "package", "infrastructure_zero"])
def test_native_harbor_import_preserves_controlled_package(tmp_path, monkeypatch, change):
    task = TaskSpec.model_validate(json.loads(_task("native")["lowered_task_spec"])["task"])
    task = task.model_copy(
        update={
            "tags": ("harbor",),
            "answer_type": AnswerType.WORKSPACE_STATE,
            "environment_requirements": EnvironmentRequirements(),
            "grader": NoGrader(reason="Harbor grades the completed trial"),
            "resources": ResourceGroups(
                verifier=tuple(
                    inline_resource(path, content)
                    for path, content in {
                        "task.toml": b'[environment]\ndocker_image = "custom-task-image"\n',
                        "instruction.md": b"Fix the repository.",
                        "tests/test.sh": b"#!/bin/bash\nexit 0\n",
                    }.items()
                )
            ),
        }
    )
    serialized = task.model_dump_json()
    receipt = {
        "task_id": "native",
        "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "status": "admitted",
        "controls": {
            "bug": {"status": "scored", "reward": 0},
            "reference": {"status": "scored", "reward": 1},
        },
    }
    if change == "package":
        serialized = task.model_copy(update={"labels": {"changed": "after-controls"}}).model_dump_json()
    elif change == "infrastructure_zero":
        receipt["controls"]["bug"]["status"] = "infra_error"
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": serialized}]), path)
    (tmp_path / "controls.jsonl").write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **kwargs: str(tmp_path / kwargs["filename"]))
    if change == "none":
        assert prebuilt.import_harbor_cohort(_cohort(1)) == str(path)
    else:
        with pytest.raises(ValueError, match=r"differs from its control|controls do not admit"):
            prebuilt.import_harbor_cohort(_cohort(1))


def test_quick_pass_writes_the_validated_cohort_as_converted_tasks(tmp_path, monkeypatch):
    path = _write(tmp_path, [_task("a"), _task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    output = tmp_path / "out"
    artifact = prebuilt._quick_view(_cohort(2), path, output)

    manifest = json.loads((output / "manifest.json").read_text())
    tasks = [
        TaskSpec.model_validate_json(item)
        for item in pq.read_table(output / "normalize").column("task_json").to_pylist()
    ]
    assert artifact.status == "completed"
    assert [task.id for task in tasks] == ["a", "b"]
    assert manifest["input_rows"] == manifest["converted_rows"] == 2
    assert manifest["rejections"] == {} and manifest["intended_use"] == "train"
