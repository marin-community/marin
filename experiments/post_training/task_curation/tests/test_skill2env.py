# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable
from shellbox.machine import HostImage, MachineSpec, NetworkPolicy, PackageSpec, UnsupportedMachineSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    ConversationTrace,
    DockerBuildContext,
    GradingAttempt,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    require_resolved_environment,
)
from taskcompendium.pipeline.conversion import convert_record, convert_row
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.sources import conversion_shards, staged_file_rows
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.skill2env.conversion import CAPTURE_ROOTS, RELEASE, convert_task
from experiments.post_training.task_curation.datasets.skill2env.tasks import sources
from experiments.post_training.task_curation.pipeline import source_recipe


def exported_row(name: str = "fixture", working_directory: str = "/workspace") -> RawRow:
    files = {
        "instruction.md": b"Repair the answer and keep the checker intact.",
        "environment/Dockerfile": b"FROM python:3.12-slim\nWORKDIR /opt/dependencies\nWORKDIR /workspace\n",
        "environment/workspace/checker.txt": b"original",
        "solution/solve.sh": b"echo correct > answer.txt\n",
        "tests/test.sh": b"python3 /tests/check.py\nexit 1\n",
        "tests/check.py": (
            b"""import json
from pathlib import Path
answer = Path('answer.txt')
checker = Path('checker.txt')
rewards = {
    'answer': float(answer.exists() and answer.read_text().strip() == 'correct'),
    'checker': float(checker.exists() and checker.read_text() == 'original'),
}
Path('/logs/verifier/reward.json').write_text(json.dumps(rewards))
"""
        ),
        "task.toml": (
            b"""schema_version = "1.3"
artifacts = ["/workspace/answer.txt"]
[task]
name = "skill2env/fixture"
[agent]
network_mode = "no-network"
[verifier]
network_mode = "no-network"
timeout_sec = 20
[environment]
network_mode = "no-network"
cpus = 2
memory_mb = 4096
storage_mb = 4096
"""
        ),
    }
    files["environment/Dockerfile"] = files["environment/Dockerfile"].replace(
        b"WORKDIR /workspace", f"WORKDIR {working_directory}".encode()
    )
    files[f"environment{working_directory}/checker.txt"] = files.pop("environment/workspace/checker.txt")
    files["task.toml"] = (
        files["task.toml"]
        .replace(b"skill2env/fixture", f"skill2env/{name}".encode())
        .replace(b"/workspace/answer.txt", f"{working_directory}/answer.txt".encode())
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path}\0{hashlib.sha256(files[path]).hexdigest()}\n".encode())
    return RawRow(
        "fixture",
        Source(
            dataset="skill2env/skill2env",
            revision=RELEASE.split("@")[1],
            row=f"tasks/{digest.hexdigest()}.tar.gz:0",
            importer_revision="fixture",
        ),
        {
            "release": RELEASE,
            "task_ref": f"skill2env/{name}@sha256:{digest.hexdigest()}",
            "files": files,
            "modes": {path: "755" if path.endswith(".sh") else "644" for path in files},
        },
    )


def write_selection(root: Path, selection) -> None:
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    refs = [row.data["task_ref"] for row in selection]
    (root / "release.json").write_text(json.dumps({"release": RELEASE, "selected": refs, "task_refs": refs}))
    for row in selection:
        digest = hashlib.sha256(row.data["task_ref"].encode()).hexdigest()
        with tarfile.open(root / "tasks" / f"{digest}.tar.gz", "w:gz") as archive:
            for name, blob in row.data["files"].items():
                member = tarfile.TarInfo(name)
                member.size = len(blob)
                member.mode = int(row.data["modes"][name], 8)
                archive.addfile(member, io.BytesIO(blob))


def test_prepared_selections_keep_distinct_stable_task_ids(tmp_path):
    config = sources()[0].config
    assert config is not None
    recipe = source_recipe(config, {}, None)
    context = ConversionContext({}, None)
    first, second = exported_row("first"), exported_row("second")
    selections = ((first,), (second,), (second, first))
    identities = []
    for index, selection in enumerate(selections):
        root = tmp_path / str(index)
        (root / "tasks").mkdir(parents=True)
        write_selection(root, selection)
        tasks = {}
        for shard in conversion_shards(str(root), recipe.source):
            for record in staged_file_rows(str(root), shard, recipe.source, context):
                converted = convert_record(record, recipe)
                assert isinstance(converted.result, NormalizedTask)
                tasks[record["data"]["task_ref"]] = converted.result.task.id
        assert len(tasks) == len(selection)
        identities.append(tasks)
    first_ids, second_ids, combined_ids = identities
    assert len(combined_ids) == len(set(combined_ids.values())) == 2
    assert combined_ids == {**first_ids, **second_ids}


def test_selected_archive_tampering_is_rejected(tmp_path):
    first = exported_row("first")
    root = tmp_path
    write_selection(root, [first])
    config = sources()[0].config
    recipe = source_recipe(config, {}, None)
    context = ConversionContext({}, None)
    # Replacing an archive under its original immutable filename must fail before conversion.
    payload = next((root / "tasks").glob("*.tar.gz"))
    with tarfile.open(payload, "w:gz") as archive:
        for name, blob in first.data["files"].items():
            if name == "instruction.md":
                blob = b"Changed instruction"
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
    with pytest.raises(ValueError, match="Archive contents differ"):
        for shard in conversion_shards(str(root), recipe.source):
            list(staged_file_rows(str(root), shard, recipe.source, context))


@pytest.mark.parametrize("wrong_release", [False, True])
def test_reader_honors_replaced_selection(tmp_path, wrong_release):
    first, second = exported_row("first"), exported_row("second")
    write_selection(tmp_path, [first])
    write_selection(tmp_path, [second])
    recipe = source_recipe(sources()[0].config, {}, None)
    context = ConversionContext({}, None)

    def records():
        return [
            record
            for shard in conversion_shards(str(tmp_path), recipe.source)
            for record in staged_file_rows(str(tmp_path), shard, recipe.source, context)
        ]

    if wrong_release:
        manifest_path = tmp_path / "release.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["release"] = "another/release@sha256:other"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="another Skill2Env release"):
            records()
    else:
        assert [record["data"]["task_ref"] for record in records()] == [second.data["task_ref"]]


def test_conversion_retains_hidden_files_and_detects_modified_exports():
    row = exported_row()
    result = convert_row(row, convert_task, ConversionContext({}, None))
    assert isinstance(result, NormalizedTask)
    assert next(file for file in result.task.resources.verifier if file.path == "test.sh").mode == "755"
    restored = TaskSpec.model_validate_json(result.task.model_dump_json())
    assert (
        resource_bytes(next(file for file in restored.resources.verifier if file.path == "test.sh"))
        == b"python3 /tests/check.py\nexit 1\n"
    )
    assert not restored.resources.all and not restored.resources.worker
    assert [file.path for file in restored.resources.oracle] == ["solution/solve.sh"]
    assert restored.source == row.source
    contract = next(resource for resource in restored.resources.verifier if resource.path == "curation-contract.json")
    assert json.loads(resource_bytes(contract))["task_ref"] == row.data["task_ref"]
    changed = {**row.data, "files": {**row.data["files"], "tests/test.sh": b"exit 0"}}
    with pytest.raises(ValueError, match="differ from the pinned task"):
        convert_task(replace(row, data=changed), ConversionContext({}, None))


def test_unbuilt_export_preserves_contexts_and_private_roles_but_cannot_execute():
    row = exported_row()
    result = convert_task(row, ConversionContext({}, None))
    assert isinstance(result, TaskSpec)
    task = TaskSpec.model_validate_json(result.model_dump_json())
    assert isinstance(task.grader, ScriptGrader) and task.grader.environment is not None
    actor = task.environment_requirements
    grader = task.grader.environment
    assert isinstance(actor.docker_build, DockerBuildContext)
    assert isinstance(grader.docker_build, DockerBuildContext)
    actor_files = {resource.path: resource_bytes(resource) for resource in actor.docker_build.files}
    grader_files = {resource.path: resource_bytes(resource) for resource in grader.docker_build.files}
    assert actor_files == {
        "Dockerfile": b"FROM python:3.12-slim\nWORKDIR /opt/dependencies\nWORKDIR /workspace\n",
        "workspace/checker.txt": b"original",
    }
    assert grader_files["workspace/checker.txt"] == b"original"
    assert grader_files["Dockerfile"] == (
        actor_files["Dockerfile"] + b"USER root\nRUN rm -rf /workspace && mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    assert not task.resources.all and not task.resources.worker
    assert {resource.path for resource in task.resources.oracle} == {"solution/solve.sh"}
    assert {"test.sh", "check.py"} <= {resource.path for resource in task.resources.verifier}
    # An inspection artifact must never silently execute in a substitute host environment.
    for environment in (actor, grader):
        with pytest.raises(UnsupportedMachineSpec, match="Unresolved Docker build context"):
            require_resolved_environment(environment)


@pytest.mark.parametrize(
    "files,components,score",
    [
        ({"answer.txt": b"correct", "checker.txt": b"original"}, {"answer": 1.0, "checker": 1.0}, 1.0),
        ({"answer.txt": b"correct", "checker.txt": b"changed"}, {"answer": 1.0, "checker": 0.0}, 0.5),
        ({"answer.txt": b"correct"}, {"answer": 1.0, "checker": 0.0}, 0.5),
        ({}, {"answer": 0.0, "checker": 0.0}, 0.0),
    ],
)
@pytest.mark.parametrize("working_directory", ["/workspace", "/workspace/fleet", "/workspace/fleet records", "/work"])
@pytest.mark.asyncio
async def test_native_script_grades_captured_workspace_and_preserves_components(
    files, components, score, working_directory, monkeypatch
):
    row = exported_row(working_directory=working_directory)
    if working_directory in ("/workspace/fleet records", "/work"):
        monkeypatch.setitem(CAPTURE_ROOTS, row.data["task_ref"].split("@")[1], (working_directory,))
    task = convert_task(row, ConversionContext({}, None))
    assert isinstance(task, TaskSpec) and isinstance(task.grader, ScriptGrader)
    # The fixture uses only stdlib; use a real local sandbox at the machine boundary.
    environment = task.environment_requirements.model_copy(update={"docker_build": None, "package_spec": PackageSpec()})
    assert task.grader.environment is not None
    grader_environment = task.grader.environment.model_copy(update={"docker_build": None, "package_spec": PackageSpec()})
    task = task.model_copy(
        update={
            "environment_requirements": environment,
            "grader": task.grader.model_copy(update={"environment": grader_environment}),
        }
    )
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done"))),
        {f"{working_directory}/{path}": blob for path, blob in files.items()},
    )
    runtime = Path(sys.prefix)
    try:
        factory = LocalMachineFactory(read_only=(runtime,), bin_dirs=(runtime / "bin",))
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    result = await grade_in_sandbox(task, attempt, factory, MachineSpec(HostImage(), network=NetworkPolicy.DENY))
    assert result.status == Outcome.GRADED, result
    assert (result.reward, result.passed, result.detail) == (score, None, {"native_rewards": components})
