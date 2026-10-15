# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable
from shellbox.machine import HostImage, MachineSpec, NetworkPolicy, PackageSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    CommandSemantics,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.skill2env.conversion import RELEASE, convert_task
from experiments.post_training.task_curation.datasets.skill2env.grade import component_reward

IMAGE = "fixture@sha256:" + "0" * 64


def exported_row() -> RawRow:
    files = {
        "instruction.md": b"Repair the answer and keep the checker intact.",
        "environment/Dockerfile": b"FROM python:3.12-slim\nWORKDIR /workspace\n",
        "environment/workspace/checker.txt": b"original",
        "solution/solve.sh": b"echo correct > /workspace/answer.txt\n",
        "tests/test.sh": b"python3 /tests/check.py\nexit 1\n",
        "tests/check.py": (
            b"""import json
from pathlib import Path
answer = Path('/workspace/answer.txt')
checker = Path('/workspace/checker.txt')
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
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path}\0{hashlib.sha256(files[path]).hexdigest()}\n".encode())
    return RawRow(
        "fixture",
        Source(
            dataset="skill2env/skill2env",
            revision=RELEASE.split("@")[1],
            row="tasks.jsonl:0",
            importer_revision="fixture",
        ),
        {
            "release": RELEASE,
            "task_ref": f"skill2env/fixture@sha256:{digest.hexdigest()}",
            "files": {path: base64.b64encode(blob).decode() for path, blob in files.items()},
            "modes": {path: "644" for path in files},
            "image": IMAGE,
            "grader_image": IMAGE,
        },
    )


def test_conversion_retains_hidden_files_and_detects_modified_exports():
    row = exported_row()
    task = convert_task(row, ConversionContext({}, None))
    assert isinstance(task, TaskSpec)
    restored = TaskSpec.model_validate_json(task.model_dump_json())
    assert (
        resource_bytes(next(file for file in restored.resources.verifier if file.path == "test.sh"))
        == b"python3 /tests/check.py\nexit 1\n"
    )
    assert not restored.resources.all and not restored.resources.worker
    assert [file.path for file in restored.resources.oracle] == ["solution/solve.sh"]
    assert restored.source.row == row.data["task_ref"]
    changed = {**row.data, "files": {**row.data["files"], "tests/test.sh": base64.b64encode(b"exit 0").decode()}}
    with pytest.raises(ValueError, match="differ from the pinned task"):
        convert_task(replace(row, data=changed), ConversionContext({}, None))


def test_unbuilt_export_is_deferred_instead_of_assuming_a_host_environment():
    row = exported_row()
    result = convert_task(
        replace(row, data={key: value for key, value in row.data.items() if key != "image"}), ConversionContext({}, None)
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "native_images_unbuilt"


@pytest.mark.parametrize(
    "files,components,score,passed",
    [
        ({"answer.txt": b"correct", "checker.txt": b"original"}, {"answer": 1.0, "checker": 1.0}, 1.0, True),
        ({"answer.txt": b"correct", "checker.txt": b"changed"}, {"answer": 1.0, "checker": 0.0}, 0.5, False),
        ({"answer.txt": b"correct"}, {"answer": 1.0, "checker": 0.0}, 0.5, False),
        ({}, {"answer": 0.0, "checker": 0.0}, 0.0, False),
    ],
)
@pytest.mark.asyncio
async def test_native_script_grades_captured_workspace_and_preserves_components(files, components, score, passed):
    task = convert_task(exported_row(), ConversionContext({}, None))
    assert isinstance(task, TaskSpec) and isinstance(task.grader, ScriptGrader)
    # The fixture uses only stdlib; use a real local sandbox at the machine boundary.
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, package_spec=PackageSpec(), working_directory="/workspace"
    )
    task = task.model_copy(
        update={
            "environment_requirements": environment,
            "grader": task.grader.model_copy(update={"environment": environment}),
        }
    )
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done"))),
        {f"/workspace/{path}": blob for path, blob in files.items()},
    )
    runtime = Path(sys.prefix)
    try:
        factory = LocalMachineFactory(read_only=(runtime,), bin_dirs=(runtime / "bin",))
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    result = await grade_in_sandbox(task, attempt, factory, MachineSpec(HostImage(), network=NetworkPolicy.DENY))
    assert result.status == Outcome.GRADED, result
    assert (result.reward, result.passed, result.detail) == (score, passed, {"native_rewards": components})


def test_component_full_pass_is_not_inferred_from_mean():
    result = component_reward({"one": 1.0, "two": 1.0, "three": 1.0, "four": 1 - 2e-9})
    assert result["reward"] > 1 - 1e-9
    assert result["passed"] is False
