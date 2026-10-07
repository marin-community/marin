# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise directory evidence through real filesystem reads and private grading upload."""

import asyncio
import json
from dataclasses import dataclass

import pytest
from shellbox.machine import Backend, DockerImage, ExitReason, MachineSpec, Result, UnsupportedMachineSpec

from taskcompendium.models import OutputDirectory, TaskSpec
from taskcompendium.runtime.grading import grade_submission
from taskcompendium.runtime.shell import ShellEnvironment, ShellFactory

from . import test_executable_ingestion
from .test_executable_ingestion import GradingMachines
from .test_runtime import FileMachine, FileMachines

executable_row = test_executable_ingestion.executable_row
executable_task = test_executable_ingestion.executable_task


@dataclass
class DirectoryMachine(FileMachine):
    """Run only the trusted evidence reader against a temporary local workspace."""

    async def run(self, command):
        process = await asyncio.create_subprocess_exec(
            *command.argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        limit = command.output_limit_bytes
        return Result(process.returncode, stdout[:limit], stderr[:limit], len(stdout) > limit, False, ExitReason.EXITED)


@pytest.fixture
def directory_task(executable_task, tmp_path):
    parameters = json.loads(executable_task.verifier.parameters_json)
    parameters["workspace"] = str(tmp_path)
    task = executable_task.model_copy(
        update={
            "output_paths": (str(tmp_path / "canonical.yml"),),
            "environment_requirements": executable_task.environment_requirements.model_copy(
                update={"capabilities": ("shell", "filesystem", "python3")}
            ),
            "output_directories": (
                OutputDirectory(root=str(tmp_path), patterns=("*.yml", "*.yaml"), max_files=8, max_bytes=1024),
            ),
            "verifier": executable_task.verifier.model_copy(update={"parameters_json": json.dumps(parameters)}),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


async def test_directory_evidence_round_trip_preserves_valid_names_and_skips_links(directory_task, tmp_path):
    nested = tmp_path / "nested:dir"
    nested.mkdir()
    workflow = nested / "ci:\nλ\\file.yml"
    workflow.write_bytes(b"jobs: {}\n")
    (tmp_path / "ignored.txt").write_bytes(b"not submitted")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "private.yml").write_bytes(b"private")
    (tmp_path / "linked-directory").symlink_to(outside, target_is_directory=True)
    (tmp_path / "linked.yml").symlink_to(outside / "private.yml")
    # Named files retain existing symlink-following behavior; directory discovery
    # follows find -P and excludes symbolic links.
    canonical = tmp_path / "canonical.yml"
    canonical.symlink_to(workflow)
    environment = ShellEnvironment(
        DirectoryMachine(), directory_task.output_paths, 10, 1024, directory_task.output_directories
    )
    evidence = await environment.evidence()
    assert evidence.files == {str(canonical): b"jobs: {}\n", str(workflow): b"jobs: {}\n"}
    machines = GradingMachines()
    await grade_submission(
        directory_task,
        {**evidence.files, "/tests/reference.yml": b"tampered", "/solution/secret.yml": b"tampered"},
        machines,
        machine_spec=MachineSpec(DockerImage(directory_task.verifier.environment_requirements.docker_image)),
    )
    uploaded = machines.machines[0].files
    assert uploaded[str(workflow)] == b"jobs: {}\n"
    assert "/tests/reference.yml" not in uploaded and "/solution/secret.yml" not in uploaded


@pytest.mark.parametrize("budget", [{"max_files": 1}, {"max_bytes": 3}])
async def test_directory_over_budget_never_returns_partial_evidence(directory_task, tmp_path, budget):
    (tmp_path / "first.yml").write_bytes(b"aa")
    (tmp_path / "second.yml").write_bytes(b"bb")
    selection = directory_task.output_directories[0].model_copy(update=budget)
    environment = ShellEnvironment(DirectoryMachine(), (), 10, 1024, (selection,))
    with pytest.raises(RuntimeError, match="Directory capture unavailable"):
        await environment.evidence()
    # Oversized captured evidence is also rejected before starting a private
    # grader, even when supplied by a caller other than the shell runtime.
    machines = GradingMachines()
    task = directory_task.model_copy(update={"output_directories": (selection,)})
    result = await grade_submission(
        task,
        {str(tmp_path / "first.yml"): b"aa", str(tmp_path / "second.yml"): b"bb"},
        machines,
        machine_spec=MachineSpec(DockerImage(task.verifier.environment_requirements.docker_image)),
    )
    assert result.status == "infra_error" and not machines.machines
    (tmp_path / "second.yml").unlink()
    assert (await environment.evidence()).files == {str(tmp_path / "first.yml"): b"aa"}


@pytest.mark.parametrize("root", ["/tests", "/logs", "/solution", "/unrelated"])
async def test_directory_private_or_outside_root_rejected_before_capture_and_upload(executable_task, root):
    task = executable_task.model_copy(
        update={"output_directories": (OutputDirectory(root=root, patterns=("*.yml",), max_files=2, max_bytes=100),)}
    )
    machines = FileMachines()
    with pytest.raises(ValueError, match=r"private mounts|workspace"):
        await ShellFactory(machines, MachineSpec(DockerImage("test")), {}, 1, 1024).create(task)
    with pytest.raises(ValueError, match=r"private mounts|workspace"):
        await grade_submission(task, {root + "/file.yml": b"x"}, machines, machine_spec=MachineSpec(DockerImage("test")))
    assert not machines.machines


async def test_directory_traversal_candidate_cannot_overwrite_private_grader(directory_task, tmp_path):
    machines = GradingMachines()
    with pytest.raises(ValueError, match="normalized"):
        await grade_submission(
            directory_task,
            {str(tmp_path) + "/../tests/reference.yml": b"tampered"},
            machines,
            machine_spec=MachineSpec(DockerImage(directory_task.verifier.environment_requirements.docker_image)),
        )
    assert not machines.machines


async def test_directory_order_matches_original_find_discovery(directory_task, tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "workflow-later.yml").write_bytes(b"later")
    (tmp_path / "nested" / "ci-first.yml").write_bytes(b"first")
    process = await asyncio.create_subprocess_exec(
        "find",
        str(tmp_path),
        "-type",
        "f",
        "(",
        "-name",
        "*.yml",
        "-o",
        "-name",
        "*.yaml",
        ")",
        "(",
        "-path",
        "*/.github/workflows/*",
        "-o",
        "-name",
        "workflow*",
        "-o",
        "-name",
        "ci*",
        "-o",
        "-name",
        "main*",
        "-o",
        "-name",
        "test*",
        ")",
        "-print0",
        stdout=asyncio.subprocess.PIPE,
    )
    found, _ = await process.communicate()
    environment = ShellEnvironment(DirectoryMachine(), (), 10, 1024, directory_task.output_directories)
    assert list((await environment.evidence()).files) == [path.decode() for path in found.split(b"\0") if path]


async def test_directory_capture_cannot_claim_shellsim_python_support(directory_task):
    machines = FileMachines()
    machines.backend = Backend.SHELLSIM
    task = directory_task.model_copy(
        update={
            "environment_requirements": directory_task.environment_requirements.model_copy(
                update={"docker_image": None, "compatible_backends": (Backend.SHELLSIM,)}
            )
        }
    )
    with pytest.raises(UnsupportedMachineSpec, match="POSIX Python 3"):
        await ShellFactory(machines, MachineSpec(DockerImage("test")), {}, 1, 1024).create(task)
    assert not machines.machines


async def test_directory_capture_python_free_image_fails_explicitly_and_closes(directory_task, monkeypatch):
    async def missing_interpreter(self, command):
        return Result(127, b"", b"python3: not found", False, False, ExitReason.EXITED)

    # The external command boundary reports a missing interpreter; no capture or
    # actor work should proceed after this runtime capability check fails.
    monkeypatch.setattr(FileMachine, "run", missing_interpreter)
    machines = FileMachines()
    image = directory_task.environment_requirements.docker_image
    with pytest.raises(UnsupportedMachineSpec, match="POSIX Python 3"):
        await ShellFactory(machines, MachineSpec(DockerImage(image)), {}, 1, 1024).create(directory_task)
    assert machines.machines[0].closed
