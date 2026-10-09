# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise directory evidence through real filesystem reads and grader upload."""

import asyncio
import io
import json
import tarfile
from dataclasses import dataclass, field

import pytest
from shellbox.machine import Backend, DockerImage, ExitReason, MachineSpec, Result, UnsupportedMachineSpec
from verifyit.spec import StdioSpec

from taskcompendium.convert.environment import grading_environment
from taskcompendium.convert.executable import SOLUTION_PATHS, workspace_task
from taskcompendium.models import GradingAttempt, OutputDirectory, Source, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellEnvironment, ShellFactory

from .test_runtime import EXITED, FileMachine, FileMachines, finished

IMAGE = "test@sha256:" + "a" * 64


@pytest.fixture
def executable_task():
    row = RawRow("program-1", Source(dataset="test/program", revision="1", row="1", importer_revision="1"), {})
    return workspace_task(
        row,
        instruction="Read two integers and print their sum in /app/solution.py.",
        spec=StdioSpec(command="python3 /app/solution.py"),
        environment=grading_environment(IMAGE),
        grader_environment=grading_environment(IMAGE),
        output_paths=SOLUTION_PATHS,
        verifier=(inline_resource("cases/input_1.txt", b"3 4\n"), inline_resource("cases/output_1.txt", b"7\n")),
        worker=(inline_resource("setup_files/readme.txt", b"Public setup"),),
    )


@dataclass
class GradingMachine(FileMachine):
    """A grader machine that unpacks the staged archive and writes a fixed verdict."""

    async def run(self, command):
        if command.argv[0] == "tar":
            with tarfile.open(fileobj=io.BytesIO(self.files[command.argv[2]])) as archive:
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    self.files["/" + member.name] = stream.read()
            return EXITED
        if command.argv[0] != "python3":
            return await super().run(command)
        self.files["/logs/verifier/verdict.json"] = json.dumps(
            {"status": "scored", "reward": 0.0, "detail": {}}
        ).encode()
        return EXITED


@dataclass
class GradingMachines:
    backend = Backend.DOCKER
    machines: list[GradingMachine] = field(default_factory=list)

    async def create(self, spec):
        machine = GradingMachine()
        self.machines.append(machine)
        return machine


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


def grading_machine(task: TaskSpec) -> MachineSpec:
    """A machine from the image the task's verifyit grader declares."""
    grader = task.grader
    assert isinstance(grader, VerifyitGrader) and grader.environment is not None
    assert grader.environment.docker_image is not None
    return MachineSpec(DockerImage(grader.environment.docker_image))


@pytest.fixture
def directory_task(executable_task, tmp_path):
    grader = executable_task.grader
    assert isinstance(grader, VerifyitGrader)
    task = executable_task.model_copy(
        update={
            "output_paths": (str(tmp_path / "canonical.yml"),),
            "environment_requirements": executable_task.environment_requirements.model_copy(
                update={"capabilities": ("shell", "filesystem", "python3")}
            ),
            "output_directories": (
                OutputDirectory(root=str(tmp_path), patterns=("*.yml", "*.yaml"), max_files=8, max_bytes=1024),
            ),
            "grader": grader.model_copy(update={"parameters": {**grader.parameters, "workspace": str(tmp_path)}}),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


@pytest.mark.asyncio
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
    files = {**evidence.files, "/tests/reference.yml": b"tampered", "/solution/secret.yml": b"tampered"}
    await grade_in_sandbox(
        directory_task, GradingAttempt(finished(directory_task), files), machines, grading_machine(directory_task)
    )
    uploaded = machines.machines[0].files
    assert uploaded[str(workflow)] == b"jobs: {}\n"
    assert "/tests/reference.yml" not in uploaded and "/solution/secret.yml" not in uploaded


@pytest.mark.parametrize("budget", [{"max_files": 1}, {"max_bytes": 3}])
@pytest.mark.asyncio
async def test_directory_over_budget_never_returns_partial_evidence(directory_task, tmp_path, budget):
    (tmp_path / "first.yml").write_bytes(b"aa")
    (tmp_path / "second.yml").write_bytes(b"bb")
    selection = directory_task.output_directories[0].model_copy(update=budget)
    environment = ShellEnvironment(DirectoryMachine(), (), 10, 1024, (selection,))
    with pytest.raises(RuntimeError, match="Directory capture unavailable"):
        await environment.evidence()
    # Oversized captured evidence is also rejected before starting a
    # grader, even when supplied by a caller other than the shell runtime.
    machines = GradingMachines()
    task = directory_task.model_copy(update={"output_directories": (selection,)})
    files = {str(tmp_path / "first.yml"): b"aa", str(tmp_path / "second.yml"): b"bb"}
    with pytest.raises(RuntimeError, match="exceeds its budget"):
        await grade_in_sandbox(task, GradingAttempt(finished(task), files), machines, grading_machine(task))
    assert not machines.machines
    (tmp_path / "second.yml").unlink()
    assert (await environment.evidence()).files == {str(tmp_path / "first.yml"): b"aa"}


@pytest.mark.parametrize("root", ["/tests", "/logs", "/solution", "/unrelated"])
@pytest.mark.asyncio
async def test_directory_grader_mount_or_outside_root_rejected_before_capture_and_upload(executable_task, root):
    task = executable_task.model_copy(
        update={"output_directories": (OutputDirectory(root=root, patterns=("*.yml",), max_files=2, max_bytes=100),)}
    )
    machines = FileMachines()
    with pytest.raises(ValueError, match=r"private mounts|workspace"):
        await ShellFactory(machines, MachineSpec(DockerImage("test")), {}, 1, 1024).create(task)
    with pytest.raises(ValueError, match=r"private mounts|workspace"):
        await grade_in_sandbox(
            task, GradingAttempt(finished(task), {root + "/file.yml": b"x"}), machines, grading_machine(task)
        )
    assert not machines.machines


@pytest.mark.asyncio
async def test_directory_traversal_candidate_cannot_overwrite_grader_files(directory_task, tmp_path):
    machines = GradingMachines()
    files = {str(tmp_path) + "/../tests/reference.yml": b"tampered"}
    with pytest.raises(ValueError, match="normalized"):
        await grade_in_sandbox(
            directory_task, GradingAttempt(finished(directory_task), files), machines, grading_machine(directory_task)
        )
    assert not machines.machines


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
