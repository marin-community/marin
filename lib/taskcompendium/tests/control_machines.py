# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control submissions check a grader through the grading path that rollouts use."""

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    Machine,
    MachineSpec,
    Result,
)
from verifyit.spec import SchemaFormat

from taskcompendium.convert.answers import (
    json_schema_task,
    numeric_answer_task,
)
from taskcompendium.models import (
    AnswerType,
    ResourceGroups,
    Source,
    TaskSpec,
)
from taskcompendium.pipeline.controls import answer_reply, control_suite, reference_reply
from taskcompendium.pipeline.models import CheckStatus, Controls, RawRow, Reply
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import (
    FixtureGradingMachines,
    ShellSimImages,
    script_graded,
)

PASS, FAIL, DEFECT = CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.DEFECT
ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
REFERENCE_CONTROLS = Controls(golden=reference_reply)
AGENT_IMAGE = "agent@sha256:" + "a" * 64


@dataclass(eq=False)
class ImageMachine:
    """A ShellSim machine standing in for ``image``; it records the paths uploaded to it."""

    machine: Machine
    image: str
    uploads: list[str] = field(default_factory=list)

    async def run(self, command: Command) -> Result:
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        if source.suffix == ".tar":
            with tarfile.open(source) as archive:
                self.uploads.extend("/" + member.name for member in archive if member.isfile())
        else:
            self.uploads.append(target)
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass
class RecordingImages(DockerMachineFactory):
    """Fresh ShellSim machines, each recording the image it stands in for."""

    backend = Backend.DOCKER
    created: list[ImageMachine] = field(default_factory=list)

    async def create(self, spec: MachineSpec) -> ImageMachine:
        assert isinstance(spec.source, DockerImage)
        machine = ImageMachine(await ShellSimImages().create(spec), spec.source.reference)
        self.created.append(machine)
        return machine


def checks(task: TaskSpec, controls: Controls, machines: FixtureGradingMachines | None = None) -> dict[str, CheckStatus]:
    return {check.check: check.status for check in control_suite(controls, machines).run(task).checks}


def in_process_task(kind: str) -> TaskSpec:
    if kind == "numeric":
        task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    else:
        task = json_schema_task(
            ROW, prompt="Return a JSON object.", schema=json.dumps({"type": "object"}), schema_format=SchemaFormat.JSON
        )
    assert isinstance(task, TaskSpec)
    return task


def answer_task() -> TaskSpec:
    """A conversation answer graded in its image by comparing /app/answer.txt with the public input."""
    task = in_process_task("numeric").model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("data/expected.txt", b"7"),))}
    )
    return script_graded(task, b'test "$(cat answer.txt)" = 7\n')


def file_task() -> TaskSpec:
    """A file answer at /app/solution.txt, with an oracle script that derives it from a worker file."""
    task = in_process_task("numeric").model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "output_paths": ("/app/solution.txt",),
            "resources": ResourceGroups(
                worker=(inline_resource("data/expected.txt", b"7"),),
                oracle=(
                    inline_resource("solution/solve.sh", b"mkdir -p /app; cp /data/expected.txt /app/solution.txt\n"),
                ),
            ),
        }
    )
    return script_graded(task, b'test "$(cat solution.txt)" = 7\n', answer_path=None)


def wrong_reply(task: TaskSpec) -> Reply:
    return answer_reply(task, "__incorrect_answer__")
