# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the self-contained TaskTrove natural-language shell family."""

import tomllib
from typing import Any

from tasktrove_verify.spec import ScriptSpec

from taskcompendium.importers.tasktrove import TaskArchive, semantic_verifier
from taskcompendium.models import (
    AnswerRequirements,
    ContainerRuntime,
    Embedded,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    ShellSimEnvironment,
    TaskMetadata,
    TaskSpecification,
    relative_path,
)

IMPORTER_REVISION = "taskcompendium-tasktrove-shell-v0.1"
FAMILY = "shell-cmd"
CONVERTER = "nl2bash"
MODE = "script"
WORKDIR = "/workspace"
OUTPUT_PATH = "/output/command_capture.txt"
CHECKER = "nl2bash_check.py"
EXPECTED = "nl2bash_expected.json"


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    metadata = tomllib.loads(raw.decode()).get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("task metadata is not a table")
    return metadata


def _resource(path: str, roles: tuple[ResourceRole, ...], data: bytes, *, executable: bool = False) -> Resource:
    relative_path(path)
    return Resource(path, roles, Embedded(data), executable=executable)


def _resources(archive: TaskArchive) -> tuple[Resource, ...]:
    resources: list[Resource] = []
    for path, data in archive.files.items():
        if path.startswith("setup_files/"):
            resources.append(_resource(path, (ResourceRole.AGENT,), data, executable=path.endswith(".sh")))
        elif path.startswith("tests/"):
            resources.append(
                _resource(path.removeprefix("tests/"), (ResourceRole.VERIFIER,), data, executable=path.endswith(".sh"))
            )
    if CHECKER not in {resource.path for resource in resources if ResourceRole.VERIFIER in resource.roles}:
        raise ValueError(f"missing {CHECKER}")
    if EXPECTED not in {resource.path for resource in resources if ResourceRole.VERIFIER in resource.roles}:
        raise ValueError(f"missing {EXPECTED}")
    return tuple(resources)


def _validate_verifier(archive: TaskArchive, verifier: Any) -> None:
    if not isinstance(verifier, ScriptSpec):
        raise ValueError("shell family does not have a script verifier")
    if verifier.path != CHECKER:
        raise ValueError("shell verifier must use the pinned checker")
    if verifier.args != (OUTPUT_PATH,):
        raise ValueError("shell verifier must grade command_capture.txt")
    if verifier.workspace != WORKDIR:
        raise ValueError("shell verifier must run from /workspace")


def _instructions(instructions: str) -> str:
    lines = instructions.splitlines()
    lines = [
        (
            "**TERMINAL ENVIRONMENT**: ShellSim with a persistent shell and virtual filesystem."
            if line.startswith("**TERMINAL ENVIRONMENT**:")
            else line.replace(
                "Do not delete any helper files generated during execution; the verifier inspects them.",
                "Do not delete any helper files generated during execution.",
            )
        )
        for line in lines
    ]
    # This fixed source template ends with an incidental Ubuntu tool inventory.
    # The task's goal, setup script, and output requirements remain intact.
    return "\n".join(lines).split("\n## Available Tools\n", 1)[0].strip()


def import_task(
    archive: TaskArchive, *, verifier_runtime: ContainerRuntime | None = None
) -> TaskSpecification | Rejected:
    """Use ShellSim for the task world and a pinned Docker image for its original checker."""
    source = archive.source
    if archive.family != FAMILY:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported shell family {archive.family!r}")
    try:
        metadata = _metadata(archive)
        if metadata.get("converter") != CONVERTER or metadata.get("mode") != MODE:
            raise LookupError("unsupported shell converter")
        if metadata.get("language") != "bash":
            raise LookupError("shell converter is not bash")
        verifier = semantic_verifier(archive.verifier)
        _validate_verifier(archive, archive.verifier)
        resources = _resources(archive)
        if verifier_runtime is None:
            raise LookupError("an immutable verifier runtime must be supplied by the caller")
        environment = ShellSimEnvironment(
            workdir=WORKDIR,
            setup_commands=("cp -a /workspace/setup_files /setup_files",),
            additional_directories=("/output",),
        )
    except LookupError as error:
        return Rejected(source, RejectionReason.UNSUPPORTED_ENVIRONMENT, str(error))
    except (KeyError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    tags = metadata.get("tags", ())
    competencies = tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()
    return TaskSpecification(
        id=f"tasktrove-{source.row}",
        instructions=_instructions(archive.instructions),
        environment=environment,
        resources=resources,
        verifier=verifier,
        verifier_runtime=verifier_runtime,
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="environment-modification"),
        answer_requirements=AnswerRequirements("final_state"),
    )
