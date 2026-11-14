# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a declared source OracleCommand to its unchanged task and runtime."""

import hashlib
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import AnswerType, TaskSpec
from taskcompendium.pipeline.fingerprints import callable_identity
from taskcompendium.pipeline.models import ControlSubmission, OracleCommand
from taskcompendium.runtime.resources import has_oracle_solution, resource_bytes

ORACLE_COMMAND_FILE = "oracle-command.json"


def oracle_resource_identity(task: TaskSpec) -> str:
    """Hash actual resource bytes and metadata across every task role."""
    return canonical_sha256(
        {
            role: [
                {
                    **resource.model_dump(mode="json", exclude={"source"}),
                    "sha256": hashlib.sha256(resource_bytes(resource)).hexdigest(),
                }
                for resource in resources
            ]
            for role, resources in (
                ("all", task.resources.all),
                ("worker", task.resources.worker),
                ("oracle", task.resources.oracle),
                ("verifier", task.resources.verifier),
            )
        }
    )


class OracleCommandBinding(BaseModel):
    """A source-declared command and the frozen inputs that justify executing it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resources_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    control_identity: dict[str, JsonValue]
    submission: OracleCommand

    def require_task(self, task: TaskSpec) -> None:
        if not has_oracle_solution(task):
            raise ValueError("An original OracleCommand requires preserved source solution resources")
        if task.answer_type != AnswerType.TEXT or task.final_tools:
            raise ValueError("This OracleCommand delivery requires a text answer without final tool calls")
        if self.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError("OracleCommand belongs to another TaskSpec")
        if self.resources_sha256 != oracle_resource_identity(task):
            raise ValueError("OracleCommand source resource bytes changed")
        if not self.submission.command.strip() or self.submission.answer_file is None:
            raise ValueError("A conversation OracleCommand requires its declared command and answer file")


def read_oracle_command(directory: Path) -> OracleCommandBinding | None:
    path = directory / ORACLE_COMMAND_FILE
    return OracleCommandBinding.model_validate_json(path.read_text()) if path.is_file() else None


def write_oracle_command(
    task: TaskSpec,
    runtime: dict,
    golden: Callable[[TaskSpec], ControlSubmission | None],
    directory: Path,
    *,
    source_revision: str,
) -> bool:
    """Write a conversation command when the declared source control supplies one."""
    submission = golden(task)
    if not isinstance(submission, OracleCommand) or submission.answer_file is None:
        return False
    binding = OracleCommandBinding(
        task_sha256=canonical_sha256(task.model_dump(mode="json")),
        runtime_sha256=canonical_sha256(runtime),
        resources_sha256=oracle_resource_identity(task),
        source_revision=source_revision,
        control_identity=callable_identity(golden),
        submission=submission,
    )
    binding.require_task(task)
    with (directory / ORACLE_COMMAND_FILE).open("x") as stream:
        stream.write(binding.model_dump_json(indent=2) + "\n")
    return True
