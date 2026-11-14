# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind original deployment choices to an unchanged Harbor TaskSpec packet."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.machine import NetworkPolicy

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

SOURCE_RUNTIME_FILE = "source-runtime.json"


class TaskSpecRuntimeBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: dict[str, Any]

    @model_validator(mode="after")
    def validate_runtime(self):
        if canonical_sha256(self.runtime) != self.runtime_sha256:
            raise ValueError("Source runtime differs from its declared hash")
        for role, machine in self.runtime.items():
            if isinstance(machine, dict) and "network" in machine and role not in {"task_machine", "verifier_machine"}:
                raise ValueError(f"Unsupported source runtime machine role: {role}")
        self.network("task_machine")
        self.network("verifier_machine")
        return self

    def require_task(self, task: TaskSpec) -> None:
        if self.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError("Source runtime belongs to another TaskSpec")

    def network(self, role: Literal["task_machine", "verifier_machine"]) -> NetworkPolicy | None:
        machine = self.runtime.get(role)
        if machine is None:
            return None
        if not isinstance(machine, dict):
            raise ValueError(f"Source runtime machine must be an object: {role}")
        if "network" not in machine:
            return None
        return NetworkPolicy(machine["network"])


class TaskNetworkPolicyReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_runtime_sha256: str | None
    actor_required: bool
    actor_declared: NetworkPolicy | None
    verifier_declared: NetworkPolicy | None
    actor_effective: NetworkPolicy | None
    verifier_effective: NetworkPolicy | None


def write_source_runtime(task: TaskSpec, runtime: dict, directory: Path) -> None:
    """Preserve the complete original runtime before measuring a packet's checksum."""
    binding = TaskSpecRuntimeBinding(
        task_sha256=canonical_sha256(task.model_dump(mode="json")),
        runtime_sha256=canonical_sha256(runtime),
        runtime=runtime,
    )
    with (directory / SOURCE_RUNTIME_FILE).open("x") as stream:
        stream.write(binding.model_dump_json(indent=2) + "\n")


def read_source_runtime(directory: Path) -> TaskSpecRuntimeBinding | None:
    path = directory / SOURCE_RUNTIME_FILE
    return TaskSpecRuntimeBinding.model_validate_json(path.read_text()) if path.is_file() else None
