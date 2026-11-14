# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind original deployment choices to an unchanged Harbor TaskSpec packet."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.machine import NetworkPolicy

from taskcompendium.harbor.adapter import declared_execution_limits, source_task_configurations
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

SOURCE_RUNTIME_FILE = "source-runtime.json"
RESOURCE_FIELDS = ("cpus", "memory_mb", "storage_mb")
MachineRole = Literal["task_machine", "verifier_machine"]
MACHINE_ROLES: tuple[MachineRole, ...] = ("task_machine", "verifier_machine")


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
            if (
                isinstance(machine, dict)
                and {"network", *RESOURCE_FIELDS}.intersection(machine)
                and role not in MACHINE_ROLES
            ):
                raise ValueError(f"Unsupported source runtime machine role: {role}")
        self.network("task_machine")
        self.network("verifier_machine")
        self.resource_limits("task_machine")
        self.resource_limits("verifier_machine")
        return self

    def require_task(self, task: TaskSpec) -> None:
        if self.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError("Source runtime belongs to another TaskSpec")
        declared = declared_execution_limits(task)
        verifier: dict[str, int | float] = {}
        for resource in source_task_configurations(task):
            configuration = tomllib.loads(resource_bytes(resource).decode())
            environment = (configuration.get("verifier") or {}).get("environment") or {}
            for name in RESOURCE_FIELDS:
                value = environment.get(name)
                if value is None:
                    continue
                if name in verifier and verifier[name] != value:
                    raise ValueError(f"Conflicting original verifier {name} declarations")
                verifier[name] = value
        for role in MACHINE_ROLES:
            original = declared if role == "task_machine" else verifier
            for name, value in self.resource_limits(role).items():
                if name in original and original[name] != value:
                    raise ValueError(f"Source runtime {role}.{name} conflicts with its original declaration")

    def resource_limits(self, role: MachineRole) -> dict[str, int]:
        """Return only explicitly declared allocation fields, preserving missing dimensions."""
        machine = self.runtime.get(role)
        if machine is None:
            return {}
        if not isinstance(machine, dict):
            raise ValueError(f"Source runtime machine must be an object: {role}")
        limits = {}
        for name in RESOURCE_FIELDS:
            value = machine.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Source runtime {role}.{name} must be a positive integer")
            limits[name] = value
        return limits

    def network(self, role: MachineRole) -> NetworkPolicy | None:
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


def network_policy_qualified(binding: TaskSpecRuntimeBinding | None, observed: TaskNetworkPolicyReceipt | None) -> bool:
    """Match observed role policies to the preserved declaration and actual actor plan."""
    actor = binding.network("task_machine") if binding else None
    verifier = binding.network("verifier_machine") if binding else None
    if observed is None:
        return actor is None and verifier is None
    return (
        observed.source_runtime_sha256 == (binding.runtime_sha256 if binding else None)
        and observed.actor_declared == actor
        and observed.verifier_declared == verifier
        and (actor is None or not observed.actor_required or observed.actor_effective == actor)
        and (verifier is None or observed.verifier_effective == verifier)
    )


def write_source_runtime(task: TaskSpec, runtime: dict, directory: Path) -> None:
    """Preserve the complete original runtime before measuring a packet's checksum."""
    binding = TaskSpecRuntimeBinding(
        task_sha256=canonical_sha256(task.model_dump(mode="json")),
        runtime_sha256=canonical_sha256(runtime),
        runtime=runtime,
    )
    binding.require_task(task)
    with (directory / SOURCE_RUNTIME_FILE).open("x") as stream:
        stream.write(binding.model_dump_json(indent=2) + "\n")


def read_source_runtime(directory: Path) -> TaskSpecRuntimeBinding | None:
    path = directory / SOURCE_RUNTIME_FILE
    return TaskSpecRuntimeBinding.model_validate_json(path.read_text()) if path.is_file() else None
