# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive backend demands from converted specs without treating backend choice as task data."""

from collections.abc import Mapping
from typing import Literal

from finestore.rl.source_identity import SourceId, original_source_id
from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.machine import Backend, NetworkPolicy
from shellbox.requirements import (
    COMMAND_FIELDS,
    MACHINE_FIELDS,
    SOURCE_FIELDS,
    Requirement,
    Support,
)
from shellbox.survey import CapabilityCell

from taskcompendium.backend_fields import (
    ENVIRONMENT_FIELDS,
    EXECUTION_FIELDS,
    HARBOR_ENVIRONMENT_FIELDS,
    RESOURCE_FIELDS,
    RUNTIME_FIELDS,
    SCRIPT_GRADER_FIELDS,
    TASK_FIELDS,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import EnvironmentRequirements, ExecutionRequirements


class RuntimeProfile(BaseModel):
    """Explicit deployment settings used where a converted spec omits a value."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    network: NetworkPolicy
    cpus: int | None = Field(gt=0)
    memory_mb: int | None = Field(gt=0)
    storage_mb: int | None = Field(gt=0)
    gpus: int = Field(ge=0)
    user: str | None
    startup_timeout: float | None = Field(gt=0)
    cleanup_timeout: float | None = Field(gt=0)


def backend_schema_sha256() -> str:
    """Identify the requirement vocabulary and owning task-schema classifications."""
    return canonical_sha256(
        {
            "environment": EnvironmentRequirements.model_json_schema(),
            "execution": ExecutionRequirements.model_json_schema(),
            "runtime_profile": RuntimeProfile.model_json_schema(),
            "requirements": [requirement.value for requirement in Requirement],
            "classifications": [
                COMMAND_FIELDS,
                ENVIRONMENT_FIELDS,
                EXECUTION_FIELDS,
                HARBOR_ENVIRONMENT_FIELDS,
                MACHINE_FIELDS,
                RESOURCE_FIELDS,
                RUNTIME_FIELDS,
                SCRIPT_GRADER_FIELDS,
                SOURCE_FIELDS,
                TASK_FIELDS,
            ],
        }
    )


class BackendFit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    support: Support
    task_counts: dict[Support, int]
    roles: dict[str, dict[Support, int]]
    blockers: dict[str, int]
    validation_errors: dict[str, int]


class BackendSourceMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    basis: Literal["converted_spec_census", "unassessed"]
    reason: str | None = None
    """Why a source has no converted population; required for unassessed rows only."""
    task_count: int
    inputs: dict[str, str]
    demands: dict[str, dict[Requirement, int]]
    resource_ranges: dict[str, dict[str, tuple[int, int]]]
    backends: dict[str, BackendFit]
    role_backend_assignments: dict[str, dict[str, int]]
    """Daytona deployment: agent on Daytona, package-lock grader on LOCAL."""
    grading_environment: dict | None
    """The published grading-boundary census, carried through without re-assessment."""
    integrity_contracts: dict | None = None
    """Optional contract census bound to the same published input population."""

    @model_validator(mode="after")
    def validate_reason(self) -> "BackendSourceMap":
        if self.basis == "unassessed":
            if not self.reason:
                raise ValueError("Unassessed sources require their reason")
        elif self.reason is not None:
            raise ValueError("Only unassessed sources carry a reason")
        return self


class BackendSurvey(BaseModel):
    """An immutable matrix and source map bound to one published catalog snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    catalog_root: str
    snapshot_id: str
    generated_at: str
    git_revision: str
    schema_sha256: str
    runtime_defaults: dict[str, RuntimeProfile]
    matrix: list[CapabilityCell]
    deployment_limits: dict[str, dict]
    local_disk_pool: str | None = None
    """Worker path of the verified LOCAL disk pool backing the local checker, if any."""
    sources: list[BackendSourceMap]

    def identity_payload(self) -> dict:
        """Preserve the published survey identity while source labels resolve canonically."""
        document = self.model_dump(mode="json")
        for source in document["sources"]:
            source["source_id"] = original_source_id(source["source_id"])
        return document

    @model_validator(mode="after")
    def validate_coverage(self) -> "BackendSurvey":
        expected = {backend.value for backend in Backend} | {"iris"}
        pairs = [(cell.backend, cell.requirement) for cell in self.matrix]
        if len(pairs) != len(set(pairs)) or set(pairs) != {
            (backend, requirement) for backend in expected for requirement in Requirement
        }:
            raise ValueError("Backend matrix must cover every backend and requirement exactly once")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("Duplicate source in backend map")
        for source in self.sources:
            if source.basis == "unassessed":
                if source.task_count or source.backends or source.demands:
                    raise ValueError("Unassessed sources cannot claim measured compatibility")
                continue
            if source.task_count <= 0 or set(source.backends) != expected:
                raise ValueError("Measured sources require a nonempty census and every backend")
            for fit in source.backends.values():
                if sum(fit.task_counts.values()) != source.task_count:
                    raise ValueError("Backend outcomes must cover the complete converted population")
        return self


def environment_demands(environment: EnvironmentRequirements) -> set[Requirement]:
    """Describe declared behavior; unknown capabilities fail classification."""
    demands = {Requirement(capability) for capability in environment.capabilities}
    if environment.command_semantics is not None:
        demands.add(Requirement(environment.command_semantics.value))
    for name in ("docker_image", "docker_build", "packages_lock", "package_spec"):
        if getattr(environment, name) is not None:
            demands.add(Requirement(name))
    if environment.packages_lock is not None:
        demands.add(Requirement.HOST_IMAGE)
    if environment.working_directory is not None:
        demands.add(Requirement.WORKDIR)
    if environment.setup_commands:
        demands.add(Requirement.SETUP)
    if environment.environment_variables:
        demands.add(Requirement.ENV)
    if environment.tool_providers:
        demands.add(Requirement.TOOL_PROVIDERS)
    return demands


def execution_resource_shape(task: Mapping, shape: Mapping | None) -> dict:
    """Apply task capacity, rejecting conflicting published allocations."""
    values = dict(shape or {})
    declared = task.get("execution_requirements")
    if declared is None:
        return values
    requirements = ExecutionRequirements.model_validate(declared)
    for name in ("cpus", "memory_mb", "storage_mb"):
        requested = getattr(requirements, name)
        if values.get(name) is not None and values[name] != requested:
            raise ValueError(f"{task.get('id')}: {name} conflicts with task execution requirements")
        values[name] = requested
    return values


def task_demands(document: Mapping) -> dict[str, set[Requirement]]:
    """Extract independent agent/grader demands, including published runtime budgets.

    Only feature names leave this function: package locks, file contents,
    environment values and backend selections never become survey metadata.
    """
    task = document.get("task", document)
    runtime = document.get("runtime") or {}
    grader = task.get("grader") or {}
    resources = task.get("resources") or {}
    roles = {}
    for role, declared, shape, files in (
        (
            "agent",
            task.get("environment_requirements"),
            runtime.get("task_machine"),
            [*resources.get("all", []), *resources.get("worker", [])],
        ),
        (
            "grader",
            grader.get("environment"),
            runtime.get("verifier_machine"),
            [*resources.get("all", []), *resources.get("verifier", [])],
        ),
    ):
        if declared is None:
            continue
        environment = EnvironmentRequirements.model_validate(declared)
        demands = environment_demands(environment)
        if task.get("integrity_contract") is not None:
            demands.update(TASK_FIELDS["integrity_contract"])
        if role == "agent":
            if task.get("output_paths") or task.get("answer_type") in {"file", "state", "workspace_state"}:
                demands.add(Requirement.FILESYSTEM)
            if task.get("tool_bindings"):
                demands.add(Requirement.TOOL_PROVIDERS)
            if grader.get("collect"):
                demands.add(Requirement.PROCESS)
        if files:
            demands.add(Requirement.FILE_UPLOAD)
        if any(resource.get("mode") is not None for resource in files):
            demands.add(Requirement.FILE_MODE)
        if any(resource.get("mtime_ns") is not None for resource in files):
            demands.add(Requirement.FILE_MTIME)
        if not demands:
            continue
        if role == "agent":
            shape = execution_resource_shape(task, shape)
        if task.get("execution_requirements") is not None:
            demands.add(Requirement.COMMAND_TIMEOUT)
        if shape is not None:
            if "network" in shape:
                demands.add(Requirement("network_" + shape["network"]))
            for field in ("cpus", "memory_mb", "storage_mb", "gpus", "startup_timeout", "cleanup_timeout"):
                if shape.get(field):
                    demands.add(Requirement(field))
            if shape.get("user") not in (None, "0", "root"):
                demands.add(Requirement.USER)
        if role == "grader" and demands:
            if grader.get("timeout") is not None:
                demands.add(Requirement.COMMAND_TIMEOUT)
            if grader.get("env"):
                demands.add(Requirement.ENV)
            if grader.get("cwd"):
                demands.add(Requirement.WORKDIR)
        if demands:
            roles[role] = demands
    return roles
