# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Signal-specific resource calibration for production Harbor oracle trials."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import ExecutionRequirements

INITIAL_REQUIREMENTS = ExecutionRequirements(
    cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
)
CALIBRATION_CAP = ExecutionRequirements(
    cpus=16, memory_mb=65536, storage_mb=131072, agent_timeout_sec=14400, verifier_timeout_sec=1800
)
CPU_BOUND_FRACTION = 0.9


class FailureSignal(StrEnum):
    OOM = "oom"
    DISK_FULL = "disk_full"
    AGENT_TIMEOUT = "agent_timeout"
    VERIFIER_TIMEOUT = "verifier_timeout"
    NON_RESOURCE = "non_resource"
    INFRASTRUCTURE = "infrastructure"


class ResourceDimension(StrEnum):
    CPUS = "cpus"
    MEMORY = "memory_mb"
    STORAGE = "storage_mb"
    AGENT_TIMEOUT = "agent_timeout_sec"
    VERIFIER_TIMEOUT = "verifier_timeout_sec"


class ResourceUsage(BaseModel):
    """Backend measurements; CPU time and elapsed time cover the agent phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    oom_kills: int = Field(default=0, ge=0)
    cpu_seconds: float | None = Field(default=None, ge=0)
    elapsed_seconds: float | None = Field(default=None, gt=0)

    def cpu_bound(self, cpus: int) -> bool:
        return (
            self.cpu_seconds is not None
            and self.elapsed_seconds is not None
            and self.cpu_seconds / self.elapsed_seconds >= CPU_BOUND_FRACTION * cpus
        )


def failure_signal(exception: dict | None, log: str, usage: ResourceUsage) -> FailureSignal:
    """Classify explicit runtime evidence; SIGKILL/exit 137 alone never proves OOM."""
    exception = exception or {}
    kind = exception.get("exception_type", "")
    message = f"{exception.get('exception_message', '')}\n{log}".lower()
    if (
        usage.oom_kills
        or kind == "OutOfMemoryError"
        or any(marker in message for marker in ("oomkilled", "oom-kill", "out of memory", "memory cgroup out of memory"))
    ):
        return FailureSignal.OOM
    if kind in {"AgentDiskFullError", "SandboxDiskFullError"} or any(
        marker in message for marker in ("no space left on device", "enospc")
    ):
        return FailureSignal.DISK_FULL
    if kind == "AgentTimeoutError":
        return FailureSignal.AGENT_TIMEOUT
    if kind == "VerifierTimeoutError":
        return FailureSignal.VERIFIER_TIMEOUT
    if kind in {"EnvironmentStartTimeoutError", "RuntimeServiceError", "RuntimeLogCollectionError", "ConnectionError"}:
        return FailureSignal.INFRASTRUCTURE
    return FailureSignal.NON_RESOURCE


def next_requirements(
    current: ExecutionRequirements,
    signal: FailureSignal,
    usage: ResourceUsage,
    fixed: frozenset[ResourceDimension] = frozenset(),
) -> ExecutionRequirements | None:
    """Double only failed dimensions, stopping when none can grow within the cap."""
    fields = {
        FailureSignal.OOM: ["memory_mb"],
        FailureSignal.DISK_FULL: ["storage_mb"],
        FailureSignal.AGENT_TIMEOUT: ["agent_timeout_sec"],
        FailureSignal.VERIFIER_TIMEOUT: ["verifier_timeout_sec"],
        FailureSignal.NON_RESOURCE: [],
        FailureSignal.INFRASTRUCTURE: [],
    }[signal]
    if signal is FailureSignal.AGENT_TIMEOUT and usage.cpu_bound(current.cpus):
        fields.append("cpus")
    values = current.model_dump()
    limits = CALIBRATION_CAP.model_dump()
    for name in fields:
        if ResourceDimension(name) in fixed:
            continue
        values[name] = min(values[name] * 2, limits[name])
    updated = ExecutionRequirements.model_validate(values)
    return updated if updated != current else None
