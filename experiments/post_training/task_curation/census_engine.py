# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The census engine: production machine identities over a converted population.

Implements the taskcompendium census protocol with the same primitives the
rollout runtime uses. Native Harbor tasks are measured through their delivered
``task.toml`` and the Daytona snapshot naming path; lowered TaskCompendium
tasks are measured from their declared environment requirements with the
pipeline's default machine shape. Package-lock graders demand a local host and
are validated with the local checker instead of contributing fabricated
snapshots. The measurement never calls a provider.
"""

import io
import tarfile
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import NetworkPolicy, UnsupportedMachineSpec
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import EnvironmentRequirements, TaskResource, grader_environment
from taskcompendium.pipeline.census import (
    EnvelopeMeasurement,
    RoleOutcome,
    TaskEnvelope,
    actor_failure_reason,
)
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.environment_fidelity import ENVIRONMENT_PREFIX
from experiments.post_training.task_curation.lowered_envelope import MachineShape

# Native identity counting runs through Harbor's isolated runtime, which not
# every environment installs: those imports stay inside the measurement functions.

PROFILE: Mapping[str, object] = {"cpus": None, "memory_mb": None, "storage_mb": None, "gpus": 0}
"""The machine shape a lowered task without native allocations measures under."""

ROLE_NETWORKS: Mapping[str, str] = {"agent": "allow", "grader": "deny"}
ACTOR_ENVIRONMENT_SOURCES = ("docker_image", "docker_build", "packages_lock", "package_spec")

SNAPSHOT_IDENTITY = tuple[str | None, tuple[str, ...]]
"""One declared machine's snapshot name and its mapping or execution refusals."""


@dataclass(frozen=True)
class DaytonaCensusEngine:
    """Measure declared machine envelopes through production identities, without provider calls."""

    def measure(self, envelopes: Iterable[TaskEnvelope]) -> EnvelopeMeasurement:
        converted = 0
        machine_slots = 0
        sizes: Counter[str] = Counter()
        execution_failures: Counter[str] = Counter()
        mapping_failures: Counter[str] = Counter()
        roles: dict[str, dict[str, RoleOutcome]] = {"agent": {}, "grader": {}}
        actor_sources: Counter[str] = Counter()
        unsafe_task_counts: Counter[str] = Counter()
        identities: dict[str, SNAPSHOT_IDENTITY] = {}
        local_checks: dict[str, str | None] = {}
        factory = DaytonaMachineFactory()
        for envelope in envelopes:
            converted += 1
            task, spec = envelope.task, envelope.spec
            if spec is None:
                mapping_failures[f"{envelope.task_id}: task is not a valid TaskSpec"] += 1
                continue
            native = "harbor" in task.get("tags", ())
            shapes: Mapping[str, Mapping] = {}
            if native:
                shapes = native_shapes(native_task_toml(task))
                native = any(row["path"] == "task.toml" for row in task["resources"]["verifier"])
            grader_env = grader_environment(spec.grader)
            task_sources: dict[str, object] = {}
            has_machine = False
            task_failures: list[str] = []
            for role, declared in (("agent", spec.environment_requirements), ("grader", grader_env)):
                shape = {**PROFILE, "network": ROLE_NETWORKS[role], **dict(shapes.get(f"{role}_machine") or {})}
                needs_machine = (native and role == "agent") or (declared is not None and any(dict(declared).values()))
                if not needs_machine:
                    continue
                if role == "agent":
                    task_sources = {name: getattr(declared, name) for name in ACTOR_ENVIRONMENT_SOURCES}
                    actor_sources.update(name for name, value in task_sources.items() if value is not None)
                    has_machine = True
                is_local = role == "grader" and declared.packages_lock is not None
                outcome = roles[role].setdefault("local" if is_local else "daytona", RoleOutcome())
                outcome.machine_slots += 1
                if is_local:
                    outcome_key = _local_key(declared, shape)
                    if outcome_key not in local_checks:
                        local_checks[outcome_key] = _check_local(declared, shape)
                    if (refusal := local_checks[outcome_key]) is not None:
                        outcome.failures[refusal] = outcome.failures.get(refusal, 0) + 1
                    continue
                runtime = {
                    "task_machine": shape if role == "agent" else None,
                    "verifier_machine": shape if role == "grader" else None,
                }
                identity_inputs = (
                    [
                        row
                        for row in task["resources"]["verifier"]
                        if row["path"] == "task.toml" or row["path"].startswith(ENVIRONMENT_PREFIX)
                    ]
                    if native and role == "agent"
                    else declared.model_dump(mode="json")
                )
                key = canonical_sha256([identity_inputs, runtime])
                if key not in identities:
                    identities[key] = _measure_identity(envelope.task_id, task, declared, shape, role, native, factory)
                name, errors = identities[key]
                if name is not None and role == "agent":
                    sizes[name] += 1
                for detail in errors:
                    outcome.failures[detail] = outcome.failures.get(detail, 0) + 1
                    if role == "agent":
                        task_failures.append(detail)
            for detail in task_failures:
                reason = actor_failure_reason(detail, task_sources)
                unsafe_task_counts[reason] += 1
                execution_failures[detail] += 1
            machine_slots += has_machine
        return EnvelopeMeasurement(
            converted=converted,
            machine_slots=machine_slots,
            snapshot_sizes=dict(sizes),
            execution_failures=dict(execution_failures),
            mapping_failures=dict(mapping_failures),
            roles=roles,
            actor_sources=dict(actor_sources),
            unsafe_task_counts=dict(unsafe_task_counts),
        )


def _measure_identity(
    task_id: str,
    task: Mapping,
    declared: EnvironmentRequirements,
    shape: Mapping,
    role: str,
    native: bool,
    factory: DaytonaMachineFactory,
) -> SNAPSHOT_IDENTITY:
    """One declared machine's snapshot name and refusals, through the production identity."""
    from experiments.post_training.task_curation.daytona_identity import declared_identity  # noqa: PLC0415

    if native:
        return _native_identity(task_id, task)
    machine_shape = MachineShape(
        cpus=shape.get("cpus"),
        memory_mb=shape.get("memory_mb"),
        storage_mb=shape.get("storage_mb"),
        gpus=shape.get("gpus", 0),
    )
    measured = declared_identity(role, declared, machine_shape, factory, network=NetworkPolicy(shape["network"]))
    return measured.name, tuple(error for error in (measured.mapping_error, measured.execution_error) if error)


def _native_identity(task_id: str, task: Mapping) -> SNAPSHOT_IDENTITY:
    """A native task's snapshot name and refusals, through its reassembled task archive."""
    from experiments.post_training.task_curation.daytona_identity import count_task_identities  # noqa: PLC0415

    measurement = count_task_identities([(task_id, _native_archive(task))])
    errors = tuple(detail for _, detail in (*measurement.execution_failures, *measurement.failures))
    if measurement.distinct != 1:
        return None, errors or (f"{task_id}: native archive produced {measurement.distinct} identities",)
    return next(iter(measurement.sizes)), errors


def _local_key(declared: EnvironmentRequirements, shape: Mapping) -> str:
    return canonical_sha256([declared.model_dump(mode="json"), dict(shape)])


def _check_local(declared: EnvironmentRequirements, shape: Mapping) -> str | None:
    """The local checker's refusal for one declared package-lock grader, or ``None`` when it runs."""
    from experiments.post_training.task_curation.daytona_identity import declared_machine  # noqa: PLC0415

    machine_shape = MachineShape(
        cpus=shape.get("cpus"),
        memory_mb=shape.get("memory_mb"),
        storage_mb=shape.get("storage_mb"),
        gpus=shape.get("gpus", 0),
    )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            machine = declared_machine(
                declared,
                Path(tmp),
                network=NetworkPolicy(shape["network"]),
                cpus=machine_shape.cpus,
                memory_mb=machine_shape.memory_mb,
                storage_mb=machine_shape.storage_mb,
                gpus=machine_shape.gpus,
            )
            # The rollout runtime drops the per-process memory limit for LOCAL.
            check_local_environment(declared, replace(machine, memory_mb=None))
        except (UnsupportedMachineSpec, ValueError) as error:
            return str(error)
    return None


def native_task_toml(task: Mapping) -> bytes:
    """A native Harbor task's delivered ``task.toml``."""
    delivered = [row for row in task["resources"]["verifier"] if row["path"] == "task.toml"]
    if not delivered:
        raise ValueError(f"harbor-tagged task carries no delivered task.toml: {task.get('id')}")
    return resource_bytes(TaskResource.model_validate(delivered[0]))


def native_shapes(task_toml: bytes) -> dict[str, dict]:
    """The native task.toml's actor and verifier machine allocations."""
    from harbor_config.models.task.config import EnvironmentConfig  # noqa: PLC0415

    document = tomllib.loads(task_toml.decode(errors="replace"))
    actor = EnvironmentConfig.model_validate(document.get("environment") or {})
    verifier = EnvironmentConfig.model_validate(
        (document.get("verifier") or {}).get("environment") or actor.model_dump()
    )
    return {
        f"{role}_machine": {
            "cpus": environment.cpus,
            "memory_mb": environment.memory_mb,
            "storage_mb": environment.storage_mb,
            "gpus": environment.gpus or 0,
            "network": "allow" if environment.allow_internet else "deny",
        }
        for role, environment in (("agent", actor), ("grader", verifier))
    }


def _native_archive(task: Mapping) -> bytes:
    """Reassemble a native task's delivered configuration and environment build context."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for row in task["resources"]["verifier"]:
            if row["path"] != "task.toml" and not row["path"].startswith(ENVIRONMENT_PREFIX):
                continue
            resource = TaskResource.model_validate(row)
            content = resource_bytes(resource)
            entry = tarfile.TarInfo(resource.path)
            entry.size = len(content)
            entry.mode = int(resource.mode or "644", 8)
            archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()
