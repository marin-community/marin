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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from shellbox.machine import MachineFactory, NetworkPolicy, UnsupportedMachineSpec
from taskcompendium.grading_environment import NATIVE_CONFIG_DOCUMENT
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

# Daytona's SDK and Harbor's runtime are optional extras: this module imports
# them only where a machine measurement needs them, so a population without
# declared machine demands (and environments without the extras) still measures.

ROLE_NETWORKS: Mapping[str, str] = {"agent": "allow", "grader": "deny"}
ACTOR_ENVIRONMENT_SOURCES = ("docker_image", "docker_build", "packages_lock", "package_spec")

SNAPSHOT_IDENTITY = tuple[str | None, tuple[str, ...]]
"""One declared machine's snapshot name and its mapping or execution refusals."""


@dataclass(frozen=True)
class DaytonaCensusEngine:
    """Measure declared machine envelopes through production identities, without provider calls."""

    def measure(self, envelopes: Iterable[TaskEnvelope]) -> EnvelopeMeasurement:
        total = EnvelopeMeasurement(roles={"agent": {}, "grader": {}})
        context = _MeasureContext()
        for envelope in envelopes:
            total = _merge(total, _measure_task(envelope, context))
        return total

    def describe(self) -> str:
        return (
            "actor on Daytona, package-lock grader on LOCAL; snapshot identity through the "
            "production naming path (Harbor's task-config mapping for natives, the shellbox "
            "MachineSpec for lowered specs); no provider calls"
        )


@dataclass
class _MeasureContext:
    """One measure() pass's caches and its lazily constructed backend machinery."""

    local_checks: dict[str, str | None] = field(default_factory=dict)
    identities: dict[str, SNAPSHOT_IDENTITY] = field(default_factory=dict)
    _factory: MachineFactory | None = field(default=None, repr=False)

    def factory(self) -> MachineFactory:
        """The Daytona factory, constructed on the first machine measurement that needs it."""
        if self._factory is None:
            from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415

            self._factory = DaytonaMachineFactory()
        return self._factory


def _merge(total: EnvelopeMeasurement, task: EnvelopeMeasurement) -> EnvelopeMeasurement:
    """Fold one task's measurement into the population aggregate."""
    return EnvelopeMeasurement(
        converted=total.converted + task.converted,
        machine_slots=total.machine_slots + task.machine_slots,
        snapshot_sizes=_sum(total.snapshot_sizes, task.snapshot_sizes),
        execution_failures=_sum(total.execution_failures, task.execution_failures),
        mapping_failures=_sum(total.mapping_failures, task.mapping_failures),
        roles={
            role: {
                backend: _merge_role(total.roles.get(role, {}).get(backend, RoleOutcome()), outcome)
                for backend, outcome in {**total.roles.get(role, {}), **task.roles.get(role, {})}.items()
            }
            for role in ("agent", "grader")
        },
        actor_sources=_sum(total.actor_sources, task.actor_sources),
        unsafe_task_counts=_sum(total.unsafe_task_counts, task.unsafe_task_counts),
    )


def _sum(total: Mapping[str, int], task: Mapping[str, int]) -> dict[str, int]:
    """Add two count mappings without assuming either is a Counter."""
    counts = Counter(total)
    counts.update(task)
    return dict(counts)


def _merge_role(total: RoleOutcome, task: RoleOutcome) -> RoleOutcome:
    """Fold one role outcome into the aggregate."""
    return RoleOutcome(
        machine_slots=total.machine_slots + task.machine_slots,
        failures={
            **total.failures,
            **{reason: total.failures.get(reason, 0) + count for reason, count in task.failures.items()},
        },
    )


def _measure_task(envelope: TaskEnvelope, context: _MeasureContext) -> EnvelopeMeasurement:
    """One task's declared machine envelopes, its actor sources and its refusal reasons."""
    task, spec = envelope.task, envelope.spec
    native = "harbor" in task.get("tags", ())
    shapes: Mapping[str, Mapping] = {}
    if native:
        shapes = native_shapes(native_task_toml(task))
        native = any(row["path"] == NATIVE_CONFIG_DOCUMENT for row in task["resources"]["verifier"])
    measurement = EnvelopeMeasurement(converted=1, roles={"agent": {}, "grader": {}})
    grader_env = grader_environment(spec.grader)
    task_sources: dict[str, object] = {}
    task_failures: list[str] = []
    machine_slots = 0
    for role, declared in (("agent", spec.environment_requirements), ("grader", grader_env)):
        shape = {**asdict(MachineShape()), "network": ROLE_NETWORKS[role], **dict(shapes.get(f"{role}_machine") or {})}
        needs_machine = (native and role == "agent") or (declared is not None and any(dict(declared).values()))
        if not needs_machine:
            continue
        outcome = _measure_role_machine(role, declared, shape, native, task, envelope.task_id, context, measurement)
        if role == "agent":
            task_sources = {name: getattr(declared, name) for name in ACTOR_ENVIRONMENT_SOURCES}
            for name, value in task_sources.items():
                if value is not None:
                    measurement.actor_sources[name] = measurement.actor_sources.get(name, 0) + 1
            machine_slots += 1
            task_failures.extend(outcome.failures)
    for detail in task_failures:
        reason = actor_failure_reason(detail, task_sources)
        measurement.unsafe_task_counts[reason] = measurement.unsafe_task_counts.get(reason, 0) + 1
        measurement.execution_failures[detail] = measurement.execution_failures.get(detail, 0) + 1
    return replace(measurement, machine_slots=machine_slots)


def _measure_role_machine(
    role: str,
    declared: EnvironmentRequirements,
    shape: Mapping,
    native: bool,
    task: Mapping,
    task_id: str,
    context: _MeasureContext,
    measurement: EnvelopeMeasurement,
) -> RoleOutcome:
    """Measure one role's declared machine and record its outcome on the task measurement."""
    is_local = role == "grader" and declared.packages_lock is not None
    outcome = measurement.roles[role].setdefault("local" if is_local else "daytona", RoleOutcome())
    outcome.machine_slots += 1
    if is_local:
        outcome_key = _local_key(declared, shape)
        if outcome_key not in context.local_checks:
            context.local_checks[outcome_key] = _check_local(declared, shape)
        if (refusal := context.local_checks[outcome_key]) is not None:
            outcome.failures[refusal] = outcome.failures.get(refusal, 0) + 1
        return outcome
    identity_inputs = _native_config_rows(task) if native and role == "agent" else declared.model_dump(mode="json")
    runtime = {
        "task_machine": shape if role == "agent" else None,
        "verifier_machine": shape if role == "grader" else None,
    }
    key = canonical_sha256([identity_inputs, runtime])
    if key not in context.identities:
        context.identities[key] = (
            _native_identity(task_id, task)
            if native and role == "agent"
            else _lowered_identity(role, declared, shape, context.factory())
        )
    name, errors = context.identities[key]
    if name is not None and role == "agent":
        measurement.snapshot_sizes[name] = measurement.snapshot_sizes.get(name, 0) + 1
    for detail in errors:
        outcome.failures[detail] = outcome.failures.get(detail, 0) + 1
    return outcome


def _lowered_identity(
    role: str, declared: EnvironmentRequirements, shape: Mapping, factory: MachineFactory
) -> SNAPSHOT_IDENTITY:
    """A lowered task machine's snapshot name and refusals, through the production identity."""
    from experiments.post_training.task_curation.daytona_identity import declared_identity  # noqa: PLC0415

    measured = declared_identity(
        role, declared, MachineShape.from_mapping(shape), factory, network=NetworkPolicy(shape["network"])
    )
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
    from shellbox.backends.local.machine import check_local_environment  # noqa: PLC0415

    from experiments.post_training.task_curation.daytona_identity import declared_machine  # noqa: PLC0415

    machine_shape = MachineShape.from_mapping(shape)
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
    delivered = [row for row in task["resources"]["verifier"] if row["path"] == NATIVE_CONFIG_DOCUMENT]
    if not delivered:
        raise ValueError(f"harbor-tagged task carries no delivered task.toml: {task.get('id')}")
    return resource_bytes(TaskResource.model_validate(delivered[0]))


def _native_config_rows(task: Mapping) -> list[Mapping]:
    """The verifier resources the native identity names: the delivered config and its build context."""
    return [
        row
        for row in task["resources"]["verifier"]
        if row["path"] == NATIVE_CONFIG_DOCUMENT or row["path"].startswith(ENVIRONMENT_PREFIX)
    ]


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
        for row in _native_config_rows(task):
            resource = TaskResource.model_validate(row)
            content = resource_bytes(resource)
            entry = tarfile.TarInfo(resource.path)
            entry.size = len(content)
            entry.mode = int(resource.mode or "644", 8)
            archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()
