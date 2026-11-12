# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the distinct Daytona snapshots a task cohort provisions.

Measurement runs the production naming path. Native cohort tars are
materialized to a build context and mapped through Harbor's task-config
mapping; lowered TaskCompendium specs build the shellbox ``MachineSpec``
directly from their declared requirements. Either way the snapshot name comes
from ``shellbox.backends.daytona.machine.snapshot_identity`` — the same
function the Daytona factory uses to name and create snapshots. A task the
production validation rejects is counted as a failure and reported; it is not
Daytona-safe. No Daytona API calls are made.
"""

import base64
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from daytona import Resources
from harbor.environments.daytona import machine_spec
from harbor_config.models.task.config import EnvironmentConfig
from shellbox.backends.daytona.machine import (
    DaytonaMachineFactory,
    UnsupportedMachineSpec,
    snapshot_identity,
    snapshot_resources,
)
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    HostImage,
    MachineSpec,
    NetworkPolicy,
)
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, VerifyitGrader, require_resolved_environment

from experiments.post_training.task_curation.environment_fidelity import ENVIRONMENT_PREFIX
from experiments.post_training.task_curation.lowered_envelope import LoweredSpec, MachineShape
from experiments.post_training.task_curation.task_tars import open_task_tar

DOCKERFILE = "Dockerfile"


def _task_environment_config(task_toml: bytes | None) -> EnvironmentConfig:
    """Harbor's own environment config for a task, defaulted when absent."""
    if task_toml is None:
        return EnvironmentConfig()
    document = tomllib.loads(task_toml.decode(errors="replace"))
    return EnvironmentConfig(**document.get("environment") or {})


@dataclass
class SnapshotIdentityCount:
    """Distinct production snapshot names with per-name cohort sizes."""

    distinct: int
    sizes: dict[str, int] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    # Tasks whose declared execution requirements the Daytona backend refuses, so a
    # snapshot count alone must not call them safe.
    execution_failures: list[tuple[str, str]] = field(default_factory=list)


def _write_context(root: Path, name: str, content: bytes, mode: int = 0o644) -> None:
    """Stage one context entry with the mode the extracted task tree carries."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    target.chmod(mode & 0o7777)


@dataclass(frozen=True)
class EnvironmentEntry:
    """One file inside a native task archive's ``environment/`` build context."""

    path: str
    content: bytes
    mode: int


def _unpack_environment(payload: bytes) -> tuple[list[EnvironmentEntry], bytes | None]:
    """The extracted ``environment/`` entries and the task's own ``task.toml``."""
    entries: list[EnvironmentEntry] = []
    task_toml: bytes | None = None
    with open_task_tar(payload) as archive:
        for member in archive.getmembers():
            if member.isdir() or member.issym():
                continue
            if member.name in {"task.toml", "./task.toml"}:
                task_toml = archive.extractfile(member).read()
                continue
            if member.name.startswith(ENVIRONMENT_PREFIX):
                entries.append(
                    EnvironmentEntry(
                        member.name[len(ENVIRONMENT_PREFIX) :], archive.extractfile(member).read(), member.mode
                    )
                )
    return entries, task_toml


def _registry_identity(reference: str) -> str:
    """Production snapshot name for a registry reference with default resources."""
    return snapshot_identity(RegistryImage(reference), Resources())


def count_task_identities(tasks: Iterable[tuple[str, bytes]]) -> SnapshotIdentityCount:
    """Count snapshot identities for native cohort tars, one per task."""
    sizes: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    for task_id, payload in tasks:
        entries, task_toml = _unpack_environment(payload)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                for entry in entries:
                    _write_context(Path(tmp), entry.path, entry.content, entry.mode)
                config = _task_environment_config(task_toml)
                if not config.docker_image and not (Path(tmp) / DOCKERFILE).exists():
                    raise UnsupportedMachineSpec(f"missing {ENVIRONMENT_PREFIX}{DOCKERFILE}")
                spec = machine_spec(
                    config,
                    Path(tmp),
                    {},
                    cpus=config.cpus,
                    memory_mb=config.memory_mb,
                    storage_mb=config.storage_mb,
                )
                name = snapshot_identity(spec.source, snapshot_resources(spec))
            except UnsupportedMachineSpec as error:
                failures.append((task_id, str(error)))
                continue
            sizes[name] += 1
    return SnapshotIdentityCount(distinct=len(sizes), sizes=dict(sizes), failures=failures)


def count_lowered_spec_identities(tasks: Iterable[tuple[str, LoweredSpec]]) -> SnapshotIdentityCount:
    """Measure each declared agent and grader machine through production identities.

    Package-lock graders require a local host and are recorded as execution
    failures for Daytona. They never borrow the agent's image or contribute a
    fabricated snapshot. Unbuilt contexts retain their identity but are refused
    by the runtime's unresolved-environment check.
    """
    sizes: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    execution_failures: list[tuple[str, str]] = []
    factory = DaytonaMachineFactory()
    for task_id, spec in tasks:
        grader = spec.task.grader
        # Only a grader that runs in a machine declares one; the other kinds borrow none.
        grader_environment = grader.environment if isinstance(grader, ScriptGrader | VerifyitGrader) else None
        slots = (
            ("agent", spec.task.environment_requirements, spec.task_machine),
            ("grader", grader_environment, spec.verifier_machine),
        )
        mapping_errors: list[str] = []
        execution_errors: list[str] = []
        for role, declared, shape in slots:
            measured = _declared_identity(role, declared, shape, factory)
            if measured.name is not None:
                sizes[measured.name] += 1
            if measured.mapping_error is not None:
                mapping_errors.append(measured.mapping_error)
            if measured.execution_error is not None:
                execution_errors.append(measured.execution_error)
        if mapping_errors:
            failures.append((task_id, "; ".join(mapping_errors)))
        if execution_errors:
            execution_failures.append((task_id, "; ".join(execution_errors)))
    return SnapshotIdentityCount(
        distinct=len(sizes), sizes=dict(sizes), failures=failures, execution_failures=execution_failures
    )


@dataclass(frozen=True)
class _DeclaredIdentity:
    """One declared machine's snapshot name, mapping refusal and execution refusal."""

    name: str | None = None
    mapping_error: str | None = None
    execution_error: str | None = None


def declared_machine(
    requirements: EnvironmentRequirements,
    context: Path,
    *,
    network: NetworkPolicy,
    cpus: int | None,
    memory_mb: int | None,
    storage_mb: int | None,
    gpus: int = 0,
) -> MachineSpec:
    """The shellbox MachineSpec a task's declared requirements and machine shape resolve to.

    A Docker build context is materialized into ``context``; a registry image, package
    spec or host requirement maps onto the matching source, with the declared
    working directory. The measurement never invents a source the requirements
    do not declare.
    """
    if requirements.docker_build is not None:
        for resource in requirements.docker_build.files:
            _write_context(
                context,
                resource.path,
                base64.b64decode(resource.source.content_base64, validate=True),
                int(resource.mode or "644", 8),
            )
        source = DockerfileSource(context, context / DOCKERFILE)
    elif requirements.docker_image is not None:
        source = RegistryImage(requirements.docker_image)
    elif requirements.packages_lock is not None:
        source = HostImage()
    else:
        if requirements.package_spec is None:
            raise ValueError("declared requirements name no environment")
        source = requirements.package_spec
    return MachineSpec(
        source=source,
        workdir=requirements.working_directory or "",
        network=network,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        gpus=gpus,
    )


def _declared_identity(
    role: str,
    declared: EnvironmentRequirements | None,
    shape: MachineShape | None,
    factory: DaytonaMachineFactory,
) -> _DeclaredIdentity:
    """Measure one declared agent or grader machine through the production identity."""
    if declared is None or shape is None:
        return _DeclaredIdentity()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            machine = declared_machine(
                declared,
                Path(tmp),
                network=NetworkPolicy.ALLOW,
                cpus=shape.cpus,
                memory_mb=shape.memory_mb,
                storage_mb=shape.storage_mb,
                gpus=shape.gpus,
            )
        except ValueError as error:
            return _DeclaredIdentity(mapping_error=f"{role}: {error}", execution_error=f"{role}: {error}")
        try:
            factory.check_environment(declared, machine)
            require_resolved_environment(declared)
        except UnsupportedMachineSpec as error:
            # An execution refusal still reports the identity: snapshot counts measure
            # what the cohort would provision, whether or not Daytona would run it.
            execution_error = f"{role}: {error}"
        else:
            execution_error = None
        if not isinstance(machine.source, RegistryImage | DockerfileSource):
            return _DeclaredIdentity(execution_error=execution_error)
        try:
            return _DeclaredIdentity(
                snapshot_identity(machine.source, snapshot_resources(machine)), execution_error=execution_error
            )
        except UnsupportedMachineSpec as error:
            return _DeclaredIdentity(mapping_error=f"{role}: {error}", execution_error=execution_error)


def count_registry_identities(references: Iterable[str]) -> SnapshotIdentityCount:
    """Count snapshot identities for registry references, resources defaulted."""
    sizes: Counter[str] = Counter()
    for reference in set(references):
        sizes[_registry_identity(reference)] += 1
    return SnapshotIdentityCount(distinct=len(sizes), sizes=dict(sizes))
