# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the distinct Daytona snapshots a task cohort provisions.

Measurement runs the production naming path: each pinned task's environment
is materialized to a build context, the shellbox ``MachineSpec`` is built
through Harbor's own task-config mapping, and the snapshot name comes from
``shellbox.backends.daytona.machine.snapshot_identity`` — the same function
the Daytona factory uses to name and create snapshots. A task the production
validation rejects is counted as a failure and reported; it is not
Daytona-safe. No Daytona API calls are made.
"""

import base64
import gzip
import io
import tarfile
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
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
from shellbox.survey import ProbeRequirements

ENVIRONMENT_PREFIX = "environment/"
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


def _unpack_environment(payload: bytes) -> tuple[list[tuple[str, bytes, int]], bytes | None]:
    """The extracted ``environment/`` entries and the task's own ``task.toml``."""
    entries: list[tuple[str, bytes, int]] = []
    task_toml: bytes | None = None
    with tarfile.open(
        fileobj=io.BytesIO(gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload)
    ) as archive:
        for member in archive.getmembers():
            if member.isdir() or member.issym():
                continue
            if member.name in {"task.toml", "./task.toml"}:
                task_toml = archive.extractfile(member).read()
                continue
            if member.name.startswith(ENVIRONMENT_PREFIX):
                entries.append((member.name[len(ENVIRONMENT_PREFIX) :], archive.extractfile(member).read(), member.mode))
    return entries, task_toml


def _registry_identity(reference: str) -> str:
    """Production snapshot name for a registry reference with default resources."""
    return snapshot_identity(RegistryImage(reference), Resources())


def count_task_identities(tasks: Iterable[tuple[str, bytes]]) -> SnapshotIdentityCount:
    """Count snapshot identities for native cohort tars, one per task."""
    sizes: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    factory = DaytonaMachineFactory()
    for task_id, payload in tasks:
        entries, task_toml = _unpack_environment(payload)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                for name, content, mode in entries:
                    _write_context(Path(tmp), name, content, mode)
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
                factory.check_environment(ProbeRequirements(), spec)
                if config.gpus:
                    raise UnsupportedMachineSpec("Native Harbor's Daytona mapping does not carry the requested GPUs")
                name = snapshot_identity(spec.source, snapshot_resources(spec))
            except UnsupportedMachineSpec as error:
                failures.append((task_id, str(error)))
                continue
            sizes[name] += 1
    return SnapshotIdentityCount(
        distinct=len(sizes), sizes=dict(sizes), failures=failures, execution_failures=list(failures)
    )


def native_task_toml(task: Mapping) -> bytes:
    # Native counting is also used in Harbor's standalone optional-dependency environment.
    from taskcompendium.models import TaskResource  # noqa: PLC0415
    from taskcompendium.runtime.resources import resource_bytes  # noqa: PLC0415

    resource = next(row for row in task["resources"]["verifier"] if row["path"] == "task.toml")
    return resource_bytes(TaskResource.model_validate(resource))


def count_native_spec_identities(tasks: Iterable[tuple[str, Mapping]]) -> SnapshotIdentityCount:
    """Count lossless Harbor envelopes through their unchanged native files and task.toml."""
    from taskcompendium.backend_requirements import execution_resource_shape  # noqa: PLC0415
    from taskcompendium.models import TaskResource  # noqa: PLC0415
    from taskcompendium.runtime.resources import resource_bytes  # noqa: PLC0415

    def archives() -> Iterable[tuple[str, bytes]]:
        for task_id, task in tasks:
            if task.get("execution_requirements") is not None:
                document = tomllib.loads(native_task_toml(task).decode())
                environment = EnvironmentConfig.model_validate(document.get("environment") or {})
                execution_resource_shape(task, environment.model_dump())
                for role, field in (("agent", "agent_timeout_sec"), ("verifier", "verifier_timeout_sec")):
                    if (document.get(role) or {}).get("timeout_sec") != task["execution_requirements"][field]:
                        raise ValueError(f"{task_id}: {field} conflicts with native task.toml")
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
            yield task_id, buffer.getvalue()

    return count_task_identities(archives())


def count_lowered_spec_identities(tasks: Iterable[tuple[str, Mapping]]) -> SnapshotIdentityCount:
    """Measure each declared agent and grader machine through production identities.

    Package-lock graders require a local host and are recorded as execution
    failures for Daytona. They never borrow the agent's image or contribute a
    fabricated snapshot. Unbuilt contexts retain their identity but are refused
    by the runtime's unresolved-environment check.
    """
    # Native Harbor counting also runs in Harbor's standalone environment, which
    # does not install TaskCompendium. Only lowered TaskCompendium inputs need it.
    from taskcompendium.backend_requirements import execution_resource_shape  # noqa: PLC0415
    from taskcompendium.models import EnvironmentRequirements, require_resolved_environment  # noqa: PLC0415

    sizes: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    execution_failures: list[tuple[str, str]] = []
    factory = DaytonaMachineFactory()
    for task_id, spec in tasks:
        task = spec.get("task") or {}
        runtime = spec.get("runtime") or {}
        grader = task.get("grader") or {}
        mapping_errors = []
        execution_errors = []
        slots = (
            ("agent", task.get("environment_requirements"), runtime.get("task_machine")),
            ("grader", grader.get("environment"), runtime.get("verifier_machine")),
        )
        for role, declared, shape in slots:
            if declared is None or shape is None:
                continue
            shape = execution_resource_shape(task, shape)
            requirements = EnvironmentRequirements.model_validate(declared)
            with tempfile.TemporaryDirectory() as tmp:
                context = Path(tmp)
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
                elif requirements.package_spec is not None:
                    source = requirements.package_spec
                else:
                    mapping_errors.append(f"{role}: no declared environment")
                    execution_errors.append(f"{role}: no declared environment")
                    continue
                machine = MachineSpec(
                    source=source,
                    workdir=requirements.working_directory or "",
                    network=NetworkPolicy(shape.get("network", "allow")),
                    cpus=shape.get("cpus"),
                    memory_mb=shape.get("memory_mb"),
                    storage_mb=shape.get("storage_mb"),
                    gpus=shape.get("gpus", 0),
                )
                try:
                    factory.check_environment(requirements, machine)
                    require_resolved_environment(requirements)
                except UnsupportedMachineSpec as error:
                    execution_errors.append(f"{role}: {error}")
                if not isinstance(source, RegistryImage | DockerfileSource):
                    continue
                try:
                    name = snapshot_identity(
                        source,
                        snapshot_resources(machine),
                    )
                except UnsupportedMachineSpec as error:
                    mapping_errors.append(f"{role}: {error}")
                else:
                    sizes[name] += 1
        if mapping_errors:
            failures.append((task_id, "; ".join(mapping_errors)))
        if execution_errors:
            execution_failures.append((task_id, "; ".join(execution_errors)))
    return SnapshotIdentityCount(
        distinct=len(sizes), sizes=dict(sizes), failures=failures, execution_failures=execution_failures
    )


def count_registry_identities(references: Iterable[str]) -> SnapshotIdentityCount:
    """Count snapshot identities for registry references, resources defaulted."""
    sizes: Counter[str] = Counter()
    for reference in set(references):
        sizes[_registry_identity(reference)] += 1
    return SnapshotIdentityCount(distinct=len(sizes), sizes=dict(sizes))
