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
import json
import math
import tarfile
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from daytona import Resources
from harbor.environments.daytona import machine_spec
from harbor_config.models.task.config import EnvironmentConfig
from shellbox.backends.daytona.machine import UnsupportedMachineSpec, snapshot_identity
from shellbox.image import RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    CommandSemantics,
    MachineSpec,
    NetworkPolicy,
    check_execution_requirements,
)

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


def _machine_resources(cpus: int | None, memory_mb: int | None, storage_mb: int | None, gpus: int = 0) -> Resources:
    """The ``Resources`` construction mirroring ``DaytonaMachineFactory._prepare``."""
    return Resources(
        cpu=cpus,
        memory=math.ceil(memory_mb / 1024) if memory_mb is not None else None,
        disk=math.ceil(storage_mb / 1024) if storage_mb is not None else None,
        gpu=gpus or None,
    )


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
                name = snapshot_identity(
                    spec.source, _machine_resources(spec.cpus, spec.memory_mb, spec.storage_mb, spec.gpus)
                )
            except UnsupportedMachineSpec as error:
                failures.append((task_id, str(error)))
                continue
            sizes[name] += 1
    return SnapshotIdentityCount(distinct=len(sizes), sizes=dict(sizes), failures=failures)


def _decoded_source(source: Mapping) -> bytes:
    if source.get("kind") == "inline_file":
        return base64.b64decode(source.get("content_base64", ""))
    return json.dumps(source, sort_keys=True).encode()


def _verifier_files(spec: Mapping) -> Iterator[tuple[str, bytes, int]]:
    resources = (spec.get("task") or {}).get("resources", {})
    items = list(resources.get("verifier", [])) + list(resources.get("all", []))
    for item in items:
        path = item.get("path", "")
        if path.startswith(ENVIRONMENT_PREFIX) and item.get("source", {}).get("kind") != "directory":
            yield (
                path[len(ENVIRONMENT_PREFIX) :],
                _decoded_source(item["source"]),
                int(item.get("mode") or "644", 8),
            )


@dataclass(frozen=True)
class _DeclaredRequirements:
    """The slice of a lowered spec's environment that execution checks read."""

    capabilities: tuple[str, ...]
    command_semantics: str | None
    docker_image: str | None


def daytona_execution_failure(requirements: Mapping) -> str | None:
    """Why the Daytona backend would refuse this task's declared requirements, if it would.

    A converted task that only names an image and ordinary capabilities passes; one that
    declares an unsupported capability (a container runtime, for instance) or needs the
    account to build its archived Dockerfile does not, whatever its snapshot count is.
    """
    declared = _DeclaredRequirements(
        capabilities=tuple(requirements.get("capabilities") or ()),
        command_semantics=requirements.get("command_semantics"),
        docker_image=requirements.get("docker_image"),
    )
    spec = MachineSpec(source=RegistryImage(declared.docker_image), env={}, network=NetworkPolicy.ALLOW)
    try:
        check_execution_requirements(
            declared,
            spec,
            semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=LINUX_PROCESS_CAPABILITIES,
            image_reference=declared.docker_image,
        )
    except UnsupportedMachineSpec as error:
        return str(error)
    return None


def count_lowered_spec_identities(tasks: Iterable[tuple[str, Mapping]]) -> SnapshotIdentityCount:
    """Count snapshot identities for lowered Harbor task specs.

    Each spec contributes verifier and task-machine identities for its declared
    registry image or materialized ``environment/`` context, using each slot's
    runtime machine shape.
    """
    sizes: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    execution_failures: list[tuple[str, str]] = []
    for task_id, spec in tasks:
        task = spec.get("task") or {}
        runtime = spec.get("runtime") or {}
        requirements = task.get("environment_requirements") or {}
        docker_image = requirements.get("docker_image")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                for name, content, mode in _verifier_files(spec):
                    _write_context(Path(tmp), name, content, mode)
                if not docker_image and not (Path(tmp) / DOCKERFILE).exists():
                    raise UnsupportedMachineSpec(f"no docker_image and no {ENVIRONMENT_PREFIX}{DOCKERFILE}")
                verifier_machine = runtime.get("verifier_machine") or {}
                task_machine_spec = runtime.get("task_machine") or {}
                verifier = machine_spec(
                    EnvironmentConfig(
                        docker_image=docker_image,
                        workdir=requirements.get("working_directory"),
                        cpus=verifier_machine.get("cpus"),
                        memory_mb=verifier_machine.get("memory_mb"),
                        storage_mb=verifier_machine.get("storage_mb"),
                    ),
                    Path(tmp),
                    {},
                    cpus=verifier_machine.get("cpus"),
                    memory_mb=verifier_machine.get("memory_mb"),
                    storage_mb=verifier_machine.get("storage_mb"),
                )
                task_machine = machine_spec(
                    EnvironmentConfig(
                        docker_image=docker_image,
                        workdir=requirements.get("working_directory"),
                        cpus=task_machine_spec.get("cpus"),
                        memory_mb=task_machine_spec.get("memory_mb"),
                        storage_mb=task_machine_spec.get("storage_mb"),
                    ),
                    Path(tmp),
                    {},
                    cpus=task_machine_spec.get("cpus"),
                    memory_mb=task_machine_spec.get("memory_mb"),
                    storage_mb=task_machine_spec.get("storage_mb"),
                )
                failure = daytona_execution_failure(requirements)
                if failure is not None:
                    execution_failures.append((task_id, failure))
                # Identities are derived while the context exists: the production hash
                # reads the Dockerfile and every file of the staged tree.
                for machine in (verifier, task_machine):
                    name = snapshot_identity(
                        machine.source,
                        _machine_resources(machine.cpus, machine.memory_mb, machine.storage_mb, machine.gpus),
                    )
                    sizes[name] += 1
            except UnsupportedMachineSpec as error:
                failures.append((task_id, str(error)))
    return SnapshotIdentityCount(
        distinct=len(sizes), sizes=dict(sizes), failures=failures, execution_failures=execution_failures
    )


def count_registry_identities(references: Iterable[str]) -> SnapshotIdentityCount:
    """Count snapshot identities for registry references, resources defaulted."""
    sizes: Counter[str] = Counter()
    for reference in set(references):
        sizes[_registry_identity(reference)] += 1
    return SnapshotIdentityCount(distinct=len(sizes), sizes=dict(sizes))
