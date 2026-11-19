# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline capability survey using production factory validation; never acquire machines."""

import importlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from shellbox.backends.local.cgroup import (
    CPU_PERIOD,
    MachineCgroup,
    delegated_cgroup_parent,
)
from shellbox.backends.local.disk import LocalDiskPool
from shellbox.image import DockerfileSource, PreparedImage, RegistryImage
from shellbox.machine import (
    CommandSemantics,
    ExecutionRequirements,
    HostImage,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    QemuBundle,
    UnsupportedMachineSpec,
)
from shellbox.requirements import Requirement, Support

PROBE_IMAGE = "docker.io/library/python@sha256:" + "0" * 64
PROBE_CPUS = 1
PROBE_MEMORY_MB = 1024
PROBE_STORAGE_MB = 1024
Checker = Callable[[ExecutionRequirements, MachineSpec], None]


class CapabilityCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    requirement: Requirement
    support: Support
    evidence: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ProbeRequirements:
    command_semantics: CommandSemantics = CommandSemantics.LINUX_PROCESS
    capabilities: tuple[str, ...] = ()
    docker_image: str | None = None


@dataclass(frozen=True)
class LocalResourceEvidence:
    """This worker's verified LOCAL enforcement machinery, probed once per survey.

    ``storage_enforced``: a LocalDiskPool was constructed on this privileged Linux
    worker (its construction probe verifies ENOSPC on a fixed-size ext4 loop
    filesystem) and the pool has free capacity for the survey probe size.
    ``cgroup_verified``: a probe machine cgroup applied and read back cpu.max and
    memory.max in the delegated cgroup v2 hierarchy that production machine
    creation uses.
    """

    storage_enforced: bool
    cgroup_verified: bool
    cgroup_parent: Path | None = None


def cgroup_limits_applied(path: Path, *, cpus: int, memory_mb: int) -> bool:
    """Whether the kernel actually applied the demanded limits to a machine cgroup."""
    try:
        quota, period = (path / "cpu.max").read_text().split()[:2]
        cpu_applied = int(quota) == cpus * CPU_PERIOD and int(period) == CPU_PERIOD
        memory_applied = int((path / "memory.max").read_text()) == memory_mb * 1024 * 1024
        return cpu_applied and memory_applied
    except (OSError, ValueError):
        return False


def _cgroup_limits_verified(parent: Path) -> Path | None:
    """Apply and read back survey-sized limits in a probe machine cgroup."""
    try:
        cgroup = MachineCgroup.create(parent, cpus=PROBE_CPUS, memory_mb=PROBE_MEMORY_MB)
    except (OSError, UnsupportedMachineSpec):
        return None
    try:
        if not cgroup_limits_applied(cgroup.path, cpus=PROBE_CPUS, memory_mb=PROBE_MEMORY_MB):
            return None
        return parent
    finally:
        cgroup.kill()
        cgroup.remove_if_empty()


def local_resource_evidence(disk_pool: LocalDiskPool | None) -> LocalResourceEvidence:
    """Verify this worker's LOCAL storage and cgroup enforcement once, without machines."""
    storage_enforced = False
    if disk_pool is not None:
        try:
            storage_enforced = shutil.disk_usage(disk_pool.root).free >= PROBE_STORAGE_MB * 1024 * 1024
        except OSError:
            storage_enforced = False
    parent = None
    discovered = delegated_cgroup_parent()
    if discovered is not None:
        parent = _cgroup_limits_verified(discovered)
    return LocalResourceEvidence(
        storage_enforced=storage_enforced, cgroup_verified=parent is not None, cgroup_parent=parent
    )


def production_checkers(*, local_disk_pool: LocalDiskPool | None = None) -> dict[str, Checker]:
    """Return all production validators without acquiring machines or provider clients."""
    checkers = {}
    for backend, class_name, module_name in (
        ("docker", "DockerMachineFactory", "docker"),
        ("docker-gvisor", "DockerGvisorMachineFactory", "docker_gvisor"),
        ("iris", "IrisMachineFactory", "iris"),
        ("qemu", "QemuMachineFactory", "qemu"),
        ("shellsim", "ShellSimMachineFactory", "shellsim"),
        ("daytona", "DaytonaMachineFactory", "daytona"),
    ):
        module = importlib.import_module(f"shellbox.backends.{module_name}.machine")
        factory_type = getattr(module, class_name)
        factory = factory_type(controller_url="http://survey.invalid") if backend == "iris" else factory_type()
        checkers[backend] = factory.check_environment
    module = importlib.import_module("shellbox.backends.local.machine")
    checkers["local"] = partial(
        module.check_local_environment,
        disk_pool=local_disk_pool,
        resource_evidence=local_resource_evidence(local_disk_pool),
    )
    return checkers


def requirement_probe(requirement: Requirement, backend: str) -> tuple[ProbeRequirements, MachineSpec] | None:
    """Generate a minimal spec that isolates one demand on a backend's native source."""
    semantics = CommandSemantics.SHELL_SIMULATOR if backend == "shellsim" else CommandSemantics.LINUX_PROCESS
    source = (
        PackageSpec()
        if backend == "shellsim"
        else (
            HostImage()
            if backend == "local"
            else QemuBundle(Path("/survey/bundle")) if backend == "qemu" else RegistryImage(PROBE_IMAGE)
        )
    )
    spec = MachineSpec(source)
    requirements = ProbeRequirements(command_semantics=semantics)
    sources = {
        Requirement.DOCKER_IMAGE: RegistryImage(PROBE_IMAGE),
        Requirement.DOCKER_BUILD: DockerfileSource(Path("/survey/context"), Path("/survey/context/Dockerfile")),
        Requirement.PREPARED_IMAGE: PreparedImage(Path("/survey/oci"), "sha256:" + "0" * 64, RegistryImage(PROBE_IMAGE)),
        Requirement.QEMU_BUNDLE: QemuBundle(Path("/survey/bundle")),
        Requirement.HOST_IMAGE: HostImage(),
        Requirement.PACKAGES_LOCK: HostImage(),
        Requirement.PACKAGE_SPEC: PackageSpec(),
    }
    if requirement in sources:
        return requirements, replace(spec, source=sources[requirement])
    if requirement in (Requirement.LINUX_PROCESS, Requirement.SHELL_SIMULATOR):
        return replace(requirements, command_semantics=CommandSemantics(requirement.value)), spec
    if requirement in {
        Requirement.SHELL,
        Requirement.FILESYSTEM,
        Requirement.PYTHON,
        Requirement.PROCESS,
        Requirement.NETWORK,
        Requirement.GIT_REPOSITORY,
        Requirement.RETAINED_COMMAND_CANCELLATION,
    }:
        return replace(requirements, capabilities=(requirement.value,)), replace(
            spec, network=NetworkPolicy.ALLOW if requirement == Requirement.NETWORK else NetworkPolicy.DENY
        )
    changes = {
        Requirement.NETWORK_ALLOW: {"network": NetworkPolicy.ALLOW},
        Requirement.NETWORK_DENY: {"network": NetworkPolicy.DENY},
        Requirement.STORAGE: {"storage_mb": 1024},
        Requirement.MEMORY: {"memory_mb": 1024},
        Requirement.CPUS: {"cpus": 1},
        Requirement.GPUS: {"gpus": 1},
        Requirement.WORKDIR: {"workdir": "/survey/workspace"},
        Requirement.ENV: {"env": {"SURVEY_LITERAL": "value"}},
        Requirement.STARTUP_TIMEOUT: {"startup_timeout": 60.0},
    }
    if requirement in changes:
        return requirements, replace(spec, **changes[requirement])
    return None


@dataclass(frozen=True)
class RuntimeSupport:
    support: Support
    evidence: str
    note: str


def _runtime_cell(backend: str, requirement: Requirement) -> RuntimeSupport:
    path = f"lib/shellbox/src/shellbox/backends/{backend}/machine.py"
    if requirement == Requirement.DOCKER_IN_DOCKER:
        return RuntimeSupport(
            Support.UNSUPPORTED, path + ":create", "No privileged Docker daemon or socket is exposed by this factory."
        )
    if requirement in {Requirement.WINDOWS, Requirement.TPU, Requirement.GPU_TYPES}:
        return RuntimeSupport(
            Support.UNSUPPORTED,
            "lib/shellbox/src/shellbox/machine.py:MachineSpec",
            "No field lowers this demand to Shellbox.",
        )
    if requirement == Requirement.MOUNTS:
        if backend == "local":
            return RuntimeSupport(
                Support.PARTIAL,
                path + ":_check_mounts",
                "HostImage read-only mounts; host paths must exist and avoid system directories.",
            )
        return RuntimeSupport(
            Support.UNSUPPORTED,
            "lib/shellbox/src/shellbox/machine.py:HostImage",
            "Mounts require HostImage, rejected by this factory.",
        )
    if requirement == Requirement.FILE_MTIME:
        if backend in {"local", "docker", "docker-gvisor"}:
            return RuntimeSupport(
                Support.VERIFY,
                path + ":upload",
                "Archive/copy transfer restores timestamps at archive precision; "
                "a live probe must grade the precision the demand needs.",
            )
        if backend in {"daytona", "iris"}:
            return RuntimeSupport(
                Support.VERIFY,
                path + ":upload",
                "Upload now restores the declared mtime via a guest touch; a live probe must grade it.",
            )
        return RuntimeSupport(
            Support.UNSUPPORTED,
            path + ":upload",
            "Single-file upload writes bytes without restoring the declared mtime.",
        )
    if requirement == Requirement.FILE_MODE and backend == "iris":
        return RuntimeSupport(
            Support.VERIFY,
            path + ":IrisMachine.upload",
            "Upload now restores declared permission bits via a guest chmod; a live probe must grade it.",
        )
    if requirement == Requirement.USER:
        if backend in {"qemu", "shellsim"}:
            return RuntimeSupport(
                Support.UNSUPPORTED, path + ":run", "Separate execution users are explicitly rejected."
            )
        return RuntimeSupport(
            Support.PARTIAL,
            path + ":run",
            "Guest account must exist; Daytona requires util-linux su; local requires host privilege.",
        )
    if requirement in {Requirement.TOOL_PROVIDERS, Requirement.SERVICES, Requirement.HEALTHCHECK, Requirement.SKILLS}:
        return RuntimeSupport(
            Support.VERIFY,
            "vendor/harbor/src/harbor/environments/base.py",
            "Adapter-owned orchestration; a live probe on the real backend must grade it.",
        )
    if requirement == Requirement.SETUP:
        return RuntimeSupport(
            Support.VERIFY,
            "lib/shellbox/src/shellbox/machine.py:Machine.run",
            "Production orchestration runs declared setup commands; whether the image supports each needs a live probe.",
        )
    if requirement == Requirement.CLEANUP_TIMEOUT:
        return RuntimeSupport(
            Support.VERIFY,
            "lib/shellbox/src/shellbox/machine.py:Machine.close",
            "Production orchestration bounds cleanup independently of factory creation; live verification pending.",
        )
    return RuntimeSupport(
        Support.VERIFY,
        path,
        "Implemented by the command/transfer path; factory acceptance does not "
        "execute a runtime probe, so a live probe must grade it.",
    )


def survey_backend(backend: str, check: Checker) -> list[CapabilityCell]:
    """Record real validation outcomes and implementation conditions for every demand."""
    cells = []
    for requirement in Requirement:
        probe = requirement_probe(requirement, backend)
        if probe is None:
            result = _runtime_cell(backend, requirement)
            cells.append(
                CapabilityCell(
                    backend=backend,
                    requirement=requirement,
                    support=result.support,
                    evidence=(result.evidence,),
                    note=result.note,
                )
            )
            continue
        requirements, spec = probe
        validator_backend = "docker" if backend == "docker-gvisor" else backend
        validator = "check_local_environment" if backend == "local" else "check_environment"
        evidence = f"lib/shellbox/src/shellbox/backends/{validator_backend}/machine.py:{validator}"
        try:
            check(requirements, spec)
        except UnsupportedMachineSpec as error:
            support, note = Support.UNSUPPORTED, str(error)
            if backend == "local" and requirement in {
                Requirement.CPUS,
                Requirement.MEMORY,
                Requirement.STORAGE,
            }:
                # the LOCAL machinery exists in code; this worker just lacks the
                # delegation or disk pool, so a live probe on a capable worker
                # must grade the cell instead of a hard unsupported
                support = Support.VERIFY
                note += " This worker lacks the enforcement machinery; a live probe on a capable worker grades it."
            if backend == "qemu" and requirement in {
                Requirement.DOCKER_IMAGE,
                Requirement.DOCKER_BUILD,
                Requirement.PREPARED_IMAGE,
            }:
                support = Support.PARTIAL
                note += "; configure real QEMU assets, image cache and bundle cache, then revalidate."
        else:
            support, note = (
                Support.SUPPORTED,
                "Generated probe accepted by the production validator; no machine acquired.",
            )
            if requirement in {Requirement.CPUS, Requirement.MEMORY, Requirement.STORAGE, Requirement.GPUS}:
                support = Support.VERIFY
                note += (
                    " The numeric probe is not a capacity or enforcement measurement;"
                    " a live probe on the real backend must grade it."
                )
                if backend == "iris" and requirement is Requirement.STORAGE:
                    note += (
                        " Enforcement requires one sandbox per task and a direct Kubernetes controller "
                        "advertising sandbox-storage-limit. SANDBOX writable-layer accounting is unqualified; "
                        "current controllers withhold that capability pending a real SANDBOX overflow."
                    )
                if backend == "docker" or backend == "docker-gvisor":
                    note += " Disk quotas depend on the host storage driver."
                if backend == "local":
                    note += (
                        " LOCAL memory/cpu enforcement is verified per worker when a delegated cgroup hierarchy "
                        "applies and reads back the demanded limits (see local_resource_evidence); otherwise a "
                        "live probe must grade it."
                    )
            if requirement == Requirement.STARTUP_TIMEOUT and backend != "daytona":
                support = Support.VERIFY
                note += (
                    " Factory ignores this field; rolloutengine supplies the outer"
                    " startup deadline and a live probe must grade it."
                )
            if backend == "iris" and requirement in {Requirement.NETWORK_DENY, Requirement.NETWORK_ALLOW}:
                support = Support.VERIFY
                note += " Iris deny permits DNS on Kubernetes; live egress and DNS probes must grade it."
        cells.append(
            CapabilityCell(backend=backend, requirement=requirement, support=support, evidence=(evidence,), note=note)
        )
    return cells


def capability_matrix(checkers: dict[str, Checker]) -> list[CapabilityCell]:
    return [cell for backend, check in checkers.items() for cell in survey_backend(backend, check)]
