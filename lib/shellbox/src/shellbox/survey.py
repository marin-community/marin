# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline capability survey using production factory validation; never acquire machines."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict

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


def production_checkers() -> dict[str, Checker]:
    """Return all production validators without acquiring machines or provider clients."""
    checkers = {}
    for backend, class_name in (
        ("docker", "DockerMachineFactory"),
        ("gvisor", "GvisorMachineFactory"),
        ("qemu", "QemuMachineFactory"),
        ("shellsim", "ShellSimMachineFactory"),
        ("daytona", "DaytonaMachineFactory"),
        ("iris", "IrisMachineFactory"),
    ):
        module = importlib.import_module(f"shellbox.backends.{backend}.machine")
        factory_type = getattr(module, class_name)
        factory = factory_type(controller_url="http://survey.invalid") if backend == "iris" else factory_type()
        checkers[backend] = factory.check_environment
    module = importlib.import_module("shellbox.backends.local.machine")
    checkers["local"] = module.check_local_environment
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
        if backend in {"local", "docker", "gvisor"}:
            return RuntimeSupport(
                Support.PARTIAL,
                path + ":upload",
                "Archive/copy transfer preserves timestamp at archive precision; nanoseconds are not guaranteed.",
            )
        return RuntimeSupport(
            Support.UNSUPPORTED,
            path + ":upload",
            "Single-file upload writes bytes without restoring the declared mtime.",
        )
    if requirement == Requirement.FILE_MODE and backend == "iris":
        return RuntimeSupport(
            Support.UNSUPPORTED, path + ":IrisMachine.upload", "Single-file upload does not restore permission bits."
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
            Support.PARTIAL,
            "vendor/harbor/src/harbor/environments/base.py",
            "Adapter-owned orchestration; not guaranteed by a Shellbox factory check.",
        )
    if requirement == Requirement.SETUP:
        return RuntimeSupport(
            Support.PARTIAL,
            "lib/rolloutengine/src/rolloutengine/machines.py:_prepare_machine",
            "Runtime executes sh commands as root after creation; commands must be supported by the image or simulator.",
        )
    if requirement == Requirement.CLEANUP_TIMEOUT:
        return RuntimeSupport(
            Support.PARTIAL,
            "lib/shellbox/src/shellbox/machine.py:Machine.close",
            "Cleanup limits require caller orchestration; the machine close contract has no timeout parameter.",
        )
    return RuntimeSupport(
        Support.PARTIAL,
        path,
        "Implemented by the command/transfer path; factory acceptance does not execute a runtime probe.",
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
        validator_backend = "docker" if backend == "gvisor" else backend
        validator = "check_local_environment" if backend == "local" else "check_environment"
        evidence = f"lib/shellbox/src/shellbox/backends/{validator_backend}/machine.py:{validator}"
        try:
            check(requirements, spec)
        except UnsupportedMachineSpec as error:
            support, note = Support.UNSUPPORTED, str(error)
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
                support = Support.PARTIAL
                note += " Numeric probe is not a capacity or enforcement measurement."
                if backend == "docker" or backend == "gvisor":
                    note += " Disk quotas require a compatible Docker storage driver; GPUs require host/runtime support."
                if backend == "local":
                    note += " CPU/memory enforcement depends on delegated cgroups; rolloutengine omits local memory."
                if backend == "iris" and requirement == Requirement.STORAGE:
                    note += " Iris disk is a scheduling request, not a filesystem quota."
            if requirement == Requirement.STARTUP_TIMEOUT and backend != "daytona":
                support = Support.PARTIAL
                note += " Factory ignores this field; rolloutengine supplies the outer startup deadline."
            if backend == "iris" and requirement in {Requirement.NETWORK_DENY, Requirement.NETWORK_ALLOW}:
                support = Support.PARTIAL
                note += " Iris deny permits DNS on Kubernetes; allow permits public internet only."
        cells.append(
            CapabilityCell(backend=backend, requirement=requirement, support=support, evidence=(evidence,), note=note)
        )
    return cells


def capability_matrix(checkers: dict[str, Checker]) -> list[CapabilityCell]:
    return [cell for backend, check in checkers.items() for cell in survey_backend(backend, check)]
