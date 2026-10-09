# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine, and what each one can run.

``machine_factories`` resolves, once for where Taskforge is running, the factories that
``ShellboxRolloutEngine`` takes, keyed by shellbox ``Backend`` value as ``MachineRuntimeSpec.backend``
names them. ``container_backend`` is the backend an image-backed machine is lowered onto on that host.
``factory_capabilities`` describes the same factories so validation can refuse a lowered task up front
with a typed reason (``task_refusals``) instead of failing inside ``MachineFactory.create`` after a trial
started. The capability table mirrors the checks each shellbox backend makes at create and run time.
"""

import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory, NetworkPolicy
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskResource, TaskSpec, VerifyitGrader

DOCKER_PROBE_TIMEOUT = 20


class MachineHost(StrEnum):
    """Where the factories run: a developer laptop, or inside an Iris task."""

    LAPTOP = "laptop"
    IRIS = "iris"


class MachineRole(StrEnum):
    """Which of a lowered task's machines a refusal concerns."""

    TASK = "task"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class FactoryCapabilities:
    """What one machine factory accepts. ``unavailable`` names why there is no factory at all.

    ``image`` is True when the backend needs a digest-pinned registry image and False when it runs
    ShellSim builtins without one. ``execution_users`` of None accepts any user; otherwise a command
    may name only those users.
    """

    image: bool
    network: frozenset[NetworkPolicy]
    execution_users: frozenset[str] | None
    cpu_and_storage_limits: bool
    gpus: bool
    file_timestamps: bool
    unavailable: str | None = None


class RefusalReason(StrEnum):
    NO_FACTORY = "no_factory"
    IMAGE = "image"
    NETWORK = "network"
    EXECUTION_USER = "execution_user"
    RESOURCE_LIMITS = "resource_limits"
    GPUS = "gpus"
    FILE_TIMESTAMPS = "file_timestamps"


@dataclass(frozen=True)
class Refusal:
    """One reason a lowered task cannot run on these factories."""

    reason: RefusalReason
    where: MachineRole
    detail: str


@dataclass(frozen=True)
class LocalDocker:
    """The laptop's Docker probe: the Skopeo binary shellbox needs, or why Docker machines are unavailable.

    Exactly one of ``skopeo`` and ``unavailable`` is set.
    """

    skopeo: Path | None
    unavailable: str | None


@cache
def local_docker() -> LocalDocker:
    """Probe the laptop once for a reachable Docker daemon and Skopeo."""
    docker = shutil.which("docker")
    if docker is None:
        return LocalDocker(skopeo=None, unavailable="docker CLI not found on PATH")
    probe = subprocess.run(
        (docker, "info", "--format", "{{.ServerVersion}}"), capture_output=True, text=True, timeout=DOCKER_PROBE_TIMEOUT
    )
    if probe.returncode:
        return LocalDocker(skopeo=None, unavailable=f"docker daemon unreachable: {probe.stderr.strip()[-500:]}")
    skopeo = shutil.which("skopeo")
    if skopeo is None:
        return LocalDocker(
            skopeo=None,
            unavailable="skopeo not found on PATH (shellbox pulls registry images with it)",
        )
    return LocalDocker(skopeo=Path(skopeo), unavailable=None)


@dataclass(frozen=True)
class LocalSandbox:
    """The Iris task's bubblewrap probe: a local factory for lock-only graders, or why there is none.

    Exactly one of ``factory`` and ``unavailable`` is set.
    """

    factory: LocalMachineFactory | None
    unavailable: str | None


@cache
def local_sandbox() -> LocalSandbox:
    """Probe once whether bubblewrap can sandbox commands here (it needs root or user namespaces on Linux)."""
    try:
        return LocalSandbox(factory=LocalMachineFactory(), unavailable=None)
    except SandboxUnavailable as error:
        return LocalSandbox(factory=None, unavailable=str(error))


ROOT_USERS = frozenset({"0", "root"})
# ShellSim runs builtins only, has no guest network, and refuses users other than root
# (shellbox.backends.shellsim.machine.ShellSimMachine.run). RolloutEngine refuses explicit resource
# timestamps on it.
SHELLSIM = FactoryCapabilities(
    image=False,
    network=frozenset({NetworkPolicy.DENY}),
    execution_users=ROOT_USERS,
    cpu_and_storage_limits=False,
    gpus=False,
    file_timestamps=False,
)
LOCAL_DOCKER = FactoryCapabilities(
    image=True,
    network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
    execution_users=None,
    cpu_and_storage_limits=True,
    gpus=False,
    file_timestamps=True,
)
# Iris exec cannot switch users: Command.user is accepted only when it equals the container's own
# uid or name. RolloutEngine runs setup commands, output capture, collect commands and artifact probes
# as user "0", so an image bound for gVisor must run as root, and no other user is accepted. Egress is
# MachineSpec.network through shellbox's EGRESS_POLICIES. Upload writes bytes through shell commands,
# so an explicit resource timestamp is lost.
IRIS_GVISOR = FactoryCapabilities(
    image=True,
    network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
    execution_users=ROOT_USERS,
    cpu_and_storage_limits=True,
    gpus=False,
    file_timestamps=False,
)
# The local backend runs lock-only graders in bubblewrap sandboxes on the host that runs Taskforge, with
# packages RolloutEngine builds from the lock; it allocates no CPU, memory, storage or GPUs, and RolloutEngine
# refuses explicit resource timestamps on any machine without an image.
LOCAL_SANDBOX = FactoryCapabilities(
    image=False,
    network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
    execution_users=ROOT_USERS,
    cpu_and_storage_limits=False,
    gpus=False,
    file_timestamps=False,
)


def _unavailable(reason: str) -> FactoryCapabilities:
    return FactoryCapabilities(
        image=True,
        network=frozenset(),
        execution_users=frozenset(),
        cpu_and_storage_limits=False,
        gpus=False,
        file_timestamps=False,
        unavailable=reason,
    )


def container_backend(where: MachineHost) -> Backend:
    """The backend an image-backed machine is lowered onto on ``where``."""
    if where is MachineHost.IRIS:
        return Backend.GVISOR
    return Backend.DOCKER


def factory_capabilities(where: MachineHost) -> Mapping[str, FactoryCapabilities]:
    """What the factories ``machine_factories(where, ...)`` returns can run, keyed the same way."""
    if where is MachineHost.IRIS:
        sandbox = local_sandbox()
        local = LOCAL_SANDBOX if sandbox.unavailable is None else _unavailable(sandbox.unavailable)
        return {Backend.SHELLSIM.value: SHELLSIM, Backend.GVISOR.value: IRIS_GVISOR, Backend.LOCAL.value: local}
    docker = local_docker()
    docker_row = LOCAL_DOCKER if docker.unavailable is None else _unavailable(docker.unavailable)
    return {Backend.SHELLSIM.value: SHELLSIM, Backend.DOCKER.value: docker_row}


def machine_factories(
    where: MachineHost, controller_url: str | None, image_cache: Path | None
) -> Mapping[str, MachineFactory]:
    """The factories ``ShellboxRolloutEngine`` takes, keyed by ``factory.backend.value``.

    On Iris the gVisor factory submits sandboxes to ``controller_url``, the controller of the task
    Taskforge runs in, and ``image_cache`` must be ``None``; the local factory, which grades lock-only
    environments, is present when bubblewrap can sandbox commands in the Iris task. On a laptop
    ``controller_url`` must be ``None`` and ``image_cache`` is the directory where the Docker factory
    keeps the registry images it pulls with Skopeo; the Docker factory is absent when the laptop has
    no Docker or Skopeo.
    """
    if (where is MachineHost.IRIS) != (controller_url is not None):
        raise ValueError(f"{where} factories take a controller URL only on Iris, got {controller_url!r}")
    if (where is MachineHost.LAPTOP) != (image_cache is not None):
        raise ValueError(f"{where} factories take an image cache only on a laptop, got {image_cache!r}")
    factories: list[MachineFactory] = [ShellSimMachineFactory()]
    if controller_url is not None:
        factories.append(IrisMachineFactory(controller_url=controller_url))
        if (local := local_sandbox().factory) is not None:
            factories.append(local)
    elif (skopeo := local_docker().skopeo) is not None:
        factories.append(DockerMachineFactory(skopeo=skopeo, image_cache=image_cache))
    return {factory.backend.value: factory for factory in factories}


def _machine_refusals(
    role: MachineRole,
    selection: MachineRuntimeSpec,
    requirements: EnvironmentRequirements,
    resources: tuple[TaskResource, ...],
    capabilities: Mapping[str, FactoryCapabilities],
) -> list[Refusal]:
    capability = capabilities.get(selection.backend)
    if capability is None or capability.unavailable is not None:
        detail = capability.unavailable if capability is not None else "no factory for this backend"
        return [Refusal(RefusalReason.NO_FACTORY, role, f"{selection.backend}: {detail}")]
    refusals = []
    if (requirements.docker_image is not None) != capability.image:
        needed = "a digest-pinned registry image" if capability.image else "no image (ShellSim builtins)"
        refusals.append(Refusal(RefusalReason.IMAGE, role, f"{selection.backend} needs {needed}"))
    if selection.network not in capability.network:
        refusals.append(Refusal(RefusalReason.NETWORK, role, f"network policy {selection.network} is not provided"))
    user = selection.user
    if user is not None and capability.execution_users is not None and user not in capability.execution_users:
        refusals.append(Refusal(RefusalReason.EXECUTION_USER, role, f"execution user {user!r}"))
    if (selection.cpus is not None or selection.storage_mb is not None) and not capability.cpu_and_storage_limits:
        refusals.append(Refusal(RefusalReason.RESOURCE_LIMITS, role, "cpus and storage_mb limits are not provided"))
    if selection.gpus and not capability.gpus:
        refusals.append(Refusal(RefusalReason.GPUS, role, f"{selection.gpus} GPUs requested"))
    stamped = sorted(resource.path for resource in resources if resource.mtime_ns is not None)
    if stamped and not capability.file_timestamps:
        refusals.append(Refusal(RefusalReason.FILE_TIMESTAMPS, role, f"explicit timestamps on {stamped}"))
    return refusals


def task_refusals(lowered: LoweredTaskSpec, capabilities: Mapping[str, FactoryCapabilities]) -> list[Refusal]:
    """Every reason ``lowered`` cannot run on these factories.

    Checks the task machine and the verifier machine, each against the capabilities of the backend it
    was lowered onto. The commands RolloutEngine itself runs as user ``0`` need no check: every row
    that restricts users accepts root.
    """
    task = lowered.task
    selections = (
        (
            MachineRole.TASK,
            lowered.runtime.task_machine,
            task.environment_requirements,
            task.resources.all + task.resources.worker,
        ),
        (
            MachineRole.VERIFIER,
            lowered.runtime.verifier_machine,
            grading_environment(task),
            task.resources.all + task.resources.worker + task.resources.verifier,
        ),
    )
    return [
        refusal
        for role, selection, requirements, resources in selections
        if selection is not None and requirements is not None
        for refusal in _machine_refusals(role, selection, requirements, resources, capabilities)
    ]


def grading_environment(task: TaskSpec) -> EnvironmentRequirements | None:
    """The environment of the task's verifier machine, or ``None`` when its grader takes none."""
    grader = task.grader
    if isinstance(grader, ScriptGrader | VerifyitGrader):
        return grader.environment
    return None
