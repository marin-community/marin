# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine, and what each one can run.

``machine_factories`` resolves the ``EnvironmentKind -> MachineFactory`` mapping that
``ShellboxRolloutEngine`` takes, once, for where Taskforge is running. ``factory_capabilities``
describes the same factories so validation can refuse a task up front with a typed reason
(``task_refusals``) instead of failing inside ``MachineFactory.create`` after a trial started.
The capability table mirrors the checks each shellbox backend makes at create time.
"""

import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineFactory, NetworkPolicy
from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    FileReward,
    MissingArtifactPolicy,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.execution import TaskExecution
from taskcompendium.models import TaskSpec, VerifierKind, VerifierSpec

DOCKER_PROBE_TIMEOUT = 20
# ShellSim accepts no execution user other than root (shellbox.backends.shellsim.machine.ShellSimMachine.run).
SHELLSIM_USERS = frozenset({"0", "root"})
# The user RolloutEngine's own grading commands run as on the task machine.
ROOT_USER = "0"


class MachineHost(StrEnum):
    """Where the factories run: a developer laptop, or inside an Iris task."""

    LAPTOP = "laptop"
    IRIS = "iris"


class ImageSourceKind(StrEnum):
    """The TaskCompendium image kinds (``RegistryImage.kind`` and ``DockerBuild.kind``)."""

    REGISTRY = "registry"
    BUILD = "build"


@dataclass(frozen=True)
class FactoryCapabilities:
    """What one machine factory accepts. ``unavailable`` names why there is no factory at all."""

    image_sources: frozenset[ImageSourceKind]
    network: frozenset[NetworkPolicy]
    execution_users: bool
    cpu_and_storage_limits: bool
    gpus: bool
    file_timestamps: bool
    unavailable: str | None = None


class RefusalReason(StrEnum):
    NO_FACTORY = "no_factory"
    IMAGE_SOURCE = "image_source"
    NETWORK = "network"
    EXECUTION_USER = "execution_user"
    RESOURCE_LIMITS = "resource_limits"
    GPUS = "gpus"
    FILE_TIMESTAMPS = "file_timestamps"


@dataclass(frozen=True)
class Refusal:
    """One reason a task cannot run on these factories. ``where`` names the environment's role."""

    reason: RefusalReason
    where: str
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
            unavailable="skopeo not found on PATH (shellbox prepares registry images and Dockerfiles with it)",
        )
    return LocalDocker(skopeo=Path(skopeo), unavailable=None)


# RolloutEngine refuses explicit file timestamps (``EnvironmentFile.mtime_ns``) on ShellSim.
SHELLSIM = FactoryCapabilities(
    image_sources=frozenset(),
    network=frozenset({NetworkPolicy.DENY}),
    execution_users=False,
    cpu_and_storage_limits=False,
    gpus=False,
    file_timestamps=False,
)
LOCAL_DOCKER = FactoryCapabilities(
    image_sources=frozenset({ImageSourceKind.REGISTRY, ImageSourceKind.BUILD}),
    network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
    execution_users=True,
    cpu_and_storage_limits=True,
    gpus=False,
    file_timestamps=True,
)
# shellbox.backends.iris refuses Command.user and GPUs, and takes only registry references. Its
# create fails on every call today, so the factory is reported unavailable. ``network`` is what the
# backend provides once its readiness poll is fixed upstream: marin's gVisor sandboxes have no DNS
# or egress (measured on the marin GCP workers), so the fixed factory
# declares DENY and refuses ALLOW. The shipped backend instead refuses DENY and claims ALLOW.
# Its upload writes file bytes through shell commands, so an explicit file timestamp is lost.
IRIS_DOCKER = FactoryCapabilities(
    image_sources=frozenset({ImageSourceKind.REGISTRY}),
    network=frozenset({NetworkPolicy.DENY}),
    execution_users=False,
    cpu_and_storage_limits=True,
    gpus=False,
    file_timestamps=False,
    unavailable=(
        "shellbox IrisMachineFactory.create fails before any sandbox starts (its readiness poll compares "
        "iris TaskState to proto ints)"
    ),
)


def factory_capabilities(where: MachineHost) -> dict[EnvironmentKind, FactoryCapabilities]:
    """What the factories ``machine_factories(where, ...)`` returns can run, per environment kind."""
    if where is MachineHost.IRIS:
        return {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: IRIS_DOCKER}
    docker = local_docker()
    if docker.unavailable is not None:
        unavailable = FactoryCapabilities(
            frozenset(), frozenset(), False, False, False, False, unavailable=docker.unavailable
        )
        return {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: unavailable}
    return {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: LOCAL_DOCKER}


def machine_factories(
    where: MachineHost, controller_url: str | None, image_cache: Path | None
) -> Mapping[EnvironmentKind, MachineFactory]:
    """The factories ``ShellboxRolloutEngine`` takes. DOCKER is absent when the laptop has no Docker.

    On Iris the DOCKER factory submits sandboxes to ``controller_url``, the controller of the task
    Taskforge runs in, and ``image_cache`` must be ``None``. On a laptop ``controller_url`` must be
    ``None`` and ``image_cache`` is the directory where the Docker factory keeps the registry images
    and Dockerfile builds it prepares with Skopeo.
    """
    if (where is MachineHost.IRIS) != (controller_url is not None):
        raise ValueError(f"{where} factories take a controller URL only on Iris, got {controller_url!r}")
    if (where is MachineHost.LAPTOP) != (image_cache is not None):
        raise ValueError(f"{where} factories take an image cache only on a laptop, got {image_cache!r}")
    factories: dict[EnvironmentKind, MachineFactory] = {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}
    if controller_url is not None:
        factories[EnvironmentKind.DOCKER] = IrisMachineFactory(controller_url=controller_url)
        return factories
    docker = local_docker()
    if docker.skopeo is not None:
        factories[EnvironmentKind.DOCKER] = DockerMachineFactory(skopeo=docker.skopeo, image_cache=image_cache)
    return factories


def environment_refusals(
    environment: EnvironmentSpec,
    users: tuple[str | None, ...],
    capabilities: Mapping[EnvironmentKind, FactoryCapabilities],
    where: str,
    installed: tuple[EnvironmentFile, ...] = (),
) -> list[Refusal]:
    """Why ``environment`` cannot be created, run commands as ``users``, or receive the ``installed``
    files (beyond its own) on these factories."""
    if environment.kind == EnvironmentKind.NULL:
        return []
    capability = capabilities.get(environment.kind)
    if capability is None or capability.unavailable is not None:
        detail = capability.unavailable if capability is not None else "no factory for this kind"
        return [Refusal(RefusalReason.NO_FACTORY, where, f"{environment.kind}: {detail}")]
    refusals = []
    if environment.image is not None and ImageSourceKind(environment.image.kind) not in capability.image_sources:
        hint = " (publish it with an ImageBuilder first)" if isinstance(environment.image, DockerBuild) else ""
        refusals.append(
            Refusal(RefusalReason.IMAGE_SOURCE, where, f"{environment.image.kind} images are not accepted{hint}")
        )
    network = NetworkPolicy.ALLOW if environment.network else NetworkPolicy.DENY
    if network not in capability.network:
        refusals.append(Refusal(RefusalReason.NETWORK, where, f"network policy {network} is not provided"))
    commands = [*environment.setup, *([environment.healthcheck.command] if environment.healthcheck else [])]
    requested = {user for user in (*users, *(command.user for command in commands)) if user is not None}
    if environment.kind == EnvironmentKind.SHELLSIM:
        requested -= SHELLSIM_USERS
    if requested and not capability.execution_users:
        refusals.append(Refusal(RefusalReason.EXECUTION_USER, where, f"execution users {sorted(requested)}"))
    if (environment.cpus is not None or environment.storage_mb is not None) and not capability.cpu_and_storage_limits:
        refusals.append(Refusal(RefusalReason.RESOURCE_LIMITS, where, "cpus and storage_mb limits are not provided"))
    if environment.gpus and not capability.gpus:
        refusals.append(Refusal(RefusalReason.GPUS, where, f"{environment.gpus} GPUs requested"))
    stamped = sorted(file.path for file in (*environment.files, *installed) if file.mtime_ns is not None)
    if stamped and not capability.file_timestamps:
        refusals.append(Refusal(RefusalReason.FILE_TIMESTAMPS, where, f"explicit timestamps on {stamped}"))
    return refusals


def _shell_verifier(spec: VerifierSpec) -> ShellVerifierSpec | None:
    return ShellVerifierSpec.model_validate_json(spec.parameters_json) if spec.kind == VerifierKind.SHELL else None


def _artifact_needs_root(artifact: VerifierArtifact) -> bool:
    """Whether RolloutEngine inspects or archives ``artifact`` with a root command on the task machine.

    Follows the grading commands that ``docs/references/task-rollouts.md`` says the engine runs as
    user ``0`` in the agent machine: AUTO kinds and SKIP policies probe the source, and directories
    with excludes are archived and the archive removed.
    """
    if artifact.kind == ArtifactKind.AUTO or artifact.missing == MissingArtifactPolicy.SKIP:
        return True
    return artifact.kind == ArtifactKind.DIRECTORY and bool(artifact.exclude)


def _stage_grader_removal_needs_root(spec: VerifierSpec, shell: ShellVerifierSpec) -> bool:
    """Whether the engine removes this stage grader's private and reward files as user ``0`` before the next stage.

    ``docs/references/task-rollouts.md`` lists this removal among the grading commands run as user ``0``.
    """
    return spec.environment is None and bool(spec.files or isinstance(shell.reward, FileReward))


def task_refusals(
    task: TaskSpec, execution: TaskExecution, capabilities: Mapping[EnvironmentKind, FactoryCapabilities]
) -> list[Refusal]:
    """Every reason ``task`` cannot run with ``execution`` on these factories.

    Covers the task machine (agent, stage, collect and in-machine verifier users, plus the root
    commands RolloutEngine itself runs there to fetch grading artifacts and remove stage graders,
    and every file installed there) and each shell verifier's separate grading environment.
    """
    stages = tuple(execution.stages.values())
    stage_commands = [command for stage in stages for command in stage.setup]
    stage_commands += [stage.healthcheck.command for stage in stages if stage.healthcheck is not None]
    verifiers = [("verifier", task.verifier)] + [
        (f"stage {stage.name} verifier", stage.verifier) for stage in task.stages
    ]
    shell_verifiers = [(where, spec, shell) for where, spec in verifiers if (shell := _shell_verifier(spec))]
    engine_root = any(
        _artifact_needs_root(artifact)
        for _, spec, shell in shell_verifiers
        if spec.environment is not None
        for artifact in shell.artifacts
    ) or any(
        _stage_grader_removal_needs_root(stage.verifier, shell)
        for stage in task.stages
        if (shell := _shell_verifier(stage.verifier))
    )
    task_users = (
        execution.agent_user,
        *(stage.agent_user for stage in stages),
        *(command.user for command in stage_commands),
        *(command.user for _, _, shell in shell_verifiers for command in shell.collect),
        *(shell.user for _, spec, shell in shell_verifiers if spec.environment is None),
        *((ROOT_USER,) if engine_root else ()),
    )
    task_files = (
        *(file for stage in stages for file in stage.workdir_files),
        *(file for _, spec in verifiers if spec.environment is None for file in spec.files),
    )
    refusals = environment_refusals(task.environment, task_users, capabilities, "task", task_files)
    for where, spec, shell in shell_verifiers:
        if spec.environment is not None:
            refusals += environment_refusals(spec.environment, (shell.user,), capabilities, where, spec.files)
    return refusals
