# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine, and what each one can run.

``machine_factories`` returns the factories ``ShellboxRolloutEngine`` takes, keyed by shellbox
``Backend`` value as ``MachineRuntimeSpec.backend`` names them. ``factory_capabilities`` describes the
same factories so validation can refuse a lowered task up front with a typed reason
(``task_refusals``) instead of failing inside ``MachineFactory.create`` after a trial started.

These factories are ShellSim's alone: a task lowered onto another backend is refused with
``NO_FACTORY``. ``validate.trials.lower`` gives a task machine no network, user, resource limits or
GPUs and gives no task a verifier machine, so only the task machine's image and file timestamps are
checked.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory


@dataclass(frozen=True)
class FactoryCapabilities:
    """What one machine factory accepts.

    ``image`` is True when the backend needs a digest-pinned registry image and False when it runs
    ShellSim builtins without one.
    """

    image: bool
    file_timestamps: bool


class RefusalReason(StrEnum):
    NO_FACTORY = "no_factory"
    IMAGE = "image"
    FILE_TIMESTAMPS = "file_timestamps"


@dataclass(frozen=True)
class Refusal:
    """One reason a lowered task's machine cannot run on these factories."""

    reason: RefusalReason
    detail: str


# ShellSim runs builtins only; RolloutEngine refuses explicit resource timestamps on it.
SHELLSIM = FactoryCapabilities(image=False, file_timestamps=False)


def factory_capabilities() -> Mapping[str, FactoryCapabilities]:
    """What the factories ``machine_factories()`` returns can run, keyed the same way."""
    return {Backend.SHELLSIM.value: SHELLSIM}


def machine_factories() -> Mapping[str, MachineFactory]:
    """The factories ``ShellboxRolloutEngine`` takes, keyed by ``factory.backend.value``: ShellSim's alone."""
    factories: list[MachineFactory] = [ShellSimMachineFactory()]
    return {factory.backend.value: factory for factory in factories}


def task_refusals(lowered: LoweredTaskSpec, capabilities: Mapping[str, FactoryCapabilities]) -> list[Refusal]:
    """Every reason the task machine of ``lowered`` cannot run on these factories; none without one."""
    selection = lowered.runtime.task_machine
    if selection is None:
        return []
    capability = capabilities.get(selection.backend)
    if capability is None:
        return [Refusal(RefusalReason.NO_FACTORY, f"{selection.backend}: no factory for this backend")]
    task = lowered.task
    refusals = []
    if (task.environment_requirements.docker_image is not None) != capability.image:
        needed = "a digest-pinned registry image" if capability.image else "no image (ShellSim builtins)"
        refusals.append(Refusal(RefusalReason.IMAGE, f"{selection.backend} needs {needed}"))
    stamped = sorted(
        resource.path for resource in task.resources.all + task.resources.worker if resource.mtime_ns is not None
    )
    if stamped and not capability.file_timestamps:
        refusals.append(Refusal(RefusalReason.FILE_TIMESTAMPS, f"explicit timestamps on {stamped}"))
    return refusals
