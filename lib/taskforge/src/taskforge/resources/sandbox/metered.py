# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record each sandbox a workpiece starts, for cost accounting.

``MeteredFactory`` wraps a ``MachineFactory`` and records a ``MachineStarted`` for every ``create``, so the
author's, adversaries', solvers', graders' and verifiers' machines are all counted at one point. It returns the
wrapped factory's machine unchanged, because callers check machine types such as ``IrisMachine``. A machine's
lifetime and measured CPU and memory come from Iris, joined on ``MachineStarted.job``.
"""

import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from shellbox.backends.iris.machine import IrisMachine
from shellbox.machine import (
    Backend,
    DockerImage,
    ExecutionRequirements,
    Machine,
    MachineFactory,
    MachineSpec,
    RegistryImage,
)

from taskforge.effects.events import Event
from taskforge.effects.issues import describe_exception
from taskforge.resources.execution import Notify


@dataclass(frozen=True, slots=True)
class MachineStarted(Event[None]):
    """One sandbox ``create``, successful or not.

    Attributes:
        job: The Iris job id of the sandbox; empty for a backend without jobs, such as ShellSim.
        backend: The factory's backend.
        image: The registry or Docker image reference, or empty for another machine source.
        cpus, memory_mb: The machine limits requested; ``None`` leaves them to the factory.
        seconds: Time spent in ``create``: scheduling, image pull and start.
        error: Why ``create`` failed, or ``None`` when the machine started.
    """

    job: str
    backend: str
    image: str
    cpus: int | None
    memory_mb: int | None
    seconds: float
    error: str | None


class MachineMeter:
    """Where ``MeteredFactory`` sends its events; ``install`` points ``notify`` at the workpiece's context."""

    def __init__(self) -> None:
        self.notify: Notify = logging.info

    async def record(self, event: MachineStarted) -> None:
        recorded = self.notify(event)
        if inspect.isawaitable(recorded):
            await recorded


def image_reference(spec: MachineSpec) -> str:
    return spec.source.reference if isinstance(spec.source, (RegistryImage, DockerImage)) else ""


@dataclass(frozen=True, slots=True)
class MeteredFactory(MachineFactory):
    factory: MachineFactory
    meter: MachineMeter

    @property
    @override
    def backend(self) -> Backend:
        return self.factory.backend

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        started = time.monotonic()
        try:
            machine = await self.factory.create(spec)
        except Exception as error:
            await self.meter.record(self._event(spec, "", started, describe_exception(error)))
            raise
        job = machine.job.job_id.to_wire() if isinstance(machine, IrisMachine) else ""
        await self.meter.record(self._event(spec, job, started, None))
        return machine

    def _event(self, spec: MachineSpec, job: str, started: float, error: str | None) -> MachineStarted:
        return MachineStarted(
            job,
            self.backend.value,
            image_reference(spec),
            spec.cpus,
            spec.memory_mb,
            time.monotonic() - started,
            error,
        )


def metered(factories: Mapping[str, MachineFactory], meter: MachineMeter) -> dict[str, MachineFactory]:
    return {name: MeteredFactory(factory, meter) for name, factory in factories.items()}
