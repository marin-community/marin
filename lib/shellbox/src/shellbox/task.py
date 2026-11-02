# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named containers sharing a task's network and lifetime."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol

from shellbox.machine import (
    Command,
    ExitReason,
    Machine,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
    UnsupportedMachineSpec,
)

_TIMED_PROBE = """\
import os
import signal
import subprocess
import sys

process = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    try:
        result = process.wait(timeout=float(sys.argv[1]))
    except subprocess.TimeoutExpired:
        result = 124
finally:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
sys.exit(result if result >= 0 else 128 - result)
"""


@dataclass(frozen=True)
class ServiceEndpoint:
    name: str
    port: int


@dataclass(frozen=True)
class ReadinessProbe:
    command: Command
    timeout: float
    interval: float
    start_period: float = 0
    attempts: int | None = None

    def __post_init__(self) -> None:
        if self.timeout <= 0 or self.interval <= 0 or self.start_period < 0:
            raise ValueError("Readiness timing requires positive timeout/interval and a nonnegative start period")
        if self.attempts is not None and self.attempts <= 0:
            raise ValueError("Readiness attempts must be positive")


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    machine: MachineSpec
    endpoints: tuple[ServiceEndpoint, ...] = ()
    readiness: ReadinessProbe | None = None
    entrypoint: tuple[str, ...] | None = None
    arguments: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TaskSpec:
    """One or more containers, with a primary container for default execution.

    All members share a network policy. Backend admission decides which policies
    and container counts it can enforce; sandbox restrictions are not part of
    this generic description.
    """

    containers: tuple[ContainerSpec, ...]
    primary: str = "task"

    def __post_init__(self) -> None:
        names = [container.name for container in self.containers]
        if len(set(names)) != len(names):
            raise ValueError("Container names must be unique within a task")
        if self.primary not in names:
            raise ValueError("A task must name one of its containers as primary")
        if any(container.machine.network is not self.network for container in self.containers):
            raise UnsupportedMachineSpec("Containers sharing a task must use the same network policy")
        ports = [endpoint.port for container in self.containers for endpoint in container.endpoints]
        if len(set(ports)) != len(ports):
            raise ValueError("Containers sharing loopback must use distinct endpoint ports")

    @property
    def primary_container(self) -> ContainerSpec:
        return next(container for container in self.containers if container.name == self.primary)

    @property
    def network(self) -> NetworkPolicy:
        return self.primary_container.machine.network


def single_container_task(spec: MachineSpec) -> TaskSpec:
    """Normalize the machine convenience API to the task execution model."""
    return TaskSpec((ContainerSpec("task", spec),))


class Task:
    """Named container handles owned and closed as one task."""

    def __init__(
        self,
        containers: Mapping[str, Machine],
        spec: TaskSpec,
        close: Callable[[], Awaitable[None]],
    ):
        self.containers = MappingProxyType(dict(containers))
        self.spec = spec
        self._close = close
        self._closing: asyncio.Future[None] | None = None

    @property
    def primary(self) -> Machine:
        return self.containers[self.spec.primary]

    def endpoint(self, container: str, endpoint: str) -> str:
        """Return a task-private address usable from its containers."""
        selected = next(member for member in self.spec.containers if member.name == container)
        port = next(item.port for item in selected.endpoints if item.name == endpoint)
        return f"{container}:{port}"

    async def wait_ready(self) -> None:
        """Wait for each declared probe, reclaiming the whole group on failure."""
        try:
            for container in self.spec.containers:
                if container.readiness is not None:
                    await wait_service_ready(self.containers[container.name], container.readiness)
        except BaseException:
            await asyncio.shield(self.close())
            raise

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._close())
        await asyncio.shield(self._closing)


class TaskFactory(MachineFactory, Protocol):
    """A factory accepting singleton and multi-container task descriptions."""

    async def create_task(self, spec: TaskSpec) -> Task: ...


async def wait_service_ready(machine: Machine, probe: ReadinessProbe) -> None:
    """Run an image-owned readiness command within its bounded startup budget."""
    command = probe.command
    if command.timeout is not None:
        # Machine timeouts may destroy the machine. Keep individual failed
        # health checks inside the guest; only the overall budget cancels it.
        command = replace(
            command,
            argv=("python3", "-c", _TIMED_PROBE, str(command.timeout), *command.argv),
            timeout=None,
        )
    async with asyncio.timeout(probe.timeout):
        if probe.start_period:
            await asyncio.sleep(probe.start_period)
        attempted = 0
        while True:
            result = await machine.run(command)
            if result.reason is ExitReason.EXITED and result.exit_code == 0:
                return
            attempted += 1
            if probe.attempts is not None and attempted >= probe.attempts:
                raise RuntimeError(
                    f"Service readiness failed after {attempted} attempts: {result.stderr.decode(errors='replace')}"
                )
            await asyncio.sleep(probe.interval)
