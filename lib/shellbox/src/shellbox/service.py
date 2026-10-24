# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private service groups with separate filesystems and one shared loopback network."""

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
class ServiceSpec:
    name: str
    machine: MachineSpec
    endpoints: tuple[ServiceEndpoint, ...] = ()
    readiness: ReadinessProbe | None = None
    entrypoint: tuple[str, ...] | None = None
    arguments: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ServiceGroupSpec:
    task: MachineSpec
    services: tuple[ServiceSpec, ...]

    def __post_init__(self) -> None:
        if self.task.network is not NetworkPolicy.DENY or any(
            service.machine.network is not NetworkPolicy.DENY for service in self.services
        ):
            raise UnsupportedMachineSpec("Private service groups require denied external networking for every member")
        names = [service.name for service in self.services]
        if len(set(names)) != len(names):
            raise ValueError("Service names must be unique within a group")
        ports = [endpoint.port for service in self.services for endpoint in service.endpoints]
        if len(set(ports)) != len(ports):
            raise ValueError("Services sharing loopback must use distinct endpoint ports")


class ServiceGroup:
    """One private network and its task and service machines, closed together."""

    def __init__(
        self,
        task: Machine,
        services: Mapping[str, Machine],
        spec: ServiceGroupSpec,
        close: Callable[[], Awaitable[None]],
    ):
        self.task = task
        self.services = MappingProxyType(dict(services))
        self.spec = spec
        self._close = close
        self._closing: asyncio.Future[None] | None = None

    def endpoint(self, service: str, endpoint: str) -> str:
        """Return the group-private host and port, usable only from group machines."""
        selected = next(member for member in self.spec.services if member.name == service)
        port = next(item.port for item in selected.endpoints if item.name == endpoint)
        return f"{service}:{port}"

    async def wait_ready(self) -> None:
        """Wait for each declared probe, reclaiming the whole group on failure."""
        try:
            for service in self.spec.services:
                if service.readiness is not None:
                    await wait_service_ready(self.services[service.name], service.readiness)
        except BaseException:
            await asyncio.shield(self.close())
            raise

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._close())
        await asyncio.shield(self._closing)


class ServiceGroupFactory(MachineFactory, Protocol):
    """A machine factory that also creates private groups of task services."""

    async def create_group(self, spec: ServiceGroupSpec) -> ServiceGroup: ...


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
