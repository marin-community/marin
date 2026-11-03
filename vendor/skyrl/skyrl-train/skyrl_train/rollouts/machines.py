"""Keep task machines and late creation results owned until worker shutdown."""

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from loguru import logger
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineFactory, MachineSpec, Result

_PASSTHROUGH_ERRORS = (TypeError, ValueError, AttributeError, LookupError, AssertionError)


class MachineOperation(StrEnum):
    CREATE = "create"
    RUN = "run"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    CLOSE = "close"


class TaskMachineError(RuntimeError):
    """A task-machine provider operation failed."""

    def __init__(self, message: str, operation: MachineOperation):
        super().__init__(message, operation)
        self.operation = operation

    def __str__(self) -> str:
        return self.args[0]


@dataclass(eq=False)
class _OwnedMachine:
    machine: Machine
    owner: "OwnedMachineFactory"
    closing: asyncio.Task[None] | None = None

    async def run(self, command: Command) -> Result:
        try:
            return await self.machine.run(command)
        except _PASSTHROUGH_ERRORS:
            raise
        except Exception as error:
            raise TaskMachineError("Task-machine command failed", MachineOperation.RUN) from error

    async def upload(self, source: Path, target: str) -> None:
        try:
            await self.machine.upload(source, target)
        except _PASSTHROUGH_ERRORS:
            raise
        except Exception as error:
            raise TaskMachineError("Task-machine upload failed", MachineOperation.UPLOAD) from error

    async def download(self, source: str, target: Path) -> None:
        try:
            await self.machine.download(source, target)
        except _PASSTHROUGH_ERRORS:
            raise
        except Exception as error:
            raise TaskMachineError("Task-machine download failed", MachineOperation.DOWNLOAD) from error

    def begin_close(self) -> asyncio.Task[None]:
        if self.closing is None:
            self.closing = asyncio.create_task(self.machine.close())
            self.closing.add_done_callback(self._closed)
        return self.closing

    def _closed(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            exception_type = "CancelledError"
        elif (error := task.exception()) is not None:
            exception_type = type(error).__name__
        else:
            self.owner.machines.discard(self)
            return
        logger.bind(provider=self.owner.identifier, operation="machine_close", exception_type=exception_type).error(
            "Task-machine provider close failed"
        )

    async def close(self) -> None:
        await asyncio.shield(self.begin_close())


@dataclass
class OwnedMachineFactory:
    """Track provider creations and machines across rollout cancellation."""

    factory: MachineFactory
    identifier: str
    creating: set[asyncio.Task] = field(default_factory=set)
    machines: set[_OwnedMachine] = field(default_factory=set)
    closing: bool = False

    @property
    def backend(self) -> Backend:
        return self.factory.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    def stop_creating(self) -> None:
        self.closing = True

    async def create(self, spec: MachineSpec) -> Machine:
        if self.closing:
            raise TaskMachineError("The task-machine factory is closed", MachineOperation.CREATE)
        creation = asyncio.current_task()
        assert creation is not None
        self.creating.add(creation)
        try:
            try:
                machine = await self.factory.create(spec)
            except (TimeoutError, *_PASSTHROUGH_ERRORS):
                raise
            except Exception as error:
                raise TaskMachineError("Task-machine creation failed", MachineOperation.CREATE) from error
            owned = _OwnedMachine(machine, self)
            self.machines.add(owned)
            if self.closing:
                await owned.close()
                raise asyncio.CancelledError
            return owned
        finally:
            self.creating.discard(creation)

    async def close(self, deadline: float) -> None:
        """Release machines with the worker's remaining shutdown budget."""
        self.stop_creating()
        cleanups = {machine.begin_close() for machine in tuple(self.machines)}
        pending = cleanups | self.creating
        if pending:
            _, pending = await asyncio.wait(pending, timeout=max(0, deadline - asyncio.get_running_loop().time()))
        failures = []
        for task in cleanups:
            if not task.done():
                continue
            if task.cancelled():
                failures.append(TaskMachineError("Task-machine provider close was cancelled", MachineOperation.CLOSE))
            elif (error := task.exception()) is not None:
                failures.append(error)
        if pending:
            logger.bind(
                provider=self.identifier, open_creations=len(self.creating), open_machines=len(self.machines)
            ).error("Task-machine shutdown deadline expired")
            failures.append(TimeoutError(f"Task-machine shutdown deadline expired for provider {self.identifier}"))
        if failures:
            raise ExceptionGroup("Task-machine cleanup failed", failures)
