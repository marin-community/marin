# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve task machines to trusted Iris Actor peers without replaying operations."""

import asyncio
import tarfile
import tempfile
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, cast
from uuid import uuid4

import cloudpickle
from iris.actor.client import ActorClient
from iris.actor.resolver import Resolver

from shellbox.backends.local.transfers import _guest_member, unpack_local_archive
from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    Machine,
    MachineFactory,
    MachineSpec,
    ProcessMachine,
    ProcessScope,
    ProcessScopeMachine,
    ProcessSession,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.process_session import acquire_owned, await_cleanup, finish_owned, own_cleanup

TRANSFER_CHUNK_BYTES = 65536
OPERATION_POLL_INTERVAL = 0.02


@dataclass(frozen=True)
class MachineServiceContract:
    """Execution support and per-machine capacity advertised by a ready service."""

    backend: Backend
    semantics: CommandSemantics
    capabilities: frozenset[str]
    images: tuple[str, ...]
    cpus: int
    memory_mb: int
    region: str
    architecture: str


def check_service_spec(contract: MachineServiceContract, spec: MachineSpec) -> RegistryImage:
    if not isinstance(spec.source, RegistryImage) or spec.source.reference not in contract.images:
        raise UnsupportedMachineSpec("Machine service requires one of its prepared registry images")
    if spec.storage_mb is not None or spec.gpus:
        raise UnsupportedMachineSpec("Machine service does not provide disk quotas or GPU allocations")
    if spec.cpus is None or spec.memory_mb is None:
        raise UnsupportedMachineSpec("Machine service requires explicit CPU and memory allocations")
    if not 0 < spec.cpus <= contract.cpus or not 0 < spec.memory_mb <= contract.memory_mb:
        raise UnsupportedMachineSpec("Machine allocation exceeds the service's per-machine capacity")
    return spec.source


class Operation(StrEnum):
    CREATE = "create"
    RUN = "run"
    CLOSE = "close"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    OPEN_PROCESS = "open_process"
    PROCESS_READ = "process_read"
    PROCESS_WRITE = "process_write"
    PROCESS_WAIT = "process_wait"
    PROCESS_CLOSE = "process_close"
    OPEN_SCOPE = "open_scope"
    SCOPE_RUN = "scope_run"
    SCOPE_TERMINATE = "scope_terminate"
    CLOSE_OWNER = "close_owner"


class OutcomeKind(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Outcome:
    kind: OutcomeKind
    value: object


@dataclass(frozen=True)
class ProcessRead:
    data: bytes
    stderr: bytes


@dataclass(frozen=True)
class ProcessExit:
    exit_code: int
    stderr: bytes


@dataclass
class MachineOwner:
    machines: dict[str, Machine] = field(default_factory=dict)
    processes: dict[str, tuple[str, ProcessSession]] = field(default_factory=dict)
    scopes: dict[str, tuple[str, ProcessScope]] = field(default_factory=dict)


class TransferBlob:
    """Own one temporary stream across multiple bounded RPC requests."""

    def __init__(self):
        with ExitStack() as resources:
            self.stream = resources.enter_context(tempfile.TemporaryFile())
            self.resources = resources.pop_all()

    def close(self) -> None:
        self.resources.close()


async def _complete_all(operations: list[Awaitable[object]]) -> None:
    results = await asyncio.gather(*operations, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise ExceptionGroup("Machine service cleanup failed", errors)


def _pack(source: Path, stream: BinaryIO) -> None:
    with tarfile.open(fileobj=stream, mode="w") as archive:
        archive.add(source, arcname="." if source.is_dir() else "payload")
    stream.seek(0)


def _unpack(stream: BinaryIO, root: Path) -> Path:
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:") as archive:
        first = archive.next()
        if first is None:
            raise ValueError("Machine service archive is empty")
        directory = first.name == "." and first.isdir()
        if not directory and (first.name != "payload" or not first.isfile()):
            raise ValueError("Machine service archive has an invalid root")
        archive.extractall(root, filter=_guest_member)
    return root if directory else root / "payload"


class MachineService:
    """Host one factory on a persistent event loop behind fast Actor RPC methods.

    Actor RPC uses cloudpickle and trusts its peers to execute host Python code.
    This service must use the same private or authenticated network boundary.
    Long operations run on the owned loop; polling and cancellation never wait
    behind an executor full of running commands.
    """

    def __init__(self, factory: MachineFactory, contract: MachineServiceContract, *, slots: int):
        if slots <= 0 or contract.cpus <= 0 or contract.memory_mb <= 0:
            raise ValueError("Machine service capacity must be positive")
        if factory.backend != contract.backend:
            raise ValueError("Machine service backend differs from its factory")
        self.factory = factory
        self.contract = contract
        self._slots = asyncio.Semaphore(slots)
        self._owners: dict[str, MachineOwner] = {}
        self._closed_owners: set[str] = set()
        self._operations: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._discarded: set[tuple[str, str]] = set()
        self._blobs: dict[tuple[str, str], TransferBlob] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="machine-service", daemon=True)
        self._thread.start()
        self._handlers: dict[Operation, Callable[..., Awaitable[object]]] = {
            Operation.CREATE: self._create,
            Operation.RUN: self._run,
            Operation.CLOSE: self._close_machine,
            Operation.UPLOAD: self._upload,
            Operation.DOWNLOAD: self._download,
            Operation.OPEN_PROCESS: self._open_process,
            Operation.PROCESS_READ: self._process_read,
            Operation.PROCESS_WRITE: self._process_write,
            Operation.PROCESS_WAIT: self._process_wait,
            Operation.PROCESS_CLOSE: self._process_close,
            Operation.OPEN_SCOPE: self._open_scope,
            Operation.SCOPE_RUN: self._scope_run,
            Operation.SCOPE_TERMINATE: self._scope_terminate,
            Operation.CLOSE_OWNER: self._close_owner,
        }

    def _on_loop[T](self, operation: Coroutine[object, object, T]) -> T:
        return asyncio.run_coroutine_threadsafe(operation, self._loop).result()

    def describe(self) -> MachineServiceContract:
        return self.contract

    def put(self, owner: str, blob: str, offset: int, data: bytes) -> None:
        self._on_loop(self._put(owner, blob, offset, data))

    async def _put(self, owner: str, blob: str, offset: int, data: bytes) -> None:
        if owner in self._closed_owners:
            raise RuntimeError("Machine service owner is closed")
        if len(data) > TRANSFER_CHUNK_BYTES:
            raise ValueError("Machine service transfer chunk is too large")
        key = owner, blob
        if key in self._discarded:
            raise RuntimeError("Machine service transfer was cancelled")
        if key not in self._blobs:
            if offset != 0:
                raise ValueError("Machine service transfer must start at zero")
            self._blobs[key] = TransferBlob()
        stream = self._blobs[key].stream
        if stream.seek(0, 2) != offset:
            raise ValueError("Machine service transfer offset changed; request must not be replayed")
        stream.write(data)

    def get(self, owner: str, blob: str, offset: int) -> bytes:
        return self._on_loop(self._get(owner, blob, offset))

    async def _get(self, owner: str, blob: str, offset: int) -> bytes:
        stream = self._blobs[owner, blob].stream
        stream.seek(offset)
        return stream.read(TRANSFER_CHUNK_BYTES)

    def start(self, owner: str, identifier: str, operation: Operation) -> None:
        self._on_loop(self._start(owner, identifier, operation))

    async def _start(self, owner: str, identifier: str, operation: Operation) -> None:
        if owner in self._closed_owners:
            raise RuntimeError("Machine service owner is closed")
        key = owner, identifier
        if key in self._discarded:
            raise RuntimeError("Machine service operation was cancelled before delivery")
        if key in self._operations:
            raise ValueError("Machine service operation must not be replayed")
        self._owners.setdefault(owner, MachineOwner())
        self._operations[key] = asyncio.create_task(self._execute(owner, identifier, operation))

    async def _execute(self, owner: str, identifier: str, operation: Operation) -> None:
        stream = self._blobs[owner, identifier].stream
        stream.seek(0)
        arguments = cloudpickle.load(stream)
        try:
            value = await self._handlers[operation](owner, *arguments)
            outcome = Outcome(OutcomeKind.SUCCESS, value)
        except asyncio.CancelledError:
            outcome = Outcome(OutcomeKind.CANCELLED, None)
        except Exception as error:
            outcome = Outcome(OutcomeKind.FAILURE, error)
        output = TransferBlob()
        try:
            cloudpickle.dump(outcome, output.stream)
        except BaseException:
            output.close()
            raise
        self._blobs[owner, identifier + "-result"] = output

    def ready(self, owner: str, identifier: str) -> bool:
        return self._on_loop(self._ready(owner, identifier))

    async def _ready(self, owner: str, identifier: str) -> bool:
        operation = self._operations[owner, identifier]
        if not operation.done():
            return False
        operation.result()
        return True

    def discard(self, owner: str, identifier: str) -> None:
        self._on_loop(self._discard(owner, identifier))

    async def _discard(self, owner: str, identifier: str) -> None:
        # A timed-out transport can deliver its first request after cancellation.
        # Retain the tombstone until this owner closes; never replay a mutation.
        self._discarded.add((owner, identifier))
        operation = self._operations.pop((owner, identifier), None)
        if operation is not None:
            operation.cancel()
            await _complete_all([operation])
        for name in (identifier, identifier + "-result"):
            stream = self._blobs.pop((owner, name), None)
            if stream is not None:
                stream.close()

    async def _create(self, owner: str, identifier: str, spec: MachineSpec) -> None:
        check_service_spec(self.contract, spec)
        await self._slots.acquire()
        try:
            machine = await acquire_owned(asyncio.create_task(self.factory.create(spec)))
        except BaseException:
            self._slots.release()
            raise
        self._owners[owner].machines[identifier] = machine

    async def _run(self, owner: str, machine: str, command: Command) -> Result:
        return await self._owners[owner].machines[machine].run(command)

    async def _close_machine(self, owner: str, identifier: str) -> None:
        state = self._owners[owner]
        machine = state.machines.pop(identifier, None)
        if machine is None:
            return
        operations: list[Awaitable[object]] = []
        for key, (machine_id, process) in tuple(state.processes.items()):
            if machine_id == identifier:
                del state.processes[key]
                operations.append(process.close())
        for key, (machine_id, scope) in tuple(state.scopes.items()):
            if machine_id == identifier:
                del state.scopes[key]
                operations.append(scope.terminate())

        async def destroy() -> None:
            try:
                await _complete_all(operations)
            finally:
                await machine.close()

        try:
            await await_cleanup(own_cleanup(destroy()))
        finally:
            self._slots.release()

    async def _upload(self, owner: str, machine: str, blob: str, target: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _unpack(self._blobs[owner, blob].stream, Path(temporary))
            await await_cleanup(own_cleanup(self._owners[owner].machines[machine].upload(source, target)))

    async def _download(self, owner: str, machine: str, source: str, blob: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "payload"
            await await_cleanup(own_cleanup(self._owners[owner].machines[machine].download(source, target)))
            output = TransferBlob()
            try:
                _pack(target, output.stream)
            except BaseException:
                output.close()
                raise
            self._blobs[owner, blob] = output

    async def _open_process(self, owner: str, machine: str, identifier: str, command: Command) -> None:
        selected = self._owners[owner].machines[machine]
        if not isinstance(selected, ProcessMachine):
            raise UnsupportedMachineSpec("Machine service factory does not support persistent process pipes")
        process = await acquire_owned(asyncio.create_task(selected.open_process(command)))
        self._owners[owner].processes[identifier] = machine, process

    async def _process_read(self, owner: str, identifier: str, maximum: int) -> ProcessRead:
        process = self._owners[owner].processes[identifier][1]
        data = await process.read(min(maximum, TRANSFER_CHUNK_BYTES))
        return ProcessRead(data, process.stderr)

    async def _process_write(self, owner: str, identifier: str, data: bytes) -> bytes:
        process = self._owners[owner].processes[identifier][1]
        await process.write(data)
        return process.stderr

    async def _process_wait(self, owner: str, identifier: str) -> ProcessExit:
        process = self._owners[owner].processes[identifier][1]
        return ProcessExit(await process.wait(), process.stderr)

    async def _process_close(self, owner: str, identifier: str) -> None:
        entry = self._owners[owner].processes.pop(identifier, None)
        if entry is not None:
            await entry[1].close()

    async def _open_scope(self, owner: str, machine: str, identifier: str) -> None:
        selected = cast(ProcessScopeMachine, self._owners[owner].machines[machine])
        scope = await selected.open_process_scope()
        self._owners[owner].scopes[identifier] = machine, scope

    async def _scope_run(self, owner: str, identifier: str, command: Command) -> Result:
        return await self._owners[owner].scopes[identifier][1].run(command)

    async def _scope_terminate(self, owner: str, identifier: str) -> None:
        entry = self._owners[owner].scopes.pop(identifier, None)
        if entry is not None:
            await entry[1].terminate()

    async def _close_owner(self, owner: str) -> None:
        self._closed_owners.add(owner)
        current = asyncio.current_task()
        operations = [task for (key, _), task in self._operations.items() if key == owner and task is not current]
        for task in operations:
            task.cancel()
        try:
            await _complete_all(operations)
        finally:
            await _complete_all([self._close_machine(owner, machine) for machine in tuple(self._owners[owner].machines)])

    async def _shutdown(self) -> None:
        try:
            await _complete_all([self._close_owner(owner) for owner in tuple(self._owners)])
        finally:
            for stream in self._blobs.values():
                stream.close()
            self._blobs.clear()

    def close(self) -> None:
        try:
            self._on_loop(self._shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()


class RemoteMachineFactory:
    """Forward machine operations once, with bounded RPC frames and owned cancellation."""

    def __init__(self, resolver: Resolver, name: str, *, rpc_timeout: float):
        self.client = ActorClient(resolver, name, call_timeout=rpc_timeout, max_call_attempts=1)
        self.contract = cast(MachineServiceContract, self.client.describe())
        self.owner = uuid4().hex

    async def _rpc[T](self, call: Callable[..., T], *arguments: object) -> T:
        task = asyncio.create_task(asyncio.to_thread(call, *arguments))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as interruption:
            try:
                await finish_owned(task)
            except Exception as error:
                interruption.add_note(f"Interrupted machine service RPC failed: {error}")
            raise

    @property
    def backend(self) -> Backend:
        return self.contract.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        source = check_service_spec(self.contract, spec)
        check_execution_requirements(
            requirements,
            spec,
            semantics=self.contract.semantics,
            capabilities=self.contract.capabilities,
            image_reference=source.reference,
        )

    async def _send(self, identifier: str, stream: BinaryIO) -> None:
        offset = 0
        while data := stream.read(TRANSFER_CHUNK_BYTES):
            await self._rpc(self.client.put, self.owner, identifier, offset, data)
            offset += len(data)
        if offset == 0:
            await self._rpc(self.client.put, self.owner, identifier, 0, b"")

    async def _receive(self, identifier: str, stream: BinaryIO) -> None:
        offset = 0
        while data := await self._rpc(self.client.get, self.owner, identifier, offset):
            stream.write(data)
            offset += len(data)
        stream.seek(0)

    async def _discard(self, identifier: str) -> None:
        await self._rpc(self.client.discard, self.owner, identifier)

    async def _call(self, operation: Operation, *arguments: object) -> object:
        identifier = uuid4().hex
        try:
            with tempfile.TemporaryFile() as stream:
                cloudpickle.dump(arguments, stream)
                stream.seek(0)
                await self._send(identifier, stream)
            await self._rpc(self.client.start, self.owner, identifier, operation)
            while not await self._rpc(self.client.ready, self.owner, identifier):
                await asyncio.sleep(OPERATION_POLL_INTERVAL)
            with tempfile.TemporaryFile() as stream:
                await self._receive(identifier + "-result", stream)
                outcome = cast(Outcome, cloudpickle.load(stream))
            if outcome.kind == OutcomeKind.FAILURE:
                raise cast(Exception, outcome.value)
            if outcome.kind == OutcomeKind.CANCELLED:
                raise asyncio.CancelledError
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self._discard(identifier)))
            except BaseException as cleanup_error:
                error.add_note(f"Machine service cancellation could not be confirmed: {cleanup_error}")
            raise
        await self._discard(identifier)
        return outcome.value

    async def create(self, spec: MachineSpec) -> "RemoteMachine":
        check_service_spec(self.contract, spec)
        identifier = uuid4().hex
        try:
            await self._call(Operation.CREATE, identifier, spec)
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self._call(Operation.CLOSE, identifier)))
            except BaseException as cleanup_error:
                error.add_note(f"Machine service late-create cleanup failed: {cleanup_error}")
            raise
        return RemoteMachine(self, identifier)

    async def close(self) -> None:
        await self._call(Operation.CLOSE_OWNER)


@dataclass
class RemoteMachine:
    factory: RemoteMachineFactory
    identifier: str

    async def run(self, command: Command) -> Result:
        return cast(Result, await self.factory._call(Operation.RUN, self.identifier, command))

    async def upload(self, source: Path, target: str) -> None:
        blob = uuid4().hex
        try:
            with tempfile.TemporaryFile() as stream:
                _pack(source, stream)
                await self.factory._send(blob, stream)
            await self.factory._call(Operation.UPLOAD, self.identifier, blob, target)
        finally:
            await finish_owned(own_cleanup(self.factory._discard(blob)))

    async def download(self, source: str, target: Path) -> None:
        blob = uuid4().hex
        try:
            await self.factory._call(Operation.DOWNLOAD, self.identifier, source, blob)
            with tempfile.TemporaryFile() as stream:
                await self.factory._receive(blob, stream)
                unpack_local_archive(stream.read(), target, source_name="payload")
        finally:
            await finish_owned(own_cleanup(self.factory._discard(blob)))

    async def close(self) -> None:
        await self.factory._call(Operation.CLOSE, self.identifier)

    async def open_process(self, command: Command) -> "RemoteProcess":
        identifier = uuid4().hex
        try:
            await self.factory._call(Operation.OPEN_PROCESS, self.identifier, identifier, command)
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self.factory._call(Operation.PROCESS_CLOSE, identifier)))
            except BaseException as cleanup_error:
                error.add_note(f"Machine service late-process cleanup failed: {cleanup_error}")
            raise
        return RemoteProcess(self.factory, identifier)

    async def open_process_scope(self) -> "RemoteProcessScope":
        identifier = uuid4().hex
        try:
            await self.factory._call(Operation.OPEN_SCOPE, self.identifier, identifier)
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self.factory._call(Operation.SCOPE_TERMINATE, identifier)))
            except BaseException as cleanup_error:
                error.add_note(f"Machine service late-scope cleanup failed: {cleanup_error}")
            raise
        return RemoteProcessScope(self.factory, identifier)


@dataclass
class RemoteProcess:
    factory: RemoteMachineFactory
    identifier: str
    stderr: bytes = b""

    async def read(self, max_bytes: int = TRANSFER_CHUNK_BYTES) -> bytes:
        result = cast(ProcessRead, await self.factory._call(Operation.PROCESS_READ, self.identifier, max_bytes))
        self.stderr = result.stderr
        return result.data

    async def write(self, data: bytes) -> None:
        for offset in range(0, len(data), TRANSFER_CHUNK_BYTES):
            self.stderr = cast(
                bytes,
                await self.factory._call(
                    Operation.PROCESS_WRITE, self.identifier, data[offset : offset + TRANSFER_CHUNK_BYTES]
                ),
            )

    async def wait(self) -> int:
        result = cast(ProcessExit, await self.factory._call(Operation.PROCESS_WAIT, self.identifier))
        self.stderr = result.stderr
        return result.exit_code

    async def close(self) -> None:
        await self.factory._call(Operation.PROCESS_CLOSE, self.identifier)


@dataclass
class RemoteProcessScope:
    factory: RemoteMachineFactory
    identifier: str

    async def run(self, command: Command) -> Result:
        return cast(Result, await self.factory._call(Operation.SCOPE_RUN, self.identifier, command))

    async def terminate(self) -> None:
        await self.factory._call(Operation.SCOPE_TERMINATE, self.identifier)
