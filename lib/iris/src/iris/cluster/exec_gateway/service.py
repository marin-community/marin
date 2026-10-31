# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous execution sessions outside the controller RPC pool."""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.timing import Timer, Timestamp
from rigging.token_authority import JwksVerifier

from iris.cluster.exec_gateway.auth import ExecGrant, ExecPodTarget, verify_exec_grant
from iris.rpc import exec_gateway_pb2

logger = logging.getLogger(__name__)

MAX_CHUNK_BYTES = 64 * 1024
MAX_BUFFER_BYTES = 1024 * 1024
MAX_SESSIONS = 256
MAX_SPENT_GRANTS = 8192
MAX_WAIT_MS = 20_000
IDLE_TIMEOUT_MS = 300_000
IO_TIMEOUT = 30


@dataclass(frozen=True)
class ExecOutput:
    stdout: bytes = b""
    stderr: bytes = b""
    exit_code: int | None = None


class ExecTransport(Protocol):
    async def send(self, data: bytes, *, eof: bool) -> None: ...

    async def receive(self) -> ExecOutput: ...

    async def close(self) -> None: ...


class ExecConnector(Protocol):
    async def open(self, target: ExecPodTarget, command: tuple[str, ...]) -> ExecTransport: ...

    async def close(self) -> None: ...


@dataclass
class _Session:
    grant: ExecGrant
    transport: ExecTransport
    touched_at: int
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    sending: asyncio.Lock = field(default_factory=asyncio.Lock)
    receiving: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    exit_code: int | None = None
    error: str = ""
    stdin_closed: bool = False
    stdin_bytes: int = 0
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    backpressure_events: int = 0
    dropped_bytes: int = 0
    receive_calls: int = 0
    receive_wait_ms: int = 0


class ExecGatewayService:
    """One event loop multiplexes all admitted execution streams.

    The caller drains each session with Receive. Slow consumers backpressure the
    transport at a fixed byte budget; they cannot grow gateway memory.
    """

    def __init__(
        self,
        verifier: JwksVerifier,
        connector: ExecConnector,
        *,
        max_sessions: int = MAX_SESSIONS,
        max_buffer_bytes: int = MAX_BUFFER_BYTES,
        idle_timeout_ms: int = IDLE_TIMEOUT_MS,
    ):
        self._verifier = verifier
        self._connector = connector
        self._max_sessions = max_sessions
        self._max_buffer_bytes = max_buffer_bytes
        self._idle_timeout_ms = idle_timeout_ms
        self._sessions: dict[str, _Session] = {}
        self._closing: set[asyncio.Task[None]] = set()
        # Retain spent grants through expiry, including closed sessions, so Open
        # retries and replay cannot launch the command a second time.
        self._spent: dict[str, int] = {}
        self._opening = 0
        self._reaper: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._reaper = asyncio.create_task(self._reap())

    def _grant(self, token: str) -> ExecGrant:
        try:
            return verify_exec_grant(self._verifier, token)
        except ValueError as error:
            raise ConnectError(Code.UNAUTHENTICATED, "Invalid or expired execution grant") from error

    def _session(self, token: str, session_id: str) -> _Session:
        grant = self._grant(token)
        session = self._sessions.get(session_id)
        if session is None:
            raise ConnectError(Code.NOT_FOUND, "Execution session is closed or unknown")
        if session.grant != grant:
            raise ConnectError(Code.PERMISSION_DENIED, "Execution grant does not own this session")
        session.touched_at = Timestamp.now().epoch_ms()
        return session

    async def open(self, request: exec_gateway_pb2.OpenRequest, ctx: RequestContext) -> exec_gateway_pb2.OpenResponse:
        del ctx
        grant = self._grant(request.grant)
        if self._shutdown is not None:
            raise ConnectError(Code.UNAVAILABLE, "Execution gateway is shutting down")
        if grant.grant_id in self._spent:
            raise ConnectError(Code.ALREADY_EXISTS, "Execution grant has already been consumed")
        if (
            len(self._sessions) + self._opening + len(self._closing) >= self._max_sessions
            or len(self._spent) >= MAX_SPENT_GRANTS
        ):
            logger.warning(
                "exec gateway capacity refused active=%d opening=%d spent=%d",
                len(self._sessions),
                self._opening,
                len(self._spent),
            )
            raise ConnectError(Code.RESOURCE_EXHAUSTED, "Execution gateway session capacity reached")
        self._spent[grant.grant_id] = grant.expires_at_ms
        self._opening += 1
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                transport = await self._connector.open(grant.target, grant.command)
            session_id = secrets.token_urlsafe(32)
            session = _Session(grant, transport, Timestamp.now().epoch_ms())
            self._sessions[session_id] = session
            session.task = asyncio.create_task(self._read(session))
            logger.info(
                "exec session opened active=%d opening=%d pod=%s container=%s",
                len(self._sessions),
                self._opening - 1,
                grant.target.pod_name,
                grant.target.container_name,
            )
            return exec_gateway_pb2.OpenResponse(session_id=session_id)
        finally:
            self._opening -= 1

    async def send(self, request: exec_gateway_pb2.SendRequest, ctx: RequestContext) -> exec_gateway_pb2.SendResponse:
        del ctx
        if len(request.stdin) > MAX_CHUNK_BYTES:
            raise ConnectError(Code.RESOURCE_EXHAUSTED, "Execution stdin chunk exceeds limit")
        session = self._session(request.grant, request.session_id)
        if session.sending.locked():
            raise ConnectError(Code.ABORTED, "Another stdin write is in progress")
        async with session.sending:
            if session.stdin_closed or session.exit_code is not None:
                raise ConnectError(Code.FAILED_PRECONDITION, "Execution stdin is closed")
            try:
                async with asyncio.timeout(IO_TIMEOUT):
                    await session.transport.send(request.stdin, eof=request.eof)
                session.stdin_closed = request.eof
                session.stdin_bytes += len(request.stdin)
            except (OSError, TimeoutError) as error:
                await self._finish(session, 1, f"Execution stdin transport failed: {error}")
                raise ConnectError(Code.UNAVAILABLE, "Execution stdin transport failed") from error
        return exec_gateway_pb2.SendResponse()

    async def receive(
        self, request: exec_gateway_pb2.ReceiveRequest, ctx: RequestContext
    ) -> exec_gateway_pb2.ReceiveResponse:
        del ctx
        session = self._session(request.grant, request.session_id)
        if session.receiving.locked():
            raise ConnectError(Code.ABORTED, "Another output receive is in progress")
        maximum = min(request.max_bytes or MAX_CHUNK_BYTES, MAX_CHUNK_BYTES)
        timer = Timer()
        async with session.receiving:
            if not session.stdout and not session.stderr and session.exit_code is None:
                session.changed.clear()
                try:
                    async with asyncio.timeout(min(request.wait_ms, MAX_WAIT_MS) / 1000):
                        await session.changed.wait()
                except TimeoutError:
                    pass
            stdout = bytes(session.stdout[:maximum])
            del session.stdout[: len(stdout)]
            stderr = bytes(session.stderr[: maximum - len(stdout)])
            del session.stderr[: len(stderr)]
            drained = not session.stdout and not session.stderr
            session.drained.set()
            session.receive_calls += 1
            session.receive_wait_ms += timer.elapsed_ms()
            return exec_gateway_pb2.ReceiveResponse(
                stdout=stdout,
                stderr=stderr,
                exited=session.exit_code is not None and drained,
                exit_code=session.exit_code or 0,
                error=session.error,
            )

    async def close_session(
        self, request: exec_gateway_pb2.CloseRequest, ctx: RequestContext
    ) -> exec_gateway_pb2.CloseResponse:
        del ctx
        session = self._session(request.grant, request.session_id)
        disposal = self._start_disposal(session)
        del self._sessions[request.session_id]
        await asyncio.shield(disposal)
        return exec_gateway_pb2.CloseResponse()

    async def _read(self, session: _Session) -> None:
        try:
            while session.exit_code is None:
                output = await session.transport.receive()
                incoming = len(output.stdout) + len(output.stderr)
                if incoming > self._max_buffer_bytes:
                    session.dropped_bytes += incoming
                    await self._finish(session, 1, "Execution output frame exceeds buffer budget")
                    return
                if len(session.stdout) + len(session.stderr) + incoming > self._max_buffer_bytes:
                    session.backpressure_events += 1
                while len(session.stdout) + len(session.stderr) + incoming > self._max_buffer_bytes:
                    session.drained.clear()
                    await session.drained.wait()
                session.stdout.extend(output.stdout)
                session.stderr.extend(output.stderr)
                session.stdout_bytes += len(output.stdout)
                session.stderr_bytes += len(output.stderr)
                if output.exit_code is not None:
                    await self._finish(session, output.exit_code)
                session.changed.set()
        except (OSError, EOFError, ValueError) as error:
            await self._finish(session, 1, f"Execution transport closed: {error}")

    async def _finish(self, session: _Session, exit_code: int, error: str = "") -> None:
        session.exit_code = exit_code
        session.error = error
        session.changed.set()
        await session.transport.close()

    def _start_disposal(self, session: _Session) -> asyncio.Task[None]:
        task = asyncio.create_task(self._dispose(session))
        self._closing.add(task)
        task.add_done_callback(self._disposal_finished)
        return task

    def _disposal_finished(self, task: asyncio.Task[None]) -> None:
        self._closing.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Execution session disposal failed (%s)", type(error).__name__)

    async def _dispose(self, session: _Session) -> None:
        if session.exit_code is None:
            session.exit_code = 137
        session.changed.set()
        if session.task is not None:
            session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)
        await session.transport.close()
        logger.info(
            "exec session closed active=%d stdin_bytes=%d stdout_bytes=%d stderr_bytes=%d "
            "queued_bytes=%d backpressure_events=%d dropped_bytes=%d receive_calls=%d "
            "receive_wait_ms=%d exit_code=%s error=%s",
            len(self._sessions),
            session.stdin_bytes,
            session.stdout_bytes,
            session.stderr_bytes,
            len(session.stdout) + len(session.stderr),
            session.backpressure_events,
            session.dropped_bytes,
            session.receive_calls,
            session.receive_wait_ms,
            session.exit_code,
            session.error.split(":", 1)[0],
        )

    async def _reap(self) -> None:
        while True:
            await asyncio.sleep(1)
            now = Timestamp.now().epoch_ms()
            expired = [
                key
                for key, session in self._sessions.items()
                if now >= session.grant.expires_at_ms or now - session.touched_at >= self._idle_timeout_ms
            ]
            for key in expired:
                session = self._sessions.pop(key)
                session.error = (
                    "Execution grant expired" if now >= session.grant.expires_at_ms else "Execution session idle"
                )
                await asyncio.shield(self._start_disposal(session))
            self._spent = {key: expiry for key, expiry in self._spent.items() if expiry > now}

    async def shutdown(self) -> None:
        if self._shutdown is None:
            self._shutdown = asyncio.create_task(self._shutdown_all())
        cancellation: asyncio.CancelledError | None = None
        while not self._shutdown.done():
            try:
                await asyncio.shield(self._shutdown)
            except asyncio.CancelledError as error:
                cancellation = error
        self._shutdown.result()
        if cancellation is not None:
            raise cancellation

    async def _shutdown_all(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
        for session in self._sessions.values():
            self._start_disposal(session)
        self._sessions.clear()
        try:
            results = await asyncio.gather(*self._closing, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            await self._connector.close()
