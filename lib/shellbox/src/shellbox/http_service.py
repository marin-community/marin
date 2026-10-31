# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose a bounded host HTTP capability inside a machine without network egress."""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from shellbox import http_service_guest
from shellbox.machine import Command, Machine, ProcessMachine, ProcessSession, UnsupportedMachineSpec
from shellbox.process_session import finish_owned, own_cleanup

logger = logging.getLogger(__name__)
MAX_BODY = http_service_guest.MAX_BODY
MAX_PENDING = http_service_guest.MAX_PENDING
MAX_PATH = http_service_guest.MAX_PATH
WIRE_LIMIT = http_service_guest.WIRE_LIMIT
TRANSPORT_TIMEOUT = 30
CLEANUP_TIMEOUT = 5


@dataclass(frozen=True)
class HttpRequest:
    method: str
    path: str
    body: bytes


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    content_type: str = "application/json"


class _Frames:
    def __init__(self, session: ProcessSession):
        self.session = session
        self.buffer = bytearray()

    async def receive(self) -> dict:
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                frame = bytes(self.buffer[:end])
                del self.buffer[: end + 1]
                item = json.loads(frame)
                if not isinstance(item, dict):
                    raise ValueError("Invalid sandbox HTTP frame")
                return item
            if len(self.buffer) >= WIRE_LIMIT:
                raise ValueError("Sandbox HTTP frame exceeds its limit")
            data = await self.session.read(min(65536, WIRE_LIMIT - len(self.buffer)))
            if not data:
                raise RuntimeError("Sandbox HTTP service process ended")
            self.buffer.extend(data)


async def _response(handler: Callable[[HttpRequest], Awaitable[HttpResponse]], request: HttpRequest) -> HttpResponse:
    try:
        response = await asyncio.wait_for(handler(request), timeout=http_service_guest.REQUEST_TIMEOUT)
        if not 100 <= response.status <= 599 or len(response.body) > MAX_BODY:
            raise ValueError("HTTP capability response exceeds its limits")
        if len(response.content_type) > MAX_PATH or "\r" in response.content_type or "\n" in response.content_type:
            raise ValueError("Invalid HTTP capability content type")
        return response
    except Exception as error:
        # Provider exception text and headers can contain host credentials.
        logger.warning("HTTP capability failed (%s)", type(error).__name__)
        return HttpResponse(502, b'{"error":"host service failed"}')


async def _cancel(tasks: set[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=CLEANUP_TIMEOUT)
    for task in done:
        if not task.cancelled():
            task.exception()
    # A handler that suppresses cancellation must not hold the service open.
    # Retain its task until it exits, consuming any eventual error.
    for task in pending:
        task.add_done_callback(_handler_finished)
        _UNFINISHED_HANDLERS.add(task)


_UNFINISHED_HANDLERS: set[asyncio.Task] = set()


def _handler_finished(task: asyncio.Task) -> None:
    _UNFINISHED_HANDLERS.discard(task)
    if not task.cancelled():
        task.exception()


async def _pump(frames: _Frames, handler: Callable[[HttpRequest], Awaitable[HttpResponse]]) -> None:
    pending: dict[str, asyncio.Task[HttpResponse]] = {}
    receive = asyncio.create_task(frames.receive())
    try:
        while True:
            done, _ = await asyncio.wait((receive, *pending.values()), return_when=asyncio.FIRST_COMPLETED)
            for request_id, task in list(pending.items()):
                if task not in done:
                    continue
                response = task.result()
                frame = (
                    json.dumps(
                        {
                            "id": request_id,
                            "status": response.status,
                            "body": base64.b64encode(response.body).decode(),
                            "content_type": response.content_type,
                        },
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                async with asyncio.timeout(TRANSPORT_TIMEOUT):
                    await frames.session.write(frame)
                del pending[request_id]
            if receive in done:
                item = receive.result()
                request_id, method, path, encoded = (item[key] for key in ("id", "method", "path", "body"))
                if (
                    item.get("type") != "request"
                    or not isinstance(request_id, str)
                    or len(request_id) != 32
                    or request_id in pending
                    or method not in ("GET", "POST")
                    or not isinstance(path, str)
                    or len(path) > MAX_PATH
                    or not isinstance(encoded, str)
                    or len(pending) >= MAX_PENDING
                ):
                    raise ValueError("Invalid sandbox HTTP request")
                body = base64.b64decode(encoded, validate=True)
                if len(body) > MAX_BODY:
                    raise ValueError("Sandbox HTTP request exceeds its limit")
                pending[request_id] = asyncio.create_task(_response(handler, HttpRequest(method, path, body)))
                receive = asyncio.create_task(frames.receive())
    finally:
        await finish_owned(own_cleanup(_cancel({receive, *pending.values()})))


@asynccontextmanager
async def serve_http(
    machine: Machine,
    handler: Callable[[HttpRequest], Awaitable[HttpResponse]],
    *,
    port: int,
) -> AsyncIterator[str]:
    """Run a host handler behind one foreground machine-local HTTP listener.

    Python 3 in the guest exchanges bounded, buffered frames over persistent
    process pipes. The handler authorizes paths and operations before using host
    authority. Caller headers and streaming are unsupported. Close this context
    before closing the machine or its container group.
    """
    if not 1 <= port <= 65535:
        raise ValueError("HTTP service port is out of range")
    if not isinstance(machine, ProcessMachine):
        raise UnsupportedMachineSpec("HTTP services require persistent process sessions")
    source = Path(http_service_guest.__file__).read_text()
    session = await machine.open_process(Command(("python3", "-u", "-c", source, str(port))))
    pump: asyncio.Task[None] | None = None
    owner = asyncio.current_task()
    assert owner is not None
    failed = False
    interrupted = False
    primary: BaseException | None = None

    def finished(task: asyncio.Task[None]) -> None:
        nonlocal failed, interrupted
        if not task.cancelled():
            # Interrupt the body promptly if the helper exits while no HTTP
            # request is active. Retrieve the actual failure after cleanup.
            failed = True
            if not owner.cancelling():
                interrupted = True
                owner.cancel()

    try:
        frames = _Frames(session)
        async with asyncio.timeout(TRANSPORT_TIMEOUT):
            if await frames.receive() != {"type": "ready"}:
                raise RuntimeError("Sandbox HTTP listener failed to start")
        pump = asyncio.create_task(_pump(frames, handler))
        pump.add_done_callback(finished)
        yield f"http://127.0.0.1:{port}"
    except BaseException as error:
        primary = error
        raise
    finally:
        if pump is not None:
            pump.remove_done_callback(finished)

        async def cleanup() -> None:
            if pump is not None:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
            async with asyncio.timeout(TRANSPORT_TIMEOUT):
                await session.close()

        closing = own_cleanup(cleanup())
        try:
            await asyncio.shield(closing)
        except BaseException as error:
            try:
                await finish_owned(closing)
            except BaseException as cleanup_error:
                error.add_note(f"HTTP service cleanup failed ({type(cleanup_error).__name__})")
            if primary is None:
                raise
        if (
            failed
            and pump is not None
            and (primary is None or (interrupted and isinstance(primary, asyncio.CancelledError)))
        ):
            if interrupted:
                owner.uncancel()
            pump.result()
