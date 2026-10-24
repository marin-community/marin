# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose a bounded host HTTP capability inside a machine without network egress."""

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from shellbox import http_service_guest
from shellbox.machine import Command, Machine

logger = logging.getLogger(__name__)
MAX_BODY = http_service_guest.MAX_BODY
MAX_PENDING = http_service_guest.MAX_PENDING
MAX_PATH = http_service_guest.MAX_PATH
WIRE_LIMIT = MAX_BODY * 2
POLL_INTERVAL = 0.1
TRANSPORT_TIMEOUT = 30
CONTROL_SCRIPT = """
import http.client, sys
c = http.client.HTTPConnection('127.0.0.1', int(sys.argv[1]), timeout=float(sys.argv[4]))
try:
    c.request('POST', '/__shellbox_http__/' + sys.argv[2], body=sys.stdin.buffer.read())
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError('HTTP service control failed')
    sys.stdout.buffer.write(r.read(int(sys.argv[3]) + 1))
finally:
    c.close()
"""


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


async def _control(machine: Machine, port: int, operation: str, body: bytes = b"") -> object:
    result = await machine.run(
        Command(
            (
                "python3",
                "-c",
                CONTROL_SCRIPT,
                str(port),
                operation,
                str(WIRE_LIMIT),
                str(http_service_guest.IDLE_POLL_TIMEOUT + 5),
            ),
            stdin=body,
            output_limit_bytes=WIRE_LIMIT + 1,
            timeout=TRANSPORT_TIMEOUT,
        )
    )
    if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > WIRE_LIMIT:
        raise RuntimeError("Sandbox HTTP service transport failed")
    return json.loads(result.stdout)


async def _response(handler: Callable[[HttpRequest], Awaitable[HttpResponse]], request: HttpRequest) -> HttpResponse:
    try:
        response = await asyncio.wait_for(handler(request), timeout=http_service_guest.REQUEST_TIMEOUT)
        if not 100 <= response.status <= 599 or len(response.body) > MAX_BODY:
            raise ValueError("HTTP capability response exceeds its limits")
        if "\r" in response.content_type or "\n" in response.content_type:
            raise ValueError("Invalid HTTP capability content type")
        return response
    except Exception as error:
        # Provider exception text and headers can contain host credentials.
        logger.warning("HTTP capability failed (%s)", type(error).__name__)
        return HttpResponse(502, b'{"error":"host service failed"}')


async def _pump(
    machine: Machine,
    port: int,
    handler: Callable[[HttpRequest], Awaitable[HttpResponse]],
    stop: asyncio.Event,
) -> None:
    pending: dict[str, asyncio.Task[HttpResponse]] = {}
    try:
        while not stop.is_set():
            for request_id, task in list(pending.items()):
                if not task.done():
                    continue
                response = task.result()
                await _control(
                    machine,
                    port,
                    "respond",
                    json.dumps(
                        {
                            "id": request_id,
                            "status": response.status,
                            "body": base64.b64encode(response.body).decode(),
                            "content_type": response.content_type,
                        }
                    ).encode(),
                )
                del pending[request_id]
            if len(pending) < MAX_PENDING:
                # Long idle polls reduce controller exec traffic. Keep polls
                # short while host work is pending so completed responses drain.
                wait = 1 if pending else http_service_guest.IDLE_POLL_TIMEOUT
                item = await _control(machine, port, "poll", json.dumps({"wait": wait}).encode())
                if item is not None:
                    if not isinstance(item, dict):
                        raise ValueError("Invalid sandbox HTTP request")
                    request_id, method, path, encoded = (item[key] for key in ("id", "method", "path", "body"))
                    if (
                        not isinstance(request_id, str)
                        or len(request_id) != 32
                        or request_id in pending
                        or method not in ("GET", "POST")
                        or not isinstance(path, str)
                        or len(path) > MAX_PATH
                        or not isinstance(encoded, str)
                    ):
                        raise ValueError("Invalid sandbox HTTP request")
                    body = base64.b64decode(encoded, validate=True)
                    if len(body) > MAX_BODY:
                        raise ValueError("Sandbox HTTP request exceeds its limit")
                    pending[request_id] = asyncio.create_task(_response(handler, HttpRequest(method, path, body)))
                    continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL)
            except TimeoutError:
                pass
    finally:
        for task in pending.values():
            task.cancel()
        await asyncio.gather(*pending.values(), return_exceptions=True)


@asynccontextmanager
async def serve_http(
    machine: Machine,
    handler: Callable[[HttpRequest], Awaitable[HttpResponse]],
    *,
    port: int,
) -> AsyncIterator[str]:
    """Run an explicit host handler behind a machine-local HTTP listener.

    The guest needs Python 3. Requests cross the existing machine exec channel;
    the guest receives no host address, authentication headers or network access.
    The handler must authorize paths and operations before using host authority.
    Responses are buffered and bounded; streaming and caller headers are unsupported.
    Close this context before closing the machine or its service group.
    """
    if not 1 <= port <= 65535:
        raise ValueError("HTTP service port is out of range")
    source = Path(http_service_guest.__file__)
    target = f"/tmp/shellbox-http-{uuid.uuid4().hex}.py"
    await machine.upload(source, target)
    stop = asyncio.Event()
    pump = None
    started = False
    try:
        result = await machine.run(Command(("python3", target, "start", str(port)), timeout=TRANSPORT_TIMEOUT))
        if result.exit_code != 0:
            raise RuntimeError("Sandbox HTTP listener failed to start")
        started = True
        pump = asyncio.create_task(_pump(machine, port, handler, stop))
        yield f"http://127.0.0.1:{port}"
    finally:
        stop.set()
        try:
            if pump is not None:
                await pump
        finally:
            try:
                if started:
                    await _control(machine, port, "stop")
            finally:
                await machine.run(Command(("rm", "-f", target), timeout=TRANSPORT_TIMEOUT))
