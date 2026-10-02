# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP handlers for dashboard-authored and built-in tools."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Protocol

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from marin.inference.python_tools import (
    PythonToolDefinitionsRequest,
    PythonToolOperation,
    PythonToolRequest,
    PythonToolSourceTooLarge,
)
from marin.inference.shell_workspace import ShellWorkspaceRequest

TOOL_WORKER_TIMEOUT = 10
_MAX_TOOL_ERROR_LENGTH = 4_000
_PYTHON_TOOL_MODULE = "marin.inference.python_tools"


class _SerializedToolRequest(Protocol):
    def to_json_bytes(self) -> bytes: ...


def _python_tool_operation_label(operation: PythonToolOperation) -> str:
    match operation:
        case PythonToolOperation.DEFINITIONS:
            return "Python tool definition"
        case PythonToolOperation.INVOKE:
            return "Python tool"


def _run_tool_worker(module: str, arguments: tuple[str, ...], payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # Keep CPython on its posix_spawn path: forking after Levanter starts JAX threads can deadlock.
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        input=payload,
        capture_output=True,
        timeout=TOOL_WORKER_TIMEOUT,
        check=False,
        close_fds=False,
    )


async def _tool_response(
    request: Request,
    *,
    label: str,
    module: str,
    arguments: tuple[str, ...],
    parse_payload: Callable[[object], _SerializedToolRequest],
) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": f"{label} request must be JSON"}, status_code=400)
    try:
        tool_request = parse_payload(payload)
    except PythonToolSourceTooLarge:
        return JSONResponse({"error": "Python tool source is too large"}, status_code=413)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        result = await asyncio.to_thread(
            _run_tool_worker,
            module,
            arguments,
            tool_request.to_json_bytes(),
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": f"{label} exceeded {TOOL_WORKER_TIMEOUT} seconds"}, status_code=408)
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace")[-_MAX_TOOL_ERROR_LENGTH:].strip()
        return JSONResponse({"error": f"{label} failed", "details": details}, status_code=422)
    return Response(result.stdout, media_type="application/json")


async def python_tool_definitions_response(request: Request) -> Response:
    operation = PythonToolOperation.DEFINITIONS
    return await _tool_response(
        request,
        label=_python_tool_operation_label(operation),
        module=_PYTHON_TOOL_MODULE,
        arguments=(operation.value,),
        parse_payload=PythonToolDefinitionsRequest.from_payload,
    )


async def invoke_tool_response(request: Request) -> Response:
    name = request.path_params["name"]
    operation = PythonToolOperation.INVOKE
    return await _tool_response(
        request,
        label=_python_tool_operation_label(operation),
        module=_PYTHON_TOOL_MODULE,
        arguments=(operation.value,),
        parse_payload=lambda payload: PythonToolRequest.from_payload(payload, name=name),
    )


async def shell_workspace_response(request: Request) -> Response:
    return await _tool_response(
        request,
        label="Shell workspace command",
        module="marin.inference.shell_workspace",
        arguments=(),
        parse_payload=ShellWorkspaceRequest.from_payload,
    )
