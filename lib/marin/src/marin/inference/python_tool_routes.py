# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP handlers for dashboard-authored Python tools."""

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

PYTHON_TOOL_TIMEOUT = 10
_MAX_TOOL_ERROR_LENGTH = 4_000


class _SerializedPythonToolRequest(Protocol):
    def to_json_bytes(self) -> bytes: ...


def _python_tool_operation_label(operation: PythonToolOperation) -> str:
    match operation:
        case PythonToolOperation.DEFINITIONS:
            return "Python tool definition"
        case PythonToolOperation.INVOKE:
            return "Python tool"


def _run_python_tool_worker(operation: PythonToolOperation, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # Keep CPython on its posix_spawn path: forking after Levanter starts JAX threads can deadlock.
    return subprocess.run(
        [sys.executable, "-m", "marin.inference.python_tools", operation.value],
        input=payload,
        capture_output=True,
        timeout=PYTHON_TOOL_TIMEOUT,
        check=False,
        close_fds=False,
    )


async def _python_tool_response(
    request: Request,
    *,
    operation: PythonToolOperation,
    parse_payload: Callable[[object], _SerializedPythonToolRequest],
) -> Response:
    label = _python_tool_operation_label(operation)
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
        result = await asyncio.to_thread(_run_python_tool_worker, operation, tool_request.to_json_bytes())
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": f"{label} exceeded {PYTHON_TOOL_TIMEOUT} seconds"}, status_code=408)
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace")[-_MAX_TOOL_ERROR_LENGTH:].strip()
        return JSONResponse({"error": f"{label} failed", "details": details}, status_code=422)
    return Response(result.stdout, media_type="application/json")


async def python_tool_definitions_response(request: Request) -> Response:
    return await _python_tool_response(
        request,
        operation=PythonToolOperation.DEFINITIONS,
        parse_payload=PythonToolDefinitionsRequest.from_payload,
    )


async def invoke_tool_response(request: Request) -> Response:
    name = request.path_params["name"]
    return await _python_tool_response(
        request,
        operation=PythonToolOperation.INVOKE,
        parse_payload=lambda payload: PythonToolRequest.from_payload(payload, name=name),
    )
