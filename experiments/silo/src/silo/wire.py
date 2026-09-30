# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Error transport between server and client.

The consumer classifies errors by name, ``status_code`` and message text, so an
error has to survive the HTTP hop with all three intact. The server sends
``{"error": <class>, "message": <str>, ...}`` with the matching HTTP status; the
client rebuilds the same class with the same message.
"""

from __future__ import annotations

from typing import Any

from silo.errors import (
    SiloConflictError,
    SiloError,
    SiloNotFoundError,
    SiloRateLimitError,
    SiloRecipeError,
)


def error_body(error: SiloError) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": type(error).__name__,
        "message": str(error),
        "status_code": error.status_code or 500,
    }
    if isinstance(error, (SiloNotFoundError, SiloConflictError)):
        body["resource"] = error.resource
        body["name"] = error.name
    if isinstance(error, SiloRateLimitError):
        body["error_code"] = error.error_code
        body["headers"] = dict(error.headers)
    return body


def error_from_body(body: dict[str, Any], http_status: int) -> SiloError:
    kind = body.get("error", "")
    message = str(body.get("message", f"HTTP {http_status}"))
    status = int(body.get("status_code") or http_status)

    if kind == "SiloNotFoundError":
        error: SiloError = SiloNotFoundError.__new__(SiloNotFoundError)
        SiloError.__init__(error, message)
        error.resource = body.get("resource", "")  # type: ignore[attr-defined]
        error.name = body.get("name", "")  # type: ignore[attr-defined]
        return error
    if kind == "SiloConflictError":
        error = SiloConflictError.__new__(SiloConflictError)
        SiloError.__init__(error, message)
        error.resource = body.get("resource", "")  # type: ignore[attr-defined]
        error.name = body.get("name", "")  # type: ignore[attr-defined]
        return error
    if kind == "SiloRateLimitError":
        return SiloRateLimitError(
            message,
            error_code=str(body.get("error_code", "capacity_exhausted")),
            headers=body.get("headers") or {},
        )
    if kind == "SiloRecipeError":
        return SiloRecipeError(message)
    return SiloError(message, status_code=status)
