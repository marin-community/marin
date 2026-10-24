# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only backend capabilities supplied at an executable entry point."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

DAYTONA_API_URL = "https://app.daytona.io/api"


@dataclass(frozen=True)
class DaytonaConnection:
    api_key: str | None = field(repr=False)
    api_url: str
    target: str | None


_DAYTONA_CONNECTION: ContextVar[DaytonaConnection | None] = ContextVar(
    "harbor_daytona_connection", default=None
)


@contextmanager
def host_backend_credentials(
    *,
    daytona_api_key: str | None,
    daytona_api_url: str = DAYTONA_API_URL,
    daytona_target: str | None = None,
) -> Iterator[None]:
    """Bind a resolved connection without placing credentials in a job config."""
    token = _DAYTONA_CONNECTION.set(
        DaytonaConnection(daytona_api_key, daytona_api_url, daytona_target)
    )
    try:
        yield
    finally:
        _DAYTONA_CONNECTION.reset(token)


def daytona_connection() -> DaytonaConnection:
    connection = _DAYTONA_CONNECTION.get()
    if (
        connection is None
        or not connection.api_key
        or not connection.api_url
        or not connection.target
    ):
        raise ValueError(
            "Daytona requires explicit host DAYTONA_API_KEY, DAYTONA_API_URL and DAYTONA_TARGET"
        )
    return connection
