# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current-client resolution for fray.

Holds the ContextVar and the two helpers that select which backend to use at
runtime.  Kept separate from fray.client so that fray.client (protocols and
exceptions) does not transitively import fray.iris_backend or
fray.local_backend, which would form a circular dependency.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Generator

from iris.client.client import get_iris_ctx

from fray.client import Client
from fray.iris_backend import FrayIrisClient
from fray.local_backend import LocalClient

logger = logging.getLogger(__name__)

_current_client_var: contextvars.ContextVar[Client | None] = contextvars.ContextVar("_current_client_var", default=None)


def current_client() -> Client:
    """Return the current fray Client.

    Resolution order:
        1. Explicitly set client (via set_current_client)
        2. Auto-detect Iris environment (get_iris_ctx() returns context)
        3. LocalClient() default
    """
    client = _current_client_var.get()
    if client is not None:
        logger.info("current_client: using explicitly set client")
        return client

    ctx = get_iris_ctx()
    if ctx is not None:
        logger.info("current_client: using Iris backend (auto-detected)")
        return FrayIrisClient.from_iris_client(ctx.client)

    logger.info("current_client: using LocalClient (fallback)")
    return LocalClient()


@contextlib.contextmanager
def set_current_client(client: Client) -> Generator[Client, None, None]:
    """Context manager that sets the current client and restores on exit."""
    token = _current_client_var.set(client)
    try:
        yield client
    finally:
        _current_client_var.reset(token)
