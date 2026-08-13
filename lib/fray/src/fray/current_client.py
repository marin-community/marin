# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the active Fray client."""

import contextlib
import contextvars
import logging
import os
from collections.abc import Generator

from iris.client.context_state import has_current_context

from fray.client import Client
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

    ctx = None
    if has_current_context() or os.environ.get("IRIS_TASK_ID"):
        # Iris is an optional Fray backend. Import its client only inside an Iris
        # context, where it is needed to resolve the concrete client.
        from iris.client.client import get_iris_ctx  # noqa: PLC0415

        ctx = get_iris_ctx()
    if ctx is not None:
        if ctx.client is None:
            raise RuntimeError("Iris context has no client")
        from fray.iris_backend import FrayIrisClient  # noqa: PLC0415

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
