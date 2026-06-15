# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Poll the Iris endpoint registry until a name resolves.

Shared by the actor-coordination call sites where one task registers an
endpoint and peers wait for it to appear — JAX coordinator discovery
(``jax_init``) and sweep gang rounds (``marin.execution.sweep_coordination``).
"""

from __future__ import annotations

import logging
import time

from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.actor.resolver import Resolver

logger = logging.getLogger(__name__)


def poll_endpoint(
    resolver: Resolver,
    name: str,
    *,
    poll_interval: float = 2.0,
    poll_timeout: float | None = None,
    waiting_log: str | None = None,
    log_interval: float = 60.0,
) -> str:
    """Poll ``resolver`` for ``name`` until it resolves; return its first url.

    Args:
        resolver: Namespaced resolver for the current job.
        name: Endpoint name to wait for.
        poll_interval: Initial backoff delay in seconds.
        poll_timeout: Maximum seconds to wait, or ``None`` to wait without a
            deadline.
        waiting_log: If given, log this message (throttled to ``log_interval``)
            while still waiting, so a long wait is visible rather than silent.
        log_interval: Minimum seconds between ``waiting_log`` messages.

    Returns:
        The resolved endpoint's url.

    Raises:
        TimeoutError: If ``poll_timeout`` elapses before the endpoint appears.
    """
    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    deadline = Deadline.from_now(Duration.from_seconds(poll_timeout)) if poll_timeout is not None else None
    last_log = 0.0
    while True:
        resolved = resolver.resolve(name)
        if not resolved.is_empty:
            return resolved.first().url
        if deadline is not None and deadline.expired():
            raise TimeoutError(f"Timed out after {poll_timeout}s waiting for endpoint '{name}'")
        if waiting_log is not None:
            now = time.monotonic()
            if now - last_log >= log_interval:
                logger.info("%s", waiting_log)
                last_log = now
        interval = backoff.next_interval()
        if deadline is not None:
            interval = min(interval, deadline.remaining_seconds())
        if interval > 0:
            time.sleep(interval)
