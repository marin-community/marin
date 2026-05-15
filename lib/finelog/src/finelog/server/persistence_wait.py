# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Server-side helper that blocks RPC handlers until rows are durable.

Append paths return the highest ``seq`` they allocated; the bg flush thread
advances ``DiskLogNamespace.max_persisted_seq`` after each successful L0
parquet write. Handlers poll that cursor on the event loop so the wait
parks a coroutine rather than a thread.
"""

from __future__ import annotations

import asyncio
import time

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from finelog.store import LogStore

# Polling cadence inside the wait. Short enough that the wake-up latency
# adds <50ms on top of the bg loop's flush cadence; long enough that an
# idle waiter doesn't burn measurable CPU.
PERSIST_POLL_INTERVAL_SEC = 0.05

# Upper bound on how long a handler will block waiting for an L0 write.
# Sized well above ``DEFAULT_FLUSH_INTERVAL_SEC`` so a single slow flush
# (GCS hiccup, oversized batch) doesn't trip clients.
DEFAULT_PERSIST_TIMEOUT_SEC = 30.0


async def await_persisted(
    log_store: LogStore,
    namespace: str,
    target_seq: int,
    *,
    timeout: float = DEFAULT_PERSIST_TIMEOUT_SEC,
    poll_interval: float = PERSIST_POLL_INTERVAL_SEC,
) -> None:
    """Block until ``log_store``'s persisted cursor for ``namespace`` reaches ``target_seq``.

    ``target_seq < 0`` is the "nothing to wait for" sentinel returned by
    empty appends and resolves immediately. On timeout, raises
    ``ConnectError(DEADLINE_EXCEEDED)``; the buffered rows remain in RAM
    and the next successful flush will pick them up.
    """
    if target_seq < 0:
        return
    deadline = time.monotonic() + timeout
    while log_store.max_persisted_seq(namespace) < target_seq:
        if time.monotonic() >= deadline:
            raise ConnectError(
                Code.DEADLINE_EXCEEDED,
                f"timed out after {timeout:.1f}s waiting for namespace {namespace!r} to persist seq>={target_seq}",
            )
        await asyncio.sleep(poll_interval)
