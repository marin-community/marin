# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pings the Iris controller via list_workers(). Success iff the RPC returns
in time. REMOTE_ERROR on RPC error, TIMEOUT on deadline exceeded."""

from __future__ import annotations

import logging

from iris.cluster.client.remote_client import RemoteClusterClient

from probes.probe import ErrorClass, ProbeOutcome, ProbeResult

logger = logging.getLogger(__name__)


class ControllerPing:
    """Iris controller liveness probe."""

    def __init__(self, client: RemoteClusterClient):
        self._client = client

    def run(self, deadline_seconds: float) -> ProbeResult:
        try:
            workers = self._client.list_workers()
        except TimeoutError as exc:
            return ProbeResult(
                outcome=ProbeOutcome.TIMEOUT,
                error_class=ErrorClass.TIMEOUT,
                error_detail=str(exc),
            )
        except Exception as exc:
            return ProbeResult.remote_error(_classify(exc), f"{type(exc).__name__}: {exc}")
        return ProbeResult.success(extras={"worker_count": len(workers)})


def _classify(exc: Exception) -> ErrorClass:
    name = type(exc).__name__
    if name == "ConnectError":
        return ErrorClass.RPC_ERROR
    if isinstance(exc, ConnectionError):
        return ErrorClass.CONNECT_ERROR
    return ErrorClass.RPC_ERROR
