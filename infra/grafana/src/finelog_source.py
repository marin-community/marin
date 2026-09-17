# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a finelog VM's internal IP and query it over Direct VPC egress.

finelog's cidr auth layer admits the RFC1918 ranges, so the query carries no
token. Discovery is a GCE instance lookup, so it holds while the monitored
cluster's controller is down.
"""

import logging
import threading
import time
from typing import Protocol

import httpx
import pyarrow as pa
from config import FINELOG_PORT, ClusterTarget
from discovery import InstanceResolutionError, resolve_internal_ip
from errors import FinelogUnavailableError
from finelog.client.log_client import LogClient
from finelog.errors import StatsError
from finelog.types import is_retryable_error
from finelog_health import FinelogHealth, FinelogRole
from google.api_core.exceptions import GoogleAPIError
from google.api_core.retry import if_transient_error
from relay_health import RelaySenderStatus, relay_sender_statuses

logger = logging.getLogger(__name__)

_LIST_RELAY_STATUS_PATH = "/finelog.stats.StatsService/ListRelayStatus"
_CONNECT_HEADERS = (("Connect-Protocol-Version", "1"), ("Content-Type", "application/json"))


class MetricSource(Protocol):
    """One cluster's queryable metric store."""

    @property
    def target(self) -> ClusterTarget: ...

    def query(self, sql: str, *, max_rows: int) -> pa.Table: ...

    def namespaces(self) -> frozenset[str]: ...

    def health(self) -> FinelogHealth: ...

    def relay_status(self) -> tuple[RelaySenderStatus, ...]: ...


class FinelogSource:
    """A query handle for one cluster's finelog, addressed by the VM's internal IP.

    The address is resolved lazily and refreshed after a connection failure, so a
    rebuilt VM's new IP is picked up without a restart.
    """

    def __init__(self, target: ClusterTarget, *, timeout_ms: int) -> None:
        self._target = target
        self._client = LogClient.connect(
            f"finelog-{target.name}",  # logical label; the resolver supplies the address
            resolver=self._resolve_address,
            timeout_ms=timeout_ms,
        )
        self._relay_http = httpx.Client(timeout=timeout_ms / 1000)
        self._relay_address: str | None = None
        self._relay_address_lock = threading.Lock()

    @property
    def target(self) -> ClusterTarget:
        return self._target

    def _resolve_address(self, _label: str) -> str:
        """Return http://<internal-ip>:<port> for the VM matching this cluster's filter."""
        ip = resolve_internal_ip(self._target.project, self._target.zone, self._target.instance_filter)
        logger.info("resolved finelog for %s to %s", self._target.name, ip)
        return f"http://{ip}:{FINELOG_PORT}"

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        """Run SQL, classifying discovery and transport failures as retryable."""
        try:
            return self._client.query(sql, max_rows=max_rows)
        except StatsError as err:
            cause = err.__cause__
            if isinstance(cause, Exception) and is_retryable_error(cause):
                raise FinelogUnavailableError(str(err)) from err
            raise
        except GoogleAPIError as err:
            if if_transient_error(err):
                raise FinelogUnavailableError(str(err)) from err
            raise
        except (InstanceResolutionError, OSError) as err:
            raise FinelogUnavailableError(str(err)) from err

    def namespaces(self) -> frozenset[str]:
        """Return the namespaces this deployment holds."""
        return frozenset(info.namespace for info in self._client.list_namespaces())

    def health(self) -> FinelogHealth:
        """Probe the query path and return a dashboard-safe health row."""
        started = time.monotonic()
        try:
            self.query('SELECT * FROM "log" LIMIT 1', max_rows=1)
        except StatsError as err:
            reported_error = err.__cause__ if isinstance(err, FinelogUnavailableError) else err
            assert isinstance(reported_error, Exception)
            logger.warning("finelog health query failed for %s: %s", self._target.name, reported_error)
            return FinelogHealth(
                cluster=self._target.name,
                server=f"finelog-{self._target.name}",
                role=FinelogRole.HUB,
                responsive=False,
                ready=0,
                desired=1,
                latency_ms=None,
                error_class=type(reported_error).__name__,
                error=str(reported_error),
            )
        return FinelogHealth(
            cluster=self._target.name,
            server=f"finelog-{self._target.name}",
            role=FinelogRole.HUB,
            responsive=True,
            ready=1,
            desired=1,
            latency_ms=round((time.monotonic() - started) * 1000),
            error_class="",
            error="",
        )

    def relay_status(self) -> tuple[RelaySenderStatus, ...]:
        # Grafana is deployed from an independent lockfile, so this wire call cannot
        # depend on a LogClient method released from the same repository revision.
        address = self._relay_server_address()
        try:
            response = self._relay_http.post(f"{address}{_LIST_RELAY_STATUS_PATH}", headers=_CONNECT_HEADERS, json={})
            response.raise_for_status()
        except httpx.TransportError:
            with self._relay_address_lock:
                self._relay_address = None
            raise
        return relay_sender_statuses(response.json())

    def _relay_server_address(self) -> str:
        with self._relay_address_lock:
            if self._relay_address is None:
                self._relay_address = self._resolve_address(self._target.name)
            return self._relay_address
