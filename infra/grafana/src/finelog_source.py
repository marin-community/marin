# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a finelog VM's internal IP and query it over Direct VPC egress.

finelog's cidr auth layer admits the RFC1918 ranges, so the query carries no
token. Discovery is a GCE instance lookup, so it holds while the monitored
cluster's controller is down.
"""

import logging
from typing import Protocol

import pyarrow as pa
from config import FINELOG_PORT, ClusterTarget
from discovery import resolve_internal_ip
from finelog.client.log_client import LogClient

logger = logging.getLogger(__name__)


class MetricSource(Protocol):
    """One cluster's queryable metric store."""

    @property
    def target(self) -> ClusterTarget: ...

    def query(self, sql: str, *, max_rows: int) -> pa.Table: ...


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

    @property
    def target(self) -> ClusterTarget:
        return self._target

    def _resolve_address(self, _label: str) -> str:
        """Return http://<internal-ip>:<port> for the VM matching this cluster's filter."""
        ip = resolve_internal_ip(self._target.project, self._target.zone, self._target.instance_filter)
        logger.info("resolved finelog for %s to %s", self._target.name, ip)
        return f"http://{ip}:{FINELOG_PORT}"

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        """Run sql against this cluster's finelog. Raises QueryResultTooLargeError past max_rows."""
        return self._client.query(sql, max_rows=max_rows)
