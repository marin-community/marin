# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a finelog VM's internal IP and query it.

The internal IP is reachable from Cloud Run over Direct VPC egress, and finelog's
``cidr`` auth layer admits the RFC1918 ranges, so the query carries no token.

Discovery is a GCE instance lookup rather than an Iris endpoint-registry lookup:
Grafana exists to be readable when the cluster is not, so resolving finelog
through the thing being monitored would put it back on the path to the data about
it.
"""

import logging
from typing import Protocol

import pyarrow as pa
from config import FINELOG_PORT, ClusterTarget
from finelog.client.log_client import LogClient
from google.cloud import compute_v1

logger = logging.getLogger(__name__)


class MetricSource(Protocol):
    """One cluster's queryable metric store."""

    @property
    def target(self) -> ClusterTarget: ...

    def query(self, sql: str, *, max_rows: int) -> pa.Table: ...


class FinelogSource:
    """A query handle for one cluster's finelog, addressed by the VM's internal IP.

    The address is looked up lazily and refreshed after a connection failure, so a
    rebuilt VM's new IP is picked up without a restart.
    """

    def __init__(self, target: ClusterTarget, *, timeout_ms: int) -> None:
        self._target = target
        self._client = LogClient.connect(
            f"finelog-{target.name}",  # logical label; the resolver supplies the real address
            resolver=self._resolve_address,
            timeout_ms=timeout_ms,
        )

    @property
    def target(self) -> ClusterTarget:
        return self._target

    def _resolve_address(self, _label: str) -> str:
        """Return ``http://<internal-ip>:<port>`` for the VM matching this cluster's filter."""
        instances = compute_v1.InstancesClient().list(
            project=self._target.project,
            zone=self._target.zone,
            filter=self._target.instance_filter,
        )
        for instance in instances:
            for interface in instance.network_interfaces:
                if interface.network_i_p:
                    logger.info("resolved finelog for %s to %s", self._target.name, interface.network_i_p)
                    return f"http://{interface.network_i_p}:{FINELOG_PORT}"
        raise RuntimeError(
            f"no finelog VM with an internal IP for {self._target.name} "
            f"({self._target.instance_filter} in {self._target.zone})"
        )

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        """Run ``sql`` against this cluster's finelog.

        Raises ``QueryResultTooLargeError`` past ``max_rows``.
        """
        return self._client.query(sql, max_rows=max_rows)
