# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge configuration, resolved once at startup from the container environment.

One bridge serves every cluster in :data:`CLUSTERS`; Grafana provisions one
datasource per entry and addresses it by name in the URL path. Both finelog VMs
live in the same project and zone and are reached over Direct VPC egress on their
internal IPs, so a single Cloud Run service covers prod and dev.

Cluster targets are defined here and changed under review, so a datasource points
at the cluster it names.
"""

import dataclasses
import os

# finelog listens here on both VMs (lib/finelog/config/{marin,marin-dev}.yaml).
FINELOG_PORT = 10001

# The bridge's loopback port, named literally by the provisioned datasource URLs
# (provisioning/datasources/finelog.yaml). A constant so the two cannot drift.
BRIDGE_PORT = 8081


@dataclasses.dataclass(frozen=True)
class ClusterTarget:
    """One finelog deployment the bridge can query.

    ``instance_filter`` is a GCE list filter selecting the VM whose internal IP
    the bridge connects to.
    """

    name: str
    project: str
    zone: str
    instance_filter: str


CLUSTERS: tuple[ClusterTarget, ...] = (
    ClusterTarget(
        name="marin",
        project="hai-gcp-models",
        zone="us-central1-a",
        instance_filter="name = finelog-marin",
    ),
    ClusterTarget(
        name="marin-dev",
        project="hai-gcp-models",
        zone="us-central1-a",
        instance_filter="name = finelog-marin-dev",
    ),
)


@dataclasses.dataclass(frozen=True)
class BridgeConfig:
    """Resolved bridge settings."""

    # Rows one query may return. finelog caps a response at 64 MiB and enforces its
    # own query deadline; this is a lower ceiling so a mis-written panel returns an
    # error rather than a result Grafana cannot render.
    max_rows: int
    # Result cache TTL. Grafana's own query caching is Enterprise-only, and a
    # shared dashboard auto-refreshing across viewers would otherwise multiply
    # straight through to the finelog hub.
    cache_ttl: float
    query_timeout_ms: int

    @staticmethod
    def from_environment() -> "BridgeConfig":
        """Read settings from the container env, falling back to prod-safe defaults."""
        return BridgeConfig(
            max_rows=int(os.environ.get("GRAFANA_BRIDGE_MAX_ROWS", "200000")),
            cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_CACHE_TTL", "20")),
            query_timeout_ms=int(os.environ.get("GRAFANA_BRIDGE_QUERY_TIMEOUT_MS", "20000")),
        )
