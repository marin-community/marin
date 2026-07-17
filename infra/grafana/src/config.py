# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge configuration: the clusters served and the runtime settings.

One bridge serves every cluster in CLUSTERS. Grafana provisions one datasource
per entry and addresses it by name in the URL path.
"""

import dataclasses
import os

# Port finelog listens on, set in lib/finelog/config/{marin,marin-dev}.yaml.
FINELOG_PORT = 10001

# Loopback port the bridge listens on; the provisioned datasource URLs
# (provisioning/datasources/finelog.yaml) use it.
BRIDGE_PORT = 8081


@dataclasses.dataclass(frozen=True)
class ClusterTarget:
    """A finelog deployment the bridge can query.

    instance_filter is a GCE list filter selecting the VM to connect to.
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

    # Maximum rows a query may return before the bridge rejects it with a 400.
    max_rows: int
    # Result cache TTL, seconds.
    cache_ttl: float
    query_timeout_ms: int

    @staticmethod
    def from_environment() -> "BridgeConfig":
        """Read settings from the container environment."""
        return BridgeConfig(
            max_rows=int(os.environ.get("GRAFANA_BRIDGE_MAX_ROWS", "200000")),
            cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_CACHE_TTL", "20")),
            query_timeout_ms=int(os.environ.get("GRAFANA_BRIDGE_QUERY_TIMEOUT_MS", "20000")),
        )
