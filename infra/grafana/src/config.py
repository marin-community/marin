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

# Port the Iris controller's HTTP/RPC surface listens on.
CONTROLLER_PORT = 10000

# Loopback port the bridge listens on; the provisioned datasource URLs
# (provisioning/datasources/finelog.yaml) use it.
BRIDGE_PORT = 8081

# The GitHub repository the ferry and build panels read.
GITHUB_REPO = "marin-community/marin"
# Ferry runs fetched per tier; commits scanned for the build panel.
FERRY_RUN_LIMIT = 14
BUILD_HISTORY = 100


@dataclasses.dataclass(frozen=True)
class ClusterTarget:
    """A cluster the bridge can query.

    instance_filter selects the finelog VM; controller_filter selects the Iris
    controller VM. Both are GCE list filters resolved to an internal IP.
    """

    name: str
    project: str
    zone: str
    instance_filter: str
    controller_filter: str


CLUSTERS: tuple[ClusterTarget, ...] = (
    ClusterTarget(
        name="marin",
        project="hai-gcp-models",
        zone="us-central1-a",
        instance_filter="name = finelog-marin",
        controller_filter="labels.iris-marin-controller=true AND status=RUNNING",
    ),
    ClusterTarget(
        name="marin-dev",
        project="hai-gcp-models",
        zone="us-central1-a",
        instance_filter="name = finelog-marin-dev",
        controller_filter="labels.iris-marin-dev-controller=true AND status=RUNNING",
    ),
)


@dataclasses.dataclass(frozen=True)
class FerryTier:
    """One workflow file backing a ferry card. label captions a multi-tier strip."""

    label: str | None
    file: str


@dataclasses.dataclass(frozen=True)
class FerryGroup:
    """One ferry card, grouping one or more tiers."""

    name: str
    tiers: tuple[FerryTier, ...]


# The ferry cards the dashboard renders, mirroring the canary/CW/datakit workflows.
FERRY_GROUPS: tuple[FerryGroup, ...] = (
    FerryGroup("Canary ferry", (FerryTier(None, "marin-canary-ferry.yaml"),)),
    FerryGroup("CW ferry", (FerryTier(None, "marin-canary-ferry-coreweave.yaml"),)),
    FerryGroup(
        "Datakit ferry",
        (
            FerryTier("tier1", "marin-canary-datakit-tier1.yaml"),
            FerryTier("tier2", "marin-canary-datakit-tier2.yaml"),
            FerryTier("tier3", "marin-canary-datakit-tier3.yaml"),
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class BridgeConfig:
    """Resolved bridge settings."""

    # Maximum rows a query may return before the bridge rejects it with a 400.
    max_rows: int
    # finelog result cache TTL, seconds.
    cache_ttl: float
    query_timeout_ms: int
    # Cache TTLs for the live Iris and GitHub endpoints, seconds.
    iris_cache_ttl: float
    github_cache_ttl: float
    # HTTP timeout for the controller RPC and GitHub calls, seconds.
    http_timeout: float
    # GitHub token; lifts the REST rate limit and is required for the GraphQL build panel.
    github_token: str | None

    @staticmethod
    def from_environment() -> "BridgeConfig":
        """Read settings from the container environment."""
        return BridgeConfig(
            max_rows=int(os.environ.get("GRAFANA_BRIDGE_MAX_ROWS", "200000")),
            cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_CACHE_TTL", "20")),
            query_timeout_ms=int(os.environ.get("GRAFANA_BRIDGE_QUERY_TIMEOUT_MS", "20000")),
            iris_cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_IRIS_CACHE_TTL", "15")),
            github_cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_GITHUB_CACHE_TTL", "60")),
            http_timeout=float(os.environ.get("GRAFANA_BRIDGE_HTTP_TIMEOUT", "10")),
            github_token=os.environ.get("GITHUB_TOKEN") or None,
        )
