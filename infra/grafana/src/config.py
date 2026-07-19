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
class K8sClusterTarget:
    """A CoreWeave cluster whose k8s API server the bridge polls read-only."""

    name: str  # iris cluster name, e.g. "cw-us-east-08a"
    api_server: str  # public CKS API server URL


# All requests authenticate with the single org-wide CW read-role token from the
# CW_READ_TOKEN env var (Secret Manager: marin-grafana-cw-read-token).
K8S_CLUSTERS: tuple[K8sClusterTarget, ...] = (
    K8sClusterTarget("cw-us-east-02a", "https://208261-34513e48.k8s.us-east-02a.coreweave.com"),
    K8sClusterTarget("cw-us-east-08a", "https://208261-d2cd61ed.k8s.us-east-08a.coreweave.com"),
    K8sClusterTarget("cw-rno2a", "https://208261-6670debc.k8s.rno2a.coreweave.com"),
)


@dataclasses.dataclass(frozen=True)
class WatchedComponent:
    """A control-plane Deployment the k8s source reports on and alerts over."""

    namespace: str
    deployment: str

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.deployment}"


WATCHED_COMPONENTS: tuple[WatchedComponent, ...] = (
    WatchedComponent("kueue-system", "kueue-controller-manager"),
    WatchedComponent("iris", "iris-controller"),
    WatchedComponent("traefik", "traefik"),
    WatchedComponent("cert-manager", "cert-manager"),
)


@dataclasses.dataclass(frozen=True)
class WatchedWebhook:
    """An admission-webhook Service whose ready-endpoint count the bridge watches."""

    namespace: str
    service: str

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.service}"


WATCHED_WEBHOOKS: tuple[WatchedWebhook, ...] = (WatchedWebhook("kueue-system", "kueue-webhook-service"),)

# Namespaces the pod-level scans (crashloops, pending) skip, by prefix. CoreWeave's
# per-node daemons dominate the pod count on large clusters (thousands of pods,
# ~20KB of JSON each) and are CoreWeave's to operate; the namespaces we own hold
# on the order of a hundred pods.
PROVIDER_NAMESPACE_PREFIXES: tuple[str, ...] = ("cw-", "kube-")


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
    # Cache TTLs for the live Iris, GitHub, and k8s endpoints, seconds.
    iris_cache_ttl: float
    github_cache_ttl: float
    k8s_cache_ttl: float
    # HTTP timeout for the controller RPC, GitHub, and k8s API calls, seconds.
    http_timeout: float
    # GitHub token; lifts the REST rate limit and is required for the GraphQL build panel.
    github_token: str | None
    # CW read-role bearer token for the k8s API servers. None does not fail the boot
    # (that would take Grafana down with it); the k8s routes serve auth error rows
    # and unreachable=1 alert rows instead.
    cw_read_token: str | None

    @staticmethod
    def from_environment() -> "BridgeConfig":
        """Read settings from the container environment."""
        return BridgeConfig(
            max_rows=int(os.environ.get("GRAFANA_BRIDGE_MAX_ROWS", "200000")),
            cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_CACHE_TTL", "20")),
            query_timeout_ms=int(os.environ.get("GRAFANA_BRIDGE_QUERY_TIMEOUT_MS", "20000")),
            iris_cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_IRIS_CACHE_TTL", "15")),
            github_cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_GITHUB_CACHE_TTL", "60")),
            k8s_cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_K8S_CACHE_TTL", "30")),
            http_timeout=float(os.environ.get("GRAFANA_BRIDGE_HTTP_TIMEOUT", "10")),
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            cw_read_token=os.environ.get("CW_READ_TOKEN") or None,
        )
