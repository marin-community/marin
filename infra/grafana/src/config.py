# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge configuration, resolved once at startup from the container environment.

One bridge serves every cluster in :data:`CLUSTERS`; Grafana provisions one
datasource per entry and addresses it by name in the URL path. Both finelog VMs
live in the same project and zone and are reached over Direct VPC egress on their
internal IPs, so a single Cloud Run service covers prod and dev.

Cluster targets are code, not environment: they change when we stand up a
cluster, which is a reviewed event, and a typo in an env var would otherwise
silently point prod dashboards at dev data.
"""

import dataclasses
import os

# The finelog namespace infra/probes writes its flat metric samples to, and the
# source of every series the bridge serves (see infra/probes/src/sample.py for the
# row shape). Copied rather than imported: infra/probes depends on marin-iris, and
# Grafana must stay readable when the cluster it monitors is not.
#
# The producer's copy is PROBE_RESULTS_NAMESPACE in infra/probes/src/infra_probes.py.
# The two are only coupled by this string, so changing it there greps to here;
# every panel silently empties if they diverge.
CANARY_METRICS_NAMESPACE = "infra.canary.metrics"

# finelog listens here on both VMs (lib/finelog/config/{marin,marin-dev}.yaml).
FINELOG_PORT = 10001

# The bridge's loopback port. A constant, not a setting: it is a private contract
# between two processes in one container, and the provisioned datasource URLs
# (provisioning/datasources/finelog.yaml) name it literally — making it
# configurable would only create a way to break every panel at once.
BRIDGE_PORT = 8081


@dataclasses.dataclass(frozen=True)
class ClusterTarget:
    """One finelog deployment the bridge can query.

    ``instance_filter`` is a GCE list filter selecting the VM; ``namespace`` is
    the finelog table its series are read from.
    """

    name: str
    project: str
    zone: str
    instance_filter: str
    namespace: str = CANARY_METRICS_NAMESPACE


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

    # Rows one query may pull back. finelog caps a response at 64 MiB but only
    # after collecting it, so the scan bound has to come from this side.
    max_rows: int
    # Result cache TTL. Grafana's own query caching is Enterprise-only, and a
    # shared dashboard auto-refreshing across viewers would otherwise multiply
    # straight through to the finelog hub.
    cache_ttl: float
    # Refusal floor for a requested window: panels asking for more than this get
    # an error rather than a scan of the whole namespace.
    max_window_hours: float
    query_timeout_ms: int

    @staticmethod
    def from_environment() -> "BridgeConfig":
        """Read settings from the container env, falling back to prod-safe defaults."""
        return BridgeConfig(
            max_rows=int(os.environ.get("GRAFANA_BRIDGE_MAX_ROWS", "200000")),
            cache_ttl=float(os.environ.get("GRAFANA_BRIDGE_CACHE_TTL", "20")),
            max_window_hours=float(os.environ.get("GRAFANA_BRIDGE_MAX_WINDOW_HOURS", "168")),
            query_timeout_ms=int(os.environ.get("GRAFANA_BRIDGE_QUERY_TIMEOUT_MS", "20000")),
        )
