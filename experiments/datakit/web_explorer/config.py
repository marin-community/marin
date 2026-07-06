# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""web_explorer configuration, resolved once at startup from the task environment.

All knobs are ``WEB_EXPLORER_*`` env vars, forwarded to the Iris task by
:mod:`experiments.datakit.web_explorer.deploy`. ``store`` is the one required
input (validated by the server entrypoint, where a ``--store`` CLI flag may
supply it instead); everything else is optional.
"""

from __future__ import annotations

import dataclasses
import os

_ENV_PREFIX = "WEB_EXPLORER_"

# Fixed Iris identifiers (not configurable): the named port the service binds/publishes,
# and the cluster-global endpoint it registers — the leading slash makes it reachable at
# ``/proxy/datakit_explorer/`` rather than a per-job path.
PORT_NAME = "datakit_explorer"
ENDPOINT_NAME = "/datakit_explorer"


@dataclasses.dataclass(frozen=True)
class WebExplorerConfig:
    """Resolved web_explorer configuration. Construct directly, or via :meth:`from_environment`."""

    store: str | None = None
    """Datakit store artifact path (``gs://…``) to explore. Required to serve."""

    ducky_url: str | None = None
    """Explicit ducky endpoint. Unset picks the environment default: the controller's
    internal proxy in-cluster, the public IAP-gated proxy for local dev."""

    lineage_cache: str | None = None
    """Path (local or ``gs://``) to cache the resolved lineage; skips the ~1-2 min
    ducky-backed re-resolve on restart."""

    query_timeout: float = 900.0
    """Per-request timeout (seconds) for calls to the ducky service."""

    source_summary: str | None = None
    """Path to a pre-baked per-source pipeline summary JSON (the Sources leaderboard)."""

    domain_centroids: str | None = None
    """Domain-centroids artifact pin, to disambiguate ``cluster_assign`` lineage."""

    quality_model: str | None = None
    """Quality-model name pin, to disambiguate ``quality`` lineage."""

    @classmethod
    def from_environment(cls) -> WebExplorerConfig:
        """Build from ``WEB_EXPLORER_*`` env vars; unset vars keep the field defaults."""

        def env(name: str) -> str | None:
            return os.environ.get(f"{_ENV_PREFIX}{name}") or None

        return cls(
            store=env("STORE"),
            ducky_url=env("DUCKY_URL"),
            lineage_cache=env("LINEAGE_CACHE"),
            query_timeout=float(os.environ.get(f"{_ENV_PREFIX}QUERY_TIMEOUT", cls.query_timeout)),
            source_summary=env("SOURCE_SUMMARY"),
            domain_centroids=env("DOMAIN_CENTROIDS"),
            quality_model=env("QUALITY_MODEL"),
        )
