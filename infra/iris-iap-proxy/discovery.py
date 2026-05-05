# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller VM discovery via GCE instance labels.

Queries the Compute Engine API for VMs matching the Iris controller label,
returns the internal IP address, and caches the result to avoid per-request
API calls. Mirrors the discovery pattern in
``iris.cluster.providers.gcp.controller.GcpControllerProvider.discover_controller``.
"""

import logging
import os
import time

from google.cloud import compute_v1

logger = logging.getLogger(__name__)

# When set, skip GCE discovery entirely and use this URL directly.
CONTROLLER_URL = os.environ.get("CONTROLLER_URL", "")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "hai-gcp-models")
CONTROLLER_ZONE = os.environ.get("CONTROLLER_ZONE", "us-central1-a")
CONTROLLER_LABEL = os.environ.get("CONTROLLER_LABEL", "iris-marin-controller")
CONTROLLER_PORT = int(os.environ.get("CONTROLLER_PORT", "10000"))

_CACHE_TTL_SECONDS = 60

_cached_url: str | None = None
_cache_expires_at: float = 0.0


def _query_controller_ip() -> str:
    """Query GCE for the controller VM's internal IP."""
    client = compute_v1.InstancesClient()
    request = compute_v1.ListInstancesRequest(
        project=GCP_PROJECT,
        zone=CONTROLLER_ZONE,
        filter=f"labels.{CONTROLLER_LABEL}=true AND status=RUNNING",
    )
    instances = list(client.list(request=request))
    if not instances:
        raise RuntimeError(
            f"No controller VM found (label={CONTROLLER_LABEL}=true, " f"project={GCP_PROJECT}, zone={CONTROLLER_ZONE})"
        )
    instance = instances[0]
    for iface in instance.network_interfaces:
        if iface.network_i_p:
            return iface.network_i_p
    raise RuntimeError(f"Controller VM {instance.name} has no internal IP")


def get_controller_url() -> str:
    """Return the controller's internal URL, refreshing the cache if stale.

    If ``CONTROLLER_URL`` is set, returns it directly (useful for local
    testing and static deployments).
    """
    if CONTROLLER_URL:
        return CONTROLLER_URL

    global _cached_url, _cache_expires_at

    now = time.monotonic()
    if _cached_url is not None and now < _cache_expires_at:
        return _cached_url

    ip = _query_controller_ip()
    url = f"http://{ip}:{CONTROLLER_PORT}"
    logger.info("Discovered controller at %s", url)
    _cached_url = url
    _cache_expires_at = now + _CACHE_TTL_SECONDS
    return url
