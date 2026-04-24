# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VM-name → ``(host, port)`` lookup for the rigging URL resolver.

The single public entrypoint is :func:`vm_address`. Today it dispatches to
GCP only — CoreWeave and k8s lookups are called out as followups in the
log-store extraction plan.

The GCP path uses the Compute Engine REST API directly via ``httpx`` and
``google.auth`` (mirroring the iris pattern in
``iris.cluster.providers.gcp.service``). Those two libraries are already
present in every environment that imports rigging in practice (``gcsfs``
pulls them in transitively), but we do not declare them as hard rigging
dependencies — the import is therefore gated and yields a clear
``NotImplementedError`` if either is unavailable.

The default port — 10002 — matches the port baked into the finelog
Dockerfile in the log-store extraction plan. Callers that want a different
port can pass ``port=...`` explicitly.
"""

from rigging.resolver._gcp_lookup import gcp_internal_ip

DEFAULT_GCP_PORT = 10002


def vm_address(name: str, provider: str, *, port: int = DEFAULT_GCP_PORT) -> tuple[str, int]:
    """Look up the address of a system-service VM by name and provider.

    Returns ``(internal_ip, port)``. Today only ``provider="gcp"`` is
    implemented; CoreWeave and k8s lookups are tracked as followups in the
    log-store extraction plan and raise ``ValueError`` here.
    """
    if provider == "gcp":
        host = gcp_internal_ip(name)
        return host, port
    raise ValueError(f"unsupported provider: {provider}")
