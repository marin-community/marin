# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin GCP Compute REST wrapper used by :func:`rigging.resolver.vm_address`.

This module is intentionally minimal — we only need "given a VM name, get
its primary internal IP". The full provisioning surface lives in iris
(``iris.cluster.providers.gcp.service``); we deliberately do not pull that
across the dep boundary. If the REST shape diverges we'll consolidate
later, per the log-store extraction plan §Phase 2.

``httpx`` and ``google.auth`` are imported at function-call time. Both are
transitive deps of every environment that already imports rigging
(``gcsfs`` → ``google-auth``; ``connect-python`` ecosystem → ``httpx`` in
the iris dev install), but we don't make them required for rigging itself
since the resolver is a sidecar. If either is missing, we raise a clear
``NotImplementedError`` directing the operator at the install path.
"""

import logging

_COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TIMEOUT_SECONDS = 10.0


logger = logging.getLogger(__name__)


def gcp_internal_ip(name: str) -> str:
    """Return the primary internal IP for the GCP VM ``name``.

    Looks the VM up via Compute Engine's ``aggregated/instances`` REST
    endpoint so the caller does not need to know which zone the VM lives
    in. Raises ``LookupError`` if no VM with that name exists in the
    active project.
    """
    project_id, token = _gcp_credentials()
    vm_data = _fetch_vm_aggregated(project_id, token, name)
    if vm_data is None:
        raise LookupError(f"no GCP VM named {name!r} found in project {project_id!r}")
    network_interfaces = vm_data.get("networkInterfaces") or []
    if not network_interfaces:
        raise LookupError(f"GCP VM {name!r} has no network interfaces")
    internal_ip = network_interfaces[0].get("networkIP")
    if not internal_ip:
        raise LookupError(f"GCP VM {name!r} has no networkIP on its primary interface")
    return internal_ip


def _gcp_credentials() -> tuple[str, str]:
    """Return ``(project_id, oauth_token)`` from ADC.

    Defers ``google.auth`` import to call time so rigging's runtime
    doesn't grow a hard ``google-auth`` dep just to support an
    optional resolver path.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise NotImplementedError(
            "GCP `vm_address` requires `google-auth` and `httpx`; install rigging in an "
            "environment that has them (the iris dev environment does)."
        ) from exc
    creds, project_id = google.auth.default(scopes=[_OAUTH_SCOPE])
    if not project_id:
        raise LookupError("google.auth.default() returned no project_id; set GOOGLE_CLOUD_PROJECT")
    creds.refresh(google.auth.transport.requests.Request())
    return project_id, creds.token


def _fetch_vm_aggregated(project_id: str, token: str, name: str) -> dict | None:
    """Page through ``aggregated/instances`` looking for ``name``.

    Defers ``httpx`` import to call time for the same reason as
    ``_gcp_credentials`` above.
    """
    try:
        import httpx
    except ImportError as exc:
        raise NotImplementedError(
            "GCP `vm_address` requires `httpx`; install rigging in an environment that "
            "has it (the iris dev environment does)."
        ) from exc

    url = f"{_COMPUTE_BASE}/projects/{project_id}/aggregated/instances"
    headers = {"Authorization": f"Bearer {token}"}
    params: dict[str, str] = {"filter": f"name eq {name}"}
    with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
        while True:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            for scope in (data.get("items") or {}).values():
                for vm in scope.get("instances") or []:
                    if vm.get("name") == name:
                        return vm
            token_param = data.get("nextPageToken")
            if not token_param:
                return None
            params["pageToken"] = token_param
