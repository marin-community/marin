# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""URL-scheme resolver: ``iris://``, ``gcp://``, or bare ``host:port`` → ``(host, port)``.

External schemes register handlers via :func:`register_scheme`; ``gcp://`` is
built in. The iris-side ``iris://`` handler lives in
``iris.client.resolver_plugin`` and is installed when ``iris.client`` is imported.
"""

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

_COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TIMEOUT_SECONDS = 10.0
_DEFAULT_GCP_PORT = 10002


@dataclass(frozen=True)
class ServiceURL:
    scheme: str
    host: str
    port: int | None
    query: dict[str, str]

    @classmethod
    def parse(cls, ref: str) -> "ServiceURL":
        parts = urlsplit(ref)
        if not parts.scheme:
            raise ValueError(f"missing scheme in URL: {ref!r}")
        if parts.username or parts.password:
            raise ValueError(f"userinfo not supported in URL: {ref!r}")
        host = parts.hostname or parts.netloc
        if not host:
            raise ValueError(f"missing host in URL: {ref!r}")
        query = {k: v[0] for k, v in parse_qs(parts.query).items() if v}
        return cls(scheme=parts.scheme, host=host, port=parts.port, query=query)


SchemeHandler = Callable[[ServiceURL], tuple[str, int]]

_HANDLERS: dict[str, SchemeHandler] = {}


def register_scheme(scheme: str, handler: SchemeHandler) -> None:
    _HANDLERS[scheme] = handler


def is_registered(scheme: str) -> bool:
    return scheme in _HANDLERS


def resolve(ref: str) -> tuple[str, int]:
    if "://" not in ref:
        host, port = ref.rsplit(":", 1)
        return host, int(port)
    url = ServiceURL.parse(ref)
    handler = _HANDLERS.get(url.scheme)
    if handler is None:
        raise ValueError(f"unsupported scheme: {url.scheme!r}")
    return handler(url)


def gcp_vm_address(name: str, *, port: int = _DEFAULT_GCP_PORT) -> tuple[str, int]:
    return _gcp_internal_ip(name), port


def _gcp_internal_ip(name: str) -> str:
    project_id, token = _gcp_credentials()
    vm = _fetch_vm_aggregated(project_id, token, name)
    if vm is None:
        raise LookupError(f"no GCP VM named {name!r} found in project {project_id!r}")
    interfaces = vm.get("networkInterfaces") or []
    if not interfaces:
        raise LookupError(f"GCP VM {name!r} has no network interfaces")
    internal_ip = interfaces[0].get("networkIP")
    if not internal_ip:
        raise LookupError(f"GCP VM {name!r} has no networkIP on its primary interface")
    return internal_ip


def _gcp_credentials() -> tuple[str, str]:
    import google.auth
    import google.auth.transport.requests

    creds, project_id = google.auth.default(scopes=[_OAUTH_SCOPE])
    if not project_id:
        raise LookupError("google.auth.default() returned no project_id; set GOOGLE_CLOUD_PROJECT")
    creds.refresh(google.auth.transport.requests.Request())
    return project_id, creds.token


def _fetch_vm_aggregated(project_id: str, token: str, name: str) -> dict | None:
    import httpx

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
            page_token = data.get("nextPageToken")
            if not page_token:
                return None
            params["pageToken"] = page_token


register_scheme("gcp", lambda url: gcp_vm_address(url.host, port=url.port or _DEFAULT_GCP_PORT))
