# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated HTTP client helpers for Marina's command line."""

from urllib.parse import urlparse

import requests
from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapServiceAccountTokenProvider,
    TokenProvider,
)
from rigging.credential_store import credentials_dir
from rigging.credentials import iap_edge_provider

DEFAULT_MARINA_URL = "https://marina.oa.dev"
DEFAULT_TIMEOUT = 60
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})


class MarinaRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Marina request failed ({status_code}): {detail}")
        self.status_code = status_code


def _cached_login_provider() -> TokenProvider | None:
    provider = iap_edge_provider("marin")
    if provider is not None:
        return provider
    for path in sorted(credentials_dir().glob("*.json")):
        candidate = iap_edge_provider(path.stem)
        if candidate is not None:
            return candidate
    return None


def marina_token_provider() -> TokenProvider:
    """Resolve an expiring-token provider for reuse across a sequence of requests."""
    return _cached_login_provider() or IapServiceAccountTokenProvider(MARIN_DESKTOP_OAUTH_CLIENT.client_id)


def bearer_token(provider: TokenProvider | None = None) -> str:
    """Return an IAP token from a cached Marin login or ambient service account."""
    provider = provider if provider is not None else marina_token_provider()
    try:
        token = provider.get_token()
    except (IapCredentialsUnavailable, IapLoginRequired) as error:
        raise RuntimeError(f"{error}; human callers should run `iris login`") from error
    if not token:
        raise RuntimeError("could not obtain an IAP token for Marina")
    return token


def marina_request(
    service_url: str,
    method: str,
    path: str,
    *,
    json_body: object | None = None,
    payload: bytes | None = None,
    params: dict[str, object] | None = None,
    session: requests.Session | None = None,
    token_provider: TokenProvider | None = None,
) -> object | None:
    """Call Marina, optionally reusing a caller-owned transport and token provider."""
    service_url = service_url.rstrip("/")
    headers: dict[str, str] = {}
    if payload is not None:
        headers["Content-Type"] = "application/gzip"
    host = urlparse(service_url).hostname
    if host not in LOCAL_HOSTS:
        headers["Authorization"] = f"Bearer {bearer_token(token_provider)}"
    request = session.request if session is not None else requests.request
    response = request(
        method,
        service_url + path,
        params=params,
        json=json_body,
        data=payload,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code >= 400:
        detail = (
            response.json().get("detail", response.text)
            if "json" in response.headers.get("content-type", "")
            else response.text
        )
        raise MarinaRequestError(response.status_code, str(detail))
    if response.status_code == 204:
        return None
    return response.json()


def publish_applet(
    service_url: str,
    payload: bytes,
    applet_id: str | None = None,
    base_version: int | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    """Publish an applet archive and return Marina's structured response."""
    path = "/api/marina/applets" + (f"/{applet_id}" if applet_id is not None else "")
    value = marina_request(
        service_url,
        "POST",
        path,
        params={key: value for key, value in {"base_version": base_version, "mode": mode}.items() if value is not None}
        or None,
        payload=payload,
    )
    if not isinstance(value, dict):
        raise RuntimeError("Marina returned a non-object publish response")
    return value
