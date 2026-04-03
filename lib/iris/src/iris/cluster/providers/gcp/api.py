# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-level HTTP client for GCP REST APIs (TPU v2, Compute v1, Cloud Logging).

Handles authentication (Application Default Credentials), token caching,
pagination, and error mapping to domain exceptions. Used by CloudGcpService
as a replacement for gcloud CLI subprocess calls.
"""

from __future__ import annotations

import json
import logging
import time

import google.auth
import google.auth.credentials
import google.auth.transport.requests
import httpx

from iris.cluster.providers.types import (
    InfraError,
    QuotaExhaustedError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)

TPU_BASE = "https://tpu.googleapis.com/v2"
COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
LOGGING_BASE = "https://logging.googleapis.com/v2"

_REFRESH_MARGIN = 300  # seconds before expiry to refresh token
_DEFAULT_TIMEOUT = 120  # seconds


class GCPApi:
    """Low-level HTTP client for GCP REST APIs with ADC auth and token caching."""

    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)
        self._creds: google.auth.credentials.Credentials | None = None
        self._token: str | None = None
        self._expires_at: float = 0.0

    def close(self) -> None:
        self._client.close()

    # -- Auth ---------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._token is None or time.monotonic() >= self._expires_at:
            self._refresh_token()
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _refresh_token(self) -> None:
        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._creds.refresh(google.auth.transport.requests.Request())
        self._token = self._creds.token
        now = time.monotonic()
        if self._creds.expiry is not None:
            self._expires_at = now + (self._creds.expiry.timestamp() - time.time()) - _REFRESH_MARGIN
        else:
            self._expires_at = now + _REFRESH_MARGIN

    # -- Error mapping ------------------------------------------------------

    def _classify_response(self, resp: httpx.Response) -> None:
        """Raise a domain exception for non-2xx responses."""
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
            error = body.get("error", {})
            message = error.get("message", resp.text)
            status = error.get("status", "")
            code = error.get("code", resp.status_code)
        except (json.JSONDecodeError, AttributeError):
            message = resp.text
            status = ""
            code = resp.status_code

        if code == 404 or status == "NOT_FOUND":
            raise ResourceNotFoundError(message)
        if code == 429 or status in ("RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"):
            raise QuotaExhaustedError(message)
        raise InfraError(f"GCP API error {code}: {message}")

    # -- Pagination ---------------------------------------------------------

    def _paginate(self, url: str, items_key: str, params: dict[str, str] | None = None) -> list[dict]:
        results: list[dict] = []
        p = dict(params or {})
        while True:
            resp = self._client.get(url, headers=self._headers(), params=p)
            self._classify_response(resp)
            data = resp.json()
            results.extend(data.get(items_key, []))
            token = data.get("nextPageToken")
            if not token:
                break
            p["pageToken"] = token
        return results

    def _paginate_raw(self, url: str, params: dict[str, str] | None = None) -> list[dict]:
        """Return raw page bodies (for aggregatedList where items_key varies)."""
        pages: list[dict] = []
        p = dict(params or {})
        while True:
            resp = self._client.get(url, headers=self._headers(), params=p)
            self._classify_response(resp)
            data = resp.json()
            pages.append(data)
            token = data.get("nextPageToken")
            if not token:
                break
            p["pageToken"] = token
        return pages

    # ======================================================================
    # TPU v2
    # ======================================================================

    def _tpu_parent(self, zone: str) -> str:
        return f"projects/{self._project_id}/locations/{zone}"

    def tpu_create(self, name: str, zone: str, body: dict) -> dict | None:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/nodes"
        resp = self._client.post(url, params={"nodeId": name}, headers=self._headers(), json=body)
        self._classify_response(resp)
        data = resp.json()
        # REST create returns a long-running operation, not the node itself.
        return data if data.get("name", "").endswith(f"/nodes/{name}") else None

    def tpu_get(self, name: str, zone: str) -> dict:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/nodes/{name}"
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def tpu_delete(self, name: str, zone: str) -> None:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/nodes/{name}"
        resp = self._client.delete(url, headers=self._headers())
        if resp.status_code != 404:
            self._classify_response(resp)

    def tpu_list(self, zone: str) -> list[dict]:
        return self._paginate(f"{TPU_BASE}/{self._tpu_parent(zone)}/nodes", "nodes")

    # ======================================================================
    # TPU v2 — Queued Resources
    # ======================================================================

    def queued_resource_create(self, name: str, zone: str, body: dict) -> None:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/queuedResources"
        resp = self._client.post(
            url,
            params={"queuedResourceId": name},
            headers=self._headers(),
            json=body,
        )
        self._classify_response(resp)

    def queued_resource_get(self, name: str, zone: str) -> dict:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/queuedResources/{name}"
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def queued_resource_delete(self, name: str, zone: str) -> None:
        url = f"{TPU_BASE}/{self._tpu_parent(zone)}/queuedResources/{name}"
        resp = self._client.delete(url, params={"force": "true"}, headers=self._headers())
        if resp.status_code != 404:
            self._classify_response(resp)

    def queued_resource_list(self, zone: str) -> list[dict]:
        return self._paginate(
            f"{TPU_BASE}/{self._tpu_parent(zone)}/queuedResources",
            "queuedResources",
        )

    # ======================================================================
    # Compute Engine v1 — Instances
    # ======================================================================

    def _instance_url(self, zone: str, name: str = "") -> str:
        path = f"{COMPUTE_BASE}/projects/{self._project_id}/zones/{zone}/instances"
        if name:
            path += f"/{name}"
        return path

    def instance_insert(self, zone: str, body: dict) -> dict:
        url = self._instance_url(zone)
        resp = self._client.post(url, headers=self._headers(), json=body)
        self._classify_response(resp)
        return resp.json()

    def instance_get(self, name: str, zone: str) -> dict:
        url = self._instance_url(zone, name)
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def instance_delete(self, name: str, zone: str) -> None:
        url = self._instance_url(zone, name)
        resp = self._client.delete(url, headers=self._headers())
        if resp.status_code != 404:
            self._classify_response(resp)

    def instance_list(self, zone: str | None = None, filter_str: str = "") -> list[dict]:
        params: dict[str, str] = {}
        if filter_str:
            params["filter"] = filter_str

        if zone:
            return self._paginate(self._instance_url(zone), "items", params)

        # Project-wide: aggregatedList, flatten across zones
        url = f"{COMPUTE_BASE}/projects/{self._project_id}/aggregated/instances"
        results: list[dict] = []
        for page in self._paginate_raw(url, params):
            for scope in page.get("items", {}).values():
                results.extend(scope.get("instances", []))
        return results

    def instance_reset(self, name: str, zone: str) -> None:
        url = self._instance_url(zone, name) + "/reset"
        resp = self._client.post(url, headers=self._headers())
        self._classify_response(resp)

    def instance_set_labels(self, name: str, zone: str, labels: dict[str, str], fingerprint: str) -> None:
        url = self._instance_url(zone, name) + "/setLabels"
        resp = self._client.post(
            url,
            headers=self._headers(),
            json={"labels": labels, "labelFingerprint": fingerprint},
        )
        self._classify_response(resp)

    def instance_set_metadata(self, name: str, zone: str, metadata_body: dict) -> None:
        url = self._instance_url(zone, name) + "/setMetadata"
        resp = self._client.post(url, headers=self._headers(), json=metadata_body)
        self._classify_response(resp)

    def instance_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> dict:
        url = self._instance_url(zone, name) + "/serialPort"
        resp = self._client.get(url, headers=self._headers(), params={"start": str(start)})
        self._classify_response(resp)
        return resp.json()

    # ======================================================================
    # Cloud Logging v2
    # ======================================================================

    def logging_list_entries(self, filter_str: str, limit: int = 200) -> list[dict]:
        url = f"{LOGGING_BASE}/entries:list"
        body = {
            "resourceNames": [f"projects/{self._project_id}"],
            "filter": filter_str,
            "pageSize": min(limit, 1000),
            "orderBy": "timestamp desc",
        }
        resp = self._client.post(url, headers=self._headers(), json=body, timeout=30)
        self._classify_response(resp)
        return resp.json().get("entries", [])
