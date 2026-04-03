# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CloudGcpService REST API integration.

Uses httpx.MockTransport to verify URL construction, error mapping,
pagination, operation waiting, and auth header injection without hitting real GCP.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import patch

import httpx
import pytest

from iris.cluster.providers.gcp.service import (
    CloudGcpService,
    TpuCreateRequest,
    VmCreateRequest,
)
from iris.cluster.providers.types import (
    InfraError,
    QuotaExhaustedError,
)
from iris.rpc import config_pb2

PROJECT = "test-project"
ZONE = "us-central1-a"


def _mock_credentials():
    """Patch google.auth.default to return a fake credential."""
    cred = type(
        "FakeCred",
        (),
        {
            "token": "fake-token",
            "expiry": None,
            "refresh": lambda self, req: None,
        },
    )()
    return patch("iris.cluster.providers.gcp.service.google.auth.default", return_value=(cred, PROJECT))


def _make_svc(handler: Callable[[httpx.Request], httpx.Response]) -> CloudGcpService:
    """Create a CloudGcpService with a mock HTTP transport and fake credentials."""
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10)
    svc = CloudGcpService(PROJECT, http_client=client)
    svc._token = "fake-token"
    svc._expires_at = float("inf")
    return svc


def _json_response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


# ========================================================================
# Error mapping
# ========================================================================


def test_404_raises_resource_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})

    svc = _make_svc(handler)
    assert svc.tpu_describe("no-such-tpu", ZONE) is None
    svc.shutdown()


def test_429_raises_quota_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
        )

    svc = _make_svc(handler)
    with pytest.raises(QuotaExhaustedError, match="Quota exceeded"):
        svc.vm_reset("some-vm", ZONE)
    svc.shutdown()


def test_500_raises_infra_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500, "message": "Internal error"}})

    svc = _make_svc(handler)
    with pytest.raises(InfraError, match="Internal error"):
        svc.vm_reset("some-vm", ZONE)
    svc.shutdown()


# ========================================================================
# Auth headers
# ========================================================================


def test_auth_header_injected():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "test", "state": "READY"})

    svc = _make_svc(handler)
    svc.tpu_describe("my-tpu", ZONE)
    svc.shutdown()

    assert len(requests_seen) == 1
    assert requests_seen[0].headers["authorization"] == "Bearer fake-token"


def test_token_refresh_on_expiry():
    with _mock_credentials():

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"name": "test", "state": "READY"})

        svc = _make_svc(handler)
        svc._token = None
        svc._expires_at = 0.0
        svc.tpu_describe("my-tpu", ZONE)
        assert svc._token == "fake-token"
        svc.shutdown()


# ========================================================================
# VM create waits for operation before describing
# ========================================================================


def test_vm_create_waits_for_operation():
    """vm_create must wait for the insert operation before describing the VM."""
    operation_done = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal operation_done
        url = str(request.url)

        if request.method == "POST" and url.endswith(f"/zones/{ZONE}/instances"):
            return _json_response(
                {"name": "op-vm-1", "status": "RUNNING", "zone": f"zones/{ZONE}", "kind": "compute#operation"}
            )

        if "/operations/op-vm-1" in url and request.method == "GET":
            operation_done = True
            return _json_response({"name": "op-vm-1", "status": "DONE"})

        if request.method == "GET" and url.endswith(f"/zones/{ZONE}/instances/test-vm"):
            if not operation_done:
                return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})
            return _json_response(
                {
                    "name": "test-vm",
                    "status": "RUNNING",
                    "zone": f"projects/{PROJECT}/zones/{ZONE}",
                    "networkInterfaces": [{"networkIP": "10.0.0.1", "accessConfigs": [{"natIP": "34.1.2.3"}]}],
                    "metadata": {},
                    "serviceAccounts": [{"email": "sa@test.iam.gserviceaccount.com"}],
                    "creationTimestamp": "2026-01-01T00:00:00Z",
                }
            )

        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    svc = _make_svc(handler)
    info = svc.vm_create(VmCreateRequest(name="test-vm", zone=ZONE, machine_type="n1-standard-4", labels={}))
    svc.shutdown()

    assert info.name == "test-vm"
    assert info.internal_ip == "10.0.0.1"
    assert operation_done, "vm_create must poll the operation before describing"


def test_tpu_create_waits_for_operation():
    """tpu_create must wait for the LRO before describing the TPU."""
    operation_done = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal operation_done
        url = str(request.url)

        if request.method == "POST" and "/nodes" in url and "nodeId=test-tpu" in url:
            return _json_response({"name": f"projects/{PROJECT}/locations/{ZONE}/operations/op-tpu-1", "done": False})

        if "/operations/op-tpu-1" in url and request.method == "GET":
            operation_done = True
            return _json_response({"name": f"projects/{PROJECT}/locations/{ZONE}/operations/op-tpu-1", "done": True})

        if request.method == "GET" and url.endswith("/nodes/test-tpu"):
            if not operation_done:
                return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})
            return _json_response(
                {
                    "name": f"projects/{PROJECT}/locations/{ZONE}/nodes/test-tpu",
                    "state": "READY",
                    "acceleratorType": "v4-8",
                    "networkEndpoints": [{"ipAddress": "10.0.0.2"}],
                    "labels": {},
                    "metadata": {},
                    "createTime": "2026-01-01T00:00:00Z",
                }
            )

        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    svc = _make_svc(handler)
    info = svc.tpu_create(
        TpuCreateRequest(
            name="test-tpu",
            zone=ZONE,
            accelerator_type="v4-8",
            runtime_version="tpu-ubuntu2204-base",
            capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
            labels={},
        )
    )
    svc.shutdown()

    assert info.name == "test-tpu"
    assert info.state == "READY"
    assert operation_done, "tpu_create must poll the operation before describing"


# ========================================================================
# TPU list — only queries requested zones
# ========================================================================


def test_tpu_list_queries_only_requested_zones():
    zones_queried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/locations/" in url and "/nodes" in url:
            zone = url.split("/locations/")[1].split("/nodes")[0]
            zones_queried.append(zone)
        return _json_response({"nodes": []})

    svc = _make_svc(handler)
    svc.tpu_list(zones=["europe-west4-b", "us-west4-a"])
    svc.shutdown()

    assert sorted(zones_queried) == ["europe-west4-b", "us-west4-a"]


def test_tpu_list_label_filtering():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "nodes": [
                    {
                        "name": f"projects/{PROJECT}/locations/{ZONE}/nodes/tpu-match",
                        "state": "READY",
                        "acceleratorType": "v4-8",
                        "labels": {"env": "test", "managed": "true"},
                    },
                    {
                        "name": f"projects/{PROJECT}/locations/{ZONE}/nodes/tpu-nomatch",
                        "state": "READY",
                        "acceleratorType": "v4-8",
                        "labels": {"env": "prod"},
                    },
                ]
            }
        )

    svc = _make_svc(handler)
    results = svc.tpu_list(zones=[ZONE], labels={"env": "test"})
    svc.shutdown()

    assert len(results) == 1
    assert results[0].name == "tpu-match"


# ========================================================================
# VM list
# ========================================================================


def test_vm_list_project_wide_uses_aggregated_list():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response(
            {
                "items": {
                    "zones/us-central1-a": {
                        "instances": [
                            {"name": "vm-1", "status": "RUNNING", "zone": f"projects/{PROJECT}/zones/us-central1-a"}
                        ],
                    },
                    "zones/us-west1-a": {
                        "warning": {"code": "NO_RESULTS_ON_PAGE"},
                    },
                }
            }
        )

    svc = _make_svc(handler)
    results = svc.vm_list(zones=[])
    svc.shutdown()

    assert len(results) == 1
    assert results[0].name == "vm-1"
    assert "/aggregated/instances" in str(requests_seen[0].url)


def test_vm_list_with_labels_passes_filter():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"items": []})

    svc = _make_svc(handler)
    svc.vm_list(zones=[ZONE], labels={"env": "test"})
    svc.shutdown()

    assert "filter=labels.env%3Dtest" in str(requests_seen[0].url)


# ========================================================================
# Pagination
# ========================================================================


def test_tpu_list_with_pagination():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _json_response(
                {
                    "nodes": [{"name": "tpu-1", "state": "READY"}],
                    "nextPageToken": "page2",
                }
            )
        return _json_response({"nodes": [{"name": "tpu-2", "state": "READY"}]})

    svc = _make_svc(handler)
    results = svc.tpu_list(zones=[ZONE])
    svc.shutdown()

    assert len(results) == 2
    assert call_count == 2


# ========================================================================
# Cloud Logging
# ========================================================================


def test_logging_read():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"entries": [{"textPayload": "line 1"}, {"textPayload": "line 2"}]})

    svc = _make_svc(handler)
    entries = svc.logging_read("some filter", limit=50)
    svc.shutdown()

    assert entries == ["line 1", "line 2"]
    body = json.loads(requests_seen[0].content)
    assert body["filter"] == "some filter"
    assert body["pageSize"] == 50


def test_logging_read_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({})

    svc = _make_svc(handler)
    assert svc.logging_read("no match") == []
    svc.shutdown()


# ========================================================================
# Delete operations ignore 404
# ========================================================================


def test_tpu_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    svc = _make_svc(handler)
    svc.tpu_delete("gone-tpu", ZONE)
    svc.shutdown()


def test_vm_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    svc = _make_svc(handler)
    svc.vm_delete("gone-vm", ZONE)
    svc.shutdown()


def test_queued_resource_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    svc = _make_svc(handler)
    svc.queued_resource_delete("gone-qr", ZONE)
    svc.shutdown()


def test_queued_resource_delete_passes_force():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/op-789"})

    svc = _make_svc(handler)
    svc.queued_resource_delete("my-qr", ZONE)
    svc.shutdown()

    assert "force=true" in str(requests_seen[0].url)
