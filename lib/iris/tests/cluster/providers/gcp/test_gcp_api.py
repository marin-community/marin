# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCPApi — HTTP client for GCP REST APIs.

Uses httpx.MockTransport to verify URL construction, error mapping,
pagination, and auth header injection without hitting real GCP.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import patch

import httpx
import pytest

from iris.cluster.providers.gcp.api import (
    COMPUTE_BASE,
    LOGGING_BASE,
    TPU_BASE,
    GCPApi,
)
from iris.cluster.providers.types import (
    InfraError,
    QuotaExhaustedError,
    ResourceNotFoundError,
)

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
    return patch("iris.cluster.providers.gcp.api.google.auth.default", return_value=(cred, PROJECT))


def _make_api(handler: Callable[[httpx.Request], httpx.Response]) -> GCPApi:
    """Create a GCPApi with a mock HTTP transport and fake credentials."""
    api = GCPApi(PROJECT)
    api._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10)
    # Inject fake token so _refresh_token isn't called
    api._token = "fake-token"
    api._expires_at = float("inf")
    return api


def _json_response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


# ========================================================================
# Error mapping
# ========================================================================


def test_404_raises_resource_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})

    api = _make_api(handler)
    with pytest.raises(ResourceNotFoundError, match="Not found"):
        api.tpu_get("no-such-tpu", ZONE)
    api.close()


def test_429_raises_quota_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
        )

    api = _make_api(handler)
    with pytest.raises(QuotaExhaustedError, match="Quota exceeded"):
        api.tpu_get("some-tpu", ZONE)
    api.close()


def test_500_raises_infra_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500, "message": "Internal error"}})

    api = _make_api(handler)
    with pytest.raises(InfraError, match="Internal error"):
        api.tpu_get("some-tpu", ZONE)
    api.close()


def test_non_json_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    api = _make_api(handler)
    with pytest.raises(InfraError, match="Bad Gateway"):
        api.tpu_get("some-tpu", ZONE)
    api.close()


# ========================================================================
# Auth headers
# ========================================================================


def test_auth_header_injected():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "test", "state": "READY"})

    api = _make_api(handler)
    api.tpu_get("my-tpu", ZONE)
    api.close()

    assert len(requests_seen) == 1
    assert requests_seen[0].headers["authorization"] == "Bearer fake-token"


def test_token_refresh_on_expiry():
    """When token is expired, _refresh_token is called."""
    with _mock_credentials():

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"name": "test", "state": "READY"})

        api = _make_api(handler)
        api._token = None  # Force refresh
        api._expires_at = 0.0
        api.tpu_get("my-tpu", ZONE)
        assert api._token == "fake-token"
        api.close()


# ========================================================================
# TPU operations — URL construction
# ========================================================================


def test_tpu_create_url_and_params():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/op-123"})

    api = _make_api(handler)
    api.tpu_create("my-tpu", ZONE, {"acceleratorType": "v4-8"})
    api.close()

    req = requests_seen[0]
    assert req.method == "POST"
    assert f"{TPU_BASE}/projects/{PROJECT}/locations/{ZONE}/nodes" in str(req.url)
    assert "nodeId=my-tpu" in str(req.url)


def test_tpu_get_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"name": f"projects/{PROJECT}/locations/{ZONE}/nodes/my-tpu", "state": "READY"})

    api = _make_api(handler)
    result = api.tpu_get("my-tpu", ZONE)
    api.close()

    assert result["state"] == "READY"


def test_tpu_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    api = _make_api(handler)
    api.tpu_delete("gone-tpu", ZONE)  # should not raise
    api.close()


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
        return _json_response(
            {
                "nodes": [{"name": "tpu-2", "state": "READY"}],
            }
        )

    api = _make_api(handler)
    results = api.tpu_list(ZONE)
    api.close()

    assert len(results) == 2
    assert results[0]["name"] == "tpu-1"
    assert results[1]["name"] == "tpu-2"
    assert call_count == 2


# ========================================================================
# Queued resource operations
# ========================================================================


def test_queued_resource_create_url():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/op-456"})

    api = _make_api(handler)
    api.queued_resource_create("my-qr", ZONE, {"tpu": {"nodeSpec": []}})
    api.close()

    req = requests_seen[0]
    assert req.method == "POST"
    assert "/queuedResources" in str(req.url)
    assert "queuedResourceId=my-qr" in str(req.url)


def test_queued_resource_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    api = _make_api(handler)
    api.queued_resource_delete("gone-qr", ZONE)  # should not raise
    api.close()


def test_queued_resource_delete_passes_force():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/op-789"})

    api = _make_api(handler)
    api.queued_resource_delete("my-qr", ZONE)
    api.close()

    assert "force=true" in str(requests_seen[0].url)


# ========================================================================
# Compute operations
# ========================================================================


def test_instance_insert_url():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/op-vm"})

    api = _make_api(handler)
    api.instance_insert(ZONE, {"name": "my-vm", "machineType": "n1-standard-4"})
    api.close()

    req = requests_seen[0]
    assert req.method == "POST"
    assert f"{COMPUTE_BASE}/projects/{PROJECT}/zones/{ZONE}/instances" in str(req.url)


def test_instance_delete_ignores_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    api = _make_api(handler)
    api.instance_delete("gone-vm", ZONE)  # should not raise
    api.close()


def test_instance_reset_url():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/reset"})

    api = _make_api(handler)
    api.instance_reset("my-vm", ZONE)
    api.close()

    assert "/my-vm/reset" in str(requests_seen[0].url)


def test_instance_set_labels_url_and_body():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"name": "operations/labels"})

    api = _make_api(handler)
    api.instance_set_labels("my-vm", ZONE, {"env": "test"}, "abc123")
    api.close()

    req = requests_seen[0]
    assert "/my-vm/setLabels" in str(req.url)
    body = json.loads(req.content)
    assert body["labels"] == {"env": "test"}
    assert body["labelFingerprint"] == "abc123"


def test_instance_get_serial_port_output():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"contents": "serial output here", "next": 42})

    api = _make_api(handler)
    result = api.instance_get_serial_port_output("my-vm", ZONE, start=10)
    api.close()

    assert result["contents"] == "serial output here"


def test_instance_list_project_wide():
    """Project-wide list uses aggregatedList and flattens across zones."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "items": {
                    "zones/us-central1-a": {
                        "instances": [{"name": "vm-1", "status": "RUNNING"}],
                    },
                    "zones/us-east1-b": {
                        "instances": [{"name": "vm-2", "status": "RUNNING"}],
                    },
                    "zones/us-west1-a": {
                        "warning": {"code": "NO_RESULTS_ON_PAGE"},
                    },
                }
            }
        )

    api = _make_api(handler)
    results = api.instance_list(zone=None)
    api.close()

    names = {r["name"] for r in results}
    assert names == {"vm-1", "vm-2"}


def test_instance_list_with_zone():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "items": [{"name": "vm-1", "status": "RUNNING"}],
            }
        )

    api = _make_api(handler)
    results = api.instance_list(zone=ZONE)
    api.close()

    assert len(results) == 1
    assert results[0]["name"] == "vm-1"


def test_instance_list_with_filter():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response({"items": []})

    api = _make_api(handler)
    api.instance_list(zone=ZONE, filter_str="labels.env=test")
    api.close()

    assert "filter=labels.env%3Dtest" in str(requests_seen[0].url)


# ========================================================================
# Cloud Logging
# ========================================================================


def test_logging_list_entries():
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return _json_response(
            {
                "entries": [
                    {"textPayload": "line 1"},
                    {"textPayload": "line 2"},
                ]
            }
        )

    api = _make_api(handler)
    entries = api.logging_list_entries("some filter", limit=50)
    api.close()

    assert len(entries) == 2
    req = requests_seen[0]
    assert req.method == "POST"
    assert f"{LOGGING_BASE}/entries:list" in str(req.url)
    body = json.loads(req.content)
    assert body["filter"] == "some filter"
    assert body["pageSize"] == 50


# ========================================================================
# Operation waiting — vm_create / tpu_create must wait for async operations
# ========================================================================


def test_instance_insert_wait_polls_until_done():
    """instance_insert_wait should poll the zone operation until DONE."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        # POST to create the instance
        if request.method == "POST" and "/instances" in str(request.url) and "/operations/" not in str(request.url):
            return _json_response(
                {"name": "op-123", "status": "RUNNING", "zone": f"zones/{ZONE}", "kind": "compute#operation"}
            )
        # GET to poll the operation
        if "/operations/op-123" in str(request.url):
            call_count += 1
            if call_count < 2:
                return _json_response({"name": "op-123", "status": "RUNNING"})
            return _json_response({"name": "op-123", "status": "DONE"})
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    api = _make_api(handler)
    api.instance_insert_wait(ZONE, {"name": "my-vm"})
    api.close()

    assert call_count == 2, "Must poll operation until DONE"


def test_instance_insert_wait_raises_on_operation_error():
    """If the operation completes with an error, raise InfraError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "/instances" in str(request.url) and "/operations/" not in str(request.url):
            return _json_response(
                {"name": "op-err", "status": "RUNNING", "zone": f"zones/{ZONE}", "kind": "compute#operation"}
            )
        if "/operations/op-err" in str(request.url):
            return _json_response(
                {
                    "name": "op-err",
                    "status": "DONE",
                    "error": {"errors": [{"code": "QUOTA_EXCEEDED", "message": "Insufficient quota"}]},
                }
            )
        return httpx.Response(404, json={"error": {"code": 404, "message": "Not found"}})

    api = _make_api(handler)
    with pytest.raises(InfraError, match="Insufficient quota"):
        api.instance_insert_wait(ZONE, {"name": "my-vm"})
    api.close()


def test_vm_create_waits_for_operation():
    """vm_create must wait for the insert operation before describing the VM.

    This is the core regression from replacing `gcloud compute instances create`
    (which blocks until RUNNING) with the REST API (which returns immediately).
    The mock simulates real GCP behavior: instance_get returns 404 until the
    zone operation reaches DONE.
    """
    from iris.cluster.providers.gcp.service import CloudGcpService, VmCreateRequest

    operation_done = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal operation_done
        url = str(request.url)

        # POST instances — create (returns operation)
        if request.method == "POST" and url.endswith(f"/zones/{ZONE}/instances"):
            return _json_response(
                {"name": "op-vm-1", "status": "RUNNING", "zone": f"zones/{ZONE}", "kind": "compute#operation"}
            )

        # GET operation poll — marks operation as done
        if "/operations/op-vm-1" in url and request.method == "GET":
            operation_done = True
            return _json_response({"name": "op-vm-1", "status": "DONE"})

        # GET instance — only succeeds after operation completed (real GCP behavior)
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

    api = _make_api(handler)
    svc = CloudGcpService(PROJECT, api=api)
    info = svc.vm_create(VmCreateRequest(name="test-vm", zone=ZONE, machine_type="n1-standard-4", labels={}))
    api.close()

    assert info.name == "test-vm"
    assert info.internal_ip == "10.0.0.1"
    assert operation_done, "vm_create must poll the operation before describing"


def test_tpu_create_waits_for_operation():
    """tpu_create must wait for the LRO before describing the TPU.

    Same race condition as vm_create: the REST API create returns an LRO,
    and the TPU node may not be visible via tpu_get until the operation completes.
    """
    from iris.cluster.providers.gcp.service import CloudGcpService, TpuCreateRequest
    from iris.rpc import config_pb2

    operation_done = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal operation_done
        url = str(request.url)

        # POST nodes — create (returns LRO)
        if request.method == "POST" and "/nodes" in url and "nodeId=test-tpu" in url:
            return _json_response({"name": f"projects/{PROJECT}/locations/{ZONE}/operations/op-tpu-1", "done": False})

        # GET operation poll (TPU LRO) — marks operation as done
        if "/operations/op-tpu-1" in url and request.method == "GET":
            operation_done = True
            return _json_response({"name": f"projects/{PROJECT}/locations/{ZONE}/operations/op-tpu-1", "done": True})

        # GET node — only succeeds after operation completed
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

    api = _make_api(handler)
    svc = CloudGcpService(PROJECT, api=api)
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
    api.close()

    assert info.name == "test-tpu"
    assert info.state == "READY"
    assert operation_done, "tpu_create must poll the operation before describing"


def test_logging_list_entries_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({})

    api = _make_api(handler)
    entries = api.logging_list_entries("no match")
    api.close()

    assert entries == []
