# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``iris.cluster.endpoints.resolve_endpoint_uri``."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import mock_open, patch

import pytest
from iris.cluster import endpoints
from iris.cluster.endpoints import register_scheme, resolve_endpoint_uri


def test_http_passthrough():
    assert resolve_endpoint_uri("http://example:1234") == "http://example:1234"
    assert resolve_endpoint_uri("https://example:1234") == "https://example:1234"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown endpoint scheme"):
        resolve_endpoint_uri("foo://bar")


def test_register_scheme_extensibility():
    calls: list[tuple[str, dict[str, str]]] = []

    def fake(uri: str, md: dict[str, str]) -> str:
        calls.append((uri, md))
        return "http://fake:1"

    register_scheme("xfake", fake)
    try:
        assert resolve_endpoint_uri("xfake://blah", {"k": "v"}) == "http://fake:1"
        assert calls == [("xfake://blah", {"k": "v"})]
    finally:
        endpoints._REGISTRY.pop("xfake", None)


# ---------------------------------------------------------------------------
# gcp://
# ---------------------------------------------------------------------------


def _gcloud_ok(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_gcp_basic():
    payload = json.dumps({"networkInterfaces": [{"networkIP": "10.0.0.5"}]})
    with patch("iris.cluster.endpoints.subprocess.run", return_value=_gcloud_ok(payload)) as run:
        out = resolve_endpoint_uri(
            "gcp://my-vm:10001",
            {"project": "p", "zone": "us-central1-a"},
        )
    assert out == "http://10.0.0.5:10001"
    cmd = run.call_args[0][0]
    assert "my-vm" in cmd
    assert "--project=p" in cmd
    assert "--zone=us-central1-a" in cmd


def test_gcp_port_from_metadata():
    payload = json.dumps({"networkInterfaces": [{"networkIP": "10.0.0.6"}]})
    with patch("iris.cluster.endpoints.subprocess.run", return_value=_gcloud_ok(payload)):
        out = resolve_endpoint_uri(
            "gcp://my-vm",
            {"project": "p", "zone": "z", "port": "9000"},
        )
    assert out == "http://10.0.0.6:9000"


def test_gcp_uses_adc_project_when_metadata_missing():
    payload = json.dumps({"networkInterfaces": [{"networkIP": "10.0.0.7"}]})
    with (
        patch("iris.cluster.endpoints.subprocess.run", return_value=_gcloud_ok(payload)) as run,
        patch("iris.cluster.endpoints.google.auth.default", return_value=(None, "adc-project")),
    ):
        out = resolve_endpoint_uri("gcp://vm:1234", {"zone": "z"})
    assert out == "http://10.0.0.7:1234"
    assert "--project=adc-project" in run.call_args[0][0]


def test_gcp_missing_project_raises():
    with patch("iris.cluster.endpoints.google.auth.default", return_value=(None, None)):
        with pytest.raises(ValueError, match=r"metadata\.project"):
            resolve_endpoint_uri("gcp://vm:1", {"zone": "z"})


def test_gcp_missing_zone_raises():
    with pytest.raises(ValueError, match=r"metadata\.zone"):
        resolve_endpoint_uri("gcp://vm:1", {"project": "p"})


def test_gcp_missing_port_raises():
    with pytest.raises(ValueError, match="port"):
        resolve_endpoint_uri("gcp://vm", {"project": "p", "zone": "z"})


def test_gcp_no_network_ip_raises():
    payload = json.dumps({"networkInterfaces": []})
    with patch("iris.cluster.endpoints.subprocess.run", return_value=_gcloud_ok(payload)):
        with pytest.raises(RuntimeError, match="networkInterfaces"):
            resolve_endpoint_uri("gcp://vm:1", {"project": "p", "zone": "z"})


def test_gcp_gcloud_failure_raises():
    err = subprocess.CalledProcessError(returncode=1, cmd=["gcloud"], stderr="boom: not found")
    with patch("iris.cluster.endpoints.subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="boom: not found"):
            resolve_endpoint_uri("gcp://vm:1", {"project": "p", "zone": "z"})


# ---------------------------------------------------------------------------
# k8s://
# ---------------------------------------------------------------------------


def test_k8s_explicit_namespace():
    assert resolve_endpoint_uri("k8s://svc.ns:1234") == "http://svc.ns.svc.cluster.local:1234"


def test_k8s_namespace_from_metadata():
    assert resolve_endpoint_uri("k8s://svc:1234", {"namespace": "iris"}) == "http://svc.iris.svc.cluster.local:1234"


def test_k8s_namespace_from_pod_file():
    mo = patch("builtins.open", new=_mock_open_returning("iris\n"))
    with mo:
        out = resolve_endpoint_uri("k8s://svc:1234")
    assert out == "http://svc.iris.svc.cluster.local:1234"


def test_k8s_port_from_metadata():
    out = resolve_endpoint_uri("k8s://svc.ns", {"port": "9001"})
    assert out == "http://svc.ns.svc.cluster.local:9001"


def test_k8s_no_namespace_raises():
    with patch("builtins.open", side_effect=OSError):
        with pytest.raises(ValueError, match="namespace"):
            resolve_endpoint_uri("k8s://svc:1234")


def test_k8s_no_port_raises():
    with pytest.raises(ValueError, match="port"):
        resolve_endpoint_uri("k8s://svc.ns")


def test_k8s_no_api_calls():
    with (
        patch("iris.cluster.endpoints.subprocess.run") as run,
        patch("iris.cluster.endpoints.google.auth.default") as adc,
    ):
        out = resolve_endpoint_uri("k8s://svc.ns:1234")
    assert out == "http://svc.ns.svc.cluster.local:1234"
    run.assert_not_called()
    adc.assert_not_called()


def _mock_open_returning(data: str):
    """Return a patcher target callable mimicking ``open()`` for a single file."""
    return mock_open(read_data=data)
