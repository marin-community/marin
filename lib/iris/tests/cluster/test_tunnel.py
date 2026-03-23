# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Cloudflare Tunnel integration module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from iris.cluster.tunnel import (
    TunnelConfig,
    TunnelHandle,
    _cf_request,
    _configure_tunnel_ingress,
    _find_dns_record,
    _find_existing_tunnel,
    _upsert_dns_record,
    start_tunnel,
    stop_tunnel,
)

TUNNEL_MODULE = "iris.cluster.tunnel"

# ---------------------------------------------------------------------------
# TunnelConfig
# ---------------------------------------------------------------------------


def test_tunnel_config_subdomain_hashes_cluster_name():
    config = TunnelConfig(cluster_name="marin-us-central2")
    assert config.subdomain.startswith("marin-")
    assert len(config.subdomain) == len("marin-") + 8  # 8 hex chars


def test_tunnel_config_subdomain_deterministic():
    a = TunnelConfig(cluster_name="marin-us-central2")
    b = TunnelConfig(cluster_name="marin-us-central2")
    assert a.subdomain == b.subdomain


def test_tunnel_config_subdomain_differs_per_cluster():
    a = TunnelConfig(cluster_name="marin-us-central2")
    b = TunnelConfig(cluster_name="marin-eu-west4")
    assert a.subdomain != b.subdomain


def test_tunnel_config_fqdn():
    config = TunnelConfig(cluster_name="test", domain="iris-ops.dev")
    assert config.fqdn.endswith(".iris-ops.dev")


def test_tunnel_config_public_url():
    config = TunnelConfig(cluster_name="test", domain="iris-ops.dev")
    assert config.public_url.startswith("https://")
    assert config.public_url.endswith(".iris-ops.dev")


def test_tunnel_config_empty_cluster_name():
    config = TunnelConfig(cluster_name="")
    assert config.subdomain == "marin"


# ---------------------------------------------------------------------------
# Cloudflare API helpers (with mocked httpx)
# ---------------------------------------------------------------------------


def _make_cf_response(result, success: bool = True) -> httpx.Response:
    """Build a mock httpx.Response that returns CF API JSON."""
    body = {"success": success, "result": result, "errors": []}
    return httpx.Response(200, json=body)


def test_cf_request_raises_on_failure():
    client = MagicMock()
    client.request.return_value = httpx.Response(
        400, json={"success": False, "errors": [{"message": "bad"}], "result": None}
    )
    with pytest.raises(RuntimeError, match="Cloudflare API error"):
        _cf_request(client, "GET", "/test", "token")


def test_find_existing_tunnel_returns_first_match():
    client = MagicMock()
    tunnel = {"id": "tun-123", "name": "iris-test"}
    client.request.return_value = _make_cf_response([tunnel])
    result = _find_existing_tunnel(client, "acct-1", "iris-test", "token")
    assert result == tunnel


def test_find_existing_tunnel_returns_none():
    client = MagicMock()
    client.request.return_value = _make_cf_response([])
    result = _find_existing_tunnel(client, "acct-1", "iris-test", "token")
    assert result is None


def test_find_dns_record_returns_match():
    client = MagicMock()
    record = {"id": "rec-1", "name": "test.iris-ops.dev", "type": "CNAME"}
    client.request.return_value = _make_cf_response([record])
    result = _find_dns_record(client, "zone-1", "test.iris-ops.dev", "token")
    assert result == record


def test_upsert_dns_record_creates_new():
    client = MagicMock()
    # First call: no existing record
    # Second call: create succeeds
    client.request.side_effect = [
        _make_cf_response([]),  # find_dns_record returns empty
        _make_cf_response({"id": "rec-new"}),  # create returns new record
    ]
    record_id = _upsert_dns_record(client, "zone-1", "test.iris-ops.dev", "tun-123", "token")
    assert record_id == "rec-new"


def test_upsert_dns_record_updates_existing():
    client = MagicMock()
    existing = {"id": "rec-existing", "name": "test.iris-ops.dev"}
    client.request.side_effect = [
        _make_cf_response([existing]),  # find_dns_record returns existing
        _make_cf_response(existing),  # update succeeds
    ]
    record_id = _upsert_dns_record(client, "zone-1", "test.iris-ops.dev", "tun-123", "token")
    assert record_id == "rec-existing"


def test_configure_tunnel_ingress():
    client = MagicMock()
    client.request.return_value = _make_cf_response({})
    _configure_tunnel_ingress(client, "acct-1", "tun-123", "test.iris-ops.dev", 10000, "token")
    call_args = client.request.call_args
    body = call_args.kwargs.get("json") or call_args[1].get("json")
    ingress = body["config"]["ingress"]
    assert ingress[0]["hostname"] == "test.iris-ops.dev"
    assert ingress[0]["service"] == "http://localhost:10000"
    assert ingress[1]["service"] == "http_status:404"


# ---------------------------------------------------------------------------
# start_tunnel / stop_tunnel integration (mocked)
# ---------------------------------------------------------------------------


@patch(f"{TUNNEL_MODULE}._launch_cloudflared")
@patch(f"{TUNNEL_MODULE}.httpx.Client")
def test_start_tunnel_full_flow(mock_client_cls, mock_launch):
    """start_tunnel creates tunnel, configures ingress, sets DNS, launches cloudflared."""
    client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    client.request.side_effect = [
        _make_cf_response([]),  # find_existing_tunnel → empty
        _make_cf_response({"id": "tun-abc"}),  # create_tunnel
        _make_cf_response({}),  # configure_tunnel_ingress
        _make_cf_response("tunnel-token-string"),  # get_tunnel_token
        _make_cf_response([]),  # find_dns_record → empty
        _make_cf_response({"id": "rec-xyz"}),  # create dns record
    ]

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_launch.return_value = mock_proc

    config = TunnelConfig(
        enabled=True,
        domain="iris-ops.dev",
        cloudflare_account_id="acct-1",
        cloudflare_zone_id="zone-1",
        api_token="fake-token",
        cluster_name="test-cluster",
    )

    handle = start_tunnel(config, controller_port=10000)

    assert handle.tunnel_id == "tun-abc"
    assert handle.dns_record_id == "rec-xyz"
    assert handle.public_url == config.public_url
    assert handle.process is mock_proc
    mock_launch.assert_called_once()


def test_start_tunnel_validates_required_fields():
    config = TunnelConfig(enabled=True, api_token="", cloudflare_account_id="", cloudflare_zone_id="")
    with pytest.raises(ValueError, match="api_token"):
        start_tunnel(config, controller_port=10000)

    config2 = TunnelConfig(enabled=True, api_token="tok", cloudflare_account_id="", cloudflare_zone_id="")
    with pytest.raises(ValueError, match="account_id"):
        start_tunnel(config2, controller_port=10000)


def test_stop_tunnel_terminates_process():
    proc = MagicMock()
    proc.poll.return_value = None  # still running
    handle = TunnelHandle(
        tunnel_id="tun-1",
        tunnel_token="tok",
        dns_record_id="rec-1",
        public_url="https://test.iris-ops.dev",
        process=proc,
    )
    config = TunnelConfig(api_token="fake", cloudflare_account_id="acct", cloudflare_zone_id="zone")

    stop_tunnel(handle, config, delete_tunnel=False)

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


@patch(f"{TUNNEL_MODULE}.httpx.Client")
def test_stop_tunnel_with_cleanup(mock_client_cls):
    """stop_tunnel with delete_tunnel=True deletes DNS and tunnel."""
    client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    client.request.side_effect = [
        _make_cf_response({}),  # delete_dns_record
        _make_cf_response({}),  # delete_tunnel
    ]

    handle = TunnelHandle(
        tunnel_id="tun-1",
        tunnel_token="tok",
        dns_record_id="rec-1",
        public_url="https://test.iris-ops.dev",
        process=None,
    )
    config = TunnelConfig(api_token="fake", cloudflare_account_id="acct", cloudflare_zone_id="zone")

    stop_tunnel(handle, config, delete_tunnel=True)

    assert client.request.call_count == 2
