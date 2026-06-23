# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the login orchestration, exercised with fake desktop + exchange seams."""

import pytest
from marin_cluster import login as login_flow
from marin_cluster.config import ClusterConfig
from rigging import credential_store
from rigging.cluster_manifest import AuthProvider, ClusterAuth, ClusterManifest, IapAuth


def _iap_config(**iap_kwargs) -> ClusterConfig:
    iap = IapAuth(url="https://iris-marin.oa.dev", **iap_kwargs)
    manifest = ClusterManifest(name="marin", dashboard_url=None, auth=ClusterAuth(AuthProvider.IAP, iap=iap))
    return ClusterConfig(manifest=manifest)


def _pin_config(monkeypatch, cfg: ClusterConfig) -> None:
    monkeypatch.setattr(ClusterConfig, "load", classmethod(lambda cls, cluster=None: cfg))


def test_login_writes_one_record(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    _pin_config(monkeypatch, _iap_config(desktop_oauth_client_id="cid", desktop_oauth_client_secret="sec"))

    seen = {}

    def fake_desktop(iap):
        seen["client_id"] = iap.desktop_oauth_client_id
        return "id-token", "refresh-token"

    def fake_exchange(endpoint, id_token):
        seen["endpoint"] = endpoint
        seen["id_token"] = id_token
        return login_flow.LoginResult(app_token="app-jwt", user_id="alice@x")

    record = login_flow.login("marin", desktop_login=fake_desktop, exchange=fake_exchange)

    assert seen == {"client_id": "cid", "endpoint": "https://iris-marin.oa.dev", "id_token": "id-token"}
    assert record.edge_refresh_token == "refresh-token"
    assert record.app_token == "app-jwt"
    assert record.metadata["user_id"] == "alice@x"
    # Persisted to the one store.
    assert credential_store.load_credentials("marin") == record


def test_login_rejects_non_iap_cluster(monkeypatch):
    manifest = ClusterManifest(name="coreweave", dashboard_url=None, auth=ClusterAuth(AuthProvider.GCP))
    _pin_config(monkeypatch, ClusterConfig(manifest=manifest))
    with pytest.raises(ValueError, match="supports IAP clusters"):
        login_flow.login("coreweave")


def test_login_requires_desktop_client_credentials(monkeypatch):
    _pin_config(monkeypatch, _iap_config())  # no client id/secret
    with pytest.raises(ValueError, match="no desktop_oauth_client_id/secret"):
        login_flow.login("marin", desktop_login=lambda iap: ("x", "y"))


def test_logout_removes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    _pin_config(monkeypatch, _iap_config(desktop_oauth_client_id="cid", desktop_oauth_client_secret="sec"))
    login_flow.login(
        "marin",
        desktop_login=lambda iap: ("id", "ref"),
        exchange=lambda e, t: login_flow.LoginResult(app_token="j", user_id="u"),
    )
    assert login_flow.logout("marin") is True
    assert login_flow.logout("marin") is False
