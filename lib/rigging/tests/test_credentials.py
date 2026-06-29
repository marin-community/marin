# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolution-order behavior for ``credentials_for`` and the interceptor chain."""

import pytest
from rigging import credentials as creds
from rigging.auth import (
    BearerTokenInjector,
    GcpAccessTokenProvider,
    IapRefreshTokenProvider,
    IapServiceAccountTokenProvider,
    StaticTokenProvider,
)
from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.credential_store import CredentialRecord
from rigging.credentials import ClientCredentials, credentials_for


@pytest.fixture(autouse=True)
def _no_real_store(monkeypatch):
    """Default: no cached login and no env override, so each test sets its own."""
    monkeypatch.delenv(creds.MARIN_CLUSTER_TOKEN_ENV, raising=False)
    monkeypatch.setattr(creds, "load_credentials", lambda cluster: None)


def _record(**kw) -> CredentialRecord:
    base = dict(cluster="marin", endpoint="https://iris")
    base.update(kw)
    return CredentialRecord(**base)


def _iap_auth(**kw) -> ClusterAuth:
    return ClusterAuth(AuthProvider.IAP, iap=IapAuth(url="https://iris", **kw))


def test_env_override_wins_over_file_and_provider(monkeypatch):
    monkeypatch.setenv(creds.MARIN_CLUSTER_TOKEN_ENV, "env-tok")
    monkeypatch.setattr(creds, "load_credentials", lambda cluster: _record(app_token="file-tok"))
    c = credentials_for("marin", ClusterAuth(AuthProvider.GCP))
    assert isinstance(c.token_provider, StaticTokenProvider)
    assert c.token_provider.get_token() == "env-tok"


def test_login_file_app_token_used_when_no_env(monkeypatch):
    monkeypatch.setattr(creds, "load_credentials", lambda cluster: _record(app_token="file-tok"))
    c = credentials_for("marin", _iap_auth())
    assert c.token_provider.get_token() == "file-tok"


def test_gcp_cluster_falls_back_to_ambient_access_token():
    c = credentials_for("marin", ClusterAuth(AuthProvider.GCP))
    assert isinstance(c.token_provider, GcpAccessTokenProvider)
    assert c.iap_provider is None


def test_static_cluster_uses_supplied_token():
    c = credentials_for("local", ClusterAuth(AuthProvider.STATIC), static_token="shared")
    assert c.token_provider.get_token() == "shared"


def test_iap_cluster_has_no_ambient_app_token():
    # The Iris JWT comes only from login; without a file there is no app provider.
    c = credentials_for("marin", _iap_auth())
    assert c.token_provider is None


def test_iap_edge_prefers_cached_refresh_over_service_account(monkeypatch):
    monkeypatch.setattr(creds, "load_credentials", lambda cluster: _record(edge_refresh_token="refresh"))
    c = credentials_for("marin", _iap_auth(programmatic_audiences=("aud",)))
    assert isinstance(c.iap_provider, IapRefreshTokenProvider)


def test_iap_edge_falls_back_to_service_account_for_ci():
    c = credentials_for("marin", _iap_auth(programmatic_audiences=("aud",)))
    assert isinstance(c.iap_provider, IapServiceAccountTokenProvider)


def test_none_cluster_attaches_nothing():
    c = credentials_for("local", ClusterAuth(AuthProvider.NONE))
    assert c.token_provider is None and c.iap_provider is None
    assert c.interceptors() == ()


def test_interceptors_map_providers_to_headers():
    c = ClientCredentials(token_provider=StaticTokenProvider("a"), iap_provider=StaticTokenProvider("e"))
    chain = c.interceptors()
    assert [i.header for i in chain] == ["authorization", "proxy-authorization"]
    assert all(isinstance(i, BearerTokenInjector) for i in chain)
