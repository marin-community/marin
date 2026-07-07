# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolution-order behavior for ``credentials_for`` and the interceptor chain."""

import pytest
from rigging import credentials as creds
from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    BearerTokenInjector,
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


def test_env_override_sets_token_provider(monkeypatch):
    # $MARIN_CLUSTER_TOKEN is the sole app-token source: an explicit bearer for
    # CI / headless runs. The controller mints no user token otherwise.
    monkeypatch.setenv(creds.MARIN_CLUSTER_TOKEN_ENV, "env-tok")
    c = credentials_for("marin", _iap_auth())
    assert isinstance(c.token_provider, StaticTokenProvider)
    assert c.token_provider.get_token() == "env-tok"


def test_iap_cluster_has_no_ambient_authorization_bearer():
    # Pure-IAP: the controller mints no user token, so with no env override there is
    # no Authorization bearer — auth rides on the IAP edge (Proxy-Authorization).
    c = credentials_for("marin", _iap_auth())
    assert c.token_provider is None


def test_iap_edge_prefers_cached_refresh_over_service_account(monkeypatch):
    monkeypatch.setattr(creds, "load_credentials", lambda cluster: _record(edge_refresh_token="refresh"))
    c = credentials_for("marin", _iap_auth(programmatic_audiences=("aud",)))
    assert isinstance(c.iap_provider, IapRefreshTokenProvider)


def test_iap_edge_falls_back_to_service_account_for_ci():
    c = credentials_for("marin", _iap_auth(programmatic_audiences=("aud",)))
    assert isinstance(c.iap_provider, IapServiceAccountTokenProvider)
    assert c.iap_provider._audience == "aud"


def test_iap_edge_service_account_falls_back_to_desktop_client_when_no_programmatic_audience():
    # With no programmatic audience configured, the edge path still mints a token
    # using the desktop client id rather than attaching none (which IAP rejects with
    # 401). Regression guard for the cross-lane CI auth outage (#6936 and siblings).
    c = credentials_for("marin", _iap_auth())
    assert isinstance(c.iap_provider, IapServiceAccountTokenProvider)
    assert c.iap_provider._audience == MARIN_DESKTOP_OAUTH_CLIENT.client_id


def test_iap_edge_desktop_fallback_honors_configured_desktop_client():
    # A cluster overriding the desktop OAuth client falls back to its id, not the
    # Marin default.
    c = credentials_for(
        "marin",
        _iap_auth(desktop_oauth_client_id="custom-desktop.apps.googleusercontent.com", desktop_oauth_client_secret="s"),
    )
    assert c.iap_provider._audience == "custom-desktop.apps.googleusercontent.com"


def test_none_cluster_attaches_nothing():
    c = credentials_for("local", ClusterAuth(AuthProvider.NONE))
    assert c.token_provider is None and c.iap_provider is None
    assert c.interceptors() == ()


def test_interceptors_map_providers_to_headers():
    c = ClientCredentials(token_provider=StaticTokenProvider("a"), iap_provider=StaticTokenProvider("e"))
    chain = c.interceptors()
    assert [i.header for i in chain] == ["authorization", "proxy-authorization"]
    assert all(isinstance(i, BearerTokenInjector) for i in chain)
