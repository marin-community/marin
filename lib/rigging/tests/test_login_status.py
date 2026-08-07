# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of ``resolve_login_status`` — the mint-on-demand / offline-fallback dump."""

import base64
import json
import time

import pytest
from rigging import login_status as ls
from rigging.auth import IapCredentialsUnavailable, IapLoginRequired
from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.credential_store import CredentialRecord
from rigging.login_status import EMAIL_METADATA_KEY, LoginMethod, LoginState, email_from_id_token, resolve_login_status


def _id_token(claims: dict) -> str:
    """A JWT whose payload is ``claims`` (unsigned; decodable with verify=False)."""

    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


class _FakeProvider:
    """A TokenProvider that returns a fixed token or raises a fixed error on mint."""

    def __init__(self, *, token: str | None = None, error: Exception | None = None):
        self._token = token
        self._error = error

    def get_token(self) -> str | None:
        if self._error is not None:
            raise self._error
        return self._token


@pytest.fixture
def store(monkeypatch):
    """Control the cached record and minted token seen by ``resolve_login_status``."""

    state: dict = {"record": None, "provider": None}
    monkeypatch.setattr(ls, "load_credentials", lambda cluster: state["record"])
    monkeypatch.setattr(ls, "edge_provider", lambda cluster, auth: state["provider"])
    return state


def _iap() -> ClusterAuth:
    return ClusterAuth(AuthProvider.IAP, iap=IapAuth(url="https://iris.example"))


def _record(**kw) -> CredentialRecord:
    base = dict(cluster="marin", endpoint="https://iris.example")
    base.update(kw)
    return CredentialRecord(**base)


def test_non_iap_cluster_needs_no_login(store):
    status = resolve_login_status("local", ClusterAuth(AuthProvider.NONE))
    assert status.state is LoginState.NO_LOGIN_REQUIRED
    assert status.method is LoginMethod.NETWORK_TRUST
    assert status.email is None and status.token_expiry is None


def test_browser_login_active_reports_live_email_and_expiry(store):
    exp = int(time.time()) + 3600
    store["record"] = _record(edge_refresh_token="rt", metadata={EMAIL_METADATA_KEY: "stale@marin.io"})
    store["provider"] = _FakeProvider(token=_id_token({"email": "alice@marin.io", "exp": exp}))

    status = resolve_login_status("marin", _iap())

    assert status.state is LoginState.ACTIVE
    assert status.method is LoginMethod.IAP_BROWSER
    # The freshly minted token wins over the value cached at login.
    assert status.email == "alice@marin.io"
    assert status.token_expiry is not None
    assert int(status.token_expiry.timestamp()) == exp


def test_browser_login_offline_falls_back_to_cached_email(store):
    store["record"] = _record(edge_refresh_token="rt", metadata={EMAIL_METADATA_KEY: "alice@marin.io"})
    store["provider"] = _FakeProvider(error=IapLoginRequired("cached IAP credentials are no longer valid"))

    status = resolve_login_status("marin", _iap())

    assert status.state is LoginState.OFFLINE
    assert status.method is LoginMethod.IAP_BROWSER
    assert status.email == "alice@marin.io"  # from the last login, not a live token
    assert status.token_expiry is None
    assert "no longer valid" in status.detail  # the mint failure travels to the user


def test_service_account_active_when_no_cached_login(store):
    exp = int(time.time()) + 1800
    store["record"] = None
    store["provider"] = _FakeProvider(token=_id_token({"email": "sa@proj.iam.gserviceaccount.com", "exp": exp}))

    status = resolve_login_status("marin", _iap())

    assert status.state is LoginState.ACTIVE
    assert status.method is LoginMethod.IAP_SERVICE_ACCOUNT
    assert status.email == "sa@proj.iam.gserviceaccount.com"


def test_not_logged_in_when_no_record_and_mint_fails(store):
    store["record"] = None
    store["provider"] = _FakeProvider(error=IapCredentialsUnavailable("no credentials"))

    status = resolve_login_status("marin", _iap())

    assert status.state is LoginState.NOT_LOGGED_IN
    assert status.method is None
    assert status.email is None
    assert "no credentials" in status.detail


def test_mint_failure_is_caught_for_any_reason(store):
    # The dump must survive an unexpected failure from the mint path, not just the
    # known auth errors — the offline fallback is intentionally broad.
    store["record"] = _record(edge_refresh_token="rt")
    store["provider"] = _FakeProvider(error=RuntimeError("connection reset"))

    status = resolve_login_status("marin", _iap())

    assert status.state is LoginState.OFFLINE
    assert "connection reset" in status.detail


def test_email_from_id_token_extracts_email():
    assert email_from_id_token(_id_token({"email": "dev@marin.io", "sub": "1"})) == "dev@marin.io"


def test_email_from_id_token_absent_or_undecodable():
    assert email_from_id_token(_id_token({"sub": "no-email-claim"})) is None
    assert email_from_id_token("not-a-jwt") is None
