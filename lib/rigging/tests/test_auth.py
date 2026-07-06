# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import google.auth.exceptions
import google.auth.impersonated_credentials
import pytest
from rigging.auth import (
    BearerTokenInjector,
    GcpAccessTokenProvider,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapRefreshTokenProvider,
    IapServiceAccountTokenProvider,
    read_desktop_client,
)


class FakeProvider:
    def __init__(self, token):
        self._token = token

    def get_token(self):
        return self._token


class FakeCtx:
    """Connect ctx whose request_headers() returns a mutable dict."""

    def __init__(self):
        self._headers: dict[str, str] = {}

    def request_headers(self) -> dict[str, str]:
        return self._headers


def test_injector_sets_its_header_sync():
    ctx = FakeCtx()
    BearerTokenInjector(FakeProvider("tok"), "authorization").on_start_sync(ctx)
    assert ctx.request_headers()["authorization"] == "Bearer tok"


def test_injector_skips_header_when_no_token_sync():
    ctx = FakeCtx()
    BearerTokenInjector(FakeProvider(None), "authorization").on_start_sync(ctx)
    assert "authorization" not in ctx.request_headers()


@pytest.mark.asyncio
async def test_injector_sets_its_header_async():
    ctx = FakeCtx()
    await BearerTokenInjector(FakeProvider("tok"), "authorization").on_start(ctx)
    assert ctx.request_headers()["authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_injector_skips_header_when_no_token_async():
    ctx = FakeCtx()
    await BearerTokenInjector(FakeProvider(None), "authorization").on_start(ctx)
    assert "authorization" not in ctx.request_headers()


def test_injector_uses_the_chosen_header():
    ctx = FakeCtx()
    BearerTokenInjector(FakeProvider("edge"), "proxy-authorization").on_start_sync(ctx)
    assert ctx.request_headers()["proxy-authorization"] == "Bearer edge"
    assert "authorization" not in ctx.request_headers()


class FakeCreds:
    """Stand-in for a google-auth Credentials whose refresh sets token/expiry."""

    def __init__(self, token: str, expiry):
        self._token = token
        self.token = None
        self.expiry = expiry
        self.refresh_calls = 0

    def refresh(self, request):
        self.refresh_calls += 1
        self.token = self._token


def test_gcp_access_token_provider_caches_until_expiry(monkeypatch):
    # Token valid well beyond the 5-minute refresh margin.
    creds = FakeCreds("access-tok", datetime.now(UTC) + timedelta(hours=1))
    monkeypatch.setattr("google.auth.default", lambda: (creds, "proj"))
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    provider = GcpAccessTokenProvider()

    assert provider.get_token() == "access-tok"
    assert creds.refresh_calls == 1

    # Second call within the validity window must not re-fetch.
    assert provider.get_token() == "access-tok"
    assert creds.refresh_calls == 1


def test_gcp_access_token_provider_refetches_after_expiry(monkeypatch):
    # Expiry already inside the refresh margin -> cache window is in the past.
    creds = FakeCreds("access-tok", datetime.now(UTC) + timedelta(seconds=60))
    monkeypatch.setattr("google.auth.default", lambda: (creds, "proj"))
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    provider = GcpAccessTokenProvider()

    assert provider.get_token() == "access-tok"
    assert creds.refresh_calls == 1
    # Expiry (60s) is inside the 300s margin, so the cache is immediately stale.
    assert provider.get_token() == "access-tok"
    assert creds.refresh_calls == 2


def test_iap_id_token_provider_caches_until_expiry(monkeypatch):
    fetch_calls = 0

    def fake_fetch(request, audience):
        nonlocal fetch_calls
        fetch_calls += 1
        assert audience == "aud-123"
        return "id-token"

    # exp far enough out that the cache window is in the future.
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", fake_fetch)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.auth.jwt.decode", lambda token, verify: {"exp": time.time() + 3600})

    provider = IapServiceAccountTokenProvider("aud-123")

    assert provider.get_token() == "id-token"
    assert fetch_calls == 1
    assert provider.get_token() == "id-token"
    assert fetch_calls == 1


def test_iap_id_token_provider_refetches_after_expiry(monkeypatch):
    fetch_calls = 0

    def fake_fetch(request, audience):
        nonlocal fetch_calls
        fetch_calls += 1
        return "id-token"

    # exp inside the 300s refresh margin -> cache immediately stale.
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", fake_fetch)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.auth.jwt.decode", lambda token, verify: {"exp": time.time() + 60})

    provider = IapServiceAccountTokenProvider("aud-123")

    assert provider.get_token() == "id-token"
    assert fetch_calls == 1
    assert provider.get_token() == "id-token"
    assert fetch_calls == 2


def _raise_no_creds(*args, **kwargs):
    raise google.auth.exceptions.DefaultCredentialsError("no ADC")


def test_iap_id_token_provider_raises_actionable_error_without_credentials(monkeypatch):
    """With neither ambient SA creds nor an impersonated ADC, get_token is actionable."""
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", _raise_no_creds)
    monkeypatch.setattr("google.auth.default", _raise_no_creds)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    provider = IapServiceAccountTokenProvider("aud-123")
    with pytest.raises(IapCredentialsUnavailable, match="impersonate"):
        provider.get_token()


def test_iap_id_token_provider_mints_from_impersonated_adc(monkeypatch):
    """fetch_id_token ignores the well-known ADC, so an impersonated ADC mints here."""
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", _raise_no_creds)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    source = MagicMock(spec=google.auth.impersonated_credentials.Credentials)
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (source, "proj"))

    class FakeIdCreds:
        def __init__(self, target_credentials, target_audience, include_email):
            self.token = "imp-id-token"

        def refresh(self, request):
            pass

    monkeypatch.setattr("google.auth.impersonated_credentials.IDTokenCredentials", FakeIdCreds)
    monkeypatch.setattr("google.auth.jwt.decode", lambda token, verify: {"exp": time.time() + 3600})

    provider = IapServiceAccountTokenProvider("aud-123")
    assert provider.get_token() == "imp-id-token"


def test_iap_id_token_provider_rejects_non_impersonated_adc(monkeypatch):
    """Bare user ADC (not impersonated) cannot mint an IAP token — hard-fail."""
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", _raise_no_creds)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (object(), "proj"))

    provider = IapServiceAccountTokenProvider("aud-123")
    with pytest.raises(IapCredentialsUnavailable):
        provider.get_token()


class FakeRefreshCreds:
    """Stand-in for google.oauth2.credentials.Credentials in the desktop flow."""

    def __init__(self):
        self.id_token = None
        self.valid = False
        self.refresh_calls = 0

    def refresh(self, request):
        self.refresh_calls += 1
        self.id_token = "refreshed-id-token"
        self.valid = True


def test_iap_refresh_token_provider_remints_then_caches(monkeypatch):
    creds = FakeRefreshCreds()
    monkeypatch.setattr("google.oauth2.credentials.Credentials", lambda **kwargs: creds)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    provider = IapRefreshTokenProvider("client-id", "secret", "refresh-tok")

    # First call refreshes (no cached id_token); the second reuses it.
    assert provider.get_token() == "refreshed-id-token"
    assert creds.refresh_calls == 1
    assert provider.get_token() == "refreshed-id-token"
    assert creds.refresh_calls == 1

    # When the access token expires (valid flips false), it re-mints.
    creds.valid = False
    assert provider.get_token() == "refreshed-id-token"
    assert creds.refresh_calls == 2


def test_iap_refresh_token_provider_raises_login_required_when_refresh_fails(monkeypatch):
    class FailingCreds:
        id_token = None
        valid = False

        def refresh(self, request):
            # An expired/revoked refresh token surfaces here as RefreshError.
            raise google.auth.exceptions.RefreshError("invalid_grant")

    monkeypatch.setattr("google.oauth2.credentials.Credentials", lambda **kwargs: FailingCreds())
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    provider = IapRefreshTokenProvider(
        "client-id",
        "secret",
        "refresh-tok",
        login_hint="log in to cluster marin to authenticate",
    )

    # The raw google-auth error becomes an actionable, self-contained IapLoginRequired.
    with pytest.raises(IapLoginRequired, match="log in to cluster marin"):
        provider.get_token()


def test_read_desktop_client_rejects_web_client_secret(tmp_path):
    secret_file = tmp_path / "web_secret.json"
    secret_file.write_text(json.dumps({"web": {"client_id": "cid", "client_secret": "secret"}}))

    with pytest.raises(ValueError, match="desktop"):
        read_desktop_client(str(secret_file))
