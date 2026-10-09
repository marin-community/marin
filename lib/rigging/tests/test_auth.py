# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import google.auth.exceptions
import google.auth.identity_pool
import google.auth.impersonated_credentials
import jwt
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.auth import (
    BearerTokenInjector,
    GcpAccessTokenProvider,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapRefreshTokenProvider,
    IapServiceAccountTokenProvider,
    RefreshingTokenProvider,
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


def test_iap_id_token_provider_mints_from_wif_external_account_adc(monkeypatch):
    """A GitHub-Actions WIF external-account ADC impersonating a SA mints here too.

    google-github-actions/auth writes an ``external_account`` (identity_pool) ADC
    with a ``service_account_impersonation_url``; fetch_id_token ignores it, so it
    reaches the impersonation path like the gcloud-impersonate ADC does.
    """
    monkeypatch.setattr("google.oauth2.id_token.fetch_id_token", _raise_no_creds)
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    sa_email = "github-iris@hai-gcp-models.iam.gserviceaccount.com"
    # A real WIF ADC as google-github-actions/auth@v3 writes it (constructed offline;
    # no token is exchanged until a refresh, which the faked IDTokenCredentials skips).
    wif_adc = google.auth.identity_pool.Credentials.from_info(
        {
            "type": "external_account",
            "audience": (
                "//iam.googleapis.com/projects/748532799086/locations/global/"
                "workloadIdentityPools/github-pool/providers/github-oidc"
            ),
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": "https://sts.googleapis.com/v1/token",
            "service_account_impersonation_url": (
                "https://iamcredentials.googleapis.com/v1/" f"projects/-/serviceAccounts/{sa_email}:generateAccessToken"
            ),
            "credential_source": {"file": "/tmp/oidc-token.txt"},
        }
    )
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (wif_adc, "proj"))

    minted: dict[str, str] = {}

    class FakeIdCreds:
        def __init__(self, target_credentials, target_audience, include_email):
            # Records what the external account resolved to: the ID token is minted
            # as the impersonated SA (signer_email) for the requested audience.
            minted["signer_email"] = target_credentials.signer_email
            minted["audience"] = target_audience
            self.token = "wif-id-token"

        def refresh(self, request):
            pass

    monkeypatch.setattr("google.auth.impersonated_credentials.IDTokenCredentials", FakeIdCreds)
    monkeypatch.setattr("google.auth.jwt.decode", lambda token, verify: {"exp": time.time() + 3600})

    provider = IapServiceAccountTokenProvider("aud-123")
    assert provider.get_token() == "wif-id-token"
    assert minted == {"signer_email": sa_email, "audience": "aud-123"}


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


@pytest.fixture
def renewal_tokens():
    def mint(expiry):
        return jwt.encode(
            {"iss": "test", "sub": "worker", "aud": "iris", "exp": expiry},
            "test-signing-key-for-auth-renewal",
            algorithm="HS256",
        )

    return mint


def test_refreshing_token_survives_bootstrap_expiry_and_restart(tmp_path, renewal_tokens):
    now = [1000.0]
    bootstrap = renewal_tokens(2000)
    renewed = renewal_tokens(4000)
    cache = tmp_path / "credentials" / "worker.jwt"
    provider = RefreshingTokenProvider(bootstrap, lambda token: renewed, cache_path=cache, now=lambda: now[0])
    assert provider.get_token() == bootstrap
    now[0] = 1750
    assert provider.get_token() == renewed
    assert cache.stat().st_mode & 0o777 == 0o600
    now[0] = 2500
    restarted = RefreshingTokenProvider(
        bootstrap, lambda token: renewal_tokens(6000), cache_path=cache, now=lambda: now[0]
    )
    assert restarted.get_token() == renewed
    now[0] = 3750
    assert restarted.get_token() == renewal_tokens(6000)


def test_refreshing_token_serializes_concurrent_renewals(renewal_tokens):
    initial = renewal_tokens(1200)
    renewed = renewal_tokens(4000)
    supplied = []

    def exchange(token):
        supplied.append(token)
        return renewed

    provider = RefreshingTokenProvider(initial, exchange, now=lambda: 1000)
    with ThreadPoolExecutor(max_workers=8) as executor:
        tokens = list(executor.map(lambda _: provider.get_token(), range(16)))
    assert tokens == [renewed] * 16
    assert supplied == [initial]


def test_refreshing_token_retries_transient_failure_without_sending_expired_token(renewal_tokens):
    now = [1000.0]
    initial = renewal_tokens(1200)

    def unavailable(token):
        raise ConnectError(Code.UNAVAILABLE, "issuer offline")

    provider = RefreshingTokenProvider(initial, unavailable, now=lambda: now[0])
    assert provider.get_token() == initial
    now[0] = 1201
    with pytest.raises(ConnectError) as error:
        provider.get_token()
    assert error.value.code == Code.UNAVAILABLE


def test_refreshing_token_does_not_hide_auth_rejection(renewal_tokens):
    def rejected(token):
        raise ConnectError(Code.UNAUTHENTICATED, "invalid credential")

    provider = RefreshingTokenProvider(renewal_tokens(1200), rejected, now=lambda: 1000)
    with pytest.raises(ConnectError) as error:
        provider.get_token()
    assert error.value.code == Code.UNAUTHENTICATED


def test_refreshing_token_renews_without_application_requests(renewal_tokens):
    stop = threading.Event()
    renewed = renewal_tokens(4000)

    def exchange(token):
        stop.set()
        return renewed

    provider = RefreshingTokenProvider(renewal_tokens(1200), exchange, now=lambda: 1000)
    provider.run(stop)
    assert provider.get_token() == renewed
