# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import UTC, datetime, timedelta

import pytest
from connectrpc._interceptor_async import MetadataInterceptor
from connectrpc._interceptor_sync import MetadataInterceptorSync
from rigging.auth import (
    BearerTokenInjector,
    GcpAccessTokenProvider,
    IapUserIdTokenProvider,
    StaticTokenProvider,
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


def test_static_token_provider_returns_token():
    assert StaticTokenProvider("abc").get_token() == "abc"


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


def test_injector_is_a_metadata_interceptor():
    """connectrpc must classify the injector as a *metadata* interceptor so the
    header is applied to every RPC shape (unary and streaming), not just unary.

    Reaches into connectrpc's interceptor module to assert the contract the
    public client wiring depends on; if the protocol moves, this fails loudly.
    """
    injector = BearerTokenInjector(FakeProvider("tok"), "authorization")
    assert isinstance(injector, MetadataInterceptorSync)
    assert isinstance(injector, MetadataInterceptor)


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

    provider = IapUserIdTokenProvider("aud-123")

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

    provider = IapUserIdTokenProvider("aud-123")

    assert provider.get_token() == "id-token"
    assert fetch_calls == 1
    assert provider.get_token() == "id-token"
    assert fetch_calls == 2
