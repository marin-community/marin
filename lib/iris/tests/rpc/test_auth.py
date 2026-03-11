# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.auth import (
    AuthInterceptor,
    AuthTokenInjector,
    StaticTokenProvider,
    StaticTokenVerifier,
    get_verified_user,
)


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


class FakeHeaders(dict):
    """Minimal headers container matching Connect RPC Headers behavior."""

    def get(self, key, default=None):
        val = super().get(key)
        if val is None:
            return default
        if isinstance(val, list):
            return val
        return [val]


def _make_ctx(headers: dict | None = None):
    """Create a fake RequestContext with optional headers."""

    class FakeCtx:
        def __init__(self):
            self._request_headers = FakeHeaders(headers or {})

        def method(self):
            return FakeMethodInfo(name="TestMethod")

        def request_headers(self):
            return self._request_headers

    return FakeCtx()


@pytest.fixture
def verifier():
    return StaticTokenVerifier({"valid-token-alice": "alice", "valid-token-bob": "bob"})


@pytest.fixture
def interceptor(verifier):
    return AuthInterceptor(verifier)


def test_auth_interceptor_passes_valid_token(interceptor):
    ctx = _make_ctx({"authorization": "Bearer valid-token-alice"})
    captured_user = []

    def handler(req, ctx):
        captured_user.append(get_verified_user())
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "request", ctx)
    assert result == "ok"
    assert captured_user == ["alice"]


def test_auth_interceptor_rejects_missing_header(interceptor):
    ctx = _make_ctx({})
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(lambda r, c: "ok", "request", ctx)
    assert exc_info.value.code == Code.UNAUTHENTICATED


def test_auth_interceptor_rejects_invalid_token(interceptor):
    ctx = _make_ctx({"authorization": "Bearer wrong-token"})
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(lambda r, c: "ok", "request", ctx)
    assert exc_info.value.code == Code.UNAUTHENTICATED


def test_auth_interceptor_rejects_malformed_header(interceptor):
    ctx = _make_ctx({"authorization": "Basic user:pass"})
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(lambda r, c: "ok", "request", ctx)
    assert exc_info.value.code == Code.UNAUTHENTICATED


def test_auth_interceptor_cleans_up_context_on_handler_error(interceptor):
    ctx = _make_ctx({"authorization": "Bearer valid-token-alice"})

    def failing_handler(req, ctx):
        assert get_verified_user() == "alice"
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    # Verified user should be cleaned up after handler exits
    assert get_verified_user() is None


def test_verified_user_is_none_without_interceptor():
    assert get_verified_user() is None


def test_static_token_verifier_valid():
    v = StaticTokenVerifier({"tok": "user1"})
    assert v.verify("tok") == "user1"


def test_static_token_verifier_invalid():
    v = StaticTokenVerifier({"tok": "user1"})
    with pytest.raises(ValueError, match="Invalid token"):
        v.verify("bad")


def test_token_injector_adds_auth_header():
    provider = StaticTokenProvider("my-token")
    injector = AuthTokenInjector(provider)
    ctx = _make_ctx({})
    captured_headers = []

    def handler(req, ctx):
        captured_headers.append(dict(ctx.request_headers()))
        return "ok"

    result = injector.intercept_unary_sync(handler, "request", ctx)
    assert result == "ok"
    assert captured_headers[0]["authorization"] == "Bearer my-token"


def test_different_users_get_different_identities(interceptor):
    users = []

    def capture_handler(req, ctx):
        users.append(get_verified_user())
        return "ok"

    ctx_alice = _make_ctx({"authorization": "Bearer valid-token-alice"})
    interceptor.intercept_unary_sync(capture_handler, "request", ctx_alice)

    ctx_bob = _make_ctx({"authorization": "Bearer valid-token-bob"})
    interceptor.intercept_unary_sync(capture_handler, "request", ctx_bob)

    assert users == ["alice", "bob"]
