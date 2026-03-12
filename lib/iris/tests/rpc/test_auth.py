# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from connectrpc._headers import Headers
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.auth import (
    AuthInterceptor,
    AuthTokenInjector,
    CompositeTokenVerifier,
    StaticTokenProvider,
    StaticTokenVerifier,
    get_verified_user,
)


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


def _make_ctx(headers: dict | None = None):
    """Create a fake RequestContext with optional headers."""

    class FakeCtx:
        def __init__(self):
            self._request_headers = Headers(headers or {})

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


def test_gcp_token_provider_uses_id_token():
    """Verify GcpTokenProvider calls fetch_id_token, not credentials.token."""
    from unittest.mock import patch

    from iris.rpc.auth import GcpTokenProvider

    provider = GcpTokenProvider(audience="https://my-audience")
    with patch("google.oauth2.id_token.fetch_id_token", return_value="fake-id-token") as mock_fetch:
        token = provider.get_token()
    assert token == "fake-id-token"
    mock_fetch.assert_called_once()
    assert mock_fetch.call_args[0][1] == "https://my-audience"


def test_gcp_token_provider_propagates_errors():
    """Verify GcpTokenProvider does NOT swallow exceptions."""
    from unittest.mock import patch

    from iris.rpc.auth import GcpTokenProvider

    provider = GcpTokenProvider(audience="https://my-audience")
    with patch("google.oauth2.id_token.fetch_id_token", side_effect=Exception("metadata server unreachable")):
        with pytest.raises(Exception, match="metadata server unreachable"):
            provider.get_token()


def test_cli_gcp_token_provider_sdk_success():
    """When the SDK returns a token, gcloud CLI is never called."""
    from unittest.mock import patch

    from iris.rpc.auth import CliGcpTokenProvider

    provider = CliGcpTokenProvider(audience="https://my-audience")
    with (
        patch("google.oauth2.id_token.fetch_id_token", return_value="sdk-token") as mock_sdk,
        patch("subprocess.run") as mock_subprocess,
    ):
        token = provider.get_token()

    assert token == "sdk-token"
    mock_sdk.assert_called_once()
    mock_subprocess.assert_not_called()


def test_cli_gcp_token_provider_falls_back_to_gcloud():
    """When SDK raises DefaultCredentialsError, falls back to gcloud CLI."""
    from subprocess import CompletedProcess
    from unittest.mock import patch

    from google.auth.exceptions import DefaultCredentialsError

    from iris.rpc.auth import CliGcpTokenProvider

    provider = CliGcpTokenProvider(audience="https://my-audience")
    with (
        patch("google.oauth2.id_token.fetch_id_token", side_effect=DefaultCredentialsError("no creds")),
        patch(
            "subprocess.run",
            return_value=CompletedProcess(args=[], returncode=0, stdout="gcloud-token\n", stderr=""),
        ) as mock_subprocess,
    ):
        token = provider.get_token()

    assert token == "gcloud-token"
    mock_subprocess.assert_called_once()


def test_cli_gcp_token_provider_all_fail():
    """When both SDK and gcloud fail, raises RuntimeError."""
    from subprocess import CompletedProcess
    from unittest.mock import patch

    from google.auth.exceptions import DefaultCredentialsError

    from iris.rpc.auth import CliGcpTokenProvider

    provider = CliGcpTokenProvider(audience="https://my-audience")
    with (
        patch("google.oauth2.id_token.fetch_id_token", side_effect=DefaultCredentialsError("no creds")),
        patch(
            "subprocess.run",
            return_value=CompletedProcess(args=[], returncode=1, stdout="", stderr="gcloud error"),
        ),
    ):
        with pytest.raises(RuntimeError, match="gcloud auth failed"):
            provider.get_token()


def test_composite_verifier_first_match():
    v1 = StaticTokenVerifier({"tok-a": "alice"})
    v2 = StaticTokenVerifier({"tok-b": "bob"})
    composite = CompositeTokenVerifier([v1, v2])
    assert composite.verify("tok-a") == "alice"


def test_composite_verifier_second_match():
    v1 = StaticTokenVerifier({"tok-a": "alice"})
    v2 = StaticTokenVerifier({"tok-b": "bob"})
    composite = CompositeTokenVerifier([v1, v2])
    assert composite.verify("tok-b") == "bob"


def test_composite_verifier_rejects_empty_list():
    with pytest.raises(ValueError, match="requires at least one verifier"):
        CompositeTokenVerifier([])


def test_composite_verifier_all_fail():
    v1 = StaticTokenVerifier({"tok-a": "alice"})
    v2 = StaticTokenVerifier({"tok-b": "bob"})
    composite = CompositeTokenVerifier([v1, v2])
    with pytest.raises(ValueError, match="All verifiers failed"):
        composite.verify("unknown-token")


# ---------------------------------------------------------------------------
# NullAuthInterceptor
# ---------------------------------------------------------------------------


def test_null_auth_interceptor_sets_anonymous_user():
    from iris.rpc.auth import NullAuthInterceptor

    interceptor = NullAuthInterceptor()
    captured_user = []

    def handler(req, ctx):
        captured_user.append(get_verified_user())
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert result == "ok"
    assert captured_user == ["anonymous"]


def test_null_auth_interceptor_custom_user():
    from iris.rpc.auth import NullAuthInterceptor

    interceptor = NullAuthInterceptor(user="custom-user")
    captured_user = []

    def handler(req, ctx):
        captured_user.append(get_verified_user())
        return "ok"

    interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert captured_user == ["custom-user"]


def test_null_auth_interceptor_resets_context():
    from iris.rpc.auth import NullAuthInterceptor

    interceptor = NullAuthInterceptor()

    def handler(req, ctx):
        assert get_verified_user() == "anonymous"
        return "ok"

    interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert get_verified_user() is None


def test_null_auth_interceptor_resets_context_on_error():
    from iris.rpc.auth import NullAuthInterceptor

    interceptor = NullAuthInterceptor()

    def failing_handler(req, ctx):
        assert get_verified_user() == "anonymous"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        interceptor.intercept_unary_sync(failing_handler, "request", _make_ctx())

    assert get_verified_user() is None


# ---------------------------------------------------------------------------
# CLI token provider factory
# ---------------------------------------------------------------------------


def test_create_client_token_provider_gcp():
    from iris.cli.main import create_client_token_provider
    from iris.rpc.auth import CliGcpTokenProvider
    from iris.rpc.config_pb2 import AuthConfig

    config = AuthConfig(gcp={"audience": "https://my-controller"})
    provider = create_client_token_provider(config)
    assert isinstance(provider, CliGcpTokenProvider)


def test_create_client_token_provider_none_when_no_provider():
    from iris.cli.main import create_client_token_provider
    from iris.rpc.config_pb2 import AuthConfig

    config = AuthConfig()
    assert create_client_token_provider(config) is None


# ---------------------------------------------------------------------------
# hash_token utility
# ---------------------------------------------------------------------------


def test_hash_token_deterministic():
    from iris.rpc.auth import hash_token

    assert hash_token("test-token") == hash_token("test-token")
    assert hash_token("a") != hash_token("b")


def test_hash_token_is_sha256_hex():
    from iris.rpc.auth import hash_token

    result = hash_token("test")
    assert len(result) == 64  # SHA-256 hex digest
    assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# DbTokenVerifier
# ---------------------------------------------------------------------------


def test_db_token_verifier_valid_key(tmp_path):
    from iris.cluster.controller.auth_setup import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "my-secret-token"
    db.create_api_key(
        key_id="k1",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id="alice",
        name="test-key",
        now=now,
    )
    verifier = DbTokenVerifier(db)
    assert verifier.verify(raw_token) == "alice"
    db.close()


def test_db_token_verifier_invalid_key(tmp_path):
    from iris.cluster.controller.auth_setup import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    verifier = DbTokenVerifier(db)
    with pytest.raises(ValueError, match="Invalid API key"):
        verifier.verify("nonexistent-token")
    db.close()


def test_db_token_verifier_revoked_key(tmp_path):
    from iris.cluster.controller.auth_setup import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "revoked-token"
    db.create_api_key(
        key_id="k1",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id="alice",
        name="test-key",
        now=now,
    )
    db.revoke_api_key("k1", now)

    verifier = DbTokenVerifier(db)
    with pytest.raises(ValueError, match="revoked"):
        verifier.verify(raw_token)
    db.close()


def test_db_token_verifier_expired_key(tmp_path):
    from iris.cluster.controller.auth_setup import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "expired-token"
    past = Timestamp.from_ms(1000)
    db.create_api_key(
        key_id="k1",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id="alice",
        name="test-key",
        now=past,
        expires_at=past,
    )

    verifier = DbTokenVerifier(db)
    with pytest.raises(ValueError, match="expired"):
        verifier.verify(raw_token)
    db.close()


def test_db_token_verifier_last_used_throttle(tmp_path):
    from iris.cluster.controller.auth_setup import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "throttle-test-token"
    db.create_api_key(
        key_id="k1",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id="alice",
        name="test-key",
        now=now,
    )

    verifier = DbTokenVerifier(db)

    # First call should set last_used_at
    verifier.verify(raw_token)
    key = db.lookup_api_key_by_hash(hash_token(raw_token))
    assert key is not None
    assert key.last_used_at is not None
    first_used = key.last_used_at.epoch_ms()

    # Immediate second call should NOT update (throttled)
    verifier.verify(raw_token)
    key2 = db.lookup_api_key_by_hash(hash_token(raw_token))
    assert key2.last_used_at.epoch_ms() == first_used

    db.close()
