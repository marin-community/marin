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
    _extract_cookie,
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


def test_auth_interceptor_accepts_session_cookie(interceptor):
    ctx = _make_ctx({"cookie": "iris_session=valid-token-bob"})
    captured_user = []

    def handler(req, ctx):
        captured_user.append(get_verified_user())
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "request", ctx)
    assert result == "ok"
    assert captured_user == ["bob"]


def test_auth_interceptor_prefers_bearer_over_cookie(interceptor):
    ctx = _make_ctx(
        {
            "authorization": "Bearer valid-token-alice",
            "cookie": "iris_session=valid-token-bob",
        }
    )
    captured_user = []

    def handler(req, ctx):
        captured_user.append(get_verified_user())
        return "ok"

    interceptor.intercept_unary_sync(handler, "request", ctx)
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
    assert exc_info.value.message == "Authentication failed"
    assert "Invalid token" not in exc_info.value.message


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


def test_gcp_access_token_verifier_valid():
    """GcpAccessTokenVerifier extracts email from tokeninfo."""
    from unittest.mock import Mock, patch

    from iris.rpc.auth import GcpAccessTokenVerifier

    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"email": "alice@example.com"}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        email = verifier.verify("fake-access-token")

    assert email == "alice@example.com"
    mock_get.assert_called_once_with(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"access_token": "fake-access-token"},
        timeout=10,
    )


def test_gcp_access_token_verifier_invalid_token():
    from unittest.mock import Mock, patch

    from iris.rpc.auth import GcpAccessTokenVerifier

    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 401

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="Token verification failed"):
            verifier.verify("bad-token")


def test_gcp_access_token_verifier_no_email():
    from unittest.mock import Mock, patch

    from iris.rpc.auth import GcpAccessTokenVerifier

    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"scope": "openid"}

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="email"):
            verifier.verify("token-without-email")


def test_gcp_access_token_verifier_checks_project_access():
    from unittest.mock import Mock, patch

    from iris.rpc.auth import GcpAccessTokenVerifier

    verifier = GcpAccessTokenVerifier(project_id="my-project")

    tokeninfo_resp = Mock()
    tokeninfo_resp.status_code = 200
    tokeninfo_resp.json.return_value = {"email": "alice@example.com"}

    project_resp = Mock()
    project_resp.status_code = 200

    with patch("requests.get", side_effect=[tokeninfo_resp, project_resp]) as mock_get:
        email = verifier.verify("valid-token")

    assert email == "alice@example.com"
    assert mock_get.call_count == 2
    mock_get.assert_any_call(
        "https://cloudresourcemanager.googleapis.com/v3/projects/my-project",
        headers={"Authorization": "Bearer valid-token"},
        timeout=10,
    )


def test_gcp_access_token_verifier_rejects_no_project_access():
    from unittest.mock import Mock, patch

    from iris.rpc.auth import GcpAccessTokenVerifier

    verifier = GcpAccessTokenVerifier(project_id="restricted-project")

    tokeninfo_resp = Mock()
    tokeninfo_resp.status_code = 200
    tokeninfo_resp.json.return_value = {"email": "alice@example.com"}

    project_resp = Mock()
    project_resp.status_code = 403

    with patch("requests.get", side_effect=[tokeninfo_resp, project_resp]):
        with pytest.raises(ValueError, match="does not have access"):
            verifier.verify("valid-token")


def test_gcp_access_token_provider_refreshes_credentials():
    from unittest.mock import MagicMock, patch

    from iris.rpc.auth import GcpAccessTokenProvider

    mock_creds = MagicMock()
    mock_creds.token = "fresh-access-token"
    mock_creds.expiry = None

    provider = GcpAccessTokenProvider()
    with patch("google.auth.default", return_value=(mock_creds, "project-id")):
        token = provider.get_token()

    assert token == "fresh-access-token"
    mock_creds.refresh.assert_called_once()


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


def test_create_client_token_provider_gcp(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    from iris.cli.main import create_client_token_provider
    from iris.rpc.config_pb2 import AuthConfig

    # Isolate from real token store
    monkeypatch.setattr("iris.cli.main.load_token", lambda *a, **kw: None)
    monkeypatch.setattr("iris.cli.main.load_any_token", lambda *a, **kw: None)

    config = AuthConfig(gcp={"project_id": "my-project"})
    provider = create_client_token_provider(config)

    # Verify it actually produces a GCP token when called
    mock_creds = MagicMock()
    mock_creds.token = "gcp-access-token"
    mock_creds.expiry = None
    with patch("google.auth.default", return_value=(mock_creds, "my-project")):
        assert provider.get_token() == "gcp-access-token"


def test_create_client_token_provider_uses_stored_token(tmp_path, monkeypatch):
    from iris.cli.main import create_client_token_provider
    from iris.cli.token_store import ClusterCredential
    from iris.rpc.config_pb2 import AuthConfig

    monkeypatch.setattr(
        "iris.cli.main.load_token",
        lambda name, **kw: ClusterCredential(url="http://x", token="stored-tok") if name == "mycluster" else None,
    )
    monkeypatch.setattr("iris.cli.main.load_any_token", lambda **kw: None)

    config = AuthConfig(gcp={"project_id": "my-project"})
    provider = create_client_token_provider(config, cluster_name="mycluster")
    assert provider.get_token() == "stored-tok"


def test_create_client_token_provider_none_when_no_provider(monkeypatch):
    from iris.cli.main import create_client_token_provider
    from iris.rpc.config_pb2 import AuthConfig

    monkeypatch.setattr("iris.cli.main.load_token", lambda *a, **kw: None)
    monkeypatch.setattr("iris.cli.main.load_any_token", lambda *a, **kw: None)

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
    from iris.cluster.controller.auth import DbTokenVerifier, create_api_key
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "my-secret-token"
    create_api_key(
        db,
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
    from iris.cluster.controller.auth import DbTokenVerifier
    from iris.cluster.controller.db import ControllerDB

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    verifier = DbTokenVerifier(db)
    with pytest.raises(ValueError, match="Invalid API key"):
        verifier.verify("nonexistent-token")
    db.close()


def test_db_token_verifier_revoked_key(tmp_path):
    from iris.cluster.controller.auth import DbTokenVerifier, create_api_key, revoke_api_key
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "revoked-token"
    create_api_key(
        db,
        key_id="k1",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id="alice",
        name="test-key",
        now=now,
    )
    revoke_api_key(db, "k1", now)

    verifier = DbTokenVerifier(db)
    with pytest.raises(ValueError, match="revoked"):
        verifier.verify(raw_token)
    db.close()


def test_db_token_verifier_expired_key(tmp_path):
    from iris.cluster.controller.auth import DbTokenVerifier, create_api_key
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "expired-token"
    past = Timestamp.from_ms(1000)
    create_api_key(
        db,
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
    from iris.cluster.controller.auth import DbTokenVerifier, create_api_key, lookup_api_key_by_hash
    from iris.cluster.controller.db import ControllerDB
    from iris.rpc.auth import hash_token
    from iris.time_utils import Timestamp

    db = ControllerDB(db_path=tmp_path / "test.sqlite3")
    now = Timestamp.now()
    db.ensure_user("alice", now)
    raw_token = "throttle-test-token"
    create_api_key(
        db,
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
    key = lookup_api_key_by_hash(db, hash_token(raw_token))
    assert key is not None
    assert key.last_used_at is not None
    first_used = key.last_used_at.epoch_ms()

    # Immediate second call should NOT update (throttled)
    verifier.verify(raw_token)
    key2 = lookup_api_key_by_hash(db, hash_token(raw_token))
    assert key2.last_used_at.epoch_ms() == first_used

    db.close()


# ---------------------------------------------------------------------------
# Streaming interceptor guards
# ---------------------------------------------------------------------------


def test_auth_interceptor_rejects_server_stream(interceptor):
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_server_stream_sync(None, None, None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


def test_auth_interceptor_rejects_client_stream(interceptor):
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_client_stream_sync(None, None, None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


def test_auth_interceptor_rejects_bidi_stream(interceptor):
    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_bidi_stream_sync(None, None, None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


# ---------------------------------------------------------------------------
# GcpAccessTokenProvider caching
# ---------------------------------------------------------------------------


def test_gcp_access_token_provider_caches_token():
    import datetime
    from unittest.mock import MagicMock, patch

    from iris.rpc.auth import GcpAccessTokenProvider

    mock_creds = MagicMock()
    mock_creds.token = "cached-token"
    mock_creds.expiry = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=1)

    provider = GcpAccessTokenProvider()
    with patch("google.auth.default", return_value=(mock_creds, "project-id")):
        token1 = provider.get_token()
        token2 = provider.get_token()

    assert token1 == "cached-token"
    assert token2 == "cached-token"
    # refresh should only be called once due to caching
    mock_creds.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# _extract_cookie helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cookie_header, name, expected",
    [
        ("iris_session=abc123", "iris_session", "abc123"),
        ("other=x; iris_session=abc123", "iris_session", "abc123"),
        ("iris_session=abc123; other=y", "iris_session", "abc123"),
        ("other=x", "iris_session", None),
        ("", "iris_session", None),
    ],
)
def test_extract_cookie(cookie_header, name, expected):
    assert _extract_cookie(cookie_header, name) == expected


def test_extract_cookie_empty_string():
    assert _extract_cookie("", "iris_session") is None
