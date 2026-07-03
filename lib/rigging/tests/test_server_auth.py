# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
from connectrpc._headers import Headers
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import (
    LOOPBACK_IDENTITY,
    AuthInterceptor,
    AuthRequest,
    GcpAccessTokenVerifier,
    IapAssertionVerifier,
    IapIdTokenVerifier,
    NullAuthInterceptor,
    StaticTokenVerifier,
    VerifiedIdentity,
    _extract_cookie,
    _verified_identity,
    build_request_authenticators,
    get_verified_identity,
    get_verified_user,
    require_identity,
    resolve_auth,
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
    return AuthInterceptor(verifier, cookie_name="iris_session")


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


def test_static_token_verifier_valid():
    v = StaticTokenVerifier({"tok": "user1"})
    result = v.verify("tok")
    assert result.user_id == "user1"
    assert result.role == "user"


def test_static_token_verifier_with_custom_role():
    v = StaticTokenVerifier({"tok": "admin1"}, roles={"admin1": "admin"})
    result = v.verify("tok")
    assert result.user_id == "admin1"
    assert result.role == "admin"


def test_static_token_verifier_invalid():
    v = StaticTokenVerifier({"tok": "user1"})
    with pytest.raises(ValueError, match="Invalid token"):
        v.verify("bad")


def _verify_oauth2_token_returning(payload):
    """Patch target factory: a stand-in for google's verify_oauth2_token."""
    return Mock(return_value=payload)


def test_iap_id_token_verifier_accepts_matching_audience():
    verifier = IapIdTokenVerifier(["desktop-client-id", "iap-client-id"])
    payload = {"aud": "desktop-client-id", "email": "alice@example.com", "email_verified": True}
    with patch("google.oauth2.id_token.verify_oauth2_token", _verify_oauth2_token_returning(payload)):
        identity = verifier.verify("id-token")
    assert identity == VerifiedIdentity(user_id="alice@example.com", role="user")


def test_iap_id_token_verifier_rejects_wrong_audience():
    verifier = IapIdTokenVerifier(["expected-aud"])
    payload = {"aud": "some-other-client", "email": "alice@example.com"}
    with patch("google.oauth2.id_token.verify_oauth2_token", _verify_oauth2_token_returning(payload)):
        with pytest.raises(ValueError, match="audience"):
            verifier.verify("id-token")


def test_iap_id_token_verifier_rejects_missing_email():
    verifier = IapIdTokenVerifier(["aud"])
    with patch("google.oauth2.id_token.verify_oauth2_token", _verify_oauth2_token_returning({"aud": "aud"})):
        with pytest.raises(ValueError, match="email"):
            verifier.verify("id-token")


def test_iap_id_token_verifier_rejects_unverified_email():
    verifier = IapIdTokenVerifier(["aud"])
    payload = {"aud": "aud", "email": "alice@example.com", "email_verified": False}
    with patch("google.oauth2.id_token.verify_oauth2_token", _verify_oauth2_token_returning(payload)):
        with pytest.raises(ValueError, match="not verified"):
            verifier.verify("id-token")


def test_iap_id_token_verifier_wraps_google_failure():
    verifier = IapIdTokenVerifier(["aud"])
    with patch("google.oauth2.id_token.verify_oauth2_token", side_effect=ValueError("bad signature")):
        with pytest.raises(ValueError, match="IAP ID token verification failed"):
            verifier.verify("id-token")


# --- IAP signed-header assertion -> implicit read-only dashboard identity -----

_ASSERTION_HEADERS = {"x-goog-iap-jwt-assertion": "signed.assertion.jwt"}


def test_iap_assertion_verifier_grants_dashboard_role():
    verifier = IapAssertionVerifier(
        "/projects/1/global/backendServices/2",
        role_resolver=lambda _email: "dashboard",
    )
    payload = {"aud": "/projects/1/global/backendServices/2", "email": "alice@example.com"}
    with patch("google.oauth2.id_token.verify_token", Mock(return_value=payload)):
        identity = verifier.identity_from_headers(_ASSERTION_HEADERS)
    assert identity == VerifiedIdentity(user_id="alice@example.com", role="dashboard")


def test_iap_assertion_verifier_resolves_provisioned_role():
    # With a role resolver injected (as the controller does), a provisioned email
    # resolves to its real role instead of the read-only dashboard default — the
    # path that lets an admin behind IAP act without running `iris login`.
    roles = {"admin@example.com": "admin"}
    verifier = IapAssertionVerifier(
        "/projects/1/global/backendServices/2",
        role_resolver=lambda email: roles.get(email, "dashboard"),
    )
    payload = {"aud": "/projects/1/global/backendServices/2", "email": "admin@example.com"}
    with patch("google.oauth2.id_token.verify_token", Mock(return_value=payload)):
        identity = verifier.identity_from_headers(_ASSERTION_HEADERS)
    assert identity == VerifiedIdentity(user_id="admin@example.com", role="admin")


def test_iap_assertion_verifier_returns_none_without_header():
    verifier = IapAssertionVerifier(
        "/projects/1/global/backendServices/2",
        role_resolver=lambda _email: "dashboard",
    )
    # No assertion header -> not an IAP request; the caller falls through to
    # loopback/optional/reject instead of getting a dashboard identity.
    assert verifier.identity_from_headers({}) is None


def test_iap_assertion_verifier_rejects_forged_assertion():
    verifier = IapAssertionVerifier(
        "/projects/1/global/backendServices/2",
        role_resolver=lambda _email: "dashboard",
    )
    with patch("google.oauth2.id_token.verify_token", side_effect=ValueError("Wrong recipient")):
        with pytest.raises(ValueError, match="IAP assertion verification failed"):
            verifier.identity_from_headers(_ASSERTION_HEADERS)


def test_iap_assertion_verifier_rejects_missing_email():
    verifier = IapAssertionVerifier(
        "/projects/1/global/backendServices/2",
        role_resolver=lambda _email: "dashboard",
    )
    with patch("google.oauth2.id_token.verify_token", Mock(return_value={"aud": "x"})):
        with pytest.raises(ValueError, match="no email"):
            verifier.identity_from_headers(_ASSERTION_HEADERS)


class _FakeAssertionVerifier:
    """Stand-in mirroring IapAssertionVerifier's header contract.

    Returns a dashboard identity when the signed-header is present and valid,
    None when it is absent, and raises when present but forged.
    """

    def identity_from_headers(self, headers):
        value = headers.get("x-goog-iap-jwt-assertion")
        if not value:
            return None
        if value == "forged":
            raise ValueError("IAP assertion verification failed")
        return VerifiedIdentity(user_id="alice@example.com", role="dashboard")


def test_resolve_auth_iap_assertion_grants_dashboard_when_tokenless():
    identity = resolve_auth(
        AuthRequest(token=None, headers={"x-goog-iap-jwt-assertion": "valid"}),
        build_request_authenticators(StaticTokenVerifier({}), _FakeAssertionVerifier()),
        optional=False,
    )
    assert identity == VerifiedIdentity(user_id="alice@example.com", role="dashboard")


def test_resolve_auth_iris_jwt_wins_over_iap_assertion():
    # A present Iris JWT outranks the implicit IAP path: a logged-in user keeps
    # their real role even though IAP also injected an assertion.
    identity = resolve_auth(
        AuthRequest(token="valid-token-alice", headers={"x-goog-iap-jwt-assertion": "valid"}),
        build_request_authenticators(StaticTokenVerifier({"valid-token-alice": "alice"}), _FakeAssertionVerifier()),
        optional=False,
    )
    assert identity == VerifiedIdentity(user_id="alice", role="user")


def test_resolve_auth_rejects_tokenless_without_assertion():
    # Behind IAP with optional=false, a tokenless call that carries no valid
    # assertion (i.e. did not pass IAP) is rejected — never anonymous-admin.
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            AuthRequest(token=None, headers={}),
            build_request_authenticators(StaticTokenVerifier({}), _FakeAssertionVerifier()),
            optional=False,
        )


def test_resolve_auth_rejects_forged_assertion():
    with pytest.raises(ValueError, match="IAP assertion verification failed"):
        resolve_auth(
            AuthRequest(token=None, headers={"x-goog-iap-jwt-assertion": "forged"}),
            build_request_authenticators(StaticTokenVerifier({}), _FakeAssertionVerifier()),
            optional=False,
        )


def test_resolve_auth_loopback_admin_when_no_assertion():
    # A genuine loopback peer (SSH tunnel) with no assertion still resolves to
    # the admin identity even when the assertion verifier is configured.
    identity = resolve_auth(
        AuthRequest(token=None, headers={}, client_address="127.0.0.1:54321"),
        build_request_authenticators(StaticTokenVerifier({}), _FakeAssertionVerifier()),
        optional=False,
    )
    assert identity == LOOPBACK_IDENTITY


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
    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"email": "alice@example.com"}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = verifier.verify("fake-access-token")

    assert result.user_id == "alice@example.com"
    assert result.role == "user"
    mock_get.assert_called_once_with(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"access_token": "fake-access-token"},
        timeout=10,
    )


def test_gcp_access_token_verifier_invalid_token():
    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 401

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="Token verification failed"):
            verifier.verify("bad-token")


def test_gcp_access_token_verifier_no_email():
    verifier = GcpAccessTokenVerifier()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"scope": "openid"}

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="email"):
            verifier.verify("token-without-email")


def test_gcp_access_token_verifier_checks_project_access():
    verifier = GcpAccessTokenVerifier(project_id="my-project")

    tokeninfo_resp = Mock()
    tokeninfo_resp.status_code = 200
    tokeninfo_resp.json.return_value = {"email": "alice@example.com"}

    project_resp = Mock()
    project_resp.status_code = 200

    with patch("requests.get", side_effect=[tokeninfo_resp, project_resp]) as mock_get:
        result = verifier.verify("valid-token")

    assert result.user_id == "alice@example.com"
    assert mock_get.call_count == 2
    mock_get.assert_any_call(
        "https://cloudresourcemanager.googleapis.com/v3/projects/my-project",
        headers={"Authorization": "Bearer valid-token"},
        timeout=10,
    )


def test_gcp_access_token_verifier_rejects_no_project_access():
    verifier = GcpAccessTokenVerifier(project_id="restricted-project")

    tokeninfo_resp = Mock()
    tokeninfo_resp.status_code = 200
    tokeninfo_resp.json.return_value = {"email": "alice@example.com"}

    project_resp = Mock()
    project_resp.status_code = 403

    with patch("requests.get", side_effect=[tokeninfo_resp, project_resp]):
        with pytest.raises(ValueError, match="does not have access"):
            verifier.verify("valid-token")


# ---------------------------------------------------------------------------
# NullAuthInterceptor
# ---------------------------------------------------------------------------


def test_null_auth_interceptor_sets_anonymous_user():
    interceptor = NullAuthInterceptor()
    captured = []

    def handler(req, ctx):
        captured.append((get_verified_user(), get_verified_identity()))
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert result == "ok"
    assert captured[0][0] == "anonymous"
    identity = captured[0][1]
    assert identity is not None
    assert identity.user_id == "anonymous"
    assert identity.role == "admin"


def test_null_auth_interceptor_custom_user():
    interceptor = NullAuthInterceptor(user="custom-user", role="user")
    captured = []

    def handler(req, ctx):
        captured.append(get_verified_identity())
        return "ok"

    interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert captured[0].user_id == "custom-user"
    assert captured[0].role == "user"


def test_null_auth_interceptor_resets_context():
    interceptor = NullAuthInterceptor()

    def handler(req, ctx):
        assert get_verified_user() == "anonymous"
        return "ok"

    interceptor.intercept_unary_sync(handler, "request", _make_ctx())
    assert get_verified_user() is None
    assert get_verified_identity() is None


def test_null_auth_interceptor_resets_context_on_error():
    interceptor = NullAuthInterceptor()

    def failing_handler(req, ctx):
        assert get_verified_user() == "anonymous"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        interceptor.intercept_unary_sync(failing_handler, "request", _make_ctx())

    assert get_verified_user() is None
    assert get_verified_identity() is None


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


# ---------------------------------------------------------------------------
# require_identity
# ---------------------------------------------------------------------------


def test_require_identity_returns_identity():
    reset = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        identity = require_identity()
        assert identity.user_id == "alice"
    finally:
        _verified_identity.reset(reset)


def test_require_identity_raises_unauthenticated():
    with pytest.raises(ConnectError) as exc_info:
        require_identity()
    assert exc_info.value.code == Code.UNAUTHENTICATED
