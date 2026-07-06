# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transport-generic, client-side authentication for Connect RPC.

This module owns only the *client* side of auth: acquiring a bearer token and
attaching it as a request header. It carries no knowledge of any particular
service — no JWT minting/verification, no role semantics, no token store. Those
service-specific concerns live with the service (e.g. iris).

Token sources are provided against ambient Google credentials:
``GcpAccessTokenProvider`` mints OAuth2 *access* tokens (for Google APIs and
loopback-trust services). Two providers mint the Google-signed OIDC *ID* token
an IAP-fronted service requires, differing only in where the credential comes
from: ``IapServiceAccountTokenProvider`` uses ``fetch_id_token`` (service
accounts, GCE metadata, impersonation — the in-cluster / CI path), while
``IapRefreshTokenProvider`` re-mints from a cached desktop-OAuth refresh token
(the human path; obtain the initial token once with ``run_iap_desktop_login``).
All cache the token until shortly before expiry and only touch the network
inside ``get_token``.

A single ``BearerTokenInjector`` attaches the token to outgoing requests under a
caller-chosen header — ``authorization`` for app auth, ``proxy-authorization``
for the IAP edge token. It is a Connect *metadata* interceptor (the ``on_start``
hook), so the header rides on every RPC shape — unary and streaming alike — for
both sync and async clients.
"""

import json
import os
import time
import webbrowser
from dataclasses import dataclass
from typing import Protocol, cast

import google.auth
import google.auth.exceptions
import google.auth.jwt
import google.auth.transport.requests
import google.oauth2.credentials
import google.oauth2.id_token

_REFRESH_MARGIN_SECONDS = 300

# OAuth scopes for the IAP desktop-login flow. "openid" makes the token endpoint
# return an OIDC ID token (the credential IAP requires); "email" puts the user's
# address in the token so the service can attribute the identity.
IAP_LOGIN_SCOPES = ["openid", "email"]
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"


def _monotonic_expiry(expiry_wall: float | None) -> float:
    """The ``time.monotonic`` deadline to cache a token until.

    Converts a wall-clock expiry to a monotonic deadline and subtracts the
    refresh margin, falling back to ``margin`` from now when the expiry is
    unknown. Caching keys off ``monotonic`` so a wall-clock step can't extend a
    token's lifetime.
    """
    now_mono = time.monotonic()
    if expiry_wall is None:
        return now_mono + _REFRESH_MARGIN_SECONDS
    return now_mono + (expiry_wall - time.time()) - _REFRESH_MARGIN_SECONDS


class IapLoginRequired(RuntimeError):
    """No usable IAP credentials are available for the human (desktop-OAuth) path.

    Raised when nothing is cached yet, or when the cached refresh token has
    expired or been revoked so the silent ID-token re-mint failed. The message is
    self-contained — it names the remedy (log in again) — so a
    CLI can surface ``str(exc)`` directly instead of synthesising its own remedy.
    """


class TokenProvider(Protocol):
    """Provides a bearer token for outgoing requests."""

    def get_token(self) -> str | None:
        """Return a token string, or None to skip auth."""
        ...


class StaticTokenProvider:
    """Returns a fixed token. Useful for testing and worker auth."""

    def __init__(self, token: str):
        self._token = token

    def get_token(self) -> str | None:
        return self._token


class GcpAccessTokenProvider:
    """Mints OAuth2 access tokens from ambient Google credentials.

    Works for all credential types: user accounts (from ``gcloud auth
    application-default login``), service accounts, and GCE metadata. Tokens are
    cached until five minutes before expiry. Credential discovery and refresh
    happen only inside ``get_token``.
    """

    def __init__(self):
        self._creds = None
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str | None:
        if self._cached_token is not None and time.monotonic() < self._expires_at:
            return self._cached_token

        if self._creds is None:
            self._creds, _ = google.auth.default()
        self._creds.refresh(google.auth.transport.requests.Request())

        expiry = self._creds.expiry.timestamp() if self._creds.expiry is not None else None
        self._cached_token = self._creds.token
        self._expires_at = _monotonic_expiry(expiry)
        return self._cached_token


class IapServiceAccountTokenProvider:
    """Mints OIDC ID tokens for IAP from ambient *service-account* credentials.

    Uses ``fetch_id_token``, which works for service accounts, GCE metadata, and
    impersonated credentials (the in-cluster / CI path) but **not** end-user
    ``gcloud`` credentials. The audience is the OAuth client id of the
    IAP-protected resource. The token is cached until five minutes before its
    ``exp`` claim; credential access happens only inside ``get_token``.
    """

    def __init__(self, audience: str):
        self._audience = audience
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str | None:
        if self._cached_token is not None and time.monotonic() < self._expires_at:
            return self._cached_token

        token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), self._audience)
        claims = google.auth.jwt.decode(token, verify=False)

        self._cached_token = token
        self._expires_at = _monotonic_expiry(claims.get("exp"))
        return self._cached_token


class IapRefreshTokenProvider:
    """Re-mints an OIDC ID token for IAP from a cached desktop-OAuth refresh token.

    IAP requires an ID token (not an access token); this silently re-mints it
    from the user's long-lived refresh token with no browser prompt. Obtain the
    initial refresh token once via the desktop login flow. The token's ``aud``
    is the desktop client id, which must be on the cluster's IAP audience
    allowlist.
    """

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, *, login_hint: str | None = None):
        # login_hint is appended to the IapLoginRequired raised when the refresh
        # token is expired/revoked, so the caller's "log in again" remedy (which
        # depends on the logical endpoint name) travels with the error.
        self._login_hint = login_hint
        self._creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            token_uri=_GOOGLE_TOKEN_URI,
            scopes=IAP_LOGIN_SCOPES,
        )

    def get_token(self) -> str | None:
        # creds.valid is False until the first refresh and once the access token
        # (minted alongside the ID token) expires; refreshing repopulates both.
        if self._creds.id_token is None or not self._creds.valid:
            try:
                self._creds.refresh(google.auth.transport.requests.Request())
            except google.auth.exceptions.RefreshError as exc:
                message = "cached IAP credentials are no longer valid (token refresh failed)"
                if self._login_hint:
                    message = f"{message}; {self._login_hint}"
                raise IapLoginRequired(message) from exc
        return self._creds.id_token


@dataclass(frozen=True)
class OAuthClient:
    """A Google OAuth client identity (client id + secret)."""

    client_id: str
    client_secret: str


# The Marin desktop ("installed") OAuth client that drives the IAP browser-login
# flow. For an installed app the "client secret" is not confidential — it is part
# of the app's public identity, not a credential (RFC 8252 §8.5) — so it ships in
# source: a login needs no Console download, and the same refresh token is usable
# from any tool. It only names the app to Google's OAuth endpoint; IAP still
# authorizes each user individually against its per-backend allowlist. A cluster
# that fronts a different desktop client overrides this via config.
MARIN_DESKTOP_OAUTH_CLIENT = OAuthClient(
    client_id="748532799086-qf8m6mvovtdmd71npm07gk1ohijsr3q5.apps.googleusercontent.com",
    client_secret="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",
)


def read_desktop_client(path: str) -> OAuthClient:
    """Read a Google *desktop* ('installed') OAuth client secret JSON from ``path``."""
    with open(path) as f:
        installed = json.load(f).get("installed")
    if installed is None:
        raise ValueError(f"{path}: expected a desktop ('installed') OAuth client secret")
    return OAuthClient(installed["client_id"], installed["client_secret"])


def run_iap_desktop_login(
    client_id: str, client_secret: str, *, port: int = 0, headless: bool = False
) -> tuple[str, str]:
    """Run the installed-app OAuth flow; return (id_token, refresh_token).

    With ``headless=False`` (default) opens the system browser and catches the
    redirect on a localhost port — the right choice on a workstation. With
    ``headless=True`` (no local browser, e.g. an SSH session) it instead prints
    the authorization URL and reads back the pasted redirect URL or code, so no
    browser or port-forward on the box is required. Even with ``headless=False``
    we fall back to that paste flow automatically when no browser is registered,
    since ``run_local_server``'s localhost redirect is unreachable from wherever
    the user would otherwise open the URL.

    Returns the freshly minted OIDC ID token and the long-lived refresh token to
    cache for silent re-minting via :class:`IapRefreshTokenProvider`.
    """
    # Lazy import: google-auth-oauthlib pulls in requests-oauthlib and is only
    # needed for the interactive login path, never by a server or a worker.
    try:
        from google_auth_oauthlib.flow import Flow, InstalledAppFlow  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise RuntimeError(
            "IAP desktop login requires google-auth-oauthlib; install it with `pip install marin-rigging[iap]`"
        ) from exc

    # Both loopback paths trip oauthlib's strict defaults, and both are expected
    # and safe here: the redirect is http://localhost (not real transport), and
    # Google expands the requested "email" scope to its full userinfo URL, which
    # otherwise raises "Scope has changed". Relax both before either flow runs.
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": _GOOGLE_AUTH_URI,
            "token_uri": _GOOGLE_TOKEN_URI,
        }
    }

    if headless or not _browser_available():
        creds = _console_oauth(Flow, client_config)
    else:
        flow = InstalledAppFlow.from_client_config(client_config, scopes=IAP_LOGIN_SCOPES)
        creds = flow.run_local_server(port=port, open_browser=True)

    if not creds.id_token:
        raise RuntimeError("OAuth flow returned no ID token (the 'openid' scope must be granted)")
    if not creds.refresh_token:
        raise RuntimeError("OAuth flow returned no refresh token (request offline access)")
    # google-auth types these as object; the guards above prove they are non-empty strings.
    return cast(str, creds.id_token), cast(str, creds.refresh_token)


def _browser_available() -> bool:
    """True when a usable web browser is registered for ``webbrowser.open``.

    ``webbrowser.get()`` raises ``webbrowser.Error`` on a box with no browser
    (a headless server or a bare SSH session), which is exactly where the
    localhost-redirect flow cannot work and the paste-the-code flow must run.
    """
    try:
        webbrowser.get()
        return True
    except webbrowser.Error:
        return False


def _console_oauth(flow_cls, client_config: dict):
    """Manual loopback OAuth: print the URL, read back the pasted redirect/code.

    Works without a local browser or a reachable localhost port — the user opens
    the URL on any machine and pastes the resulting ``http://localhost/?code=...``
    URL (which the browser fails to load, but whose address bar holds the code).
    Callers must relax OAUTHLIB_INSECURE_TRANSPORT / OAUTHLIB_RELAX_TOKEN_SCOPE
    first (``run_iap_desktop_login`` does), since the http loopback redirect and
    Google's scope expansion both trip oauthlib's strict defaults.
    """
    flow = flow_cls.from_client_config(client_config, scopes=IAP_LOGIN_SCOPES, redirect_uri="http://localhost")
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    print(
        "\nOpen this URL in a browser, authorize, then paste the FULL redirected URL\n"
        "(it looks like http://localhost/?...code=...; the page itself will not load):\n\n"
        f"{auth_url}\n"
    )
    response = input("Redirected URL (or just the code): ").strip()
    if response.startswith("http"):
        flow.fetch_token(authorization_response=response)
    else:
        flow.fetch_token(code=response)
    return flow.credentials


class BearerTokenInjector:
    """Metadata interceptor that attaches ``<header>: Bearer <token>``.

    Implemented against Connect's metadata interceptor protocol (``on_start`` /
    ``on_start_sync``) rather than the unary hooks, so ``connectrpc`` applies it
    to every RPC shape — unary, client-stream, server-stream, and bidi — for
    both sync and async clients. No header is set when the provider returns None
    (the loopback / SSH-tunnel-trust case).

    The header is the lever between app auth and edge auth: app tokens ride in
    ``authorization``, the IAP edge token in ``proxy-authorization`` (so the
    app-level header stays free for the service's own JWT).
    """

    def __init__(self, provider: TokenProvider, header: str):
        self._provider = provider
        self.header = header

    def _apply(self, ctx) -> None:
        token = self._provider.get_token()
        if token:
            ctx.request_headers()[self.header] = f"Bearer {token}"

    def on_start_sync(self, ctx):
        self._apply(ctx)

    def on_end_sync(self, token, ctx, error) -> None:
        return

    async def on_start(self, ctx):
        self._apply(ctx)

    async def on_end(self, token, ctx, error) -> None:
        return
