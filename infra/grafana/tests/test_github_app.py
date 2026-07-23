# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for GitHub App installation-token minting."""

import json
import time

import httpx
import jwt
import pytest
from config import GithubAppCredentials, _github_app_credentials
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from errors import UpstreamError
from github_app import GithubAppAuth


def _keypair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    public_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    return private_pem, public_pem


def _auth(private_pem: str) -> GithubAppAuth:
    return GithubAppAuth(
        GithubAppCredentials(app_id="123", installation_id="456", private_key=private_pem),
        repository="marin-community/marin",
    )


def _fetch(auth: GithubAppAuth, handler) -> httpx.Response:
    """Drive one authenticated request through the auth flow against a mock GitHub."""
    client = httpx.Client(transport=httpx.MockTransport(handler), auth=auth)
    return client.get("https://api.github.com/anything")


def test_mints_scoped_token_and_sends_it(monkeypatch):
    private_pem, public_pem = _keypair()
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            claims = jwt.decode(
                request.headers["authorization"].removeprefix("Bearer "),
                public_pem,
                algorithms=["RS256"],
                # The app JWT's exp is stamped from the mocked clock, so skip the
                # library's wall-clock expiry check; the signature still verifies.
                options={"verify_exp": False},
            )
            seen["jwt_iss"] = claims["iss"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(201, json={"token": "ghs_minted", "expires_at": "2026-07-23T20:00:00Z"})
        seen["sent_auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    _fetch(_auth(private_pem), handler)

    assert seen["jwt_iss"] == "123"
    assert seen["body"]["repositories"] == ["marin"]
    assert seen["body"]["permissions"]["contents"] == "read"
    assert seen["sent_auth"] == "Bearer ghs_minted"


def test_caches_token_across_requests(monkeypatch):
    private_pem, _ = _keypair()
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    mints = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal mints
        if request.url.path.endswith("/access_tokens"):
            mints += 1
            return httpx.Response(201, json={"token": f"ghs_{mints}", "expires_at": "2026-07-23T20:00:00Z"})
        return httpx.Response(200, json={})

    auth = _auth(private_pem)
    client = httpx.Client(transport=httpx.MockTransport(handler), auth=auth)
    client.get("https://api.github.com/a")
    client.get("https://api.github.com/b")

    assert mints == 1


def test_refreshes_after_expiry(monkeypatch):
    private_pem, _ = _keypair()
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    tokens = iter(["ghs_first", "ghs_second"])
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            # Each token is valid for 100s; the skew forces a refresh well before that.
            expiry = time.gmtime(clock["now"] + 100)
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", expiry)
            return httpx.Response(201, json={"token": next(tokens), "expires_at": iso})
        sent.append(request.headers["authorization"])
        return httpx.Response(200, json={})

    auth = _auth(private_pem)
    client = httpx.Client(transport=httpx.MockTransport(handler), auth=auth)
    client.get("https://api.github.com/a")
    clock["now"] += 200  # past this token's expiry
    client.get("https://api.github.com/b")

    assert sent == ["Bearer ghs_first", "Bearer ghs_second"]


def test_mint_failure_raises_upstream_error(monkeypatch):
    private_pem, _ = _keypair()
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    with pytest.raises(UpstreamError) as excinfo:
        _fetch(_auth(private_pem), handler)
    assert excinfo.value.status_code == 502


def test_credentials_resolve_when_fully_configured(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "2")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "pem")
    assert _github_app_credentials() == GithubAppCredentials("1", "2", "pem")


def test_credentials_are_none_when_unset(monkeypatch):
    for key in ("GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert _github_app_credentials() is None


def test_credentials_reject_partial_config(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    for key in ("GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError):
        _github_app_credentials()
