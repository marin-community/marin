# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for GitHub App installation-token minting.

The GitHub API is faked at the httpx boundary; the assertions cover our logic —
the signed JWT, the token's repo/permission scope, caching, and error mapping.
"""

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


def _auth(private_pem: str, repositories=("marin-community/marin",)) -> GithubAppAuth:
    return GithubAppAuth(GithubAppCredentials(client_id="cid", private_key=private_pem), repositories)


def _github(auth: GithubAppAuth, handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), auth=auth)


def test_mints_a_scoped_readonly_token_and_sends_it(monkeypatch):
    private_pem, public_pem = _keypair()
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/installation"):
            # The app authenticates to the lookup with a JWT issued by the client id.
            claims = jwt.decode(
                request.headers["authorization"].removeprefix("Bearer "),
                public_pem,
                algorithms=["RS256"],
                options={"verify_exp": False},  # exp is stamped from the mocked clock
            )
            seen["issuer"] = claims["iss"]
            return httpx.Response(200, json={"id": 42})
        if request.url.path.endswith("/access_tokens"):
            assert request.url.path == "/app/installations/42/access_tokens"
            seen["token_body"] = json.loads(request.content)
            return httpx.Response(201, json={"token": "ghs_minted", "expires_at": "2026-07-23T20:00:00Z"})
        seen["sent_auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    _github(_auth(private_pem, ["marin-community/marin", "marin-community/vllm"]), handler).get("https://x/data")

    assert seen["issuer"] == "cid"
    assert seen["token_body"]["repositories"] == ["marin", "vllm"]
    assert seen["token_body"]["permissions"] == {
        k: "read" for k in ("metadata", "contents", "checks", "statuses", "actions")
    }
    assert seen["sent_auth"] == "Bearer ghs_minted"


def test_reuses_cached_token_then_refreshes_after_expiry(monkeypatch):
    private_pem, _ = _keypair()
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    mints = 0
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal mints
        if request.url.path.endswith("/installation"):
            return httpx.Response(200, json={"id": 42})
        if request.url.path.endswith("/access_tokens"):
            mints += 1
            # A GitHub installation token lives an hour; expiry is stamped from the clock.
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock["now"] + 3600))
            return httpx.Response(201, json={"token": f"ghs_{mints}", "expires_at": iso})
        sent.append(request.headers["authorization"])
        return httpx.Response(200, json={})

    client = _github(_auth(private_pem), handler)
    client.get("https://x/a")
    client.get("https://x/b")  # cached: no second mint
    clock["now"] += 3600  # past expiry
    client.get("https://x/c")  # refreshes

    assert mints == 2
    assert sent == ["Bearer ghs_1", "Bearer ghs_1", "Bearer ghs_2"]


def test_upstream_failure_surfaces_as_502(monkeypatch):
    private_pem, _ = _keypair()
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/installation"):
            return httpx.Response(200, json={"id": 42})
        return httpx.Response(404, text="not found")  # token mint fails

    with pytest.raises(UpstreamError) as excinfo:
        _github(_auth(private_pem), handler).get("https://x/data")
    assert excinfo.value.status_code == 502


def test_rejects_repositories_from_multiple_owners():
    private_pem, _ = _keypair()
    # One installation token is per-owner, so a mixed-owner set cannot be honored.
    with pytest.raises(ValueError):
        _auth(private_pem, ["marin-community/marin", "vllm-project/vllm"])


def test_credentials_resolve_when_fully_configured(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "cid")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "pem")
    assert _github_app_credentials() == GithubAppCredentials("cid", "pem")


def test_credentials_reject_partial_config(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "cid")
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError):
        _github_app_credentials()
