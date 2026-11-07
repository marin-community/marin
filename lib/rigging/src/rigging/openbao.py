# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenBao identity verification and remote service-token signing.

Only public keys leave Transit. Workloads receive audience-scoped JWTs;
the broker's OpenBao credential stays in its private rotating token file.
"""

import base64
import json
import ssl
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from rigging.grants import Grant, GrantDenied, Permission, permissions_cover
from rigging.shared_grants import authority_origin
from rigging.token_authority import JwksVerifier, VerifiedClaims, public_jwk_from_pem

TRANSIT_KEY_CACHE_TTL = 300


class OpenBaoClient:
    """Make bounded requests using a caller-managed, rotating Bao token file."""

    def __init__(self, address: str, token_file: Path, *, ca_bundle: str | bool = True, timeout: float = 10):
        if ca_bundle is False:
            raise ValueError("OpenBao TLS verification cannot be disabled")
        self.address = authority_origin(address)
        self.token_file = token_file
        self.ca_bundle = ca_bundle
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
            raise ValueError("expected an absolute OpenBao API path")
        response = requests.request(
            method,
            self.address + "/v1" + path,
            json=body,
            headers={"X-Vault-Token": self.token_file.read_text().strip()},
            timeout=self.timeout,
            verify=self.ca_bundle,
            allow_redirects=False,
        )
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("OpenBao redirected a credential-bearing request")
        return response.json() if response.content else {}


class OpenBaoIdentityVerifier:
    """Verify a fixed Bao issuer and map entity IDs to standing principals."""

    def __init__(self, *, issuer: str, audience: str, subjects: Mapping[str, str], ca_file: str | None = None):
        parsed = urlsplit(issuer)
        if (
            not parsed.hostname
            or (
                parsed.scheme != "https"
                and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
            )
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OpenBao identity issuer requires HTTPS outside loopback and no URL credentials or query")
        if not audience or not subjects:
            raise ValueError("OpenBao identity verification requires an audience and trusted subjects")
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.subjects = dict(subjects)
        self.keys = jwt.PyJWKClient(
            self.issuer + "/.well-known/keys", timeout=10, ssl_context=ssl.create_default_context(cafile=ca_file)
        )

    def verify(self, token: str) -> str:
        try:
            key = self.keys.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256", "EdDSA"],
                issuer=self.issuer,
                audience=self.audience,
                options={"require": ["iss", "sub", "aud", "exp", "iat"], "strict_aud": True},
            )
        except jwt.PyJWTError as exc:
            raise ValueError("Invalid OpenBao identity token") from exc
        principal = self.subjects.get(claims["sub"])
        if principal is None:
            raise ValueError("OpenBao entity has no standing principal mapping")
        return principal


def _encoded_json(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()


def _public_pem(value: str) -> str:
    return (
        Ed25519PublicKey.from_public_bytes(base64.b64decode(value))
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


class OpenBaoTransitSigner:
    """Sign the existing Rigging EdDSA JWT contract with a Transit key."""

    def __init__(self, client: OpenBaoClient, *, issuer: str, key_version: int, key: str = "marin-grants"):
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in key):
            raise ValueError("invalid Transit key name")
        if key_version < 1:
            raise ValueError("Transit signing requires an explicit positive key version")
        self.client = client
        self.issuer = issuer
        self.key = key
        self.key_version = key_version

    def public_keys(self) -> tuple[str, ...]:
        data = self.client.request("GET", f"/transit/keys/{self.key}")["data"]
        if str(self.key_version) not in data["keys"]:
            raise ValueError("The configured Transit signing version is unavailable")
        return tuple(_public_pem(entry["public_key"]) for entry in data["keys"].values())

    def public_jwks(self) -> dict:
        return {"keys": [public_jwk_from_pem(pem) for pem in self.public_keys()]}

    def mint(self, claims: dict, *, audience: str, ttl_seconds: int) -> str:
        if ttl_seconds <= 0 or not audience:
            raise ValueError("a positive lifetime and audience are required")
        if {"iss", "aud", "iat", "exp"} & claims.keys():
            raise ValueError("registered token claims belong to the signer")
        data = self.client.request("GET", f"/transit/keys/{self.key}")["data"]
        version = self.key_version
        kid = public_jwk_from_pem(_public_pem(data["keys"][str(version)]["public_key"]))["kid"]
        now = int(time.time())
        header = _encoded_json({"alg": "EdDSA", "typ": "at+jwt", "kid": kid})
        payload = _encoded_json({**claims, "iss": self.issuer, "aud": audience, "iat": now, "exp": now + ttl_seconds})
        signing_input = f"{header}.{payload}"
        result = self.client.request(
            "POST",
            f"/transit/sign/{self.key}",
            {"input": base64.b64encode(signing_input.encode()).decode(), "key_version": version},
        )
        _, signed_version, signature = result["data"]["signature"].split(":")
        if signed_version != f"v{version}":
            raise ValueError("Transit returned an unexpected signing key version")
        encoded_signature = base64.urlsafe_b64encode(base64.b64decode(signature)).rstrip(b"=").decode()
        return f"{signing_input}.{encoded_signature}"


class OpenBaoStorageIssuer:
    """Issue a fixed provider identity only when its entire policy fits a grant.

    CoreWeave derives permissions from issuer and subject. It does not downscope
    that policy using our grant JWT, so a narrower request must be rejected.
    """

    def __init__(
        self,
        client: OpenBaoClient,
        *,
        role: str,
        issuer: str,
        subject: str,
        permissions: tuple[Permission, ...],
    ):
        if not role or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in role):
            raise ValueError("invalid identity role name")
        if not permissions or any(permission.service != "cw_s3" for permission in permissions):
            raise ValueError("CoreWeave bindings require explicit cw_s3 permissions")
        self.client = client
        self.role = role
        self.permissions = permissions
        self.verifier = OpenBaoIdentityVerifier(
            issuer=issuer,
            audience="https://coreweave.com/iam",
            subjects={subject: "storage"},
            ca_file=client.ca_bundle if isinstance(client.ca_bundle, str) else None,
        )

    def issue(self, grant: Grant) -> dict:
        if not permissions_cover(grant.permissions, self.permissions):
            raise GrantDenied("The provider identity would exceed this grant")
        result = self.client.request("GET", f"/identity/oidc/token/{self.role}")["data"]
        token = result["token"]
        self.verifier.verify(token)
        # The signature has been checked above; these claims bound file refresh.
        claims = jwt.decode(token, options={"verify_signature": False})
        if claims["exp"] > grant.expires_at:
            raise GrantDenied("The fixed provider token lifetime would exceed this grant")
        return {"token": token, "expires_at": claims["exp"]}


class OpenBaoTransitVerifier:
    """Refresh public Transit keys on rotation and expire cached trust after five minutes."""

    def __init__(
        self,
        signer: OpenBaoTransitSigner,
        *,
        audiences: frozenset[str],
        now: Callable[[], float] = time.monotonic,
    ):
        self._now = now
        self.signer = signer
        self.audiences = audiences
        self._lock = threading.Lock()
        self._refreshing = False
        self._refresh()

    def _refresh(self) -> None:
        keys = self.signer.public_keys()
        kids = {public_jwk_from_pem(key)["kid"] for key in keys}
        verifier = JwksVerifier(issuers={self.signer.issuer: keys}, expected_audiences=self.audiences, leeway_seconds=0)
        self._state = kids, verifier, self._now() + TRANSIT_KEY_CACHE_TTL
        self._refresh_after = self._now() + 1

    def verify(self, token: str) -> VerifiedClaims:
        try:
            header = jwt.get_unverified_header(token)
            issuer = jwt.decode(token, options={"verify_signature": False}).get("iss")
        except jwt.PyJWTError as exc:
            raise ValueError("Malformed grant token") from exc
        kid = header.get("kid")
        if issuer != self.signer.issuer or header.get("alg") != "EdDSA" or not isinstance(kid, str) or not kid:
            raise ValueError("Unexpected grant issuer, algorithm, or key identifier")
        kids, verifier, expires_at = self._state
        if kid in kids and self._now() < expires_at:
            return verifier.verify(token)
        refresh = False
        with self._lock:
            if not self._refreshing and self._now() >= self._refresh_after:
                self._refreshing = True
                refresh = True
        if refresh:
            try:
                self._refresh()
            finally:
                with self._lock:
                    self._refresh_after = self._now() + 1
                    self._refreshing = False
        kids, verifier, expires_at = self._state
        if self._now() >= expires_at:
            raise ValueError("Grant signing key metadata has expired")
        if kid not in kids:
            raise ValueError("Unknown grant signing key")
        return verifier.verify(token)
