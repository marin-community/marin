# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Asymmetric (EdDSA/Ed25519) service-token signing and verification.

The generic mechanism a Marin service uses to be its own signing authority:
mint short-lived JWTs with a per-cluster Ed25519 private key, publish the public
half as a JWKS / PEM, and verify a presented token against a configured
allowlist of trusted issuers. It carries no service policy — no role/RBAC
semantics, no revocation, no login exchange — so it never imports iris. Higher
layers (``rigging.server_auth``, iris's ``JwtTokenManager``) build policy on top.

Two invariants make this security-critical:

- Per-plane audience discipline (RFC 8725). A single signing key serves every
  plane, so the key itself provides no cross-plane isolation; every minted token
  names exactly one ``aud`` (plane) and every verifier requires its expected
  ``aud`` set — a token minted for one plane is rejected at another
  (:class:`JwksVerifier` is fail-closed: ``expected_audiences`` is mandatory).
- No SSRF key discovery. ``issuers`` is a configured ``iss -> [public PEM]``
  allowlist; the verifier resolves the key by the token's ``iss`` (then ``kid``)
  and never fetches a JWKS URL derived from the token.

The token contract is frozen and shared byte-for-byte with finelog's Rust
verifier: EdDSA header ``{"alg":"EdDSA","typ":"JWT","kid":<thumbprint>}`` where
``kid`` is the RFC 7638 JWK thumbprint of the public key; the public key is
emittable both as a JWKS entry and as a SubjectPublicKeyInfo PEM.
"""

import base64
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

# Frozen token/JWK contract (mirrored in finelog's Rust verifier).
ALGORITHM = "EdDSA"
_OKP_KTY = "OKP"
_ED25519_CRV = "Ed25519"
_SIG_USE = "sig"

# Registered claims the signer owns and always sets. A caller supplying any of
# these in its claims dict is a bug (its value would be silently overridden), so
# minting rejects it up front.
_SIGNER_OWNED_CLAIMS = frozenset({"iss", "aud", "iat", "exp"})

# Claims a verifier requires present — a token missing any is malformed.
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud"]

DEFAULT_LEEWAY_SECONDS = 60


@dataclass(frozen=True)
class SigningKey:
    """An Ed25519 signing key: the private half plus its derived public views.

    ``kid`` is the RFC 7638 thumbprint of ``public_pem`` — stable for a given
    key, so the header ``kid`` a verifier resolves on is reproducible.
    """

    kid: str
    private_pem: str  # Ed25519 PKCS8 PEM (the private half)
    public_pem: str  # SubjectPublicKeyInfo PEM (derived from the private half)


@dataclass(frozen=True)
class Ed25519Keypair:
    """A freshly generated keypair with every serialized view a caller needs.

    Returned by :func:`generate_ed25519_keypair` for ``iris cluster init-keys``:
    the private PEM goes to a SecretSpec destination, the public PEM / JWK go to
    the trust config of every verifier.
    """

    kid: str
    private_pem: str
    public_pem: str
    public_jwk: dict


def _load_ed25519_private_key(private_pem: str) -> Ed25519PrivateKey:
    key = load_pem_private_key(private_pem.encode("utf-8"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"expected an Ed25519 private key, got {type(key).__name__}")
    return key


def _load_ed25519_public_key(public_pem: str) -> Ed25519PublicKey:
    key = load_pem_public_key(public_pem.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"expected an Ed25519 public key, got {type(key).__name__}")
    return key


def _private_pem(key: Ed25519PrivateKey) -> str:
    return key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("ascii")


def _public_pem(key: Ed25519PublicKey) -> str:
    return key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo).decode("ascii")


def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwk_thumbprint(x_b64url: str) -> str:
    """RFC 7638 JWK thumbprint of an Ed25519 (OKP) public key.

    The canonical form has exactly the required members in lexicographic order
    ``{"crv","kty","x"}`` with no whitespace; the thumbprint is the base64url
    (no padding) SHA-256 of that JSON.
    """
    canonical = json.dumps(
        {"crv": _ED25519_CRV, "kty": _OKP_KTY, "x": x_b64url},
        separators=(",", ":"),
        sort_keys=True,
    )
    return _b64url_nopad(hashlib.sha256(canonical.encode("ascii")).digest())


def _public_jwk(key: Ed25519PublicKey) -> dict:
    x = _b64url_nopad(key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw))
    return {
        "kty": _OKP_KTY,
        "crv": _ED25519_CRV,
        "use": _SIG_USE,
        "alg": ALGORITHM,
        "kid": _jwk_thumbprint(x),
        "x": x,
    }


def generate_ed25519_keypair() -> Ed25519Keypair:
    """Generate a fresh Ed25519 keypair with its kid, PEMs, and public JWK."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    jwk = _public_jwk(public_key)
    return Ed25519Keypair(
        kid=jwk["kid"],
        private_pem=_private_pem(private_key),
        public_pem=_public_pem(public_key),
        public_jwk=jwk,
    )


def signing_key_from_private_pem(private_pem: str) -> SigningKey:
    """Load a private PEM and derive the public PEM and RFC 7638 ``kid``."""
    private_key = _load_ed25519_private_key(private_pem)
    public_key = private_key.public_key()
    return SigningKey(
        kid=_public_jwk(public_key)["kid"],
        private_pem=private_pem,
        public_pem=_public_pem(public_key),
    )


def public_jwk_from_pem(public_pem: str) -> dict:
    """Convert a SubjectPublicKeyInfo PEM to a JWKS entry (with its ``kid``)."""
    return _public_jwk(_load_ed25519_public_key(public_pem))


class JwtSigner:
    """Mints EdDSA service tokens for one issuer with one signing key.

    Every token carries ``iss`` (this signer's issuer), the per-mint ``aud``,
    and ``iat``/``exp`` (from ``ttl_seconds``); the header carries the key's
    ``kid``. All other claims (``sub``, ``role``, ``jti``, ``scope``, …) are the
    caller's — this primitive invents no policy claims.
    """

    def __init__(self, key: SigningKey, *, issuer: str):
        if not issuer:
            raise ValueError("JwtSigner requires a non-empty issuer")
        self._key = key
        self._issuer = issuer

    def mint(self, claims: dict, *, audience: str, ttl_seconds: int) -> str:
        """Mint a signed EdDSA token.

        Sets ``iss``/``aud``/``iat``/``exp`` and the header ``kid``; ``claims``
        supplies everything else. Raises ``ValueError`` if ``claims`` tries to
        set a signer-owned registered claim, or on a non-positive ``ttl_seconds``.
        """
        if not audience:
            raise ValueError("mint requires a non-empty audience")
        if ttl_seconds <= 0:
            raise ValueError("mint requires a positive ttl_seconds")
        conflicting = _SIGNER_OWNED_CLAIMS & claims.keys()
        if conflicting:
            raise ValueError(f"claims must not set signer-owned registered claims: {sorted(conflicting)}")
        now = int(time.time())
        payload = {**claims, "iss": self._issuer, "aud": audience, "iat": now, "exp": now + ttl_seconds}
        return jwt.encode(payload, self._key.private_pem, algorithm=ALGORITHM, headers={"kid": self._key.kid})

    def public_jwks(self, *, also: Sequence[str] = ()) -> dict:
        """Public JWKS for ``/.well-known/jwks.json``: this key plus ``also``.

        ``also`` are extra public-key PEMs (retained previous keys) served
        during a rotation overlap so verifiers accept both old and new tokens.
        """
        keys = [_public_jwk(_load_ed25519_public_key(self._key.public_pem))]
        keys.extend(public_jwk_from_pem(pem) for pem in also)
        return {"keys": keys}


@dataclass(frozen=True)
class VerifiedClaims:
    """The verified claim set — raw claims, not an identity.

    Mapping ``sub``/``aud``/``scope`` to a role or identity is service policy;
    this is only the cryptographically verified payload.
    """

    sub: str
    iss: str
    aud: str
    scope: str | None
    claims: dict


class JwksVerifier:
    """Verifies EdDSA tokens against a configured issuer -> public-key allowlist.

    ``issuers`` maps each trusted ``iss`` to a list of its trusted public-key
    PEMs (a list to span a rotation overlap). ``expected_audiences`` is the
    surface's allowed ``aud`` set and is required and fail-closed: a token
    whose ``aud`` is outside it is rejected — the per-plane replay guard.

    Key resolution is by the token's ``iss`` then, among that issuer's keys, by
    the header ``kid`` (falling back to trying each of the issuer's keys when the
    header carries no ``kid`` or an unrecognized one). The key set is never
    discovered from the token (no SSRF).
    """

    def __init__(
        self,
        *,
        issuers: Mapping[str, Sequence[str]],
        expected_audiences: frozenset[str],
        leeway_seconds: int = DEFAULT_LEEWAY_SECONDS,
    ):
        if not expected_audiences:
            raise ValueError("JwksVerifier requires a non-empty expected_audiences (fail-closed audience discipline)")
        self._expected_audiences = frozenset(expected_audiences)
        self._leeway_seconds = leeway_seconds
        self._issuer_keys: dict[str, tuple[tuple[str, Ed25519PublicKey], ...]] = {}
        for iss, pems in issuers.items():
            if isinstance(pems, str):
                raise ValueError(f"issuer {iss!r}: expected a sequence of PEMs, not a bare string")
            entries = tuple((public_jwk_from_pem(pem)["kid"], _load_ed25519_public_key(pem)) for pem in pems)
            if not entries:
                raise ValueError(f"issuer {iss!r} has no trusted public keys")
            self._issuer_keys[iss] = entries

    def verify(self, token: str) -> VerifiedClaims:
        """Verify signature, expiry, issuer, and audience; return the claims.

        Raises ``ValueError`` on any failure — bad signature, expired, an ``aud``
        outside ``expected_audiences``, an unknown ``iss``, a non-EdDSA ``alg``,
        or a malformed token — matching ``server_auth.TokenVerifier``.
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
            unverified_claims = jwt.decode(token, options={"verify_signature": False})
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Malformed token: {exc}") from exc

        iss = unverified_claims.get("iss")
        if not isinstance(iss, str) or iss not in self._issuer_keys:
            raise ValueError(f"Untrusted or missing issuer: {iss!r}")

        payload = self._verify_against_issuer(token, iss, unverified_header.get("kid"))
        return VerifiedClaims(
            sub=payload.get("sub", ""),
            iss=payload["iss"],
            aud=payload["aud"],
            scope=payload.get("scope"),
            claims=payload,
        )

    def _candidate_keys(self, iss: str, kid: str | None) -> tuple[Ed25519PublicKey, ...]:
        entries = self._issuer_keys[iss]
        if kid is not None:
            matched = tuple(key for entry_kid, key in entries if entry_kid == kid)
            if matched:
                return matched
        return tuple(key for _kid, key in entries)

    def _verify_against_issuer(self, token: str, iss: str, kid: str | None) -> dict:
        signature_error: Exception | None = None
        for key in self._candidate_keys(iss, kid):
            try:
                return jwt.decode(
                    token,
                    key,
                    algorithms=[ALGORITHM],
                    audience=list(self._expected_audiences),
                    issuer=iss,
                    leeway=self._leeway_seconds,
                    options={"require": _REQUIRED_CLAIMS},
                )
            except jwt.InvalidSignatureError as exc:
                # Wrong key among this issuer's set (e.g. mid-rotation) — try the
                # next. Signature is the only key-dependent check.
                signature_error = exc
            except jwt.InvalidTokenError as exc:
                # Signature matched but a claim failed (expired / wrong aud /
                # wrong iss / missing required), or alg != EdDSA — key-independent.
                raise ValueError(f"Token verification failed: {exc}") from exc
        raise ValueError(f"Token signature verification failed: {signature_error}")
