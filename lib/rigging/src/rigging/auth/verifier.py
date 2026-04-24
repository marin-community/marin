# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless JWT verification.

`JwtVerifier` performs HMAC-SHA256 signature/expiry checks plus an in-memory
revocation set lookup. It has no database handle and no persistent state —
processes that only need to validate tokens (e.g. the log server) can
construct one from a signing key alone.

Token issuance and DB-backed revocation hydration live alongside the
controller (see `iris.cluster.controller.auth`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt

JWT_ALGORITHM = "HS256"


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Identity of an authenticated caller, extracted from JWT claims."""

    user_id: str
    role: str


class JwtVerifier:
    """Verifies HMAC-SHA256 JWT tokens against an in-memory revocation set.

    Verification is a pure crypto check followed by a set lookup — no I/O,
    no DB, no network. Callers that mint tokens (the controller) populate
    the revocation set via `revoke(jti)` or by replaying DB rows through
    `set_revocations(jtis)`; the verifier itself never touches the DB.
    """

    def __init__(self, signing_key: str):
        self._signing_key = signing_key
        self._revoked_jtis: set[str] = set()

    @property
    def signing_key(self) -> str:
        """HMAC secret used to verify JWTs. Do not log or serialize."""
        return self._signing_key

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify JWT signature and expiry, then check the revocation set.

        Raises:
            ValueError: if the signature is invalid, the token is expired,
                or its `jti` is in the revocation set.
        """
        return self.verify_full(token)[0]

    def verify_full(self, token: str) -> tuple[VerifiedIdentity, dict]:
        """Like `verify`, but also returns the decoded claims dict.

        Useful for issuers that need the `jti` (or other claims) for
        write-back side effects without re-decoding.
        """
        try:
            payload = jwt.decode(token, self._signing_key, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid token: {exc}") from exc

        jti = payload.get("jti", "")
        if jti in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        identity = VerifiedIdentity(
            user_id=payload["sub"],
            role=payload.get("role", "user"),
        )
        return identity, payload

    def revoke(self, jti: str) -> None:
        """Add a JTI to the in-memory revocation set."""
        self._revoked_jtis.add(jti)

    def set_revocations(self, jtis: set[str]) -> None:
        """Replace the revocation set. Used to rehydrate from a DB on startup."""
        self._revoked_jtis = set(jtis)

    def is_revoked(self, jti: str) -> bool:
        return jti in self._revoked_jtis
