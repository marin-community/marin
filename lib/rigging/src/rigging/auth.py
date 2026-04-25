# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless JWT verification.

Issuance and DB-backed revocation hydration live in
``iris.cluster.controller.auth``; the verifier only does signature/expiry checks
plus an in-memory revocation set lookup.
"""

from dataclasses import dataclass

import jwt

JWT_ALGORITHM = "HS256"


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    user_id: str
    role: str


class JwtVerifier:
    def __init__(self, signing_key: str):
        self._signing_key = signing_key
        self._revoked_jtis: set[str] = set()

    @property
    def signing_key(self) -> str:
        return self._signing_key

    def verify(self, token: str) -> VerifiedIdentity:
        return self.verify_full(token)[0]

    def verify_full(self, token: str) -> tuple[VerifiedIdentity, dict]:
        try:
            payload = jwt.decode(token, self._signing_key, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid token: {exc}") from exc

        if payload.get("jti", "") in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        identity = VerifiedIdentity(
            user_id=payload["sub"],
            role=payload.get("role", "user"),
        )
        return identity, payload

    def revoke(self, jti: str) -> None:
        self._revoked_jtis.add(jti)

    def set_revocations(self, jtis: set[str]) -> None:
        self._revoked_jtis = set(jtis)
