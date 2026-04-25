# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless JWT verification.

Signature and expiry only. Revocation, when needed, is layered on by the
issuer-side caller (e.g. ``iris.cluster.controller.auth.JwtTokenManager``)
and never crosses a process boundary.
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

        identity = VerifiedIdentity(
            user_id=payload["sub"],
            role=payload.get("role", "user"),
        )
        return identity, payload
