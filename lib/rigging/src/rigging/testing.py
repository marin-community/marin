# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support doubles for services built on rigging's auth stack.

Shipped alongside the library (not under ``tests/``) so both rigging's own suite
and downstream consumers such as iris import one copy rather than each carrying
its own.
"""

from rigging.server_auth import VerifiedIdentity


class MockVerifier:
    """Deterministic :class:`~rigging.server_auth.TokenVerifier` double: maps fixed tokens to identities.

    A stand-in for a real service verifier (which checks JWTs it signed) in tests
    that only need a bearer token to resolve to a known user. Not a production auth
    mechanism.

    Args:
        tokens: Mapping of token string to username. Every user gets role ``"user"``.
    """

    def __init__(self, tokens: dict[str, str]):
        self._tokens = tokens

    def verify(self, token: str) -> VerifiedIdentity:
        user = self._tokens.get(token)
        if user is None:
            raise ValueError("Invalid token")
        return VerifiedIdentity(user_id=user, role="user")
