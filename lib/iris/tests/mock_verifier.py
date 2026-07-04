# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-only token verifier for iris auth tests."""

from rigging.server_auth import VerifiedIdentity


class MockVerifier:
    """Test-only :class:`~rigging.server_auth.TokenVerifier`: maps fixed tokens to identities.

    A deterministic stand-in for a real service verifier (e.g. the controller's
    JWT manager) in tests that only need a bearer token to resolve to a known
    user. Not a production auth mechanism.

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
