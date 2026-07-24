# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control-plane target selection for Rust-streamed federated endpoints."""

from collections.abc import Callable
from dataclasses import dataclass

PeerAddressResolver = Callable[[str], str | None]
FederationTokenMinter = Callable[[], str | None]


@dataclass(frozen=True, slots=True)
class FederatedProxyTarget:
    """Peer URL and short-lived credential returned to the native listener."""

    upstream_url: str
    authorization: str


class FederatedEndpointHandoff:
    """Resolve a mirrored endpoint to its peer and mint the peer credential."""

    def __init__(self, peer_address: PeerAddressResolver, mint_token: FederationTokenMinter) -> None:
        self._peer_address = peer_address
        self._mint_token = mint_token

    def target(
        self,
        *,
        peer_id: str,
        encoded_name: str,
        sub_path: str,
        query: str,
    ) -> FederatedProxyTarget | None:
        """Return the peer proxy target, or ``None`` when the peer is unavailable."""
        base = self._peer_address(peer_id)
        token = self._mint_token()
        if base is None or token is None:
            return None
        upstream_url = f"{base.rstrip('/')}/proxy/{encoded_name}/{sub_path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"
        return FederatedProxyTarget(
            upstream_url=upstream_url,
            authorization=f"Bearer {token}",
        )
