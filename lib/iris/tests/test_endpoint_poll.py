# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.runtime.endpoint_poll import poll_endpoint


@dataclass
class FakeResolver:
    results: list[ResolveResult] = field(default_factory=list)
    call_count: int = 0

    def resolve(self, name: str) -> ResolveResult:
        idx = min(self.call_count, len(self.results) - 1)
        result = self.results[idx]
        self.call_count += 1
        return result


def test_poll_endpoint_default_interval() -> None:
    """Works with the default poll_interval=2.0 (must not crash on ExponentialBackoff)."""
    found = ResolveResult(name="coord", endpoints=[ResolvedEndpoint(url="1.2.3.4:8476", actor_id="ep-1")])
    resolver = FakeResolver(results=[found])
    assert poll_endpoint(resolver, "coord", poll_timeout=10.0) == "1.2.3.4:8476"


def test_poll_endpoint_returns_url() -> None:
    """Returns the url from the first resolved endpoint."""
    found = ResolveResult(name="coord", endpoints=[ResolvedEndpoint(url="1.2.3.4:8476", actor_id="ep-1")])
    resolver = FakeResolver(results=[found])
    assert poll_endpoint(resolver, "coord", poll_interval=0.01, poll_timeout=5.0) == "1.2.3.4:8476"


def test_poll_endpoint_timeout() -> None:
    """Raises TimeoutError when the endpoint never appears within poll_timeout."""
    resolver = FakeResolver(results=[ResolveResult(name="coord", endpoints=[])])
    with pytest.raises(TimeoutError, match="Timed out"):
        poll_endpoint(resolver, "coord", poll_interval=0.01, poll_timeout=0.1)
