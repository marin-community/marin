# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Daytona credentials for one account."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DaytonaConnection:
    """Resolve one explicitly selected Daytona account.

    The SDK otherwise reads ``DAYTONA_API_KEY`` from the environment, which can
    silently authenticate against the wrong account. The key may stay unset on
    hosts that do not use Daytona; creating a client requires it, so an
    incomplete connection fails where the account is used instead of falling
    back to ambient credentials. Neither the connection nor its client factory
    exposes the API key through ``repr``, and this module imports without the
    optional Daytona SDK, which is only loaded when a client is created.

    The v2 API has no client-side region or endpoint setting: sandboxes and
    images use the project's default region, and only ``DAYTONA_V2_API_URL``
    overrides the endpoint.
    """

    api_key: str | None = field(default=None, repr=False)

    def client_factory(self) -> Callable[[], Any]:
        """Validate the connection and return a callable that builds v2 SDK clients."""
        if not self.api_key:
            raise ValueError("Daytona requires an explicit api_key")
        api_key = self.api_key

        def client() -> Any:
            from daytona import AsyncDaytona, DaytonaConfig  # noqa: PLC0415  # optional SDK, loaded at client creation
            from daytona.v2 import _transport  # noqa: PLC0415  # optional SDK, loaded at client creation

            # Over HTTP/2 the sandbox proxy stalls stdin writes to a process whose log the
            # client follows on the same connection, which hangs the HTTP relay. The SDK
            # enables HTTP/2 whenever h2 is importable and offers no setting, so turn it off.
            _transport._HTTP2_AVAILABLE = False
            return AsyncDaytona(DaytonaConfig(api_key=api_key)).v2

        return client
