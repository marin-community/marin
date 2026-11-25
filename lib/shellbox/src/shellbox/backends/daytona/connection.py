# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Daytona credentials for one account."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Every Marin account runs against the same hosted Daytona API and region.
DAYTONA_API_URL = "https://app.daytona.io/api"
DAYTONA_TARGET = "us"


@dataclass(frozen=True)
class DaytonaConnection:
    """Resolve one explicitly selected Daytona account.

    The SDK otherwise reads ambient environment variables, which can silently
    authenticate against the wrong account. Only the per-account API key
    varies; the API URL and target are module constants. Fields may stay unset
    on hosts that do not use Daytona; creating a client requires the key, so a
    missing connection fails where the account is used instead of falling back
    to ambient credentials. Neither the connection nor its client factory
    exposes the API key through ``repr``, and this module imports without the
    optional Daytona SDK, which is only loaded when a client is created.
    """

    api_key: str | None = field(default=None, repr=False)

    def client_factory(self) -> Callable[[], Any]:
        """Validate the connection and return a callable that builds SDK clients."""
        if not self.api_key:
            raise ValueError("Daytona requires an explicit api_key")
        api_key = self.api_key

        def client() -> Any:
            from daytona import AsyncDaytona, DaytonaConfig  # noqa: PLC0415  # optional SDK, loaded at client creation

            return AsyncDaytona(DaytonaConfig(api_key=api_key, api_url=DAYTONA_API_URL, target=DAYTONA_TARGET))

        return client
