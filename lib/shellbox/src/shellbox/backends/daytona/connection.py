# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Daytona credentials for one account."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DAYTONA_API_URL = "https://app.daytona.io/api"


@dataclass(frozen=True)
class DaytonaConnection:
    """Resolve one explicitly selected Daytona account.

    The SDK otherwise reads ambient environment variables, which can silently
    authenticate against the wrong account. Fields may stay unset on hosts
    that do not use Daytona; creating a client requires every value, so an
    incomplete connection fails where the account is used instead of falling
    back to ambient credentials. Neither the connection nor its client
    factory exposes the API key through ``repr``, and this module imports
    without the optional Daytona SDK, which is only loaded when a client is
    created.
    """

    api_key: str | None = field(default=None, repr=False)
    api_url: str | None = None
    target: str | None = None

    def client_factory(self) -> Callable[[], Any]:
        """Validate the connection and return a callable that builds SDK clients."""
        if not (self.api_key and self.api_url and self.target):
            raise ValueError("Daytona requires an explicit api_key, api_url and target")
        api_key, api_url, target = self.api_key, self.api_url, self.target

        def client() -> Any:
            from daytona import AsyncDaytona, DaytonaConfig  # noqa: PLC0415  # optional SDK, loaded at client creation

            return AsyncDaytona(DaytonaConfig(api_key=api_key, api_url=api_url, target=target))

        return client
