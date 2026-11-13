# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Daytona credentials for one account."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from daytona import AsyncDaytona, DaytonaConfig

DEFAULT_DAYTONA_API_URL = "https://app.daytona.io/api"


@dataclass(frozen=True)
class DaytonaConnection:
    """Resolve one explicitly selected Daytona account.

    The SDK otherwise reads ambient environment variables, which can silently
    authenticate against the wrong account. Fields may stay unset on hosts
    that do not use Daytona; creating a client requires every value, so an
    incomplete connection fails where the account is used instead of falling
    back to ambient credentials. The API key is hidden from ``repr``.
    """

    api_key: str | None = field(default=None, repr=False)
    api_url: str | None = None
    target: str | None = None

    def client_factory(self) -> Callable[[], AsyncDaytona]:
        """Validate the connection and return a callable that builds SDK clients."""
        if not (self.api_key and self.api_url and self.target):
            raise ValueError("Daytona requires an explicit api_key, api_url and target")
        return partial(
            AsyncDaytona,
            DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self.target,
            ),
        )
