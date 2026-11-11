# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Daytona credentials for one account."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from daytona import AsyncDaytona, DaytonaConfig

DEFAULT_DAYTONA_API_URL = "https://app.daytona.io/api"


@dataclass(frozen=True)
class DaytonaConnection:
    """Resolve one explicitly selected Daytona account.

    The SDK otherwise reads ambient environment variables, which can silently
    authenticate against the wrong account. Every field is required and passed
    to the SDK directly.
    """

    api_key: str
    api_url: str
    target: str

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_url or not self.target:
            raise ValueError("Daytona requires an explicit api_key, api_url and target")

    def client_factory(self) -> Callable[[], AsyncDaytona]:
        return partial(
            AsyncDaytona,
            DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self.target,
            ),
        )
