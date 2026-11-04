# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only backend capabilities supplied at an executable entry point."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from harbor.utils.env import protected_environment_values

DAYTONA_API_URL = "https://app.daytona.io/api"
OPENAI_API_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class DaytonaConnection:
    api_key: str | None = field(repr=False)
    api_url: str
    target: str | None


@dataclass(frozen=True)
class HostCapabilities:
    """Host authority passed outside serialized job and task configuration."""

    daytona: DaytonaConnection | None = field(default=None, repr=False)
    model_authentication: Mapping[str, str] = field(default_factory=dict, repr=False)
    protected_values: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_authentication",
            MappingProxyType(dict(self.model_authentication)),
        )


def host_capabilities_from_environment(
    environment: Mapping[str, str],
    *,
    model_authentication: Mapping[str, str],
) -> HostCapabilities:
    """Resolve backend and model authority from one executable-owned snapshot."""
    return HostCapabilities(
        daytona=DaytonaConnection(
            api_key=environment.get("DAYTONA_API_KEY"),
            api_url=environment.get("DAYTONA_API_URL", DAYTONA_API_URL),
            target=environment.get("DAYTONA_TARGET"),
        ),
        model_authentication=model_authentication,
        protected_values=protected_environment_values(environment)
        | frozenset(
            value
            for value in model_authentication.values()
            if value and value != "EMPTY"
        ),
    )
