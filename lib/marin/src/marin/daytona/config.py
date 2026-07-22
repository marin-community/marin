# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-secret configuration for Daytona clients."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DaytonaConfig:
    """Connection settings whose API key lives in the process environment.

    Args:
        endpoint: Daytona API endpoint, or ``None`` to use the SDK default.
        target: Optional Daytona target/region selected by a caller.
        api_key_env: Name of the environment variable containing the API key.
    """

    endpoint: str | None = None
    target: str | None = None
    api_key_env: str = "DAYTONA_API_KEY"

    def __post_init__(self) -> None:
        if not self.api_key_env or "=" in self.api_key_env:
            raise ValueError("api_key_env must be a non-empty environment variable name")


@dataclass(frozen=True)
class DaytonaCredentials:
    """Authenticated Daytona connection settings.

    This value must not be rendered in logs or CLI output.
    """

    config: DaytonaConfig
    api_key: str


def resolve_daytona_credentials(config: DaytonaConfig, environ: dict[str, str] | None = None) -> DaytonaCredentials:
    """Read the configured Daytona API key from the current process environment."""

    values = os.environ if environ is None else environ
    api_key = values.get(config.api_key_env)
    if not api_key:
        raise ValueError(f"Daytona API key is not set in ${config.api_key_env}")
    return DaytonaCredentials(config=config, api_key=api_key)
