# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed construction of the optional Daytona SDK client."""

from typing import Any

from marin.daytona.config import DaytonaCredentials


def create_daytona_client(credentials: DaytonaCredentials) -> Any:
    """Create a Daytona SDK client without exposing credentials in caller APIs."""

    try:
        from daytona import Daytona  # noqa: PLC0415
        from daytona import DaytonaConfig as SdkConfig  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Daytona operations require `uv sync --extra daytona`") from exc

    settings: dict[str, str] = {"api_key": credentials.api_key}
    if credentials.config.endpoint:
        settings["api_url"] = credentials.config.endpoint
    if credentials.config.target:
        settings["target"] = credentials.config.target
    return Daytona(SdkConfig(**settings))
