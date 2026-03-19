# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic restart permission dispatch for provider implementations."""

from enum import StrEnum
from collections.abc import Callable

from iris.rpc import config_pb2


class RestartPermissionError(RuntimeError):
    """Raised when the active identity cannot perform restart operations."""


class RestartScope(StrEnum):
    """Restart operation scope."""

    CLUSTER = "cluster"
    CONTROLLER = "controller"


RestartPermissionChecker = Callable[[config_pb2.IrisClusterConfig, RestartScope], None]


def _ensure_gcp_restart_permissions(config: config_pb2.IrisClusterConfig, scope: RestartScope) -> None:
    """Lazily load the GCP checker to keep this module platform-agnostic at import time."""
    from iris.cluster.platform.gcp import ensure_gcp_restart_permissions

    ensure_gcp_restart_permissions(config, scope)


_RESTART_PERMISSION_CHECKERS: dict[str, RestartPermissionChecker] = {
    "gcp": _ensure_gcp_restart_permissions,
}


def register_restart_permission_checker(platform_kind: str, checker: RestartPermissionChecker) -> None:
    """Register or override a restart permission checker for a platform kind."""
    _RESTART_PERMISSION_CHECKERS[platform_kind] = checker


def ensure_restart_permissions(
    config: config_pb2.IrisClusterConfig,
    scope: RestartScope = RestartScope.CLUSTER,
) -> None:
    """Run restart permission checks for the configured platform.

    Platforms without a registered checker are treated as pass-through.
    """
    platform_kind = config.platform.WhichOneof("platform")
    if not platform_kind:
        return

    checker = _RESTART_PERMISSION_CHECKERS.get(platform_kind)
    if checker is None:
        return

    try:
        checker(config, scope)
    except RestartPermissionError:
        # Preserve existing wrappers from provider-specific checkers.
        raise
    except RuntimeError as e:
        raise RestartPermissionError(str(e)) from e
