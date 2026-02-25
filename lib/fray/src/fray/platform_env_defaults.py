# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared platform-aware launch-time environment defaults.

Apply default environment variables based on device configuration while
preserving user-provided overrides. Each DeviceConfig class (CpuConfig,
GpuConfig, TpuConfig) provides its own defaults via ``default_env_vars()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class _HasDefaultEnvVars(Protocol):
    def default_env_vars(self) -> dict[str, str]: ...


def apply_platform_default_env_vars(
    device: _HasDefaultEnvVars,
    env_vars: Mapping[str, str],
) -> dict[str, str]:
    """Apply device-specific env-var defaults without overriding user-provided values."""
    merged = dict(env_vars)
    for key, value in device.default_env_vars().items():
        merged.setdefault(key, value)
    return merged
