# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared platform-aware launch-time environment defaults.

This module provides a single mechanism for applying default environment
variables based on device/platform characteristics while preserving
user-provided overrides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformEnvDefaultRule:
    """One platform default env-var rule.

    Attributes:
        env_var: Environment variable name to default.
        value: Default value to apply if env_var is absent.
        device_kind: Optional required device kind (e.g. "tpu", "gpu", "cpu").
        device_variant_prefixes: Optional variant prefixes that must match.
    """

    env_var: str
    value: str
    device_kind: str | None = None
    device_variant_prefixes: tuple[str, ...] = ()


DEFAULT_PLATFORM_ENV_RULES: tuple[PlatformEnvDefaultRule, ...] = (
    PlatformEnvDefaultRule(
        env_var="JAX_PLATFORMS",
        value="cpu",
        device_kind="cpu",
    ),
    PlatformEnvDefaultRule(
        env_var="JAX_PLATFORMS",
        value="",
        device_kind="tpu",
    ),
    PlatformEnvDefaultRule(
        env_var="JAX_PLATFORMS",
        value="",
        device_kind="gpu",
    ),
    PlatformEnvDefaultRule(
        env_var="LIBTPU_INIT_ARGS",
        value="--xla_tpu_scoped_vmem_limit_kib=50000",
        device_kind="tpu",
        device_variant_prefixes=("v5p-", "v6e-"),
    ),
)


def apply_platform_default_env_vars(
    device: object,
    env_vars: Mapping[str, str],
    *,
    rules: Sequence[PlatformEnvDefaultRule] = DEFAULT_PLATFORM_ENV_RULES,
) -> dict[str, str]:
    """Apply matching platform defaults without overriding user-provided values."""
    merged_env_vars = dict(env_vars)
    device_kind = getattr(device, "kind", None)
    device_variant = getattr(device, "variant", None)

    for rule in rules:
        if rule.device_kind is not None and device_kind != rule.device_kind:
            continue
        if rule.device_variant_prefixes:
            if not isinstance(device_variant, str) or not device_variant.startswith(rule.device_variant_prefixes):
                continue
        merged_env_vars.setdefault(rule.env_var, rule.value)

    return merged_env_vars
