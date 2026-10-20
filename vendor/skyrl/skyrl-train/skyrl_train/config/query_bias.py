# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration semantics for Grug router query-bias updates."""

import math
from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig


class GrugQueryBiasUpdateMode(StrEnum):
    """Policy for updating Grug router query-bias buffers during RL."""

    FROZEN = "frozen"
    INTERPOLATE = "interpolate"
    LOSS_FREE = "loss_free"
    REPLACE = "replace"


@dataclass(frozen=True)
class GrugQueryBiasUpdate:
    """Validated parameters for one Grug query-bias update mechanism."""

    mode: GrugQueryBiasUpdateMode
    interpolation_weight: float | None = None
    update_rate: float | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not GrugQueryBiasUpdateMode.FROZEN


def _finite_number(raw_value: object, config_key: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{config_key} must be a number; got {raw_value}") from error
    if not math.isfinite(value):
        raise ValueError(f"{config_key} must be finite; got {raw_value}")
    return value


def resolve_grug_query_bias_update(policy_config: DictConfig) -> GrugQueryBiasUpdate:
    """Validate and resolve the selected Grug query-bias update."""

    mode_key = "grug_query_bias_update_mode"
    if mode_key not in policy_config:
        raise ValueError(f"missing required policy configuration: {mode_key}")
    raw_mode = policy_config[mode_key]
    try:
        mode = GrugQueryBiasUpdateMode(raw_mode)
    except ValueError as error:
        valid_modes = tuple(item.value for item in GrugQueryBiasUpdateMode)
        raise ValueError(f"invalid {mode_key}: {raw_mode}. Must be one of {valid_modes}") from error

    interpolation_key = "grug_query_bias_interpolation_weight"
    raw_interpolation_weight = policy_config.get(interpolation_key)
    rate_key = "grug_query_bias_update_rate"
    raw_update_rate = policy_config.get(rate_key)

    if mode is GrugQueryBiasUpdateMode.INTERPOLATE:
        if raw_interpolation_weight is None:
            raise ValueError(f"{interpolation_key} is required when {mode_key}=interpolate")
        interpolation_weight = _finite_number(raw_interpolation_weight, interpolation_key)
        if not 0.0 < interpolation_weight < 1.0:
            raise ValueError(f"{interpolation_key} must be between 0 and 1 exclusively; got {raw_interpolation_weight}")
    else:
        if raw_interpolation_weight is not None:
            raise ValueError(f"{interpolation_key} is only valid when {mode_key}=interpolate")
        interpolation_weight = None

    if mode is GrugQueryBiasUpdateMode.LOSS_FREE:
        if raw_update_rate is None:
            raise ValueError(f"{rate_key} is required when {mode_key}=loss_free")
        update_rate = _finite_number(raw_update_rate, rate_key)
        if update_rate <= 0.0:
            raise ValueError(f"{rate_key} must be positive; got {raw_update_rate}")
    else:
        if raw_update_rate is not None:
            raise ValueError(f"{rate_key} is only valid when {mode_key}=loss_free")
        update_rate = None

    return GrugQueryBiasUpdate(
        mode=mode,
        interpolation_weight=interpolation_weight,
        update_rate=update_rate,
    )
