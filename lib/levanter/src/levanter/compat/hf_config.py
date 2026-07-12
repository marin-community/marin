# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, TypeVar, cast

ConfigT = TypeVar("ConfigT")


def hf_config_from_kwargs(config_cls: type[ConfigT], **kwargs: Any) -> ConfigT:
    """Construct HF configs with runtime kwargs that generated stubs may omit."""
    return cast(Any, config_cls)(**kwargs)


def hf_rope_parameter_mapping(hf_config: Any, *, include_rope_scaling: bool = False) -> Mapping[str, Any] | None:
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        return rope_parameters

    if include_rope_scaling:
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, Mapping):
            return rope_scaling

    return None


def hf_rope_config(
    hf_config: Any,
    default_theta: float = 10_000.0,
    *,
    rope_parameter_keys: tuple[str, ...] = (),
) -> tuple[float, Any | None]:
    rope_theta = getattr(hf_config, "rope_theta", None)
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_parameters = hf_rope_parameter_mapping(hf_config)

    if rope_parameter_keys and rope_theta is None and rope_parameters is not None:
        rope_theta = rope_parameters.get("rope_theta")
        selected_rope_config = None
        for key in rope_parameter_keys:
            if rope_theta is not None:
                break
            attention_rope = rope_parameters.get(key)
            if isinstance(attention_rope, Mapping):
                rope_theta = attention_rope.get("rope_theta")
                selected_rope_config = attention_rope

        if selected_rope_config is not None:
            rope_scaling = selected_rope_config
        elif rope_scaling is None:
            rope_scaling = rope_parameters
    elif rope_parameters is not None:
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta")
        if rope_scaling is None:
            rope_scaling = rope_parameters

    if rope_theta is None:
        rope_theta = default_theta

    return float(rope_theta), rope_scaling
