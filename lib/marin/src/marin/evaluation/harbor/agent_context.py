# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor context limits without dependencies outside the standard library."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

MODEL_INFO_KEY = "model_info"
MAX_INPUT_TOKENS_KEY = "max_input_tokens"
MAX_OUTPUT_TOKENS_KEY = "max_output_tokens"

# Harbor's own budgets, applied when neither the model catalog nor the policy states a limit.
DEFAULT_MODEL_INFO: Mapping[str, Any] = MappingProxyType(
    {
        MAX_INPUT_TOKENS_KEY: 32768,
        MAX_OUTPUT_TOKENS_KEY: 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
)

# Model-catalog fields named in policy-limit errors.
_MODEL_CONFIG_FIELD: Mapping[str, str] = MappingProxyType(
    {
        MAX_INPUT_TOKENS_KEY: "serve.max_model_len",
        MAX_OUTPUT_TOKENS_KEY: "generation.max_gen_toks",
    }
)


def served_model_info(max_model_len: int | None, max_gen_toks: int | None) -> dict[str, int]:
    """Omit unset catalog limits so the policy or Harbor defaults can supply them."""
    model_info: dict[str, int] = {}
    if max_model_len is not None:
        model_info[MAX_INPUT_TOKENS_KEY] = max_model_len
    if max_gen_toks is not None:
        model_info[MAX_OUTPUT_TOKENS_KEY] = max_gen_toks
    return model_info


def _model_info_mapping(value: object, source: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Harbor {source} model_info must be a mapping")
    return value


def reconciled_model_info(served: object, policy: object) -> dict[str, Any]:
    """Reconcile policy limits with the model catalog and Harbor defaults.

    Policy limits may reserve headroom below served limits but cannot exceed them.

    Raises:
        ValueError: A policy context limit exceeds the corresponding served limit.
    """
    served_info = _model_info_mapping(served, "model")
    policy_info = _model_info_mapping(policy, "agent")
    for key, served_value in served_info.items():
        if key not in policy_info:
            continue
        policy_value = policy_info[key]
        model_field = _MODEL_CONFIG_FIELD[key]
        if not isinstance(policy_value, int):
            raise ValueError(f"Harbor agent model_info.{key} must be an integer, got {policy_value!r}")
        if policy_value > served_value:
            raise ValueError(
                f"Harbor agent model_info.{key} is {policy_value} but the served model's "
                f"{model_field} is only {served_value}; lower the policy limit or raise {model_field}"
            )
    return {**DEFAULT_MODEL_INFO, **served_info, **policy_info}
