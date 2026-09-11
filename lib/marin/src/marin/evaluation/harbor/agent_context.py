# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The context budget a Harbor agent runs under, derived from how the model is served.

Both the Marin launcher and the isolated Harbor driver import this module, so it depends on the
standard library alone.
"""

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

# The model-catalog field behind each agent limit, named when a policy contradicts it.
_MODEL_CONFIG_FIELD: Mapping[str, str] = MappingProxyType(
    {
        MAX_INPUT_TOKENS_KEY: "serve.max_model_len",
        MAX_OUTPUT_TOKENS_KEY: "generation.max_gen_toks",
    }
)


def served_model_info(max_model_len: int | None, max_gen_toks: int | None) -> dict[str, int]:
    """The Harbor ``model_info`` limits implied by how a model is served.

    A limit the model catalog leaves unset is omitted, so the policy and then Harbor's own default
    still decide it.
    """
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
    """The agent ``model_info`` for one run: served-model limits, policy kwargs, Harbor defaults.

    Raises:
        ValueError: the policy states a context limit the served model contradicts.
    """
    served_info = _model_info_mapping(served, "model")
    policy_info = _model_info_mapping(policy, "agent")
    for key, served_value in served_info.items():
        if key in policy_info and policy_info[key] != served_value:
            raise ValueError(
                f"Harbor agent model_info.{key} is {policy_info[key]!r} but the served model's "
                f"{_MODEL_CONFIG_FIELD.get(key, key)} is {served_value!r}; align the policy with the "
                f"model configuration or drop the agent kwarg"
            )
    return {**DEFAULT_MODEL_INFO, **served_info, **policy_info}
