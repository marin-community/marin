"""Probability convention for objectives that consume rollout logprobs."""

import math
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

MIN_NON_GREEDY_TEMPERATURE = 1e-5

ROLLOUT_LOGPROB_VALIDATION_KEY = "validate_rollout_logprob_sampling"
ROLLOUT_LOGPROB_ENGINE_OPTIONS = {"logprobs_mode": "processed_logprobs", "generation_config": "vllm"}

# Trainer recomputation applies temperature and no other sampling processor.
ROLLOUT_LOGPROB_NEUTRAL_SAMPLING = {
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "min_tokens": 0,
}
ROLLOUT_LOGPROB_UNSUPPORTED_PROCESSORS = (
    "logits_processors",
    "logit_bias",
    "allowed_token_ids",
    "bad_words",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
)


def _sampling_mismatches(params: Mapping[str, Any]) -> list[str]:
    mismatches = []
    temperature_value = params.get("temperature", 1.0)
    try:
        temperature = float(temperature_value)
    except (TypeError, ValueError):
        mismatches.append(f"temperature={temperature_value!r} is not numeric")
    else:
        if (
            isinstance(temperature_value, bool)
            or not math.isfinite(temperature)
            or temperature < MIN_NON_GREEDY_TEMPERATURE
        ):
            mismatches.append(
                f"temperature must be finite and >= {MIN_NON_GREEDY_TEMPERATURE:g} for non-greedy sampling"
            )
    for key, neutral in ROLLOUT_LOGPROB_NEUTRAL_SAMPLING.items():
        value = params.get(key)
        if value is not None and value != neutral:
            mismatches.append(f"{key}={value!r} (required value: {neutral!r})")
    for key in ROLLOUT_LOGPROB_UNSUPPORTED_PROCESSORS:
        if params.get(key):
            mismatches.append(f"{key} modifies the sampling distribution")
    response_format = params.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        mismatches.append("constrained response_format")
    if params.get("tool_choice") not in (None, "none", "auto"):
        mismatches.append("constrained tool_choice")
    return mismatches


def validate_behavior_logprob_sampling(params: Mapping[str, Any]) -> None:
    """Require rollout and trainer probabilities to describe one distribution."""
    mismatches = _sampling_mismatches(params)
    if mismatches:
        raise ValueError(
            "Behavior-logprob training requires full-distribution, temperature-only sampling: " + "; ".join(mismatches)
        )


def configure_behavior_logprob_sampling(generator: DictConfig) -> None:
    """Configure serving to return probabilities compatible with the trainer."""
    validate_behavior_logprob_sampling(generator.sampling_params)
    options = generator.engine_init_kwargs
    mismatches = []
    for key, required in ROLLOUT_LOGPROB_ENGINE_OPTIONS.items():
        if key in options and options[key] != required:
            mismatches.append(f"{key}={options[key]!r} (required value: {required!r})")
    if options.get("override_generation_config") or options.get("logits_processors"):
        mismatches.append("engine-level generation overrides or logits processors")
    if mismatches:
        raise ValueError(
            "Behavior-logprob training requires processed rollout logprobs without checkpoint or engine-level "
            "generation overrides: " + "; ".join(mismatches)
        )
    for key, value in ROLLOUT_LOGPROB_NEUTRAL_SAMPLING.items():
        if generator.sampling_params.get(key) is None:
            OmegaConf.update(generator.sampling_params, key, value, force_add=True)
    for key, required in ROLLOUT_LOGPROB_ENGINE_OPTIONS.items():
        if key not in options:
            OmegaConf.update(options, key, required, force_add=True)
    OmegaConf.update(options, ROLLOUT_LOGPROB_VALIDATION_KEY, True, force_add=True)
