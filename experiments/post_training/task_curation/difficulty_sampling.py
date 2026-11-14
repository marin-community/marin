# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build recorded settings and provider requests from the same protocol values."""

from typing import Any

from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, SamplingSettings, Solver, ThinkingPolicy


def sampling_settings(solver: Solver, deviations: list[str]) -> SamplingSettings:
    protocol = PROTOCOLS[PROTOCOL_ID]
    context, input_tokens, output_tokens, top_p, top_k, min_p, repetition, presence, frequency = protocol.sampling
    return SamplingSettings(
        context_tokens=int(context),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        temperature=protocol.temperature,
        top_p=top_p,
        top_k=int(top_k),
        min_p=min_p,
        repetition_penalty=repetition,
        presence_penalty=presence,
        frequency_penalty=frequency,
        thinking=protocol.thinking[solver],
        deviations=deviations,
    )


def request_parameters(settings: SamplingSettings) -> dict[str, Any]:
    extra: dict[str, Any] = {
        name: value
        for name, value in {
            "top_k": settings.top_k,
            "min_p": settings.min_p,
            "repetition_penalty": settings.repetition_penalty,
        }.items()
        if value is not None
    }
    if settings.thinking in {ThinkingPolicy.OFF, ThinkingPolicy.ON}:
        extra["chat_template_kwargs"] = {"enable_thinking": settings.thinking == ThinkingPolicy.ON}
    parameters: dict[str, Any] = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "presence_penalty": settings.presence_penalty,
        "frequency_penalty": settings.frequency_penalty,
        "extra_body": extra,
    }
    if settings.thinking == ThinkingPolicy.LOW:
        parameters["reasoning_effort"] = "low"
    return parameters
