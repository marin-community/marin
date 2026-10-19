# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateless completion normalization and endpoint payload formatting."""

from __future__ import annotations

import base64
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from lm_eval.models.openai_completions import (
    LocalChatCompletion,
    LocalCompletionsAPI,
    OpenAIChatCompletion,
    OpenAICompletionsAPI,
)
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import simple_parse_args_string

from eval.completion_response import (
    MISSING_FINAL_CLASSIFICATIONS,
    CompletionClassification,
    CompletionContentPolicy,
    CompletionResponse,
    CompletionText,
    completion_response_from_chat_choice,
)
from eval.contracts.failures import FailureCategory
from eval.generation_stops import bounded_request_stops, chat_request_stops

logger = logging.getLogger("eval.robust_api")

_COMPLETION_PATCH_FLAG = "_marin_completion_normalization_patched"
_OPENAI_PAYLOAD_PATCH_FLAG = "_marin_openai_payload_patched"
_CHAT_TEMPLATE_KWARGS_ATTR = "_evalchemy_chat_template_kwargs"
_EXTRA_BODY_ATTR = "_evalchemy_extra_body"
_CONTEXT_OUTPUT_CAP_KWARG = "_evalchemy_context_output_cap"
_ENCODED_MAPPING_PREFIX = "base64:"
_INVALID_RESPONSE_FRACTION = 0.5
_OPENAI_FIXED_GENERATION_MODEL = re.compile(r"^(?:gpt-5|o[134])(?:$|[-.])", re.IGNORECASE)
def completion_response_quality_invalid(classifications: Counter[CompletionClassification]) -> bool:
    """Return whether a generation run has too many unusable chat completions."""
    total = sum(classifications.values())
    missing_final = sum(classifications[classification] for classification in MISSING_FINAL_CLASSIFICATIONS)
    return total > 0 and missing_final / total >= _INVALID_RESPONSE_FRACTION


def completion_response_text_unavailable(classifications: Counter[CompletionClassification]) -> bool:
    """Return whether at least half of responses contain no scorer text."""
    total = sum(classifications.values())
    return total > 0 and classifications[CompletionClassification.EMPTY] / total >= _INVALID_RESPONSE_FRACTION


def openai_model_requires_fixed_generation(model: object) -> bool:
    """Return whether an official OpenAI model rejects stops and temperature zero."""
    return isinstance(model, str) and bool(_OPENAI_FIXED_GENERATION_MODEL.match(model))


def configure_generation_overrides(model: object, overrides: dict) -> None:
    """Retain caller settings separately from source defaults for API adapters."""
    if isinstance(model, LocalCompletionsAPI):
        model._evalchemy_generation_overrides = dict(overrides)


def configured_generation_overrides(model: object) -> Mapping[str, Any]:
    """Return explicit caller settings, or an empty mapping before configuration."""
    return getattr(model, "_evalchemy_generation_overrides", {})


def parse_generation_overrides(value: str | dict | None) -> dict:
    """Parse caller settings and make temperature decisive for local decoders."""
    overrides = simple_parse_args_string(value) if isinstance(value, str) else dict(value or {})
    if "temperature" in overrides and "do_sample" not in overrides:
        overrides["do_sample"] = float(overrides["temperature"]) > 0
    return overrides


def _parse_request_body_mapping(value: str | Mapping | None, name: str) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str) and value.startswith(_ENCODED_MAPPING_PREFIX):
        value = base64.urlsafe_b64decode(value.removeprefix(_ENCODED_MAPPING_PREFIX)).decode("utf-8")
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(parsed)


def apply_completion_normalization() -> bool:
    """Normalize each choice without retaining completed responses on the model."""
    if getattr(LocalChatCompletion, _COMPLETION_PATCH_FLAG, False):
        return True
    original_init = LocalChatCompletion.__init__

    def __init__(self, *args, completion_content_policy="combine", **kwargs):
        self.completion_content_policy = CompletionContentPolicy(completion_content_policy)
        original_init(self, *args, **kwargs)

    def parse_generations(self, outputs, **kwargs):
        batches = outputs if isinstance(outputs, list) else [outputs]
        generated = []
        for output in batches:
            try:
                choices = sorted(output["choices"], key=lambda choice: choice["index"])
                if not choices:
                    raise ValueError("completion contains no choices")
                responses = [completion_response_from_chat_choice(output, choice) for choice in choices]
            except (IndexError, KeyError, TypeError, ValueError):
                responses = [CompletionResponse(
                    content=None, reasoning_content=None, finish_reason=None, usage=None,
                    provider_metadata={}, raw_choice={}, failure_category=FailureCategory.MALFORMED_MODEL_RESPONSE.value,
                )]
            for response in responses:
                generated.append(CompletionText(response.normalized_content(self.completion_content_policy),
                                                response, self.completion_content_policy))
        return generated

    LocalChatCompletion.__init__ = __init__
    LocalChatCompletion.parse_generations = parse_generations
    setattr(LocalChatCompletion, _COMPLETION_PATCH_FLAG, True)
    return True


def apply_openai_payload_controls() -> bool:
    """Bound API stops and restrict OpenAI-specific controls to OpenAI models."""
    if (
        getattr(LocalCompletionsAPI, _OPENAI_PAYLOAD_PATCH_FLAG, False)
        and getattr(LocalChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, False)
        and getattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, False)
    ):
        return True

    original_completions_payload = LocalCompletionsAPI._create_payload
    original_local_chat_init = LocalChatCompletion.__init__
    original_local_chat_payload = LocalChatCompletion._create_payload
    original_openai_chat_payload = OpenAIChatCompletion._create_payload

    def _local_chat_init(self, *args, chat_template_kwargs=None, extra_body=None, **kwargs):
        setattr(
            self,
            _CHAT_TEMPLATE_KWARGS_ATTR,
            _parse_request_body_mapping(chat_template_kwargs, "chat_template_kwargs"),
        )
        setattr(self, _EXTRA_BODY_ATTR, _parse_request_body_mapping(extra_body, "extra_body"))
        original_local_chat_init(self, *args, **kwargs)

    def _merged_stop_kwargs(gen_kwargs, eos, default_until=None):
        request_kwargs = dict(gen_kwargs or {})
        until = request_kwargs.get("until", default_until)
        stop = handle_stop_sequences(list(until) if isinstance(until, list) else until, eos)
        return request_kwargs, stop

    def _bounded_generation_kwargs(gen_kwargs, eos, default_until=None):
        request_kwargs, stop = _merged_stop_kwargs(gen_kwargs, eos, default_until)
        request_kwargs["until"] = bounded_request_stops(stop)
        return request_kwargs

    def _chat_generation_kwargs(gen_kwargs, eos):
        # Chat servers apply stop strings to reasoning content, and the turn
        # ends at EOS anyway, so only tokenizer sentinels may cross the wire.
        request_kwargs, stop = _merged_stop_kwargs(gen_kwargs, eos)
        request_kwargs["until"] = chat_request_stops(stop)
        return request_kwargs

    def _apply_context_output_cap(payload, context_output_cap):
        if context_output_cap is None:
            return payload
        token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
        payload[token_key] = min(payload.get(token_key, context_output_cap), context_output_cap)
        return payload

    def _caller_generation_payload(self, payload):
        if isinstance(self, (OpenAICompletionsAPI, OpenAIChatCompletion)):
            return payload
        overrides = getattr(self, "_evalchemy_generation_overrides", None)
        if overrides is None:
            return payload
        # Source and adapter defaults must not override the serving model's
        # sampling policy. Explicit caller controls still cross the wire.
        for key in ("temperature", "seed"):
            payload.pop(key, None)
        for key, value in overrides.items():
            if key == "max_gen_toks":
                token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                payload[token_key] = value
            elif key not in ("do_sample", "until"):
                payload[key] = value
        if overrides.get("do_sample") is False and "temperature" not in overrides:
            payload["temperature"] = 0
        return payload

    def _finalize_generation_payload(self, payload, context_output_cap):
        payload = _caller_generation_payload(self, payload)
        return _apply_context_output_cap(payload, context_output_cap)

    def _generation_kwargs_and_context_cap(gen_kwargs):
        request_kwargs = dict(gen_kwargs or {})
        return request_kwargs, request_kwargs.pop(_CONTEXT_OUTPUT_CAP_KWARG, None)

    def _create_local_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos=None,
        **kwargs,
    ):
        gen_kwargs, context_output_cap = _generation_kwargs_and_context_cap(gen_kwargs)
        payload = original_local_chat_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=_chat_generation_kwargs(gen_kwargs, eos),
            seed=seed,
            eos=None,
            **kwargs,
        )
        if not payload.get("stop"):
            # No sentinels to send: the endpoint ends the turn at EOS.
            payload.pop("stop", None)
        payload.update(getattr(self, _EXTRA_BODY_ATTR, None) or {})
        chat_template_kwargs = getattr(self, _CHAT_TEMPLATE_KWARGS_ATTR, None)
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        return _finalize_generation_payload(self, payload, context_output_cap)

    def _create_completions_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos=None,
        **kwargs,
    ):
        gen_kwargs, context_output_cap = _generation_kwargs_and_context_cap(gen_kwargs)
        if not generate:
            payload = original_completions_payload(
                self,
                messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=seed,
                eos=eos,
                **kwargs,
            )
            return _finalize_generation_payload(self, payload, context_output_cap)
        payload = original_completions_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=_bounded_generation_kwargs(gen_kwargs, eos),
            seed=seed,
            eos=None,
            **kwargs,
        )
        return _finalize_generation_payload(self, payload, context_output_cap)

    def _create_openai_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ):
        gen_kwargs, context_output_cap = _generation_kwargs_and_context_cap(gen_kwargs)
        request_kwargs = _bounded_generation_kwargs(gen_kwargs, eos, ["<|endoftext|>"])
        selected_stops = list(request_kwargs["until"])
        temperature = request_kwargs.get("temperature", 0)
        payload = original_openai_chat_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=request_kwargs,
            seed=seed,
            eos=None,
            **kwargs,
        )
        if openai_model_requires_fixed_generation(self.model):
            payload.pop("stop", None)
            payload["temperature"] = 1
        else:
            payload["stop"] = selected_stops
            payload["temperature"] = temperature
        return _finalize_generation_payload(self, payload, context_output_cap)

    LocalCompletionsAPI._create_payload = _create_completions_payload
    LocalChatCompletion.__init__ = _local_chat_init
    LocalChatCompletion._create_payload = _create_local_payload
    OpenAIChatCompletion._create_payload = _create_openai_payload
    setattr(LocalCompletionsAPI, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    setattr(LocalChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    setattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    logger.info("OpenAI payload controls: patched bounded stop selection and GPT-5 family matching.")
    return True


# Apply on import so `from eval import robust_api` is enough to activate the patch.
_COMPLETION_NORMALIZATION_APPLIED = apply_completion_normalization()
_OPENAI_PAYLOAD_CONTROLS_APPLIED = apply_openai_payload_controls()
