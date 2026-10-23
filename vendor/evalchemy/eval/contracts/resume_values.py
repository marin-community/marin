"""Lossless JSON values for Evalchemy resume manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eval.completion_response import CompletionContentPolicy, CompletionResponse, CompletionText, FailedGeneration

_RESUME_VALUE_TYPE = "__evalchemy_resume_value__"
_COMPLETION_TEXT = "completion_text"
_FAILED_GENERATION = "failed_generation"


def encode_resume_value(value: Any) -> Any:
    """Encode string subclasses whose metadata affects scoring and coverage."""
    if isinstance(value, CompletionText):
        response = value.response
        return {
            _RESUME_VALUE_TYPE: _COMPLETION_TEXT,
            "value": str(value),
            "content_policy": value.content_policy.value,
            "response": {
                "content": response.content,
                "reasoning_content": response.reasoning_content,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "provider_metadata": response.provider_metadata,
                "raw_choice": response.raw_choice,
                "failure_category": response.failure_category,
            },
        }
    if isinstance(value, FailedGeneration):
        return {
            _RESUME_VALUE_TYPE: _FAILED_GENERATION,
            "failure_category": value.failure_category,
        }
    if isinstance(value, Mapping):
        return {key: encode_resume_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [encode_resume_value(item) for item in value]
    return value


def decode_resume_value(value: Any) -> Any:
    """Restore scoring-bearing string subclasses from a resume manifest."""
    if isinstance(value, Mapping):
        value_type = value.get(_RESUME_VALUE_TYPE)
        if value_type == _FAILED_GENERATION:
            return FailedGeneration(str(value["failure_category"]))
        if value_type == _COMPLETION_TEXT:
            response_value = value["response"]
            if not isinstance(response_value, Mapping):
                raise ValueError("completion_text resume payload has a non-object response")
            response = CompletionResponse(
                content=response_value.get("content"),
                reasoning_content=response_value.get("reasoning_content"),
                finish_reason=response_value.get("finish_reason"),
                usage=response_value.get("usage"),
                provider_metadata=response_value.get("provider_metadata", {}),
                raw_choice=response_value.get("raw_choice", {}),
                failure_category=response_value.get("failure_category"),
            )
            return CompletionText(
                str(value["value"]),
                response,
                CompletionContentPolicy(str(value["content_policy"])),
            )
        return {key: decode_resume_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_resume_value(item) for item in value]
    return value
