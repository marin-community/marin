"""Normalize OpenAI-compatible chat completion choices before scoring."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from eval.contracts.failures import FailureCategory


class CompletionContentPolicy(StrEnum):
    """Select which completion fields are exposed to benchmark scorers."""

    COMBINE = "combine"
    FINAL_ONLY = "final_only"
    REASONING_ONLY = "reasoning_only"


class CompletionClassification(StrEnum):
    """Describe whether a chat completion contains a usable final answer."""

    FINAL = "final"
    REASONING_AND_FINAL = "reasoning_and_final"
    REASONING_ONLY = "reasoning_only"
    REASONING_ONLY_TRUNCATED = "reasoning_only_truncated"
    EMPTY = "empty"


MISSING_FINAL_CLASSIFICATIONS = frozenset(
    {
        CompletionClassification.REASONING_ONLY,
        CompletionClassification.REASONING_ONLY_TRUNCATED,
        CompletionClassification.EMPTY,
    }
)


@dataclass(frozen=True)
class CompletionResponse:
    """A provider-independent view of an OpenAI-compatible completion choice."""

    content: str | None
    reasoning_content: str | None
    finish_reason: str | None
    usage: Mapping[str, Any] | None
    provider_metadata: Mapping[str, Any]
    raw_choice: Mapping[str, Any]
    failure_category: str | None = None

    @property
    def classification(self) -> CompletionClassification:
        if self.content and self.reasoning_content:
            return CompletionClassification.REASONING_AND_FINAL
        if self.content:
            return CompletionClassification.FINAL
        if self.reasoning_content and self.finish_reason == "length":
            return CompletionClassification.REASONING_ONLY_TRUNCATED
        if self.reasoning_content:
            return CompletionClassification.REASONING_ONLY
        return CompletionClassification.EMPTY

    def normalized_content(self, policy: CompletionContentPolicy | str = CompletionContentPolicy.COMBINE) -> str:
        """Return scorer text without silently discarding reasoning content."""
        selected_policy = CompletionContentPolicy(policy)
        content = self.content or ""
        reasoning = self.reasoning_content or ""
        if selected_policy == CompletionContentPolicy.FINAL_ONLY:
            return content
        if selected_policy == CompletionContentPolicy.REASONING_ONLY:
            return reasoning
        if content and reasoning:
            return f"{reasoning}\n\n{content}"
        return reasoning or content

    def artifact(self, policy: CompletionContentPolicy | str = CompletionContentPolicy.COMBINE) -> dict[str, Any]:
        """Return the auditable response fields for a sample artifact."""
        failure_category = self.failure_category
        if failure_category is None and self.classification in MISSING_FINAL_CLASSIFICATIONS:
            failure_category = FailureCategory.MALFORMED_MODEL_RESPONSE.value
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "provider_metadata": self.provider_metadata,
            "raw_choice": self.raw_choice,
            "classification": self.classification,
            "normalized_content": self.normalized_content(policy),
            "content_policy": CompletionContentPolicy(policy),
            "failure_category": failure_category,
        }


class CompletionText(str):
    """Scorer text that retains the OpenAI response used to produce it."""

    response: CompletionResponse
    content_policy: CompletionContentPolicy

    def __new__(
        cls,
        value: str,
        response: CompletionResponse,
        content_policy: CompletionContentPolicy | str,
    ) -> "CompletionText":
        text = super().__new__(cls, value)
        text.response = response
        text.content_policy = CompletionContentPolicy(content_policy)
        return text

    def artifact(self) -> dict[str, Any]:
        """Return the raw response fields that produced this scorer text."""
        return self.response.artifact(self.content_policy)

    def __getnewargs__(self):
        return str(self), self.response, self.content_policy


class FailedGeneration(str):
    """Empty scorer text carrying a classified endpoint failure."""

    def __new__(cls, category: str) -> "FailedGeneration":
        text = super().__new__(cls, "")
        text.failure_category = category
        return text

    def artifact(self) -> dict[str, Any]:
        return {
            "classification": CompletionClassification.EMPTY,
            "normalized_content": "",
            "failure_category": self.failure_category,
        }

    def __getnewargs__(self):
        return (self.failure_category,)


def completion_response_from_chat_choice(response: Mapping[str, Any], choice: Mapping[str, Any]) -> CompletionResponse:
    """Build a normalized response from an OpenAI chat completion choice."""
    message = choice.get("message", {})
    if not isinstance(message, Mapping):
        raise ValueError("OpenAI chat completion choice has a non-object message.")

    content = message.get("content")
    reasoning_content = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or choice.get("reasoning_content")
        or choice.get("reasoning")
    )
    if content is not None and not isinstance(content, str):
        raise ValueError("OpenAI chat completion content must be a string or null.")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ValueError("OpenAI chat completion reasoning_content must be a string or null.")

    usage = response.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise ValueError("OpenAI chat completion usage must be an object when present.")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("OpenAI chat completion finish_reason must be a string or null.")

    provider_metadata = {key: value for key, value in response.items() if key not in {"choices", "usage"}}
    return CompletionResponse(
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=usage,
        provider_metadata=provider_metadata,
        raw_choice=dict(choice),
    )
