"""Mini-SWE 2.1 models installed alongside vllm_tokenization in the sandbox."""

import json
import os
from typing import Any

import httpx
import openai
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.openrouter_model import OpenRouterModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from openai import OpenAI
from vllm_tokenization import ContextBudgetExceeded, completion_budget

_SUMMARY_SOURCE_MULTIPLIER = 4
_SUMMARY_OUTPUT_TOKENS = 1024
_SUMMARY_SYSTEM_PROMPT = """You compact the work history of a software-engineering agent.
Preserve concrete facts needed to continue: files inspected or changed, commands and their
important outputs, test results, hypotheses, unresolved errors, and the next intended action.
Do not invent facts. Return only the compact handoff summary."""
_SUMMARY_MESSAGE_PREFIX = "Earlier work was compacted into this handoff summary:\n"


def _provider_message(message: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key != "extra"}


def _message_characters(message: dict[str, Any]) -> int:
    return len(json.dumps(_provider_message(message), sort_keys=True, default=str))


def _history_characters(messages: list[dict[str, Any]]) -> int:
    return sum(_message_characters(message) for message in messages)


def _turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _bounded_source(text: str, limit: int) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    marker = "\n\n[... source characters omitted ...]\n\n"
    retained = max(0, limit - len(marker))
    head = retained // 2
    tail = retained - head
    omitted = len(text) - retained
    return (
        f"{text[:head]}{marker}{text[-tail:] if tail else ''}"[:limit],
        omitted,
    )


def _render_summary_source(messages: list[dict[str, Any]]) -> str:
    rendered = []
    for message in messages:
        rendered.append(
            f"[{message.get('role', 'unknown')}]\n"
            f"{json.dumps(_provider_message(message), sort_keys=True, default=str)}"
        )
    return "\n\n".join(rendered)


def _summary_message(text: str, character_budget: int) -> dict[str, Any]:
    content = f"{_SUMMARY_MESSAGE_PREFIX}{text}"
    message = {"role": "user", "content": content}
    while content and _message_characters(message) > character_budget:
        excess = _message_characters(message) - character_budget
        content = content[: -max(1, excess)]
        message = {"role": "user", "content": content}
    return message


class OpenAIModel(OpenRouterModel):
    """Use Mini-SWE's tool handling with an OpenAI-compatible transport."""

    abort_exceptions = [
        openai.AuthenticationError,
        openai.NotFoundError,
        openai.PermissionDeniedError,
    ]

    def __init__(self, *, api_base: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_base = (
            api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        if (
            not self.api_base
            and "/" in self.config.model_name
            and not self.config.model_name.startswith("openai/")
        ):
            raise ValueError(
                "Configure an OpenAI-compatible api_base for this provider"
            )

    def _completion(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        with OpenAI(
            base_url=self.api_base,
            api_key=os.environ.get("MSWEA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY",
            max_retries=0,
        ) as client:
            model = self.config.model_name
            if model.startswith(("hosted_vllm/", "openai/")):
                model = model.split("/", 1)[1]
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            ).model_dump()

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return self._completion(
            messages, tools=[BASH_TOOL], **(self.config.model_kwargs | kwargs)
        )


class ContextLimitedModel(OpenAIModel):
    """Preserve native actions and stop cleanly when the context is exhausted."""

    def __init__(
        self,
        *,
        api_base: str,
        context_window: int,
        max_output_tokens: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_base=api_base, **kwargs)
        self.abort_exceptions = [*OpenAIModel.abort_exceptions, LimitsExceeded]
        self.api_base = api_base
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens

    def serialize(self) -> dict[str, Any]:
        result = super().serialize()
        result["info"]["config"]["model"].update(
            api_base=self.api_base,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
        )
        return result

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        request_kwargs = self.config.model_kwargs | kwargs
        try:
            with httpx.Client(timeout=60.0) as client:
                max_tokens = completion_budget(
                    client,
                    api_base=self.api_base,
                    model=self.config.model_name.removeprefix("hosted_vllm/"),
                    messages=messages,
                    tools=[BASH_TOOL],
                    request_kwargs=request_kwargs,
                    context_window=self.context_window,
                    max_output_tokens=self.max_output_tokens,
                )
        except ContextBudgetExceeded as error:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": str(error),
                    "extra": {"exit_status": "ContextLimit", "submission": ""},
                }
            ) from error
        return super()._query(messages, **{**kwargs, "max_tokens": max_tokens})


class K2HorizonContextLimitedModel(ContextLimitedModel):
    """Supply the reasoning field required by K2's template after tool calls."""

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        formatted = [
            (
                {
                    **message,
                    "reasoning": message.get("reasoning")
                    or message.get("reasoning_content")
                    or "",
                }
                if message["role"] == "assistant"
                else message
            )
            for message in messages
        ]
        return super()._query(formatted, **kwargs)


class _ContextCompactionMixin:
    """Compact completed Mini-SWE turns without discarding the saved transcript."""

    def __init__(
        self,
        *,
        summarization_threshold_chars: int,
        **kwargs: Any,
    ) -> None:
        if summarization_threshold_chars < 1:
            raise ValueError("summarization_threshold_chars must be positive")
        super().__init__(**kwargs)
        self.summarization_threshold_chars = summarization_threshold_chars
        self._summary_message: dict[str, Any] | None = None
        self._summarized_history_messages = 0
        self._compaction_events: list[dict[str, Any]] = []

    def serialize(self) -> dict[str, Any]:
        result = super().serialize()
        result["info"]["config"]["model"][
            "summarization_threshold_chars"
        ] = self.summarization_threshold_chars
        result["info"]["context_compaction"] = {
            "summarized_history_messages": self._summarized_history_messages,
            "current_summary": self._summary_message,
            "events": self._compaction_events,
        }
        return result

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        request_messages = self._request_messages(messages)
        return super().query(request_messages, **kwargs)

    def _request_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) < 2:
            return list(messages)

        prefix = list(messages[:2])
        unsummarized = list(messages[2 + self._summarized_history_messages :])
        active_history = (
            [self._summary_message] if self._summary_message is not None else []
        ) + unsummarized
        before_chars = _history_characters(active_history)
        if before_chars <= self.summarization_threshold_chars:
            return prefix + active_history

        summary_budget = max(1, self.summarization_threshold_chars // 2)
        recent_budget = self.summarization_threshold_chars - summary_budget
        turns = _turns(unsummarized)
        kept_turns: list[list[dict[str, Any]]] = []
        kept_chars = 0
        while turns:
            candidate = turns[-1]
            candidate_chars = _history_characters(candidate)
            if candidate_chars + kept_chars > recent_budget:
                break
            kept_turns.insert(0, turns.pop())
            kept_chars += candidate_chars

        summarized_messages = [message for turn in turns for message in turn]
        summary_sources = (
            [self._summary_message] if self._summary_message is not None else []
        ) + summarized_messages
        assert summary_sources

        had_summary = self._summary_message is not None
        summary_text, summary_metadata = self._summarize(
            summary_sources, summary_budget
        )
        self._summary_message = _summary_message(summary_text, summary_budget)
        self._summarized_history_messages += len(summarized_messages)
        if not summarized_messages:
            self._summarized_history_messages += sum(len(turn) for turn in kept_turns)

        kept_messages = [message for turn in kept_turns for message in turn]
        compacted = [self._summary_message, *kept_messages]
        self._compaction_events.append(
            {
                "threshold_chars": self.summarization_threshold_chars,
                "history_chars_before": before_chars,
                "history_chars_after": _history_characters(compacted),
                "newly_summarized_messages": len(summary_sources)
                - (1 if had_summary else 0),
                **summary_metadata,
            }
        )
        return prefix + compacted

    def _summarize(
        self, messages: list[dict[str, Any]], output_chars: int
    ) -> tuple[str, dict[str, Any]]:
        source, omitted_chars = _bounded_source(
            _render_summary_source(messages),
            self.summarization_threshold_chars * _SUMMARY_SOURCE_MULTIPLIER,
        )
        request_kwargs = {
            key: value
            for key, value in self.config.model_kwargs.items()
            if key != "max_tokens"
        }
        summary_messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ]
        try:
            with httpx.Client(timeout=60.0) as client:
                max_tokens = completion_budget(
                    client,
                    api_base=self.api_base,
                    model=self.config.model_name.removeprefix("hosted_vllm/"),
                    messages=summary_messages,
                    tools=[],
                    request_kwargs=request_kwargs,
                    context_window=self.context_window,
                    max_output_tokens=min(
                        _SUMMARY_OUTPUT_TOKENS, self.max_output_tokens
                    ),
                )
            response = self._completion(
                messages=summary_messages,
                max_tokens=max_tokens,
                **request_kwargs,
            )
            content = response["choices"][0]["message"].get("content")
            if isinstance(content, str) and content.strip():
                usage = response.get("usage")
                return content.strip()[:output_chars], {
                    "source_characters": len(source),
                    "source_characters_omitted": omitted_chars,
                    "summary_characters": min(len(content.strip()), output_chars),
                    "summary_prompt_tokens": (
                        usage.get("prompt_tokens") if usage is not None else None
                    ),
                    "summary_completion_tokens": (
                        usage.get("completion_tokens") if usage is not None else None
                    ),
                    "fallback_reason": None,
                }
            fallback_reason = "empty_summary"
        except Exception as error:
            fallback_reason = type(error).__name__

        fallback, fallback_omitted = _bounded_source(source, output_chars)
        return fallback, {
            "source_characters": len(source),
            "source_characters_omitted": omitted_chars + fallback_omitted,
            "summary_characters": len(fallback),
            "summary_prompt_tokens": None,
            "summary_completion_tokens": None,
            "fallback_reason": fallback_reason,
        }


class ContextCompactingModel(_ContextCompactionMixin, ContextLimitedModel):
    """Context-limited Mini-SWE model with audited history compaction."""


class K2HorizonContextCompactingModel(
    _ContextCompactionMixin, K2HorizonContextLimitedModel
):
    """K2 Mini-SWE model with audited history compaction."""
