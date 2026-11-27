# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminus's chat history and its model client over an ``AgentHarness`` completion.

Adapted from Harbor's ``Chat``, ``OpenAILLM`` request construction and ``ContextWindowManager`` (Apache-2.0); see
``rolloutengine.agents.terminus``.

Harbor counts prompt tokens with the served model's ``/tokenize`` route. Here each served request's reported
prompt tokens (``ModelTurn.usage``, else its prompt token IDs) count the chat through that request's user
message. Messages after the last counted one are estimated without a tokenizer, with Harbor's local formula
(three tokens per message plus three priming tokens) and Harbor's character estimate of three characters per
token. The estimate covers at most the last reply and the messages that are not yet sent.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rolloutengine.harness.external import AgentCompletion
from rolloutengine.models import LENGTH_STOP_REASON, ContextWindowExceeded, ModelTurn, Prompt, TokenUsage

logger = logging.getLogger(__name__)

MESSAGES_PER_TURN = 2
CHAT_REPLY_PRIMING_TOKENS = 3
CHAT_TOKENS_PER_MESSAGE = 3
CHARACTERS_PER_TOKEN = 3


@dataclass(frozen=True)
class LLMResponse:
    content: str
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]] | None
    usage: TokenUsage | None
    prompt_tokens: int | None
    """The served prompt's token count: the reported usage, else the prompt token IDs, else ``None``."""


class ContextLengthExceededError(Exception):
    """The request does not fit the model's context window."""


class OutputLengthExceededError(Exception):
    """The response was cut by the output-token limit; ``partial_response`` keeps it."""

    def __init__(self, message: str, partial_response: LLMResponse):
        super().__init__(message)
        self.partial_response = partial_response
        self.truncated_response = partial_response.content


class ContextManagementInfrastructureError(Exception):
    """Context recovery could not construct a request that leaves room for a response."""


def assistant_message_from_response(response: LLMResponse, interleaved_thinking: bool) -> dict[str, Any]:
    """Build an assistant history message using the configured thinking mode."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if interleaved_thinking:
        # Some chat templates require this field on every assistant message,
        # including synthetic history and responses without reasoning text.
        message["reasoning_content"] = response.reasoning_content or ""
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    return message


def add_vllm_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy assistant ``reasoning_content`` into ``reasoning``, the field vLLM's chat endpoint replays."""
    return [
        (
            {**message, "reasoning": message["reasoning_content"]}
            if message.get("role") == "assistant" and message.get("reasoning_content") is not None
            else message
        )
        for message in messages
    ]


def trim_latest_turn(message_history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Return history without its latest turn, or None if nothing can be trimmed."""
    if len(message_history) <= 1:
        return None
    remove_count = min(MESSAGES_PER_TURN, len(message_history) - 1)
    return message_history[:-remove_count]


def estimated_message_tokens(message: Mapping[str, Any]) -> int:
    return CHAT_TOKENS_PER_MESSAGE + sum(len(str(value)) // CHARACTERS_PER_TOKEN for value in message.values())


class TerminusLLM:
    """Send Terminus's requests through the harness completion, with Harbor's request options.

    Every request carries the configured temperature and reasoning effort, when set, and the call's
    ``max_tokens`` and ``extra_body``. A ``"length"`` stop raises ``OutputLengthExceededError``, and a context
    rejection raises ``ContextLengthExceededError``.
    """

    def __init__(
        self,
        complete: AgentCompletion,
        *,
        context_limit: int,
        output_limit: int,
        temperature: float | None,
        reasoning_effort: str | None,
    ):
        self._complete = complete
        self._context_limit = context_limit
        self._output_limit = output_limit
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def get_model_context_limit(self) -> int:
        return self._context_limit

    def get_model_output_limit(self) -> int:
        return self._output_limit

    async def call(self, prompt: str, message_history: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        messages = add_vllm_reasoning([*message_history, {"role": "user", "content": prompt}])
        options: dict[str, Any] = {}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._reasoning_effort is not None:
            options["reasoning_effort"] = self._reasoning_effort
        options.update({key: value for key, value in kwargs.items() if value is not None})
        try:
            turn = await self._complete(Prompt(tuple(messages), options))
        except ContextWindowExceeded as error:
            raise ContextLengthExceededError(str(error)) from error
        response = _response(turn)
        if turn.stop_reason == LENGTH_STOP_REASON:
            raise OutputLengthExceededError(
                "The model hit its max_tokens limit, so the response was truncated.",
                partial_response=response,
            )
        return response


def _response(turn: ModelTurn) -> LLMResponse:
    message = turn.message
    if turn.usage is not None:
        prompt_tokens: int | None = turn.usage.prompt_tokens
    else:
        prompt_tokens = len(turn.prompt_token_ids) or None
    return LLMResponse(
        content=message.get("content") or "",
        reasoning_content=message.get("reasoning_content") or turn.metadata.get("reasoning"),
        tool_calls=message.get("tool_calls") or None,
        usage=turn.usage,
        prompt_tokens=prompt_tokens,
    )


class Chat:
    """Terminus's main conversation: each prompt and the assistant message that answered it."""

    def __init__(self, model: TerminusLLM, interleaved_thinking: bool = False):
        self._model = model
        self._messages: list[dict[str, Any]] = []
        self._interleaved_thinking = interleaved_thinking
        # The served prompt token count through each user message, keyed by the message object. A history
        # rewrite keeps prefixes or builds new messages, so a message object keeps the messages before it.
        self._served_tokens: dict[int, tuple[dict[str, Any], int]] = {}

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    async def chat(self, prompt: str, **kwargs: Any) -> LLMResponse:
        llm_response = await self._model.call(prompt=prompt, message_history=self._messages, **kwargs)
        self.append_response(prompt, llm_response)
        return llm_response

    def append_response(self, prompt: str, llm_response: LLMResponse) -> None:
        """Append one returned model response to the conversation."""
        user_message = {"role": "user", "content": prompt}
        if llm_response.prompt_tokens is not None:
            self._served_tokens[id(user_message)] = (user_message, llm_response.prompt_tokens)
        self._messages.extend([user_message, assistant_message_from_response(llm_response, self._interleaved_thinking)])

    def count_tokens(self, extra_messages: list[dict[str, Any]] | None = None) -> int:
        """Count the prompt tokens of the chat followed by ``extra_messages``."""
        messages = [*self._messages, *(extra_messages or [])]
        for index in range(len(messages) - 1, -1, -1):
            served = self._served_tokens.get(id(messages[index]))
            if served is not None and served[0] is messages[index]:
                return served[1] + sum(estimated_message_tokens(message) for message in messages[index + 1 :])
        return CHAT_REPLY_PRIMING_TOKENS + sum(estimated_message_tokens(message) for message in messages)


@dataclass(frozen=True)
class ContextManagedResponse:
    """An LLM response and the exact history accepted by the provider."""

    response: LLMResponse
    accepted_history: list[dict[str, Any]]


class ContextWindowManager:
    """Fit recovery requests using the provider as the admission authority.

    Token counts remain useful for starting recovery before the hard boundary,
    but a client-side estimate cannot reproduce every provider chat template.
    This class therefore retries context-rejected recovery requests with less
    copied history. It never changes the caller's history and never dispatches a
    request with a non-positive output budget.
    """

    def __init__(self, llm: TerminusLLM):
        self._llm = llm

    async def call(
        self,
        *,
        prompt: str,
        message_history: list[dict[str, Any]],
        max_tokens: int,
        call_kwargs: Mapping[str, Any],
    ) -> ContextManagedResponse:
        if max_tokens < 1:
            raise ContextManagementInfrastructureError(
                f"Context recovery computed a non-positive output budget: {max_tokens}."
            )

        retained_history = list(message_history)
        while True:
            try:
                response = await self._llm.call(
                    prompt=prompt,
                    message_history=retained_history,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                return ContextManagedResponse(response, retained_history)
            except ContextLengthExceededError as error:
                trimmed_history = trim_latest_turn(retained_history)
                if trimmed_history is None:
                    raise ContextManagementInfrastructureError(
                        "The provider rejected a context-recovery request after all "
                        "copied conversation turns were removed."
                    ) from error
                logger.debug(
                    "Provider rejected context recovery with %s messages (%s); retrying with %s messages",
                    len(retained_history),
                    type(error).__name__,
                    len(trimmed_history),
                )
                retained_history = trimmed_history
