from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any

from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    ContextManagementInfrastructureError,
    LLMResponse,
)

logger = logging.getLogger(__name__)
MESSAGES_PER_TURN = 2


def trim_latest_turn(
    message_history: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return history without its latest turn, or None if nothing can be trimmed."""
    if len(message_history) <= 1:
        return None
    remove_count = min(MESSAGES_PER_TURN, len(message_history) - 1)
    return message_history[:-remove_count]


@dataclass(frozen=True)
class ContextManagedResponse:
    """An LLM response and the exact history accepted by the provider."""

    response: LLMResponse
    accepted_history: list[dict[str, Any]]


class ContextWindowManager:
    """Fit recovery requests using the provider as the admission authority.

    Token counts remain useful for starting recovery before the hard boundary,
    but a client-side tokenizer cannot reproduce every provider chat template.
    This class therefore retries context-rejected recovery requests with less
    copied history. It never changes the caller's history and never dispatches a
    request with a non-positive output budget.
    """

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    async def call(
        self,
        *,
        prompt: str,
        message_history: list[dict[str, Any]],
        max_tokens: int,
        call_kwargs: Mapping[str, object],
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
                    "Provider rejected context recovery with %s messages (%s); "
                    "retrying with %s messages",
                    len(retained_history),
                    type(error).__name__,
                    len(trimmed_history),
                )
                retained_history = trimmed_history
