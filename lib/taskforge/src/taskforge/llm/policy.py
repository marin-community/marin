# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling and continuation policy for GLM-5.3 calls, and how the provider is called.

``LLMPolicy`` is how one logical call samples. ``LLMProviderConfig`` is how the client calls the
provider: the model, retries and their backoff, the infrastructure hold, the stall timeout and the
connection pool.

``max_tokens`` defaults to the model's output limit and is never lowered: a prompt too long for
that limit to fit the context window is an error (``ContextExhausted``), not something the client
works around, and nothing is ever truncated. A prompt that long is a sign that its parts belong in
files and the model in an agentic environment that reads them. When a reply stops on
``finish_reason == "length"`` the client keeps the partial output and continues, up to
``max_continuations`` times, in one of two ways measured live on GLM-5.3:

* Cut off in the answer: the partial answer becomes the assistant turn and a user turn asks the
  model to continue. (vLLM ``continue_final_message`` is unusable here: the reasoning parser files
  the continued answer under ``reasoning``.)
* Cut off while still reasoning (no answer text yet): the assistant turn is left open as
  ``<think>`` plus the partial reasoning and sent with ``continue_final_message``, so the model
  resumes the same reasoning and the parser splits the rest into reasoning and answer. A user
  "continue" turn here instead opens a fresh reasoning block that talks about the cut-off and
  loops; carrying the reasoning as ``reasoning_content`` does not prevent that.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

GLM_MAX_OUTPUT_TOKENS = 131_072

CONTINUE_PROMPT = (
    "Your previous reply was cut off by the output limit. Continue exactly where it stopped, "
    "starting with the next character. Do not repeat or summarize anything already written."
)
THINK_OPEN = "<think>"
CONTINUE_FINAL_MESSAGE_FIELDS: dict[str, object] = {"continue_final_message": True, "add_generation_prompt": False}

Message = dict[str, object]


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class LLMPolicy:
    """How one logical call samples.

    Attributes:
        max_tokens: Output budget per request. Defaults to the model maximum.
        reasoning_effort: Passed to the chat template as ``reasoning_effort``.
        temperature: Sampling temperature.
        max_continuations: Continuation rounds after ``finish_reason == "length"``; 0 disables.
    """

    max_tokens: int = GLM_MAX_OUTPUT_TOKENS
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 0.7
    max_continuations: int = 4

    def template_kwargs(self) -> dict[str, object]:
        """The ``chat_template_kwargs`` this policy sends."""
        return {"reasoning_effort": str(self.reasoning_effort)}

    def sampling_fields(self) -> dict[str, object]:
        """Request-body fields this policy contributes, other than ``max_tokens``."""
        return {"temperature": self.temperature, "chat_template_kwargs": self.template_kwargs()}


@dataclass(frozen=True)
class LLMProviderConfig:
    """How the client calls the LLM provider.

    Attributes:
        model: The model id requests name; it must match the endpoint's model.
        max_attempts: Attempts per request (each continuation is its own request); when the last one
            fails the call raises ``ModelUnavailable``. Holds do not count.
        hold_timeout: Seconds an infrastructure hold may last before ``ModelUnavailable``.
        backoff_initial: First delay, in seconds, between attempts and between health polls.
        backoff_maximum: Largest delay, in seconds.
        backoff_factor: Growth of the delay after each wait.
        stall_timeout: Seconds with no streamed token before an attempt is abandoned.
        max_connections: httpx connection-pool size; sized for hundreds of concurrent calls.
        structured_repairs: How many times an invalid tool call is sent back for correction before the call fails.
    """

    model: str
    max_attempts: int
    hold_timeout: float
    backoff_initial: float
    backoff_maximum: float
    backoff_factor: float
    stall_timeout: float
    max_connections: int
    structured_repairs: int


def continuation_messages(messages: Sequence[Message], partial: str) -> list[Message]:
    """Return ``messages`` followed by the partial answer as the assistant turn and a request to continue."""
    return [*messages, {"role": "assistant", "content": partial}, {"role": "user", "content": CONTINUE_PROMPT}]


def reasoning_continuation_messages(messages: Sequence[Message], reasoning: str) -> list[Message]:
    """Return ``messages`` followed by an open assistant turn holding the partial reasoning.

    Send with ``CONTINUE_FINAL_MESSAGE_FIELDS`` so the model extends that turn.
    """
    return [*messages, {"role": "assistant", "content": THINK_OPEN + reasoning}]
