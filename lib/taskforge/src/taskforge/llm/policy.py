# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling and continuation settings for GLM-5.3 calls, and the settings for calling the provider.

``LLMPolicy`` sets how a logical call samples. ``LLMProviderConfig`` sets how the client calls the
provider: the model, the retries and their backoff, the infrastructure hold, the stall timeout and
the connection pool.

``max_tokens`` defaults to the model's output limit. The client sends it unchanged with the whole
prompt, so a prompt too long for that budget to fit the context window raises ``ContextExhausted``.
A prompt that long usually means its parts belong in files, for a model in an agentic environment
to read.

When a reply stops with ``finish_reason == "length"``, the client saves the partial output and
continues, up to ``max_continuations`` times. The continuation depends on where the cut fell, as
measured live on GLM-5.3:

* Cut off in the answer: the client sends the partial answer as the assistant turn, followed by a
  user turn asking the model to continue. vLLM's ``continue_final_message`` fails here because the
  reasoning parser files the continued answer under ``reasoning``.
* Cut off while still reasoning, before any answer text: the client leaves the assistant turn open
  as ``<think>`` plus the partial reasoning and sends it with ``continue_final_message``. The model
  resumes the same reasoning, with the parser splitting the rest into reasoning and answer. A user
  "continue" turn here would instead open a fresh reasoning block, which talks about the cut-off
  and loops. Carrying the reasoning as ``reasoning_content`` leaves that loop in place.
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
    """Sampling settings for a logical call.

    ``temperature`` and ``top_p`` default to GLM-5.3's published ``generation_config.json``
    (temperature 1.0, top_p 0.95, no top_k), so an unset policy samples as the model publishes.
    Chat completions and the token-id route both send exactly ``sampling_params()``, so the two
    transports sample the same distribution.

    Attributes:
        max_tokens: Output budget per request. Defaults to the model maximum.
        reasoning_effort: Passed to the chat template as ``reasoning_effort``.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        max_continuations: Continuation rounds after ``finish_reason == "length"``; 0 disables.
    """

    max_tokens: int = GLM_MAX_OUTPUT_TOKENS
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    temperature: float = 1.0
    top_p: float = 0.95
    max_continuations: int = 4

    def template_kwargs(self) -> dict[str, object]:
        """The ``chat_template_kwargs`` this policy sends."""
        return {"reasoning_effort": str(self.reasoning_effort)}

    def sampling_params(self) -> dict[str, float]:
        """The sampling fields every request sends, on either transport."""
        return {"temperature": self.temperature, "top_p": self.top_p}


@dataclass(frozen=True)
class LLMProviderConfig:
    """How the client calls the LLM provider.

    Attributes:
        model: The model id sent with each request; it must match the endpoint's model.
        max_attempts: Attempts per request, with each continuation a separate request. When the last
            attempt fails, the call raises ``ModelUnavailable``. Only failures outside a hold count.
        hold_timeout: Seconds an infrastructure hold may last before ``ModelUnavailable``.
        backoff_initial: First delay, in seconds, between attempts and between health polls.
        backoff_maximum: Largest delay, in seconds.
        backoff_factor: Growth of the delay after each wait.
        stall_timeout: Seconds without a streamed token before the client abandons an attempt.
        max_connections: httpx connection-pool size; sized for hundreds of concurrent calls.
        structured_repairs: How many times an invalid structured reply is sent back for correction before the
            call fails.
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
