# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a call samples, what its model needs, and how long the model client waits for an unavailable provider.

``LLMPolicy`` sets how a logical call samples; it is part of every recorded request's replay key.
``ModelProfile`` holds what one model needs whoever serves it: sampling defaults, its context window, its output
limit and how a replayed tool call's arguments are repaired. ``GLM_5_3`` is the library default for every role; any
other profile's ``name`` and output limit join the replay keys of the calls it shapes (``replay_identity``).
``RetryPolicy`` is the one retry owner's budget (``taskforge.llm.client.ModelClient``).

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

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

GLM_MAX_OUTPUT_TOKENS = 131_072
GLM_CONTEXT_TOKENS = 262_144
GLM_PROFILE = "glm-5.3"

CONTINUE_PROMPT = (
    "Your previous reply was cut off by the output limit. Continue exactly where it stopped, "
    "starting with the next character. Do not repeat or summarize anything already written."
)
THINK_OPEN = "<think>"
CONTINUE_FINAL_MESSAGE_FIELDS: dict[str, object] = {"continue_final_message": True, "add_generation_prompt": False}
INVALID_ARGUMENTS_KEY = "invalid_arguments"

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

    Attributes:
        max_tokens: Output budget per request, sent only when the model profile limits output.
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
        """The sampling fields every request sends."""
        return {"temperature": self.temperature, "top_p": self.top_p}


def replay_arguments(arguments: str) -> str:
    """GLM's chat template rejects a replayed call whose arguments are not a JSON object, so wrap anything else.

    The arguments come back unchanged when they parse to an object, else as ``{"invalid_arguments": <raw text>}``.
    """
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return arguments
    return json.dumps({INVALID_ARGUMENTS_KEY: arguments})


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """What a model needs from its requests, wherever it is served.

    ``sampling`` sits under the policy's ``sampling.*`` keys; ``<role>.context_tokens`` may not exceed
    ``context_tokens``. With ``max_output_tokens`` the client sends ``min(max_tokens, max_output_tokens)`` and reports a
    tool call that used the whole budget as ``length``; None leaves output to the endpoint. ``replay_arguments`` repairs
    a sampled tool call's arguments so that the conversation replays.

    ``name`` identifies the profile's output-affecting behaviour in replay keys, since a callable has no stable hash:
    give a profile with its own ``replay_arguments`` its own name, and a new one whenever that formatter changes.
    """

    sampling: LLMPolicy
    context_tokens: int
    max_output_tokens: int | None
    replay_arguments: Callable[[str], str]
    name: str = GLM_PROFILE

    def __post_init__(self) -> None:
        if self.name == GLM_PROFILE and self.replay_arguments is not replay_arguments:
            raise ValueError(f"A profile with its own replay_arguments needs a name other than {GLM_PROFILE!r}")

    def replay_identity(self) -> tuple[str, int | None] | None:
        """What replay keys carry about this profile: None for the default, so recorded journals keep their keys."""
        identity = (self.name, self.max_output_tokens)
        return None if identity == (GLM_PROFILE, None) else identity


GLM_5_3 = ModelProfile(LLMPolicy(), GLM_CONTEXT_TOKENS, None, replay_arguments)
"""The default for every role: the router's GLM-5.3 and DeepSeek V4.1 Flash share its chat-template conventions."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How long the model client waits out an unavailable provider, and how many seconds a call may spend generating
    (``generation_timeout``; None is unbounded). The backoff spaces attempts and availability polls."""

    unavailable_wait: float
    backoff_initial: float
    backoff_maximum: float
    backoff_factor: float
    generation_timeout: float | None = None


def continuation_messages(messages: Sequence[Message], partial: str) -> list[Message]:
    """Return ``messages`` followed by the partial answer as the assistant turn and a request to continue."""
    return [*messages, {"role": "assistant", "content": partial}, {"role": "user", "content": CONTINUE_PROMPT}]


def reasoning_continuation_messages(messages: Sequence[Message], reasoning: str) -> list[Message]:
    """Return ``messages`` followed by an open assistant turn holding the partial reasoning.

    Send with ``CONTINUE_FINAL_MESSAGE_FIELDS`` so the model extends that turn.
    """
    return [*messages, {"role": "assistant", "content": THINK_OPEN + reasoning}]
