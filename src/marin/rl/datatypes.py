"""Common dataclasses and interfaces for Marin's RL subsystem.

This module purposefully contains *only* lightweight type definitions and
interfaces that are shared across the training, environment, and inference
components.  Implementation-specific logic should live elsewhere to avoid
introducing heavy dependencies at import time.
"""

from collections.abc import Callable
from dataclasses import dataclass
import dataclasses
import warnings
import numpy as np
from typing import Any

try:
    from openai.types.chat import ChatCompletion, Choice
except ImportError:
    ChatCompletion = Any
    Choice = Any


__all__ = [
    "GroupKey",
    "InferenceEndpoint",
    "RLExample",
    "Rollout",
    "RolloutGroup",
    "RolloutRecord",
    "RolloutSink",
    "SampledBatch",
    "Turn",
]


# ---------------------------------------------------------------------------
# Rollouts & Turns
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class InferenceMetadata:
    """Metadata about the inference of a turn."""

    model_version: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    input_seed: int | None = None


@dataclass(slots=True, frozen=True)
class Turn:
    """A single message-level interaction within a rollout."""

    message: str
    role: str
    tokens: list[str] | None = None
    logprobs: np.ndarray | None = None
    reward: float | None = None
    inference_metadata: InferenceMetadata | dict | None = None
    timestamp: int | None = None

    @staticmethod
    def from_prompt(prompt: str, input_seed: int | None) -> "Turn":
        """
        Build a Turn from a prompt.
        """
        return Turn(message=prompt, role="user")

    @staticmethod
    def from_openai_response(response: ChatCompletion, reward: float | None, input_seed: int | None) -> "Turn":
        """
        Build a Turn from an OpenAI ChatCompletion-like response object.

        (ChatGPT decided to code this in a very defensive way... Gonna go with it.)

        Notes
        -----
        - Supports both the modern `choice.logprobs.content` shape
          (list of {token, logprob, top_logprobs, bytes}) and a simpler
          `choice.logprobs.logprobs` vector if present.
        - Token *IDs* are generally not exposed by Chat Completions; we store token
          *strings* if available, else leave None.  # TODO: fill with token IDs if/when exposed.
        """
        # Choose first choice (warn if multiple)
        try:
            choices = response.choices
        except AttributeError as e:
            raise ValueError("Response object missing 'choices'.") from e

        if not choices:
            raise ValueError("Response has no choices.")

        if len(choices) > 1:
            warnings.warn(f"Multiple choices in response; using the first. Count={len(choices)}", stacklevel=2)

        choice = choices[0]

        # Extract role and content (handle tool messages gracefully)
        msg = getattr(choice, "message", None)
        if msg is None:
            raise ValueError("Choice missing 'message'.")

        role = getattr(msg, "role", None) or "assistant"
        content = getattr(msg, "content", None)
        if content is None:
            # Tool/function messages sometimes have no textual content.
            content = ""

        # Extract timestamp if present (OpenAI returns a UNIX epoch 'created' on the root)
        timestamp = getattr(response, "created", None)

        # make sure no tool calls are present (not implemented yet)
        if getattr(choice, "tool_calls", None) is not None and len(getattr(choice, "tool_calls", [])) > 0:
            raise NotImplementedError("Tool calls are not supported yet")

        # Extract logprobs/tokens in a flexible way
        tokens_list: list[str] = []
        lps_list: list[float] = []

        lp_obj = getattr(choice, "logprobs", None)
        if lp_obj is not None:
            # Preferred: new-style per-token entries at logprobs.content
            content_lps = getattr(lp_obj, "content", None)
            if content_lps:
                for entry in content_lps:
                    # entry typically has .token (str) and .logprob (float)
                    tok = getattr(entry, "token", None)
                    lp = getattr(entry, "logprob", None)
                    if tok is not None and lp is not None:
                        tokens_list.append(tok)
                        lps_list.append(lp)
            else:
                # Fallback: a flat vector at logprobs.logprobs (rare)
                flat_lps = getattr(lp_obj, "logprobs", None)
                if flat_lps is not None:
                    try:
                        lps_list = list(flat_lps)
                        # No tokens exposed in this shape
                    except Exception:
                        pass

        # Finalize arrays; if nothing available, honor the dataclass contract: None => not recorded
        logprobs_arr = np.array(lps_list, dtype=float) if lps_list else None

        # Build inference metadata
        model_version = getattr(response, "model", "unknown")
        finish_reason = getattr(choice, "finish_reason", None)

        # usage may include prompt_tokens, completion_tokens, total_tokens
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        else:
            usage_dict = {}

        meta = InferenceMetadata(
            model_version=model_version,
            finish_reason=finish_reason,
            usage=usage_dict,
            input_seed=input_seed,
        )

        return Turn(
            message=content,
            role=role,
            tokens=tokens_list or None,
            logprobs=logprobs_arr,
            reward=reward,
            inference_metadata=meta,
            timestamp=timestamp,
        )

    @staticmethod
    def system_text(text: str) -> "Turn":
        """Helper to create a system turn from plain text."""
        return Turn(message=text, role="system")

    @staticmethod
    def assistant_text(text: str, *, reward: float | None = None, input_seed: int | None = None) -> "Turn":
        """Helper to create an assistant turn from plain text.

        Includes minimal inference metadata if an input_seed is provided.
        """
        meta = InferenceMetadata(input_seed=input_seed) if input_seed is not None else None
        return Turn(message=text, role="assistant", reward=reward, inference_metadata=meta)


@dataclass(slots=True, frozen=True)
class Rollout:
    """A sequence of :class:`Turn` objects plus auxiliary metadata."""

    environment: str
    problem_id: str
    rollout_uid: str

    turns: list[Turn]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.turns)


# A callable that accepts a *batch* of :class:`RolloutGroup` objects.
RolloutSink = Callable[[list["RolloutGroup"]], None]


# ---------------------------------------------------------------------------
# Inference endpoint placeholder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceEndpoint:
    """Location of an OAI-compatible inference server.

    For now this is just a plain address string (e.g. "http://host:8000").
    """

    address: str
    model: str


# ---------------------------------------------------------------------------
# Replay buffer data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutRecord:
    """Fully materialized rollout plus associated metadata.

    Attributes
    ----------
    environment:
        Name of the environment that produced the rollout.
    example_id:
        Identifier for the dataset example or task instance.
    policy_version:
        Version of the policy used to generate the rollout.
    replica_id:
        Identifier for the environment replica that produced the rollout.
    rollout_uid:
        Unique identifier for deduplicating rollouts.
    turns:
        Ordered list of :class:`Turn` objects comprising the rollout.
    metadata:
        Additional implementation-defined metadata.
    created_ts:
        UNIX timestamp when the rollout was generated.
    """

    environment: str
    example_id: str
    policy_version: str
    rollout_uid: str
    turns: list[Turn]
    created_ts: float
    metadata: dict[str, Any]
    reward: float | None = None
    replica_id: str = "unknown"

    @property
    def total_reward(self):
        return sum(turn.reward for turn in self.turns if turn.reward is not None)


@dataclass(frozen=True)
class RolloutGroup:
    """A sealed collection of rollouts sharing the same group key.

    Attributes
    ----------
    id:
        Deterministic identifier of the group.
    environment:
        Name of the environment that produced the rollouts.
    example_id:
        Identifier of the dataset example shared by all rollouts in the group.
    policy_version:
        Policy version associated with the rollouts.
    rollouts:
        List of :class:`RolloutRecord` objects belonging to the group.
    sealed_ts:
        UNIX timestamp when the group was sealed.
    metadata:
        Additional metadata about the group (e.g. counts, replica info).
    """

    id: str
    environment: str
    example_id: str
    policy_version: str
    rollouts: list[RolloutRecord]
    sealed_ts: float
    metadata: dict[str, Any] = None


@dataclass(frozen=True)
class GroupKey:
    """Key identifying a rollout group before it is sealed."""

    environment: str
    example_id: str


@dataclass(frozen=True)
class SampledBatch:
    """Batch of group identifiers returned by the replay buffer sampler."""

    batch_id: str
    group_ids: list[str]
    ts: float


@dataclass(frozen=True)
class RLExample:
    """A single RL training example.

    Attributes:
        tokens: Token sequence for the example
        loss_mask: Boolean mask indicating which positions to compute loss on
        advantage: Advantage values for each position
        generator_log_probs: Log probabilities from the generator model
    """

    tokens: np.ndarray  # i32["pos"]
    loss_mask: np.ndarray  # bool["pos"]
    advantage: np.ndarray  # float["pos"]
    generator_log_probs: np.ndarray  # float["pos"]
