# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str
    model: str
    api_key: str | None = None

    def url(self, path: str) -> str:
        """Return an endpoint URL under the API root."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True)
class RunningModel:
    """A model that is already being served by a launcher-owned backend."""

    endpoint: OpenAIEndpoint
    tokenizer: str | None = None


@dataclass(frozen=True)
class ModelDeployment:
    """Configuration needed by a launcher to serve a model artifact."""

    model_name: str
    model_path: str
    tokenizer: str | None = None
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)


class ModelLauncher(Protocol):
    """Launch a model and own its serving lifecycle."""

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        """Return a context manager that yields a running served model."""
        ...


class TokenRolloutFinishReason(StrEnum):
    """Backend-independent reason a token rollout stopped."""

    STOP = "stop"
    LENGTH = "length"
    EOS_TOKEN = "eos_token"
    CANCELLED = "cancelled"
    ERROR = "error"


class TokenRolloutFailureReason(StrEnum):
    """Backend-independent reason a tokenized rollout request failed."""

    INVALID_REQUEST = "invalid_request"
    BACKEND_ERROR = "backend_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TokenizerIdentity:
    """Tokenizer identity needed to replay tokenized rollout batches."""

    name_or_path: str
    revision: str | None = None
    vocab_size: int | None = None
    chat_template_hash: str | None = None
    special_token_ids: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name_or_path:
            raise ValueError("name_or_path must be non-empty")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive when set")
        for name, token_id in self.special_token_ids.items():
            if not name:
                raise ValueError("special token names must be non-empty")
            if token_id < 0:
                raise ValueError("special token IDs must be non-negative")


@dataclass(frozen=True)
class PolicyIdentity:
    """Policy/checkpoint identity for a batch of generated rollouts."""

    policy_name: str
    checkpoint_ref: str
    checkpoint_step: int | None = None
    weight_version: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.policy_name:
            raise ValueError("policy_name must be non-empty")
        if not self.checkpoint_ref:
            raise ValueError("checkpoint_ref must be non-empty")
        if self.checkpoint_step is not None and self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be non-negative when set")


@dataclass(frozen=True)
class TokenSamplingParameters:
    """Low-level sampling parameters for tokenized rollout requests."""

    max_tokens: int
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    return_logprobs: bool = True

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1] when set")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative when set")
        _validate_non_negative_ints("stop_token_ids", self.stop_token_ids)


@dataclass(frozen=True)
class TokenizedRolloutRequest:
    """One already-tokenized prompt to sample one or more completions for."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: TokenSamplingParameters
    n_generations: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.n_generations <= 0:
            raise ValueError("n_generations must be positive")
        _validate_non_negative_ints("prompt_token_ids", self.prompt_token_ids)


@dataclass(frozen=True)
class TokenizedRolloutBatchRequest:
    """A tokenized rollout batch that avoids OpenAI-compatible JSON surfaces."""

    batch_id: str
    tokenizer: TokenizerIdentity
    policy: PolicyIdentity
    requests: tuple[TokenizedRolloutRequest, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if not self.requests:
            raise ValueError("requests must be non-empty")
        request_ids = [request.request_id for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_id values must be unique within a batch")


@dataclass(frozen=True)
class TokenRolloutTiming:
    """Host-observed timing in seconds for a rollout batch or sequence."""

    queued: float | None = None
    prefill: float | None = None
    decode: float | None = None
    device: float | None = None
    host: float | None = None
    total: float | None = None

    def __post_init__(self):
        for field_name in ("queued", "prefill", "decode", "device", "host", "total"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when set")


@dataclass(frozen=True)
class TokenRolloutAdmissionMetadata:
    """Admission and scheduler metadata for a tokenized rollout batch."""

    queued_tokens: int | None = None
    admitted_tokens: int | None = None
    prefill_admissions: int = 0
    prefill_prompt_tokens_per_admission: tuple[int, ...] = ()
    decode_rounds: int | None = None
    backend_request_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for field_name in ("queued_tokens", "admitted_tokens", "decode_rounds"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when set")
        if self.prefill_admissions < 0:
            raise ValueError("prefill_admissions must be non-negative")
        if len(self.prefill_prompt_tokens_per_admission) != self.prefill_admissions:
            raise ValueError("prefill_prompt_tokens_per_admission must match prefill_admissions")
        _validate_non_negative_ints(
            "prefill_prompt_tokens_per_admission",
            self.prefill_prompt_tokens_per_admission,
        )


@dataclass(frozen=True)
class MoeRouterReplayMetadata:
    """Reference to MoE router data sufficient for later replay or auditing."""

    format: str
    payload_ref: str | None = None
    layer_names: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.format:
            raise ValueError("format must be non-empty")
        for layer_name in self.layer_names:
            if not layer_name:
                raise ValueError("layer_names must not contain empty names")


@dataclass(frozen=True)
class ExpertLoadAccounting:
    """Per-expert load accounting for MoE rollout backends."""

    num_experts: int
    tokens_per_expert: tuple[int, ...]
    dropped_tokens: int = 0
    capacity: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if len(self.tokens_per_expert) != self.num_experts:
            raise ValueError("tokens_per_expert length must match num_experts")
        _validate_non_negative_ints("tokens_per_expert", self.tokens_per_expert)
        if self.dropped_tokens < 0:
            raise ValueError("dropped_tokens must be non-negative")
        if self.capacity is not None and self.capacity < 0:
            raise ValueError("capacity must be non-negative when set")


@dataclass(frozen=True)
class TokenizedRollout:
    """One sampled token sequence with prompt/completion boundaries preserved."""

    request_id: str
    generation_index: int
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    finish_reason: TokenRolloutFinishReason
    prompt_mask: tuple[bool, ...]
    completion_mask: tuple[bool, ...]
    stop_token_id: int | None = None
    timing: TokenRolloutTiming = field(default_factory=TokenRolloutTiming)
    router_replay: MoeRouterReplayMetadata | None = None
    expert_load: ExpertLoadAccounting | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        _validate_non_negative_ints("prompt_token_ids", self.prompt_token_ids)
        _validate_non_negative_ints("completion_token_ids", self.completion_token_ids)
        if len(self.completion_token_ids) != len(self.completion_logprobs):
            raise ValueError("completion_token_ids and completion_logprobs must have the same length")
        if len(self.prompt_mask) != len(self.prompt_token_ids):
            raise ValueError("prompt_mask length must match prompt_token_ids")
        if len(self.completion_mask) != len(self.completion_token_ids):
            raise ValueError("completion_mask length must match completion_token_ids")
        if self.stop_token_id is not None and self.stop_token_id < 0:
            raise ValueError("stop_token_id must be non-negative when set")

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return prompt and completion tokens as one sequence."""
        return self.prompt_token_ids + self.completion_token_ids

    @property
    def loss_mask(self) -> tuple[bool, ...]:
        """Return the prompt and completion masks as one sequence."""
        return self.prompt_mask + self.completion_mask


@dataclass(frozen=True)
class TokenizedRolloutFailure:
    """One failed request or generation in a tokenized rollout batch."""

    request_id: str
    reason: TokenRolloutFailureReason
    generation_index: int | None = None
    message: str = ""
    backend_request_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.generation_index is not None and self.generation_index < 0:
            raise ValueError("generation_index must be non-negative when set")


@dataclass(frozen=True)
class TokenizedRolloutBatchResult:
    """Result for a tokenized rollout batch."""

    batch_id: str
    tokenizer: TokenizerIdentity
    policy: PolicyIdentity
    rollouts: tuple[TokenizedRollout, ...]
    failures: tuple[TokenizedRolloutFailure, ...] = ()
    timing: TokenRolloutTiming = field(default_factory=TokenRolloutTiming)
    admission: TokenRolloutAdmissionMetadata = field(default_factory=TokenRolloutAdmissionMetadata)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")


class TokenRolloutBackend(Protocol):
    """Backend that can generate tokenized rollout batches without OpenAI serialization."""

    def generate_token_rollouts(self, batch: TokenizedRolloutBatchRequest) -> TokenizedRolloutBatchResult:
        """Generate tokenized rollouts for an already-tokenized batch."""
        ...


def _validate_non_negative_ints(field_name: str, values: tuple[int, ...]) -> None:
    for value in values:
        if value < 0:
            raise ValueError(f"{field_name} must contain only non-negative values")
