# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`RolloutBackend` protocol — abstracts Together + Gemini batch APIs.

Each provider implements this protocol so the Stage-2 StepSpec doesn't care
which API produced a given rollout. One backend per provider; one batch
submission per (model, backend). No mixing of providers in a single batch.

See `.agents/logbooks/dpo_sft.md` § D-011 for the design decision.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import pathlib
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


class BatchStatus(enum.StrEnum):
    """Coarse-grained batch state across providers.

    Provider-specific intermediate states (Together's VALIDATING vs Gemini's
    "queued") are normalised into these. Anything not COMPLETED, FAILED,
    CANCELLED, or EXPIRED maps to PENDING.
    """

    PENDING = "pending"  # not done — keep polling
    COMPLETED = "completed"  # output_file_id is downloadable
    FAILED = "failed"  # terminal; no output
    CANCELLED = "cancelled"  # terminal; user or system cancelled
    EXPIRED = "expired"  # terminal; provider didn't finish in window


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters; hashed into the rollout cache key."""

    temperature: float
    top_p: float
    max_new_tokens: int
    # Reasoning-model controls. Most backends ignore these.
    thinking_level: str | None = None  # Gemini 3.x: "minimal" | "low" | "medium" | "high"

    def hash_short(self) -> str:
        """Stable short hash for cache keys."""
        payload = repr(dataclasses.asdict(self)).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


@dataclasses.dataclass(frozen=True)
class RolloutRequest:
    """One generation request submitted to a backend.

    `prompt_id` and `model_id` together identify the rollout for caching.
    The backend serialises `messages` into its own request format internally.
    """

    prompt_id: str
    model_id: str
    messages: list[dict[str, str]]
    sampling: SamplingConfig
    seed: int  # Backends that don't honour seed record None in metadata.


@dataclasses.dataclass(frozen=True)
class Rollout:
    """One generation result downloaded from a backend.

    Unified across providers — the Stage-3 verifier scoring code consumes this
    schema regardless of where the rollout came from.
    """

    prompt_id: str
    model_id: str
    backend: str  # provider name: "together" | "gemini"
    response_text: str  # the assistant's response (without thinking tokens)
    finish_reason: str | None  # provider-reported reason: "stop", "length", "error", etc.
    input_tokens: int | None  # prompt token count
    output_tokens: int | None  # response tokens (excludes thinking)
    thinking_tokens: int | None  # only set for reasoning models; None otherwise
    seed: int | None  # echo of the requested seed; None if backend ignored it
    sampling_config_hash: str  # SamplingConfig.hash_short(); sanity check on cache key
    raw_provider_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class BatchHandle:
    """Provider-issued handle for an async batch job."""

    backend: str
    batch_id: str
    model_id: str
    submitted_at_iso: str  # for telemetry / logbook entries
    expected_request_count: int  # so we can detect under-delivery on download


@runtime_checkable
class RolloutBackend(Protocol):
    """Provider-agnostic interface for a batch rollout backend.

    Implementers:
    - `experiments.ifbench.rollout.together_backend.TogetherBackend`
    - `experiments.ifbench.rollout.gemini_backend.GeminiBackend`

    The Stage-2 StepSpec calls these methods in order:
        handle = backend.submit_batch(model_id, requests, jsonl_dir)
        while backend.poll(handle) is BatchStatus.PENDING:
            sleep(...)
        rollouts = list(backend.download(handle))
    """

    name: str  # e.g. "together", "gemini"

    def submit_batch(
        self,
        model_id: str,
        requests: Iterable[RolloutRequest],
        jsonl_dir: pathlib.Path,
    ) -> BatchHandle:
        """Serialise `requests` to a .jsonl file under `jsonl_dir` and submit.

        `jsonl_dir` is the per-batch staging directory; the backend names the
        file (with a SHA-stable name so reruns hit the local cache too).
        Raises on submission errors that we shouldn't keep polling through.
        """
        ...

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Return the current normalised status. Idempotent; safe to call repeatedly."""
        ...

    def download(self, handle: BatchHandle) -> Iterable[Rollout]:
        """Yield one Rollout per response. Raises if status is not COMPLETED.

        If the provider returns fewer responses than `handle.expected_request_count`,
        the missing requests are NOT silently dropped — the implementation logs the
        gap and the caller decides whether to fail. (Stage-2 wait-on-all merge step
        treats partial as acceptable per D-011.)
        """
        ...
