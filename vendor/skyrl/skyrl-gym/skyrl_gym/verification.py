"""Harness-independent contracts between verifiers and trajectory runners."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol, TypeAlias

Message: TypeAlias = Mapping[str, Any]
UNKNOWN_STOP_REASON = "unknown"
VERIFIER_RUNTIME_ERROR = "VerifierRuntimeError"


def _normalize_finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return normalized


def _optional_finite(value: float | None, *, field_name: str) -> float | None:
    return None if value is None else _normalize_finite(value, field_name=field_name)


@dataclass(frozen=True)
class RolloutEvidence:
    """Normalized evidence produced by a rollout, independent of its harness."""

    messages: tuple[Message, ...] = ()
    response: str | None = None
    stop_reason: str = UNKNOWN_STOP_REASON
    generated_token_count: int | None = None
    response_token_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generated_token_count is not None and self.generated_token_count < 0:
            raise ValueError("generated_token_count must be non-negative")


class VerificationStatus(StrEnum):
    """Whether a verifier produced a usable verdict."""

    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    # The environment was configured not to verify; the trajectory still trains.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class VerificationResult:
    """A verifier verdict or an explicit explanation that no verdict exists."""

    status: VerificationStatus
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    score_min: float = 0.0
    score_max: float = 1.0

    def __post_init__(self) -> None:
        minimum = _normalize_finite(self.score_min, field_name="score_min")
        maximum = _normalize_finite(self.score_max, field_name="score_max")
        if maximum <= minimum:
            raise ValueError("Verifier score range must be increasing")
        object.__setattr__(self, "score_min", minimum)
        object.__setattr__(self, "score_max", maximum)
        if self.status is VerificationStatus.VERIFIED:
            if self.score is None:
                raise ValueError("verified results require a score")
            object.__setattr__(self, "score", _normalize_finite(self.score, field_name="score"))
            if self.reason is not None:
                raise ValueError("verified results cannot carry an unavailability reason")
            return
        if self.score is not None or self.passed is not None:
            raise ValueError(f"{self.status.value} results cannot carry a verifier verdict")
        if not self.reason:
            raise ValueError(f"{self.status.value} results require a reason")

    @classmethod
    def verified(
        cls,
        score: float,
        *,
        passed: bool | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        score_min: float = 0.0,
        score_max: float = 1.0,
    ) -> "VerificationResult":
        return cls(
            status=VerificationStatus.VERIFIED,
            score=score,
            passed=passed,
            diagnostics={} if diagnostics is None else diagnostics,
            score_min=score_min,
            score_max=score_max,
        )

    @classmethod
    def unavailable(cls, reason: str, *, diagnostics: Mapping[str, Any] | None = None) -> "VerificationResult":
        return cls(
            status=VerificationStatus.UNAVAILABLE,
            reason=reason,
            diagnostics={} if diagnostics is None else diagnostics,
        )

    @classmethod
    def skipped(cls, reason: str, *, diagnostics: Mapping[str, Any] | None = None) -> "VerificationResult":
        return cls(
            status=VerificationStatus.SKIPPED,
            reason=reason,
            diagnostics={} if diagnostics is None else diagnostics,
        )

    @classmethod
    def error(cls, reason: str, *, diagnostics: Mapping[str, Any] | None = None) -> "VerificationResult":
        return cls(
            status=VerificationStatus.ERROR,
            reason=reason,
            diagnostics={} if diagnostics is None else diagnostics,
        )


def normalized_verifier_score(result: VerificationResult) -> float:
    """Scale a verified score to [0, 1] using its declared native bounds."""
    if result.status is not VerificationStatus.VERIFIED or result.score is None:
        raise ValueError("A verified score is required for normalization")
    minimum, maximum = result.score_min, result.score_max
    return min(1.0, max(0.0, (result.score - minimum) / (maximum - minimum)))


@dataclass(frozen=True)
class RewardResult:
    """Raw verifier outcome and the reward channels derived from it."""

    unshaped_reward: float | None
    optimization_reward: float
    token_rewards: tuple[float, ...] | None = None
    components: Mapping[str, float] = field(default_factory=dict)
    token_credit: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unshaped_reward",
            _optional_finite(self.unshaped_reward, field_name="unshaped_reward"),
        )
        object.__setattr__(
            self,
            "optimization_reward",
            _normalize_finite(self.optimization_reward, field_name="optimization_reward"),
        )
        if self.token_rewards is not None:
            object.__setattr__(
                self,
                "token_rewards",
                tuple(
                    _normalize_finite(value, field_name=f"token_rewards[{index}]")
                    for index, value in enumerate(self.token_rewards)
                ),
            )
        object.__setattr__(
            self,
            "components",
            {
                name: _normalize_finite(value, field_name=f"components[{name!r}]")
                for name, value in self.components.items()
            },
        )
        if self.token_credit is not None:
            object.__setattr__(
                self,
                "token_credit",
                tuple(
                    _normalize_finite(value, field_name=f"token_credit[{index}]")
                    for index, value in enumerate(self.token_credit)
                ),
            )

    def validate_for(self, evidence: RolloutEvidence) -> None:
        """Validate reward channels that depend on the associated evidence."""
        if self.token_credit is not None:
            if len(self.token_credit) != len(evidence.response_token_ids):
                raise ValueError("token_credit must align with response_token_ids")
        if self.token_rewards is not None:
            if len(self.token_rewards) != len(evidence.response_token_ids):
                raise ValueError("token_rewards must align with response_token_ids")

    def to_trainer_reward(self) -> float | list[float]:
        """Return the legacy scalar or token-list trainer transport."""
        return self.optimization_reward if self.token_rewards is None else list(self.token_rewards)


@dataclass(frozen=True)
class TrainingDisposition:
    """Whether a trajectory contributes tokens and group-baseline statistics."""

    loss_eligible: bool
    baseline_eligible: bool
    reason: str
    exception_type: str | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("training disposition requires a reason")

    @classmethod
    def train(cls, reason: str = "verified") -> "TrainingDisposition":
        return cls(loss_eligible=True, baseline_eligible=True, reason=reason)

    @classmethod
    def mask(cls, reason: str, *, exception_type: str | None = None) -> "TrainingDisposition":
        return cls(
            loss_eligible=False,
            baseline_eligible=False,
            reason=reason,
            exception_type=exception_type,
        )


class Verifier(Protocol):
    """Native interface for a verifier that consumes normalized rollout evidence."""

    def verify(self, evidence: RolloutEvidence) -> VerificationResult: ...
