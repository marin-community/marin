# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task error types and categories shared by multiple producers.

Frameworks own their remaining names. Compose their maps with this common map;
no framework exception classes are imported here.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, Field


class ErrorCategory(StrEnum):
    """Semantic category for a persisted task exception."""

    INFRASTRUCTURE = "infrastructure"  # Outside the agent; SkyRL excludes the sample.
    AGENT = "agent"  # Attributable to the agent; SkyRL assigns zero reward.
    PASSTHROUGH = "passthrough"  # Usable work remains; SkyRL keeps the verifier reward.
    UNKNOWN = "unknown"  # Requires the consumer's configured fallback.


SHARED_ERROR_CATEGORIES = MappingProxyType(
    {
        "APIConnectionError": ErrorCategory.INFRASTRUCTURE,
        "APITimeoutError": ErrorCategory.INFRASTRUCTURE,
        "AgentSetupTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "AgentTimeoutError": ErrorCategory.AGENT,
        "ContextLengthExceededError": ErrorCategory.AGENT,
        "EnvironmentStartTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "InternalServerError": ErrorCategory.INFRASTRUCTURE,
        "RewardFileEmptyError": ErrorCategory.INFRASTRUCTURE,
        "RewardFileNotFoundError": ErrorCategory.INFRASTRUCTURE,
        "TrialTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "VerifierOutputParseError": ErrorCategory.INFRASTRUCTURE,
        "VerifierRuntimeError": ErrorCategory.INFRASTRUCTURE,
        "VerifierTimeoutError": ErrorCategory.UNKNOWN,
    }
)


def compose_error_categories(*sources: Mapping[str, ErrorCategory]) -> Mapping[str, ErrorCategory]:
    """Compose category maps, rejecting names owned by more than one source."""
    categories: dict[str, ErrorCategory] = {}
    for source in sources:
        duplicates = categories.keys() & source.keys()
        if duplicates:
            raise ValueError(f"Task error names have multiple owners: {sorted(duplicates)}")
        categories.update(source)
    return MappingProxyType(categories)


def error_category(exception_type: str, *, categories: Mapping[str, ErrorCategory]) -> ErrorCategory:
    return categories.get(exception_type, ErrorCategory.UNKNOWN)


def errors_by_category(category: ErrorCategory, *, categories: Mapping[str, ErrorCategory]) -> frozenset[str]:
    return frozenset(name for name, value in categories.items() if value == category)


def known_error_types(*, categories: Mapping[str, ErrorCategory]) -> frozenset[str]:
    return frozenset(categories)


DEFAULT_RETRY_EXCLUSIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)


class RetrySettings(BaseModel):
    """Retry budget and name filters shared by task execution paths."""

    max_retries: int = Field(default=0, ge=0)
    include_exceptions: set[str] | None = Field(default=None, description="No name inclusion filter when None")
    exclude_exceptions: set[str] | None = Field(
        default_factory=lambda: set(DEFAULT_RETRY_EXCLUSIONS),
        description="Excluded names take precedence over included names",
    )
    wait_multiplier: float = 1.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 60.0
