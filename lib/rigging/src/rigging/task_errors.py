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

    INFRASTRUCTURE = "infrastructure"  # Outside the agent; SkyRL masks by default.
    AGENT = "agent"  # Attributable to the agent; SkyRL defaults to zero reward.
    PASSTHROUGH = "passthrough"  # Usable work remains; SkyRL defaults to the verifier reward.
    UNKNOWN = "unknown"  # Requires the consumer's configured fallback.


# Maps are keyed by exception class name, not by class. Keying by class would
# make Rigging import the heavy framework and SDK packages that raise these
# errors, and would fix Harbor's exception types at the lowest layer of the
# dependency stack. Persisted task records also carry only the name.
# TODO: after the task-error cleanup, move the Harbor-specific names below into
# the framework that owns them, leaving only names that several producers share.
API_CONNECTION_ERROR = "APIConnectionError"
API_TIMEOUT_ERROR = "APITimeoutError"
AGENT_SETUP_TIMEOUT_ERROR = "AgentSetupTimeoutError"
AGENT_TIMEOUT_ERROR = "AgentTimeoutError"
CONTEXT_LENGTH_EXCEEDED_ERROR = "ContextLengthExceededError"
ENVIRONMENT_START_TIMEOUT_ERROR = "EnvironmentStartTimeoutError"
INTERNAL_SERVER_ERROR = "InternalServerError"
REWARD_FILE_EMPTY_ERROR = "RewardFileEmptyError"
REWARD_FILE_NOT_FOUND_ERROR = "RewardFileNotFoundError"
TRIAL_TIMEOUT_ERROR = "TrialTimeoutError"
VERIFIER_OUTPUT_PARSE_ERROR = "VerifierOutputParseError"
VERIFIER_RUNTIME_ERROR = "VerifierRuntimeError"
VERIFIER_TIMEOUT_ERROR = "VerifierTimeoutError"


SHARED_ERROR_CATEGORIES = MappingProxyType(
    {
        API_CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        API_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        AGENT_SETUP_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        AGENT_TIMEOUT_ERROR: ErrorCategory.AGENT,
        CONTEXT_LENGTH_EXCEEDED_ERROR: ErrorCategory.AGENT,
        ENVIRONMENT_START_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        INTERNAL_SERVER_ERROR: ErrorCategory.INFRASTRUCTURE,
        REWARD_FILE_EMPTY_ERROR: ErrorCategory.INFRASTRUCTURE,
        REWARD_FILE_NOT_FOUND_ERROR: ErrorCategory.INFRASTRUCTURE,
        TRIAL_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_OUTPUT_PARSE_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_RUNTIME_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_TIMEOUT_ERROR: ErrorCategory.UNKNOWN,
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
        AGENT_TIMEOUT_ERROR,
        VERIFIER_TIMEOUT_ERROR,
        REWARD_FILE_NOT_FOUND_ERROR,
        REWARD_FILE_EMPTY_ERROR,
        VERIFIER_OUTPUT_PARSE_ERROR,
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
