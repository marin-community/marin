"""Harbor task runtime, verifier, and retry settings."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

HARBOR_EXCEPTION = "harbor_exception_type"
# Harbor trials one run executes at once when its config leaves n_concurrent_trials unset.
DEFAULT_CONCURRENT_TRIALS = 16


def concurrent_trials(harbor: Mapping[str, Any]) -> int:
    """The Harbor trials one run executes at once, from a ``terminal_bench_config.harbor`` mapping."""
    value = harbor.get("n_concurrent_trials")
    return DEFAULT_CONCURRENT_TRIALS if value is None else int(value)


class EnvironmentType(StrEnum):
    DOCKER = "docker"
    DAYTONA = "daytona"


class EnvironmentConfig(BaseModel):
    type: EnvironmentType = EnvironmentType.DOCKER
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class VerifierConfig(BaseModel):
    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    disable: bool = False


DEFAULT_RETRY_EXCLUSIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)


class RetryConfig(BaseModel):
    max_retries: int = Field(default=0, ge=0)
    include_exceptions: set[str] | None = None
    exclude_exceptions: set[str] | None = Field(default_factory=lambda: set(DEFAULT_RETRY_EXCLUSIONS))
    wait_multiplier: float = 1.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 60.0
