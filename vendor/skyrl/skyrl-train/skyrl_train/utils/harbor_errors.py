from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from loguru import logger

from skyrl_train.error_treatment import ErrorTreatment


class ErrorCategory(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    AGENT = "agent"
    PASSTHROUGH = "passthrough"
    UNKNOWN = "unknown"


# Persisted exception names from the Harbor revision in uv.lock. The Harbor
# integration test compares this snapshot with that revision's taxonomy.
_ERROR_CATEGORIES: Mapping[str, ErrorCategory] = MappingProxyType(
    {
        "ContextManagementInfrastructureError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactUploadTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactWriterBacklogError": ErrorCategory.INFRASTRUCTURE,
        "EnvironmentStartTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "TrialTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "SandboxBuildFailedError": ErrorCategory.INFRASTRUCTURE,
        "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
        "HealthcheckError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaSandboxStopError": ErrorCategory.INFRASTRUCTURE,
        "BridgeOutageError": ErrorCategory.INFRASTRUCTURE,
        "EnrootMemoryLimitExceededError": ErrorCategory.INFRASTRUCTURE,
        "MissingExtraError": ErrorCategory.INFRASTRUCTURE,
        "AgentKilledBySignalError": ErrorCategory.INFRASTRUCTURE,
        "AgentSetupTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "ModelAuthenticationError": ErrorCategory.INFRASTRUCTURE,
        "ContextBudgetExceededError": ErrorCategory.INFRASTRUCTURE,
        "LLMRequestTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "OpenAITransportConnectTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "TmuxBatchProtocolError": ErrorCategory.INFRASTRUCTURE,
        "TmuxCommandError": ErrorCategory.INFRASTRUCTURE,
        "TmuxSessionEndedError": ErrorCategory.INFRASTRUCTURE,
        "DownloadVerifierDirError": ErrorCategory.INFRASTRUCTURE,
        "RewardFileNotFoundError": ErrorCategory.INFRASTRUCTURE,
        "RewardFileEmptyError": ErrorCategory.INFRASTRUCTURE,
        "VerifierRuntimeError": ErrorCategory.INFRASTRUCTURE,
        "VerifierOutputParseError": ErrorCategory.INFRASTRUCTURE,
        "AddTestsDirError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactPersistenceError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactWriteTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "ContainerRegistryThrottledError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaProcessError": ErrorCategory.INFRASTRUCTURE,
        "PhaseCancellationError": ErrorCategory.INFRASTRUCTURE,
        "SandboxDiskFullError": ErrorCategory.INFRASTRUCTURE,
        "Tau3RuntimeInfrastructureError": ErrorCategory.INFRASTRUCTURE,
        "AgentTimeoutError": ErrorCategory.AGENT,
        "ModelInitiatedTmuxFailure": ErrorCategory.AGENT,
        "AgentDiskFullError": ErrorCategory.AGENT,
        "ContextLengthExceededError": ErrorCategory.AGENT,
        "NonZeroAgentExitCodeError": ErrorCategory.AGENT,
        "OutputLengthExceededError": ErrorCategory.PASSTHROUGH,
        "TurnCapExhaustedError": ErrorCategory.PASSTHROUGH,
        "MalformedModelResponseError": ErrorCategory.PASSTHROUGH,
        "TrialNotScoredError": ErrorCategory.UNKNOWN,
        "VerificationNotCompletedError": ErrorCategory.UNKNOWN,
        "VerifierTimeoutError": ErrorCategory.UNKNOWN,
    }
)


def error_category(exception_type: str) -> ErrorCategory:
    return _ERROR_CATEGORIES.get(exception_type, ErrorCategory.UNKNOWN)


def errors_by_category(category: ErrorCategory) -> frozenset[str]:
    return frozenset(name for name, value in _ERROR_CATEGORIES.items() if value is category)


def known_error_types() -> frozenset[str]:
    return frozenset(_ERROR_CATEGORIES)


AGENT_TIMEOUT_ERROR = "AgentTimeoutError"
PASSTHROUGH_WITHOUT_LOGPROBS_ERROR = "PassthroughWithoutLogprobs"


DEFAULT_ERROR_TREATMENT = ErrorTreatment.ZERO


def _exception_names(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(name.strip() for name in value.split(",") if name.strip())
    return frozenset(value)


@dataclass(frozen=True)
class ErrorHandlingConfig:
    """Typed training treatment for terminal agent and serving failures."""

    enable_error_classification: bool = False
    passthrough_exceptions: frozenset[str] = field(default_factory=frozenset)
    mask_exceptions: frozenset[str] = field(default_factory=frozenset)
    zero_exceptions: frozenset[str] = field(default_factory=frozenset)
    default_error_treatment: ErrorTreatment = DEFAULT_ERROR_TREATMENT
    preserve_logprobs_on_timeout: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ErrorHandlingConfig":
        """Validate the schema-derived mapping at the terminal-bench boundary."""
        defaults = DEFAULT_ERROR_HANDLING_CONFIG
        return cls(
            enable_error_classification=bool(
                config.get("enable_error_classification", defaults.enable_error_classification)
            ),
            passthrough_exceptions=_exception_names(
                config.get("passthrough_exceptions", defaults.passthrough_exceptions)
            ),
            mask_exceptions=_exception_names(config.get("mask_exceptions", defaults.mask_exceptions)),
            zero_exceptions=_exception_names(config.get("zero_exceptions", defaults.zero_exceptions)),
            default_error_treatment=ErrorTreatment(
                config.get("default_error_treatment", defaults.default_error_treatment)
            ),
            preserve_logprobs_on_timeout=bool(
                config.get("preserve_logprobs_on_timeout", defaults.preserve_logprobs_on_timeout)
            ),
        )


DEFAULT_ERROR_HANDLING_CONFIG = ErrorHandlingConfig()


def retry_excluded_exception_types(
    configured_exclusions: Iterable[str] | None,
    error_handling: ErrorHandlingConfig,
) -> frozenset[str]:
    """Return retry exclusions with every pass-through classification made terminal."""
    classification_candidates = set(known_error_types())
    if error_handling.default_error_treatment is not ErrorTreatment.PASSTHROUGH:
        classification_candidates.difference_update(errors_by_category(ErrorCategory.UNKNOWN))
    classification_candidates.update(error_handling.passthrough_exceptions)
    classification_candidates.update(error_handling.mask_exceptions)
    classification_candidates.update(error_handling.zero_exceptions)

    passthrough_exceptions = {
        exception_type
        for exception_type in classification_candidates
        if classify_exception_type(exception_type, error_handling) is ErrorTreatment.PASSTHROUGH
    }
    return frozenset(configured_exclusions or ()) | passthrough_exceptions


_CATEGORY_TREATMENTS = {
    ErrorCategory.INFRASTRUCTURE: ErrorTreatment.MASK,
    ErrorCategory.AGENT: ErrorTreatment.ZERO,
    ErrorCategory.PASSTHROUGH: ErrorTreatment.PASSTHROUGH,
}
_TASK_INFRASTRUCTURE_ERRORS = frozenset({"TaskMachineError", "TemplateError"})


def classify_exception_type(exception_type: str, config: ErrorHandlingConfig) -> ErrorTreatment:
    """Classify a persisted terminal exception name, honoring overrides."""
    override_fields = (
        (config.passthrough_exceptions, ErrorTreatment.PASSTHROUGH),
        (config.mask_exceptions, ErrorTreatment.MASK),
        (config.zero_exceptions, ErrorTreatment.ZERO),
    )
    for exception_types, treatment in override_fields:
        if exception_type in exception_types:
            return treatment

    category = (
        ErrorCategory.INFRASTRUCTURE
        if exception_type in _TASK_INFRASTRUCTURE_ERRORS
        else error_category(exception_type)
    )
    if category is not ErrorCategory.UNKNOWN:
        return _CATEGORY_TREATMENTS[category]

    logger.error(
        "Unknown terminal exception type {}; applying explicit default_error_treatment={}",
        exception_type,
        config.default_error_treatment.value,
    )
    return config.default_error_treatment


def treatment_excludes_from_baseline(treatment: ErrorTreatment, *, verifier_available: bool) -> bool:
    """Translate a treatment into the RLOO-N exclusion bit for the available result."""
    return treatment is ErrorTreatment.MASK or (treatment is ErrorTreatment.PASSTHROUGH and not verifier_available)


def passthrough_logprob_error_type(
    treatment: ErrorTreatment,
    *,
    has_rollout_logprobs: bool,
    rollout_logprobs_required: bool,
) -> str | None:
    """Return the masking error for a passthrough result without required behavior logprobs."""
    if treatment is not ErrorTreatment.PASSTHROUGH or not rollout_logprobs_required or has_rollout_logprobs:
        return None
    return PASSTHROUGH_WITHOUT_LOGPROBS_ERROR
