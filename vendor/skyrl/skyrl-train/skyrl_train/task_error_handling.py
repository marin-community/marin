"""Training treatment for terminal task failures."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from loguru import logger
from rigging.task_errors import ErrorCategory, ErrorIdentity

from skyrl_train.error_treatment import ErrorTreatment

PASSTHROUGH_WITHOUT_LOGPROBS_ERROR = "PassthroughWithoutLogprobs"


DEFAULT_ERROR_TREATMENT = ErrorTreatment.ZERO


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
        """Read the training error policy and fill omitted fields with defaults."""
        defaults = DEFAULT_ERROR_HANDLING_CONFIG
        return cls(
            enable_error_classification=bool(
                config.get("enable_error_classification", defaults.enable_error_classification)
            ),
            passthrough_exceptions=frozenset(config.get("passthrough_exceptions", defaults.passthrough_exceptions)),
            mask_exceptions=frozenset(config.get("mask_exceptions", defaults.mask_exceptions)),
            zero_exceptions=frozenset(config.get("zero_exceptions", defaults.zero_exceptions)),
            default_error_treatment=ErrorTreatment(
                config.get("default_error_treatment", defaults.default_error_treatment)
            ),
            preserve_logprobs_on_timeout=bool(
                config.get("preserve_logprobs_on_timeout", defaults.preserve_logprobs_on_timeout)
            ),
        )


DEFAULT_ERROR_HANDLING_CONFIG = ErrorHandlingConfig()


_CATEGORY_TREATMENTS = MappingProxyType(
    {
        ErrorCategory.INFRASTRUCTURE: ErrorTreatment.MASK,
        ErrorCategory.AGENT: ErrorTreatment.ZERO,
        ErrorCategory.PASSTHROUGH: ErrorTreatment.PASSTHROUGH,
    }
)


def classify_failure(failure: ErrorIdentity, config: ErrorHandlingConfig) -> ErrorTreatment:
    """Classify persisted failure metadata, honoring per-name training overrides."""
    exception_type = failure.exception_type
    override_fields = (
        (config.passthrough_exceptions, ErrorTreatment.PASSTHROUGH),
        (config.mask_exceptions, ErrorTreatment.MASK),
        (config.zero_exceptions, ErrorTreatment.ZERO),
    )
    for exception_types, treatment in override_fields:
        if exception_type in exception_types:
            return treatment

    if failure.category is not ErrorCategory.UNKNOWN:
        return _CATEGORY_TREATMENTS[failure.category]

    logger.error(
        "Unknown terminal exception type {}; applying explicit default_error_treatment={}",
        exception_type,
        config.default_error_treatment.value,
    )
    return config.default_error_treatment


def treatment_excludes_from_baseline(treatment: ErrorTreatment, *, verifier_available: bool) -> bool:
    """Exclude masked rollouts and unverified passthrough rollouts from the baseline."""
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
