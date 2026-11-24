"""Training treatment for terminal task failures."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

from loguru import logger
from rigging.task_errors import ErrorCategory, error_category, errors_by_category, known_error_types
from skyrl_train.task_errors import ERROR_CATEGORIES

from skyrl_train.error_treatment import ErrorTreatment


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
    classification_candidates = set(known_error_types(categories=ERROR_CATEGORIES))
    if error_handling.default_error_treatment is not ErrorTreatment.PASSTHROUGH:
        classification_candidates.difference_update(errors_by_category(ErrorCategory.UNKNOWN, categories=ERROR_CATEGORIES))
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

    category = error_category(exception_type, categories=ERROR_CATEGORIES)
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
