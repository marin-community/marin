"""Benchmark-independent evaluation failure taxonomy and classification."""

from __future__ import annotations

import importlib
from enum import StrEnum
from functools import lru_cache

from eval.limits import ContextWindowExceededError, MissingContextLengthError


class FailurePhase(StrEnum):
    """Lifecycle boundary at which an exception escaped."""

    PREPARATION = "preparation"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    GRADING = "grading"
    SERIALIZATION = "serialization"


class FailureCategory(StrEnum):
    """Failure classes needed at the task execution boundary."""

    PREPARATION = "preparation"
    RESOURCE = "resource"
    AGENT_TIMEOUT = "AgentTimeoutError"
    MODEL_TRANSPORT = "model_transport"
    MALFORMED_MODEL_RESPONSE = "malformed_model_response"
    GENERATION_POLICY = "generation_policy"
    GENERATION = "generation"
    GRADER_INFRASTRUCTURE = "grader_infrastructure"
    SAMPLE_EXECUTION = "sample_execution"
    GRADING = "grading"
    INCOMPLETE_EVALUATION = "incomplete_evaluation"
    SERIALIZATION = "serialization"
    INVALID_RESULT = "invalid_result"


TRANSPORT_FAILURE_CATEGORIES = frozenset({FailureCategory.MODEL_TRANSPORT, FailureCategory.AGENT_TIMEOUT})


class ModelRequestValidationError(ValueError):
    """Raised before generation when a representative request is malformed."""


class MalformedJudgeResponse(ValueError):
    """A completed judgment failed its declared semantic response format."""


@lru_cache(maxsize=1)
def transport_error_types() -> tuple[type[BaseException], ...]:
    """Exception types an HTTP client raises when the endpoint itself failed.

    ``ConnectionError`` covers the stdlib transport family; the HTTP client
    libraries are optional at import time because this module must import in a
    dependency-free install (the e2e harness), so each library contributes its
    exception base only when present. A library exception raised against the
    model endpoint -- connection reset, timeout, or any HTTP status the client
    surfaced as an error -- is a model-transport failure, not grader
    infrastructure.
    """
    types: list[type[BaseException]] = [ConnectionError]
    for module_name, exception_name in (
        ("requests.exceptions", "RequestException"),
        ("aiohttp", "ClientError"),
        ("openai", "APIError"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        exception_type = getattr(module, exception_name, None)
        if isinstance(exception_type, type) and issubclass(exception_type, BaseException):
            types.append(exception_type)
    return tuple(types)


def classify_task_exception(phase: FailurePhase, exception: BaseException) -> FailureCategory:
    """Map escaped exceptions onto the benchmark-independent failure taxonomy."""
    if phase is FailurePhase.PREPARATION:
        if isinstance(exception, (FileNotFoundError, ModuleNotFoundError, ImportError)):
            return FailureCategory.RESOURCE
        return FailureCategory.PREPARATION
    if phase is FailurePhase.SERIALIZATION:
        return FailureCategory.SERIALIZATION
    if isinstance(exception, TimeoutError):
        return FailureCategory.AGENT_TIMEOUT
    if phase in {FailurePhase.EVALUATION, FailurePhase.GRADING}:
        # Only the EVALUATION phase talks to the model endpoint; an HTTP client
        # error escaping there is model transport. The same error during
        # GRADING is a judge-side failure and stays grader infrastructure.
        if phase is FailurePhase.EVALUATION and isinstance(exception, transport_error_types()):
            return FailureCategory.MODEL_TRANSPORT
        return FailureCategory.GRADER_INFRASTRUCTURE
    if isinstance(exception, (ContextWindowExceededError, MissingContextLengthError, ModelRequestValidationError)):
        return FailureCategory.GENERATION_POLICY
    if isinstance(exception, transport_error_types()):
        return FailureCategory.MODEL_TRANSPORT
    return FailureCategory.GENERATION
