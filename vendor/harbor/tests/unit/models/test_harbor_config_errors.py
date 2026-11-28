"""Check Harbor's category map, resume policy, and persisted exception identity."""

import inspect

import pytest
from harbor.models.trial.result import ExceptionInfo
from harbor.trial.errors import (
    AgentSetupTimeoutError, AgentTimeoutError, EnvironmentStartTimeoutError,
    RuntimeServiceError, TrialTimeoutError, VerifierTimeoutError,
)
from harbor_config.errors import ERROR_CATEGORIES, is_resume_retryable
from rigging.task_errors import (
    ErrorCategory,
    ErrorIdentity,
    RetryDisposition,
    error_category,
    errors_by_category,
    known_error_types,
)

# Modules whose exception classes can surface as a persisted trial
# ``ExceptionInfo.exception_type``. A new exception added to any of these is a
# public contract change and must receive an explicit category decision.
_PERSISTED_EXCEPTION_MODULES = [
    "harbor.agents.base",
    "harbor.agents.terminus_2.tmux_session",
    "harbor.trial.errors",
    "harbor.llms.base",
    "harbor.verifier.verifier",
    "harbor.environments.base",
    "harbor.environments.daytona",
    "harbor.environments.qemu",
    "harbor.environments.shellsim",
    "harbor.agents.installed.base",
    "harbor.utils.artifact_writer",
]


def _harbor_exception_names() -> set[str]:
    names: set[str] = set()
    for module_name in _PERSISTED_EXCEPTION_MODULES:
        module = pytest.importorskip(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, BaseException):
                continue
            # Only classes Harbor defines; skip stdlib bases re-exported here.
            if not cls.__module__.startswith("harbor"):
                continue
            names.add(cls.__name__)
    return names


def test_context_management_infrastructure_error_is_infrastructure() -> None:
    assert (
        error_category(
            "ContextManagementInfrastructureError", categories=ERROR_CATEGORIES
        )
        is ErrorCategory.INFRASTRUCTURE
    )


def test_llm_request_timeout_is_retryable_infrastructure() -> None:
    assert (
        error_category("LLMRequestTimeoutError", categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("LLMRequestTimeoutError")


def test_openai_connect_timeout_is_retryable_infrastructure() -> None:
    assert (
        error_category(
            "OpenAITransportConnectTimeoutError", categories=ERROR_CATEGORIES
        )
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("OpenAITransportConnectTimeoutError")


@pytest.mark.parametrize(
    "name", ["InternalServerError", "APIConnectionError", "APITimeoutError"]
)
def test_provider_transport_failure_is_retryable_infrastructure(name: str) -> None:
    assert (
        error_category(name, categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable(name)


def test_phase_cancellation_error_is_retryable_infrastructure() -> None:
    assert (
        error_category("PhaseCancellationError", categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("PhaseCancellationError")


def test_tau3_runtime_failure_is_retryable_infrastructure() -> None:
    assert (
        error_category("Tau3RuntimeInfrastructureError", categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("Tau3RuntimeInfrastructureError")


@pytest.mark.parametrize(
    "name",
    ["TmuxBatchProtocolError", "TmuxCommandError", "TmuxSessionEndedError"],
)
def test_tmux_control_failure_is_retryable_infrastructure(name: str) -> None:
    assert (
        error_category(name, categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable(name)


@pytest.mark.parametrize(
    "name",
    [
        "ArtifactPersistenceError",
        "ArtifactUploadTimeoutError",
        "ArtifactWriteTimeoutError",
        "ArtifactWriterBacklogError",
    ],
)
def test_artifact_persistence_failure_is_retryable_infrastructure(name: str) -> None:
    assert (
        error_category(name, categories=ERROR_CATEGORIES)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable(name)


@pytest.mark.parametrize(
    "name",
    sorted(_harbor_exception_names()),
)
def test_every_persisted_harbor_exception_has_an_explicit_category(name: str) -> None:
    """A new persisted Harbor exception without a category decision fails here."""
    assert name in known_error_types(categories=ERROR_CATEGORIES), (
        f"{name} is persisted as a trial outcome but has no entry in the "
        f"harbor_config.errors taxonomy; assign it an ErrorCategory."
    )


def test_unknown_exception_is_unknown() -> None:
    assert (
        error_category("SomeUnrecognizedException", categories=ERROR_CATEGORIES)
        is ErrorCategory.UNKNOWN
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("CancelledError", True),
        ("SandboxBuildFailedError", True),
        ("ContainerRegistryThrottledError", True),
        ("ContextBudgetExceededError", True),
        ("ContextManagementInfrastructureError", True),
        ("DaytonaProcessError", True),
        ("AgentTimeoutError", False),
        ("TurnCapExhaustedError", False),
        ("BadRequestError", False),
    ],
)
def test_is_resume_retryable(name: str, expected: bool) -> None:
    assert is_resume_retryable(name) is expected


def test_resume_retryable_is_distinct_from_category() -> None:
    # Retryable infrastructure failures are still categorized as infrastructure
    # for RL treatment; retryability and reward treatment are separate axes.
    for name in (
        "SandboxBuildFailedError",
        "ContextBudgetExceededError",
        "ContextManagementInfrastructureError",
        "TrialTimeoutError",
    ):
        assert is_resume_retryable(name)
        assert (
            error_category(name, categories=ERROR_CATEGORIES)
            is ErrorCategory.INFRASTRUCTURE
        )
    # CancelledError is resume-retryable but is not a Harbor-owned exception, so
    # it is absent from the category registry (consumers treat it separately).
    assert is_resume_retryable("CancelledError")
    assert (
        error_category("CancelledError", categories=ERROR_CATEGORIES)
        is ErrorCategory.UNKNOWN
    )


def test_native_exception_identity_round_trips_with_transport_cause():
    error = RuntimeServiceError("trial failed")
    error.__cause__ = TimeoutError("provider timed out")
    info = ExceptionInfo.from_exception(error)
    restored = ExceptionInfo.model_validate_json(info.model_dump_json())
    assert restored.exception_type == "RuntimeServiceError"
    assert restored.category is ErrorCategory.INFRASTRUCTURE
    assert restored.cause_type == "TimeoutError"
    assert restored.retry_disposition is RetryDisposition.TRANSIENT
    assert restored.exception_message == "trial failed"


@pytest.mark.parametrize("cause", [None, TimeoutError, PermissionError])
@pytest.mark.parametrize(
    "native_type, disposition",
    [
        (TrialTimeoutError, RetryDisposition.TRANSIENT),
        (EnvironmentStartTimeoutError, RetryDisposition.TRANSIENT),
        (AgentSetupTimeoutError, RetryDisposition.TRANSIENT),
        (AgentTimeoutError, RetryDisposition.UNKNOWN),
        (VerifierTimeoutError, RetryDisposition.UNKNOWN),
    ],
)
def test_native_timeout_identity_and_saved_records_survive_serialization(native_type, disposition, cause):
    error = native_type("deadline expired")
    if cause is not None:
        error.__cause__ = cause("phase deadline")
    native = ExceptionInfo.from_exception(error)
    saved = ExceptionInfo(
        exception_type=native_type.__name__, exception_message="deadline expired",
        exception_traceback="", occurred_at=native.occurred_at,
    )
    for failure in (native, saved):
        persisted = ErrorIdentity.model_validate_json(failure.model_dump_json())
        assert persisted.exception_type == native_type.__name__
        assert persisted.category == native.category
        expected = RetryDisposition.PERMANENT if cause is PermissionError and failure is native else disposition
        assert persisted.retry_disposition is expected
