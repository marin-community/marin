"""Contract tests for the shared Harbor error taxonomy.

These pin the published ``harbor_config.errors`` seam that downstream consumers
(MarinSkyRL's rollout adapter) rely on instead of duplicating Harbor exception
names. The taxonomy must (a) classify ``ContextManagementInfrastructureError``
as infrastructure, (b) cover every exception Harbor can persist as a trial
outcome so contract drift fails loudly, and (c) keep resume-retryability
distinct from the RL treatment category.
"""

import inspect

import pytest

from harbor_config import ErrorCategory, error_category, errors_by_category
from harbor_config.errors import is_resume_retryable, known_error_types

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
    "harbor.environments.daytona.environment",
    "harbor.environments.apptainer.apptainer",
    "harbor.environments.enroot.enroot",
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
        error_category("ContextManagementInfrastructureError")
        is ErrorCategory.INFRASTRUCTURE
    )


def test_llm_request_timeout_is_retryable_infrastructure() -> None:
    assert error_category("LLMRequestTimeoutError") is ErrorCategory.INFRASTRUCTURE
    assert is_resume_retryable("LLMRequestTimeoutError")


def test_openai_connect_timeout_is_retryable_infrastructure() -> None:
    assert (
        error_category("OpenAITransportConnectTimeoutError")
        is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("OpenAITransportConnectTimeoutError")


def test_phase_cancellation_error_is_retryable_infrastructure() -> None:
    assert error_category("PhaseCancellationError") is ErrorCategory.INFRASTRUCTURE
    assert is_resume_retryable("PhaseCancellationError")


def test_tau3_runtime_failure_is_retryable_infrastructure() -> None:
    assert (
        error_category("Tau3RuntimeInfrastructureError") is ErrorCategory.INFRASTRUCTURE
    )
    assert is_resume_retryable("Tau3RuntimeInfrastructureError")


@pytest.mark.parametrize(
    "name",
    ["TmuxBatchProtocolError", "TmuxCommandError", "TmuxSessionEndedError"],
)
def test_tmux_control_failure_is_retryable_infrastructure(name: str) -> None:
    assert error_category(name) is ErrorCategory.INFRASTRUCTURE
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
    assert error_category(name) is ErrorCategory.INFRASTRUCTURE
    assert is_resume_retryable(name)


def test_turn_cap_exhaustion_is_scored_passthrough() -> None:
    """A turn-capped rollout did real work; the verifier reward is usable."""
    assert error_category("TurnCapExhaustedError") is ErrorCategory.PASSTHROUGH
    assert "TurnCapExhaustedError" in errors_by_category(ErrorCategory.PASSTHROUGH)


@pytest.mark.parametrize(
    "name",
    sorted(_harbor_exception_names()),
)
def test_every_persisted_harbor_exception_has_an_explicit_category(name: str) -> None:
    """A new persisted Harbor exception without a category decision fails here."""
    assert name in known_error_types(), (
        f"{name} is persisted as a trial outcome but has no entry in the "
        f"harbor_config.errors taxonomy; assign it an ErrorCategory."
    )


def test_unknown_exception_is_unknown() -> None:
    assert error_category("SomeUnrecognizedException") is ErrorCategory.UNKNOWN


def test_errors_by_category_partitions_known_types() -> None:
    categorized = {
        name: cat for cat in ErrorCategory for name in errors_by_category(cat)
    }
    # Every known type appears under exactly one category.
    assert set(categorized) == known_error_types()
    # Agent timeouts and context-length failures are agent-attributable.
    assert "AgentTimeoutError" in errors_by_category(ErrorCategory.AGENT)
    assert "ContextLengthExceededError" in errors_by_category(ErrorCategory.AGENT)
    # The context-management infrastructure failure is masked from the baseline.
    assert "ContextManagementInfrastructureError" in errors_by_category(
        ErrorCategory.INFRASTRUCTURE
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
        assert error_category(name) is ErrorCategory.INFRASTRUCTURE
    # CancelledError is resume-retryable but is not a Harbor-owned exception, so
    # it is absent from the category registry (consumers treat it separately).
    assert is_resume_retryable("CancelledError")
    assert error_category("CancelledError") is ErrorCategory.UNKNOWN
