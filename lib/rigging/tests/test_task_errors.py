# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import httpx
import pytest
from rigging.task_errors import (
    DETERMINISTIC_INFRASTRUCTURE_ERROR_TYPES,
    ErrorCategory,
    ErrorIdentity,
    RetryDisposition,
    RetrySettings,
    exception_identity,
    record_retry_disposition,
    retry_delay,
)


class ProviderFailure(RuntimeError):
    def __init__(self, status_code):
        super().__init__("provider response text")
        self.status_code = status_code


@pytest.mark.parametrize(
    "status, expected",
    [
        (503, RetryDisposition.TRANSIENT),
        (HTTPStatus.UNAUTHORIZED, RetryDisposition.PERMANENT),
        (404, RetryDisposition.UNKNOWN),
        (409, RetryDisposition.TRANSIENT),
        (500, RetryDisposition.TRANSIENT),
        (599, RetryDisposition.TRANSIENT),
    ],
)
def test_wrapped_provider_identity_preserves_retryability_in_json(status, expected):
    provider = ProviderFailure(status)
    provider.__cause__ = ConnectionError("provider connection")
    error = RuntimeError("task operation")
    error.__cause__ = provider
    failure = exception_identity(error, operation="create")
    persisted = ErrorIdentity.model_validate_json(failure.model_dump_json())
    assert persisted.exception_type == "RuntimeError"
    assert persisted.cause_type == "ProviderFailure"
    assert persisted.provider_status_code == status
    assert persisted.operation == "create"
    assert persisted.retry_disposition is expected
    assert "provider response text" not in failure.model_dump_json()


def test_provider_hidden_transport_cause_is_retryable_without_persisting_request():
    try:
        raise httpx.ConnectError("request capability must remain private")
    except httpx.ConnectError:
        try:
            raise ProviderFailure(None) from None
        except ProviderFailure as error:
            failure = exception_identity(error)
    assert failure.cause_type == "ConnectError"
    assert failure.retry_disposition is RetryDisposition.TRANSIENT
    assert "request capability" not in failure.model_dump_json()


@pytest.mark.parametrize(
    "error, expected",
    [
        (httpx.ConnectError("connection lost"), RetryDisposition.TRANSIENT),
        (httpx.UnsupportedProtocol("invalid request scheme"), RetryDisposition.PERMANENT),
        (RuntimeError("unknown failure"), RetryDisposition.UNKNOWN),
    ],
)
def test_transport_identity_distinguishes_connection_and_request_failures(error, expected):
    wrapped = RuntimeError("task machine failed")
    wrapped.__cause__ = error
    assert exception_identity(wrapped).retry_disposition is expected


def test_retry_budget_backoff_and_cause_filter_survive_serialization():
    policy = RetrySettings(
        jitter=0,
        max_retries=3,
        min_wait_sec=2,
        wait_multiplier=2,
        max_wait_sec=5,
        include_exceptions={"ConnectionError"},
    )
    policy = RetrySettings.model_validate_json(policy.model_dump_json())
    failure = ErrorIdentity(
        exception_type="TaskMachineError",
        category=ErrorCategory.INFRASTRUCTURE,
        cause_type="ConnectionError",
        retry_disposition=RetryDisposition.TRANSIENT,
    )
    assert [retry_delay(failure, policy, attempt) for attempt in range(4)] == [2, 4, 5, None]
    excluded = policy.model_copy(update={"exclude_categories": {ErrorCategory.INFRASTRUCTURE}})
    assert retry_delay(failure, excluded, 0) is None
    permanent = failure.model_copy(update={"retry_disposition": RetryDisposition.PERMANENT})
    assert retry_delay(permanent, policy, 0) is None


def test_unknown_failure_requires_explicit_category_or_name_inclusion():
    failure = ErrorIdentity(exception_type="NewVerifierError")
    policy = RetrySettings(jitter=0, max_retries=1)
    assert retry_delay(failure, policy, 0) is None
    selected = policy.model_copy(update={"include_categories": {ErrorCategory.UNKNOWN}})
    assert retry_delay(failure, selected, 0) == 1
    excluded = selected.model_copy(update={"exclude_exceptions": {"NewVerifierError"}})
    assert retry_delay(failure, excluded, 0) is None


def test_a_known_wrapper_does_not_override_an_unknown_provider_status():
    failure = ErrorIdentity(exception_type="DaytonaConnectionError", provider_status_code=404)
    assert retry_delay(failure, RetrySettings(jitter=0, max_retries=1), 0) is None


def test_deterministic_infrastructure_names_are_permanent_regardless_of_cause():
    for name in sorted(DETERMINISTIC_INFRASTRUCTURE_ERROR_TYPES):
        assert record_retry_disposition(name, None, None) is RetryDisposition.PERMANENT
        assert record_retry_disposition(name, "ConnectionError", None) is RetryDisposition.PERMANENT


def test_infrastructure_selection_cannot_retry_a_deterministic_setup_failure():
    failure = ErrorIdentity(exception_type="EnvironmentSetupError", category=ErrorCategory.INFRASTRUCTURE)
    assert failure.retry_disposition is RetryDisposition.PERMANENT
    policy = RetrySettings(jitter=0, max_retries=3, include_categories={ErrorCategory.INFRASTRUCTURE})
    assert retry_delay(failure, policy, 0) is None


def test_framework_transient_names_keep_their_disposition_beside_deterministic_names():
    teardown = "PhaseCancellationError"
    assert record_retry_disposition(teardown, None, None, retryable_error_types=frozenset({teardown})) is (
        RetryDisposition.TRANSIENT
    )
    assert record_retry_disposition("EnvironmentSetupError", None, None) is RetryDisposition.PERMANENT
