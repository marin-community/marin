# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import httpx
import pytest
from rigging.task_errors import (
    ErrorCategory,
    ErrorIdentity,
    RetryDisposition,
    RetrySettings,
    exception_identity,
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
    )
    policy = RetrySettings.model_validate_json(policy.model_dump_json())
    failure = ErrorIdentity(
        exception_type="TaskMachineError",
        category=ErrorCategory.INFRASTRUCTURE,
        cause_type="ConnectionError",
        retry_disposition=RetryDisposition.TRANSIENT,
    )
    assert [retry_delay(failure, policy, attempt) for attempt in range(4)] == [2, 4, 5, None]
    excluded = policy.model_copy(update={"exclude_exceptions": {"ConnectionError"}})
    assert retry_delay(failure, excluded, 0) is None
    permanent = failure.model_copy(update={"retry_disposition": RetryDisposition.PERMANENT})
    assert retry_delay(permanent, policy, 0) is None


def test_unknown_failure_stays_terminal_without_filters():
    failure = ErrorIdentity(exception_type="NewVerifierError")
    policy = RetrySettings(jitter=0, max_retries=1)
    assert retry_delay(failure, policy, 0) is None


def test_a_known_wrapper_does_not_override_an_unknown_provider_status():
    failure = ErrorIdentity(exception_type="DaytonaConnectionError", provider_status_code=404)
    assert retry_delay(failure, RetrySettings(jitter=0, max_retries=1), 0) is None


@pytest.mark.parametrize(
    "owned, cause, expected",
    [
        (RetryDisposition.TRANSIENT, None, RetryDisposition.TRANSIENT),
        (RetryDisposition.PERMANENT, None, RetryDisposition.PERMANENT),
        # A permanent cause outranks a producer's transient call.
        (RetryDisposition.TRANSIENT, PermissionError, RetryDisposition.PERMANENT),
        # A producer's permanent call outranks a transient transport cause.
        (RetryDisposition.PERMANENT, ConnectionError, RetryDisposition.PERMANENT),
    ],
)
def test_exception_identity_keeps_a_producer_owned_disposition(owned, cause, expected):
    class ProducerClassified(RuntimeError):
        retry_disposition = owned

    error = ProducerClassified("producer classified this failure")
    if cause is not None:
        error.__cause__ = cause("cause evidence")
    identity = exception_identity(error)
    assert identity.retry_disposition is expected
    assert identity.cause_type == (None if cause is None else cause.__name__)
