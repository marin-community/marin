# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import httpx
import pytest
from rigging.task_errors import ErrorIdentity, RetryDisposition, exception_identity


class ProviderFailure(RuntimeError):
    def __init__(self, status_code):
        super().__init__("provider response text")
        self.status_code = status_code


@pytest.mark.parametrize(
    "status, expected",
    [
        (503, RetryDisposition.TRANSIENT),
        (HTTPStatus.UNAUTHORIZED, RetryDisposition.PERMANENT),
        (401, RetryDisposition.PERMANENT),
        (409, RetryDisposition.UNKNOWN),
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
