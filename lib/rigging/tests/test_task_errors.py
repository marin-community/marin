# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import httpx
import pytest
from rigging.task_errors import ErrorIdentity, RetryDisposition, error_category, exception_identity


@pytest.mark.parametrize(
    "name, expected",
    json.loads((Path(__file__).parent / "fixtures/task_error_categories.json").read_text()).items(),
)
def test_existing_error_category_is_preserved(name, expected):
    assert error_category(name).value == expected


class ProviderFailure(RuntimeError):
    def __init__(self, status_code):
        super().__init__("provider response text")
        self.status_code = status_code


@pytest.mark.parametrize(
    "status, expected",
    [(503, RetryDisposition.TRANSIENT), (401, RetryDisposition.PERMANENT), (409, RetryDisposition.UNKNOWN)],
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
