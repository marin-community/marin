# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from rigging.task_errors import (
    ErrorCategory,
    ErrorIdentity,
    RetryDisposition,
    RetrySettings,
    error_category,
    retry_delay,
)


@pytest.mark.parametrize(
    "name, expected",
    json.loads((Path(__file__).parent / "fixtures/task_error_categories.json").read_text()).items(),
)
def test_existing_error_category_is_preserved(name, expected):
    assert error_category(name).value == expected


@pytest.mark.parametrize(
    "failure, settings, expected",
    [
        (ErrorIdentity("APIConnectionError"), RetrySettings(max_retries=2), 1.0),
        (ErrorIdentity("ModelAuthenticationError"), RetrySettings(max_retries=2), None),
        (ErrorIdentity("UnrecognizedFailure"), RetrySettings(max_retries=2), None),
        (
            ErrorIdentity("UnrecognizedFailure"),
            RetrySettings(max_retries=2, include_categories={ErrorCategory.UNKNOWN}),
            1.0,
        ),
        (
            ErrorIdentity("TaskMachineError", cause_type="ProviderBusy", retry_disposition=RetryDisposition.TRANSIENT),
            RetrySettings(max_retries=2, include_exceptions={"ProviderBusy"}),
            1.0,
        ),
        (
            ErrorIdentity("TaskMachineError", cause_type="ProviderBusy", retry_disposition=RetryDisposition.TRANSIENT),
            RetrySettings(
                max_retries=2, include_categories={ErrorCategory.INFRASTRUCTURE}, exclude_exceptions={"ProviderBusy"}
            ),
            None,
        ),
        (
            ErrorIdentity("APIConnectionError"),
            RetrySettings(
                max_retries=2,
                include_exceptions={"APIConnectionError"},
                exclude_categories={ErrorCategory.INFRASTRUCTURE},
            ),
            None,
        ),
        (
            ErrorIdentity("ModelAuthenticationError"),
            RetrySettings(max_retries=2, include_categories={ErrorCategory.INFRASTRUCTURE}),
            None,
        ),
    ],
)
def test_retry_policy_resolves_identity_and_filter_precedence(failure, settings, expected):
    assert retry_delay(failure, settings, retries=0) == expected


def test_retry_budget_and_backoff_are_independent_of_treatment_category():
    failure = ErrorIdentity("APIConnectionError")
    settings = RetrySettings(max_retries=4, min_wait_sec=2, wait_multiplier=3, max_wait_sec=10)
    assert [retry_delay(failure, settings, retries=n) for n in range(5)] == [2, 6, 10, 10, None]
    assert error_category(failure.exception_type) is ErrorCategory.INFRASTRUCTURE
