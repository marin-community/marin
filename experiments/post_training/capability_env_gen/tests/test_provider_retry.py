from types import SimpleNamespace

import pytest

from capability_pipeline.provider_retry import (
    ProviderRateLimitExhausted,
    provision_with_rate_limit_retry,
)


class DaytonaRateLimitError(Exception):
    def __init__(self, *, headers=None, status_code=429, error_code="throttled"):
        self.headers = headers or {}
        self.status_code = status_code
        self.error_code = error_code


def test_provisioning_retries_rate_limit_and_records_every_attempt():
    calls = 0
    delays = []

    def create():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise DaytonaRateLimitError()
        return SimpleNamespace(id="fresh-sandbox")

    sandbox, attempts = provision_with_rate_limit_retry(create, sleep=delays.append)
    assert sandbox.id == "fresh-sandbox"
    assert delays == [5.0, 10.0]
    assert attempts == [
        {
            "attempt": 1,
            "state": "rate_limited",
            "status_code": 429,
            "error_code": "throttled",
            "retry_after_seconds": 5.0,
        },
        {
            "attempt": 2,
            "state": "rate_limited",
            "status_code": 429,
            "error_code": "throttled",
            "retry_after_seconds": 10.0,
        },
        {"attempt": 3, "state": "created"},
    ]


def test_provisioning_honors_bounded_retry_after_and_exhausts_truthfully():
    delays = []

    def create():
        raise DaytonaRateLimitError(headers={"Retry-After": "120"})

    with pytest.raises(ProviderRateLimitExhausted) as failure:
        provision_with_rate_limit_retry(create, sleep=delays.append, max_attempts=2)
    assert delays == [60.0]
    assert failure.value.attempts[-1] == {
        "attempt": 2,
        "state": "rate_limited",
        "status_code": 429,
        "error_code": "throttled",
    }


def test_provisioning_does_not_retry_unrelated_provider_failure():
    calls = 0

    def create():
        nonlocal calls
        calls += 1
        raise RuntimeError("semantic execution failure")

    with pytest.raises(RuntimeError, match="semantic execution failure"):
        provision_with_rate_limit_retry(create, sleep=lambda _: None)
    assert calls == 1


@pytest.mark.parametrize("headers", [None, {"Retry-After": "nan"}])
def test_provisioning_invalid_retry_after_uses_bounded_fallback(headers):
    delays = []
    calls = 0

    def create():
        nonlocal calls
        calls += 1
        if calls == 1:
            error = DaytonaRateLimitError()
            error.headers = headers
            raise error
        return SimpleNamespace(id="fresh-sandbox")

    _, attempts = provision_with_rate_limit_retry(create, sleep=delays.append)
    assert delays == [5.0]
    assert attempts[0]["retry_after_seconds"] == 5.0
