import httpx
from huggingface_hub.errors import RepositoryNotFoundError
import pytest
import rigging.timing

from marinskyrl.hugging_face_retry import call_with_hugging_face_retry


def test_hugging_face_retry_recovers_from_transport_failure(monkeypatch) -> None:
    attempts = 0
    delays = []

    def flaky_read() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadError("injected interrupted Hub read")
        return "complete"

    monkeypatch.setattr(rigging.timing.time, "sleep", delays.append)

    assert call_with_hugging_face_retry(flaky_read, operation="test Hub read") == "complete"
    assert attempts == 3
    assert delays == [2.0, 4.0]


def test_hugging_face_retry_does_not_retry_missing_repository(monkeypatch) -> None:
    attempts = 0
    request = httpx.Request("GET", "https://huggingface.co/missing/model")
    error = RepositoryNotFoundError("missing repository", response=httpx.Response(404, request=request))

    def missing_repository() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _delay: pytest.fail("fatal error was retried"))

    with pytest.raises(RepositoryNotFoundError):
        call_with_hugging_face_retry(missing_repository, operation="test missing repository")
    assert attempts == 1
