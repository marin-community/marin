# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datasets
import httpx
import pytest
from huggingface_hub.utils import HfHubHTTPError
from marin.utils import load_dataset_with_backoff


def _hub_error(status_code: int) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/org/name")
    return HfHubHTTPError("hub request failed", response=httpx.Response(status_code, request=request))


def _flaky_load_dataset(monkeypatch: pytest.MonkeyPatch, error: Exception) -> list[dict]:
    calls: list[dict] = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return "loaded"

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


@pytest.mark.parametrize(
    "error",
    [
        _hub_error(503),
        _hub_error(429),
        httpx.ConnectError("connection refused"),
    ],
)
def test_transient_hub_errors_are_retried(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    calls = _flaky_load_dataset(monkeypatch, error)

    result = load_dataset_with_backoff(context="test", initial_delay=0.001, max_delay=0.001, path="org/name")

    assert result == "loaded"
    assert len(calls) == 2


def test_client_hub_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    error = _hub_error(404)
    calls = _flaky_load_dataset(monkeypatch, error)

    with pytest.raises(HfHubHTTPError):
        load_dataset_with_backoff(context="test", initial_delay=0.001, max_delay=0.001, path="org/name")

    assert len(calls) == 1
