# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import requests
from huggingface_hub.errors import HfHubHTTPError

import levanter.compat.hf_checkpoints as hf_checkpoints
from levanter.compat.hf_checkpoints import load_tokenizer


def test_load_tokenizer_retries_transient_hf_errors(monkeypatch):
    calls = 0
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def fake_from_pretrained(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            response = requests.Response()
            response.status_code = 504
            raise HfHubHTTPError("Gateway Time-out", response=response)
        return object()

    monkeypatch.setattr(hf_checkpoints.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(hf_checkpoints.time, "sleep", fake_sleep)
    monkeypatch.setattr(hf_checkpoints.random, "random", lambda: 0.0)

    tokenizer = load_tokenizer("dummy-model-id")
    assert tokenizer is not None
    assert calls == 3
    assert len(sleep_calls) == 2
