# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.tool_normalization import model, relay
from experiments.tool_normalization.model import (
    CapacityUnavailable,
    LengthDeferred,
    ModelConfig,
    complete,
    initialize_ledger,
)
from experiments.tool_normalization.verify import model_value

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


@pytest.fixture
def request_config(tmp_path):
    (tmp_path / "key").write_text("test-key")
    (tmp_path / "capacity.json").write_text(
        json.dumps(
            {
                "generation": "test",
                "conversion_slots": 1,
                "science_slots": 0,
                "expires_at_unix": time.time() + 3600,
            }
        )
    )
    config = ModelConfig(
        base_url="http://old/v1",
        model="glm-5.3",
        temperature=1,
        top_p=0.95,
        thinking_token_budget=10,
        generation="test",
        api_key_file=tmp_path / "key",
        context_tokens=1000,
        output_tokens=100,
        request_timeout=60,
        capacity_file=tmp_path / "capacity.json",
        ledger=tmp_path / "ledger.sqlite3",
        max_attempts=3,
        stop_file=tmp_path / "STOP",
        deadline_unix=time.time() + 600,
        relay_job=None,
    )
    initialize_ledger(config)
    return config


@pytest.fixture
def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}",
    )


@pytest.fixture
def clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(model.time, "time", lambda: now[0])
    monkeypatch.setattr(model.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def transport(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


def completion(finish="stop", content='{"ok":true}'):
    event = {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
    return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")


def invoke(config, tokenizer, directory):
    return complete(config, tokenizer, directory, "convert", {}, SCHEMA, reasoning_effort="low")


def lease_state(config):
    with sqlite3.connect(config.ledger) as db:
        return db.execute("SELECT state FROM requests").fetchone()[0]


def test_complete_output_length_defers_and_releases_lease(tmp_path, monkeypatch, request_config, tokenizer):
    requests = []

    def handler(request):
        requests.append(request)
        return completion("length", "truncated")

    transport(monkeypatch, handler)
    directory = tmp_path / "request"
    with pytest.raises(LengthDeferred):
        invoke(request_config, tokenizer, directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["finish_reason"] == "length"
    assert len(requests) == 1
    assert lease_state(request_config) == "terminal"


@pytest.mark.parametrize("failure", ["outage", "schema", "connection"])
def test_complete_repeated_failures_exhaust_attempts(tmp_path, monkeypatch, request_config, tokenizer, clock, failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "connection":
            raise httpx.ConnectError("relay moved", request=request)
        return httpx.Response(503) if failure == "outage" else completion(content='{"wrong":0}')

    transport(monkeypatch, handler)
    directory = tmp_path / "request"
    with pytest.raises((httpx.ConnectError, model.TransientRequestError, model.jsonschema.ValidationError)):
        invoke(request_config, tokenizer, directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert len(requests) == request_config.max_attempts
    assert len(receipt["attempts"]) == request_config.max_attempts
    assert lease_state(request_config) == ("terminal" if failure == "schema" else "uncertain")


@pytest.mark.parametrize("stop", ["file", "deadline"])
def test_complete_cancellation_during_retry_preserves_uncertain_lease(
    tmp_path, monkeypatch, request_config, tokenizer, clock, stop
):
    requests = []

    def handler(request):
        requests.append(request)
        if stop == "file":
            request_config.stop_file.touch()
        else:
            clock[0] = request_config.deadline_unix
        raise httpx.ConnectError("relay unavailable", request=request)

    transport(monkeypatch, handler)
    with pytest.raises(CapacityUnavailable):
        invoke(request_config, tokenizer, tmp_path / "request")
    assert len(requests) == 1
    assert lease_state(request_config) == "uncertain"


def test_complete_moved_relay_refreshes_endpoint_and_verifies_receipt(
    tmp_path, monkeypatch, request_config, tokenizer, clock
):
    endpoints = SimpleNamespace(endpoints=[SimpleNamespace(url="http://new")])
    resolver = SimpleNamespace(resolve=lambda name: endpoints)
    client = SimpleNamespace(resolver_for_job=lambda job: resolver)
    monkeypatch.setattr(relay, "iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr(
        relay.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"data":[{"id":"glm-5.3"}]}')
    )
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if request.url.host == "old":
            raise httpx.ConnectError("relay moved", request=request)
        return completion()

    transport(monkeypatch, handler)
    directory = tmp_path / "request"
    result = invoke(replace(request_config, relay_job=relay.RELAY_JOB), tokenizer, directory)
    assert result == {"ok": True}
    assert urls == ["http://old/v1/chat/completions", "http://new/v1/chat/completions"]
    assert model_value(directory, SCHEMA) == result
    assert lease_state(request_config) == "terminal"
