# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests
from marin.inference.types import OpenAIEndpoint
from tokenizers import pre_tokenizers

from experiments.post_training.chat_template import GRUG_TOKENS
from experiments.post_training.probe_sets import continuations
from experiments.post_training.probe_sets.continuations import (
    continuation_row,
    render_prompt_ids,
    request_body,
    sample_token_ids,
)
from experiments.post_training.probe_sets.tokenizer_test_support import MARKERS, byte_level_tokenizer
from experiments.post_training.stage0.protocol_gate import Decode

PROBE = {"probe_id": "p-0", "source": "aime_2024", "messages": [{"role": "user", "content": "add two and five"}]}


def byte_ids(tokenizer, text: str) -> list[int]:
    """``text`` as one token per byte: a valid but non-canonical tokenization."""
    return tokenizer.convert_tokens_to_ids(
        list(pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(text)[0][0])
    )


def reply_tags(row, prompt_ids):
    return row["token_type_tags"][len(prompt_ids) :]


def test_an_ended_reply_keeps_b_s_own_tokenization_and_tags_its_thinking():
    tokenizer = byte_level_tokenizer()
    prompt_ids = render_prompt_ids(tokenizer, "grug", PROBE["messages"])
    close, think_open, think_close = (tokenizer.convert_tokens_to_ids(t) for t in MARKERS[3:6])
    carry = byte_ids(tokenizer, "carry")  # B spelled it byte by byte; the tokenizer would merge it
    assert carry != tokenizer.encode("carry", add_special_tokens=False)
    answer = tokenizer.encode("it is 7", add_special_tokens=False)
    response = [think_open, *carry, think_close, *answer, close]
    row = continuation_row(tokenizer, "grug", "P_think", PROBE, prompt_ids, response)
    assert row["token_ids"] == prompt_ids + response and row["turn_ended"]
    expected = ["delimiter", *["reasoning"] * len(carry), "delimiter", *["final_message"] * len(answer), "delimiter"]
    assert reply_tags(row, prompt_ids) == expected
    assert len(row["token_type_tags"]) == len(row["token_ids"]) == len(row["position_tags"])


def test_a_reply_cut_at_the_cap_gets_no_closing_token():
    tokenizer = byte_level_tokenizer()
    prompt_ids = render_prompt_ids(tokenizer, "grug", PROBE["messages"])
    think_open = tokenizer.convert_tokens_to_ids(GRUG_TOKENS.think_open)
    response = [think_open, *tokenizer.encode("carry carry", add_special_tokens=False)]  # thinking never closed
    row = continuation_row(tokenizer, "grug", "P_think", PROBE, prompt_ids, response)
    assert row["token_ids"] == prompt_ids + response and not row["turn_ended"]
    assert reply_tags(row, prompt_ids) == ["delimiter", *["reasoning"] * (len(response) - 1)]


def test_stray_bytes_that_form_no_character_keep_their_alignment():
    tokenizer = byte_level_tokenizer()
    prompt_ids = render_prompt_ids(tokenizer, "grug", PROBE["messages"])
    close = tokenizer.convert_tokens_to_ids(GRUG_TOKENS.turn_close)
    half = byte_ids(tokenizer, "é")[:1]  # the first byte of a two-byte character
    answer = tokenizer.encode("it is 7", add_special_tokens=False)
    response = [*half, *answer, close]
    row = continuation_row(tokenizer, "grug", "P_think", PROBE, prompt_ids, response)
    assert row["token_ids"] == prompt_ids + response
    assert reply_tags(row, prompt_ids) == ["final_message"] * (1 + len(answer)) + ["delimiter"]


def test_the_request_samples_token_ids_and_stops_only_on_the_turn_close():
    body = request_body("b", [1, 2], seed=3, decode=Decode(repetition_penalty=1.05), stop_id=9)
    assert body["prompt"] == [1, 2] and body["return_token_ids"] and body["ignore_eos"]
    assert body["stop_token_ids"] == [9] and body["repetition_penalty"] == 1.05 and not body["skip_special_tokens"]


class _Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._payload


def test_a_transient_endpoint_failure_is_retried(monkeypatch):
    # Regression: one 504 from the shared vLLM endpoint killed a finished-almost
    # generation pass. Seeded bodies reproduce the sample, so retry.
    endpoint = OpenAIEndpoint(base_url="http://endpoint/v1", model="b")
    attempts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        attempts.append(url)
        if len(attempts) < 3:
            return _Response(504)
        return _Response(200, {"choices": [{"token_ids": [5, 6]}]})

    monkeypatch.setattr(continuations.requests, "post", fake_post)
    monkeypatch.setattr(continuations.time, "sleep", lambda _seconds: None)
    assert sample_token_ids(endpoint, {"model": "b"}) == [5, 6]
    assert len(attempts) == 3


def test_a_client_error_raises_without_retry(monkeypatch):
    endpoint = OpenAIEndpoint(base_url="http://endpoint/v1", model="b")
    attempts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        attempts.append(url)
        return _Response(400)

    monkeypatch.setattr(continuations.requests, "post", fake_post)
    monkeypatch.setattr(continuations.time, "sleep", lambda _seconds: None)
    with pytest.raises(requests.HTTPError):
        sample_token_ids(endpoint, {"model": "b"})
    assert len(attempts) == 1


def _probe_row(probe_id: str) -> dict:
    return {"probe_id": probe_id, "source": "aime_2024", "messages": [{"role": "user", "content": "add two and five"}]}


def test_a_killed_pass_restarts_from_the_raw_file(monkeypatch, tmp_path):
    # Regression: two passes were lost whole (a 504, then a job timeout) because
    # the raw file landed only after the last completion. Now the file is
    # rewritten as completions land and a restart samples only what is missing.
    endpoint = OpenAIEndpoint(base_url="http://endpoint/v1", model="b")
    raw_path = str(tmp_path / "P_X.jsonl")
    rows = [_probe_row(f"p-{index}") for index in range(3)]
    prompt_ids = [[1], [2], [3]]
    done = {
        "p-0": {"probe_id": "p-0", "prompt_ids": [1], "response_ids": [10, 11]},
        "p-1": {"probe_id": "p-1", "prompt_ids": [2], "response_ids": [20]},
    }
    raw_path_file = tmp_path / "P_X.jsonl"
    raw_path_file.write_text("".join(json.dumps(done[probe]) + "\n" for probe in ("p-0", "p-1")))

    sampled_ids = iter([[30], [31]])

    def fake_sample(_endpoint, _body):
        return next(sampled_ids)

    monkeypatch.setattr(continuations, "sample_token_ids", fake_sample)
    responses = continuations._sampled_responses(endpoint, rows, prompt_ids, stop_id=9, raw_path=raw_path)
    assert responses == [[10, 11], [20], [30]]

    # The raw file holds all three probes, so a third pass samples nothing.
    stored = {row["probe_id"]: row for row in continuations.read_jsonl(raw_path)}
    assert sorted(stored) == ["p-0", "p-1", "p-2"]
    sampled_ids2 = iter([])
    monkeypatch.setattr(continuations, "sample_token_ids", lambda _e, _b: next(sampled_ids2))
    assert continuations._sampled_responses(endpoint, rows, prompt_ids, stop_id=9, raw_path=raw_path) == responses


def test_a_raw_file_with_a_different_prompt_is_rejected(monkeypatch, tmp_path):
    endpoint = OpenAIEndpoint(base_url="http://endpoint/v1", model="b")
    raw_path = str(tmp_path / "P_X.jsonl")
    rows = [_probe_row("p-0")]
    raw_path_file = tmp_path / "P_X.jsonl"
    raw_path_file.write_text(json.dumps({"probe_id": "p-0", "prompt_ids": [99], "response_ids": [1]}) + "\n")
    with pytest.raises(ValueError, match="different prompt"):
        continuations._sampled_responses(endpoint, rows, [[1]], stop_id=9, raw_path=raw_path)
