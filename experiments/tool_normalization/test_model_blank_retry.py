# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded blank-answer retry policy for the GLM pool client (2026-11-24 full-32-r2).

Covers the confirmed reasoning-only empty-answer failure class: one retry with entry
rotation, second blank fatal, malformed JSON/schema still fatal, first-attempt count-only
metadata preserved in the terminal receipt.
"""

import json
import time
from pathlib import Path

import pytest

from experiments.tool_normalization import model as model_module
from experiments.tool_normalization.model import (
    ModelConfig,
    PermanentRequestError,
    complete,
    initialize_ledger,
)

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class FakeResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return b""


class FakeStream:
    def __init__(self, response: FakeResponse):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *args):
        return False


class FakeClient:
    """Scripted per-attempt SSE responses; records the POST URL for rotation asserts.

    The script list is SHARED and consumed in place across client instantiations: the
    production code builds a new httpx.Client per attempt, so per-attempt state must
    live outside the client object.
    """

    posted: list[str] = []

    def __init__(self, script: list[list[str]], posted: list[str]):
        self._script = script
        self._posted = posted

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, **kwargs):
        type(self).posted = self._posted
        self._posted.append(url)
        return FakeStream(FakeResponse(self._script.pop(0)))


class StubTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [1, 2, 3]}


def sse_line(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def blank_attempt_lines(reasoning_key: str = "reasoning_content") -> list[str]:
    """done + stop + nonempty reasoning + positive completion usage + EMPTY content.

    reasoning_key selects the delta field name: vLLM 0.34 renamed reasoning_content to
    reasoning; the client accumulator must cover both or a real blank would miss the
    nonempty-reasoning predicate.
    """
    return [
        sse_line({"choices": [{"delta": {reasoning_key: "thinking through the mapping"}}]}),
        sse_line({"choices": [], "usage": USAGE}),
        sse_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]


def valid_attempt_lines() -> list[str]:
    return [
        sse_line({"choices": [{"delta": {"content": '{"ok": true}'}}]}),
        sse_line({"choices": [], "usage": USAGE}),
        sse_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]


def malformed_attempt_lines() -> list[str]:
    return [
        sse_line({"choices": [{"delta": {"content": "not-json"}}]}),
        sse_line({"choices": [], "usage": USAGE}),
        sse_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]


@pytest.fixture
def pool_config(tmp_path: Path) -> ModelConfig:
    (tmp_path / "keys").mkdir()
    entries = []
    for i, name in enumerate(("a", "b")):
        key = tmp_path / "keys" / f"key-{i}.txt"
        key.write_text(f"jwt-{name}")
        entries.append((f"http://entry-{name}/v1", str(key), f"s3://bucket/key-{i}.jwt"))
    (tmp_path / "capacity.json").write_text(
        json.dumps({"generation": "g", "conversion_slots": 4, "science_slots": 0, "expires_at_unix": time.time() + 3600})
    )
    stop_file = tmp_path / "STOP"
    config = ModelConfig(
        base_url="http://entry-a/v1",
        model="glm-5.3",
        temperature=0.2,
        top_p=0.9,
        thinking_token_budget=0,
        generation="g",
        api_key_file=tmp_path / "keys" / "key-0.txt",
        context_tokens=100_000,
        output_tokens=1_000,
        request_timeout=30,
        capacity_file=tmp_path / "capacity.json",
        ledger=tmp_path / "ledger.sqlite3",
        max_attempts=4,
        stop_file=stop_file,
        deadline_unix=time.time() + 600,
        relay_job=None,
        pool_entries=tuple(entries),
    )
    initialize_ledger(config)
    return config


def _run_complete(monkeypatch, tmp_path: Path, config: ModelConfig, script: list[list[str]]) -> tuple[list[str], Path]:
    posted: list[str] = []
    monkeypatch.setattr(model_module.httpx, "Client", lambda **kwargs: FakeClient(script, posted))
    # complete() creates the stage directory itself (exist_ok=False).
    directory = tmp_path / "stage"
    value = complete(config, StubTokenizer(), directory, "system", {}, SCHEMA, reasoning_effort="low")
    assert value == {"ok": True}
    return posted, directory


def test_blank_answer_retries_once_and_succeeds(monkeypatch, tmp_path, pool_config) -> None:
    posted, directory = _run_complete(monkeypatch, tmp_path, pool_config, [blank_attempt_lines(), valid_attempt_lines()])
    # Exactly one retry, rotated to the next entry.
    assert [u.split("//")[1].split("/")[0] for u in posted] == ["entry-a", "entry-b"]
    receipt = json.loads((directory / "receipt.json").read_text())
    # The recovered failure stays visible: count-only metadata for the blank attempt,
    # including the endpoint the failed attempt actually used (rotation moved the retry).
    assert len(receipt["blank_attempts"]) == 1
    blank = receipt["blank_attempts"][0]
    assert blank["content_chars"] == 0
    assert blank["reasoning_chars"] > 0
    assert blank["usage"] == USAGE
    assert blank["finish_reason"] == "stop"
    assert blank["base_url"] == "http://entry-a/v1"
    assert blank["model"] == "glm-5.3"
    assert receipt["lease_state"] == "terminal"


def test_second_blank_is_fatal(monkeypatch, tmp_path, pool_config) -> None:
    posted: list[str] = []
    script = [blank_attempt_lines(), blank_attempt_lines()]
    monkeypatch.setattr(model_module.httpx, "Client", lambda **kwargs: FakeClient(script, posted))
    directory = tmp_path / "stage2"
    with pytest.raises(PermanentRequestError, match="blank answer persisted"):
        complete(pool_config, StubTokenizer(), directory, "system", {}, SCHEMA, reasoning_effort="low")
    assert [u.split("//")[1].split("/")[0] for u in posted] == ["entry-a", "entry-b"]
    receipt = json.loads((directory / "receipt.json").read_text())
    assert len(receipt["blank_attempts"]) == 2
    assert receipt["lease_state"] == "terminal"


def test_malformed_json_remains_fatal(monkeypatch, tmp_path, pool_config) -> None:
    monkeypatch.setattr(model_module.httpx, "Client", lambda **kwargs: FakeClient([malformed_attempt_lines()], []))
    directory = tmp_path / "stage3"
    with pytest.raises(PermanentRequestError, match="schema violation"):
        complete(pool_config, StubTokenizer(), directory, "system", {}, SCHEMA, reasoning_effort="low")
    receipt = json.loads((directory / "receipt.json").read_text())
    assert "blank_attempts" not in receipt


def test_blank_predicate_covers_vllm_reasoning_rename(monkeypatch, tmp_path, pool_config) -> None:
    # vLLM 0.34 renamed delta.reasoning_content -> delta.reasoning: the blank predicate
    # must see nonempty reasoning through EITHER field name or a real blank would skip
    # the bounded retry and die as a schema violation.
    script = [blank_attempt_lines(reasoning_key="reasoning"), valid_attempt_lines()]
    posted, directory = _run_complete(monkeypatch, tmp_path, pool_config, script)
    assert [u.split("//")[1].split("/")[0] for u in posted] == ["entry-a", "entry-b"]
    receipt = json.loads((directory / "receipt.json").read_text())
    assert len(receipt["blank_attempts"]) == 1
    assert receipt["blank_attempts"][0]["reasoning_chars"] > 0
