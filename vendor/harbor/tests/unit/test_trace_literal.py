"""Tests for the trace_literal feature.

Covers three layers:
  1. The ``SUPPORTS_LITERAL_TRACES`` agent capability flag.
  2. ``traces_utils`` literal token extraction (``_extract_literal_turns``)
     and the ``include_literal_tokens`` plumbing through
     ``collect_conversations_from_trial``.
  3. The recording proxy's pure helpers
     (``inject_literal_params``, ``extract_literal_from_response``) and its
     end-to-end ``forward()`` recording path.
"""

import asyncio
import json

import httpx
from starlette.testclient import TestClient
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.literal import (
    extract_literal_from_response,
    inject_literal_params,
)
from harbor.literal.proxy import CORRELATION_ID_HEADER, TASK_AGENT_HEADER, RecordProxy
from harbor.utils.traces_utils import (
    _extract_literal_turns,
    collect_conversations_from_trial,
)


# ---------------------------------------------------------------------------
# 1. Capability flag
# ---------------------------------------------------------------------------


def test_supports_literal_traces_default_false():
    assert BaseAgent.SUPPORTS_LITERAL_TRACES is False


def test_terminus_2_supports_literal_traces():
    assert Terminus2.SUPPORTS_LITERAL_TRACES is True


def test_installed_text_only_agents_do_not_support_literal():
    # Installed agents that only parse CLI logs must NOT claim literal support.
    # ClaudeCode reconstructs trajectories from CLI event files and never sees
    # token IDs, so it correctly leaves the flag at its default.
    assert ClaudeCode.SUPPORTS_LITERAL_TRACES is False


# ---------------------------------------------------------------------------
# Helpers for trajectory fixtures
# ---------------------------------------------------------------------------


def _base_run_meta(trial_name: str) -> dict:
    return {
        "agent_name": "terminus-2",
        "model_name": "test-model",
        "model_provider": "test-provider",
        "start_time": "2024-01-01T00:00:00Z",
        "task_name": "test-task",
        "trial_name": trial_name,
        "run_id": "test-run",
    }


def _write_trajectory(trial_dir: Path, steps: list[dict]) -> None:
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {
        "agent": {"name": "terminus-2", "model_name": "test-model"},
        "steps": steps,
    }
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))


_LITERAL_AGENT_STEP = {
    "source": "agent",
    "message": "All done.",
    "metrics": {
        "prompt_token_ids": [1, 2, 3],
        "completion_token_ids": [4, 5],
        "logprobs": [-0.1, -0.2],
    },
    "observation": {"results": [{"content": "output"}]},
}


# ---------------------------------------------------------------------------
# 2a. _extract_literal_turns
# ---------------------------------------------------------------------------


def test_extract_literal_turns_populates_from_metrics():
    steps = [
        {"source": "system", "message": "Task."},
        _LITERAL_AGENT_STEP,
    ]
    turns = _extract_literal_turns(steps)
    assert len(turns) == 1
    assert turns[0]["prompt_token_ids"] == [1, 2, 3]
    assert turns[0]["completion_token_ids"] == [4, 5]
    assert turns[0]["logprobs"] == [-0.1, -0.2]


def test_extract_literal_turns_empty_when_no_metrics():
    steps = [
        {"source": "system", "message": "Task."},
        {"source": "agent", "message": "All done."},
    ]
    assert _extract_literal_turns(steps) == []


def test_extract_literal_turns_skips_copied_context():
    steps = [
        {
            "source": "agent",
            "is_copied_context": True,
            "message": "copied",
            "metrics": {"prompt_token_ids": [9], "completion_token_ids": [9]},
        },
        _LITERAL_AGENT_STEP,
    ]
    turns = _extract_literal_turns(steps)
    assert len(turns) == 1
    assert turns[0]["prompt_token_ids"] == [1, 2, 3]


def test_extract_literal_turns_partial_metrics():
    # logprobs may be absent (e.g. OpenHands-SDK path) — turn still recorded.
    steps = [
        {
            "source": "agent",
            "message": "hi",
            "metrics": {"prompt_token_ids": [1], "completion_token_ids": [2]},
        }
    ]
    turns = _extract_literal_turns(steps)
    assert len(turns) == 1
    assert turns[0]["logprobs"] is None


# ---------------------------------------------------------------------------
# 2b. collect_conversations_from_trial plumbing
# ---------------------------------------------------------------------------


def test_collect_with_literal_emits_token_columns(tmp_path):
    trial_dir = tmp_path / "trial-1"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task instructions."},
            _LITERAL_AGENT_STEP,
        ],
    )
    convs = collect_conversations_from_trial(
        trial_dir, _base_run_meta("trial-1"), include_literal_tokens=True
    )
    assert len(convs) == 1
    row = convs[0]
    assert row["prompt_token_ids"] == [[1, 2, 3]]
    assert row["completion_token_ids"] == [[4, 5]]
    assert row["logprobs"] == [[-0.1, -0.2]]


def test_collect_without_literal_omits_token_columns(tmp_path):
    trial_dir = tmp_path / "trial-1"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task instructions."},
            _LITERAL_AGENT_STEP,
        ],
    )
    convs = collect_conversations_from_trial(trial_dir, _base_run_meta("trial-1"))
    assert len(convs) == 1
    row = convs[0]
    assert "prompt_token_ids" not in row
    assert "completion_token_ids" not in row
    assert "logprobs" not in row


def test_collect_with_literal_noops_on_text_only_trajectory(tmp_path):
    trial_dir = tmp_path / "trial-1"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task instructions."},
            {"source": "agent", "message": "All done."},
        ],
    )
    convs = collect_conversations_from_trial(
        trial_dir, _base_run_meta("trial-1"), include_literal_tokens=True
    )
    assert len(convs) == 1
    # Literal flag requested but trajectory has no token data → columns absent.
    row = convs[0]
    assert "prompt_token_ids" not in row
    assert "completion_token_ids" not in row


def test_collect_with_literal_multi_turn(tmp_path):
    trial_dir = tmp_path / "trial-1"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task."},
            {
                "source": "agent",
                "message": "first",
                "metrics": {
                    "prompt_token_ids": [1],
                    "completion_token_ids": [2],
                    "logprobs": [-0.5],
                },
                "observation": {"results": [{"content": "obs1"}]},
            },
            {
                "source": "agent",
                "message": "second",
                "metrics": {
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [4],
                    "logprobs": [-0.1],
                },
            },
        ],
    )
    convs = collect_conversations_from_trial(
        trial_dir, _base_run_meta("trial-1"), include_literal_tokens=True
    )
    # Two agent steps → two episodes; the first episode has 1 turn, the second 2.
    assert len(convs) == 2
    first, second = convs
    assert first["completion_token_ids"] == [[2]]
    assert second["completion_token_ids"] == [[2], [4]]


# ---------------------------------------------------------------------------
# 3a. Proxy pure helpers
# ---------------------------------------------------------------------------


def test_inject_literal_params_sets_defaults():
    body = {"model": "glm-5.2", "messages": []}
    out = inject_literal_params(body)
    assert out["logprobs"] is True
    # top_logprobs=1 (not 0): vLLM v0.20.0 tpu-inference IndexErrors on 0.
    assert out["top_logprobs"] == 1
    assert out["return_token_ids"] is True
    # Original body untouched (copy semantics).
    assert "logprobs" not in body


def test_inject_literal_params_preserves_existing_values():
    body = {"model": "m", "messages": [], "logprobs": False, "top_logprobs": 5}
    out = inject_literal_params(body)
    # Caller explicitly disabled logprobs → respected.
    assert out["logprobs"] is False
    assert out["top_logprobs"] == 5


def test_extract_literal_from_response_full():
    resp = {
        "id": "resp-1",
        "model": "glm-5.2",
        "prompt_token_ids": [10, 11, 12],
        "choices": [
            {
                "provider_specific_fields": {"token_ids": [13, 14]},
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.2},
                    ]
                },
            }
        ],
    }
    lit = extract_literal_from_response(resp)
    assert lit["prompt_token_ids"] == [10, 11, 12]
    assert lit["completion_token_ids"] == [13, 14]
    assert lit["logprobs"] == [-0.1, -0.2]
    assert lit["model"] == "glm-5.2"
    assert lit["response_id"] == "resp-1"


def test_extract_literal_from_response_missing_fields():
    resp = {"id": "resp-2", "model": "m", "choices": [{}]}
    lit = extract_literal_from_response(resp)
    assert lit["prompt_token_ids"] is None
    assert lit["completion_token_ids"] is None
    assert lit["logprobs"] is None


def test_extract_literal_from_response_empty_choices():
    lit = extract_literal_from_response({"choices": []})
    assert lit["completion_token_ids"] is None


def test_extract_completion_token_ids_fallback_choice_key():
    # Some vLLM builds place token_ids directly on the choice.
    resp = {"choices": [{"token_ids": [99]}]}
    lit = extract_literal_from_response(resp)
    assert lit["completion_token_ids"] == [99]


def test_extract_literal_from_token_completion_response():
    response = {
        "model": "m",
        "choices": [
            {
                "prompt_token_ids": [1, 2],
                "token_ids": [3, 4],
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
            }
        ],
    }

    literal = extract_literal_from_response(response)

    assert literal["prompt_token_ids"] == [1, 2]
    assert literal["completion_token_ids"] == [3, 4]
    assert literal["logprobs"] == [-0.1, -0.2]


# ---------------------------------------------------------------------------
# 3b. Proxy end-to-end forward() + recording
# ---------------------------------------------------------------------------


def test_proxy_forward_records_literal(tmp_path, monkeypatch):
    """RecordProxy.forward injects params, forwards, and logs the literal pair."""
    log_path = tmp_path / "literal.jsonl"

    captured: dict = {}

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    class FakeClient:
        is_closed = False

        async def request(self, method, url, json=None, headers=None):
            captured["method"] = method
            captured["url"] = url
            captured["sent_body"] = json
            captured["sent_headers"] = headers
            return FakeResponse(
                200,
                {
                    "id": "r-1",
                    "model": json["model"] if json else "m",
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "provider_specific_fields": {"token_ids": [3]},
                            "logprobs": {"content": [{"logprob": -0.3}]},
                        }
                    ],
                },
            )

    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)

    async def _fake_get_client():
        return FakeClient()

    monkeypatch.setattr(proxy, "_get_client", _fake_get_client)

    status, resp_json, elapsed = asyncio.run(
        proxy.forward(
            "/v1/chat/completions",
            {"model": "glm-5.2", "messages": []},
            trial_id="trial-1",
        )
    )

    assert status == 200
    assert resp_json["id"] == "r-1"
    # Injected params reached the upstream.
    assert captured["sent_body"]["logprobs"] is True
    assert captured["sent_body"]["return_token_ids"] is True
    assert captured["sent_headers"] == {CORRELATION_ID_HEADER: "trial-1"}
    assert elapsed >= 0.0
    # Log written with literal extraction.
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status_code"] == 200
    assert entry["literal"]["prompt_token_ids"] == [1, 2]
    assert entry["literal"]["completion_token_ids"] == [3]
    assert entry["literal"]["logprobs"] == [-0.3]


def test_proxy_forward_records_non_json_upstream(tmp_path, monkeypatch):
    """A non-JSON upstream response is still recorded (literal=None)."""
    log_path = tmp_path / "literal.jsonl"

    class FakeResponse:
        status_code = 500

        def json(self):
            raise ValueError("not json")

    class FakeClient:
        is_closed = False

        async def request(self, method, url, json=None, headers=None):
            return FakeResponse()

    proxy = RecordProxy("http://upstream:8000", log_path)

    async def _fake_get_client():
        return FakeClient()

    monkeypatch.setattr(proxy, "_get_client", _fake_get_client)

    status, resp_json, _ = asyncio.run(
        proxy.forward("/v1/chat/completions", {"model": "m", "messages": []})
    )
    assert status == 500
    assert resp_json is None
    entry = json.loads(log_path.read_text().strip())
    assert entry["literal"] is None
    assert entry["status_code"] == 500


# ---------------------------------------------------------------------------
# 3c. Proxy streaming tee: relay SSE bytes unchanged + capture literal
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Minimal stand-in for an httpx streaming Response."""

    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code = status_code
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


# One realistic vLLM chat-completion SSE stream: two content deltas each carrying
# a completion token id (provider_specific_fields.token_ids) + a logprob, the
# prompt_token_ids on the first chunk, terminated by [DONE].
_SSE_CHUNKS = [
    b'data: {"id":"c1","model":"m","prompt_token_ids":[1,2],'
    b'"choices":[{"delta":{"content":"Hel",'
    b'"provider_specific_fields":{"token_ids":[10]}},'
    b'"logprobs":{"content":[{"token":"Hel","logprob":-0.1}]}}]}\n\n',
    b'data: {"id":"c1","model":"m",'
    b'"choices":[{"delta":{"content":"lo",'
    b'"provider_specific_fields":{"token_ids":[11]}},'
    b'"logprobs":{"content":[{"token":"lo","logprob":-0.2}]}}]}\n\n',
    b"data: [DONE]\n\n",
]


def _run_relay(proxy: RecordProxy, resp, request_body: dict) -> list[bytes]:
    async def _collect():
        out: list[bytes] = []
        async for chunk in proxy.relay_and_record(request_body, resp):
            out.append(chunk)
        return out

    return asyncio.run(_collect())


def test_relay_and_record_relays_bytes_and_captures_literal(tmp_path):
    """Streaming path: SSE bytes relayed byte-for-byte AND a literal record written."""
    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)
    resp = _FakeStreamResponse(200, _SSE_CHUNKS)

    relayed = _run_relay(proxy, resp, {"model": "m", "messages": [], "stream": True})

    # 1. The client receives the upstream stream byte-for-byte (no re-framing).
    assert b"".join(relayed) == b"".join(_SSE_CHUNKS)
    # 2. The upstream response is closed.
    assert resp.closed is True
    # 3. A literal record is captured with accumulated token ids + logprobs.
    entry = json.loads(log_path.read_text().strip())
    assert entry["status_code"] == 200
    lit = entry["literal"]
    assert lit["prompt_token_ids"] == [1, 2]
    assert lit["completion_token_ids"] == [10, 11]
    assert lit["logprobs"] == [-0.1, -0.2]
    assert lit["response_id"] == "c1"


def test_relay_and_record_handles_bytewise_fragmentation(tmp_path):
    """SSE events split across arbitrary byte-chunk boundaries still parse."""
    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)
    # Re-fragment the whole stream into 7-byte chunks, splitting mid-JSON-line.
    whole = b"".join(_SSE_CHUNKS)
    fragments = [whole[i : i + 7] for i in range(0, len(whole), 7)]
    resp = _FakeStreamResponse(200, fragments)

    relayed = _run_relay(proxy, resp, {"model": "m", "messages": [], "stream": True})

    assert b"".join(relayed) == whole
    lit = json.loads(log_path.read_text().strip())["literal"]
    assert lit["completion_token_ids"] == [10, 11]
    assert lit["logprobs"] == [-0.1, -0.2]


def test_relay_and_record_multibyte_char_split_across_chunks(tmp_path):
    """A multi-byte UTF-8 char split across byte-chunks must not corrupt parsing."""
    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)
    # Content delta contains a 4-byte emoji; the SSE line still carries token id + logprob.
    line = (
        'data: {"id":"c1","model":"m","prompt_token_ids":[1],'
        '"choices":[{"delta":{"content":"\U0001f600",'
        '"provider_specific_fields":{"token_ids":[42]}},'
        '"logprobs":{"content":[{"logprob":-0.5}]}}]}\n\n'
    ).encode("utf-8") + b"data: [DONE]\n\n"
    # Fragment into 5-byte chunks, guaranteed to split the 4-byte emoji.
    fragments = [line[i : i + 5] for i in range(0, len(line), 5)]
    resp = _FakeStreamResponse(200, fragments)

    relayed = _run_relay(proxy, resp, {"model": "m", "messages": [], "stream": True})

    # Relay is byte-exact (emoji intact for the client).
    assert b"".join(relayed) == line
    # And the literal line parsed despite the split multibyte char.
    lit = json.loads(log_path.read_text().strip())["literal"]
    assert lit["completion_token_ids"] == [42]
    assert lit["logprobs"] == [-0.5]


def test_relay_and_record_non_200_relays_and_records_null_literal(tmp_path):
    """A non-200 upstream stream is relayed verbatim; literal is None."""
    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)
    body = b'{"error":"boom"}'
    resp = _FakeStreamResponse(500, [body])

    relayed = _run_relay(proxy, resp, {"model": "m", "messages": [], "stream": True})

    assert b"".join(relayed) == body
    entry = json.loads(log_path.read_text().strip())
    assert entry["status_code"] == 500
    assert entry["literal"] is None


def test_open_upstream_stream_injects_literal_params_and_keeps_stream(tmp_path):
    """open_upstream_stream enriches the body (logprobs/token_ids) and keeps stream=True."""
    captured: dict = {}

    class FakeClient:
        is_closed = False

        def build_request(self, method, url, json=None, headers=None):
            captured["method"] = method
            captured["url"] = url
            captured["body"] = json
            captured["headers"] = headers
            return ("request", json, headers)

        async def send(self, request, stream=False):
            captured["stream_flag"] = stream
            return _FakeStreamResponse(200, [b"data: [DONE]\n\n"])

    proxy = RecordProxy("http://upstream:8000", tmp_path / "literal.jsonl")

    async def _fake_get_client():
        return FakeClient()

    proxy._get_client = _fake_get_client  # type: ignore[assignment]

    enriched, resp = asyncio.run(
        proxy.open_upstream_stream(
            "/v1/chat/completions",
            {"model": "m", "messages": [], "stream": True},
            trial_id="trial-stream",
        )
    )
    assert resp.status_code == 200
    assert enriched["logprobs"] is True
    assert enriched["return_token_ids"] is True
    assert enriched["stream"] is True
    assert captured["stream_flag"] is True
    assert captured["url"] == "http://upstream:8000/v1/chat/completions"
    assert captured["headers"] == {CORRELATION_ID_HEADER: "trial-stream"}


def test_proxy_app_streaming_request_tees_sse(tmp_path):
    """End-to-end via app(): a stream=true request gets an SSE relay + literal record."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from starlette.testclient import TestClient
    import httpx

    stub = FastAPI()

    @stub.post("/v1/chat/completions")
    async def _chat(request: Request):  # noqa: ANN001
        async def _gen():
            for chunk in _SSE_CHUNKS:
                yield chunk

        return StreamingResponse(_gen(), media_type="text/event-stream")

    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm.local", log_path, timeout=10.0)
    stub_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stub), base_url="http://vllm.local"
    )

    async def _fake_get_client():
        return stub_client

    proxy._get_client = _fake_get_client  # type: ignore[assignment]

    with TestClient(proxy.app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [], "stream": True},
        )

    assert r.status_code == 200
    # The client saw the upstream SSE bytes unchanged.
    assert r.content == b"".join(_SSE_CHUNKS)
    # And a literal record was captured from the tee'd stream.
    entry = json.loads(log_path.read_text().strip())
    assert entry["literal"]["prompt_token_ids"] == [1, 2]
    assert entry["literal"]["completion_token_ids"] == [10, 11]
    assert entry["literal"]["logprobs"] == [-0.1, -0.2]
    # No correlation header on this request → trial_id recorded as None.
    assert entry["trial_id"] is None


def test_proxy_app_stamps_correlation_id_header_on_entry(tmp_path):
    """S2b: the x-ot-trial-id request header is captured onto the JSONL entry so a
    shared controller log can be split back per trial."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from starlette.testclient import TestClient
    import httpx

    stub = FastAPI()

    @stub.post("/v1/chat/completions")
    async def _chat(request: Request):  # noqa: ANN001
        assert request.headers[CORRELATION_ID_HEADER] == "trial-abc-123"

        async def _gen():
            for chunk in _SSE_CHUNKS:
                yield chunk

        return StreamingResponse(_gen(), media_type="text/event-stream")

    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm.local", log_path, timeout=10.0)
    stub_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stub), base_url="http://vllm.local"
    )

    async def _fake_get_client():
        return stub_client

    proxy._get_client = _fake_get_client  # type: ignore[assignment]

    with TestClient(proxy.app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [], "stream": True},
            headers={CORRELATION_ID_HEADER: "trial-abc-123"},
        )

    assert r.status_code == 200
    entry = json.loads(log_path.read_text().strip())
    assert entry["trial_id"] == "trial-abc-123"
    assert entry["literal"]["completion_token_ids"] == [10, 11]


def test_proxy_forward_stamps_correlation_id(tmp_path):
    """Non-streaming forward() also stamps trial_id (defensive; opencode streams)."""
    import asyncio

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "id": "r1",
                "model": "m",
                "choices": [
                    {
                        "provider_specific_fields": {"token_ids": [7, 8]},
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                    }
                ],
                "prompt_token_ids": [1, 2, 3],
            }

    class _Client:
        async def request(self, method, url, json=None, headers=None):
            assert headers == {CORRELATION_ID_HEADER: "trial-xyz"}
            return _Resp()

    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://upstream:8000", log_path, timeout=10.0)

    async def _fake_get_client():
        return _Client()

    proxy._get_client = _fake_get_client  # type: ignore[assignment]
    asyncio.run(
        proxy.forward(
            "/v1/chat/completions",
            {"model": "m", "messages": []},
            trial_id="trial-xyz",
        )
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["trial_id"] == "trial-xyz"


# ---------------------------------------------------------------------------
# Sanity: traces_utils export param threading does not break text-only path
# ---------------------------------------------------------------------------


def test_export_traces_accepts_include_literal_tokens(tmp_path, monkeypatch):
    """export_traces must accept the new kwarg without raising on text-only data."""
    from harbor.utils import traces_utils as _traces_utils_mod
    from harbor.utils.traces_utils import export_traces

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = job_dir / "trial-x"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task."},
            {"source": "agent", "message": "done"},
        ],
    )
    # export_traces loads run metadata from result.json.
    result_payload = {
        "config": {"agent": {"name": "terminus-2", "model_name": "test-model"}},
        "agent_info": {"name": "terminus-2", "model_info": {"name": "test-model"}},
        "started_at": "2024-01-01T00:00:00Z",
        "task_name": "test-task",
        "trial_name": "trial-x",
        "run_id": "test-run",
    }
    (job_dir / "result.json").write_text(json.dumps(result_payload))
    _traces_utils_mod._RESULT_JSON_CACHE.clear()

    ds = export_traces(
        job_dir,
        recursive=True,
        use_rich_progress=False,
        include_literal_tokens=True,
        verbose=False,
    )
    # Text-only trajectory → no token columns, but the row still exports.
    assert len(ds) == 1
    assert "prompt_token_ids" not in ds.column_names


def test_mini_swe_proxy_preserves_agent_identity_and_context_budget(
    tmp_path, monkeypatch
):
    headers = {CORRELATION_ID_HEADER: "mini-trial", TASK_AGENT_HEADER: "mini-swe-agent"}
    messages = [{"role": "user", "content": "run bash"}]

    def upstream(request):
        assert request.headers[CORRELATION_ID_HEADER] == headers[CORRELATION_ID_HEADER]
        assert request.headers[TASK_AGENT_HEADER] == headers[TASK_AGENT_HEADER]
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            assert body == {
                "model": "m",
                "messages": messages,
                "add_generation_prompt": True,
            }
            return httpx.Response(
                200, json={"tokens": [1, 2], "count": 2, "max_model_len": 128}
            )
        assert request.url.path == "/v1/chat/completions"
        assert body["return_token_ids"] is True
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1, 2],
                "choices": [
                    {"token_ids": [99], "logprobs": {"token_logprobs": [-0.1]}}
                ],
            },
        )

    stub = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: stub)
    path = tmp_path / "literal.jsonl"
    with TestClient(RecordProxy("http://vllm", path).app()) as client:
        budget = client.post(
            "/tokenize",
            headers=headers,
            json={
                "model": "m",
                "messages": messages,
                "add_generation_prompt": True,
            },
        )
        assert budget.json()["count"] == 2
        assert not path.exists()
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "m", "messages": messages},
        )
        assert response.status_code == 200
    entry = json.loads(path.read_text())
    assert entry["trial_id"] == "mini-trial"
    assert entry["literal"]["completion_token_ids"] == [99]
    assert entry["literal"]["logprobs"] == [-0.1]
    asyncio.run(stub.aclose())
