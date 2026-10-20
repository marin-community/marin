"""Episode logging: the raw request+response (with logprobs/token_ids) is
written to disk ONLY when a logging_path is passed.

This replaces litellm's ``logger_fn`` post_api_call hook. It is the hot-path
write that stalls the shared rollout loop on a remote logging_path; Terminus-2
gates it by passing None (``enable_episode_logging`` default-off), so the RL
hot path never writes. When enabled, the multi-MB serialization + write run on
the background artifact writer, not the caller thread. These tests pin the
None-guard + the write behavior + that api keys are hashed, not persisted in
the clear.

Ref: agent_logs/2026-07-03_r5_engine_starvation_rootcause.md
"""

import json

import pytest

from harbor.llms.lite_llm import LiteLLM
from harbor.utils.artifact_writer import get_artifact_writer


def _llm() -> LiteLLM:
    return LiteLLM(model_name="fake-provider/fake-model")


_RAW_RESPONSE = {
    "model": "fake-model",
    "choices": [
        {
            "message": {"content": "ok", "reasoning_content": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    "prompt_token_ids": [1, 2, 3],
}


@pytest.mark.asyncio
async def test_call_writes_nothing_when_logging_path_none(tmp_path):
    """call() with logging_path=None must not write any episode log (RL hot path)."""
    llm = _llm()

    async def _fake(_completion_kwargs):
        return _RAW_RESPONSE

    llm._acreate_chat_raw = _fake  # type: ignore[assignment]
    llm._acreate_chat_litellm = _fake  # type: ignore[assignment]
    await llm.call(prompt="hi", message_history=[], logging_path=None)
    get_artifact_writer().flush(prefix=str(tmp_path), timeout=30)
    # No files written anywhere under tmp_path.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_call_writes_episode_log_when_path_given(tmp_path):
    """call() with a logging_path persists the raw response (with token ids)
    via the background artifact writer."""
    path = tmp_path / "debug.json"
    llm = _llm()

    async def _fake(_completion_kwargs):
        return _RAW_RESPONSE

    llm._acreate_chat_raw = _fake  # type: ignore[assignment]
    llm._acreate_chat_litellm = _fake  # type: ignore[assignment]
    await llm.call(prompt="hi", message_history=[], logging_path=path)
    get_artifact_writer().flush(prefix=str(tmp_path), timeout=30)
    assert path.exists()
    assert json.loads(path.read_text())["response"]["prompt_token_ids"] == [1, 2, 3]


def test_format_episode_log_preserves_response():
    payload = json.loads(
        _llm()._format_episode_log(
            {"model": "fake-model", "messages": []}, _RAW_RESPONSE
        )
    )
    # Both the request and the response (with logprobs/token_ids) are preserved.
    assert payload["response"]["prompt_token_ids"] == [1, 2, 3]
    assert payload["request"]["model"] == "fake-model"


def test_format_episode_log_hashes_api_key():
    formatted = _llm()._format_episode_log(
        {"model": "m", "api_key": "sk-secret-123"}, _RAW_RESPONSE
    )
    payload = json.loads(formatted)
    # The raw key is never persisted; only its sha256 is.
    assert "api_key" not in payload["request"]
    assert "api_key_sha256" in payload["request"]
    assert "sk-secret-123" not in formatted
