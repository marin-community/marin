"""Regression test: terminus-2 must never send an empty ``messages: []`` request.

Root cause: on the very first episode of a run, ``_run_agent_loop`` calls
``_check_proactive_summarization`` -> ``_count_total_tokens(chat)`` *before* the
initial prompt has been appended to the ``Chat`` (the prompt lives only in
``_trajectory_steps`` until the first ``chat.chat()``). With a vLLM ``api_base``
set, ``_count_total_tokens`` POSTed ``{"messages": []}`` to ``/tokenize``. HF's
``apply_chat_template`` indexes ``conversation[0]`` before its own empty-guard
runs (``tokenization_utils_base``), which vLLM re-raises as a ``ValueError`` and
fails the request.

The fix short-circuits ``_count_total_tokens`` to 0 when the chat has no
messages, so the HTTP probe is never dispatched with an empty array. This also
covers the ``_unwind_messages_to_free_tokens`` path that can drop a 2-message
chat to zero before re-counting.
"""

import logging

import pytest

from harbor.agents.terminus_2.terminus_2 import Terminus2


class _ApiBaseLLM:
    """LLM stub that advertises a vLLM ``_api_base`` so ``_count_total_tokens``
    would take the ``/tokenize`` HTTP branch if the empty-chat guard were
    missing."""

    _api_base = "http://fake-vllm:8000/v1"

    def get_model_context_limit(self) -> int:
        return 32768


class _EmptyChat:
    @property
    def messages(self):
        return []


def _make_agent() -> Terminus2:
    agent = object.__new__(Terminus2)
    agent.logger = logging.getLogger("test-terminus-2-empty-messages")
    agent._llm = _ApiBaseLLM()
    agent.model_alias = "fake/model"
    return agent


def test_count_total_tokens_empty_chat_returns_zero_without_http(monkeypatch):
    """An empty chat must return 0 tokens and issue NO ``/tokenize`` request."""
    import httpx

    def _boom(*args, **kwargs):
        raise AssertionError(
            "httpx.Client.post must not be called for an empty chat — "
            "an empty messages: [] request would crash vLLM's /tokenize."
        )

    monkeypatch.setattr(httpx.Client, "post", _boom)

    agent = _make_agent()
    assert agent._count_total_tokens(_EmptyChat()) == 0


@pytest.mark.asyncio
async def test_proactive_summarization_on_empty_first_episode_chat(monkeypatch):
    """The first-episode proactive check runs on an empty chat and must not
    hit ``/tokenize`` (the real trigger path in ``_run_agent_loop``)."""
    import httpx

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("empty-messages /tokenize request was dispatched")
        ),
    )

    agent = _make_agent()
    agent._enable_summarize = True
    agent._proactive_summarization_threshold = 8000

    # Empty chat: 0 tokens used, full context free -> no summarization, no probe.
    result = await agent._check_proactive_summarization(
        _EmptyChat(), "task instruction", session=None
    )
    assert result is None
