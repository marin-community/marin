"""Regression: the per-TURN hot-path work in the Terminus-2 agent loop must not
block the shared RolloutCoordinator asyncio event loop.

A LIVE py-spy on the CoreWeave RL path (v0g, running the round-1 HEAD) proved the
round-1 offload was INCOMPLETE — coordinators were still serialized on the single
asyncio dispatch loop behind PER-TURN blocking calls, so the vLLM engines stayed
under-fed (Waiting=0). The quoted stacks were:

  1. the trajectory dump -> a SYNCHRONOUS ``s3://`` write (``fsspec/asyn.py
     sync`` -> s3fs ``flush/close``) plus the CPU-bound
     ``format_trajectory_json`` regex pass, on the loop thread, EVERY turn.
  2. ``LiteLLM.get_model_context_limit`` -> a synchronous ``/v1/models`` HTTP GET
     on EVERY proactive-summarization check (the context limit is constant).
  3. ``Terminus2._count_total_tokens`` -> a FRESH ``httpx.Client`` + a new SSL
     context (``ssl.py create_default_context``) built PER CALL.

Site 1 later regressed AGAIN through ``asyncio.to_thread``: the dump merely
moved onto the shared default executor, which fills up under concurrent trials
(py-spy: 12-30 blocked s3fs upload threads per coordinator) and starves every
other ``to_thread`` user. These tests lock the current fixes:
  - ``Terminus2._persist_trajectory`` and ``_split_trajectory_on_summarization``
    perform NO storage write on the caller thread — serialization + write run
    on the bounded background artifact writer;
  - ``LiteLLM.get_model_context_limit`` memoizes its result -> no repeated HTTP;
  - ``Terminus2._count_total_tokens`` reuses ONE process-wide pooled httpx
    client instead of constructing a client + SSL context per call.
"""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from harbor.agents.terminus_2 import terminus_2 as t2
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.lite_llm import LiteLLM
from harbor.utils.artifact_writer import ArtifactWriter
from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def _gated_writer(tmp_path, monkeypatch) -> tuple[ArtifactWriter, threading.Event]:
    """A single-worker artifact writer whose worker is parked behind a gate, so
    any synchronous caller-thread write is directly observable: nothing can
    land on disk until the gate opens."""
    gate = threading.Event()
    writer = ArtifactWriter(num_workers=1)

    def parked() -> str:
        gate.wait(timeout=30)
        return "gate"

    writer.submit_optional(tmp_path / "gate.txt", parked, description="gate")
    monkeypatch.setattr(t2, "get_artifact_writer", lambda: writer)
    return writer, gate


# ── site 1: per-turn trajectory dump (S3 write + JSON format) off the loop ────


@pytest.mark.unit
async def test_persist_trajectory_never_writes_on_caller_thread(
    tmp_path, monkeypatch
) -> None:
    writer, gate = _gated_writer(tmp_path, monkeypatch)
    agent = make_bare_terminus2(tmp_path)

    agent._persist_trajectory(agent._summarization_count)

    # The call returned with the background worker parked: a synchronous write
    # on the caller thread would already be visible here.
    assert not (tmp_path / "trajectory.json").exists(), (
        "trajectory dump wrote synchronously on the caller thread"
    )

    gate.set()
    writer.flush(timeout=30)
    # Semantics preserved: the trajectory JSON actually landed.
    assert '"session_id"' in (tmp_path / "trajectory.json").read_text()


@pytest.mark.unit
async def test_split_trajectory_on_summarization_defers_boundary_write(
    tmp_path, monkeypatch
) -> None:
    writer, gate = _gated_writer(tmp_path, monkeypatch)
    agent = make_bare_terminus2(tmp_path)
    agent._linear_history = True
    agent._summarization_count = 2
    # One boundary dump already completed; this split writes segment index 1.
    agent._continuation_segments_persisted = 1

    agent._split_trajectory_on_summarization("handoff")

    # No synchronous write on the caller thread...
    assert not (tmp_path / "trajectory.cont-1.json").exists(), (
        "summarization-split dump wrote synchronously on the caller thread"
    )

    gate.set()
    writer.flush(timeout=30)
    # ...but the boundary segment lands, with the continuation index matching
    # the number of completed boundary dumps.
    assert (
        '"continuation_index": 1' in (tmp_path / "trajectory.cont-1.json").read_text()
    )


# ── site 2: context-limit memoization (no repeated HTTP GET) ──────────────────


@pytest.mark.unit
def test_get_model_context_limit_memoized_no_repeat_probe() -> None:
    llm = LiteLLM(
        model_name="test-integration/unknown-hash-model",
        api_base="http://localhost:8000/v1",
    )
    calls = {"n": 0}

    def _probe():
        calls["n"] += 1
        return 32768

    llm._get_vllm_max_model_len = _probe  # type: ignore[method-assign]

    # 32768 minus the _VLLM_CONTEXT_SAFETY_BUFFER shaved off the live vLLM wall.
    first = llm.get_model_context_limit()
    assert first == llm.get_model_context_limit() == llm.get_model_context_limit()
    # The constant limit is resolved ONCE; later per-turn checks are cache hits.
    assert calls["n"] == 1, "context limit re-probed the vLLM server per call"
    assert llm._cached_context_limit == first


@pytest.mark.unit
def test_get_model_context_limit_does_not_cache_fallback() -> None:
    """When neither the registry nor the vLLM probe resolves a limit, the 1e6
    fallback must NOT be memoized so a later call can still resolve the real
    limit once the server is up."""
    llm = LiteLLM(model_name="test-integration/no-base")  # no api_base -> probe None
    assert llm.get_model_context_limit() == 1000000
    assert getattr(llm, "_cached_context_limit", None) is None


# ── site 3: shared pooled httpx client for /tokenize (no per-call SSL ctx) ─────


@pytest.mark.unit
def test_get_shared_tokenize_client_is_a_singleton() -> None:
    t2._TOKENIZE_HTTP_CLIENT = None  # reset module-level cache for a clean assert
    c1 = t2._get_shared_tokenize_client()
    c2 = t2._get_shared_tokenize_client()
    assert c1 is c2, "tokenize client is rebuilt per call (fresh SSL context)"


@pytest.mark.unit
def test_count_total_tokens_uses_shared_client_not_fresh_httpx() -> None:
    agent = object.__new__(Terminus2)
    agent._llm = SimpleNamespace(_api_base="http://localhost:8000/v1")
    agent.model_alias = "hosted_vllm/test-model"

    chat = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

    fake_client = mock.MagicMock()
    fake_client.post.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"count": 7}
    )

    with (
        mock.patch.object(
            t2, "_get_shared_tokenize_client", return_value=fake_client
        ) as shared,
        mock.patch("httpx.Client") as fresh_client_ctor,
    ):
        result = agent._count_total_tokens(chat)

    assert result == 7
    shared.assert_called_once()  # went through the pooled client
    fresh_client_ctor.assert_not_called()  # no per-call client / SSL context
    # vLLM serves /tokenize at the server root (not under /v1).
    posted_url = fake_client.post.call_args[0][0]
    assert posted_url == "http://localhost:8000/tokenize"
