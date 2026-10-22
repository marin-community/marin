"""Round-3 regression: two more per-turn blocking calls must not run ON the
shared RolloutCoordinator asyncio event loop.

A LIVE py-spy on the CoreWeave RL path (v0h, running the round-2 HEAD) still
caught the coordinator event loop stalled — engines under-fed (Waiting=0, KV
floor, ~1/3 TDP) — at TWO sites round-1/round-2 left on the loop:

  1. ``Terminus2._count_total_tokens`` -> a SYNCHRONOUS ``httpx`` POST to vLLM's
     ``/tokenize``, reached EVERY turn via ``_check_proactive_summarization``.
     Round-2 pooled the client + SSL context but the POST itself still ran on
     the loop thread (the method is sync + shared with non-async callers). Fix:
     a new ``async def _acount_total_tokens``; the async callers await it, the
     sync callers keep the sync method.  (Round-4 note: this async method was
     originally an ``asyncio.to_thread`` wrapper; it is now a TRULY-ASYNC
     ``httpx.AsyncClient`` POST — see ``test_coordinator_round4_offload.py`` for
     the concurrency-math reason. The ``is-a-coroutine`` guard below still holds.)
  2. ``Trial._record_exception`` -> a SYNCHRONOUS s3fs ``write_text`` of
     ``exception.txt``, fired CONSTANTLY on the benign per-turn noise path
     (``AgentTimeoutError`` in ``single_step._run_agent``; sandbox-stop /
     log-upload failures in ``_stop_agent_environment``). Fix: the method
     captures ``traceback.format_exc()`` on the loop thread (thread-local
     ``sys.exc_info()``) and enqueues the blocking write on the bounded
     optional background artifact lane. ``_enforce_verifier_result_invariant``
     (its sibling exception.txt write on the finalize path) routes the same way.

Each "off-loop" assertion compares the thread that ran the blocking call against
the event-loop thread; they MUST differ.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.trial import trial as trial_module
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.artifact_writer import ArtifactWriter


class _ThreadRecordingWriter:
    """Stand-in for a ``UPath`` exception path: records the thread the write ran
    on and persists the bytes to a real local file so behaviour is exercised."""

    def __init__(self, path: Path):
        self._path = path
        self.parent = path.parent
        self.thread_id: int | None = None

    def __str__(self) -> str:
        return str(self._path)

    def open(self, mode: str):
        self.thread_id = threading.get_ident()
        return self._path.open(mode)


# ── site 1: per-turn token count (/tokenize POST) off the loop ────────────────


@pytest.mark.unit
def test_acount_total_tokens_is_a_coroutine() -> None:
    import asyncio

    # The async callers (_check_proactive_summarization + the hint path) must be
    # able to `await` it; the sync method is kept intact for _unwind_* callers.
    assert asyncio.iscoroutinefunction(Terminus2._acount_total_tokens)
    assert not asyncio.iscoroutinefunction(Terminus2._count_total_tokens)


@pytest.mark.unit
async def test_acount_total_tokens_uses_truly_async_client() -> None:
    """Round-4: the per-turn probe goes through the ASYNC client and is awaited.

    (The round-3 ``!= loop_thread`` off-loop assertion no longer applies — the
    probe is now truly-async and runs ON the loop but yields during the network
    wait; the concurrency guarantee is asserted in the round-4 suite.)
    """
    agent = object.__new__(Terminus2)
    agent._llm = SimpleNamespace(_api_base="http://localhost:8000/v1")
    agent.model_alias = "hosted_vllm/test-model"
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

    called: dict[str, str] = {}

    async def _recording_post(url, json):  # noqa: A002 - mirrors AsyncClient.post
        called["url"] = url
        return SimpleNamespace(status_code=200, json=lambda: {"count": 42})

    fake_async_client = SimpleNamespace(post=_recording_post)

    def _boom(*a, **k):  # sync client must NOT be used on the async hot path
        raise AssertionError("sync /tokenize client used on the async hot path")

    with (
        mock.patch(
            "harbor.agents.terminus_2.terminus_2.shared_async_tokenize_client",
            return_value=fake_async_client,
        ),
        mock.patch(
            "harbor.agents.terminus_2.terminus_2._get_shared_tokenize_client",
            side_effect=_boom,
        ),
    ):
        result = await agent._acount_total_tokens(chat)

    assert result == 42  # count still computed, semantics preserved
    assert called["url"].endswith("/tokenize")
    # /v1 stripped from the server root (probe shape preserved from round-2/3).
    assert called["url"] == "http://localhost:8000/tokenize"


# ── site 2: per-turn exception.txt write off the loop ─────────────────────────


@pytest.mark.unit
def test_record_exception_is_a_coroutine() -> None:
    import asyncio

    # All 4 callers (trial.run x2, _stop_agent_environment, single_step._run_agent)
    # are async and await it; the write itself is enqueued fire-and-forget.
    assert asyncio.iscoroutinefunction(SingleStepTrial._record_exception)


@pytest.mark.unit
async def test_record_exception_writes_traceback_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    artifact_writer = ArtifactWriter()
    monkeypatch.setattr(trial_module, "get_artifact_writer", lambda: artifact_writer)
    writer = _ThreadRecordingWriter(tmp_path / "exception.txt")

    trial = object.__new__(SingleStepTrial)
    trial._result = SimpleNamespace(exception_info=None)
    trial.paths = SimpleNamespace(exception_message_path=writer)
    trial.logger = SimpleNamespace(debug=lambda *a, **k: None)

    try:
        raise ValueError("boom-round3")
    except ValueError as exc:
        await trial._record_exception(exc)
    artifact_writer.flush(timeout=30)

    assert writer.thread_id is not None, "exception.txt was never written"
    assert writer.thread_id != loop_thread, "exception.txt write ran ON the event loop"
    # The durable write is preserved AND the correct traceback text was captured
    # on the loop thread (sys.exc_info() is thread-local — a naive to_thread of
    # traceback.format_exc() would have lost it).
    written = (tmp_path / "exception.txt").read_text()
    assert "ValueError" in written and "boom-round3" in written
    assert trial._result.exception_info is not None  # in-memory record still set


@pytest.mark.unit
async def test_enforce_verifier_result_invariant_writes_message_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    artifact_writer = ArtifactWriter()
    monkeypatch.setattr(trial_module, "get_artifact_writer", lambda: artifact_writer)
    writer = _ThreadRecordingWriter(tmp_path / "exception.txt")

    trial = object.__new__(SingleStepTrial)
    trial.config = SimpleNamespace(verifier=SimpleNamespace(disable=False))
    trial.task = SimpleNamespace(has_steps=False)
    trial._result = SimpleNamespace(verifier_result=None, exception_info=None)
    trial._reached_verifier = False
    trial.paths = SimpleNamespace(exception_message_path=writer)

    await trial._enforce_verifier_result_invariant()
    artifact_writer.flush(timeout=30)

    assert writer.thread_id is not None, "invariant message was never written"
    assert writer.thread_id != loop_thread, "invariant write ran ON the event loop"
    assert "never reached" in (tmp_path / "exception.txt").read_text()
    assert trial._result.exception_info is not None
