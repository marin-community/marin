"""Round-4 regression: the per-turn `/tokenize` probe must be TRULY ASYNC (not
``to_thread``), and the daytona download mkdirs must be offloaded off the loop.

A LIVE py-spy on the CoreWeave RL path (v0i, running the round-3 HEAD) still
caught the RolloutCoordinator's single asyncio loop saturation-bursty (~1/3 TDP,
KV floor) despite the round-1/2/3 offloads. The remaining driver is a
CONCURRENCY-MATH one, not an ordinary "move it off the loop" one:

  1. ``Terminus2._acount_total_tokens`` -> the per-turn vLLM `/tokenize` POST,
     reached EVERY turn (~169 concurrent trials per coordinator) via
     ``_check_proactive_summarization`` (which always runs, even with
     ``enable_summarize=False``). Round-3 moved it off the loop via
     ``asyncio.to_thread`` — but that routes through the DEFAULT
     ``ThreadPoolExecutor`` (bounded ~min(32, cpu+4) threads), so 169 per-turn
     probes SERIALIZE ~32-wide AND contend the same default executor litellm's
     ``acompletion`` uses for its ``run_in_executor(None, ...)`` preamble. The
     fix for a HIGH-CONCURRENCY hot-path op is a TRULY-ASYNC ``httpx.AsyncClient``
     POST ``await``-ed on the loop: all N probes are in-flight at once (each
     yields during the network wait) instead of queueing behind 32 threads.
     THIS is asserted by ``test_..._probes_run_concurrently_not_serialized``.

     (Distinct from the LLM completion itself, which was ALREADY async —
     ``lite_llm.call`` -> ``litellm.acompletion`` -> ``async_completion`` ->
     ``await async_httpx_client.post``. No sync ``litellm.completion`` exists in
     harbor, so there was nothing to swap there.)

  2. ``DaytonaEnvironment._download_dir_to_local`` -> a blocking ``Path.mkdir``
     per downloaded file ON the loop. Low-frequency (per download, not per turn),
     so ``asyncio.to_thread`` is the correct offload (batched into one call for
     the whole dir set). Asserted by ``test_download_dir_mkdirs_run_off_loop``.
"""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from harbor.agents.terminus_2.terminus_2 import Terminus2


# ── site 1: the /tokenize probe is truly-async (concurrent, not serialized) ───


@pytest.mark.unit
async def test_acount_total_tokens_probes_run_concurrently_not_serialized() -> None:
    """N concurrent probes must all be in-flight at once (async), NOT serialized.

    A truly-async ``await client.post`` lets every concurrent trial's probe wait
    on the network simultaneously. A ``to_thread`` implementation would cap
    concurrency at the default pool width (~32) — this test would then observe a
    max-in-flight well below N. We assert max-in-flight == N (here N=64, > the
    default pool) to lock in the async property.
    """
    N = 64
    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def _barrier_post(url, json):  # noqa: A002 - mirrors AsyncClient.post
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Hold until every coroutine has entered — only possible if they are all
        # genuinely in-flight concurrently on the one loop (not queued 32-wide).
        if in_flight >= N:
            release.set()
        await release.wait()
        in_flight -= 1
        return SimpleNamespace(status_code=200, json=lambda: {"count": 7})

    fake_async_client = SimpleNamespace(post=_barrier_post)

    def _make_agent() -> Terminus2:
        agent = object.__new__(Terminus2)
        agent._llm = SimpleNamespace(_api_base="http://localhost:8000/v1")
        agent.model_alias = "hosted_vllm/test-model"
        return agent

    chat = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

    with mock.patch(
        "harbor.agents.terminus_2.terminus_2.shared_async_tokenize_client",
        return_value=fake_async_client,
    ):
        results = await asyncio.wait_for(
            asyncio.gather(
                *(_make_agent()._acount_total_tokens(chat) for _ in range(N))
            ),
            timeout=5.0,
        )

    assert results == [7] * N
    assert max_in_flight == N, (
        f"probes serialized: only {max_in_flight}/{N} concurrent — the hot-path "
        "probe is not truly async (still funneling through a bounded pool)."
    )


@pytest.mark.unit
async def test_acount_total_tokens_falls_back_off_loop_when_no_vllm() -> None:
    """No vLLM api_base -> litellm token_counter fallback, kept OFF the loop."""
    loop_thread = threading.get_ident()

    agent = object.__new__(Terminus2)
    agent._llm = SimpleNamespace(_api_base=None)  # no vLLM -> fallback path
    agent.model_alias = "gpt-4o"
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

    seen: dict[str, int] = {}

    def _fake_fallback(self, _chat, extra_messages=None) -> int:
        seen["thread"] = threading.get_ident()
        return 11

    with mock.patch.object(Terminus2, "_token_counter_fallback", _fake_fallback):
        result = await agent._acount_total_tokens(chat)

    assert result == 11
    assert seen["thread"] != loop_thread, "CPU-bound fallback ran ON the event loop"


@pytest.mark.unit
def test_async_tokenize_client_is_pooled_per_loop() -> None:
    """One AsyncClient per event loop, reused (no per-call construction)."""
    from harbor.llms.vllm_tokenization import shared_async_tokenize_client

    async def _twice():
        return (
            shared_async_tokenize_client(),
            shared_async_tokenize_client(),
        )

    a, b = asyncio.run(_twice())
    assert a is b  # same client reused within a loop

    # A different loop gets its own client (loop-affine).
    c = asyncio.run(_wrap(shared_async_tokenize_client))
    assert c is not a


async def _wrap(fn):
    return fn()


# ── site 2: daytona download mkdirs offloaded off the loop ────────────────────


@pytest.mark.unit
async def test_download_dir_mkdirs_run_off_loop(tmp_path: Path) -> None:
    from harbor.environments.daytona import environment as denv

    loop_thread = threading.get_ident()
    seen_threads: list[int] = []

    real_mkdirs = denv._mkdirs_local

    def _recording_mkdirs(dirs):
        seen_threads.append(threading.get_ident())
        real_mkdirs(dirs)

    # Minimal fake sandbox fs: one file under source_dir.
    src = "/work"
    file_path = "/work/sub/a.txt"

    async def _search_files(source_dir, pattern):
        return SimpleNamespace(files=[file_path])

    async def _get_file_info(fp):
        return SimpleNamespace(is_dir=False)

    downloaded: dict[str, object] = {}

    async def _download_files(files):
        downloaded["files"] = files

    env = object.__new__(denv.DaytonaEnvironment)
    env._sandbox = SimpleNamespace(
        fs=SimpleNamespace(
            search_files=_search_files,
            get_file_info=_get_file_info,
            download_files=_download_files,
        )
    )
    env.logger = SimpleNamespace(debug=lambda *a, **k: None)

    with mock.patch.object(denv, "_mkdirs_local", _recording_mkdirs):
        await env._download_dir_to_local(src, tmp_path)

    # The mkdir batch ran, off the event-loop thread, and created the dirs.
    assert seen_threads, "_mkdirs_local was never called"
    assert all(t != loop_thread for t in seen_threads), "mkdir ran ON the event loop"
    assert (tmp_path / "sub").is_dir()
    assert "files" in downloaded  # download still issued after the mkdirs
