# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from collections import Counter
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from marin.inference.load_test import LoadRequest
from prometheus_client.parser import text_string_to_metric_families

from experiments.post_training import mochi_decode_window as window


def test_window_counts_token_ids_at_interval_boundaries():
    trace = window.DecodeTrace(LoadRequest("fixture", "completions", {"prompt": [1]}), 0, 17, started=5)
    trace.emissions = [(9, 3), (10, 2), (20, 4), (40, 8)]
    result = window.request_window(trace, 10, 40)
    assert result["window_tokens"] == 6
    assert result["window_chunks"] == 2
    assert result["decode_tokens_per_second"] == 0.2
    assert result["tpot_seconds"] == 5
    assert result["tokens_before_window"] == 3


@pytest.mark.parametrize("mode", ["plain", "dspark", "missing_rank", "preemption", "nonprogressive", "early_completion"])
def test_prefetched_stream_protocol_and_metric_qualification(monkeypatch, mode):
    lock = threading.Lock()
    active, generated, prompts, drafts = (Counter() for _ in range(4))
    received = []
    scrapes = [0]
    client_type = httpx.AsyncClient

    class Stream(httpx.AsyncByteStream):
        def __init__(self, body, rank):
            self.body = body
            self.rank = rank

        async def __aiter__(self):
            rank = self.rank
            maximum = self.body["max_tokens"]
            probe = maximum == 8
            if mode == "early_completion" and not probe:
                maximum = 6
            with lock:
                active[rank] += 1
                prompts[rank] += 1
            count = 0
            try:
                while count < maximum:
                    size = min(1 if probe else 3, maximum - count)
                    count += size
                    choice = {"text": "x" * size}
                    if mode != "nonprogressive" or not probe:
                        choice["token_ids"] = [12] * size
                    with lock:
                        generated[rank] += size
                        if not probe:
                            drafts[rank] += 1
                    yield ("data: " + json.dumps({"choices": [choice]}) + "\n\n").encode()
                    await asyncio.sleep(0.002)
                choice = {"token_ids": [12] * count} if mode == "nonprogressive" and probe else None
                result = {
                    "choices": [] if choice is None else [choice],
                    "usage": {"prompt_tokens": 1, "completion_tokens": count},
                }
                yield ("data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n").encode()
            finally:
                with lock:
                    active[rank] -= 1

    async def handle(request):
        body = json.loads(request.content)
        rank = int(request.headers["X-data-parallel-rank"])
        received.append((rank, body))
        return httpx.Response(200, stream=Stream(body, rank))

    def scrape(self):
        with lock:
            scrapes[0] += 1
            lines = []
            for rank in range(2):
                values = {
                    "vllm:prompt_tokens_total": prompts[rank],
                    "vllm:generation_tokens_total": generated[rank],
                    # A preemption between the first scrape and acceptance-counter initialization must invalidate.
                    "vllm:num_preemptions_total": int(mode == "preemption" and scrapes[0] > 1),
                    "vllm:spec_decode_num_drafts_total": drafts[rank],
                    "vllm:spec_decode_num_draft_tokens_total": 4 * drafts[rank],
                    "vllm:spec_decode_num_accepted_tokens_total": 2 * drafts[rank],
                }
                if not (mode == "missing_rank" and rank == 1):
                    values["vllm:num_requests_running"] = active[rank]
                lines.extend(f'{name}{{engine="{rank}"}} {value}' for name, value in values.items())
        return tuple(text_string_to_metric_families("\n".join(lines) + "\n"))

    monkeypatch.setattr(window.httpx, "AsyncClient", partial(client_type, transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(window.PrometheusScraper, "scrape", scrape)
    config = window.DecodeWindowConfig(
        data_parallel_size=2,
        running_per_engine=2,
        gpu_count=2,
        seed=17,
        max_new_tokens=32768,
        first_token_timeout=10,
        settle=0.01,
        measurement=0.12,
        metrics_interval=0.02,
    )
    operation = window.steady_decode_window(
        SimpleNamespace(server_url="http://fixture/v1", metrics_url="http://fixture/metrics"),
        [LoadRequest(str(token), "completions", {"prompt": [token]}) for token in (11, 12)],
        config,
        speculative=mode != "plain",
    )
    if mode in ("early_completion", "nonprogressive"):
        with pytest.raises(ValueError):
            asyncio.run(operation)
        assert sum(active.values()) == 0
        return
    result = asyncio.run(operation)
    assert result["valid"] is (mode in ("plain", "dspark"))
    assert len(result["requests"]) == 4
    assert result["summary"]["intentionally_censored_requests"] == 4
    assert all(
        row["window_tokens"] > row["window_chunks"] and row["window_tokens"] % 3 == 0 for row in result["requests"]
    )
    if mode != "plain":
        acceptance = result["counter_window"]["speculative_decoding"]
        assert acceptance["mean_acceptance_length"] == 3
        assert acceptance["draft_acceptance_rate"] == 0.5
    for rank in range(2):
        bodies = [body for requested_rank, body in received if requested_rank == rank and body["max_tokens"] != 8]
        assert {tuple(body["prompt"]) for body in bodies} == {(11,), (12,)}
        assert all("ignore_eos" not in body for body in bodies)
    assert sum(active.values()) == 0
