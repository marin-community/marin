# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
import pytest
from marin.inference.load_test import (
    LoadRequest,
    RequestMeasurement,
    measure_request,
    summarize_metrics,
    summarize_requests,
)
from marin.inference.load_workloads import replay_requests, synthetic_requests


@pytest.mark.parametrize("data_parallel_rank", [None, 0, 3])
def test_streaming_usage_counts_tokens_and_ignores_role_only_events(data_parallel_rank):
    now = [100.0]
    received = []
    events = [
        (100.5, {"choices": [{"delta": {"role": "assistant"}}]}),
        (101.0, {"choices": [{"delta": {"reasoning_content": "three output tokens"}}]}),
        (105.0, {"choices": [{"delta": {"content": "two more"}, "finish_reason": "length"}]}),
        (106.0, {"choices": [], "usage": {"prompt_tokens": 128, "completion_tokens": 5}}),
        (107.0, "[DONE]"),
    ]

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for timestamp, event in events:
                now[0] = timestamp
                yield f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n".encode()

    async def handle(request):
        received.append((json.loads(request.content), request.headers.get("X-data-parallel-rank")))
        return httpx.Response(200, stream=Stream())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await measure_request(
                client,
                "http://model/v1",
                "policy",
                LoadRequest("task-1", "chat/completions", {"messages": []}),
                seed=17,
                clock=lambda: now[0],
                data_parallel_rank=data_parallel_rank,
            )

    result = asyncio.run(run())

    assert result.error is None
    assert result.prompt_tokens == 128 and result.output_tokens == 5
    assert result.ttft == 1.0 and result.tpot == 1.0
    assert result.ended == 107.0 and result.finish_reason == "length"
    body, requested_rank = received[0]
    assert body["stream_options"] == {"include_usage": True}
    assert body["seed"] == 17
    assert requested_rank == (None if data_parallel_rank is None else str(data_parallel_rank))
    assert result.requested_data_parallel_rank == data_parallel_rank


def test_truncated_stream_remains_a_failed_request_even_with_partial_usage():
    async def run():
        event = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=f"data: {json.dumps(event)}\n\n"))
        async with httpx.AsyncClient(transport=transport) as client:
            return await measure_request(
                client, "http://model/v1", "policy", LoadRequest("x", "completions", {}), seed=1
            )

    result = asyncio.run(run())
    assert result.error is not None
    summary = summarize_requests([result], result.started, result.ended)
    assert summary["failed"] == 1 and summary["succeeded"] == 0


def test_server_counter_window_excludes_warmup_and_preserves_counter_reset():
    def sample(timestamp, output, prompt, hits, queries):
        return {
            "time": timestamp,
            "series": [
                {"name": name, "value": value, "labels": {"engine": "0"}}
                for name, value in [
                    ("vllm:generation_tokens_total", output),
                    ("vllm:prompt_tokens_total", prompt),
                    ("vllm:prefix_cache_hits_total", hits),
                    ("vllm:prefix_cache_queries_total", queries),
                ]
            ],
        }

    before, after = sample(120, 40, 100, 2, 4), sample(130, 240, 500, 8, 14)
    result = summarize_metrics([before, after], gpu_count=4)
    assert result["output_tokens_per_second_per_gpu"] == 5
    assert result["input_tokens_per_second_per_gpu"] == 10
    assert result["prefix_hit_fraction"] == 0.6
    reset = summarize_metrics([after, sample(140, 10, 10, 0, 0)], gpu_count=4)
    assert reset["output_tokens_per_second_per_gpu"] is None


def test_request_drain_is_not_counted_as_steady_window_throughput():
    rows = [
        RequestMeasurement("warm", "a", 0, 0.5, 2, 4, 10, 20, "length", None),
        RequestMeasurement("measured", "b", 1.5, 1.6, 2.0, 4, 10, 5, "length", None),
    ]
    result = summarize_requests(rows, start=1, end=3)
    assert result["admitted"] == 1 and result["overlapping_succeeded"] == 2
    assert result["completed_in_window"] == 0
    assert result["drain_seconds"] == 1
    assert result["request_output_tokens_per_second_including_drain"] == pytest.approx(5 / 3)


def test_replay_never_includes_the_answer_being_regenerated():
    rows = [
        {
            "uid": "42",
            "data_source": "micro210",
            "output_response": "secret final answer",
            "messages": [
                {"role": "user", "content": "Repair the fixture"},
                {"role": "assistant", "content": "call a tool"},
                {"role": "tool", "content": "fixture evidence"},
                {"role": "assistant", "content": "secret final answer"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="No replayable model requests"):
        replay_requests(rows, output_tokens=64)
    rows[0]["tools"] = [{"type": "function", "function": {"name": "inspect"}}]
    requests = replay_requests(rows, output_tokens=64)
    assert len(requests) == 1
    assert requests[0].body["messages"][1]["content"] == "call a tool"
    assert requests[-1].body["messages"][-1]["role"] == "tool"
    assert "secret final answer" not in json.dumps([request.body for request in requests])


def test_synthetic_workload_reproduces_text_and_varies_seed_without_model_tokens():
    options = {"context_repetitions": 3, "output_tokens": 256, "count": 4, "prompt_group_size": 1}
    first = synthetic_requests(seed=17, **options)
    assert first == synthetic_requests(seed=17, **options)
    assert first != synthetic_requests(seed=29, **options)
    assert all("messages" in item.body and "prompt_token_ids" not in item.body for item in first)


def test_synthetic_prompt_groups_share_context_without_merging_request_identity():
    requests = synthetic_requests(seed=17, context_repetitions=3, output_tokens=256, count=24, prompt_group_size=8)
    prompts = [json.dumps(request.body["messages"]) for request in requests]

    assert len({request.identity for request in requests}) == 24
    assert len(set(prompts)) == 3
    assert [prompts.count(prompt) for prompt in set(prompts)] == [8, 8, 8]
    assert len({json.dumps(request.body["messages"]) for request in requests[:8]}) == 1
