# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
from marin.inference.load_test import LoadRequest
from rolloutengine.repetition import detect_tail_loop

from experiments.post_training.long_loop.context_cohort import (
    ResponseCapture,
    complete_cohort,
    response_signals,
    summarize_cohort,
)


def test_finite_cohort_preserves_each_prompt_seed_and_natural_stops():
    sent = []
    artifacts = []
    metadata = {
        identity: {
            "split": "train",
            "assistant_prefix": "",
            "answer_type": "text",
            "answer_format": {"kind": "plain_text"},
        }
        for identity in ("a", "b")
    }

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        events = [
            {"choices": [{"delta": {"content": "answer"}, "token_ids": [40, 41], "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 2}},
        ]
        return httpx.Response(
            200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await complete_cohort(
                client,
                "http://model/v1",
                "policy",
                [
                    LoadRequest(identity, "chat/completions", {"messages": [{"role": "user", "content": identity}]})
                    for identity in metadata
                ],
                metadata,
                concurrency=3,
                repeats=2,
                max_tokens=32768,
                seed=17,
                on_result=lambda row, artifact: artifacts.append(artifact),
            )

    rows = asyncio.run(run())
    assert {(body["messages"][0]["content"], body["seed"]) for body in sent} == {
        ("a", 17),
        ("a", 18),
        ("b", 17),
        ("b", 18),
    }
    assert len(sent) == 4
    assert all(body["max_tokens"] == 32768 and body["ignore_eos"] is False for body in sent)
    assert all(
        artifact["message"]["content"] == "answer" and artifact["token_ids"] == [40, 41] for artifact in artifacts
    )
    summary = summarize_cohort(rows, expected=4)
    assert summary["cohort_complete"] and summary["length_stop_fraction_of_successful"] == 0
    assert summary["response_length"] == {"p50": 2, "p90": 2, "p99": 2}
    assert summary["loop_observed_attempts"] == 4
    partial = summarize_cohort(rows[:2], expected=4)
    assert not partial["cohort_complete"] and partial["censored_or_unstarted"] == 2


def test_streamed_tool_fragments_reconstruct_parseable_answer_and_exact_ids():
    capture = ResponseCapture()
    capture.add(
        {
            "choices": [
                {
                    "token_ids": [1, 2],
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "submit_answer", "arguments": '{"answer":'},
                            }
                        ]
                    },
                }
            ]
        }
    )
    capture.add(
        {"choices": [{"token_ids": [3], "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"42"}'}}]}}]}
    )
    result = response_signals(
        capture,
        {"assistant_prefix": "", "answer_type": "text", "answer_format": {"kind": "answer_call"}},
        output_tokens=3,
    )
    assert capture.message("")["tool_calls"][0]["function"]["arguments"] == '{"answer":"42"}'
    assert result["source_submission_parseable"] and result["explicit_final_marker"]
    assert result["hard_tail_loop"] is False


def test_loop_coverage_and_boxed_fallback_are_not_claimed_as_final_answers():
    capture = ResponseCapture()
    capture.add({"choices": [{"token_ids": [1, 2] * 8, "delta": {"content": "still reasoning"}}]})
    metadata = {"assistant_prefix": "", "answer_type": "text", "answer_format": {"kind": "boxed"}}
    observed = response_signals(capture, metadata, output_tokens=16)
    assert observed["source_submission_parseable"]  # Normal Boxed extraction accepts raw text.
    assert not observed["explicit_final_marker"]
    assert observed["hard_tail_loop_span"] == {"start": 0, "period": 2, "end": 16}
    missing = response_signals(capture, metadata, output_tokens=17)
    assert missing["hard_tail_loop"] is None and not missing["token_id_coverage_complete"]


def test_tail_loop_ignores_prior_segments_and_preserves_original_coordinates():
    tokens = [4, 5] * 6 + [9] + [1, 2] * 4
    mask = [1] * 12 + [0] + [1] * 8
    loop = detect_tail_loop(tokens, mask, max_period_tokens=64, tail_tokens=256, minimum_occurrences=4)
    assert loop is not None and (loop.start, loop.period, loop.end) == (13, 2, 21)
    recovered = detect_tail_loop(
        [*tokens, 8, 7, 6], [*mask, 1, 1, 1], max_period_tokens=64, tail_tokens=256, minimum_occurrences=4
    )
    assert recovered is None
