"""Regression coverage for OpenAI-compatible reasoning responses."""

import asyncio
import json
from collections import Counter

import pytest
import requests
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion

from eval import robust_api  # noqa: F401 - installs the lm-eval adapter patch
from eval.completion_response import (
    CompletionClassification,
    CompletionContentPolicy,
    CompletionText,
    FailedGeneration,
    completion_response_from_chat_choice,
)
from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import record_sample_metrics

from eval.sample_logging import canonicalize_samples
from eval.task import BaseBenchmark


def _adapter(policy: CompletionContentPolicy = CompletionContentPolicy.COMBINE):
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = policy
    return adapter


def _openai_adapter(model: str) -> OpenAIChatCompletion:
    adapter = object.__new__(OpenAIChatCompletion)
    adapter.model = model
    adapter._max_gen_toks = 256
    return adapter


def _openai_payload(model: str) -> dict:
    return _openai_adapter(model)._create_payload(
        messages=[{"role": "user", "content": "Question"}],
        gen_kwargs={
            "do_sample": False,
            "max_gen_toks": 32,
            "temperature": 0,
            "until": ["<|im_end|>", "\nQuestion:"],
        },
    )


def _local_chat_payload(stops: list[str]) -> dict:
    adapter = object.__new__(LocalChatCompletion)
    adapter.model = "served-model"
    adapter._max_gen_toks = 256
    return adapter._create_payload(
        messages=[{"role": "user", "content": "Question"}],
        gen_kwargs={"max_gen_toks": 32, "until": stops},
    )


def _local_completions_payload(stops: list[str]) -> dict:
    adapter = object.__new__(LocalCompletionsAPI)
    adapter.model = "served-model"
    adapter._max_gen_toks = 256
    return adapter._create_payload(
        messages="Question",
        generate=True,
        gen_kwargs={"max_gen_toks": 32, "until": stops},
    )


def test_gpt5_payload_uses_openai_provider_generation_controls():
    payload = _openai_payload("gpt-5")

    assert "stop" not in payload
    assert payload["temperature"] == 1


@pytest.mark.parametrize("model", ["qwen3.5-9b", "provider-model-5"])
def test_non_openai_alias_with_five_retains_configured_generation_controls(model):
    payload = _openai_payload(model)

    assert payload["stop"][:2] == ["\nQuestion:", "<|im_end|>"]
    assert payload["temperature"] == 0


def test_local_chat_payload_sends_only_token_sentinels():
    payload = _local_chat_payload(
        [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "Question:",
            "\nQ:",
            "\n[Question]",
            "\nUser:",
        ]
    )

    assert payload["stop"] == ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]


def test_local_chat_payload_without_sentinels_ends_the_turn_at_eos():
    payload = _local_chat_payload(["Question:", "\nQ:", "\nUser:", "\nAssistant:"])

    assert "stop" not in payload


def test_local_completions_payload_uses_the_same_bounded_stop_policy():
    stops = ["<|im_end|>", "<|eot_id|>", "Question:", "\nQ:", "\n[Question]", "\nUser:"]

    assert _local_completions_payload(stops)["stop"] == ["Question:", "\nQ:", "\n[Question]", "\nUser:"]




def test_local_chat_completion_preserves_reasoning_when_final_content_is_null():
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "2 + 2 = 4"},
            }
        ]
    }

    generated = _adapter().parse_generations(response)

    assert generated == ["2 + 2 = 4"]
    assert isinstance(generated[0], CompletionText)


def test_failed_completion_is_empty_and_classified_in_sample_artifact():
    response = FailedGeneration("model_transport")
    record = canonicalize_samples(
        "task",
        [{"resps": [[response]], "metrics": ["accuracy"], "accuracy": 0.0}],
    )[0]

    assert response == ""
    assert record["failure_category"] == "model_transport"
    assert record["completion_responses"][0][0]["normalized_content"] == ""












def test_successful_empty_chat_responses_invalidate_result_quality():
    responses = [
        completion_response_from_chat_choice(
            {"id": f"chatcmpl-{index}", "usage": {"completion_tokens": 2}},
            {"index": 0, "finish_reason": "stop", "message": {"content": None}},
        )
        for index in range(3)
    ]

    classifications = Counter(response.classification for response in responses)

    assert robust_api.completion_response_quality_invalid(classifications)


@pytest.mark.parametrize(
    "choice",
    [
        {"index": 0, "finish_reason": "stop", "message": {"content": None, "reasoning_content": "reasoning"}},
        {"index": 0, "finish_reason": "stop", "message": {"content": None, "reasoning": "reasoning"}},
        {"index": 0, "finish_reason": "stop", "message": {"content": None}, "reasoning_content": "reasoning"},
        {"index": 0, "finish_reason": "stop", "message": {"content": None}, "reasoning": "reasoning"},
    ],
)
def test_reasoning_aliases_are_scored_and_audited_in_every_task_path(choice):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}, "choices": [choice]}
    generated = _adapter().parse_generations(response)[0]

    record = json.loads(json.dumps(_scored_record(generated)))

    assert str(generated) == "reasoning"
    assert record["completion_responses"][0][0]["reasoning_content"] == "reasoning"
    assert record["completion_responses"][0][0]["classification"] == CompletionClassification.REASONING_ONLY
    assert record["completion_responses"][0][0]["raw_choice"] == choice


def test_vllm_qwen_message_reasoning_is_not_scored_as_empty():
    response = {
        "id": "chatcmpl-vllm-qwen",
        "usage": {"completion_tokens": 512},
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "9 eggs times $2 per egg equals 18"},
            }
        ],
    }

    generated = _adapter().parse_generations(response)[0]

    assert str(generated) == "9 eggs times $2 per egg equals 18"
    assert generated.response.classification == CompletionClassification.REASONING_ONLY_TRUNCATED


@pytest.mark.parametrize(
    ("message", "finish_reason", "expected", "classification"),
    [
        ({"content": "final"}, "stop", "final", CompletionClassification.FINAL),
        (
            {"content": "final", "reasoning_content": "reasoning"},
            "stop",
            "reasoning\n\nfinal",
            CompletionClassification.REASONING_AND_FINAL,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "length",
            "reasoning",
            CompletionClassification.REASONING_ONLY_TRUNCATED,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "stop",
            "reasoning",
            CompletionClassification.REASONING_ONLY,
        ),
        ({"content": None, "reasoning_content": None}, "stop", "", CompletionClassification.EMPTY),
    ],
)
def test_completion_response_normalizes_every_content_shape(message, finish_reason, expected, classification):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}}
    choice = {"index": 0, "finish_reason": finish_reason, "message": message}

    completion = completion_response_from_chat_choice(response, choice)

    assert completion.normalized_content() == expected
    assert completion.classification == classification
    assert completion.artifact()["raw_choice"] == choice
    assert completion.artifact()["usage"] == {"completion_tokens": 4}
    assert completion.artifact()["provider_metadata"] == {"id": "completion-1"}


@pytest.mark.parametrize("policy", [CompletionContentPolicy.COMBINE, CompletionContentPolicy.FINAL_ONLY])
@pytest.mark.parametrize(
    ("message", "finish_reason", "expected", "classification"),
    [
        ({"content": "final"}, "stop", "final", CompletionClassification.FINAL),
        (
            {"content": "final", "reasoning_content": "reasoning"},
            "stop",
            "reasoning\n\nfinal",
            CompletionClassification.REASONING_AND_FINAL,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "length",
            "reasoning",
            CompletionClassification.REASONING_ONLY_TRUNCATED,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "stop",
            "reasoning",
            CompletionClassification.REASONING_ONLY,
        ),
        ({"content": None, "reasoning_content": None}, "stop", "", CompletionClassification.EMPTY),
    ],
)
def test_reasoning_responses_are_scored_and_audited_in_every_task_path(
    policy, message, finish_reason, expected, classification
):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}}
    choice = {"index": 0, "finish_reason": finish_reason, "message": message}
    generated = _adapter(policy).parse_generations({**response, "choices": [choice]})[0]

    record = json.loads(json.dumps(_scored_record(generated)))

    expected_scorer_text = expected if policy == CompletionContentPolicy.COMBINE else message.get("content") or ""
    assert str(generated) == expected_scorer_text
    assert record["completion_responses"][0][0]["classification"] == classification
    assert record["completion_responses"][0][0]["raw_choice"] == choice
    assert record["completion_responses"][0][0]["usage"] == {"completion_tokens": 4}
    assert record["completion_responses"][0][0]["content_policy"] == policy
















def _scored_record(output):
    from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
    from eval.execution.engine import CollectSamples, Evaluator

    class Backend:
        async def execute(self, request):
            return output

    async def grade(spec, outputs):
        return Grade({"accuracy": 1.0})

    spec = EvalSpec("task", SourceRef("fixture"), "item", 0,
                    (ModelRequest(RequestKind.GENERATE, "question"),), GraderSpec("fixture"))
    sink = CollectSamples()
    async def run():
        async for completed in Evaluator(Backend(), grade, sink, workers=1).run([spec]):
            assert completed.error is None
    asyncio.run(run())
    return sink.samples["task"][0]
