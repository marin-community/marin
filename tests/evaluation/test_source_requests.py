# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay source-oracle messages through the maintained chat transport profiles."""

import json
from pathlib import Path

import httpx
import openai
import pytest
from marin.evaluation.task_driver import EvaluationRunConfig, ModelIdentity, generation_options
from rolloutengine.chat import ChatModel
from rolloutengine.models import ModelRequest, RolloutLimits

from experiments.post_training.task_curation.datasets.evalchemy import sources

FIXTURES = Path(__file__).parents[2] / "experiments/post_training/task_curation/tests/fixtures"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "benchmark,fixture,source_name,thinking",
    [
        ("HumanEvalPlus", "code", "humanevalplus", False),
        ("MBPPPlus", "code", "mbppplus", False),
        ("CruxEval", "code", "cruxeval", False),
        ("FinanceBench", "judged", "FinanceBench", False),
        ("IFBench", "instructions", "IFBench", False),
        ("SimpleQA", "judged", "SimpleQA", True),
    ],
)
async def test_maintained_chat_profiles_reach_endpoint_with_exact_source_messages(
    benchmark,
    fixture,
    source_name,
    thinking,
):
    oracle = json.loads((FIXTURES / f"evalchemy_{fixture}_oracle.json").read_text())
    reference = next(case for case in oracle["cases"] if case.get("suite", case.get("name")) == source_name)
    messages = (
        [{"role": "user", "content": reference["prompts"][0]}]
        if fixture == "code"
        else reference["specs"][0]["requests"][0]["prompt"]
    )
    protocol = next(source.config.protocol for source in sources() if source.config.protocol.benchmark == benchmark)
    config = EvaluationRunConfig(
        model=ModelIdentity(
            name="fixture",
            revision="model-commit",
            tokenizer="fixture",
            tokenizer_revision="tokenizer-commit",
            context_window=65536,
            serving={"backend": "fake"},
        ),
        max_tokens=4096,
        concurrency=1,
        source_limit=1,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=10,
            tool_turn_timeout=10,
            total_turn_timeout=20,
            attempt_timeout=30,
            verifier_timeout=10,
            cleanup_timeout=10,
        ),
    )
    captured = []

    def serve(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "source-profile",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        turn = await ChatModel(client, "fixture", 4096)(
            ModelRequest(messages=tuple(messages), options=generation_options(protocol, config, 0))
        )
    assert turn.text == "answer"
    assert len(captured) == 1
    assert captured[0]["messages"] == messages
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": thinking}
