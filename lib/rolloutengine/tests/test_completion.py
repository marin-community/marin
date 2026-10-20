# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-turn conversations and assistant prefills share model rendering and grading."""

import pytest
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, ConversationInput, TextMessage
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import StructureClosureSpec

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.completion import CompletionHarness
from rolloutengine.models import ModelTurn
from rolloutengine.test_support import arithmetic_task, rollout_spec


@pytest.mark.parametrize(
    "prompt_tokens,response_tokens,text",
    [((10, 11), (20, 21), "1}"), ((40, 41, 42), (70, 71, 72), " 1 } trailing output")],
)
async def test_completion_grades_prefill_with_the_models_own_tokenization(prompt_tokens, response_tokens, text):
    requests = []

    async def model(request):
        requests.append(request)
        return ModelTurn(
            {"role": "assistant", "content": text},
            prompt_tokens,
            response_tokens,
            tuple(-0.1 for _ in response_tokens),
            "stop",
            text="<think>private reasoning</think>" + text,
        )

    task = arithmetic_task(
        answer_type=AnswerType.TEXT,
        context=ConversationInput(
            events=(
                TextMessage(role="user", content="Complete the JSON prefix."),
                TextMessage(role="assistant", content='{"x":'),
            )
        ),
        grader=verifyit_package(
            StructureClosureSpec(prefix='{"x":', expected='{"x":1}', format=ClosureFormat.JSON, terminal_markers=())
        ).grader,
    )
    engine = RolloutEngine({}, {"completion": CompletionHarness()})
    result = await engine.run_task(rollout_spec(task, harness="completion"), model)
    assert result.grade.reward == 1.0
    assert len(requests) == 1
    assert requests[0].messages[-1] == {"role": "assistant", "content": '{"x":'}
    assert requests[0].prompt_token_ids is None
    assert requests[0].options["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    record = result.rollout.conversations[0]
    assert record.prompt_token_ids == prompt_tokens
    assert record.response_token_ids == response_tokens
    assert record.loss_mask == tuple(1 for _ in response_tokens)


async def test_completion_answers_an_ordinary_conversation_without_prefill():
    requests = []

    async def model(request):
        requests.append(request)
        return ModelTurn({"role": "assistant", "content": "12"}, (1, 2), (3,), (-0.1,), "stop")

    result = await RolloutEngine({}, {"completion": CompletionHarness()}).run_task(
        rollout_spec(arithmetic_task(), harness="completion"), model
    )
    assert result.grade.reward == 1.0
    assert requests[0].messages[0] == {"role": "user", "content": "What is six plus six?"}
    assert "extra_body" not in requests[0].options
