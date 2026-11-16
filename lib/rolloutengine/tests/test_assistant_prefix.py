# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    FunctionDefinition,
    TextMessage,
)
from taskcompendium.submission import chat_request
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import StructureClosureSpec

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.assistant_prefix import AssistantPrefixHarness
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import GradedRolloutInterrupted, ModelTurn
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_spec


def closure_task():
    return arithmetic_task(
        answer_type=AnswerType.TEXT,
        context=ConversationInput(
            events=(
                TextMessage(role="user", content="Complete the JSON prefix."),
                TextMessage(role="assistant", content='{"x":'),
            )
        ),
        grader=verifyit_package(StructureClosureSpec('{"x":', '{"x":1}', ClosureFormat.JSON)).grader,
    )


async def test_assistant_prefix_harness_grades_one_continuation_with_actual_model_tokens():
    requests = []
    prompt_tokens = (10, 11)
    response_tokens = (20, 21)
    text = "1}"

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

    # Exercise serialized TaskSpecs rather than selecting semantics from the final role implicitly.
    task = closure_task()
    task = type(task).model_validate_json(task.model_dump_json())
    result = await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(
        rollout_spec(task, harness="assistant_prefix"), model
    )
    assert result.grade.reward == 1.0
    assert len(requests) == 1
    assert requests[0].messages[-1] == {"role": "assistant", "content": '{"x":'}
    assert requests[0].options["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    assert requests[0].prompt_token_ids is None
    record = result.rollout.conversations[0]
    assert record.prompt_token_ids == prompt_tokens
    assert record.response_token_ids == response_tokens
    assert record.loss_mask == tuple(1 for _ in response_tokens)


@pytest.mark.parametrize("channel", ["reasoning_content", "reasoning", "tool_calls"])
async def test_parsed_assistant_prefix_is_an_invalid_attempt_instead_of_a_normal_zero_reward(channel):
    message: dict = {"role": "assistant", "content": "", channel: "1}"}
    if channel == "tool_calls":
        message[channel] = [{"id": "call", "type": "function", "function": {"name": "write_result", "arguments": "{}"}}]
    model = ReplayModel([message])
    with pytest.raises(GradedRolloutInterrupted) as caught:
        await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(
            rollout_spec(closure_task(), harness="assistant_prefix"), model.complete
        )
    assert caught.value.grade.reward is None
    assert caught.value.rollout.failure.exception_type == "ValueError"


async def test_assistant_prefix_advertises_final_tool_schemas_without_executing_them():
    tool = FunctionDefinition(name="write_result", parameters={"type": "object"})
    task = closure_task().model_copy(update={"final_tools": (tool,)})
    model = ReplayModel([{"role": "assistant", "content": "1}"}])
    result = await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(
        rollout_spec(task, harness="assistant_prefix"), model.complete
    )
    assert result.grade.reward == 1
    assert model.requests[0].options["tools"] == [{"type": "function", "function": tool.model_dump(exclude_none=True)}]
    assert len(model.requests) == 1


async def test_finished_assistant_history_keeps_ordinary_chat_and_tool_harness_semantics():
    task = arithmetic_task(
        context=ConversationInput(
            events=(
                TextMessage(role="user", content="Tell me a fact about six."),
                TextMessage(role="assistant", content="Six is the smallest perfect number."),
            )
        )
    )
    assert chat_request(task)["messages"][-1]["role"] == "user"
    assert "extra_body" not in chat_request(task)
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    result = await RolloutEngine({}, {"shell": ShellHarness(command_timeout=None)}).run_task(
        rollout_spec(task, harness="shell"), model.complete
    )
    assert result.grade.reward == 1
    assert model.requests[0].messages[-1]["role"] == "user"
    assert "extra_body" not in model.requests[0].options
