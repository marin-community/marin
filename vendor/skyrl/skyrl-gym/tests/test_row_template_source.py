import json

import pytest
from skyrl_train.dataset.tasks import source_row_task

from skyrl_gym.answer_tasks import grade_prompt_only
from skyrl_gym.task_sessions import AnswerTaskSession

SESSION = {
    "max_turns": 1,
    "max_total_tokens": None,
    "model_turn_timeout": None,
    "tool_turn_timeout": None,
    "total_turn_timeout": None,
    "attempt_timeout": None,
    "verifier_timeout": None,
    "cleanup_timeout": 60.0,
}
TOOLS = [{"type": "function", "function": {"name": "get_price", "parameters": {"type": "object"}}}]


def arrow_row(thinking: bool) -> dict:
    """A tool row as Arrow returns it: every message carries every field of the column."""
    blank = {"role": None, "content": None, "tool_call_id": None, "tool_calls": None}
    call = {"id": "call_1", "type": "function", "function": {"name": "get_price", "arguments": '{"company": "Apple"}'}}
    return {
        "prompt": [
            {**blank, "role": "user", "content": "Price of Apple?"},
            {**blank, "role": "assistant", "content": "", "tool_calls": [call]},
            {**blank, "role": "tool", "content": '{"price": 150}', "tool_call_id": "call_1"},
            {**blank, "role": "user", "content": "And Microsoft?"},
        ],
        "env_class": "prompt_only",
        "teacher_route": "tool_thinking" if thinking else "tool_nonthinking",
        "chat_template_kwargs": json.dumps({"enable_thinking": thinking}),
        "tools": json.dumps(TOOLS),
        "data_source": "fixture",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
async def test_a_mix_row_reaches_the_template_with_its_own_mode_and_tools(thinking: bool) -> None:
    task = source_row_task(
        arrow_row(thinking), 0, source_name="fixture", environment_configs={"session": SESSION, "prompt_only": {}}
    )
    session = AnswerTaskSession(task, None, grader=grade_prompt_only)
    try:
        start = await session.prepare()
    finally:
        await session.close()
    assert start.options["chat_template_kwargs"] == {"enable_thinking": thinking}
    assert start.options["tools"] == TOOLS
    assert [m["role"] for m in start.messages] == ["user", "assistant", "tool", "user"]
    assert start.messages[1]["tool_calls"][0]["id"] == start.messages[2]["tool_call_id"]
