# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mined thoughtcrime rows convert to one-turn tool-calling tasks whose golden reply passes the grader."""

import json

from taskcompendium.models import AnswerType, AssistantToolCalls, ToolResult
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, IntendedUse

from experiments.post_training.task_curation.datasets.thoughtcrime import sources
from experiments.post_training.task_curation.tests.conversion import converted_task

BASH = {
    "name": "bash",
    "description": "Execute a bash command.",
    "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
}
MESSAGES = [
    {"role": "system", "content": "You are an agent.\n\n<tools>\n- bash\n</tools>"},
    {"role": "user", "content": "List the workspace."},
    {
        "role": "assistant",
        "content": "<|start_think|>Look first.<|end_think|>",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "a.txt\n"},
]
ROW = {
    "digest": "0" * 64,
    "messages": json.dumps(MESSAGES),
    "tools": json.dumps([BASH]),
    "recorded_violation": "stray_end",
    "prompt_tokens": 1300,
    "benchmark": "bixbench-cli-marin-corrections",
    "task": "bix-1-q1",
    "model": "Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
    "run": "20261017-022041-bixbench-pi",
    "trial": "bix-1-q1__abc",
    "position": 4,
    "reply": json.dumps({"role": "assistant", "content": "<|start_think|>Read it.<|end_think|>Reading.<|end_think|>"}),
}


def test_thoughtcrime_rows_become_tool_calling_tasks_whose_golden_reply_scores_one():
    (source,) = sources()
    config = source.config
    assert config is not None and config.controls is not None and config.intended_use == IntendedUse.EVAL

    task = converted_task(config, ROW)

    assert task.answer_type == AnswerType.NATIVE_ACTION
    assert [tool.name for tool in task.final_tools] == ["bash"]
    assert isinstance(task.context.events[2], AssistantToolCalls)
    assert task.context.events[3] == ToolResult(call_id="call-1", content="a.txt\n")
    assert task.tags == (
        "bixbench-cli-marin-corrections",
        "Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
        "stray_end",
    )
    verification = run_controls(task, controls=config.controls, machines=None)
    assert verification.checks and all(check.status == CheckStatus.PASS for check in verification.checks)
