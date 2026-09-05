# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.gpt_oss_rollouts import row_to_chat_doc as gpt_oss_row_to_chat_doc
from marin.datakit.download.swe_rebench_openhands import row_to_chat_doc as openhands_row_to_chat_doc
from marin.datakit.download.swe_zero_12m import row_to_chat_doc as swe_zero_row_to_chat_doc


def test_gpt_oss_drops_assistant_reply_that_demonstrates_inline_tool_protocol() -> None:
    row = {
        "user_content": "How does function calling work?",
        "assistant_thinking": "Explain the format.",
        "assistant_content": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
    }

    assert gpt_oss_row_to_chat_doc(row) == []


def test_openhands_drops_unparsed_inline_tool_action() -> None:
    row = {
        "resolved": 0,
        "trajectory": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "Inspect it.\n<tool_call><parameter=path>x.py</parameter></function></tool_call>",
            },
            {"role": "user", "content": "Please continue."},
        ],
    }

    assert openhands_row_to_chat_doc(row) == []


def test_swe_zero_converts_reasoning_bash_and_observation() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Respond with THOUGHT and a bash block."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "THOUGHT: Inspect the file.\n\n```bash\nsed -n '1,80p' app.py\n```",
            },
            {"role": "user", "content": "Observation: file contents"},
            {
                "role": "assistant",
                "content": "THOUGHT: The edit is complete.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "user", "content": "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        ]
    }

    [document] = swe_zero_row_to_chat_doc(row)
    messages = document["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == "<|start_think|>Inspect the file.<|end_think|>"
    assert messages[2]["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command":"sed -n \'1,80p\' app.py"}',
    }
    assert messages[3]["content"] == "file contents"
    assert messages[-1]["content"] == "<|start_think|>The edit is complete.<|end_think|>\n\nTask complete."


def test_swe_zero_drops_trajectory_ending_with_observation() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "```bash\ncat app.py\n```"},
            {"role": "user", "content": "Observation: file contents"},
        ]
    }

    assert swe_zero_row_to_chat_doc(row) == []
