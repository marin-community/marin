# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.chat_normalize import validate_chat_messages
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.calibforge_relay import row_to_chat_doc
from openai_harmony import Message

PROMPT = "Task: write /app/out.txt containing the word done.\n\nCurrent terminal state:\nroot@box:/app#"


def _reply(analysis: str, keystrokes: str, task_complete: bool = False) -> dict:
    return {
        "analysis": analysis,
        "plan": "Run the next command.",
        "commands": [{"keystrokes": keystrokes, "duration": 0.1}],
        "task_complete": task_complete,
    }


def _row(messages: list[dict]) -> dict:
    return {"sid": "episode-1", "teacher": "Qwen3.8-27B", "source": "horizon_0929", "passed": True, "messages": messages}


def test_relay_episode_keeps_teacher_reasoning_and_marks_trained_turns():
    student_reply = (
        "<|start_think|>Look first.<|end_think|>Let me look:\n\n<|start_think|>Then write.<|end_think|>"
        + json.dumps(_reply("Nothing there yet.", "ls\n"))[:-1]  # the harness accepts the missing closing brace
    )
    teacher_reply = "<|start_think|>Write the file and finish.<|end_think|>" + json.dumps(
        _reply("Ready to write.", "echo done > /app/out.txt\n", task_complete=True)
    )
    row = _row(
        [
            {"role": "system", "owner": "system", "content": "Reasoning: /think", "trained": False},
            {"role": "user", "owner": "environment", "content": PROMPT, "trained": False},
            {"role": "assistant", "owner": "student", "content": student_reply, "trained": False},
            {
                "role": "user",
                "owner": "environment",
                "content": "New Terminal Output:\nroot@box:/app#",
                "trained": False,
            },
            {"role": "assistant", "owner": "teacher", "content": teacher_reply, "trained": True},
        ]
    )

    [doc] = row_to_chat_doc(row)
    validate_chat_messages([Message.from_dict(message) for message in doc["messages"]])
    rendered = render_chat_record(doc)["text"]

    assert doc["source_id"] == "episode-1"
    assert doc["source_train_turns"] == [False, False, False, True]
    assert "Reasoning: /think" not in rendered
    assert PROMPT in rendered
    assert "<|start_think|>Look first.Let me look:\n\nThen write.<|end_think|>" in rendered
    assert "<|start_think|>Write the file and finish.<|end_think|>" in rendered
    assert '"keystrokes": "echo done > /app/out.txt\\n"' in rendered


@pytest.mark.parametrize("trained", [False, True])
def test_assistant_reply_without_command_drops_the_episode(trained):
    row = _row(
        [
            {"role": "user", "owner": "environment", "content": PROMPT, "trained": False},
            {
                "role": "assistant",
                "owner": "teacher",
                "content": "<|start_think|>Out of tokens.<|end_think|>",
                "trained": trained,
            },
        ]
    )

    assert row_to_chat_doc(row) == []


@pytest.mark.parametrize("close_action", [False, True])
def test_reasoning_example_keeps_the_actual_command(close_action):
    example = json.dumps(_reply("Example only.", "echo example\n"))
    action = _reply("Write the file.", "echo done > /app/out.txt\n", task_complete=True)
    action_json = json.dumps(action)
    content = (
        "<|start_think|>An example would be "
        + example
        + ". Use the actual action.<|end_think|>"
        + (action_json if close_action else action_json[:-1])
    )
    [doc] = row_to_chat_doc(
        _row(
            [
                {"role": "user", "content": PROMPT, "trained": False},
                {"role": "assistant", "content": content, "trained": True},
            ]
        )
    )
    final = doc["messages"][-1]
    assert json.loads(final["content"][0]["text"]) == action
    assert doc["messages"][-2]["content"][0]["text"] == ("An example would be " + example + ". Use the actual action.")


def test_command_payload_wrapped_in_reasoning_is_retained():
    action = _reply("Finish.", "echo done > /app/out.txt\n", task_complete=True)
    [doc] = row_to_chat_doc(
        _row(
            [
                {"role": "user", "content": PROMPT, "trained": False},
                {
                    "role": "assistant",
                    "content": "<|start_think|>" + json.dumps(action) + "<|end_think|>",
                    "trained": True,
                },
            ]
        )
    )
    assert json.loads(doc["messages"][-1]["content"][0]["text"]) == action
