# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the open-thoughts/AgentTrove converter.

Covers the positive ``original_teacher`` allowlist (OpenAI ToS compliance)
and the conversation rendering path that preserves embedded tool-use JSON.
"""

from __future__ import annotations

from marin.datakit.download.agenttrove import (
    ALLOWED_TEACHERS,
    is_allowed_teacher,
    render_message,
    row_to_doc,
)


def _row(teacher: str | None, conversations: list[dict] | None) -> dict:
    return {"original_teacher": teacher, "conversations": conversations}


def test_allowed_teachers_excludes_openai_families():
    # Positive-allowlist guarantee: nothing OpenAI-shaped slips through, even
    # under exact-string match or case variation in the upstream metadata.
    for blocked in [
        "GPT-5-nano",
        "GPT-5-mini",
        "GPT-5",
        "GPT 5.1 Nano",
        "gpt-5-nano",
        "OpenAI o1",
        None,
        "",
        "Some Future Unaudited Teacher",
    ]:
        assert not is_allowed_teacher(blocked), f"{blocked!r} must be filtered out"


def test_allowed_teachers_passes_glm():
    assert "GLM-4.6" in ALLOWED_TEACHERS
    assert is_allowed_teacher("GLM-4.6")


def test_row_to_doc_drops_gpt_teacher_rows():
    for teacher in ["GPT-5-nano", "GPT-5-mini", "GPT 5.1 Nano"]:
        assert (
            row_to_doc(
                _row(
                    teacher,
                    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                )
            )
            == []
        )


def test_row_to_doc_drops_unknown_teacher_by_default():
    # Forward compatibility: an unrecognized teacher is dropped, not kept.
    assert row_to_doc(_row("SomeNewModel-2027", [{"role": "user", "content": "hi"}])) == []


def test_row_to_doc_drops_empty_conversations():
    assert row_to_doc(_row("GLM-4.6", None)) == []
    assert row_to_doc(_row("GLM-4.6", [])) == []


def test_row_to_doc_keeps_allowed_teacher_and_renders_turns():
    row = _row(
        "GLM-4.6",
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    )

    [doc] = row_to_doc(row)

    assert doc["source"] == "open-thoughts/AgentTrove"
    assert doc["text"] == "<user>\nWhat is 2+2?\n</user>\n\n<assistant>\n4\n</assistant>"
    # SHA-256 hexdigest is 64 chars; ids must be stable across runs.
    assert len(doc["id"]) == 64
    assert doc["id"] == row_to_doc(row)[0]["id"]


def test_row_to_doc_preserves_embedded_tool_calls():
    # Terminus-2 (the agent in AgentTrove) emits tool calls as JSON inside
    # the ``content`` field; the rendered document must keep that JSON
    # intact so downstream training sees the full tool-use trace.
    tool_call_json = (
        '{"analysis": "list the directory",'
        ' "plan": "run ls",'
        ' "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}],'
        ' "task_complete": false}'
    )
    row = _row(
        "GLM-4.6",
        [
            {"role": "user", "content": "Task: list files."},
            {"role": "assistant", "content": tool_call_json},
            {"role": "user", "content": "Terminal output:\nfile1\nfile2"},
        ],
    )

    [doc] = row_to_doc(row)

    assert tool_call_json in doc["text"]
    assert doc["text"].count("<assistant>") == 1
    assert doc["text"].count("</assistant>") == 1


def test_render_message_handles_missing_fields():
    assert render_message({"role": "user", "content": "hi"}) == "<user>\nhi\n</user>"
    assert render_message({"role": "user", "content": None}) == "<user>\n\n</user>"
    assert render_message({"content": "orphan"}) == "<unknown>\norphan\n</unknown>"
