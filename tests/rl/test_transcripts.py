# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the SkyRL rollout transcript renderer, on trimmed real records."""

import json

import pytest
from marin.rl.transcripts import (
    DEFAULT_TRUNCATE_CHARS,
    render_documents,
    to_html,
    to_markdown,
)

# Trimmed eval-dump trial row (bfclparity-pi doc 12; token ids and long text elided).
# Its assistant turn carries a tool call with null content, followed by the tool's reply.
EVAL_DUMP_ROW = {
    "uid": "12",
    "attempt_id": "a73813cd0729",
    "step": -1,
    "ordinal": 87,
    "task_id": "bfclparity-pi/bfcl-multiple-16",
    "input_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think",
    "output_response": '<tool_call>\n{"name": "bash", "arguments": {"command": "echo hi"}}',
    "messages": [
        {"role": "system", "content": "You are an expert coding assistant operating inside pi."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "# Task\n\nWhat is the required minimum population size?"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-b14c4605f90e32a0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "echo hi"}'},
                }
            ],
        },
        {"role": "tool", "content": "(no output)", "tool_call_id": "chatcmpl-tool-b14c4605f90e32a0"},
        {
            "role": "assistant",
            "content": "The file has been written. The required minimum effective population size is 50.",
        },
    ],
    "score": [0.0, 1.0],
    "stop_reason": "stop",
    "exception_type": None,
    "env_class": "harbor_trial",
    "data_source": "bfclparity-pi",
    "env_metrics": {"rollout_retries": 0.0, "truncation_penalized": 0.0},
    "verification": {"status": "verified", "reason": None, "score": 1.0, "passed": True},
}

# The same shape with the writer's null `messages` (evidence messages absent).
EVAL_DUMP_ROW_NULL_MESSAGES = {
    **EVAL_DUMP_ROW,
    "uid": "13",
    "messages": None,
}

# Trimmed retention-archive schema_v7 turn record with the Marin chat template's think markers.
SCHEMA_V7_RECORD = {
    "disposition": {"error_treatment": None, "exception_type": None, "server_error": None},
    "global_step": 7,
    "metrics": {
        "episode/assistant_turns": 1,
        "episode/conversations": 1,
        "episode/generated_tokens": 83,
        "episode/tool_calls": 0,
    },
    "phase": "train",
    "prompt": {"messages": [{"content": "Question: 40 pupils; half like blue.", "role": "user"}]},
    "provenance": {"model_source_identity": "mochi-long-loop/v5-sft2100@2026.11.24.128:78d9cd9d"},
    "record_id": "ee260d9873060396c4bd87dfdfd2f653e793c6eb3b9fed1bc8e9b5effedaa3be",
    "response": {
        "generation_limit": 16384,
        "messages": [
            {"content": "Question: 40 pupils; half like blue.", "role": "user"},
            {"annotations": None, "content": "15", "reasoning": None, "role": "assistant"},
        ],
        "step_boundaries": [{"row_index": 0, "stop_reason": "length", "token_end": 83, "token_start": 0}],
        "stop_reason": "length",
        "text": "<|start_think|>Half of 40 is 20.<|end_think|>15",
        "token_ids": [128002, 1687],
    },
    "reward": {"components": {"overlong": 0.0}, "outcome": 1.0, "shaped": 1.0},
    "run_id": "long-loop/mochi-v5-exp3-rl@2026.11.24.28",
    "schema_version": 7,
    "trajectory": {
        "environment_class": "taskcompendium",
        "environment_extras": {"data_source": "openai/gsm8k", "teacher_route": None},
        "instance_id": "eval-None-0--399",
        "repetition_id": 0,
    },
    "verification_result": {"diagnostics": {}, "passed": None, "reason": None, "score": 1.0, "status": "verified"},
    "verifier": None,
}


def _markdown(record) -> str:
    ((_, blocks),) = list(render_documents(record))
    return to_markdown(blocks)


def test_eval_dump_row_renders_turns_and_verdict() -> None:
    text = _markdown(EVAL_DUMP_ROW)
    assert "# Rollout 12" in text
    assert "- **source:** bfclparity-pi" in text
    assert "### system" in text and "### user" in text and "### assistant" in text
    assert "**Tool call: `bash`**" in text
    assert '"command": "echo hi"' in text
    assert "### tool — chatcmpl-tool-b14c4605f90e32a0" in text
    assert "## Verifier verdict" in text
    assert "- **verifier status:** verified" in text
    assert "- **verifier score:** 1" in text
    assert "- **verifier passed:** true" in text
    assert "- **stop reason:** stop" in text


def test_eval_dump_row_with_null_messages_uses_raw_prompt_and_response() -> None:
    text = _markdown(EVAL_DUMP_ROW_NULL_MESSAGES)
    assert "### user" in text
    assert "Reasoning: /think" in text
    assert "### assistant" in text
    assert '"name": \\"bash\\"' in text or "<tool_call>" in text


def test_turn_record_renders_thinking_in_details_and_header_facts() -> None:
    text = _markdown(SCHEMA_V7_RECORD)
    assert "# Rollout eval-None-0--399" in text
    assert "- **source:** openai/gsm8k" in text
    assert "- **step:** 7" in text
    assert "- **model:** mochi-long-loop/v5-sft2100@2026.11.24.128:78d9cd9d" in text
    assert "- **generated tokens:** 83" in text
    assert "<details>" in text and "<summary>Thinking</summary>" in text
    assert "Half of 40 is 20." in text
    assert "</details>" in text
    assert "### user" in text and "### assistant" in text
    # The messages lack the think block, so the raw response supplies it.
    assert "### assistant — raw response" in text
    assert "## Verifier verdict" in text
    assert "- **reward outcome:** 1" in text


def test_messages_carrying_thinking_skip_the_raw_response_section() -> None:
    record = json.loads(json.dumps(SCHEMA_V7_RECORD))
    record["response"]["messages"][-1]["content"] = "<|start_think|>Half of 40 is 20.<|end_think|>15"
    text = _markdown(record)
    assert "<summary>Thinking</summary>" in text
    assert "raw response" not in text
    assert text.count("Half of 40 is 20.") == 1


def test_length_stopped_record_reports_stop_reason() -> None:
    text = _markdown(SCHEMA_V7_RECORD)
    assert "- **stop reason:** length" in text


def test_tool_results_truncate_with_elision_marker() -> None:
    long_output = "x" * (DEFAULT_TRUNCATE_CHARS + 500)
    record = {
        "uid": "t1",
        "attempt_id": "a",
        "output_response": "done",
        "messages": [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "content": long_output, "tool_call_id": "c1"},
        ],
        "verification": {"status": "verified", "score": 1.0},
    }
    text = _markdown(record)
    assert f"… {500} chars elided]" in text
    assert long_output not in text


def test_cumulative_records_collapse_shared_prefix_and_keep_each_outcome() -> None:
    first = json.loads(json.dumps(SCHEMA_V7_RECORD))
    first["global_step"] = 7
    second = json.loads(json.dumps(SCHEMA_V7_RECORD))
    second["global_step"] = 7
    second["response"]["messages"] = [
        *SCHEMA_V7_RECORD["response"]["messages"],
        {"content": "Question: 41 pupils.", "role": "user"},
        {"content": "16", "role": "assistant"},
    ]
    second["response"]["stop_reason"] = "stop"
    ((slug, blocks),) = list(render_documents([first, second]))
    text = to_markdown(blocks)
    assert slug == "eval-None-0--399-r0"
    # The shared prefix renders once; each record's fresh turns and outcome both appear.
    assert text.count("Question: 40 pupils; half like blue.") == 1
    assert "Question: 41 pupils." in text
    assert text.count("## Verifier verdict") == 1


def test_same_instance_rerolled_at_a_later_step_renders_separately() -> None:
    later = json.loads(json.dumps(SCHEMA_V7_RECORD))
    later["global_step"] = 12
    later["record_id"] = "different-rollout"
    later["response"]["messages"] = [
        {"content": "Question: 40 pupils; half like blue.", "role": "user"},
        {"content": "21", "role": "assistant"},
    ]
    documents = list(render_documents([SCHEMA_V7_RECORD, later]))
    # One instance re-rolled at a later training step is a separate rollout, even
    # though both documents share a uid and repetition.
    assert len(documents) == 2
    texts = [to_markdown(blocks) for _, blocks in documents]
    assert any("- **step:** 7" in text for text in texts)
    assert any("- **step:** 12" in text for text in texts)


def test_records_group_by_uid_into_separate_documents() -> None:
    other = json.loads(json.dumps(SCHEMA_V7_RECORD))
    other["trajectory"] = {**SCHEMA_V7_RECORD["trajectory"], "instance_id": "eval-None-0--400"}
    documents = {slug: to_markdown(blocks) for slug, blocks in render_documents([SCHEMA_V7_RECORD, other])}
    assert sorted(documents) == ["eval-None-0--399-r0", "eval-None-0--400-r0"]


def test_same_instance_at_different_repetition_renders_separately() -> None:
    other = json.loads(json.dumps(SCHEMA_V7_RECORD))
    other["record_id"] = "different-rollout"
    other["trajectory"] = {**SCHEMA_V7_RECORD["trajectory"], "repetition_id": 3}
    other["response"]["messages"] = [
        {"content": "Question: 41 pupils.", "role": "user"},
        {"content": "16", "role": "assistant"},
    ]
    documents = {slug: to_markdown(blocks) for slug, blocks in render_documents([SCHEMA_V7_RECORD, other])}
    # Different repetitions of one instance are separate rollouts, not cumulative steps.
    assert sorted(documents) == ["eval-None-0--399-r0", "eval-None-0--399-r3"]
    assert "Question: 40 pupils" in documents["eval-None-0--399-r0"]
    assert "Question: 41 pupils." in documents["eval-None-0--399-r3"]


def test_jsonl_text_and_non_record_input() -> None:
    line = json.dumps(SCHEMA_V7_RECORD)
    documents = list(render_documents(line + "\n"))
    assert len(documents) == 1
    with pytest.raises(ValueError, match="not a SkyRL"):
        list(render_documents({"schema_version": 6, "response": {}}))


def test_html_render_escapes_and_renders_details() -> None:
    ((_, blocks),) = list(render_documents(SCHEMA_V7_RECORD))
    page = to_html(blocks)
    assert "<!DOCTYPE html>" in page
    assert "<details><summary>Thinking</summary>" in page
    assert "<h3>user</h3>" in page
    # Message text is escaped, not interpreted.
    assert "<p>Question: 40 pupils; half like blue.</p>" in page
