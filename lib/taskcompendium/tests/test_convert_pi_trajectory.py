# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pi transcripts become prefix rows, and rows become one-turn tasks graded on reasoning markup."""

import json

import pytest
from verifyit.reasoning_markup import ReasoningViolation

from taskcompendium.convert.pi_trajectory import (
    HAZARD_SPLIT,
    UnsupportedTranscript,
    parse_pi_transcript,
    prefix_row,
    prefix_row_task,
    prompt_messages,
)
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    FunctionDefinition,
    Source,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.models import RuntimeEvidence, grading_attempt
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import require_submission_compatibility

BASH = {
    "name": "bash",
    "description": "Execute a bash command.",
    "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
    "constrainedSampling": {"type": "json_schema", "strict": "prefer"},
}
SOURCE = Source(dataset="open-athena/prefixes", revision="abc123", row="hazard.parquet:3", importer_revision="1")


def _message_end(message: dict) -> str:
    return json.dumps({"type": "message_end", "message": message})


def _assistant(text: str, call_id: str, command: str, prompt_tokens: int) -> str:
    return _message_end(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text},
                {"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": command}},
            ],
            "stopReason": "toolUse",
            "usage": {"input": prompt_tokens, "output": 10, "cacheRead": 0},
        }
    )


def _tool_result(call_id: str, text: str) -> str:
    return _message_end({"role": "toolResult", "toolCallId": call_id, "content": [{"type": "text", "text": text}]})


TRANSCRIPT = [
    json.dumps({"type": "session", "version": 3}),
    json.dumps({"type": "agent_start"}),
    _message_end(
        {
            "role": "system",
            "content": "",
            "sections": {"preamble": "You are an agent.", "tools": "<tools>\n- bash\n</tools>"},
            "toolsAdded": [BASH],
        }
    ),
    _message_end({"role": "user", "content": [{"type": "text", "text": "List the workspace."}]}),
    json.dumps({"type": "message_update", "message": {"role": "assistant", "content": []}}),
    _assistant("<|start_think|>Look first.<|end_think|>", "call-1", "ls", 1200),
    _tool_result("call-1", "a.txt\n"),
    _assistant("<|start_think|>Read it.<|end_think|>Reading.<|end_think|>", "call-2", "cat a.txt", 1300),
    _tool_result("call-2", "hello\n"),
    json.dumps({"type": "compaction_start", "reason": "threshold"}),
    _assistant("<|start_think|>After compaction.<|end_think|>", "call-3", "pwd", 400),
]


def _row(transcript, reply) -> dict:
    return prefix_row(transcript, reply, split=HAZARD_SPLIT, benchmark="bixbench", model="m", run="run-1", trial="t")


def test_transcript_yields_prefix_rows_classified_by_the_recorded_reply():
    transcript = parse_pi_transcript(TRANSCRIPT)

    assert transcript.compacted
    assert transcript.tools == (
        FunctionDefinition(name="bash", description="Execute a bash command.", parameters=BASH["parameters"]),
    )
    assert [(reply.violation, reply.prompt_tokens) for reply in transcript.replies] == [
        (None, 1200),
        (ReasoningViolation.STRAY_END, 1300),
    ]

    hazard = transcript.replies[1]
    assert prompt_messages(transcript, hazard) == [
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

    row = _row(transcript, hazard)
    assert (row["violation"], row["prompt_tokens"], row["position"]) == ("stray_end", 1300, 4)
    task = prefix_row_task(RawRow("hazard-3", SOURCE, row), ConversionContext({}, None))
    require_submission_compatibility(task)
    assert (task.id, task.source) == ("hazard-3", SOURCE)
    assert task.final_tools == transcript.tools
    assert task.context.events == transcript.events[:4]
    assert task.tags == ("hazard", "bixbench", "m", "stray_end")

    def grade(text: str) -> tuple[Outcome, float | None]:
        reply = AssistantToolCalls(
            calls=(ConversationToolCall(call_id="new", name="bash", arguments={"command": "cat a.txt"}),),
            content=text,
        )
        trace = ConversationTrace(events=(*task.context.events, reply))
        result = grade_task(task, grading_attempt(trace, RuntimeEvidence({}, "{}")))
        return result.status, result.reward

    assert grade("<|start_think|>Read it.<|end_think|>") == (Outcome.GRADED, 1.0)
    assert grade("<|start_think|>Read it.<|end_think|>Reading.<|end_think|>") == (Outcome.GRADED, 0.0)
    assert grade("<|start_think|>Read it.") == (Outcome.GRADED, 0.0)


def test_identical_prompts_share_a_digest_across_runs():
    first = parse_pi_transcript(TRANSCRIPT)
    second = parse_pi_transcript(TRANSCRIPT)
    assert _row(first, first.replies[0])["digest"] == _row(second, second.replies[0])["digest"]
    assert _row(first, first.replies[0])["digest"] != _row(first, first.replies[1])["digest"]


def test_text_only_replies_cut_off_logs_and_media_content_are_handled():
    transcript = parse_pi_transcript(
        [
            _message_end({"role": "system", "content": "", "sections": {"preamble": "Agent."}, "toolsAdded": [BASH]}),
            _message_end({"role": "user", "content": [{"type": "text", "text": "Say hi."}]}),
            _message_end(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<|start_think|>Greet"}],
                    "stopReason": "length",
                    "usage": {"input": 50, "output": 8192, "cacheRead": 0},
                }
            ),
        ]
    )
    (reply,) = transcript.replies
    # A span cut off by the length limit is truncation, not a malformed span.
    assert reply.violation is None
    assert transcript.events[-1] == TextMessage(role="assistant", content="<|start_think|>Greet")

    cut_off = parse_pi_transcript([*TRANSCRIPT[:7], TRANSCRIPT[7][:-40]])
    assert cut_off.truncated and len(cut_off.replies) == 1

    with pytest.raises(UnsupportedTranscript):
        parse_pi_transcript(
            [
                _message_end({"role": "system", "content": "", "sections": {}, "toolsAdded": [BASH]}),
                _message_end({"role": "user", "content": [{"type": "image", "data": "..."}]}),
            ]
        )
