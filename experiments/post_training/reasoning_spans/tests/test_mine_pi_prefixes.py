# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hazard prefixes are budgeted, paired with nearby controls, and deduplicated across runs."""

import json

from taskcompendium.convert.pi_trajectory import parse_pi_transcript

from experiments.post_training.reasoning_spans.mine_pi_prefixes import (
    MiningConfig,
    MiningReport,
    Trajectory,
    mine_rows,
    select_replies,
)

BASH = {
    "name": "bash",
    "description": "Execute a bash command.",
    "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
}
WELL_FORMED = "<|start_think|>Plan.<|end_think|>"
DOUBLE_THINK = "<|start_think|>Plan.<|end_think|>Then.<|end_think|>"


def _message_end(message: dict) -> str:
    return json.dumps({"type": "message_end", "message": message})


def _transcript(replies: list[tuple[str, int]], task: str = "Do the task.") -> list[str]:
    """A transcript whose assistant replies have the given texts and prompt lengths."""
    lines = [
        _message_end({"role": "system", "content": "", "sections": {"preamble": "Agent."}, "toolsAdded": [BASH]}),
        _message_end({"role": "user", "content": [{"type": "text", "text": task}]}),
    ]
    for index, (text, prompt_tokens) in enumerate(replies):
        call_id = f"call-{index}"
        command = f"step {index}"
        lines.append(
            _message_end(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": command}},
                    ],
                    "stopReason": "toolUse",
                    "usage": {"input": prompt_tokens, "output": 20, "cacheRead": 0},
                }
            )
        )
        lines.append(_message_end({"role": "toolResult", "toolCallId": call_id, "content": [{"type": "text", "text": "ok"}]}))
    return lines


def test_select_replies_budgets_hazards_and_pairs_the_nearest_well_formed_reply():
    transcript = parse_pi_transcript(
        _transcript(
            [
                (WELL_FORMED, 1000),
                (DOUBLE_THINK, 1100),
                (WELL_FORMED, 1200),
                (DOUBLE_THINK, 1300),
                (DOUBLE_THINK, 1400),
                (DOUBLE_THINK, 9000),
            ]
        )
    )

    selection = select_replies(transcript, max_prompt_tokens=5000, max_hazards=2)

    assert [reply.prompt_tokens for reply in selection.hazards] == [1100, 1300]
    # The first hazard sits between two well-formed replies and takes the earlier; the second takes the remaining one.
    assert [reply.prompt_tokens for reply in selection.controls] == [1000, 1200]


def test_mine_rows_deduplicates_identical_prompts_and_reports_counts():
    config = MiningConfig(output_path="unused", runs=("run-a", "run-b", "run-c"), check_prompt_tokens=False)
    lines = _transcript([(WELL_FORMED, 1000), (DOUBLE_THINK, 1100)])
    first = Trajectory("run-a", "trial-1", "bixbench", "model-a", parse_pi_transcript(lines))
    repeated = Trajectory("run-b", "trial-1", "bixbench", "model-b", parse_pi_transcript(lines))
    other_task = Trajectory(
        "run-b", "trial-2", "bfcl", "model-b", parse_pi_transcript(_transcript([(DOUBLE_THINK, 900)], "Other task."))
    )
    # The same opening prompt as ``first``, answered with a malformed span by a different checkpoint.
    promoter = Trajectory("run-c", "trial-1", "bixbench", "model-c", parse_pi_transcript(_transcript([(DOUBLE_THINK, 950)])))
    report = MiningReport()

    rows = mine_rows([first, repeated, other_task, promoter], config, None, report)

    assert [row["split"] for row in rows] == ["hazard", "hazard", "hazard"]
    assert (report.hazard_rows, report.control_rows, report.duplicate_rows, report.controls_promoted_to_hazard) == (
        3,
        0,
        2,
        1,
    )
    assert report.hazard_rows_by_run == {"run-a": 1, "run-b": 1, "run-c": 1}
    assert report.hazard_replies_by_violation == {"stray_end": 4}
    promoted = next(row for row in rows if row["run"] == "run-c")
    assert (promoted["model"], promoted["violation"], promoted["position"]) == ("model-c", "stray_end", 2)
    other = next(row for row in rows if row["benchmark"] == "bfcl")
    assert json.loads(other["messages"])[1]["content"] == "Other task."
