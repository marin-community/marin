# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Align replaced observations with downstream reasoning for the independent audit."""

from experiments.tool_normalization.terminal_protocol import terminal_protocol


def observation_alignment(original: list[dict], candidate: dict) -> list[dict]:
    terminal = terminal_protocol(original)
    rows = []
    call_index = 0
    turns = candidate["turns"]
    for position, turn in enumerate(turns):
        index = turn["source_message_index"]
        following = turns[position + 1]["source_message_index"] if position + 1 < len(turns) else None
        end = following if following is not None else len(original)
        old = [
            i
            for i in range(index + 1, end)
            if original[i]["role"] in ("tool", "ipython") or (terminal and i in terminal.observation_indices)
        ]
        count = len(turn["calls"])
        if old or count:
            rows.append(
                {
                    "source_message_index": index,
                    "original_observation_indices": old,
                    "actual_tool_call_ids": [f"replay_{i}" for i in range(call_index, call_index + count)],
                    "next_source_assistant_index": following,
                }
            )
        call_index += count
    return rows


def validate_observation_review(alignment: list[dict], reviews: list[dict]) -> None:
    expected = [row["source_message_index"] for row in alignment]
    if [row["source_message_index"] for row in reviews] != expected:
        raise ValueError("Audit must review every replaced observation group exactly once in order")
    if any(row["consistent"] is not True or not row["evidence"].strip() for row in reviews):
        raise ValueError("Dependent reasoning disagrees with actual observations or lacks evidence")
