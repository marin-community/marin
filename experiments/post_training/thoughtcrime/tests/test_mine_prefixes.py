# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Double-think prefixes come from a trajectory's first malformed reply, one per benchmark task."""

import json

from taskcompendium.convert.pi_trajectory import parse_pi_transcript

from experiments.post_training.thoughtcrime.mine_prefixes import (
    MiningConfig,
    MiningReport,
    Trajectory,
    mine_rows,
    source_runs,
)

BASH = {
    "name": "bash",
    "description": "Execute a bash command.",
    "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
}
WELL_FORMED = "<|start_think|>Plan.<|end_think|>"
DOUBLE_THINK = "<|start_think|>Plan.<|end_think|>Then.<|end_think|>"
UNCLOSED = "<|start_think|>Plan."


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
        lines.append(
            _message_end({"role": "toolResult", "toolCallId": call_id, "content": [{"type": "text", "text": "ok"}]})
        )
    return lines


def _trajectory(trial: str, model: str, replies: list[tuple[str, int]], benchmark: str = "bixbench") -> Trajectory:
    task = trial.split("__")[0]
    transcript = parse_pi_transcript(_transcript(replies, f"Solve {task} as {model}."))
    return Trajectory(f"run-{model}", trial, benchmark, task, model, transcript)


def test_mine_rows_keeps_one_first_failure_double_think_per_task():
    config = MiningConfig(output_path="unused", max_prompt_tokens=5000, check_prompt_tokens=False)
    trajectories = [
        # Two checkpoints reach a double-think on the same task; one row survives.
        _trajectory("bix-1-q1__a", "model-a", [(WELL_FORMED, 1000), (DOUBLE_THINK, 1100), (DOUBLE_THINK, 1200)]),
        _trajectory("bix-1-q1__b", "model-b", [(DOUBLE_THINK, 900)]),
        # The first malformed reply is unclosed, so the later double-think has a failure in its history.
        _trajectory("bix-2-q1__a", "model-a", [(UNCLOSED, 1000), (DOUBLE_THINK, 1100)]),
        # The first double-think's prompt exceeds the budget; later ones are not considered.
        _trajectory("bix-3-q1__a", "model-a", [(DOUBLE_THINK, 6000), (DOUBLE_THINK, 1100)]),
        _trajectory("bix-4-q1__a", "model-a", [(WELL_FORMED, 1000)]),
        _trajectory("simple_1__a", "model-b", [(WELL_FORMED, 1000), (DOUBLE_THINK, 1300)], benchmark="bfcl"),
    ]
    report = MiningReport()

    rows = mine_rows(trajectories, config, None, report)

    assert sorted((row.benchmark, row.task) for row in rows) == [("bfcl", "simple_1"), ("bixbench", "bix-1-q1")]
    bix = next(row for row in rows if row.task == "bix-1-q1")
    candidates = {row.digest for row in mine_rows(trajectories[:1], config, None, MiningReport())} | {
        row.digest for row in mine_rows(trajectories[1:2], config, None, MiningReport())
    }
    assert bix.digest == min(candidates)
    assert (bix.recorded_violation, bix.position) == ("stray_end", 4 if bix.model == "model-a" else 2)
    assert report.first_violations == {"stray_end": 4, "unclosed": 1, "none": 1}
    assert (report.double_think_trajectories, report.first_double_think_over_budget) == (3, 1)
    assert report.rows_by_benchmark == {"bixbench": 1, "bfcl": 1}


def test_source_runs_are_marin_pi_runs_before_the_cutoff():
    names = [
        "20261019-104831-open-athena-Snowball-Smartie-RL-2026.10.15.32-step14-bixbench-pi-72d5",
        "20261017-022041-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-tau3-pi-db16",
        "20261103-204830-dml-quickloop-d25-step25-59d7-bfclparity-pi-bfed",
        "20261018-010101-Qwen-Qwen3.6-35B-A3B-bixbench-pi-1234",
        "20261018-010101-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-swebench100-pi-1234",
        "20261105-000000-open-athena-Snowball-Smartie-RL-2026.10.15.32-step14-bixbench-pi-9999",
    ]

    assert source_runs(names, "20261104-000000") == tuple(sorted(names[:3]))
