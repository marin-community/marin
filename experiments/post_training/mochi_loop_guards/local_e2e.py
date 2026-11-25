# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end run: a tiny looping trial through the stall stop, grading, and the audit.

Runs one in-process RolloutEngine attempt whose model repeats one Terminus command with an
unchanged observation, with ``repeat_stall_batches=3``. The stop fires, the task is graded
on the work done, the graded rollout is serialized as a ``dumped_evals``-style row, and the
cohort audit computes the #12264 §2.1 metrics over it:

    uv run --package marin-core python -m experiments.post_training.mochi_loop_guards.local_e2e \\
        --output <scratch dir>
"""

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from experiments.post_training.mochi_loop_guards.stall_harness import (
    ConstantModel,
    ScriptedHarness,
    arithmetic_task,
    engine,
    rollout_spec,
)
from vendor.skyrl.scripts.loop_audit import audit_domain

logger = logging.getLogger(__name__)


async def run_trial(output: Path) -> dict:
    """Run the stalled attempt and serialize its graded rollout as an eval-trace row."""
    from rolloutengine.models import REPEAT_STALL_STOP_REASON  # noqa: PLC0415

    harness = ScriptedHarness(replies=8)
    model = ConstantModel(json_message("cat notes.txt"))
    result = await engine(model, harnesses={"scripted": harness}).run_task(
        rollout_spec(arithmetic_task(), harness="scripted", max_turns=12, repeat_stall_batches=3)
    )
    rollout = result.rollout
    assert rollout.stop_reason == REPEAT_STALL_STOP_REASON, rollout.stop_reason
    conversation = rollout.conversations[0]
    row = {
        "uid": "local-e2e-0",
        "attempt_id": rollout.run_id,
        "step": -1,
        "ordinal": 0,
        "is_last_step": True,
        "identity_disposition": "task_spec",
        "task_id": "local-e2e/stall",
        "prompt_token_ids": list(conversation.prompt_token_ids),
        "response_token_ids": list(conversation.response_token_ids),
        "input_prompt": "",
        "output_response": "",
        "messages": list(conversation.messages),
        "score": result.grade.reward,
        "stop_reason": rollout.stop_reason,
        "exception_type": None,
        "error_treatment": None,
        "server_error": None,
        "env_class": "local",
        "subset_id": None,
        "data_source": "local/e2e",
        "env_metrics": dict(rollout.metrics),
        "verification": {
            "status": "verified",
            "reason": None,
            "score": result.grade.reward,
            "passed": result.grade.reward == 1.0,
        },
    }
    path = output / "dumped_evals" / "local-e2e" / "eval_only" / "local_e2e.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")
    return row


def json_message(command: str) -> str:
    return json.dumps({"analysis": "a", "plan": "read the note", "commands": [{"keystrokes": command, "duration": 1.0}]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory for the trace row and the audit output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    row = asyncio.run(run_trial(output))
    cohort = {
        "rows": [{"task_id": "local-e2e/stall", "prompt_hash": None}],
        "max_period_tokens": 4096,
        "budget": 32768,
        "minimum_occurrences": 8,
        "interrupt": False,
    }
    domain = audit_domain("local-e2e", cohort, [row])
    audit = {"incidence": asdict(domain.incidence), "agent": asdict(domain.agent), "target_status": domain.target_status}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
