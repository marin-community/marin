# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Watch the OPD run's step metrics and stop it on a nonfinite metric or a rising teacher gap.

gap_t is the step's mean over scored response tokens of (log p_student - log p_teacher), read as the negative of
``distillation/teacher_advantage_mean``: the trainer logs the mean of clip(teacher - student, +/-5) times the route
weight, every route weight is 1.0, and the student log-probs are the trainer's own (old) log-probs. The run stops
after 5 consecutive strict increases (6 values, no epsilon), or at the first step with a nonfinite metric.

Run detached; it polls W&B and exits when it fires, when the last step is logged, or when the run ends::

    setsid -f python -m experiments.post_training.identity_opd.stop_hook --run-path ENTITY/PROJECT/RUN_ID \\
        --record stop-hook.json --job /USER/JOB --notify margay ... > stop-hook.log 2>&1
"""

import argparse
import itertools
import json
import math
import subprocess
import time
from datetime import UTC, datetime

import wandb

GAP_METRIC = "distillation/teacher_advantage_mean"
STEP_METRIC = "trainer/global_step"
RISES_TO_STOP = 5
FINISHED_STATES = frozenset({"finished", "crashed", "failed", "killed"})


def step_rows(history: list[dict]) -> list[dict]:
    """One row per trainer step that logged the gap, in step order."""
    rows = {}
    for row in history:
        if row.get(GAP_METRIC) is not None and row.get(STEP_METRIC) is not None:
            rows[int(row[STEP_METRIC])] = row
    return [rows[step] for step in sorted(rows)]


def gaps(rows: list[dict]) -> list[tuple[int, float]]:
    return [(int(row[STEP_METRIC]), -float(row[GAP_METRIC])) for row in rows]


def stop_reason(rows: list[dict]) -> str | None:
    for row in rows:
        for key, value in sorted(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                return f"nonfinite {key}={value} at step {int(row[STEP_METRIC])}"
    series = gaps(rows)
    window = series[-(RISES_TO_STOP + 1) :]
    if len(window) == RISES_TO_STOP + 1 and all(b[1] > a[1] for a, b in itertools.pairwise(window)):
        steps = ", ".join(f"{step}: {gap:.6f}" for step, gap in window)
        return f"gap rose {RISES_TO_STOP} consecutive steps ({steps})"
    return None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _notify(handles: list[str], message: str) -> None:
    for handle in handles:
        subprocess.run(
            ["agent-swarm", "send", "--to", handle, "--context", "OPD stop hook", "--message", message],
            check=False,
            timeout=60,
        )


def watch(args: argparse.Namespace) -> dict:
    api = wandb.Api(timeout=60)
    while True:
        run = api.run(args.run_path)
        rows = step_rows(list(run.scan_history()))
        reason = stop_reason(rows)
        last_step = int(rows[-1][STEP_METRIC]) if rows else None
        if reason is not None or last_step == args.last_step or run.state in FINISHED_STATES:
            return {
                "checked_at": _now(),
                "run_path": args.run_path,
                "run_state": run.state,
                "gaps": gaps(rows),
                "stop_reason": reason,
            }
        time.sleep(args.interval)
        api = wandb.Api(timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-path", required=True, help="W&B ENTITY/PROJECT/RUN_ID")
    parser.add_argument("--record", required=True, help="JSON file the outcome is written to")
    parser.add_argument("--job", help="Iris job to cancel when the hook fires; without it the hook only reports")
    parser.add_argument("--iris-config", default="lib/iris/config/marin.yaml")
    parser.add_argument("--notify", nargs="*", default=[], help="agent-swarm handles to message")
    parser.add_argument("--last-step", type=int, default=20)
    parser.add_argument("--interval", type=int, default=120, help="seconds between polls")
    args = parser.parse_args()
    record = watch(args)
    if record["stop_reason"] is not None and args.job:
        result = subprocess.run(
            [".venv/bin/iris", "--config", args.iris_config, "job", "cancel", args.job],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        record["cancel"] = {"job": args.job, "returncode": result.returncode, "output": result.stdout + result.stderr}
    with open(args.record, "w") as handle:
        json.dump(record, handle, indent=1)
    outcome = "FIRED: " + record["stop_reason"] if record["stop_reason"] else f"ended ({record['run_state']}), no stop"
    _notify(args.notify, f"OPD stop hook {outcome}. Record: {args.record}")


if __name__ == "__main__":
    main()
