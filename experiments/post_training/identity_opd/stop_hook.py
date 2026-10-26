# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Watch the OPD run's step metrics and stop it on a nonfinite metric or a rising teacher gap.

gap_t is the step's mean over scored response tokens of (log p_student - log p_teacher), read as the negative of
``distillation/teacher_advantage_mean``: the trainer logs the mean of clip(teacher - student, +/-5) times the route
weight, every route weight is 1.0, and the student log-probs are the trainer's own (old) log-probs. The run stops
after 5 consecutive strict increases (6 values, no epsilon), or at the first step with a nonfinite metric.

Run detached. It first waits for the job's primary W&B run (the one whose name starts with ``--run-name-prefix``),
polling the Iris job each time: if the job reaches a terminal state before the W&B run exists, the hook records
that and exits without a cancel. It then polls W&B and exits when it fires, when the last step is logged, or when the
run ends. Every exit, including SIGTERM, writes the record file, prints it as one JSON line and notifies::

    setsid -f python -m experiments.post_training.identity_opd.stop_hook --project ENTITY/PROJECT \\
        --run-name-prefix RUN_NAME --record stop-hook.json --job /USER/JOB --notify margay ... > stop-hook.log 2>&1
"""

import argparse
import itertools
import json
import math
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

import wandb
from iris.resources.state import TERMINAL_JOB_STATES

GAP_METRIC = "distillation/teacher_advantage_mean"
STEP_METRIC = "trainer/global_step"
RISES_TO_STOP = 5
FINISHED_STATES = frozenset({"finished", "crashed", "failed", "killed"})
RESOLVE_TIMEOUT_SECONDS = 4 * 3600
JOB_TERMINAL = "job terminal before W&B run"
KILLED = "killed (SIGTERM)"


class Killed(Exception):
    """SIGTERM reached the hook."""


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


def parse_job_list(text: str) -> dict[str, str]:
    """Job id to state from ``iris job list`` output."""
    states = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("/"):
            states[fields[0]] = fields[1]
    return states


def iris_job_states(args: argparse.Namespace) -> dict[str, str]:
    """States of the job and its descendants; empty when the query fails, so the next poll retries."""
    result = subprocess.run(
        [".venv/bin/iris", "--config", args.iris_config, "job", "list", "--prefix", args.job, "--limit", "200"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        print(f"{_now()} iris job list failed ({result.returncode}): {result.stderr.strip()[-500:]}", flush=True)
        return {}
    return parse_job_list(result.stdout)


def primary_runs(api, args: argparse.Namespace) -> list:
    try:
        runs = api.runs(args.project, per_page=50)
        return [run for run in runs if run.name.startswith(args.run_name_prefix) and run.job_type != "worker_monitor"]
    except ValueError:
        return []  # the project exists only after the run's first log


def resolve_run(args: argparse.Namespace, api_factory, job_states, sleep) -> tuple[str | None, dict[str, str]]:
    """The primary run's path, or None with the Iris states once the job is terminal before the run exists."""
    deadline = time.monotonic() + RESOLVE_TIMEOUT_SECONDS
    while True:
        runs = primary_runs(api_factory(), args)
        if len(runs) > 1:
            raise RuntimeError(f"more than one primary run starts with {args.run_name_prefix}: {[r.id for r in runs]}")
        if runs:
            return f"{args.project}/{runs[0].id}", {}
        states = job_states(args)
        if states.get(args.job) in TERMINAL_JOB_STATES:
            return None, states
        if time.monotonic() > deadline:
            raise RuntimeError(f"no W&B run starting with {args.run_name_prefix} after {RESOLVE_TIMEOUT_SECONDS} s")
        print(f"{_now()} no W&B run yet; Iris states {states}", flush=True)
        sleep(args.interval)


def watch(run_path: str, args: argparse.Namespace, api_factory, sleep) -> dict:
    while True:
        run = api_factory().run(run_path)
        rows = step_rows(list(run.scan_history()))
        reason = stop_reason(rows)
        last_step = int(rows[-1][STEP_METRIC]) if rows else None
        if reason is not None or last_step == args.last_step or run.state in FINISHED_STATES:
            return {
                "checked_at": _now(),
                "run_path": run_path,
                "run_state": run.state,
                "gaps": gaps(rows),
                "stop_reason": reason,
                "fired": reason is not None,
            }
        sleep(args.interval)


def _raise_killed(signum, frame) -> None:
    raise Killed


def run_hook(args: argparse.Namespace, *, api_factory, job_states, cancel, notify, sleep=time.sleep) -> dict:
    """Resolve the run, watch it, cancel the job if the hook fires; always write, print and send the record."""
    previous_handler = signal.signal(signal.SIGTERM, _raise_killed)
    record: dict = {"job": args.job, "run_name_prefix": args.run_name_prefix}
    try:
        run_path, states = resolve_run(args, api_factory, job_states, sleep)
        if run_path is None:
            record |= {"checked_at": _now(), "stop_reason": JOB_TERMINAL, "fired": False, "iris_states": states}
        else:
            print(f"{_now()} run path {run_path}", flush=True)
            record |= watch(run_path, args, api_factory, sleep)
            if record["fired"]:
                record["cancel"] = cancel(args)
    except Killed:
        record |= {"checked_at": _now(), "stop_reason": KILLED, "fired": False}
    except Exception as error:
        record |= {"checked_at": _now(), "stop_reason": f"hook error: {type(error).__name__}: {error}", "fired": False}
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    with open(args.record, "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record), flush=True)
    if record["fired"]:
        outcome = "FIRED: " + record["stop_reason"]
    elif record["stop_reason"] is not None:
        outcome = record["stop_reason"]
    else:
        outcome = f"ended ({record['run_state']}), no stop"
    notify(args.notify, f"OPD stop hook {outcome}. Record: {args.record}")
    return record


def cancel_job(args: argparse.Namespace) -> dict:
    result = subprocess.run(
        [".venv/bin/iris", "--config", args.iris_config, "job", "cancel", args.job],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return {"job": args.job, "returncode": result.returncode, "output": result.stdout + result.stderr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="W&B ENTITY/PROJECT")
    parser.add_argument("--run-name-prefix", required=True, help="name prefix of the job's primary W&B run")
    parser.add_argument("--record", required=True, help="JSON file the outcome is written to")
    parser.add_argument("--job", required=True, help="Iris job to watch, and to cancel when the hook fires")
    parser.add_argument("--iris-config", default="lib/iris/config/marin.yaml")
    parser.add_argument("--notify", nargs="*", default=[], help="agent-swarm handles to message")
    parser.add_argument("--last-step", type=int, default=20)
    parser.add_argument("--interval", type=int, default=120, help="seconds between polls")
    args = parser.parse_args()
    record = run_hook(
        args,
        api_factory=lambda: wandb.Api(timeout=60),
        job_states=iris_job_states,
        cancel=cancel_job,
        notify=_notify,
    )
    sys.exit(143 if record["stop_reason"] == KILLED else 0)


if __name__ == "__main__":
    main()
