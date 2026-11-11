# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the four-attempt T1 batch when all canonical Harbor results are sealed."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.audit_terminal_teacher import result_paths
from experiments.post_training.phase1b.select_terminal_teacher import ATTEMPT_CAP, ROWS_PER_TASK, plan_turns
from experiments.post_training.probe_sets.continuations import read_jsonl

POLL_INTERVAL = 300
EXPECTED_TRIALS = 4_000
EXPECTED_INITIAL_TASKS = 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--task-metadata", required=True)
    parser.add_argument("--t1-model", required=True)
    parser.add_argument("--b-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-wait", type=int, default=100_800)
    parser.add_argument("--once", action="store_true", help="check the current result count and exit without writes")
    args = parser.parse_args()
    if args.max_wait <= 0:
        parser.error("max-wait must be positive")
    for uri in (args.results_root, args.task_metadata, args.t1_model, args.b_model, args.out):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b data paths must stay in marin-rno2a")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()
    deadline = time.monotonic() + args.max_wait
    while True:
        count = len(result_paths(args.results_root))
        logging.info("T1 canonical Harbor results: %d/%d", count, EXPECTED_TRIALS)
        if count > EXPECTED_TRIALS:
            raise ValueError("unexpected extra canonical T1 results")
        if args.once:
            return
        if count == EXPECTED_TRIALS:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"T1 batch did not seal within {args.max_wait} seconds")
        time.sleep(POLL_INTERVAL)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.post_training.phase1b.audit_terminal_teacher",
            "--results-root",
            args.results_root,
            "--t1-model",
            args.t1_model,
            "--b-model",
            args.b_model,
            "--out",
            args.out,
        ],
        check=True,
    )
    metadata = read_jsonl(args.task_metadata)
    if len(metadata) < EXPECTED_INITIAL_TASKS:
        raise ValueError("candidate pool lacks the original thousand terminal tasks")
    trials = read_jsonl(f"{args.out.rstrip('/')}/trials.jsonl")
    _, summary, items = plan_turns(trials, metadata)
    initial = items[:EXPECTED_INITIAL_TASKS]
    if any(item["attempts"] != ROWS_PER_TASK for item in initial):
        raise ValueError("initial terminal tasks do not each have four trials")
    deficits = [
        {
            "task_name": item["task_name"],
            "attempts": item["attempts"],
            "usable_episodes": item["usable_episodes"],
            "needed": max(0, ROWS_PER_TASK - item["usable_episodes"]),
            "remaining_attempts": ATTEMPT_CAP - item["attempts"],
        }
        for item in initial
        if item["usable_episodes"] < ROWS_PER_TASK
    ]
    plan = {
        "audit_root": args.out,
        "task_metadata": args.task_metadata,
        "initial_tasks": EXPECTED_INITIAL_TASKS,
        "initial_trials": EXPECTED_TRIALS,
        "deficit_tasks": len(deficits),
        "replacement_attempts_needed": sum(item["needed"] for item in deficits),
        "exhausted_tasks": sum(item["needed"] > item["remaining_attempts"] for item in deficits),
        "selected_rows_before_replacements": summary["selected_rows"],
        "selected_tasks_before_replacements": summary["selected_tasks"],
        "deficits": deficits,
    }
    filesystem, path = filesystem_for(f"{args.out.rstrip('/')}/topup-plan.json")
    with filesystem.open(path, "w") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    logging.info("Terminal audit and top-up plan written: %s", args.out)


if __name__ == "__main__":
    main()
