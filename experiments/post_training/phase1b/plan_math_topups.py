# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan only replacement math-teacher attempts for a fixed 1,000-task panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.grade_math_teacher import read_jsonl, write_jsonl
from experiments.post_training.phase1b.select_math_teacher import sample_index

ROWS_PER_TASK = 4
MAX_ATTEMPTS_PER_TASK = 12
TASK_COUNT = 1_000


def plan(tasks: list[dict], verdicts: list[dict]) -> tuple[list[dict], dict]:
    if len(tasks) != TASK_COUNT or len({task["task_id"] for task in tasks}) != TASK_COUNT:
        raise ValueError(f"expected {TASK_COUNT} distinct ordered math tasks")
    target_ids = {task["task_id"] for task in tasks}
    by_task = defaultdict(list)
    seen = set()
    for verdict in verdicts:
        sample_id = verdict["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate math attempt: {sample_id}")
        seen.add(sample_id)
        task_id = verdict["task_id"]
        if task_id in target_ids:
            by_task[task_id].append(verdict)

    requests = []
    skipped = Counter()
    for task in tasks:
        task_id = task["task_id"]
        attempts = by_task[task_id]
        indices = sorted(sample_index(row["sample_id"], task_id) for row in attempts)
        if indices != list(range(len(indices))):
            raise ValueError(f"{task_id}: attempts are not contiguous from zero: {indices}")
        ended = sum(row["turn_ended"] for row in attempts)
        if ended > ROWS_PER_TASK:
            continue
        needed = ROWS_PER_TASK - ended
        if len(attempts) + needed > MAX_ATTEMPTS_PER_TASK:
            raise ValueError(f"{task_id}: {needed} missing rows cannot fit the fixed attempt cap")
        skipped.update(row["drop_reason"] for row in attempts if row["drop_reason"])
        for index in range(len(attempts), len(attempts) + needed):
            requests.append(
                {
                    "probe_id": f"{task_id}#sample-{index}",
                    "source": task_id,
                    "messages": [{"role": "user", "content": task["prompt"]}],
                }
            )
    manifest = {
        "tasks": len(tasks),
        "tasks_with_shortage": len({row["source"] for row in requests}),
        "new_tasks": sum(not by_task[task["task_id"]] for task in tasks),
        "replacement_attempts": len(requests),
        "prior_attempts_on_target_tasks": sum(len(by_task[task["task_id"]]) for task in tasks),
        "prior_drops": dict(skipped),
        "selection": "fixed task order, only missing turn-close rows, contiguous sample index, 12-attempt cap",
    }
    return requests, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--grade-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for uri in (args.tasks, *args.grade_root, args.out):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b math top-up paths must stay in marin-rno2a")
    configure_coreweave_s3()
    tasks = read_jsonl(args.tasks)
    verdicts = [row for root in args.grade_root for row in read_jsonl(f"{root.rstrip('/')}/verdicts.jsonl")]
    requests, manifest = plan(tasks, verdicts)
    payload = "".join(json.dumps(row) + "\n" for row in requests).encode()
    write_jsonl(args.out, requests)
    manifest.update(
        tasks_uri=args.tasks, grade_roots=args.grade_root, requests_sha256=hashlib.sha256(payload).hexdigest()
    )
    filesystem, path = filesystem_for(f"{args.out}.manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps({"out": args.out, **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
