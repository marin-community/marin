# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select four turn-ended math-teacher completions per Phase 1b prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.grade_math_teacher import read_jsonl, write_jsonl

ROWS_PER_TASK = 4
TASK_COUNT = 1_000


def sample_index(sample_id: str, task_id: str) -> int:
    prefix = f"{task_id}#sample-"
    if not sample_id.startswith(prefix):
        raise ValueError(f"sample {sample_id} does not belong to {task_id}")
    return int(sample_id.removeprefix(prefix))


def select_rows(order: list[str], grade_roots: list[str]) -> tuple[list[dict], dict[str, int], dict]:
    """Keep the first four turn-ended samples by attempt index, or report shortages."""
    if len(order) != TASK_COUNT or len(set(order)) != TASK_COUNT:
        raise ValueError(f"expected {TASK_COUNT} distinct ordered math tasks")
    verdicts = []
    candidates = []
    for root in grade_roots:
        verdicts.extend(read_jsonl(f"{root.rstrip('/')}/verdicts.jsonl"))
        candidates.extend(read_jsonl(f"{root.rstrip('/')}/ended-candidates.jsonl"))
    by_sample = {row["sample_id"]: row for row in verdicts}
    candidate_by_sample = {row["sample_id"]: row for row in candidates}
    if len(by_sample) != len(verdicts) or len(candidate_by_sample) != len(candidates):
        raise ValueError("duplicate M1 sample IDs across grade batches")
    if set(candidate_by_sample) != {row["sample_id"] for row in verdicts if row["turn_ended"]}:
        raise ValueError("M1 candidate rows do not match turn-ended verdicts")
    attempts = Counter(row["task_id"] for row in verdicts)
    if any(count > 12 for count in attempts.values()):
        raise ValueError("M1 prompt exceeded the fixed 12-attempt cap")
    ended_by_task = defaultdict(list)
    for verdict in verdicts:
        if not verdict["turn_ended"]:
            continue
        task_id = verdict["task_id"]
        sample_id = verdict["sample_id"]
        candidate = candidate_by_sample[sample_id]
        if candidate["task_id"] != task_id:
            raise ValueError(f"M1 candidate task mismatch for {sample_id}")
        ended_by_task[task_id].append({**candidate, "graded_correct": verdict["graded_correct"]})
    shortages = {
        task_id: ROWS_PER_TASK - len(ended_by_task[task_id])
        for task_id in order
        if len(ended_by_task[task_id]) < ROWS_PER_TASK
    }
    if shortages:
        return [], shortages, {}
    selected = [
        row
        for task_id in order
        for row in sorted(ended_by_task[task_id], key=lambda item: sample_index(item["sample_id"], task_id))[
            :ROWS_PER_TASK
        ]
    ]
    selected_ids = {row["sample_id"] for row in selected}
    counts = {
        "attempts": len(verdicts),
        "non_turn_close": sum(not row["turn_ended"] for row in verdicts),
        "other_eos": sum(row["drop_reason"] == "other_eos" for row in verdicts),
        "cap_or_unterminated": sum(row["drop_reason"] == "cap_or_unterminated" for row in verdicts),
        "ended": len(candidates),
        "eligible_task_pool": sum(len(ended_by_task[task_id]) >= ROWS_PER_TASK for task_id in order),
        "dropped_candidate_tasks": sum(len(ended_by_task[task_id]) < ROWS_PER_TASK for task_id in order),
        "selected_correct": sum(row["graded_correct"] is True for row in verdicts if row["sample_id"] in selected_ids),
        "selected_incorrect": sum(
            row["graded_correct"] is False for row in verdicts if row["sample_id"] in selected_ids
        ),
        "selected_ungraded": sum(row["graded_correct"] is None for row in verdicts if row["sample_id"] in selected_ids),
    }
    return selected, {}, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="ordered candidate math task pool")
    parser.add_argument(
        "--grade-root", action="append", required=True, help="one or more completed M1 grade batch roots"
    )
    parser.add_argument("--selected-tasks-out", required=True, help="ordered 1,000-task selected math panel")
    parser.add_argument("--out", required=True, help="selected R_train.rows.jsonl")
    args = parser.parse_args()
    for uri in (args.tasks, *args.grade_root, args.selected_tasks_out, args.out):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b math selection paths must stay in marin-rno2a")
    configure_coreweave_s3()
    candidate_tasks = read_jsonl(args.tasks)
    candidate_ids = [row["task_id"] for row in candidate_tasks]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate math task IDs")
    correct = Counter()
    ended = Counter()
    attempts = Counter()
    other_eos = Counter()
    cap_or_unterminated = Counter()
    for root in args.grade_root:
        for verdict in read_jsonl(f"{root.rstrip('/')}/verdicts.jsonl"):
            correct[verdict["task_id"]] += verdict["graded_correct"] is True
            ended[verdict["task_id"]] += verdict["turn_ended"]
            attempts[verdict["task_id"]] += 1
            other_eos[verdict["task_id"]] += verdict["drop_reason"] == "other_eos"
            cap_or_unterminated[verdict["task_id"]] += verdict["drop_reason"] == "cap_or_unterminated"
    out_of_panel = set(attempts) - set(candidate_ids)
    eligible_tasks = [row for row in candidate_tasks if ended[row["task_id"]] >= ROWS_PER_TASK]
    if len(eligible_tasks) < TASK_COUNT:
        print(json.dumps({"eligible_tasks": len(eligible_tasks), "needed": TASK_COUNT}, sort_keys=True))
        raise SystemExit(2)
    selected_tasks = eligible_tasks[:TASK_COUNT]
    order = [row["task_id"] for row in selected_tasks]
    selected_ids = set(order)
    selected, shortages, counts = select_rows(order, args.grade_root)
    if shortages:
        print(
            json.dumps(
                {"shortage_tasks": len(shortages), "missing_rows": sum(shortages.values()), "shortages": shortages},
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    write_jsonl(args.selected_tasks_out, selected_tasks)
    write_jsonl(args.out, selected)
    per_item_audit_uri = f"{args.out}.items.jsonl"
    write_jsonl(
        per_item_audit_uri,
        [
            {
                "task_id": row["task_id"],
                "source": row.get("source", "unknown"),
                "attempts": attempts[row["task_id"]],
                "turn_ended": ended[row["task_id"]],
                "graded_correct": correct[row["task_id"]],
                "other_eos": other_eos[row["task_id"]],
                "cap_or_unterminated": cap_or_unterminated[row["task_id"]],
                "selected": row["task_id"] in selected_ids,
            }
            for row in candidate_tasks
        ],
    )
    filesystem, path = filesystem_for(args.out)
    payload = "".join(json.dumps(row) + "\n" for row in selected).encode()
    manifest = {
        "tasks": len(order),
        "candidate_tasks": len(candidate_tasks),
        "out_of_panel_tasks": len(out_of_panel),
        "out_of_panel_attempts": sum(attempts[task_id] for task_id in out_of_panel),
        "eligible_tasks": len(eligible_tasks),
        "dropped_zero_ended_tasks": sum(ended[task_id] == 0 for task_id in candidate_ids),
        "dropped_one_to_three_ended_tasks": sum(0 < ended[task_id] < ROWS_PER_TASK for task_id in candidate_ids),
        "candidate_zero_correct_tasks": sum(correct[task_id] == 0 for task_id in candidate_ids),
        "candidate_source_counts": dict(Counter(row.get("source", "unknown") for row in candidate_tasks)),
        "selected_source_counts": dict(Counter(row.get("source", "unknown") for row in selected_tasks)),
        "rows": len(selected),
        "selection": "first four turn-ended attempts by sample index, without correctness filtering; 12-attempt cap",
        "grade_roots": args.grade_root,
        "selected_tasks_out": args.selected_tasks_out,
        "per_item_audit_uri": per_item_audit_uri,
        "selected_tasks_sha256": (
            hashlib.sha256("".join(json.dumps(row) + "\n" for row in selected_tasks).encode()).hexdigest()
        ),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        **counts,
    }
    manifest_uri = f"{args.out}.manifest.json"
    manifest_filesystem, manifest_path = filesystem_for(manifest_uri)
    with manifest_filesystem.open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {"out": args.out, "bytes": filesystem.info(path)["size"], **manifest},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
