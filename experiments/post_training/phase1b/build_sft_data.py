# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lay out 4,000 unfiltered R and J teacher rows for matched 125-step SFT cells."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1a.matrix import STAGE_C_STEPS

ROWS_PER_TASK = 4
TASKS_PER_STEP = 8
TASKS_PER_SOURCE = STAGE_C_STEPS * TASKS_PER_STEP
ROWS_PER_STEP = TASKS_PER_STEP * ROWS_PER_TASK
ROWS_PER_SOURCE = STAGE_C_STEPS * ROWS_PER_STEP
PROMPT_CAP = 4_096
RESPONSE_CAP = 16_384
TURN_CLOSE = 128009
ROW_COLUMNS = ("task_id", "data_source", "teacher_route", "input_ids", "assistant_masks")


def read_jsonl(uri: str) -> list[dict]:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validated_row(row: dict, task_ids: set[str], source: str, teacher: str) -> dict:
    if row["task_id"] not in task_ids or row["data_source"] != source or row["teacher_route"] != teacher:
        raise ValueError(f"{source}: unexpected teacher row {row['task_id']}")
    ids = row["input_ids"]
    mask = row["assistant_masks"]
    if not ids or len(ids) != len(mask) or not any(mask) or set(mask) - {0, 1}:
        raise ValueError(f"{row['task_id']}: invalid tokenized completion mask")
    first = mask.index(1)
    if first == 0 or first > PROMPT_CAP or len(ids) - first > RESPONSE_CAP:
        raise ValueError(f"{row['task_id']}: exceeds the Phase 1a SFT token budget")
    if ids[-1] != TURN_CLOSE or mask[-1] != 1:
        raise ValueError(f"{row['task_id']}: incomplete or unsupervised final assistant turn")
    return {key: row[key] for key in ROW_COLUMNS}


def math_rows(rows: list[dict], order: list[str], teacher_route: str) -> list[dict]:
    """Validate four ended math samples per ordered prompt."""
    if len(order) != TASKS_PER_SOURCE or len(set(order)) != len(order):
        raise ValueError(f"task_R: expected {TASKS_PER_SOURCE} unique ordered task IDs")
    if any("sample_id" not in row for row in rows) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("task_R: missing or duplicate teacher sample IDs")
    grouped = defaultdict(list)
    task_ids = set(order)
    for row in rows:
        grouped[row["task_id"]].append(validated_row(row, task_ids, "task_R", teacher_route))
    if set(grouped) != set(order) or any(len(grouped[task_id]) != ROWS_PER_TASK for task_id in order):
        raise ValueError(f"task_R: expected four rows for each of {TASKS_PER_SOURCE} tasks")
    return [row for task_id in order for row in grouped[task_id]]


def terminal_rows(rows: list[dict], task_names: list[str]) -> list[dict]:
    """Validate four unfiltered terminal episodes per selected task."""
    if len(task_names) != TASKS_PER_SOURCE or len(set(task_names)) != len(task_names) or len(rows) != ROWS_PER_SOURCE:
        raise ValueError("task_J: invalid selected tasks or row count")
    if any("sample_id" not in row for row in rows) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("task_J: missing or duplicate teacher sample IDs")
    task_ids = set(task_names)
    validated = [validated_row(row, task_ids, "task_J", "T1") for row in rows]
    counts = defaultdict(int)
    for row in validated:
        counts[row["task_id"]] += 1
    if set(counts) != task_ids or set(counts.values()) != {ROWS_PER_TASK}:
        raise ValueError("task_J: expected four episodes for every selected task")
    return validated


def ordered_rows(r_rows: list[dict], j_rows: list[dict], condition: str) -> list[dict]:
    """Interleave 32 R and 32 J rows at each matched optimizer step."""
    result = []
    for step in range(STAGE_C_STEPS):
        start = step * ROWS_PER_STEP
        stop = start + ROWS_PER_STEP
        if condition in ("theta_R", "theta_RJ"):
            result.extend(r_rows[start:stop])
        if condition in ("theta_J", "theta_RJ"):
            result.extend(j_rows[start:stop])
    return result


def write_parquet(uri: str, rows: list[dict]) -> dict:
    table = pa.Table.from_pylist(rows)
    output = io.BytesIO()
    pq.write_table(table, output)
    payload = output.getvalue()
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "wb") as handle:
        handle.write(payload)
    return {"rows": len(rows), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-rows")
    parser.add_argument("--j-rows")
    parser.add_argument("--r-tasks", help="ordered selected math-train-final-1000.jsonl")
    parser.add_argument("--r-teacher-route", help="identity label for the math teacher checkpoint")
    parser.add_argument("--j-tasks", help="ordered selected J_train.rows.jsonl.tasks.jsonl")
    parser.add_argument("--r-only", action="store_true", help="seal the independent theta_R cell before J is ready")
    parser.add_argument("--j-only", action="store_true", help="seal the independent theta_J cell before R is ready")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.r_only and args.j_only:
        parser.error("--r-only and --j-only are mutually exclusive")
    if args.r_only and (args.j_rows or args.j_tasks):
        parser.error("--r-only cannot take J rows or tasks")
    if args.j_only and (args.r_rows or args.r_tasks or args.r_teacher_route):
        parser.error("--j-only cannot take R rows or tasks")
    if not args.j_only and not (args.r_rows and args.r_tasks and args.r_teacher_route):
        parser.error("R layout needs --r-rows, --r-tasks and --r-teacher-route")
    if not args.r_only and not (args.j_rows and args.j_tasks):
        parser.error("J layout needs --j-rows and --j-tasks")
    for uri in filter(None, (args.r_rows, args.r_tasks, args.j_rows, args.j_tasks, args.out)):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b SFT data paths must stay in marin-rno2a")
    configure_coreweave_s3()
    r_rows = []
    r_order = []
    if not args.j_only:
        r_order = [row["task_id"] for row in read_jsonl(args.r_tasks)]
        r_rows = math_rows(read_jsonl(args.r_rows), r_order, args.r_teacher_route)
    j_rows = []
    if not args.r_only:
        j_order = [row["task_name"] for row in read_jsonl(args.j_tasks)]
        if set(r_order) & set(j_order):
            raise ValueError("R and J training task IDs overlap")
        j_rows = terminal_rows(read_jsonl(args.j_rows), j_order)
    manifest = {"cells": {}}
    if not args.j_only:
        manifest.update(r_rows=args.r_rows, r_tasks=args.r_tasks, r_teacher_route=args.r_teacher_route)
    if not args.r_only:
        manifest.update(j_rows=args.j_rows, j_tasks=args.j_tasks)
    if args.r_only:
        conditions = ("theta_R",)
    elif args.j_only:
        conditions = ("theta_J",)
    else:
        conditions = ("theta_R", "theta_J", "theta_RJ")
    for condition in conditions:
        rows = ordered_rows(r_rows, j_rows, condition)
        uri = f"{args.out.rstrip('/')}/{condition}/seed0/m1.parquet"
        manifest["cells"][condition] = {"uri": uri, **write_parquet(uri, rows)}
    filesystem, path = filesystem_for(f"{args.out.rstrip('/')}/manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
