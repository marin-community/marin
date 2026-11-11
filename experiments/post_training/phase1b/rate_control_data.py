# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build distinct R-prime teacher requests and paired rate-control SFT files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.build_sft_data import ROW_COLUMNS, ROWS_PER_SOURCE, write_parquet
from experiments.post_training.phase1b.grade_math_teacher import read_jsonl, write_jsonl

TASKS_PER_STEP = 8
ROWS_PER_TASK = 4
ORIGINAL_R_SHA256 = "e2930392a09f8114a47e274608c8f8e2a7df67089d77d28d7bb1c5c1087b8623"
ORIGINAL_J_SHA256 = "74ad997a0dcd0a2808a94095fa438e101c690136b50a9902264448fd5eeaa906"
ORIGINAL_RJ_SHA256 = "f73501e7f996e758a931e4e8d550d8b79d1cc782f69f27f40271d3b0364170fc"


def _regional(uri: str) -> str:
    if not uri.startswith("s3://marin-rno2a/"):
        raise ValueError(f"Phase 1b data must stay in RNO2A: {uri}")
    return uri


def extra_requests(tasks: list[dict], source_probes: list[dict], start_index: int) -> list[dict]:
    """Reuse the exact screened prompts with fresh deterministic sampling seeds."""
    if len(tasks) != 1_000 or len({row["task_id"] for row in tasks}) != len(tasks):
        raise ValueError("expected the sealed 1,000 unique R tasks")
    messages_by_task: dict[str, list[dict]] = {}
    for probe in source_probes:
        task_id = probe["source"]
        if task_id in messages_by_task and probe["messages"] != messages_by_task[task_id]:
            raise ValueError(f"source probes disagree on prompt {task_id}")
        messages_by_task[task_id] = probe["messages"]
    rows = []
    old_ids = {row["probe_id"] for row in source_probes}
    for task in tasks:
        task_id = task["task_id"]
        messages = [{"role": "user", "content": task["prompt"]}]
        if task_id in messages_by_task and messages_by_task[task_id] != messages:
            raise ValueError(f"source prompt changed for {task_id}")
        for index in range(start_index, start_index + ROWS_PER_TASK):
            probe_id = f"{task_id}#sample-{index}"
            if probe_id in old_ids:
                raise ValueError(f"attempt ID already exists: {probe_id}")
            rows.append({"probe_id": probe_id, "source": task_id, "messages": messages})
    return rows


def topup_requests(tasks: list[dict], verdicts: list[dict], start_index: int) -> list[dict]:
    """Replace only unended attempts, retaining four distinct turns per task."""
    order = [task["task_id"] for task in tasks]
    if len(order) != 1_000 or len(set(order)) != len(order):
        raise ValueError("expected the sealed 1,000 unique R tasks")
    ended = defaultdict(int)
    seen = set()
    for verdict in verdicts:
        sample_id = verdict["sample_id"]
        if sample_id in seen or verdict["task_id"] not in order:
            raise ValueError(f"duplicate or off-panel extra attempt: {sample_id}")
        seen.add(sample_id)
        ended[verdict["task_id"]] += bool(verdict["turn_ended"])
    rows = []
    for task in tasks:
        task_id = task["task_id"]
        for index in range(start_index, start_index + max(0, ROWS_PER_TASK - ended[task_id])):
            probe_id = f"{task_id}#sample-{index}"
            if probe_id in seen:
                raise ValueError(f"extra attempt ID already exists: {probe_id}")
            rows.append(
                {"probe_id": probe_id, "source": task_id, "messages": [{"role": "user", "content": task["prompt"]}]}
            )
    return rows


def _parquet_rows(uri: str, expected_sha256: str) -> list[dict]:
    filesystem, path = filesystem_for(_regional(uri))
    with filesystem.open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{uri}: SHA256 {digest} differs from the sealed manifest")
    return pq.read_table(io.BytesIO(payload)).to_pylist()


def _blocks(rows: list[dict], order: list[str], source: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["data_source"] != source or tuple(row) != ROW_COLUMNS:
            raise ValueError(f"unexpected {source} row schema or source")
        grouped[row["task_id"]].append(row)
    if set(grouped) != set(order) or any(len(grouped[task_id]) != ROWS_PER_TASK for task_id in order):
        raise ValueError(f"expected four {source} rows for each selected task")
    return grouped


def _extra_blocks(candidates: list[dict], order: list[str]) -> tuple[dict[str, list[dict]], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    sample_ids = set()
    for row in candidates:
        if row["task_id"] not in order or row["data_source"] != "task_R" or row["teacher_route"] != "S38":
            raise ValueError("extra R candidate has an unexpected task or teacher")
        if row["sample_id"] in sample_ids:
            raise ValueError(f"duplicate extra R candidate: {row['sample_id']}")
        sample_ids.add(row["sample_id"])
        grouped[row["task_id"]].append(row)
    short = {
        task_id: ROWS_PER_TASK - len(grouped[task_id]) for task_id in order if len(grouped[task_id]) < ROWS_PER_TASK
    }
    if short:
        raise ValueError(f"extra R collection needs {sum(short.values())} top-ups across {len(short)} tasks")
    selected = {
        task_id: [
            {key: row[key] for key in ROW_COLUMNS}
            for row in sorted(grouped[task_id], key=lambda item: int(item["sample_id"].split("#sample-")[1]))[
                :ROWS_PER_TASK
            ]
        ]
        for task_id in order
    }
    selected_ids = [row["sample_id"] for task_id in order for row in grouped[task_id]]
    return selected, {
        "ended_candidates": len(candidates),
        "selected_rows": ROWS_PER_SOURCE,
        "candidate_sample_ids": len(set(selected_ids)),
    }


def _permuted_indices(seed: int) -> list[int]:
    indices = list(range(1_000))
    random.Random(seed).shuffle(indices)
    return indices


def seal(
    tasks: list[dict],
    j_tasks: list[dict],
    original_r: list[dict],
    original_j: list[dict],
    original_rj: list[dict],
    extra_candidates: list[dict],
    out: str,
) -> dict:
    """Write 64-distinct-R and paired seed permutations without changing token rows."""
    r_order = [row["task_id"] for row in tasks]
    j_order = [row["task_name"] for row in j_tasks]
    if len(r_order) != 1_000 or len(j_order) != 1_000:
        raise ValueError("expected 1,000 R and J task blocks")
    r = _blocks(original_r, r_order, "task_R")
    j = _blocks(original_j, j_order, "task_J")
    if len(original_rj) != 2 * ROWS_PER_SOURCE:
        raise ValueError("sealed joint data has the wrong row count")
    expected_joint = []
    for first in range(0, 1_000, TASKS_PER_STEP):
        for task_id in r_order[first : first + TASKS_PER_STEP]:
            expected_joint.extend(r[task_id])
        for task_id in j_order[first : first + TASKS_PER_STEP]:
            expected_joint.extend(j[task_id])
    if original_rj != expected_joint:
        raise ValueError("sealed joint rows do not equal their R and J source files")
    extra, extra_audit = _extra_blocks(extra_candidates, r_order)
    prime = []
    for task_id in r_order:
        prime.extend(r[task_id])
        prime.extend(extra[task_id])
    files = {"theta_R_prime/seed0/m1.parquet": write_parquet(f"{out}/theta_R_prime/seed0/m1.parquet", prime)}
    for seed in (1, 2):
        indices = _permuted_indices(seed)
        r_rows = [row for index in indices for row in r[r_order[index]]]
        joint_rows = []
        for first in range(0, 1_000, TASKS_PER_STEP):
            for index in indices[first : first + TASKS_PER_STEP]:
                joint_rows.extend(r[r_order[index]])
            for index in indices[first : first + TASKS_PER_STEP]:
                joint_rows.extend(j[j_order[index]])
        files[f"theta_R/seed{seed}/m1.parquet"] = write_parquet(f"{out}/theta_R/seed{seed}/m1.parquet", r_rows)
        files[f"theta_RJ/seed{seed}/m1.parquet"] = write_parquet(f"{out}/theta_RJ/seed{seed}/m1.parquet", joint_rows)
    manifest = {
        "original_sha256": {"R": ORIGINAL_R_SHA256, "J": ORIGINAL_J_SHA256, "RJ": ORIGINAL_RJ_SHA256},
        "permutation": "random.Random(seed).shuffle(range(1000)) at paired four-row task-block granularity",
        "extra": extra_audit,
        "files": files,
    }
    filesystem, path = filesystem_for(f"{out}/manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    request = sub.add_parser("requests")
    request.add_argument("--tasks", required=True)
    request.add_argument("--source-probes", required=True)
    request.add_argument("--start-index", type=int, default=12)
    request.add_argument("--out", required=True)
    topup = sub.add_parser("topups")
    topup.add_argument("--tasks", required=True)
    topup.add_argument("--grade-root", action="append", required=True)
    topup.add_argument("--start-index", type=int, required=True)
    topup.add_argument("--out", required=True)
    build = sub.add_parser("seal")
    build.add_argument("--r-tasks", required=True)
    build.add_argument("--j-tasks", required=True)
    build.add_argument("--run-data", required=True)
    build.add_argument("--extra-candidates", action="append", required=True)
    build.add_argument("--out", required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if isinstance(value, str) and value.startswith("s3://"):
            _regional(value)
        if isinstance(value, list):
            for item in value:
                _regional(item)
    configure_coreweave_s3()
    if args.command == "requests":
        rows = extra_requests(read_jsonl(args.tasks), read_jsonl(args.source_probes), args.start_index)
        write_jsonl(args.out, rows)
        print(json.dumps({"requests": len(rows), "out": args.out}))
        return
    if args.command == "topups":
        verdicts = [row for root in args.grade_root for row in read_jsonl(f"{root.rstrip('/')}/verdicts.jsonl")]
        rows = topup_requests(read_jsonl(args.tasks), verdicts, args.start_index)
        write_jsonl(args.out, rows)
        print(json.dumps({"requests": len(rows), "out": args.out}))
        return
    base = args.run_data.rstrip("/")
    candidates = [
        row for uri in args.extra_candidates for row in read_jsonl(f"{uri.rstrip('/')}/ended-candidates.jsonl")
    ]
    manifest = seal(
        read_jsonl(args.r_tasks),
        read_jsonl(args.j_tasks),
        _parquet_rows(f"{base}/theta_R/seed0/m1.parquet", ORIGINAL_R_SHA256),
        _parquet_rows(f"{base}/theta_J/seed0/m1.parquet", ORIGINAL_J_SHA256),
        _parquet_rows(f"{base}/theta_RJ/seed0/m1.parquet", ORIGINAL_RJ_SHA256),
        candidates,
        args.out.rstrip("/"),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
