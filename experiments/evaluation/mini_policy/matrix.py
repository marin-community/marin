# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export aligned task rewards and audit identity overlap before model fitting."""

import concurrent.futures
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

PREFIX = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030"


def read(url):
    fs, path = filesystem_for(url)
    return json.loads(fs.cat(path))


def main():
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    manifest = json.loads(Path("experiments/evaluation/mini_policy/cohort.json").read_text())
    models = manifest["models"]
    model_index = {model: i for i, model in enumerate(models)}
    audit = read(PREFIX + "/assembly_audit.json")
    cells = [row for row in audit if row.get("trials_url")]
    benchmarks = sorted({row["benchmark"] for row in cells})
    grouped = defaultdict(dict)
    targets = np.full((len(models), len(benchmarks)), np.nan)
    sems = targets.copy()

    def load(cell):
        data = read(cell["trials_url"])
        return cell, data

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for cell, data in pool.map(load, cells):
            i = model_index[cell["model"]]
            b = cell["benchmark"]
            j = benchmarks.index(b)
            rewards = [row["reward"] for row in data["trials"] if row["scored"]]
            targets[i, j] = np.mean(rewards)
            sems[i, j] = np.std(rewards, ddof=1) / np.sqrt(len(rewards))
            per_task = defaultdict(list)
            for row in data["trials"]:
                per_task[row["task_id"]].append(row)
            for task, rows in per_task.items():
                grouped[(b, task)][i] = rows
    task_keys = sorted(grouped)
    responses = np.full((len(models), len(task_keys)), np.nan)
    counts = np.zeros_like(responses, dtype=int)
    attempted = counts.copy()
    tasks = []
    for j, (benchmark, task) in enumerate(task_keys):
        source_ids = {}
        for i, rows in grouped[(benchmark, task)].items():
            rewards = [row["reward"] for row in rows if row["scored"]]
            attempted[i, j] = len(rows)
            counts[i, j] = len(rewards)
            if rewards:
                responses[i, j] = np.mean(rewards)
            source_ids[models[i]] = list(dict.fromkeys(str(row.get("source_doc_id", task)) for row in rows))
        strata = next(
            (row.get("strata", {}) for rows in grouped[(benchmark, task)].values() for row in rows if row.get("strata")),
            {},
        )
        tasks.append({"benchmark": benchmark, "task_id": task, "source_ids": source_ids, "strata": strata})
    overlap = []
    for benchmark in benchmarks:
        idx = [j for j, t in enumerate(tasks) if t["benchmark"] == benchmark]
        complete = np.isfinite(responses[:, idx]).all(axis=0)
        overlap.append(
            {
                "benchmark": benchmark,
                "union_tasks": len(idx),
                "complete_tasks": int(complete.sum()),
                "per_model_tasks": np.isfinite(responses[:, idx]).sum(axis=1).tolist(),
            }
        )
    rejected = [
        {
            "model": r["model"],
            "benchmark": r["benchmark"],
            "status": r["status"],
            "error": r.get("error"),
            "delta": r.get("score_delta"),
        }
        for r in audit
        if r["status"] not in {"reconciled", "reconciled_source_update", "reconciled_native_evidence"}
    ]
    metadata = {
        "models": models,
        "benchmarks": benchmarks,
        "tasks": tasks,
        "predictor_exclusions": manifest["predictor_exclusions"],
        "overlap": overlap,
        "unresolved_cells": rejected,
        "assembly_audit_sha256": hashlib.sha256(json.dumps(audit, sort_keys=True, default=str).encode()).hexdigest(),
    }
    (output / "matrix_audit.json").write_text(json.dumps({"overlap": overlap, "unresolved_cells": rejected}, indent=2))
    np.savez_compressed(
        output / "responses.npz", responses=responses, counts=counts, attempted=attempted, targets=targets, sems=sems
    )
    (output / "tasks.json").write_text(json.dumps(metadata))
    fs, path = filesystem_for(PREFIX + "/matrix/responses.npz")
    fs.pipe_file(path, (output / "responses.npz").read_bytes())
    fs, path = filesystem_for(PREFIX + "/matrix/tasks.json")
    fs.pipe_file(path, (output / "tasks.json").read_bytes())
    print(
        json.dumps({"shape": responses.shape, "unresolved_cells": len(rejected), "overlap": overlap}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
