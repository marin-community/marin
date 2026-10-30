# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay frozen GSM8K responses under the target and prior vendored graders."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import time
from itertools import repeat
from pathlib import Path

import numpy as np
from eval.graders import minerva_math
from eval.lm_eval_tasks.gsm8k import utils as target
from finestore.reader import ReadView
from marin.evaluation.metric_selection import primary_filter
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.assemble import original_record, read
from experiments.evaluation.mini_policy.grader_reference import gsm8k_before as before
from experiments.evaluation.mini_policy.identities import document_identity

DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/gsm8k-grader-parity-20261030"


def save(name, document):
    payload = json.dumps(document, indent=2, allow_nan=False).encode()
    fs, key = filesystem_for(DEST + "/" + name)
    fs.pipe_file(key, payload)


def replay(cell, source):
    metadata = read(source + "/matrix/tasks.json")
    fs, key = filesystem_for(source + "/matrix/responses.npz")
    arrays = np.load(io.BytesIO(fs.cat(key)))
    tasks = {task["task_id"]: j for j, task in enumerate(metadata["tasks"]) if task["benchmark"] == "gsm8k-0shot"}
    model = metadata["models"].index(cell["model"])
    record = original_record(cell, read(cell["evidence_url"]))
    if record is None:
        raise ValueError("Original GSM8K record missing")
    view = ReadView(record["results_path"])
    rows = view.scan("samples", columns=["doc_id", "doc", "output", "filter"]).to_pylist()
    selected_filter = primary_filter([row["filter"] for row in rows])
    rows = [row for row in rows if row["filter"] == selected_filter]
    if len(rows) != 1319:
        raise ValueError(f"Expected 1319 raw GSM8K responses, found {len(rows)}")
    samples = []
    for row in rows:
        doc = json.loads(row["doc"])
        identity = document_identity("gsm8k-0shot", doc, row["doc_id"])
        samples.append(
            {
                "doc_id": row["doc_id"],
                "task_id": identity,
                "doc": doc,
                "response": row["output"] or "",
                "expected": float(arrays["responses"][model, tasks[identity]]),
            }
        )
    if {row["task_id"] for row in samples} != set(tasks):
        raise ValueError("Raw GSM8K identities differ from frozen panel")
    grades = {}
    timing = {}
    # Balance order across models and clear the shared symbolic parse cache.
    order = [("target", target), ("before", before)]
    if int(hashlib.sha256(cell["model"].encode()).hexdigest()[:2], 16) % 2:
        order.reverse()
    for name, grader in order:
        minerva_math._sympy_parses.cache_clear()
        started, cpu = time.monotonic(), time.process_time()
        values = []
        for row in samples:
            extraction = grader.extract_gsm8k_flexible_answer(row["response"])
            score = grader.process_results(row["doc"], [extraction])["exact_match"]
            values.append((extraction, score))
        timing[name] = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started}
        grades[name] = values
    trials = []
    for row, corrected, prior in zip(samples, grades["target"], grades["before"], strict=True):
        trials.append(
            {
                "doc_id": row["doc_id"],
                "task_id": row["task_id"],
                "expected": row["expected"],
                "replayed": corrected[1],
                "extracted": corrected[0],
                "prior_replayed": prior[1],
                "prior_extracted": prior[0],
                "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
            }
        )
    result = {
        "model": cell["model"],
        "source": record["results_path"],
        "sample_metadata": view.table_metadata_path("samples"),
        "filter": selected_filter,
        "trials": trials,
        "target_mismatches": sum(row["expected"] != row["replayed"] for row in trials),
        "prior_mismatches": sum(row["expected"] != row["prior_replayed"] for row in trials),
        "target_mean": float(np.mean([row["replayed"] for row in trials])),
        "expected_mean": float(np.mean([row["expected"] for row in trials])),
        "timing": timing,
        "target_sha256": hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
        "before_sha256": hashlib.sha256(Path(before.__file__).read_bytes()).hexdigest(),
    }
    save(hashlib.sha256(cell["model"].encode()).hexdigest()[:16] + ".json", result)
    small = {key: value for key, value in result.items() if key != "trials"}
    print(json.dumps(small), flush=True)
    return small


def main():
    global DEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    DEST = args.destination
    cells = [row for row in read(args.matrix_source + "/assembly_audit.json") if row["benchmark"] == "gsm8k-0shot"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(replay, cells, repeat(args.matrix_source)))
    summary = {
        "models": len(results),
        "raw_trials": len(results) * 1319,
        "target_mismatches": sum(row["target_mismatches"] for row in results),
        "prior_mismatches": sum(row["prior_mismatches"] for row in results),
        "results": results,
        "timing_scope": (
            "Per-model grading loops only; CPU time excludes archive I/O. Arm order is balanced and symbolic "
            "parse caches are cleared before each arm."
        ),
    }
    save("summary.json", summary)
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["target_mismatches"]:
        raise ValueError("Adopted GSM8K grader does not reproduce every frozen grade")
    print("GSM8K_PARITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
