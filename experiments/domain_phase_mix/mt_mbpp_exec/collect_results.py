# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect the MT-MBPP generations and grades into ``results/`` for the dataset release.

For each mixture and language, reads the chosen plan's generation artifact (the plan ``summarize_results`` used) and
joins every completion with its grade: ``results/generations/<mixture>/<language>.jsonl`` holds the task id, the
completion cut at its closing fence, whether its test was valid, and whether it passed. ``summary.json`` and
``components.csv`` from ``summarize_results`` are copied beside them.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.collect_results --project DIR
"""

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    results = args.project / "results"
    out = results / "release"
    out.mkdir(parents=True, exist_ok=True)
    plans = {p.stem: json.loads(p.read_text()) for p in args.project.glob("plan_*.json")}
    for row in csv.DictReader((results / "components.csv").open()):
        plan = plans[row["plan"]]
        (checkpoint,) = [r for r in plan["rows"] if r["name"] == row["checkpoint"]]
        task = f"mt_mbpp_{row['language']}"
        samples = json.loads(
            gzip.decompress(
                inference.existing.read_bytes(inference.result_root(plan, checkpoint, task, 0) + "/samples.json.gz")
            )
        )
        grades_path = args.project / "grading" / row["plan"] / row["checkpoint"] / f"{task}.jsonl.gz"
        with gzip.open(grades_path, "rt") as handle:
            grades = {g["task_id"]: g for g in map(json.loads, handle)}
        target = out / "generations" / row["mixture"] / f"{row['language']}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as handle:
            for s in samples:
                tid = s["metadata"]["id"]
                grade = grades.get(tid)
                handle.write(
                    json.dumps(
                        {
                            "mixture": row["mixture"],
                            "checkpoint": row["checkpoint"],
                            "language": row["language"],
                            "task_id": tid,
                            "doc_id": s["doc_id"],
                            "completion": s["generation"].split("```")[0],
                            "test_valid": grade is not None,
                            "passed": bool(grade and grade["passed"]),
                        }
                    )
                    + "\n"
                )
    for name in ("summary.json", "components.csv"):
        shutil.copy(results / name, out / name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
