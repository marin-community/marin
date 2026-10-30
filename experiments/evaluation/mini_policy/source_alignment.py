# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check current canonical source identities against the frozen historical panel."""

import hashlib
import json
import os
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import SOURCE, read_bytes
from experiments.evaluation.mini_policy.identities import document_identity

EVALCHEMY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/source-index-20261030-v2"
HARBOR = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/harbor-source-index-20261030"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/source-alignment-20261030-v2"


def main():
    raw_matrix = read_bytes(SOURCE + "/matrix/tasks.json")
    matrix = json.loads(raw_matrix)
    exclusions = set(matrix["predictor_exclusions"])
    summaries = json.loads(read_bytes(EVALCHEMY + "/summary.json")) + json.loads(read_bytes(HARBOR + "/summary.json"))
    by_benchmark = {
        row["benchmark"]: {task["task_id"] for task in matrix["tasks"] if task["benchmark"] == row["benchmark"]}
        for row in summaries
    }
    report = []
    for root in [EVALCHEMY, HARBOR]:
        for summary in json.loads(read_bytes(root + "/summary.json")):
            benchmark = summary["benchmark"]
            if benchmark in exclusions:
                raise ValueError("Excluded benchmark in predictor source index")
            raw = read_bytes(root + "/" + benchmark + ".json")
            source = json.loads(raw)
            if root == EVALCHEMY:
                rows = [row for row in source["rows"] if benchmark != "cruxeval" or row["namespace"] == "output"]
            else:
                rows = [
                    {**row, "task_id": document_identity(benchmark, {}, row["task_name"])}
                    for dataset in source["resolved"]
                    for row in dataset["rows"]
                ]
            actual = {row["task_id"] for row in rows}
            expected = by_benchmark[benchmark]
            report.append(
                {
                    "benchmark": benchmark,
                    "current_tasks": len(actual),
                    "historical_tasks": len(expected),
                    "missing_from_current": sorted(expected - actual),
                    "new_in_current": sorted(actual - expected),
                    "source_index_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_index": root + "/" + benchmark + ".json",
                }
            )
    if len(report) != 21:
        raise ValueError(f"Expected 21 predictor source indices, got {len(report)}")
    document = {
        "matrix_sha256": hashlib.sha256(raw_matrix).hexdigest(),
        "benchmarks": report,
        "identity_adapter": (
            "DS-1000 registry basenames receive their declared ds-1000 dataset namespace; all other identities unchanged"
        ),
        "all_identities_match": all(not row["missing_from_current"] and not row["new_in_current"] for row in report),
    }
    payload = json.dumps(document, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "alignment.json").write_bytes(payload)
    fs, key = filesystem_for(DEST + "/alignment.json")
    fs.pipe_file(key, payload)
    print(
        json.dumps(
            {
                "all_identities_match": document["all_identities_match"],
                "benchmarks": [
                    {
                        "benchmark": r["benchmark"],
                        "missing": len(r["missing_from_current"]),
                        "new": len(r["new_in_current"]),
                    }
                    for r in report
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
