# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-evaluate archived predictions using order-invariant half-point aggregation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import evaluate, read_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    settings = json.loads(read_bytes(args.study + "/settings.json"))
    evaluated = np.asarray([settings["models"].index(model) for model in settings["evaluated_models"]])
    original = json.loads(read_bytes(args.study + "/summary.json"))
    results = []
    for row in original["results"]:
        source = args.study + f"/oof-{row['method']}-{row['seed']}-{row['budget']}.json"
        data = json.loads(read_bytes(source))
        before = row["metrics"]
        after = evaluate(
            np.asarray(data["predicted"])[evaluated],
            np.asarray(data["actual"])[evaluated],
            np.asarray(data["predicted_sem"])[evaluated],
            np.asarray(data["actual_sem"])[evaluated],
        )
        results.append(
            {
                **row,
                "metrics": after,
                "prior_spearman": before["tournament_spearman"],
                "maximum_score_roundoff": float(
                    np.max(
                        np.abs(
                            np.asarray(before["predicted_tournament_scores"])
                            - np.asarray(after["predicted_tournament_scores"])
                        )
                    )
                ),
            }
        )
    document = {
        "study": args.study,
        "definition": (
            "Sum integer half-points before a single common-denominator division; no fitting, probe, grading, or "
            "benchmark comparison changes."
        ),
        "evaluator_sha256": (
            hashlib.sha256(Path("experiments/evaluation/mini_policy/experiment.py").read_bytes()).hexdigest()
        ),
        "results": results,
    }
    payload = json.dumps(document, indent=2).encode()
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "summary.json"
    output.write_bytes(payload)
    fs, key = filesystem_for(args.destination + "/summary.json")
    fs.pipe_file(key, payload)
    print(
        json.dumps(
            [
                {
                    "method": row["method"],
                    "budget": row["budget"],
                    "before": row["prior_spearman"],
                    "after": row["metrics"]["tournament_spearman"],
                    "maximum_score_roundoff": row["maximum_score_roundoff"],
                }
                for row in results
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
