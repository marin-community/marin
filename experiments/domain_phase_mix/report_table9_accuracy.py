# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report all 51 Table-9 components, including missing and deferred accuracy tasks.

Reads completed artifacts only; never runs inference or executes generated code.
Use the same uv math extras as grade_table9_accuracy to verify grader identity.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import fsspec
from marin.evaluation.olmo_base_eval.accuracy import (
    CHOICE_BACKFILL_TASKS,
    GENERATION_BACKFILL_TASKS,
    coverage_report,
    validate_task_samples,
)
from marin.evaluation.olmo_base_eval.components import table9_components

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference
from experiments.domain_phase_mix.grade_table9_accuracy import PRIMARY_METRICS, graded_result

NORMALIZED_OVERLAP = {"arc_easy", "arc_challenge", "hellaswag", "piqa"}


def overlap_scores(plan: dict, row: dict) -> dict:
    """Verify small summaries and their completion receipt without downloading raw samples again."""
    root = inference.existing.output_uri(plan, row)
    fs, _ = fsspec.core.url_to_fs(root)
    if not fs.exists(root + "/SUCCESS.json"):
        return {}
    marker = inference.existing.read_json(root + "/SUCCESS.json")
    if marker["plan_sha256"] != inference.existing.plan_sha256(plan) or marker["row"] != row:
        raise ValueError("Overlap completion belongs to another checkpoint or protocol")
    if set(marker["artifacts"]) != set(inference.existing.ARTIFACTS):
        raise ValueError("Overlap completion omits retained samples or provenance")
    reports = {}
    for name in ("summary_metrics.json", "provenance.json"):
        data = inference.existing.read_bytes(root + "/" + name)
        if marker["artifacts"][name] != {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
            raise ValueError(f"Overlap artifact changed: {name}")
        reports[name] = json.loads(data)
    provenance, summary = reports["provenance.json"], reports["summary_metrics.json"]
    if provenance["row"] != row or provenance["plan"] != plan or provenance["plan_sha256"] != marker["plan_sha256"]:
        raise ValueError("Overlap provenance differs")
    counts = provenance["expected_task_samples"]
    if set(counts) != inference.existing.EXPECTED_TASKS:
        raise ValueError("Overlap report omits expected task leaves")
    out = {}
    for task, count in counts.items():
        if count <= 0 or summary["n-samples"][task] != {"original": count, "effective": count}:
            raise ValueError(f"Partial overlap evaluation: {task}")
        leaf = task.removesuffix("_5shot").removesuffix("_0shot")
        metric = "acc_norm,none" if leaf in NORMALIZED_OVERLAP else "acc,none"
        out[leaf] = {
            "primary_metric": metric,
            "metrics": summary["results"][task],
            "count": count,
            "source": root,
            "protocol": "lm-eval overlap; not native Table-9 prompt parity",
        }
    return out


def checkpoint_report(plan: dict, overlap_plan: dict, row: dict) -> dict:
    matching = [r for r in overlap_plan["rows"] if r["name"] == row["name"]]
    if len(matching) != 1 or matching[0] != row:
        raise ValueError("Backfill and overlap must refer to the identical checkpoint inventory row")
    tasks = overlap_scores(overlap_plan, row)
    stages = {}
    for name in (*CHOICE_BACKFILL_TASKS, *GENERATION_BACKFILL_TASKS):
        marker = inference.completed_task(plan, row, name, 0)
        stages[name] = "missing" if marker is None else marker["stage"]
        if marker is None:
            continue
        root = inference.result_root(plan, row, name, 0)
        if name in GENERATION_BACKFILL_TASKS:
            result = graded_result(plan, row, name, 0)
            if result is None:
                continue
            tasks[name] = result | {"primary_metric": PRIMARY_METRICS[name], "source": result["grading_root"]}
            stages[name] = "scored"
            continue
        samples = json.loads(gzip.decompress(inference.existing.read_bytes(root + "/samples.json.gz")))
        metrics = {
            key: validate_task_samples(name, samples, list(range(len(samples))), key) for key in samples[0]["metrics"]
        }
        tasks[name] = {
            "primary_metric": plan["request_manifest"]["tasks"][name]["metric"],
            "metrics": metrics,
            "count": len(samples),
            "source": root,
        }
    scores = {name: t["metrics"][t["primary_metric"]] for name, t in tasks.items()}
    return {
        "checkpoint": row,
        "coverage": coverage_report(scores),
        "tasks": tasks,
        "backfill_stages": stages,
        "backfill_protocol_sha256": inference.digest(inference.protocol(plan)),
        "overlap_plan_sha256": inference.existing.plan_sha256(overlap_plan),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--overlap-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, overlap = json.loads(args.plan.read_text()), json.loads(args.overlap_plan.read_text())
    reports = [checkpoint_report(plan, overlap, row) for row in plan["rows"]]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "coverage.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    with (args.output / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "component", "status", "accuracy"])
        writer.writeheader()
        for report in reports:
            coverage = report["coverage"]
            for name in table9_components():
                status = (
                    "scored"
                    if name in coverage["components"]
                    else ("deferred" if name in coverage["deferred"] else "missing")
                )
                writer.writerow(
                    {
                        "checkpoint": report["checkpoint"]["name"],
                        "component": name,
                        "status": status,
                        "accuracy": coverage["components"].get(name),
                    }
                )
            print(json.dumps({"checkpoint": report["checkpoint"]["name"], "coverage": coverage}), flush=True)


if __name__ == "__main__":
    main()
