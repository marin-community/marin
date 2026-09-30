"""Summarize recorded evidence without turning incomplete work into success."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .inference import atomic_json


def summarize(root):
    root = Path(root)
    stages = defaultdict(
        lambda: {
            "states": Counter(),
            "responses": 0,
            "validated_results": 0,
            "structurally_repaired_results": 0,
            "error_types": Counter(),
            "finish_reasons": Counter(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "ttft_seconds": [],
            "duration_seconds": [],
        }
    )
    intervals = []
    errors = []
    for path in sorted((root / "items").glob("*/*/status.json")):
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            errors.append({"path": str(path), "error": type(e).__name__})
            continue
        stage = stages[path.parent.parent.name]
        stage["states"][status["state"]] += 1
        if status.get("error_type"):
            stage["error_types"][status["error_type"]] += 1
        result_path = path.with_name("result.json")
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                stage["validated_results"] += 1
                stage["structurally_repaired_results"] += int(
                    result.get("structural_repair_used") is True
                )
            except (OSError, ValueError) as e:
                errors.append({"path": str(result_path), "error": type(e).__name__})
        response_path = path.with_name("response.json")
        if not response_path.exists():
            continue
        try:
            response = json.loads(response_path.read_text())
        except (OSError, ValueError) as e:
            errors.append({"path": str(response_path), "error": type(e).__name__})
            continue
        stage["responses"] += 1
        stage["finish_reasons"][response.get("finish_reason", "missing")] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            stage[key] += response.get("usage", {}).get(key, 0)
        for key, destination in (
            ("ttft_seconds", "ttft_seconds"),
            ("elapsed_seconds", "duration_seconds"),
        ):
            if response.get(key) is not None:
                stage[destination].append(response[key])
        if (
            response.get("started_unix") is not None
            and response.get("elapsed_seconds") is not None
        ):
            intervals.extend(
                [
                    (response["started_unix"], 1),
                    (response["started_unix"] + response["elapsed_seconds"], -1),
                ]
            )
    peak, running = 0, 0
    for _, delta in sorted(intervals):
        running += delta
        peak = max(peak, running)
    for stage in stages.values():
        stage["states"] = dict(stage["states"])
        stage["error_types"] = dict(stage["error_types"])
        stage["finish_reasons"] = dict(stage["finish_reasons"])
        stage["valid_without_structural_repair"] = (
            stage["validated_results"] - stage["structurally_repaired_results"]
        )
        for key in ("ttft_seconds", "duration_seconds"):
            values = sorted(stage.pop(key))
            stage[key + "_median"] = statistics.median(values) if values else None
            stage[key + "_p90"] = (
                values[min(len(values) - 1, int(len(values) * 0.9))] if values else None
            )
    report_file = root / "report.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else None
    # Even a completed proposal report is not evidence of task/runtime completion.
    return {
        "stages": dict(stages),
        "evidence_scope": "files present in this local snapshot; incomplete downloads undercount stages and token usage",
        "read_errors": errors,
        "proposal_report": report,
        "peak_concurrent_completed_requests_lower_bound": peak,
        "concurrency_scope": "completed successful transports only; excludes interrupted/live requests and other tenants",
        "runtime_validated_tasks": report.get("runtime_validated_tasks", 0)
        if report
        else 0,
        "pipeline_complete": False,
        "completion_note": "This report audits proposal evidence only; synthesis/runtime/quality gates are separate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    summary = summarize(args.run)
    if args.output:
        atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
