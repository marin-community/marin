# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a durable conversion run without conflating content holds and request failures."""

import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path


def summarize(root: Path) -> dict:
    receipts = [(path, json.loads(path.read_text())) for path in root.glob("records/*/*/receipt.json")]
    if not receipts:
        raise ValueError("No model receipts to measure")
    with sqlite3.connect(root / "queue.sqlite3") as db:
        counts = dict(db.execute("SELECT status,count(*) FROM rows GROUP BY status"))
    elapsed = max(r["completed_at_unix"] for _, r in receipts) - min(r["started_at_unix"] for _, r in receipts)
    timing = json.loads((root / "timing.json").read_text())
    run_elapsed = timing["completed_at_unix"] - timing["started_at_unix"]
    durations = sorted(r["completed_at_unix"] - r["started_at_unix"] for _, r in receipts)
    failures = sum(
        not (
            r["finish_reason"] == "stop"
            and r["done"]
            and r["lease_state"] == "terminal"
            and (path.parent / "value.json").exists()
        )
        for path, r in receipts
    )
    uncertain = sum(r["lease_state"] != "terminal" for _, r in receipts)
    prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) for _, r in receipts)
    output = sum((r.get("usage") or {}).get("completion_tokens", 0) for _, r in receipts)
    cached = sum(
        ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) for _, r in receipts
    )
    stages = {}
    for stage in ("review", "audit"):
        times = sorted(
            r["completed_at_unix"] - r["started_at_unix"] for path, r in receipts if path.parent.name == stage
        )
        if times:
            stages[stage] = {
                "requests": len(times),
                "median_seconds": statistics.median(times),
                "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
            }
    p95 = durations[math.ceil(0.95 * len(durations)) - 1]
    completed = sum(count for status, count in counts.items() if status not in ("pending", "running"))
    return {
        "row_counts": counts,
        "requests": len(receipts),
        "failed_requests": failures,
        "uncertain_requests": uncertain,
        "elapsed_request_window_seconds": elapsed,
        "elapsed_run_seconds": run_elapsed,
        "accepted_per_hour": counts.get("converted_adapter_checked", 0) * 3600 / run_elapsed,
        "completed_records_per_hour": completed * 3600 / run_elapsed,
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "prompt_tokens_per_second": prompt / run_elapsed,
        "completion_tokens_per_second": output / run_elapsed,
        "cached_prompt_fraction": cached / prompt if prompt else 0,
        "median_request_seconds": statistics.median(durations),
        "p95_request_seconds": p95,
        "stages": stages,
        "artifact_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "minimum_duration_and_rows_met": elapsed >= 120 and completed >= 96,
        "escalation_health_pass": uncertain == 0 and failures / len(receipts) <= 0.05 and p95 <= 120,
        "scope": "Supported two-source conversion cohort; content holds are separate from model request failures.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / "throughput.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
