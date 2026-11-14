# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine disjoint conversion shards into a wall-clock serving measurement."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from experiments.tool_normalization.throughput import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [summarize(root) for root in args.run_root]
    timings = [json.loads((root / "timing.json").read_text()) for root in args.run_root]
    elapsed = max(t["completed_at_unix"] for t in timings) - min(t["started_at_unix"] for t in timings)
    identities = [
        json.loads(line)["id"] for root in args.run_root for line in (root / "sources.jsonl").read_text().splitlines()
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Replica shards contain duplicate source identities")
    counts = Counter()
    for report in reports:
        counts.update(report["row_counts"])
    receipts = [json.loads(path.read_text()) for root in args.run_root for path in root.glob("records/*/*/receipt.json")]
    durations = sorted(r["completed_at_unix"] - r["started_at_unix"] for r in receipts)
    completed = sum(n for status, n in counts.items() if status not in {"pending", "running"})
    accepted = counts["converted_adapter_checked"] + counts["header_normalized_no_calls"]
    requests = sum(r["requests"] for r in reports)
    failures = sum(r["failed_requests"] for r in reports)
    uncertain = sum(r["uncertain_requests"] for r in reports)
    report = {
        "run_roots": [str(root.resolve()) for root in args.run_root],
        "replicas": len(args.run_root),
        "workers": sum(json.loads((root / "config.json").read_text())["workers"] for root in args.run_root),
        "source_ids": sorted(identities),
        "row_counts": dict(counts),
        "elapsed_seconds": elapsed,
        "completed_records": completed,
        "accepted_records": accepted,
        "completed_records_per_hour": completed * 3600 / elapsed,
        "accepted_records_per_hour": accepted * 3600 / elapsed,
        "requests": requests,
        "failed_requests": failures,
        "uncertain_requests": uncertain,
        "p50_request_seconds": durations[math.ceil(0.5 * len(durations)) - 1],
        "p95_request_seconds": durations[math.ceil(0.95 * len(durations)) - 1],
        "prompt_tokens_per_second": sum(r["prompt_tokens"] for r in reports) / elapsed,
        "completion_tokens_per_second": sum(r["completion_tokens"] for r in reports) / elapsed,
        "minimum_duration_and_rows_met": elapsed >= 120 and completed >= 96,
        "request_reliability_pass": uncertain == 0 and failures / requests <= 0.05,
        "all_rows_terminal": counts["pending"] == 0 and counts["running"] == 0,
        "individual_reports": reports,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"source_ids", "individual_reports"}}))


if __name__ == "__main__":
    main()
