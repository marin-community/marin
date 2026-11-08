# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export bounded native RL telemetry with UTC windows and raw provenance."""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def collect(deployment: str, job_prefix: str, start_ms: int, end_ms: int, output: Path) -> None:
    if not job_prefix.startswith("/") or any(c in job_prefix for c in "'%_\\"):
        raise ValueError("Use an exact Iris root prefix without SQL wildcard characters")
    if not 0 <= start_ms < end_ms:
        raise ValueError("Specify a bounded UTC time window")
    query = f"""SELECT timestamp_ms, job_id, execution_uid, process_index, name,
        body_json, attributes_json, resource_attributes_json, "cluster"
        FROM "telemetry_v1.marinskyrl"
        WHERE timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
          AND job_id LIKE '{job_prefix}%'
          AND kind = 'event'
        ORDER BY timestamp_ms LIMIT 200001"""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "finelog", "query", deployment],
        input=query,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) > 200000:
        raise ValueError("Telemetry window exceeds export limit; split the window")
    output.mkdir(parents=True, exist_ok=True)
    (output / "query.sql").write_text(query)
    (output / "native-raw.jsonl").write_text(result.stdout)
    spans = []
    for row in rows:
        body = json.loads(row["body_json"] or "{}")
        attrs = json.loads(row["attributes_json"] or "{}")
        if "started_unix_ms" not in body or "finished_unix_ms" not in body:
            continue
        spans.append(
            {
                "job": row["job_id"],
                "execution": row["execution_uid"],
                "process": row["process_index"],
                "cluster": row["cluster"],
                "event": row["name"],
                "phase": body.get("phase", attrs.get("phase", row["name"])),
                "start_utc": datetime.fromtimestamp(body["started_unix_ms"] / 1000, UTC).isoformat(),
                "end_utc": datetime.fromtimestamp(body["finished_unix_ms"] / 1000, UTC).isoformat(),
                "duration_seconds": body.get(
                    "duration_seconds", (body["finished_unix_ms"] - body["started_unix_ms"]) / 1000
                ),
                "attributes": attrs,
            }
        )
    (output / "native-spans.jsonl").write_text("".join(json.dumps(span) + "\n" for span in spans))
    grouped = defaultdict(list)
    for span in spans:
        grouped[span["phase"]].append(span["duration_seconds"])
    lines = [
        "# Native timing observations",
        "",
        f"Deployment: {deployment}. Root: `{job_prefix}`.",
        "",
        "Durations may overlap across workers and nested phases; their sum is not wall time.",
        "Missing phases are unmeasured, not zero. Queue/image/compile and Harbor attempt timestamps "
        "require their separate evidence.",
        "",
        "| Phase | Windows | Summed observed seconds |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {phase} | {len(values)} | {sum(values):.3f} |" for phase, values in sorted(grouped.items()))
    (output / "native-timing.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--job-prefix", required=True)
    parser.add_argument("--start-ms", required=True, type=int)
    parser.add_argument("--end-ms", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.deployment, args.job_prefix, args.start_ms, args.end_ms, args.output)
