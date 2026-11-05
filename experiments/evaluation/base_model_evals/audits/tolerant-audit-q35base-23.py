#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tolerant audit of the .23 eval dumps.

The dump writer emits one JSON object per line, but rows containing a raw
newline inside a JSON string get split across two physical lines, which breaks
strict per-line parsing. This script joins broken lines until each buffer
parses, tracks repairs, and emits the same shape as audit_results.py.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

RESULT_URI = "s3://marin-rno2a/marin/checkpoints/mini840-e2e-qwen35-35b-a3b-base/2026.10.31.23"
DUMP_URL = RESULT_URI + "/exports/dumped_evals/eval_only"
OUTPUT = Path(
    "/Users/benfeuer/Documents/experiments/active/base-model-evals/artifacts/2026.10.31.150013/full-audits/qwen35-35b-a3b-base-full-2026.10.31.23-audit.json"
)
DOWNLOAD_DIR = Path("/tmp/q35base-23-dumps")


def list_files() -> list[str]:
    out = subprocess.run(
        ["uv", "run", "--frozen", "fsutil", "ls", DUMP_URL],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".jsonl") and not parts[-1].startswith("-"):
            names.append(parts[-1])
    return sorted(set(names))


def download(name: str) -> Path:
    path = DOWNLOAD_DIR / name
    if not path.exists():
        subprocess.run(
            ["uv", "run", "--frozen", "fsutil", "cp", f"{DUMP_URL}/{name}", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return path


def parse_tolerant(lines: list[str]) -> tuple[list[dict], int]:
    rows, repairs = [], 0
    buffer = ""
    for line in lines:
        buffer = buffer + line if buffer else line
        try:
            rows.append(json.loads(buffer))
            buffer = ""
        except json.JSONDecodeError:
            repairs += 1
            continue
    if buffer.strip():
        rows.append({"__unparsed_tail__": buffer[-400:]})
    return rows, repairs


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = Counter()
    completed = 0
    for name in list_files():
        if name == "aggregated_results.jsonl":
            continue
        text = download(name).read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        rows, repairs = parse_tolerant(lines)
        broken = [r for r in rows if "__unparsed_tail__" in r]
        rows = [r for r in rows if "__unparsed_tail__" not in r]
        for row in rows:
            status = row["verification"]["status"]
            if status != "verified":
                failures[row.get("exception_type") or status] += 1
        verified = [r for r in rows if r["verification"]["status"] == "verified"]
        summaries.append(
            {
                "file": name,
                "rollouts": len(rows),
                "verified": len(verified),
                "mean_verified_reward": (
                    sum(r["verification"]["score"] for r in verified) / len(verified) if verified else None
                ),
                "response_tokens": sum(len(r["response_token_ids"]) for r in rows),
                "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
                "line_repairs": repairs,
                "unparsed_tails": len(broken),
            }
        )
        completed += len(rows)
        flag = " (REPAIRED)" if repairs else ""
        mean = summaries[-1]["mean_verified_reward"]
        mean_s = "n/a" if mean is None else f"{mean:.3f}"
        print(
            f"{name:42s} rows={len(rows):4d} verified={len(verified):4d} mean={mean_s} repairs={repairs} tails={len(broken)}{flag}",
            flush=True,
        )
    report = {
        "result_uri": RESULT_URI,
        "expected_rollouts": 1626,
        "completed_rollouts": completed,
        "all_verified": (
            completed == 1626 and not failures and not any(s["line_repairs"] or s["unparsed_tails"] for s in summaries)
        ),
        "unverified_types": dict(failures),
        "parser": "tolerant line-join (raw-newline dump artifact)",
        "benchmarks": summaries,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\ncompleted={completed} failures={dict(failures)} -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
