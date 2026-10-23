#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote vetted GSM8K replay scores into a campaign tracker."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

BENCHMARK = "gsm8k-0shot"
SCORE_CELL = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+)\)$")


def update_tracker(tracker: str, results: list[dict]) -> str:
    """Replace GSM8K cells after verifying each source identity and score."""
    lines = tracker.splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    benchmark_index = header.index(BENCHMARK)
    by_model = {result["model"]: result for result in results}
    if len(by_model) != len(results):
        raise ValueError("regrade summary contains duplicate models")

    replaced = set()
    for index, line in enumerate(lines):
        if not line.startswith("| "):
            continue
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) != len(header) or row[0] not in by_model:
            continue
        result = by_model[row[0]]
        match = SCORE_CELL.fullmatch(row[benchmark_index])
        if match is None:
            raise ValueError(f"{row[0]}: GSM8K cell is no longer a sealed score")
        if match.group(2) != result["source"]:
            raise ValueError(f"{row[0]}: GSM8K source changed after replay")
        if abs(float(match.group(1)) - float(result["tracker_score"])) > 1e-12:
            raise ValueError(f"{row[0]}: GSM8K tracker score changed after replay")
        row[benchmark_index] = f"{float(result['corrected_score']):.3f} ({result['regrade_result']})"
        lines[index] = "| " + " | ".join(row) + " |"
        replaced.add(row[0])
    missing = set(by_model) - replaced
    if missing:
        raise ValueError("regraded models absent from tracker: " + ", ".join(sorted(missing)))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--score-changes", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.score_changes.read_text())
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        updated = update_tracker(args.tracker.read_text(), results)
        with tempfile.NamedTemporaryFile("w", dir=args.tracker.parent, delete=False) as temporary:
            temporary.write(updated)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, args.tracker)


if __name__ == "__main__":
    main()
