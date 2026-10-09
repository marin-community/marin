#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace tracker cells only after their fixed-grader replay has a sealed result."""

import argparse
import json
import re
from pathlib import Path

import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

SCORE_CELL = re.compile(r"([0-9.]+) \((s3://[^)]+)\)")
BENCHMARKS = ("math500", "aime24", "olympiadbench")
EXPECTED_TRIALS = {"math500": 500, "aime24": 300, "olympiadbench": 300}


def read_result(uri: str) -> dict | None:
    """Return a sealed replay result if its object exists."""
    filesystem, key = filesystem_for(uri)
    if not filesystem.exists(key):
        return None
    with filesystem.open(key, "rb") as source:
        return json.load(source)


def update_tracker(tracker: Path, sources: Path, output_prefix: str) -> list[tuple[str, str, str]]:
    """Apply verified replay links and scores while preserving every other cell."""
    manifest = yaml.safe_load(sources.read_text())["models"]
    by_model = {row["model"]: row["benchmarks"] for row in manifest}
    if len(manifest) != 21 or len(by_model) != 21:
        raise ValueError("expected 21 unique source models")
    original = tracker.read_text()
    lines = original.splitlines(keepends=True)
    headers = [part.strip() for part in lines[0].split("|")[1:-1]]
    columns = {benchmark: headers.index(benchmark) + 1 for benchmark in BENCHMARKS}
    seen = set()
    changed = []
    for line_index in range(3, len(lines)):
        line = lines[line_index]
        parts = line.rstrip("\n").split("|")
        model = parts[1].strip()
        if model not in by_model or model in seen:
            raise ValueError(f"tracker model absent or duplicated in frozen manifest: {model}")
        seen.add(model)
        for benchmark in BENCHMARKS:
            source = by_model[model][benchmark]
            uri = f"{output_prefix.rstrip('/')}/results/{benchmark}/{model.replace('/', '--')}.json"
            result = read_result(uri)
            if result is None:
                continue
            if result["model"] != model or result["benchmark"] != benchmark or result["source"] != source["source"]:
                raise ValueError(f"{model}/{benchmark}: result identity differs from frozen source")
            if result["num_trials"] != EXPECTED_TRIALS[benchmark]:
                raise ValueError(f"{model}/{benchmark}: incomplete trial count")
            if benchmark == "olympiadbench" and result["num_judge_failed"]:
                raise ValueError(f"{model}/{benchmark}: failed judge requests")
            score = float(result["corrected_score"])
            if not 0 <= score <= 1:
                raise ValueError(f"{model}/{benchmark}: score outside [0, 1]")
            column = columns[benchmark]
            cell = parts[column].strip()
            match = SCORE_CELL.fullmatch(cell)
            if match is None:
                raise ValueError(f"{model}/{benchmark}: current tracker cell is not a sealed score")
            current_score, current_uri = float(match.group(1)), match.group(2)
            if current_uri not in (source["source"], uri):
                raise ValueError(f"{model}/{benchmark}: tracker link changed since source manifest")
            if min(abs(current_score - float(source["tracker_score"])), abs(current_score - score)) > 0.0005:
                raise ValueError(f"{model}/{benchmark}: tracker score changed since source manifest")
            replacement = f"{score:.3f} ({uri})"
            if replacement != cell:
                parts[column] = f" {replacement} "
                changed.append((model, benchmark, replacement))
        lines[line_index] = "|".join(parts) + ("\n" if line.endswith("\n") else "")
    if seen != set(by_model):
        raise ValueError("tracker omits source-manifest models")
    updated = "".join(lines)
    if updated != original:
        tracker.write_text(updated)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    changed = update_tracker(args.tracker, args.sources, args.output_prefix)
    for model, benchmark, replacement in changed:
        print(f"{model} {benchmark}: {replacement}")
    print(f"updated {len(changed)} tracker cells")


if __name__ == "__main__":
    main()
