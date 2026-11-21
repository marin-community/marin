#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze sealed tracker cells as an immutable grader-replay source manifest."""

import argparse
import re
from pathlib import Path

import yaml

SCORE_CELL = re.compile(r"([0-9.]+) \((s3://[^)]+)\)")


def frozen_sources(tracker: Path, benchmarks: tuple[str, ...]) -> dict:
    """Return sealed source identities for every tracker model and benchmark."""
    lines = tracker.read_text().splitlines()
    headers = [part.strip() for part in lines[0].split("|")[1:-1]]
    columns = {benchmark: headers.index(benchmark) + 1 for benchmark in benchmarks}
    models = []
    for line in lines[3:]:
        parts = line.split("|")
        if len(parts) != len(headers) + 2:
            raise ValueError("malformed tracker row")
        model = parts[1].strip()
        frozen = {}
        for benchmark, column in columns.items():
            match = SCORE_CELL.fullmatch(parts[column].strip())
            if match is None:
                raise ValueError(f"{model}/{benchmark}: tracker cell is not sealed")
            frozen[benchmark] = {"tracker_score": float(match.group(1)), "source": match.group(2)}
        models.append({"model": model, "benchmarks": frozen})
    if not models or len({row["model"] for row in models}) != len(models):
        raise ValueError("expected nonempty, unique tracker models")
    return {"models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    args = parser.parse_args()
    payload = frozen_sources(args.tracker, tuple(args.benchmark))
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
