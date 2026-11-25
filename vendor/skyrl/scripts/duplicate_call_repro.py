# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce #7743's median exact-duplicate call fractions on its published traces.

The audit's ``dup_tool_call_frac`` metric must reproduce the published medians before S0.
The traces parse `<tool_call>` blocks with JSON ``name``/``arguments``; a duplicate is a
(name, canonical arguments) pair already seen in the same trial. Run:

    uv run --project vendor/skyrl --with torch python vendor/skyrl/scripts/duplicate_call_repro.py \\
        --input <parquet files or directory> --output <json path>
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rolloutengine.repetition import canonical_arguments

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
# The published suite labels, keyed by the traces' run IDs.
EXPECTED_MEDIANS = {
    "ID (dev_set_v2)": 0.339,
    "SWE-bench Verified": 0.203,
    "Terminal-Bench 2": 0.369,
}
RUN_SUITES = {
    "34f2d275-74f7-4677-8c66-05c4a673b742": "ID (dev_set_v2)",
    "1f56d588-e1ec-43b3-a998-2e8f565ce622": "SWE-bench Verified",
    "3919a7b2-342c-43ad-ac96-7781e21c6ce0": "Terminal-Bench 2",
}


def parse_tool_calls(content: str) -> list[tuple[str, str]]:
    """The (name, canonical arguments) pairs of every `<tool_call>` block in one message."""
    calls = []
    for block in TOOL_CALL_BLOCK.findall(content or ""):
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        name = parsed.get("name") or parsed.get("function") or ""
        calls.append((str(name), canonical_arguments(parsed.get("arguments"))))
    return calls


def trial_duplicate_fraction(messages: Sequence[Mapping[str, Any]]) -> float | None:
    """The share of calls whose (name, canonical arguments) pair appeared earlier in the trial."""
    seen: set[tuple[str, str]] = set()
    total = 0
    duplicates = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in parse_tool_calls(message.get("content") or ""):
            total += 1
            if call in seen:
                duplicates += 1
            seen.add(call)
    if not total:
        return None
    return duplicates / total


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = []
    for path in paths:
        table = pq.read_table(path, columns=["conversations", "run_id"])
        for conversation, run_id in zip(
            table.column("conversations").to_pylist(), table.column("run_id").to_pylist(), strict=True
        ):
            rows.append({"conversations": conversation, "run_id": run_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="Parquet files or a directory of them")
    parser.add_argument("--output", required=True, help="JSON result path")
    args = parser.parse_args()
    paths = [Path(item) for item in args.input]
    expanded = [path for item in paths for path in (sorted(item.glob("*.parquet")) if item.is_dir() else [item])]
    rows = read_rows(expanded)
    by_suite: dict[str, list[float]] = {}
    calls_by_suite: dict[str, list[int]] = {}
    skipped = 0
    for row in rows:
        suite = RUN_SUITES.get(row["run_id"])
        if suite is None:
            skipped += 1
            continue
        fraction = trial_duplicate_fraction(row["conversations"])
        if fraction is None:
            skipped += 1
            continue
        by_suite.setdefault(suite, []).append(fraction)
        calls_by_suite.setdefault(suite, []).append(
            sum(len(parse_tool_calls(message.get("content") or "")) for message in row["conversations"])
        )
    document = {
        "trials": len(rows),
        "skipped_no_calls": skipped,
        "suites": {
            suite: {
                "trials": len(values),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "median_tool_calls": float(np.median(calls_by_suite[suite])),
                "expected_median": EXPECTED_MEDIANS[suite],
                "matches": bool(abs(float(np.median(values)) - EXPECTED_MEDIANS[suite]) <= 0.001),
            }
            for suite, values in sorted(by_suite.items())
        },
    }
    Path(args.output).write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    if set(by_suite) != set(EXPECTED_MEDIANS) or not all(
        result["matches"] for result in document["suites"].values()
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
