# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade safe, normalized task-22 rows without deserializing candidate pickle."""

import argparse
import json
from pathlib import Path
from typing import TextIO

MAX_ROW_CHARACTERS = 1024 * 1024


def read_row(stream: TextIO) -> list[str] | None:
    """Read a normalized row; return None at EOF and raise ValueError for malformed rows."""
    line = stream.readline(MAX_ROW_CHARACTERS + 1)
    if not line:
        return None
    if len(line) > MAX_ROW_CHARACTERS:
        raise ValueError("Normalized row exceeds the grading limit")
    row = json.loads(line)
    if not isinstance(row, list) or any(not isinstance(value, str) for value in row):
        raise ValueError("A normalized row must be a JSON array of strings")
    return row


def score_rows(candidate: Path, gold: Path) -> dict:
    """Match the official gold-prefix comparison, including its treatment of extra rows.

    Gold parsing failures propagate as infrastructure errors. Malformed or short
    candidate output scores zero. Both files must already have undergone the
    same task-specific normalization in separate machines.
    """
    matched = 0
    total = 0
    rejected = False
    with gold.open(encoding="utf-8") as reference, candidate.open(encoding="utf-8") as prediction:
        while (expected := read_row(reference)) is not None:
            total += 1
            if rejected:
                continue
            try:
                observed = read_row(prediction)
            except (ValueError, UnicodeError):
                rejected = True
                continue
            if observed is None:
                rejected = True
            elif observed == expected:
                matched += 1
    return {
        "status": "scored",
        "reward": float(not rejected and matched == total),
        "detail": {"matched_rows": matched, "gold_rows": total, "malformed_or_missing_candidate_rows": rejected},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("verdict", type=Path)
    args = parser.parse_args()
    verdict = score_rows(args.candidate, args.gold)
    args.verdict.parent.mkdir(parents=True, exist_ok=True)
    args.verdict.write_text(json.dumps(verdict, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
