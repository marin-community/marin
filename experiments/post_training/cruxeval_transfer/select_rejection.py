# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the first execution-correct E sample per prompt without changing raw traces."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.cruxeval_transfer.constants import COLLECTION_SIZE


def select_first_correct(audit: list[dict], provenance: list[dict]) -> tuple[list[dict], list[str]]:
    grades = {row["custom_id"]: row for row in audit}
    if len(grades) != len(audit) or set(grades) != {row["custom_id"] for row in provenance}:
        raise ValueError("Grades must cover every distinct source response")
    groups = {}
    for row in provenance:
        groups.setdefault(row["prompt_id"], []).append(row)
    selected = []
    uncovered = []
    for prompt, rows in sorted(groups.items()):
        if len(rows) != 8 or {row["sample"] for row in rows} != set(range(8)):
            raise ValueError("Expected eight distinct samples per prompt")
        correct = [row for row in sorted(rows, key=lambda row: row["sample"]) if grades[row["custom_id"]]["correct"]]
        if correct:
            selected.append(correct[0])
        else:
            uncovered.append(prompt)
    return selected, uncovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = [json.loads(line) for line in args.audit.read_text().splitlines()]
    provenance = [json.loads(line) for line in args.provenance.read_text().splitlines()]
    selected, uncovered = select_first_correct(audit, provenance)
    if len(audit) != COLLECTION_SIZE or len(selected) + len(uncovered) != 1600:
        raise ValueError("Incomplete collection")
    result = {
        "selection": "first execution-correct sample in index order 0..7; no reasoning or finish-reason filter",
        "selected": selected,
        "uncovered": uncovered,
        "counts": dict(Counter(row["task"] for row in selected)),
        "selected_samples": dict(Counter(row["sample"] for row in selected)),
        "selected_finish_reasons": dict(Counter(row["finish_reason"] for row in selected)),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(args.provenance.read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ("selected", "uncovered")}))
    print(f"SELECTED {len(selected)} UNCOVERED {len(uncovered)}")


if __name__ == "__main__":
    main()
