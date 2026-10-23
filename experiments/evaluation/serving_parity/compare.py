# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate complete canary artifacts against frozen references and same-night peers."""

import argparse
import dataclasses
import json
from itertools import combinations
from pathlib import Path

from marin.testing.inference.sequence_parity import (
    AccuracyObservation,
    ScoredSequence,
    ScoredToken,
    SequenceThresholds,
    compare_accuracy,
    compare_sequences,
)
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

BANDS = (0, 1024, 2048, 2304, 2560, 4096, 5279)
THRESHOLDS = SequenceThresholds(pooled_mean=0.025, band_mean=0.04, top1_fraction=0.03)


def read_sequence(path: StoragePath) -> ScoredSequence:
    row = json.loads(path.read_text())
    return ScoredSequence(row["case_id"], row["input_sha256"], tuple(ScoredToken(**x) for x in row["tokens"]))


def compare_arms(left: str, right: str) -> dict:
    candidate, reference = StoragePath(left), StoragePath(right)
    # A partial artifact prefix, including one whose evaluator silently failed, cannot pass.
    for root in (candidate, reference):
        json.loads((root / "complete.json").read_text())
    a = json.loads((candidate / "provenance.json").read_text())
    b = json.loads((reference / "provenance.json").read_text())
    for key in ("model", "revision", "manifest_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Cannot compare different {key}")
    expected = read_sequence(reference / "teacher-1" / "0.json")
    teachers = {}
    for concurrency in (1, 4):
        for index in range(concurrency):
            actual = read_sequence(candidate / f"teacher-{concurrency}" / f"{index}.json")
            teachers[f"{concurrency}/{index}"] = dataclasses.asdict(
                compare_sequences([actual], [expected], boundaries=BANDS, thresholds=THRESHOLDS)
            )
    accuracy = {}
    for task in ("gsm8k-0shot", "boolq"):

        def items(root: StoragePath, task: str = task) -> list[AccuracyObservation]:
            rows = json.loads((root / f"{task}-items.json").read_text())
            if len(rows) != 100:
                raise ValueError("Expected all 100 frozen samples")
            return [AccuracyObservation(x["id"], x["correct"], x["finish_reason"]) for x in rows]

        accuracy[task] = dataclasses.asdict(compare_accuracy(items(candidate), items(reference), max_z=2))
    failed = any(r["failures"] for r in [*teachers.values(), *accuracy.values()])
    return dict(status="NUMERICAL_FAIL" if failed else "PASS", teacher=teachers, accuracy=accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="NAME=artifact prefix")
    parser.add_argument("--reference", required=True, help="Reviewed immutable reference artifact prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    arms = dict(item.split("=", 1) for item in args.arm)
    if len(arms) != len(args.arm) or len(arms) < 3:
        raise ValueError("Require three distinct hardware arms")
    pairs = [(name, "reference", path, args.reference) for name, path in arms.items()]
    pairs += [(a, b, arms[a], arms[b]) for a, b in combinations(arms, 2)]
    reports = {}
    for a, b, left, right in pairs:
        try:
            reports[f"{a} vs {b}"] = compare_arms(left, right)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            reports[f"{a} vs {b}"] = dict(status="INCOMPLETE", reason=str(exc))
    args.output.write_text(json.dumps(reports, indent=2))
    if any(row["status"] != "PASS" for row in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
