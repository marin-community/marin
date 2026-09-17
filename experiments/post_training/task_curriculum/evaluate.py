# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a checked-in curriculum catalog against its pilot evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.post_training.task_curriculum.catalog import load_curriculum, load_macro_catalog
from experiments.post_training.task_curriculum.rubric import UnitEvidence, evaluate_unit, load_rubric

PACKAGE_ROOT = Path(__file__).parent


def _nonnegative_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _validate_macro_evaluation(raw: object) -> None:
    if not isinstance(raw, dict):
        raise ValueError("macro_assignment must be an object")
    reference = raw.get("blind_reference")
    if not isinstance(reference, dict):
        raise ValueError("macro_assignment.blind_reference must be an object")
    exact = _nonnegative_integer(reference.get("exact_macro"), "exact_macro")
    ambiguous = _nonnegative_integer(reference.get("ambiguous_macro"), "ambiguous_macro")
    gap = _nonnegative_integer(reference.get("inventory_gap"), "inventory_gap")
    if exact + ambiguous + gap != 64:
        raise ValueError("macro blind-reference counts must cover 64 pilot tasks")
    micro_gap = _nonnegative_integer(reference.get("micro_vocabulary_gap"), "micro_vocabulary_gap")
    if micro_gap > 64:
        raise ValueError("micro_vocabulary_gap exceeds pilot task count")

    comparison = raw.get("embedding_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("macro_assignment.embedding_comparison must be an object")
    for name, row in comparison.items():
        if not isinstance(row, dict):
            raise ValueError(f"embedding comparison {name} must be an object")
        total = _nonnegative_integer(row.get("exact_total"), f"{name}.exact_total")
        top_one = _nonnegative_integer(row.get("exact_top_one"), f"{name}.exact_top_one")
        top_three = _nonnegative_integer(row.get("exact_top_three"), f"{name}.exact_top_three")
        if not top_one <= top_three <= total:
            raise ValueError(f"embedding comparison {name} has inconsistent top-k counts")

    threshold = raw.get("conservative_threshold")
    if not isinstance(threshold, dict):
        raise ValueError("macro_assignment.conservative_threshold must be an object")
    threshold_tasks = 0
    for split in ("calibration", "holdout"):
        row = threshold.get(split)
        if not isinstance(row, dict):
            raise ValueError(f"macro threshold {split} must be an object")
        tasks = _nonnegative_integer(row.get("tasks"), f"{split}.tasks")
        exact_count = _nonnegative_integer(row.get("exact_macro"), f"{split}.exact_macro")
        abstention_count = _nonnegative_integer(row.get("ambiguous_or_gap"), f"{split}.ambiguous_or_gap")
        correct = _nonnegative_integer(row.get("exact_confident_correct"), f"{split}.exact_confident_correct")
        wrong = _nonnegative_integer(row.get("exact_confident_wrong"), f"{split}.exact_confident_wrong")
        correct_abstentions = _nonnegative_integer(row.get("correct_abstentions"), f"{split}.correct_abstentions")
        if exact_count + abstention_count != tasks:
            raise ValueError(f"macro threshold {split} counts do not cover the split")
        if correct + wrong > exact_count or correct_abstentions > abstention_count:
            raise ValueError(f"macro threshold {split} outcomes exceed reference counts")
        threshold_tasks += tasks
    if threshold_tasks != 64:
        raise ValueError("macro threshold splits must cover 64 pilot tasks")


def evaluate_catalog(
    macro_path: Path,
    curriculum_path: Path,
    rubric_path: Path,
    evidence_path: Path,
) -> dict[str, object]:
    """Validate a catalog and return its rubric results as JSON-compatible data."""

    macro_catalog = load_macro_catalog(macro_path)
    curriculum = load_curriculum(curriculum_path, macro_catalog)
    rubric = load_rubric(rubric_path)
    evidence_raw = json.loads(evidence_path.read_text())
    if evidence_raw.get("schema_version") != "task-curriculum-pilot-evaluation-v1":
        raise ValueError("evidence has an unsupported schema_version")
    if evidence_raw.get("catalog_version") != curriculum.version:
        raise ValueError("evidence catalog_version does not match curriculum")
    _validate_macro_evaluation(evidence_raw.get("macro_assignment"))
    evidence_rows = evidence_raw.get("unit_evidence")
    if not isinstance(evidence_rows, list):
        raise ValueError("unit_evidence must be a list")
    evidence = [UnitEvidence(**row) for row in evidence_rows]
    evidence_by_id = {row.unit_id: row for row in evidence}
    unit_ids = {unit.id for unit in curriculum.units}
    if set(evidence_by_id) != unit_ids or len(evidence_by_id) != len(evidence):
        raise ValueError("unit_evidence must contain exactly one row for every curriculum unit")

    results = [evaluate_unit(evidence_by_id[unit.id], rubric) for unit in curriculum.units]
    return {
        "catalog_version": curriculum.version,
        "rubric_version": rubric.version,
        "macro_snapshot": macro_catalog.snapshot,
        "macro_areas": len(macro_catalog.macro_areas),
        "micro_areas": len(macro_catalog.micros_by_id),
        "units": len(curriculum.units),
        "units_without_micro_area": [unit.id for unit in curriculum.units if not unit.micro_area_ids],
        "ready_units": [result.unit_id for result in results if result.ready],
        "rubric_results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-catalog", type=Path, default=PACKAGE_ROOT / "macro_areas.json")
    parser.add_argument("--curriculum", type=Path, default=PACKAGE_ROOT / "math_v0.json")
    parser.add_argument("--rubric", type=Path, default=PACKAGE_ROOT / "rubric_v1.json")
    parser.add_argument("--evidence", type=Path, default=PACKAGE_ROOT / "pilot_evaluation.json")
    args = parser.parse_args()
    print(json.dumps(evaluate_catalog(args.macro_catalog, args.curriculum, args.rubric, args.evidence), indent=2))


if __name__ == "__main__":
    main()
