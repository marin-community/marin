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
    if evidence_raw.get("catalog_version") != curriculum.version:
        raise ValueError("evidence catalog_version does not match curriculum")
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
