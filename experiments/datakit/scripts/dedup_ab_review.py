# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate exhaustive semantic labels for a fuzzy-dedup A/B audit.

The score artifact contains one row for every marker in either arm. A label
covers one dropped member and the canonical against which it was judged.
Coverage is exact: every drop must have one label, and every marker occurrence
must be either a labeled member or a canonical referenced by a label.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from rigging.filesystem import StoragePath


@dataclass(frozen=True, order=True)
class OccurrenceKey:
    variant: str
    source_main_dir: str
    basename: str
    doc_id: str


@dataclass(frozen=True, order=True)
class DropKey:
    member: OccurrenceKey
    canonical: OccurrenceKey


class DedupLabel(BaseModel):
    """One exhaustive judgment of a dropped member against its canonical."""

    variant: Literal["baseline", "treatment"]
    member_source_main_dir: str
    member_basename: str
    member_id: str
    canonical_source_main_dir: str
    canonical_basename: str
    canonical_id: str
    label: Literal["false_positive", "true_duplicate"]
    method: Literal["raw_identity", "low_overlap", "semantic"]
    basis: str = Field(min_length=1)


class DedupLabels(BaseModel):
    """Labels and provenance for one immutable audit score artifact."""

    version: Literal["v1"] = "v1"
    scores_dir: str
    method: str = Field(min_length=1)
    labels: list[DedupLabel]


def _score_occurrence(record: dict[str, Any]) -> OccurrenceKey:
    return OccurrenceKey(
        variant=record["variant"],
        source_main_dir=record["source_main_dir"],
        basename=record["basename"],
        doc_id=record["id"],
    )


def _score_drop(record: dict[str, Any]) -> DropKey:
    variant = record["variant"]
    return DropKey(
        member=_score_occurrence(record),
        canonical=OccurrenceKey(
            variant=variant,
            source_main_dir=record["canonical_source_main_dir"],
            basename=record["canonical_basename"],
            doc_id=record["canonical_id"],
        ),
    )


def _label_drop(label: DedupLabel) -> DropKey:
    return DropKey(
        member=OccurrenceKey(
            variant=label.variant,
            source_main_dir=label.member_source_main_dir,
            basename=label.member_basename,
            doc_id=label.member_id,
        ),
        canonical=OccurrenceKey(
            variant=label.variant,
            source_main_dir=label.canonical_source_main_dir,
            basename=label.canonical_basename,
            doc_id=label.canonical_id,
        ),
    )


def _unique_map[T](items: Iterable[tuple[T, Any]], kind: str) -> dict[T, Any]:
    result: dict[T, Any] = {}
    for key, value in items:
        if key in result:
            raise AssertionError(f"Duplicate {kind}: {key}")
        result[key] = value
    return result


def _require_machine_evidence(label: DedupLabel, score: dict[str, Any]) -> None:
    if label.method == "raw_identity":
        if label.label != "true_duplicate" or not score["exact_raw_text"]:
            raise AssertionError(f"raw_identity label lacks exact raw identity: {_label_drop(label)}")
        return
    if label.method == "low_overlap":
        if label.label != "false_positive" or score["evidence_class"] != "strong_false_positive":
            raise AssertionError(f"low_overlap label lacks strong false-positive evidence: {_label_drop(label)}")


def validate_label_coverage(
    score_records: Iterable[dict[str, Any]],
    labels: DedupLabels,
) -> dict[str, Any]:
    """Require one valid label per drop and coverage of every marker occurrence."""
    score_rows = list(score_records)
    score_occurrences = _unique_map(
        ((_score_occurrence(record), record) for record in score_rows),
        "score occurrence",
    )
    drops = _unique_map(
        ((_score_drop(record), record) for record in score_rows if record["role"] == "drop"),
        "score drop",
    )
    label_rows = _unique_map(((_label_drop(label), label) for label in labels.labels), "drop label")

    missing = sorted(drops.keys() - label_rows.keys())
    extra = sorted(label_rows.keys() - drops.keys())
    if missing or extra:
        raise AssertionError(
            f"Label/drop mismatch: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
        )

    covered_occurrences: set[OccurrenceKey] = set()
    for key, label in label_rows.items():
        score = drops[key]
        _require_machine_evidence(label, score)
        covered_occurrences.add(key.member)
        covered_occurrences.add(key.canonical)

    uncovered = sorted(score_occurrences.keys() - covered_occurrences)
    unknown = sorted(covered_occurrences - score_occurrences.keys())
    if uncovered or unknown:
        raise AssertionError(
            f"Marker coverage mismatch: uncovered={len(uncovered)} {uncovered[:5]}, "
            f"unknown={len(unknown)} {unknown[:5]}"
        )

    labels_by_variant = Counter((label.variant, label.label) for label in labels.labels)
    methods_by_variant = Counter((label.variant, label.method) for label in labels.labels)
    variants: dict[str, dict[str, Any]] = {}
    for variant in ("baseline", "treatment"):
        false_positives = labels_by_variant[variant, "false_positive"]
        true_duplicates = labels_by_variant[variant, "true_duplicate"]
        total = false_positives + true_duplicates
        variants[variant] = {
            "drops": total,
            "false_positives": false_positives,
            "true_duplicates": true_duplicates,
            "candidate_precision": true_duplicates / total if total else 1.0,
            "methods": {
                method: methods_by_variant[variant, method] for method in ("raw_identity", "low_overlap", "semantic")
            },
        }

    return {
        "valid": True,
        "scores_dir": labels.scores_dir,
        "score_markers": len(score_occurrences),
        "labeled_drops": len(drops),
        "covered_markers": len(covered_occurrences),
        "variants": variants,
    }


def _score_records(scores_dir: str) -> Iterator[dict[str, Any]]:
    paths = sorted(str(path) for path in StoragePath(f"{scores_dir.rstrip('/')}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No score Parquet files under {scores_dir}")
    for path in paths:
        with StoragePath(path).open("rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches():
                yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    labels = DedupLabels.model_validate_json(StoragePath(args.labels).read_text())
    result = validate_label_coverage(_score_records(labels.scores_dir), labels)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        StoragePath(args.output).write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
