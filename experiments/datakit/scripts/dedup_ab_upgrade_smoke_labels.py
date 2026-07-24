# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the manually reviewed issue #6854 smoke labels to v2 pair locations."""

import argparse
import json
from typing import Any

from rigging.filesystem import StoragePath

from experiments.datakit.scripts.dedup_ab_review import DedupLabel, DedupLabels, _pair_records


def upgrade_smoke_labels(
    *,
    historical: dict[str, Any],
    pair_records: list[dict[str, Any]],
    scores_dir: str,
    pairs_dir: str,
) -> DedupLabels:
    """Bind historical judgments to hash-verified materialized pair records."""
    if historical.get("version") != "v1":
        raise ValueError(f"Expected historical smoke labels v1, got {historical.get('version')!r}")

    historical_by_ids: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label in historical["labels"]:
        key = (label["variant"], label["member_id"], label["canonical_id"])
        if key in historical_by_ids:
            raise AssertionError(f"Duplicate historical smoke label: {key}")
        historical_by_ids[key] = label

    labels: list[DedupLabel] = []
    matched: set[tuple[str, str, str]] = set()
    for pair in pair_records:
        key = (pair["variant"], pair["member_id"], pair["canonical_id"])
        historical_label = historical_by_ids.get(key)
        if historical_label is None:
            raise AssertionError(f"Materialized smoke pair has no historical label: {key}")
        expected_source_suffix = f"/{historical_label['source']}/outputs/main"
        if not pair["member_source_main_dir"].endswith(expected_source_suffix):
            raise AssertionError(
                f"Historical smoke source differs for {key}: "
                f"{pair['member_source_main_dir']!r} does not end with {expected_source_suffix!r}"
            )
        matched.add(key)
        labels.append(
            DedupLabel(
                variant=pair["variant"],
                member_source_main_dir=pair["member_source_main_dir"],
                member_basename=pair["member_basename"],
                member_id=pair["member_id"],
                canonical_source_main_dir=pair["canonical_source_main_dir"],
                canonical_basename=pair["canonical_basename"],
                canonical_id=pair["canonical_id"],
                label=historical_label["label"],
                method="semantic",
                basis=historical_label["basis"],
            )
        )

    unmatched = sorted(historical_by_ids.keys() - matched)
    if unmatched:
        raise AssertionError(f"Historical smoke labels have no materialized pair: {unmatched}")
    return DedupLabels(
        scores_dir=scores_dir,
        pairs_dir=pairs_dir,
        method=(
            "Every smoke pair was manually read in full. This v2 artifact binds those judgments "
            "to hash-verified member and canonical locations."
        ),
        labels=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-labels", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--pairs-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    historical = json.loads(StoragePath(args.historical_labels).read_text())
    labels = upgrade_smoke_labels(
        historical=historical,
        pair_records=list(_pair_records(args.pairs_dir)),
        scores_dir=args.scores_dir,
        pairs_dir=args.pairs_dir,
    )
    StoragePath(args.output).write_text(labels.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
