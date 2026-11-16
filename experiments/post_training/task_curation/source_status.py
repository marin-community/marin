# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed rows for the Atlas source-status page.

The page composes three kinds of rows: the loaded catalog declarations, the
pending intake queue, and declaration-only metadata exclusions. Catalog
declarations arrive from the served API; pending and metadata rows are
published here as a static extras artifact so neither requires a catalog
population, conversion pipeline or schema change.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from finestore.rl.status_taxonomy import (
    PendingStage,
    REASON_EXPLANATIONS,
    SourceStatus,
)

from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES
from experiments.post_training.task_curation.within_source_binding import read_dedup_receipts

_RECEIPT_CHECKS = {
    "actor census": "experiments.post_training.task_curation.datasets.daytona_measurements:actor_census_receipts",
    "lexical": "experiments.post_training.task_curation.datasets.lexical_measurements:lexical_measurements",
    "grading": "experiments.post_training.task_curation.datasets.grading_measurements:grading_census_receipts",
    "sample": "experiments.post_training.task_curation.datasets.sample_qc_results:sample_qc_receipts",
    "oracle": "experiments.post_training.task_curation.datasets.oracle_releases:oracle_release_receipts",
    "oracle admitted": (
        "experiments.post_training.task_curation.datasets.oracle_admitted_releases:oracle_admitted_release_receipts"
    ),
    "retained": "experiments.post_training.task_curation.datasets.retained_releases:retained_release_receipts",
}


def _receipt_keys() -> dict[str, set[str]]:
    """Import each receipt registry lazily and return its covered source IDs."""
    keys: dict[str, set[str]] = {}
    for label, target in _RECEIPT_CHECKS.items():
        module_name, attribute = target.split(":")
        module = __import__(module_name, fromlist=[attribute])
        registry = getattr(module, attribute)()
        keys[label] = set(registry)
    dedup = read_dedup_receipts(
        Path(__file__).with_name("datasets") / "within_source_dedup.json", load_json=read_registry_json
    )
    keys["dedup"] = set(dedup)
    return keys


def pending_stage(source_id: str, receipts: dict[str, set[str]]) -> PendingStage:
    """Derive a pending source's stage from the measurement receipts it lacks.

    The stage describes catalog-evidence state: a queued source has no receipts
    at all, a measured source has some, and an admission-pending source has
    every receipt the standard admission path produces.
    """
    covered = [source_id in keys for keys in receipts.values()]
    if all(covered):
        return PendingStage.ADMISSION_PENDING
    if any(covered):
        return PendingStage.MEASUREMENTS_PENDING
    return PendingStage.INTAKE_QUEUED


def pending_rows() -> list[dict[str, Any]]:
    """Rows for every queued source that is not configured for the catalog yet."""
    receipts = _receipt_keys()
    return [
        {
            "id": source.info.id,
            "name": source.name,
            "display_name": source.info.upstream_name,
            "status": SourceStatus.PENDING.value,
            "stage": pending_stage(source.info.id, receipts).value,
            "origin": source.info.origin.value,
            "upstream_dataset": source.info.upstream_dataset,
            "upstream_url": source.info.upstream_url,
            "notes": source.info.notes,
            "reasons": [],
        }
        for source in PENDING_MEASUREMENT_SOURCES
    ]


def metadata_rows() -> list[dict[str, Any]]:
    """Declaration-only Excluded rows for known-but-rejected upstreams."""
    return [
        {
            "id": row.id,
            "name": row.name,
            "display_name": row.display_name,
            "status": SourceStatus.EXCLUDED.value,
            "stage": None,
            "origin": row.origin,
            "upstream_dataset": row.upstream_dataset,
            "upstream_url": row.upstream_url,
            "dataset_id": row.dataset_id,
            "dataset_revision": row.dataset_revision,
            "notes": row.notes,
            "reasons": [claim.model_dump(mode="json") for claim in row.claims],
        }
        for row in metadata_exclusions().values()
    ]


def reason_explanations() -> dict[str, dict[str, str]]:
    """The canonical reason explanations, rendered by the page, guide and docs."""
    return {
        reason.value: explanation.model_dump(mode="json") for reason, explanation in REASON_EXPLANATIONS.items()
    }


def status_extras() -> dict[str, Any]:
    """The full static extras artifact for the status page."""
    return {
        "pending": pending_rows(),
        "metadata_exclusions": metadata_rows(),
        "reason_explanations": reason_explanations(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(status_extras(), indent=2) + "\n"
    args.output.write_text(payload)
    print(f"wrote {args.output} ({len(payload.encode())} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
