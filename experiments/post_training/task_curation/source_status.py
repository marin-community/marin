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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finestore.rl.status_taxonomy import REASON_EXPLANATIONS, PendingStage, SourceStatus

from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
from experiments.post_training.task_curation.datasets.grading_measurements import grading_census_receipts
from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.datasets.oracle_admitted_releases import oracle_admitted_release_receipts
from experiments.post_training.task_curation.datasets.oracle_releases import oracle_release_receipts
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.datasets.sample_qc_results import sample_qc_receipts
from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES
from experiments.post_training.task_curation.within_source_binding import read_dedup_receipts


@dataclass(frozen=True)
class StatusRow:
    """One status-page row shared by pending and metadata-only sources."""

    id: str
    name: str
    display_name: str
    status: str
    origin: str
    upstream_dataset: str
    upstream_url: str
    notes: str
    stage: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    reasons: tuple[dict[str, Any], ...] = field(default=())

    def payload(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = list(self.reasons)
        return row


def receipt_registry_keys() -> dict[str, set[str]]:
    """Return each standard measurement registry's covered source IDs."""
    keys: dict[str, set[str]] = {
        "actor census": set(actor_census_receipts()),
        "lexical": set(lexical_measurements()),
        "grading": set(grading_census_receipts()),
        "sample": set(sample_qc_receipts()),
        "oracle": set(oracle_release_receipts()),
        "oracle admitted": set(oracle_admitted_release_receipts()),
        "retained": set(retained_release_receipts()),
    }
    dedup = read_dedup_receipts(
        Path(__file__).with_name("datasets") / "within_source_dedup.json", load_json=read_registry_json
    )
    keys["dedup"] = set(dedup)
    return keys


def pending_stage(source_id: str, receipts: dict[str, set[str]]) -> PendingStage:
    """Derive a pending source's stage from the measurement receipts it lacks.

    The stage describes catalog-evidence state: a queued source has no
    measurement receipts at all, a measured source has some, and an
    admission-pending source has every receipt the standard admission path
    produces.
    """
    covered = [source_id in keys for keys in receipts.values()]
    if all(covered):
        return PendingStage.ADMISSION_PENDING
    if any(covered):
        return PendingStage.MEASUREMENTS_PENDING
    return PendingStage.INTAKE_QUEUED


def pending_rows() -> list[dict[str, Any]]:
    """Rows for every queued source that is not configured for the catalog yet."""
    receipts = receipt_registry_keys()
    return [
        StatusRow(
            id=source.info.id,
            name=source.name,
            display_name=source.info.upstream_name,
            status=SourceStatus.PENDING.value,
            origin=source.info.origin.value,
            upstream_dataset=source.info.upstream_dataset,
            upstream_url=source.info.upstream_url,
            notes=source.info.notes,
            stage=pending_stage(source.info.id, receipts).value,
        ).payload()
        for source in PENDING_MEASUREMENT_SOURCES
    ]


def metadata_rows() -> list[dict[str, Any]]:
    """Declaration-only Excluded rows for known-but-rejected upstreams."""
    return [
        StatusRow(
            id=row.id,
            name=row.name,
            display_name=row.display_name,
            status=SourceStatus.EXCLUDED.value,
            origin=row.origin,
            upstream_dataset=row.upstream_dataset,
            upstream_url=row.upstream_url,
            dataset_id=row.dataset_id,
            dataset_revision=row.dataset_revision,
            notes=row.notes,
            reasons=[claim.model_dump(mode="json") for claim in row.claims],
        ).payload()
        for row in metadata_exclusions().values()
    ]


def reason_explanations() -> dict[str, dict[str, str]]:
    """The canonical reason explanations, rendered by the page, guide and docs."""
    return {reason.value: explanation.model_dump(mode="json") for reason, explanation in REASON_EXPLANATIONS.items()}


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
