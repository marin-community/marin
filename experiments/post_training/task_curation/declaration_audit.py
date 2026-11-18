# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit every Atlas receipt family against the canonical RlDataSource layer.

The Atlas registry holds receipts ABOUT sources; the canonical source layer is the
``RlDataSource`` declarations cataloged by ``all_sources()``, extended by the pending-intake
declarations (``PENDING_MEASUREMENT_SOURCES``, which the Atlas measures before admission) and
recorded against the retirement ledger for sources deliberately taken out of declaration. When
an Atlas flow publishes a new source version (a re-pin, a retained or remediated release, a
variant) or adds a source, the same landing must update that declaration. This audit enforces
the pairing: a receipt that names a source or pin the declaration layer does not carry fails,
and a declared source without the receipts the Atlas requires of it fails. Staged-but-unbound
receipts are the drift this gate exists to stop.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from experiments.post_training.task_curation.registry import read_registry_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeclaredPin:
    """The (source, dataset pin, converter version) triple a declaration layer carries."""

    source_id: str
    dataset_id: str | None
    dataset_revision: str | None
    converter_version: str | None

    def matches(self, other: "DeclaredPin") -> bool:
        return (self.dataset_id, self.dataset_revision, self.converter_version) == (
            other.dataset_id,
            other.dataset_revision,
            other.converter_version,
        )


@dataclass(frozen=True)
class ReleasePin:
    """A published release pin a receipt or a selected source names."""

    repo: str
    revision: str


@dataclass(frozen=True)
class CanonicalLayer:
    """The declaration layer an audit measures receipts against."""

    source_ids: frozenset[str]
    declared: dict[str, DeclaredPin]
    selected: dict[str, ReleasePin]
    retired: frozenset[str]
    archived_releases: dict[str, ReleasePin]
    pending: frozenset[str] = frozenset()
    superseded_by_filtration: frozenset[str] = frozenset()

    def acknowledges(self, source_id: str) -> bool:
        return source_id in self.source_ids or source_id in self.retired


def audit_families(
    families: dict[str, dict[str, DeclaredPin]],
    *,
    layer: CanonicalLayer,
    releases: dict[str, ReleasePin] | None = None,
) -> tuple[list[str], list[str]]:
    """Return the failing findings and the informational notes for one audit pass.

    ``families`` maps a family name to its {source_id: receipt pin} rows. Census and dedup
    rows must match the declared cohort pin (pending-intake declarations and the archived
    bindings' recorded original pins count as declared); lexical and grading-census rows must
    match the declared or the selected pin; ``releases`` rows (retained release receipts) must
    match the source's selected release, be the archived-original lineage record of an admitted
    source, be the filtration input of a source an oracle release superseded, or be a staged
    remediated release awaiting a pending source's intake adoption (reported, not failed).
    Every row's source must exist in the canonical layer at all — declared, pending intake, or
    retired on the ledger.
    """
    findings: list[str] = []
    notes: list[str] = []
    for family, rows in sorted(families.items()):
        for source_id, pin in sorted(rows.items()):
            if not layer.acknowledges(source_id):
                findings.append(
                    f"{family}: {source_id} names a source the declaration layer does not declare "
                    f"(registry-only receipt at {pin.dataset_revision})"
                )
                continue
            if source_id in layer.retired:
                continue
            if family in {"census receipts", "within-source dedup"}:
                expected = layer.declared.get(source_id)
                if expected is None or not expected.matches(pin):
                    observed = (
                        f"{expected.dataset_id}@{expected.dataset_revision} ({expected.converter_version})"
                        if expected is not None
                        else "no declared cohort pin"
                    )
                    findings.append(
                        f"{family}: {source_id} receipt pins {pin.dataset_id}@{pin.dataset_revision} "
                        f"({pin.converter_version}) but the canonical layer declares {observed}"
                    )
            elif family in {"lexical receipts", "grading census receipts"}:
                declared_match = layer.declared.get(source_id)
                selected_match = layer.selected.get(source_id)
                at_declared = declared_match is not None and declared_match.matches(pin)
                at_selected = (
                    selected_match is not None
                    and selected_match.repo == pin.dataset_id
                    and selected_match.revision == pin.dataset_revision
                )
                if not at_declared and not at_selected:
                    findings.append(
                        f"{family}: {source_id} receipt pin {pin.dataset_id}@{pin.dataset_revision} "
                        f"({pin.converter_version}) matches no declared or selected pin"
                    )
    if releases is not None:
        for source_id, release in sorted(releases.items()):
            if not layer.acknowledges(source_id):
                findings.append(
                    f"retained release receipts: {source_id} names a source the declaration layer does not declare"
                )
                continue
            if source_id in layer.retired:
                continue
            selected_release = layer.selected.get(source_id)
            archived_release = layer.archived_releases.get(source_id)
            matches = selected_release is not None and (selected_release.repo, selected_release.revision) == (
                release.repo,
                release.revision,
            )
            archived = archived_release is not None and (archived_release.repo, archived_release.revision) == (
                release.repo,
                release.revision,
            )
            if matches or archived or source_id in layer.superseded_by_filtration:
                continue
            if source_id in layer.pending:
                notes.append(
                    f"retained release receipts: {source_id} is pending intake; its receipt for "
                    f"{release.repo}@{release.revision} stays staged until adoption selects it"
                )
                continue
            observed = (
                f"{selected_release.repo}@{selected_release.revision}"
                if selected_release is not None
                else "no selected release"
            )
            findings.append(
                f"retained release receipts: {source_id} selects {observed} "
                f"but its receipt publishes {release.repo}@{release.revision}"
            )
    return findings, notes


def declared_pins(sources) -> dict[str, DeclaredPin]:
    """The declared cohort pin of every configured source, keyed by canonical source id."""
    return {
        source.info.id: DeclaredPin(
            source_id=source.info.id,
            dataset_id=source.dataset.name if source.dataset else None,
            dataset_revision=source.dataset.revision if source.dataset else None,
            converter_version=source.version,
        )
        for source in sources
    }


def selected_pins(sources) -> dict[str, ReleasePin]:
    """The selected (post-retention) release pin of every assembled source."""
    pins = {}
    for source in sources:
        release = source.info.current_release or (
            ReleasePin(repo=source.dataset.name, revision=source.dataset.revision)
            if source.dataset is not None
            else None
        )
        if release is not None:
            pins[source.info.id] = ReleasePin(repo=release.repo, revision=release.revision)
    return pins


def canonical_layer() -> tuple[CanonicalLayer, dict[str, dict[str, DeclaredPin]], dict[str, ReleasePin]]:
    """Assemble the canonical layer and every receipt family's rows, and the release rows."""
    from experiments.post_training.task_curation.datasets.daytona_measurements import (  # noqa: PLC0415
        actor_census_receipts,
    )
    from experiments.post_training.task_curation.datasets.grading_measurements import (  # noqa: PLC0415
        grading_census_receipts,
    )
    from experiments.post_training.task_curation.datasets.lexical_measurements import (  # noqa: PLC0415
        lexical_measurements,
    )
    from experiments.post_training.task_curation.datasets.oracle_admitted_releases import (  # noqa: PLC0415
        oracle_admitted_release_receipts,
    )
    from experiments.post_training.task_curation.datasets.oracle_releases import (  # noqa: PLC0415
        oracle_release_receipts,
    )
    from experiments.post_training.task_curation.datasets.retained_releases import (  # noqa: PLC0415
        retained_release_receipts,
    )
    from experiments.post_training.task_curation.retirement import retired_admission_source_ids  # noqa: PLC0415
    from experiments.post_training.task_curation.sources import (  # noqa: PLC0415
        PENDING_MEASUREMENT_SOURCES,
        all_sources,
        configured_sources,
    )
    from experiments.post_training.task_curation.within_source_binding import (  # noqa: PLC0415
        read_dedup_receipts,
    )

    sources = all_sources()
    configured = configured_sources()
    declared = declared_pins(configured)
    selected = selected_pins(sources.values())
    source_ids = {source.info.id for source in sources.values()} | {
        source.info.id for source in PENDING_MEASUREMENT_SOURCES
    }
    pending_ids = {source.info.id for source in PENDING_MEASUREMENT_SOURCES}
    retired = frozenset(retired_admission_source_ids())

    # A pending-intake source is declared at its intake pin: its measurement receipts must
    # match that pin, and a remediated/retained release published for it stays staged for
    # intake until adoption selects it.
    declared.update(declared_pins(PENDING_MEASUREMENT_SOURCES))

    # An admitted source whose oracle campaign recorded an archived original carries the
    # archived baseline instead of a live declaration: its recorded conversion pin counts as
    # declared, and its retained receipt is the archived lineage record, not the selection.
    archived_releases: dict[str, ReleasePin] = {}
    for source_id, receipt in oracle_admitted_release_receipts().items():
        archived = receipt.archived_original
        if archived is None:
            continue
        plan = archived.retained.publication.plan
        declared[source_id] = DeclaredPin(
            source_id=source_id,
            dataset_id=plan.source_dataset_id,
            dataset_revision=plan.source_revision,
            converter_version=plan.converter_version,
        )
        source_ids.add(source_id)
        archived_releases[source_id] = ReleasePin(
            repo=archived.retained.publication.repo, revision=archived.retained.publication.revision
        )

    # A retained release whose source then went through oracle filtration is the filtration
    # input: the filtered or admitted release supersedes it as the selection.
    superseded_by_filtration = frozenset(oracle_release_receipts()) | frozenset(oracle_admitted_release_receipts())

    census_rows = {
        source_id: DeclaredPin(
            source_id=source_id,
            dataset_id=receipt.input.dataset_id,
            dataset_revision=receipt.input.dataset_revision,
            converter_version=receipt.input.converter_version,
        )
        for source_id, receipt in actor_census_receipts().items()
    }
    lexical_rows = {
        source_id: DeclaredPin(
            source_id=source_id,
            dataset_id=receipt.input.dataset_id,
            dataset_revision=receipt.input.dataset_revision,
            converter_version=receipt.input.converter_version,
        )
        for source_id, receipt in lexical_measurements().items()
    }
    grading_rows = {
        source_id: DeclaredPin(
            source_id=source_id,
            dataset_id=receipt.input.dataset_id,
            dataset_revision=receipt.input.dataset_revision,
            converter_version=receipt.input.converter_version,
        )
        for source_id, receipt in grading_census_receipts().items()
    }
    dedup_rows = {
        source_id: DeclaredPin(
            source_id=source_id,
            dataset_id=receipt.source_dataset_id,
            dataset_revision=receipt.source_revision,
            converter_version=receipt.converter_version,
        )
        for source_id, receipt in read_dedup_receipts(
            Path(__file__).with_name("datasets") / "within_source_dedup.json", load_json=read_registry_json
        ).items()
    }
    release_rows = {
        source_id: ReleasePin(repo=receipt.publication.repo, revision=receipt.publication.revision)
        for source_id, receipt in retained_release_receipts().items()
    }
    layer = CanonicalLayer(
        source_ids=frozenset(source_ids),
        declared=declared,
        selected=selected,
        retired=retired,
        archived_releases=archived_releases,
        pending=frozenset(pending_ids),
        superseded_by_filtration=superseded_by_filtration,
    )
    families = {
        "census receipts": census_rows,
        "lexical receipts": lexical_rows,
        "grading census receipts": grading_rows,
        "within-source dedup": dedup_rows,
    }
    return layer, families, release_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the audit summary JSON here")
    args = parser.parse_args()

    layer, families, release_rows = canonical_layer()

    findings, notes = audit_families(families, layer=layer, releases=release_rows)

    # Every configured source the Atlas measures must carry its dedup receipt: the assembly
    # raises on a missing row, so name the missing ones explicitly before it does.
    from experiments.post_training.task_curation.sources import configured_sources  # noqa: PLC0415

    missing_dedup = sorted(
        source.info.id
        for source in configured_sources()
        if source.info.id not in layer.retired and source.info.id not in families["within-source dedup"]
    )
    findings.extend(
        f"within-source dedup: {source_id} is declared but has no dedup receipt" for source_id in missing_dedup
    )

    summary = {
        "sources": len(layer.source_ids),
        "census_receipts": len(families["census receipts"]),
        "lexical_receipts": len(families["lexical receipts"]),
        "grading_census_receipts": len(families["grading census receipts"]),
        "dedup_rows": len(families["within-source dedup"]),
        "retained_release_receipts": len(release_rows),
        "findings": findings,
        "staged_for_intake": notes,
        "state": "ok" if not findings else "mismatched",
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if findings:
        raise SystemExit(f"declaration audit failed with {len(findings)} mismatches")
    LOGGER.info(
        "Declaration audit OK: %d sources, all receipt families bind (%d staged-for-intake notes)",
        len(layer.source_ids), len(notes),
    )


if __name__ == "__main__":
    main()
