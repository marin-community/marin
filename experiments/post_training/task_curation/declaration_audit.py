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
from dataclasses import dataclass, field
from pathlib import Path

from finestore.rl.status_taxonomy import SourceStatus

from experiments.post_training.task_curation.ensure_current_projections import (
    PROJECTION_WAIVER_CONFIG,
    DeclarationPin,
    ProjectionGap,
    ProjectionWaivers,
    apply_projection_waivers,
    atlas_catalog_selection,
    atlas_contract_receipts,
    atlas_curriculum_rows,
    projection_gaps,
    record_projection_refresh_seal,
    served_declaration_pins,
    served_pin_map,
)
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
    admitted_census_converters: dict[str, str] = field(default_factory=dict)

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
                selected = layer.selected.get(source_id)
                # A sealed repair cohort's census and dedup receipts measure its published
                # sealed population while its raw declaration pins the licensed upstream
                # (ruling 66). The selected pin counts as declared only for a source whose
                # oracle-admitted release exists and whose receipt records the same census
                # converter identity as that admission; every other selected-pin match
                # stays a finding.
                admitted_converter = layer.admitted_census_converters.get(source_id)
                measured_selected = (
                    selected is not None
                    and selected.repo == pin.dataset_id
                    and selected.revision == pin.dataset_revision
                    and admitted_converter is not None
                    and admitted_converter == pin.converter_version
                )
                if expected is None or not (expected.matches(pin) or measured_selected):
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
    from experiments.post_training.task_curation.sources import RETAINED_BINDING_DRIFTS  # noqa: PLC0415

    if releases is not None:
        for source_id, release in sorted(releases.items()):
            if source_id in RETAINED_BINDING_DRIFTS:
                # The population build kept the configured pin and recorded the drift;
                # the audit reports it as a waivable served-catalog gap (ruling 2026-11-19).
                continue
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


def canonical_layer() -> tuple[CanonicalLayer, dict[str, dict[str, DeclaredPin]], dict[str, ReleasePin], dict]:
    """Assemble the canonical layer, the receipt family rows, the release rows and the sources."""
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
        bind_measured_receipts,
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
    # A pending source handed to the next release keeps its verified retained selection,
    # so its receipts at that selected pin are not orphaned (ruling 2026-11-25). A
    # retained receipt that does not verify leaves the configured pin selected, and its
    # receipts still report.
    retained_ids = set(retained_release_receipts())
    handed_on = [
        source
        for source in PENDING_MEASUREMENT_SOURCES
        if source.info.id in retained_ids and source.info.id not in selected
    ]
    selected.update(selected_pins(bind_measured_receipts(handed_on)))

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
    admitted_census_converters = {
        source_id: receipt.actor.input.converter_version
        for source_id, receipt in oracle_admitted_release_receipts().items()
    }

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
        admitted_census_converters=admitted_census_converters,
    )
    families = {
        "census receipts": census_rows,
        "lexical receipts": lexical_rows,
        "grading census receipts": grading_rows,
        "within-source dedup": dedup_rows,
    }
    return layer, families, release_rows, sources


@dataclass(frozen=True)
class DeclarationAudit:
    findings: list[str]
    notes: list[str]
    drift_sources: frozenset[str]


def audit_declarations(
    families: dict[str, dict[str, DeclaredPin]],
    *,
    layer: CanonicalLayer,
    releases: dict[str, ReleasePin],
    configured_ids: frozenset[str],
) -> DeclarationAudit:
    """Audit each source once so projection exceptions can use the same run's drift evidence."""
    source_ids = set(layer.source_ids) | set(releases) | set(configured_ids)
    for rows in families.values():
        source_ids.update(rows)
    findings, notes, drift_sources = [], [], set()
    for source_id in sorted(source_ids):
        source_findings, source_notes = audit_families(
            {family: {source_id: rows[source_id]} for family, rows in families.items() if source_id in rows},
            layer=layer,
            releases={source_id: releases[source_id]} if source_id in releases else {},
        )
        if (
            source_id in configured_ids
            and source_id not in layer.retired
            and source_id not in families["within-source dedup"]
        ):
            source_findings.append(f"within-source dedup: {source_id} is declared but has no dedup receipt")
        if source_findings:
            drift_sources.add(source_id)
        findings.extend(source_findings)
        notes.extend(source_notes)
    return DeclarationAudit(findings, notes, frozenset(drift_sources))


def audit_typed_status() -> list[str]:
    """Audit the typed status contract over the plain-JSON extras rows.

    The registry layer's declarations validate through the typed models —
    ``SourceStatus``/``SourceUse``/``TaskFormat`` typing plus the declaration
    conformance validator refuse a Pending source without a stage, an Excluded
    source without a reason, or any value outside its enum before the audit
    could see them. The extras artifact is plain JSON with no model, so the
    audit measures its rows here.
    """
    from finestore.rl.curation import SourceUse  # noqa: PLC0415
    from finestore.rl.status_taxonomy import (  # noqa: PLC0415
        PendingStage,
        SourceMethod,
        SourceStatus,
    )

    from experiments.post_training.task_curation.source_status import status_extras  # noqa: PLC0415

    extras = status_extras()
    statuses = {status.value for status in SourceStatus}
    stages = {stage.value for stage in PendingStage}
    uses = {use.value for use in SourceUse}
    methods = {method.value for method in SourceMethod}
    findings: list[str] = []
    for row in [*extras["pending"], *extras["remediation_queue"], *extras["metadata_exclusions"]]:
        if row["status"] not in statuses:
            findings.append(f"extras status {row['status']!r} is outside SourceStatus: {row['id']}")
        if row.get("stage") and row["stage"] not in stages:
            findings.append(f"extras stage {row['stage']!r} is outside PendingStage: {row['id']}")
        if row.get("use") is not None and row["use"] not in uses:
            findings.append(f"extras Use {row['use']!r} is outside SourceUse: {row['id']}")
        for tag in row.get("method_tags") or []:
            if tag not in methods:
                findings.append(f"extras Method tag {tag!r} is outside SourceMethod: {row['id']}")
    return findings


def audit_replacement_links() -> list[str]:
    """Ruling 2026-11-23: a superseded source's replacement must exist in the Atlas.

    A Deprecated superseded source carries a typed ``replacement_source_id``;
    the linked source must be declared in the Atlas in any status, and the
    status page renders the replacement's own status beside the link.
    """
    from experiments.post_training.task_curation.sources import all_sources  # noqa: PLC0415

    declared = all_sources()
    findings: list[str] = []
    for source_id, source in sorted(declared.items()):
        replacement = source.info.replacement_source_id
        if replacement is None:
            continue
        if replacement == source_id:
            findings.append(f"replacement link: {source_id} supersedes itself")
        elif replacement not in declared:
            findings.append(f"replacement link: {source_id} links {replacement}, which the Atlas does not declare")
    return findings


def exempt_unmeasured_sources(sources: dict) -> frozenset[str]:
    """The sources the projection audit raises no census demand for (ruling 2026-11-26).

    Exactly the sources that are Excluded or Deprecated with an empty retained
    population: they have no derived-contract or curriculum projection to bind (the
    coordinator extended the ruling to Deprecated with zero tasks, 2026-11-26 5:25 AM).
    Every other unmeasured source — Available, Pending, or any source with tasks —
    still reports.
    """
    exempt_statuses = (SourceStatus.EXCLUDED, SourceStatus.DEPRECATED)
    return frozenset(
        source.info.id
        for source in sources.values()
        if source.status in exempt_statuses and (source.info.count or 0) == 0
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the audit summary JSON here")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--waivers", type=Path, help="Projection landing rule with strict refresh-seal expiry")
    mode.add_argument(
        "--seal-refresh", action="store_true", help="Record refresh completion after a passing strict audit"
    )
    args = parser.parse_args()
    waivers = ProjectionWaivers.model_validate_json(args.waivers.read_bytes()) if args.waivers else None

    layer, families, release_rows, sources = canonical_layer()

    from experiments.post_training.task_curation.sources import configured_sources  # noqa: PLC0415

    declarations = audit_declarations(
        families,
        layer=layer,
        releases=release_rows,
        configured_ids=frozenset(source.info.id for source in configured_sources()),
    )
    findings, notes = declarations.findings, declarations.notes

    # The typed status contract is a landing gate, not a page-render nicety: a
    # Pending source carries its computed PendingStage, an Excluded source
    # carries a typed ExclusionReason, and every served Use, Method, Format and
    # Status value comes from its enum. The registry declarations validate
    # structurally; the extras rows are plain JSON and can drift silently, so
    # the audit measures both.
    findings.extend(audit_typed_status())

    # A superseded Deprecated source's replacement link must resolve inside the
    # Atlas (ruling 2026-11-23); the status page renders the replacement's
    # status beside the link.
    findings.extend(audit_replacement_links())

    # The serving projections (the Atlas grader and curriculum-area columns) must also bind
    # every current pin: a derived-contract census receipt and a curriculum alignment at the
    # exact pin, served through a catalog whose declarations carry the same binding.

    projection_pins = [
        DeclarationPin(
            source_id=source_id,
            dataset_id=source.dataset.name if source.dataset else None,
            dataset_revision=source.dataset.revision if source.dataset else None,
            pipeline=(
                source.pipeline_reference.model_dump(mode="json") if source.pipeline_reference is not None else None
            ),
            actor_binding=source.info.actor_census_input,
        )
        for source_id, source in sources.items()
    ]
    # Ruling 2026-11-26: a source that is Excluded with an empty retained population has
    # no derived-contract or curriculum projection to bind, so the audit raises no actor
    # census demand for it; every other unmeasured source (Available, Pending, or any
    # source with tasks) still reports and the audit keeps failing closed on real gaps.
    selection = atlas_catalog_selection()
    projection_findings, projection_notes = projection_gaps(
        projection_pins,
        atlas_contract_receipts(),
        atlas_curriculum_rows(),
        served_pins=served_pin_map(served_declaration_pins(selection)),
        exempt_unmeasured=exempt_unmeasured_sources(sources),
    )
    if atlas_catalog_selection() != selection:
        raise ValueError("Catalog selection changed during declaration audit; retry")
    from experiments.post_training.task_curation.sample_qc_binding import SAMPLE_BINDING_DRIFTS  # noqa: PLC0415
    from experiments.post_training.task_curation.sources import RETAINED_BINDING_DRIFTS  # noqa: PLC0415

    # A retained release whose conversion pin drifted from the configured declaration is
    # a serving-projection gap at the receipt's pinned revision: reported, waivable under
    # the ruling-60b rule, and never a crash (ruling 2026-11-19).
    projection_findings.extend(
        ProjectionGap(
            source_id=source_id,
            projection="served-catalog",
            state="stale-pin",
            last_computed_revision=pinned_revision,
        )
        for source_id, pinned_revision in sorted(RETAINED_BINDING_DRIFTS.items())
    )
    projection_findings.extend(
        ProjectionGap(
            source_id=source_id,
            projection="served-catalog",
            state="stale-pin",
            last_computed_revision=pinned_revision,
        )
        for source_id, pinned_revision in sorted(SAMPLE_BINDING_DRIFTS.items())
    )
    waiver_result = apply_projection_waivers(
        projection_findings,
        waivers,
        selection,
        declarations.drift_sources,
    )
    projection_blocking = waiver_result.blocking
    findings.extend(
        f"serving projection {gap.projection}/{gap.state}: {gap.source_id}"
        + (f" (last computed at {gap.last_computed_revision})" if gap.last_computed_revision else "")
        for gap in projection_blocking
    )
    notes.extend(projection_notes)

    summary = {
        "sources": len(layer.source_ids),
        "census_receipts": len(families["census receipts"]),
        "lexical_receipts": len(families["lexical receipts"]),
        "grading_census_receipts": len(families["grading census receipts"]),
        "dedup_rows": len(families["within-source dedup"]),
        "retained_release_receipts": len(release_rows),
        "findings": findings,
        "projection_waivers": [
            {"source_projection": record.key, **record.model_dump(mode="json")} for record in waiver_result.waived
        ],
        "waiver_notes": waiver_result.notes,
        "declaration_drift_sources": sorted(declarations.drift_sources),
        "waived_projection_count": len(projection_findings) - len(projection_blocking),
        "waiver_scope": waivers.model_dump(mode="json") if waivers else None,
        "catalog_load_id": selection.load_id,
        "refresh_sealed": selection.refresh_sealed,
        "staged_for_intake": notes,
        "state": "mismatched" if findings else "ok-with-waivers" if projection_findings else "ok",
    }
    if args.seal_refresh and not findings:
        scope = ProjectionWaivers.model_validate_json(PROJECTION_WAIVER_CONFIG.read_bytes())
        selection = record_projection_refresh_seal(selection, scope, summary)
        summary["refresh_sealed"] = selection.refresh_sealed
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if findings:
        raise SystemExit(f"declaration audit failed with {len(findings)} mismatches")
    LOGGER.info(
        "Declaration audit OK: %d sources, all receipt families bind (%d staged-for-intake notes)",
        len(layer.source_ids),
        len(notes),
    )


if __name__ == "__main__":
    main()
