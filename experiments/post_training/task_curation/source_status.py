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
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from finestore.rl.curation import CurationCatalog, SnapshotSource, SourceUse, source_display_name
from finestore.rl.status_taxonomy import (
    REASON_EXPLANATIONS,
    ExclusionReason,
    PendingStage,
    SourceMethod,
    SourceStatus,
)

from experiments.post_training.task_curation.consumer_scope import CONSUMER_RUNTIME_COMMIT
from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
from experiments.post_training.task_curation.datasets.grading_measurements import grading_census_receipts
from experiments.post_training.task_curation.environment_artifacts import verify_declared_environment_artifacts
from experiments.post_training.task_curation.exclusion_registry import manual_exclusion_claims
from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.source_owners import source_owner
from experiments.post_training.task_curation.sources import all_sources, pending_measurement_sources
from infra.marina.apps.rl_data_catalog.server.admission import population_admission_blocker
from infra.marina.apps.rl_data_catalog.server.methods import variant_projection


@dataclass(frozen=True)
class StatusRow:
    """One status-page row shared by pending and metadata-only sources."""

    id: str
    display_name: str
    status: str
    origin: str
    upstream_dataset: str
    upstream_url: str
    notes: str
    stage: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    owner: str | None = None
    """The owning lane rendered beside the unresolved reasons; None only off the registry."""
    use: SourceUse | None = None
    """The declared intended use; metadata-only rows carry none."""
    method_tags: tuple[SourceMethod, ...] = ()
    """The variants' Method values; intake sources show RL only."""
    reasons: tuple[dict[str, Any], ...] = field(default=())

    def payload(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = list(self.reasons)
        row["method_tags"] = [tag.value for tag in self.method_tags]
        return row


def intake_columns(source: RlDataSource) -> dict[str, Any]:
    """The typed Use and Method values one row renders from its declaration."""
    return {
        "use": source.intended_use,
        "method_tags": tuple(
            SourceMethod(tag) for tag in variant_projection({"id": source.info.id}, None)["method_tags"]
        ),
    }


def receipt_registry_keys() -> dict[str, set[str]]:
    """Source IDs with nonempty task data and complete executable-grader evidence."""
    actors = actor_census_receipts()
    graders = grading_census_receipts()
    return {
        "actor census": {source_id for source_id, receipt in actors.items() if receipt.input.measured_tasks > 0},
        "grading": {
            source_id
            for source_id, receipt in graders.items()
            if population_admission_blocker(receipt.input, receipt.input.measured_tasks, receipt.evidence) is None
        },
    }


def pending_stage(source_id: str, receipts: dict[str, set[str]]) -> PendingStage | None:
    """Describe missing task data or graders; complete preparation has no hold stage."""
    covered = [source_id in keys for keys in receipts.values()]
    if covered and all(covered):
        return None
    if any(covered):
        return PendingStage.MEASUREMENTS_PENDING
    return PendingStage.INTAKE_QUEUED


def served_snapshot_sources(catalog: CurationCatalog, snapshot_id: str) -> Iterator[SnapshotSource]:
    """A served snapshot's sources; a pre-stage Pending row takes its stage from receipt evidence.

    Served snapshots store a Pending row without a stage (the stage is computed, never
    entered), so every reader binds the same evidence: the measurement registries.
    """
    receipts = cache(receipt_registry_keys)
    return catalog.sources(snapshot_id, pending_stage_of=lambda source_id: pending_stage(source_id, receipts()))


def pending_rows() -> list[dict[str, Any]]:
    """Rows for every queued source that is not configured for the catalog yet.

    A queued row whose manual exclusion claims are already recorded renders
    **Excluded** with those claims and its owner (the 2026-11-26 15:20 EST user
    rule: nothing stays Pending indefinitely — a row with no live driver to
    admission leaves Pending, keeps its typed reason and its clearing path, and
    returns to intake only when the claims clear). The rest queue Pending with
    their computed stage.
    """
    receipts = receipt_registry_keys()
    rows = []
    for source in pending_measurement_sources():
        decided = manual_exclusion_claims(source.info.id)
        owner = source_owner(source.info.id)
        common = dict(
            id=source.info.id,
            display_name=source_display_name(source.info.id),
            origin=source.info.origin.value,
            upstream_dataset=source.info.upstream_dataset,
            upstream_url=source.info.upstream_url,
            notes=source.info.notes,
            owner=owner.owner if owner else None,
            **intake_columns(source),
        )
        if decided:
            rows.append(
                StatusRow(
                    status=SourceStatus.EXCLUDED.value,
                    reasons=[claim.model_dump(mode="json") for claim in decided],
                    **common,
                ).payload()
            )
        else:
            stage = pending_stage(source.info.id, receipts)
            rows.append(
                StatusRow(
                    status=SourceStatus.PENDING.value if stage else SourceStatus.AVAILABLE.value,
                    stage=stage.value if stage else None,
                    **common,
                ).payload()
            )
    return rows


def remediation_rows() -> list[dict[str, Any]]:
    """Rows for configured sources queued for lexical remediation work.

    These rows keep every claim visible on the status page while their status
    is Pending: the remediation agent produces the clean or remediated
    private-release receipt, and the source returns to Available. Each row
    carries the source's computed lexical-remediation stage.
    """
    return [
        StatusRow(
            id=source.info.id,
            display_name=source.declaration.display_name,
            status=SourceStatus.PENDING.value,
            origin=source.info.origin.value,
            upstream_dataset=source.info.upstream_dataset,
            upstream_url=source.info.upstream_url,
            notes=source.info.notes,
            stage=source.stage.value if source.stage else None,
            owner=(source_owner(source.info.id).owner if source_owner(source.info.id) else None),
            **intake_columns(source),
            reasons=[claim.model_dump(mode="json") for claim in source.declaration.exclusion_claims],
        ).payload()
        for source in all_sources().values()
        if source.status is SourceStatus.PENDING and source.exclusion_claims
    ]


def metadata_rows() -> list[dict[str, Any]]:
    """Declaration-only Excluded rows for known-but-rejected upstreams."""
    return [
        StatusRow(
            id=row.id,
            display_name=source_display_name(row.id),
            status=SourceStatus.EXCLUDED.value,
            origin=row.origin,
            upstream_dataset=row.upstream_dataset,
            upstream_url=row.upstream_url,
            dataset_id=row.dataset_id,
            dataset_revision=row.dataset_revision,
            owner=(source_owner(row.id).owner if source_owner(row.id) else None),
            notes=row.notes,
            reasons=[claim.model_dump(mode="json") for claim in row.claims],
        ).payload()
        for row in metadata_exclusions().values()
    ]


def reason_explanations() -> dict[str, dict[str, str]]:
    """The canonical reason explanations, rendered by the page, guide and docs."""
    return {reason.value: explanation.model_dump(mode="json") for reason, explanation in REASON_EXPLANATIONS.items()}


def require_typed_rows(rows: list[dict[str, Any]]) -> None:
    """Fail the build when an extras row breaks the typed status contract.

    Every row's Status, PendingStage, ExclusionReason, Use and Method value must
    be a member of its enum, every Pending row carries its computed PendingStage
    and every Excluded row carries at least one typed ExclusionReason; a
    violation here fails the release build's status-extras generation, not a
    later page render.
    """

    def member(enum: type, value: Any, source_id: str, field: str) -> str:
        try:
            return enum(value).value
        except ValueError:
            raise ValueError(f"Untyped {field} value {value!r} on {source_id} is outside {enum.__name__}") from None

    for row in rows:
        source_id = row["id"]
        member(SourceStatus, row["status"], source_id, "status")
        if row["status"] == SourceStatus.PENDING.value and not row["stage"]:
            raise ValueError(f"Pending source carries no PendingStage: {source_id}")
        if row["display_name"] != source_display_name(source_id):
            raise ValueError(f"Row display name differs from the identity-derived display name: {source_id}")
        if row["status"] in (SourceStatus.PENDING.value, SourceStatus.EXCLUDED.value) and not row.get("owner"):
            raise ValueError(f"Unresolved {row['status']} source carries no owner: {source_id}")
        if row["status"] == SourceStatus.PENDING.value:
            entry = source_owner(source_id)
            if entry is not None and not entry.active:
                raise ValueError(
                    f"Pending source's owner shows no activity since the previous release "
                    f"(user rule 2026-11-26 15:20 EST): {source_id} ({entry.owner})"
                )
        if row["stage"]:
            member(PendingStage, row["stage"], source_id, "stage")
        for claim in row["reasons"]:
            member(ExclusionReason, claim["reason"], source_id, "reason")
        if row["status"] == SourceStatus.EXCLUDED.value and not row["reasons"]:
            raise ValueError(f"Excluded source carries no ExclusionReason: {source_id}")
        if row.get("use") is not None:
            member(SourceUse, row["use"], source_id, "use")
        for tag in row.get("method_tags") or ():
            member(SourceMethod, tag, source_id, "method")


def consumer_inputs(sources: Iterable[RlDataSource]) -> dict[str, dict[str, Any]]:
    """Freeze source runtime selection and consumer evidence for the serving app."""
    return {
        source.info.id: {
            **(
                {"consumer_runtime_commit": source.info.consumer_runtime_commit}
                if source.info.consumer_runtime_commit is not None
                else {}
            ),
            "consumer_requirements": [need.value for need in source.info.consumer_requirements],
            "consumer_receipt": (
                source.info.consumer_receipt.model_dump(mode="json") if source.info.consumer_receipt else None
            ),
            "consumer_decision": (
                source.info.consumer_decision.model_dump(mode="json") if source.info.consumer_decision else None
            ),
        }
        for source in sources
        if source.info.consumer_requirements
    }


def status_extras() -> dict[str, Any]:
    """The full static extras artifact for the status page.

    ``pending`` lists only the unconfigured intake queue: configured sources —
    including the remediation-queued rows — arrive through the served catalog,
    and the page overlays their stage from ``stages`` so no row is ever
    composed twice. ``remediation_queue`` is the worker-facing copy of the
    lexical remediation queue with each row's claims. The canonical taxonomy
    tables the page renders from are served by the ``/api/taxonomy`` endpoint.
    """
    remediation = remediation_rows()
    pending = pending_rows()
    metadata = metadata_rows()
    require_typed_rows([*pending, *remediation, *metadata])
    served_unresolved = {
        source.info.id: source_owner(source.info.id).owner if source_owner(source.info.id) else None
        for source in all_sources().values()
        if source.status in (SourceStatus.PENDING, SourceStatus.EXCLUDED)
    }
    owners = {
        **{row["id"]: row["owner"] for row in [*pending, *remediation, *metadata]},
        **served_unresolved,
    }
    missing = sorted(source_id for source_id, owner in owners.items() if not owner)
    if missing:
        raise ValueError(f"Unresolved sources carry no owner in the registry: {missing[:8]}")
    # The release-time driver check (user rule 2026-11-26 15:20 EST): a Pending row
    # whose owner shows no activity since the previous release fails the build.
    # require_typed_rows covers the built rows; this covers the served catalog's
    # Pending rows, which arrive through the owners overlay alone.
    inactive = sorted(
        source.info.id
        for source in all_sources().values()
        if source.status is SourceStatus.PENDING
        and (entry := source_owner(source.info.id)) is not None
        and not entry.active
    )
    if inactive:
        raise ValueError(
            f"Pending sources' owners show no activity since the previous release "
            f"(user rule 2026-11-26 15:20 EST): {inactive[:8]}"
        )
    return {
        "owners": owners,
        "consumer_inputs": consumer_inputs(all_sources().values()),
        "pending": pending,
        "stages": {
            **{row["id"]: row["stage"] for row in remediation if row["stage"]},
        },
        "remediation_queue": remediation,
        "metadata_exclusions": metadata,
        "grader_drift": grader_drift_rows(),
        "reason_explanations": reason_explanations(),
    }


def grader_drift_rows(reference=None, *, repo_root=None) -> list[dict[str, Any]]:
    """The advisory re-seal queue: train publications whose stored graders predate main's fixes.

    Measured live against origin/main (ruling 2026-11-26 6:45 AM EST): the
    publication's sealed recipe files at their converter revision against main's
    current blobs, with the main fix commits and PRs that touch the changed
    files. ``reference`` overrides the measured main side and ``repo_root`` the
    checkout for tests. A source with no sealed census binding has no stored
    grader to compare and never queues.
    """
    from experiments.post_training.task_curation import grader_drift  # noqa: PLC0415 — beside its only use

    sources = all_sources()
    if not sources:
        return []
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    paths = sorted({recipe_file for source in sources.values() for recipe_file in source.files})
    if reference is None:
        reference = grader_drift.measure_main_recipe_reference(
            repo_root, source_revision=grader_drift.head_revision(repo_root), paths=paths
        )
    rows = []
    for source_id, source in sorted(sources.items()):
        declaration = source.declaration.model_dump(mode="json")
        if not grader_drift.is_drift_population(declaration):
            continue
        sealed = grader_drift.measure_sealed_recipe(source_id, declaration, repo_root)
        finding = grader_drift.compare_recipe(source_id, sealed, reference)
        if finding is None:
            continue
        rows.append(
            {
                "id": source_id,
                "display_name": source_display_name(source_id),
                "finding": finding.finding,
                "converter_revision": finding.converter_revision,
                "converter_version": finding.converter_version,
                "changed_files": [file.model_dump(mode="json") for file in finding.changed_files],
                "fix_commits": [commit.model_dump(mode="json") for commit in finding.fix_commits],
                "main_revision": reference.revision,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer-output", type=Path)
    args = parser.parse_args()
    blocking, notes = verify_declared_environment_artifacts(grading_census_receipts())
    for note in notes:
        print(f"environment artifact: {note}", file=sys.stderr)
    if blocking:
        for finding in blocking:
            print(
                f"environment artifact {finding.reason}: {finding.source_id} declares {finding.artifact_uri} "
                f"({finding.task_count} tasks) and the scripted in-region rebuild failed: {finding.detail}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"declared environment artifacts are missing after rebuild attempts: {len(blocking)} finding(s)"
        )
    extras = status_extras()
    payload = json.dumps(extras, indent=2) + "\n"
    if args.consumer_output is not None:
        encoded = json.dumps(extras["consumer_inputs"], sort_keys=True, separators=(",", ":"))
        chunks = [encoded[start : start + 72] for start in range(0, len(encoded), 72)]
        literals = [repr(chunk) if '"' in chunk else json.dumps(chunk) for chunk in chunks]
        value = literals[0] if len(literals) == 1 else "(\n" + "".join(f"    {part}\n" for part in literals) + ")"
        args.consumer_output.write_text(
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + f"CONSUMER_RUNTIME_COMMIT = {json.dumps(CONSUMER_RUNTIME_COMMIT)}\n"
            + f"CONSUMER_INPUTS = {value}\n"
        )
    # The release's clean frontend build empties dist/ before the build command runs,
    # so the output's parent directories cannot be assumed to exist.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"wrote {args.output} ({len(payload.encode())} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
