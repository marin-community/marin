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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finestore.rl.curation import SourceUse, source_display_name
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
from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.datasets.oracle_admitted_releases import oracle_admitted_release_receipts
from experiments.post_training.task_curation.datasets.oracle_releases import oracle_release_receipts
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.datasets.sample_qc_results import sample_qc_receipts
from experiments.post_training.task_curation.environment_artifacts import verify_declared_environment_artifacts
from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES, all_sources
from experiments.post_training.task_curation.within_source_binding import read_dedup_receipts
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
            display_name=source_display_name(source.info.id),
            status=SourceStatus.PENDING.value,
            origin=source.info.origin.value,
            upstream_dataset=source.info.upstream_dataset,
            upstream_url=source.info.upstream_url,
            notes=source.info.notes,
            stage=pending_stage(source.info.id, receipts).value,
            **intake_columns(source),
        ).payload()
        for source in PENDING_MEASUREMENT_SOURCES
    ]


CONSUMER_PENDING_CORE_RECEIPTS = ("actor census", "lexical", "grading", "sample", "dedup")


def consumer_pending_stage(source_id: str, receipts: dict[str, set[str]]) -> PendingStage:
    """The stage a consumer-decision-Pending source queues under.

    Every core admission-measurement receipt present means the source awaits
    configuration and admission only (``ADMISSION_PENDING``); a missing core
    receipt keeps the source at its consumer check. The oracle and retained
    registries hold release evidence rather than admission gates: no-oracle
    sources never register there, admitted or not.
    """
    covered = all(source_id in receipts[name] for name in CONSUMER_PENDING_CORE_RECEIPTS)
    return PendingStage.ADMISSION_PENDING if covered else PendingStage.CONSUMER_CHECK_PENDING


def remediation_rows() -> list[dict[str, Any]]:
    """Rows for configured sources queued for lexical remediation work.

    These rows keep every claim visible on the status page while their status
    is Pending: the remediation agent produces the clean or remediated
    private-release receipt, and the source returns to Available. Each row
    carries the source's computed stage, so a source whose binding constraint
    is the snapshot budget shows its repair stage here too.
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
    return {
        "consumer_inputs": consumer_inputs(all_sources().values()),
        "pending": pending,
        "stages": {
            **{row["id"]: row["stage"] for row in remediation if row["stage"]},
            **{
                source.info.id: consumer_pending_stage(source.info.id, receipt_registry_keys()).value
                for source in all_sources().values()
                if source.approved_consumer_status is SourceStatus.PENDING and not source.exclusion_claims
            },
        },
        "remediation_queue": remediation,
        "metadata_exclusions": metadata,
        "reason_explanations": reason_explanations(),
    }


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
