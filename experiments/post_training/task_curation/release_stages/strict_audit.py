# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict audit: verify the served catalog, backend evidence and rendered bounds.

One maintained module replaces the per-release catalog checks and audits (see
INVENTORY.md): ``check_v14_catalog.py``, ``check_v15_catalog.py``,
``check_v92_inputs.py``, ``audit_final_catalog.py`` and their siblings. The audit
reads the served catalog through an injected reader factory and verifies it
against the release's sealed documents — declarations, backend survey, identity
survey — then re-derives the curriculum invariants through the maintained
checker. Rendered bounds and evidence views are audited through the maintained
``browser_checks`` and ``evidence_sweep`` only when a Playwright page factory is
injected; a run without one records ``skipped-in-preflight`` in its receipt so
the graph can see the browser evidence was not audited.

Every mismatch is collected and reported at once: one refusal names every
mismatched source and document, never just the first (same discipline as
``admission_check.review_admissions``). A dry run verifies with zero writes.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from finestore.rl.status_taxonomy import PendingStage, SourceStatus
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import BackendSurvey
from taskcompendium.source_fingerprints import SourceIdentitySurvey

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.release_browser import evidence_sweep
from infra.marina.apps.rl_data_catalog.release_check import CatalogReader, browser_checks
from infra.marina.apps.rl_data_catalog.release_curriculum import curriculum_invariants
from infra.marina.apps.rl_data_catalog.server.content_access import backend_metadata, content_is_withheld
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

STRICT_AUDIT_SUFFIX = "strict-audit"
AUDIT_PHASES = ("before", "after")
SKIPPED_BROWSER = "skipped-in-preflight"
DECLARATION_FIELDS = (
    "dataset_id",
    "dataset_revision",
    "pipeline",
    "task_count",
    "status",
    "intended_use",
    "within_source_dedup",
)
BACKEND_DOCUMENT_FIELDS = (
    "catalog_root",
    "snapshot_id",
    "git_revision",
    "schema_sha256",
    "runtime_defaults",
    "deployment_limits",
    "local_disk_pool",
)


@dataclass(frozen=True)
class StrictAuditInputs:
    """Typed inputs for one strict audit; every value is a graph output."""

    release_id: str
    source_revision: str
    origin: str
    output_root: str
    local: bool
    declarations: ArtifactRef
    backend: ArtifactRef
    identity: ArtifactRef
    expected_source_count: int
    retained_tasks: int
    phase: str

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        for name in ("release_id", "source_revision", "origin", "output_root"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.phase not in AUDIT_PHASES:
            raise ValueError(f"phase must be one of {AUDIT_PHASES}, got {self.phase!r}")
        if self.expected_source_count <= 0 or self.retained_tasks <= 0:
            raise ValueError("expected_source_count and retained_tasks must be positive")


def default_reader_factory(origin: str) -> CatalogReader:
    """Build the maintained served-catalog reader at the audit's origin."""
    return CatalogReader(origin, requests.Session())


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def _merge_mismatches(target: dict[str, list[str]], extra: dict[str, list[str]]) -> None:
    for subject, reasons in extra.items():
        target.setdefault(subject, []).extend(reasons)


def _survey_rows(survey: BackendSurvey) -> dict[str, dict]:
    """The sealed survey rows as sealed: a field the model gained after the seal
    (R40: ``BackendFit.blocker_reasons``) must not appear as its default."""
    return {
        canonical_source_id(row.source_id): row.model_dump(mode="json", exclude_unset=True) for row in survey.sources
    }


def _survey_groups(survey: SourceIdentitySurvey) -> dict[str, dict]:
    """Per source: the sealed duplicate-group document it belongs to, if any."""
    groups: dict[str, dict] = {}
    for group in survey.groups:
        document = group.model_dump(mode="json")
        for member in group.members:
            groups[canonical_source_id(member.source_id)] = document
    return groups


# The catalog load serves an Available declaration as typed Pending at one of these
# backend stages (the registry declaration is untouched); see load.py's downgrades.
LOAD_TIME_PENDING_STAGES = frozenset(
    stage.value
    for stage in (
        PendingStage.BACKEND_VERIFICATION_PENDING,
        PendingStage.BACKEND_PROFILE_PENDING,
        PendingStage.BACKEND_PREPARATION_PENDING,
    )
)


def served_status_matches(row: dict, declaration: dict) -> bool:
    """The served status is the declared one, or the load's typed backend Pending downgrade."""
    if row.get("status") == declaration.get("status"):
        return True
    return (
        declaration.get("status") == SourceStatus.AVAILABLE.value
        and row.get("status") == SourceStatus.PENDING.value
        and row.get("stage") in LOAD_TIME_PENDING_STAGES
    )


def served_backend_row(survey_row: dict, declaration: dict) -> dict:
    """The serving projection of one sealed survey row.

    The catalog load serves the declaration's grading evidence in place of the survey's
    survey-time copy (recorded as ``grading_evidence_rebound``), and the API serves only
    backend metadata for a source whose content is withheld.
    """
    row = {**survey_row, "grading_environment": declaration.get("grading_environment")}
    return backend_metadata(row) if content_is_withheld(declaration) else row


def served_retained_tasks(served: dict) -> int:
    """The served catalog's total retained task count."""
    return sum(int(row["task_count"]) for row in served["sources"])


def audit_served_population(
    served: dict,
    declarations: dict[str, dict],
    survey: BackendSurvey,
    identity: SourceIdentitySurvey,
    *,
    expected_source_count: int,
    retained_tasks: int,
    source_revision: str,
) -> dict[str, list[str]]:
    """The served sources against the sealed documents, with per-source mismatches.

    The hosted check reuses this projection for the published revision.
    """
    mismatches: dict[str, list[str]] = {}
    served_rows = {row["id"]: row for row in served["sources"]}
    expected_ids = set(declarations)
    if set(served_rows) != expected_ids:
        mismatches.setdefault("catalog", []).append(
            "served sources and sealed declarations disagree"
            f" (served-only {sorted(set(served_rows) - expected_ids)},"
            f" declared-only {sorted(expected_ids - set(served_rows))})"
        )
    if len(served_rows) != expected_source_count:
        mismatches.setdefault("catalog", []).append(
            f"{len(served_rows)} served sources, expected {expected_source_count}"
        )
    retained = served_retained_tasks(served)
    if retained != retained_tasks:
        mismatches.setdefault("catalog", []).append(
            f"served catalog retains {retained} tasks, expected {retained_tasks}"
        )
    if served["snapshot_id"] != survey.snapshot_id:
        mismatches.setdefault("catalog", []).append(
            f"served snapshot {served['snapshot_id']!r} differs from the backend survey {survey.snapshot_id!r}"
        )
    if survey.git_revision != source_revision:
        mismatches.setdefault("catalog", []).append(
            f"backend survey pins revision {survey.git_revision!r}, expected {source_revision!r}"
        )
    if set(identity.fingerprints) != expected_ids:
        mismatches.setdefault("identity-survey", []).append("identity fingerprints do not cover the release population")
    if identity.snapshot_id != survey.snapshot_id or identity.catalog_root != survey.catalog_root:
        mismatches.setdefault("identity-survey", []).append(
            "identity survey binds a different catalog snapshot than the backend survey"
        )
    backend_rows = _survey_rows(survey)
    orphaned = sorted(set(backend_rows) - expected_ids)
    if orphaned:
        mismatches.setdefault("backend-survey", []).append(
            f"backend survey carries rows outside the release population: {orphaned}"
        )
    identity_groups = _survey_groups(identity)
    for source_id in sorted(expected_ids & set(served_rows)):
        row = served_rows[source_id]
        declaration = declarations[source_id]
        for field in DECLARATION_FIELDS:
            if field == "status":
                if not served_status_matches(row, declaration):
                    mismatches.setdefault(source_id, []).append("served status differs from the sealed declaration")
                continue
            if row.get(field) != declaration.get(field):
                mismatches.setdefault(source_id, []).append(f"served {field} differs from the sealed declaration")
        expected_row = backend_rows.get(source_id)
        if expected_row is None:
            mismatches.setdefault(source_id, []).append("backend survey carries no row for this source")
        elif row.get("backend_capabilities") != served_backend_row(expected_row, declaration):
            mismatches.setdefault(source_id, []).append(
                "served backend capabilities differ from the sealed backend survey row"
            )
        if row.get("duplicate_group") != identity_groups.get(source_id):
            mismatches.setdefault(source_id, []).append("served duplicate group differs from the sealed identity survey")
    return mismatches


def audit_backends_document(survey: BackendSurvey, document: dict) -> list[str]:
    """The served api/backends document against the sealed backend survey."""
    survey_document = survey.model_dump(mode="json")
    return [
        f"served {field} differs from the sealed backend survey"
        for field in BACKEND_DOCUMENT_FIELDS
        if document.get(field) != survey_document.get(field)
    ]


def audit_identity_duplicates(served: dict, identity: SourceIdentitySurvey, duplicates: dict) -> dict[str, list[str]]:
    """The served duplicate survey against the sealed identity survey (hosted only)."""
    mismatches: dict[str, list[str]] = {}
    sealed_groups = sorted(
        (group.model_dump(mode="json") for group in identity.groups), key=lambda group: group["group_id"]
    )
    if sorted(duplicates.get("groups", []), key=lambda group: group["group_id"]) != sealed_groups:
        mismatches.setdefault("api/source-duplicates", []).append(
            "served duplicate groups differ from the sealed identity survey"
        )
    overlaps = {row["id"]: row["partial_overlaps"] for row in served["sources"] if row["partial_overlaps"]}
    if duplicates.get("partial_overlaps") != overlaps:
        mismatches.setdefault("api/source-duplicates", []).append(
            "served partial-overlap notes differ from the served source rows"
        )
    return mismatches


def audit_curriculum(reader: CatalogReader, mismatches: dict[str, list[str]]) -> dict:
    """Re-derive the curriculum invariants through the maintained checker."""
    try:
        releases = reader.get("api/curriculum/releases")["releases"]
        return curriculum_invariants(reader.get, releases)
    except ValueError as error:
        mismatches.setdefault("curriculum", []).append(str(error)[:500])
        return {}


def audit_browser_evidence(
    inputs: StrictAuditInputs,
    dry_run: bool,
    page_factory: Callable[[], Any] | None,
    reader: CatalogReader,
    served: dict,
    stored_groups: list[dict],
    mismatches: dict[str, list[str]],
) -> dict:
    """Walk the rendered bounds and evidence views through the maintained sweeps.

    A dry run never launches a browser (zero writes), and a write run without an
    injected page factory records the skip so its receipt shows the browser
    evidence was not audited.
    """
    if dry_run or page_factory is None:
        return {"browser_checks": SKIPPED_BROWSER}
    shots = Path(inputs.output_root) / "browser-shots"
    shots.mkdir(parents=True, exist_ok=True)
    root = f"{inputs.origin.rstrip('/')}/"
    try:
        return {
            "browser_checks": browser_checks(
                reader, root, served, shots, local=inputs.local, stored_groups=stored_groups
            ),
            "evidence_views": evidence_sweep(page_factory(), root, served["sources"]),
        }
    except ValueError as error:
        mismatches.setdefault("browser", []).append(str(error)[:500])
        return {"browser_checks": "failed", "evidence_views": []}


def run_strict_audit(
    inputs: StrictAuditInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
    reader_factory: Callable[[str], CatalogReader] | None = None,
    page_factory: Callable[[], Any] | None = None,
) -> dict:
    """Audit the served catalog; ``dry_run`` verifies with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "strict_audit", {"mode": mode, "phase": inputs.phase}, alert_feeds)
    try:
        inputs.validate()
        build_reader = reader_factory if reader_factory is not None else default_reader_factory
        reader = build_reader(inputs.origin)
        declarations = read_ref(inputs.declarations)
        if not isinstance(declarations, dict) or not declarations:
            raise ValueError("sealed declarations document is empty or malformed")
        survey = BackendSurvey.model_validate(read_ref(inputs.backend))
        identity = SourceIdentitySurvey.model_validate(read_ref(inputs.identity))
        served = reader.get("api/sources")
        mismatches: dict[str, list[str]] = {}
        _merge_mismatches(
            mismatches,
            audit_served_population(
                served,
                declarations,
                survey,
                identity,
                expected_source_count=inputs.expected_source_count,
                retained_tasks=inputs.retained_tasks,
                source_revision=inputs.source_revision,
            ),
        )
        backend_reasons = audit_backends_document(survey, reader.get("api/backends"))
        if backend_reasons:
            mismatches.setdefault("api/backends", []).extend(backend_reasons)
        duplicates_status = "skipped-local"
        stored_groups: list[dict] = []
        if not inputs.local:
            duplicates = reader.get("api/source-duplicates")
            _merge_mismatches(mismatches, audit_identity_duplicates(served, identity, duplicates))
            stored_groups = sorted(duplicates.get("stored_groups", []), key=lambda group: group["group_id"])
            duplicates_status = "checked"
        curriculum = audit_curriculum(reader, mismatches)
        browser = audit_browser_evidence(inputs, dry_run, page_factory, reader, served, stored_groups, mismatches)
        if mismatches:
            detail = "; ".join(f"{subject}: {'; '.join(reasons)}" for subject, reasons in sorted(mismatches.items()))
            raise ValueError(f"Strict audit refused {len(mismatches)} subject(s): {detail}")
        served_rows = served["sources"]
        checks: list[dict] = [
            {
                "name": "served-population",
                "sources": len(served_rows),
                "retained_tasks": served_retained_tasks(served),
            },
            {"name": "served-declarations", "fields": list(DECLARATION_FIELDS)},
            {"name": "served-backend-capabilities", "rows": len(survey.sources)},
            {"name": "backends-document", "fields": list(BACKEND_DOCUMENT_FIELDS)},
            {"name": "identity-duplicates", "status": duplicates_status},
            {"name": "curriculum-invariants", "releases": len(curriculum.get("releases", []))},
            {"name": "browser-evidence", **browser},
        ]
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{STRICT_AUDIT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "phase": inputs.phase,
            "origin": inputs.origin,
            "local": inputs.local,
            "source_count": len(served_rows),
            "checks": checks,
            "mismatches": {},
            "errors": [],
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "strict_audit",
                {"mode": mode, "phase": inputs.phase, "source_count": len(served_rows)},
                alert_feeds,
            )
            return proof
        receipt = seal(f"{inputs.output_root}/strict-audit-{inputs.phase}.json", proof)
        proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        alerts.stage_finished(
            inputs.release_id,
            "strict_audit",
            {"mode": mode, "phase": inputs.phase, "source_count": len(served_rows), **proof["receipt"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id,
            "strict_audit",
            {"mode": mode, "phase": inputs.phase, "error": str(error)[:500]},
            alert_feeds,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON StrictAuditInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in ("declarations", "backend", "identity"):
        document[name] = ArtifactRef(**document[name])
    inputs = StrictAuditInputs(**document)
    proof = run_strict_audit(inputs, dry_run=args.mode == "dry-run")
    label = "STRICT-AUDIT-DRY-RUN" if args.mode == "dry-run" else "STRICT-AUDIT-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
