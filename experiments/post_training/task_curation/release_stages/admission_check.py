# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check source preparation before loading and retain runtime backend evidence.

Missing data, missing graders, and unresolved grader images require preparation.
Backend qualification and runtime profiles are advisory and follow the source ID.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from finestore.rl.curation import QualifiedSourceDeclaration, SamplePanelCurrency
from finestore.rl.status_taxonomy import PendingStage, SourceStatus
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import BackendSurvey

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.history_staging import (
    DEFAULT_APPLET_ID,
    DEFAULT_SERVICE_URL,
    Query,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.release_stages.paced_query import RetryControl, open_paced_query
from experiments.post_training.task_curation.retirement import retired_admission_source_ids
from infra.marina.apps.rl_data_catalog.load import _docker_build_preparation_pending
from infra.marina.apps.rl_data_catalog.server.admission import (
    backend_verification_evidence,
    has_runnable_production_support,
    require_available_lexical_clearance,
    require_declaration_admission,
)
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

ADMISSION_CHECK_SUFFIX = "admission-check"
PRIOR_AVAILABLE_QUERY = (
    "SELECT s.id AS source_id FROM atlas_sources s WHERE s.load_id=:load_id AND s.declaration->>'status'='Available'"
)


@dataclass(frozen=True)
class AdmissionCheckInputs:
    """Typed inputs for one admission check; every value is a graph output."""

    release_id: str
    source_revision: str
    declarations: ArtifactRef
    backend: ArtifactRef
    snapshot_id: str
    catalog_root: str
    prior_load_id: str
    output_root: str
    expected_source_count: int
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID

    def validate(self) -> None:
        """Refuse malformed inputs before any query runs."""
        if len(self.prior_load_id) != 32 or any(char not in "0123456789abcdef" for char in self.prior_load_id):
            raise ValueError(f"prior_load_id must be a 32-hex load ID, got {self.prior_load_id!r}")
        if not self.release_id or not self.source_revision or not self.snapshot_id:
            raise ValueError("release_id, source_revision and snapshot_id are required")
        if not self.catalog_root or not self.output_root:
            raise ValueError("catalog_root and output_root are required")
        if self.expected_source_count <= 0:
            raise ValueError(f"expected_source_count must be positive, got {self.expected_source_count}")


@dataclass(frozen=True)
class AdmissionReport:
    """The full admission picture for one release population, blocked sources first."""

    blocked: dict[str, str] = field(default_factory=dict)
    gate_failures: dict[str, str] = field(default_factory=dict)
    downgraded: dict[str, tuple[str, str]] = field(default_factory=dict)
    verification_pending: dict[str, list[str]] = field(default_factory=dict)
    sample_pin_stale: dict[str, bool] = field(default_factory=dict)
    review_source_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.gate_failures


def prior_available_sources(query: Query, prior_load_id: str) -> set[str]:
    """Sources already Available in the previously selected catalog (grandfathered)."""
    rows = query(PRIOR_AVAILABLE_QUERY, {"load_id": prior_load_id})["rows"]
    return {row["source_id"] for row in rows}


def review_admissions(
    declarations: dict[str, dict],
    backend_survey: BackendSurvey,
    previously_available: set[str],
) -> AdmissionReport:
    """The load-time admission gate over every source, reporting all blockers at once.

    This mirrors ``load_catalog``'s preparation checks and backend evidence without its SQL or its load-row writes. The
    load re-runs the gate at write time; this stage exists so the same refusals
    surface in a dry run instead of on the cluster.
    """
    blocked: dict[str, str] = {}
    gate_failures: dict[str, str] = {}
    downgraded: dict[str, tuple[str, str]] = {}
    verification_pending: dict[str, list[str]] = {}
    sample_pin_stale: dict[str, bool] = {}
    backend_rows = {row.source_id: row for row in backend_survey.sources}
    retired = retired_admission_source_ids() & set(declarations)
    if retired:
        raise ValueError(f"Retired sources remain in the release population: {sorted(retired)}")
    for source_id in sorted(declarations):
        declaration = QualifiedSourceDeclaration.model_validate(declarations[source_id])
        if canonical_source_id(declaration.id) != source_id:
            raise ValueError(f"declaration identity {declaration.id!r} does not bind population row {source_id!r}")
        backend_row = backend_rows.get(source_id)
        if backend_row is None:
            blocked[source_id] = "backend survey carries no row for this source"
            continue
        verdicts = {name: fit.support.value for name, fit in backend_row.backends.items()}
        pending = backend_verification_evidence(verdicts, backend_survey.deployment_classes)
        if (
            declaration.status is SourceStatus.AVAILABLE
            and not has_runnable_production_support(verdicts, backend_survey.deployment_classes)
            and _docker_build_preparation_pending(backend_row)
        ):
            downgraded[source_id] = (
                PendingStage.BACKEND_PREPARATION_PENDING,
                "Grader Docker build context is unresolved; prepare the grader image.",
            )
        # Available rows must pass full population admission; every other status
        # publishes as metadata without admission or selectability.
        if (
            declaration.status is SourceStatus.AVAILABLE
            and filtration(declaration.model_dump(mode="json"))["state"] != "excluded"
        ):
            failures: list[str] = []
            try:
                require_declaration_admission(declaration)
            except ValueError as failure:
                failures.append(f"require_declaration_admission: {failure}")
            if source_id not in previously_available:
                try:
                    require_available_lexical_clearance(declaration)
                except ValueError as failure:
                    failures.append(f"require_available_lexical_clearance: {failure}")
            if (
                declaration.independent_qc is not None
                and declaration.independent_qc.panel_currency is SamplePanelCurrency.ARCHIVED_LINEAGE
            ):
                sample_pin_stale[source_id] = True
            if failures:
                gate_failures[source_id] = "; ".join(failures)
        if pending:
            verification_pending[source_id] = list(pending)
    if blocked or gate_failures:
        combined = {**blocked, **gate_failures}
        detail = "; ".join(f"{source_id}: {reason}" for source_id, reason in sorted(combined.items()))
        raise ValueError(f"Admission blocked for {len(combined)} source(s): {detail}")
    return AdmissionReport(
        blocked=blocked,
        gate_failures=gate_failures,
        downgraded=downgraded,
        verification_pending=verification_pending,
        sample_pin_stale=sample_pin_stale,
        review_source_count=len(declarations),
    )


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


def run_admission_check(
    inputs: AdmissionCheckInputs,
    *,
    dry_run: bool,
    query: Query | None = None,
    feeds: tuple | None = None,
) -> dict:
    """Run the admission sweep; ``dry_run`` performs the same review with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "admission_check", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        declarations_document = read_ref(inputs.declarations)
        if len(declarations_document) != inputs.expected_source_count:
            raise ValueError(f"{len(declarations_document)} declarations, expected {inputs.expected_source_count}")
        backend_document = read_ref(inputs.backend)
        survey = BackendSurvey.model_validate(backend_document)
        control = RetryControl()
        query = query if query is not None else open_paced_query(inputs.service_url, inputs.applet_id, control)
        available = prior_available_sources(query, inputs.prior_load_id)
        report = review_admissions(declarations_document, survey, available)
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{ADMISSION_CHECK_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "declarations": {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256},
            "backend": {"uri": inputs.backend.uri, "sha256": inputs.backend.sha256},
            "snapshot_id": inputs.snapshot_id,
            "prior_load_id": inputs.prior_load_id,
            "previously_available": len(available),
            "review_source_count": report.review_source_count,
            "blocked": {},
            "downgraded": {source_id: list(route) for source_id, route in sorted(report.downgraded.items())},
            "verification_pending": {
                source_id: list(pending) for source_id, pending in sorted(report.verification_pending.items())
            },
            "sample_pin_stale": {source_id: stale for source_id, stale in sorted(report.sample_pin_stale.items())},
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if not dry_run:
            receipt = seal(f"{inputs.output_root}/admission-check.json", proof)
            proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "admission_check",
            {
                "mode": mode,
                "review_source_count": report.review_source_count,
                "downgraded": len(report.downgraded),
                **(proof.get("receipt", {})),
            },
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "admission_check", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON AdmissionCheckInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["declarations"] = ArtifactRef(**document["declarations"])
    document["backend"] = ArtifactRef(**document["backend"])
    inputs = AdmissionCheckInputs(**document)
    proof = run_admission_check(inputs, dry_run=args.mode == "dry-run")
    label = "ADMISSION-CHECK-DRY-RUN" if args.mode == "dry-run" else "ADMISSION-CHECK-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
