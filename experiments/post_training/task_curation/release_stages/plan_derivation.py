# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The plan stage: derive the release plan from authoritative state (R41 brief §1).

The release plan is no longer hand-written. One stage derives it from:

- the admitted declarations on the pinned ``atlas-env-audit`` revision (the sources
  layer this module runs inside — the caller runs it in a checkout that contains
  the config's freeze revision, normally the detached snapshot the submission
  bundles, so a mid-cut admission never enters the derivation);
- the prior release's sealed state (the production catalog pointer, the prior
  load's applet SQL state, and the config's digest-verified prior-state pins);
- the frozen population artifacts for every declared source, resolved through the
  maintained census-binding chain (``declared_binding`` → the census artifact
  record → :func:`population_for_census`);
- the derived-contract receipts registry (the same documents
  ``snapshot_publication.seal_contracts`` seals), which measures the contract
  counts the loads verify.

Every ``expected_*`` count is measured here from the same inputs the release
stages read; the target load IDs are derived deterministically from the sealed
target identity; the measured-dependency pin is computed from the requirement
module at the freeze revision. Only the config's policy choices are human input.

Determinism: the plan document carries no wall-clock value — ``artifact_version``
comes from the freeze revision's own commit date — so re-deriving at the same
freeze is byte-identical and resume-safe. Live state that moved between
derivations (a new catalog load, a re-measured census receipt) makes the
re-derivation differ, and the write-once seal refuses it: a changed release is a
decision, not a side effect. Timestamps and input digests live in the derivation
receipt.

Fail fast: every unresolvable population, unknown target registry entry, prior
drift and missing prior pin is refused here, itemized with its reason, minutes
into the release instead of hours into a census.
"""

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marin.execution.artifact import read_record, result_type_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.atlas_deployment import (
    ATLAS_SERVICE,
    AtlasService,
    region_deployment,
)
from experiments.post_training.task_curation.datasets.derived_contract_measurements import (
    derived_contract_receipts,
)
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, population_for_census
from experiments.post_training.task_curation.release_stages import census as census_stage
from experiments.post_training.task_curation.release_stages import declarations as declarations_stage
from experiments.post_training.task_curation.release_stages.admission_check import prior_available_sources
from experiments.post_training.task_curation.release_stages.graph import PLAN_KIND, ReleasePlan, write_plan
from experiments.post_training.task_curation.release_stages.history_staging import Query
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.release_config import (
    ReleaseConfig,
    config_fingerprint,
)
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.sources import all_sources
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

DERIVATION_RECEIPT_KIND = "atlas-release-derivation"
DERIVATION_RECEIPT_NAME = "derivation-receipt.json"

# The in-repo fingerprint index the identity compose reads targets from; a module
# constant so tests can point the derivation at a fixture index.
REGISTRY_INDEX_PATH = Path(__file__).parent / "plans" / "registry-index.json"

# The measured dependency pins the requirement vocabulary the release's surveys
# were measured under; the module file's digest and the schema digest are both
# computed here and re-verified by catalog_load at load time.
_MEASURED_MODULE = "taskcompendium.models"

# The runtime defaults every target's qualification campaign starts from
# (deployment policy the backend validation stage re-proves per task).
VALIDATION_RUNTIME_DEFAULTS: dict[str, dict] = {
    "agent": {
        "cleanup_timeout": 120.0,
        "cpus": None,
        "gpus": 0,
        "memory_mb": None,
        "network": "allow",
        "startup_timeout": 900.0,
        "storage_mb": None,
        "user": None,
    },
    "grader": {
        "cleanup_timeout": 120.0,
        "cpus": None,
        "gpus": 0,
        "memory_mb": None,
        "network": "deny",
        "startup_timeout": 900.0,
        "storage_mb": None,
        "user": None,
    },
}


class PriorCatalogState(BaseModel):
    """The prior release's sealed catalog and applet state, measured read-only."""

    snapshot_id: str
    source_ids: tuple[str, ...]
    load_id: str
    curriculum_load_id: str
    review_panel_source_ids: tuple[str, ...]
    available_source_ids: tuple[str, ...]
    snapshot_uri: str
    snapshot_sha256: str


def _verify_freeze(repo_root: Path, revision: str) -> str:
    """The deriving checkout must contain the freeze revision; returns the head.

    The submission runs this inside the detached snapshot AT the freeze revision,
    so the declarations layer the derivation reads is the frozen layer. In the
    in-region bundle mode (no git), the submitter's attestation carries it.
    """
    from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — avoids the guard/derivation cycle at module load
        attested_revision,
    )

    attested = attested_revision()
    if attested is not None:
        return attested
    pinned = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(f"the deriving checkout does not contain the freeze revision {revision}; cut from it")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _freeze_artifact_version(repo_root: Path, revision: str) -> str:
    """The artifact version from the freeze revision's own commit date (deterministic)."""
    raw = subprocess.run(
        ["git", "show", "-s", "--format=%cI", revision],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    moment = datetime.fromisoformat(raw)
    return f"{moment.year:04d}.{moment.month:02d}.{moment.day:02d}.{moment.hour:02d}{moment.minute:02d}"


def _verify_prior_pin(ref: ArtifactRef, name: str) -> dict:
    """Digest-verify one prior-state pin and return its document."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"prior pin {name} digest mismatch: {ref.uri}")
    return json.loads(payload)


def _verify_prior_pins(config: ReleaseConfig) -> dict[str, ArtifactRef]:
    """The prior-state pins the config carries, digest-verified up front.

    URIs are pointers the config names; the digests arrive with them and are
    verified here so a moved document fails in minutes, not at a stage.
    """
    pins: dict[str, ArtifactRef] = {}
    prior = config.prior_release
    for name in (
        "declarations",
        "identity",
        "backend",
        "composed_release",
        "composed_coverage",
        "source_counts",
    ):
        ref = getattr(prior, name)
        if ref is None:
            raise ValueError(f"prior_release.{name} is required")
        _verify_prior_pin(ref, f"prior_release.{name}")
        pins[name] = ref
    return pins


def _read_registry_index(targets: tuple[str, ...]) -> ArtifactRef:
    """The in-repo registry index at the freeze revision, digest-computed, target-covering."""
    payload = REGISTRY_INDEX_PATH.read_bytes()
    index = json.loads(payload)
    sources = index.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("the registry index carries no sources mapping")
    missing = [target for target in targets if target not in sources]
    if missing:
        raise ValueError(
            f"the registry index at the freeze revision lacks {len(missing)} target entries;"
            " refresh the export before the cut: " + str(sorted(missing)[:5])
        )
    return ArtifactRef(uri=str(REGISTRY_INDEX_PATH), sha256=hashlib.sha256(payload).hexdigest())


def _measured_dependency(repo_root: Path, revision: str) -> dict:
    """The requirement module's measured pin, computed from the freeze checkout."""
    spec = importlib.util.find_spec(_MEASURED_MODULE)
    if spec is None or spec.origin is None:
        raise ValueError(f"the measured module {_MEASURED_MODULE} is not importable")
    module_path = Path(spec.origin).resolve()
    try:
        relative = module_path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"the measured module {module_path} lives outside the checkout") from error
    from taskcompendium.backend_requirements import backend_schema_sha256  # noqa: PLC0415 — the schema owner

    return {
        "module": str(relative),
        "revision": revision,
        "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "schema_sha256": backend_schema_sha256(),
    }


def _oracle_empty(source: Any) -> Any | None:
    """The source's 0-admission oracle filtration, if any (the census ships it excluded)."""
    evidence = getattr(source.info, "oracle_filtration", None)
    if evidence is not None and evidence.admitted_count == 0:
        return evidence
    return None


def _resolve_population(source_id: str, source: Any, region_prefix: str) -> tuple[str, int, str]:
    """One source's frozen population: its path, task count and evidence kind.

    Follows the maintained census-binding chain; the oracle-empty sources ship the
    excluded zero-task declaration with the oracle receipt as their evidence (the
    census never loads their population pin).
    """
    empty = _oracle_empty(source)
    if empty is not None:
        return str(empty.validation.uri), 0, "oracle-empty"
    binding = census_stage.declared_binding(source.declaration, source_id)
    population = population_for_census(binding.evidence_url, source_id)
    for name in ("path", "specs_output_dir"):
        pinned = getattr(population, name)
        if not str(pinned).startswith(region_prefix):
            raise ValueError(
                f"population {name} for {source_id} is outside the release region prefix" f" {region_prefix}: {pinned}"
            )
    if population.task_count != binding.measured_tasks:
        raise ValueError(
            f"the frozen population for {source_id} holds {population.task_count} tasks,"
            f" the declared binding measured {binding.measured_tasks}"
        )
    return population.path, population.task_count, "population"


def _target_integrity_contract(source_id: str, source: Any) -> dict | None:
    """One target's integrity-contract record, projected from its census evidence.

    The record is the census artifact's own contract summary — the same document
    shape ``snapshot_publication`` sealed for prior releases. Targets whose census
    evidence is a retained population record (no actor census) carry no contract
    record; the backend validation stage treats per-target records as optional.
    """
    if _oracle_empty(source) is not None:
        return None
    binding = census_stage.declared_binding(source.declaration, source_id)
    evidence_dir = str(StoragePath(binding.evidence_url).parent)
    record = read_record(evidence_dir)
    if record is None or record.result_type != result_type_name(ActorCensus):
        return None
    actor = ActorCensus.raw_load(evidence_dir)
    summary = actor.integrity_contracts
    if not isinstance(summary, dict) or not summary.get("groups"):
        return None
    return summary


def _derive_target_load_id(kind: str, plan_identity: dict) -> str:
    """One target load ID: 32-hex over the sealed target identity (deterministic)."""
    document = {"kind": kind, **plan_identity}
    raw = json.dumps(document, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def read_prior_catalog_state(
    catalog_root: str, query: Query, *, catalog_factory: Any = None, receipt_keys_fn: Any = None
) -> PriorCatalogState:
    """Measure the prior release's catalog and applet state (every read read-only)."""
    from finestore.rl.curation import CurationCatalog  # noqa: PLC0415 — the maintained catalog reader

    catalog = (catalog_factory or CurationCatalog)(catalog_root)
    snapshot = catalog.current()
    # Pre-stage Pending rows convert through the maintained evidence-computed
    # stage callback (never hand-entered); the receipt registry is read once.
    from experiments.post_training.task_curation.source_status import (  # noqa: PLC0415 — beside its only use
        pending_stage,
        receipt_registry_keys,
    )

    receipts = (receipt_keys_fn or receipt_registry_keys)()
    # The served snapshot carries historical source IDs; resolve them through the
    # permanent alias table so the prior state is canonical (collapsing two
    # historical entries into one canonical ID is a refusal, not a silent merge).
    from infra.marina.apps.rl_data_catalog.server.source_identity import (  # noqa: PLC0415 — beside its only use
        canonical_source_map,
    )

    def canonical_ids(rows) -> tuple[str, ...]:
        mapped = canonical_source_map({row: row for row in rows})
        return tuple(sorted(mapped))

    source_ids = canonical_ids(
        source.id
        for source in catalog.sources(snapshot.snapshot_id, pending_stage_of=lambda sid: pending_stage(sid, receipts))
    )
    load_row = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if len(load_row) != 1 or not load_row[0].get("load_id"):
        raise ValueError(f"the catalog pointer carries no selected load: {load_row!r}")
    load_id = load_row[0]["load_id"]
    curriculum_row = query("SELECT load_id FROM curriculum_current WHERE name='latest'", {})["rows"]
    if len(curriculum_row) != 1 or not curriculum_row[0].get("load_id"):
        raise ValueError(f"the curriculum pointer carries no selected load: {curriculum_row!r}")
    curriculum_load_id = curriculum_row[0]["load_id"]
    panels: list[str] = []
    after: str | None = None
    while True:
        page = query(
            "SELECT source_id FROM atlas_review_current WHERE load_id=:load"
            " AND (CAST(:after AS TEXT) IS NULL OR source_id>CAST(:after AS TEXT))"
            " ORDER BY source_id LIMIT 500",
            {"load": load_id, "after": after},
        )["rows"]
        if not page:
            break
        panels.extend(row["source_id"] for row in page)
        after = page[-1]["source_id"]
    available = prior_available_sources(query, load_id)
    # The snapshot's writer directory is the publication convention (writer id
    # "<snapshot_id>-catalog"); the digest pins the snapshot record itself.
    snapshot_uri = f"{catalog_root.rstrip('/')}/data/curation_snapshots/w={snapshot.snapshot_id}-catalog"
    record = read_record(snapshot_uri)
    snapshot_sha256 = (
        hashlib.sha256(json.dumps(record.result, sort_keys=True).encode()).hexdigest()
        if record is not None and record.result is not None
        else ""
    )
    return PriorCatalogState(
        snapshot_id=snapshot.snapshot_id,
        source_ids=source_ids,
        load_id=load_id,
        curriculum_load_id=curriculum_load_id,
        review_panel_source_ids=canonical_ids(panels),
        available_source_ids=canonical_ids(available),
        snapshot_uri=snapshot_uri,
        snapshot_sha256=snapshot_sha256,
    )


def _measure_contract_counts(targets: tuple[str, ...]) -> tuple[int, int]:
    """The contract counts the loads verify, measured from the receipts registry.

    The same documents ``snapshot_publication.seal_contracts`` seals; a target the
    registry does not cover refuses here with the itemized list.
    """
    receipts = derived_contract_receipts()
    missing = [target for target in targets if target not in receipts]
    if missing:
        raise ValueError(
            f"the derived-contract receipts registry lacks {len(missing)} targets;"
            " land their contract censuses before the cut: " + str(sorted(missing)[:5])
        )
    return len(targets), sum(receipts[target].input.measured_tasks for target in targets)


@dataclasses.dataclass(frozen=True)
class DerivedRelease:
    """One derivation's output: the release plan document and its receipt."""

    plan_document: dict
    receipt: dict

    @property
    def release_id(self) -> str:
        return self.plan_document["release_id"]


def _measure_grader_drift_reference(repo_root: Path, source_revision: str, declared: dict[str, dict]) -> Any:
    """Measure main's recipe side and every declared source's sealed recipe side.

    The drift stage is advisory (grader fixes must not block training), but its
    measurement is not optional: the derivation fetches origin/main fail-fast, so
    deriving a release whose checkout cannot see main refuses instead of sealing
    a drift reference against a stale main.
    """
    from experiments.post_training.task_curation import grader_drift  # noqa: PLC0415 — beside its only use

    return grader_drift.measure_grader_drift_reference(repo_root, source_revision=source_revision, declared=declared)


def derive_release_plan(
    config: ReleaseConfig,
    *,
    repo_root: Path,
    query: Query,
    sources_fn: Any = all_sources,
    prior_catalog_reader: Any = read_prior_catalog_state,
    drift_reference_fn: Any | None = None,
    service: AtlasService | None = None,
) -> DerivedRelease:
    """Derive the release plan from the config and authoritative state.

    All measurements happen at the freeze revision's declarations layer; the
    caller supplies the applet query client (read-only SQL). The returned plan
    document validates as :class:`ReleasePlan` and the receipt records every
    measured input. ``drift_reference_fn`` overrides the main-recipe drift
    measurement (tests inject a fixture reference; the real derivation measures
    origin/main fail-fast).
    """
    config.validate_shape()
    deployment = region_deployment(config.cluster)
    resolved_service = service or ATLAS_SERVICE
    head = _verify_freeze(repo_root, config.freeze_source_revision)
    artifact_version = _freeze_artifact_version(repo_root, config.freeze_source_revision)

    sources = sources_fn()
    declared = declarations_stage.current_declarations(lambda: sources)
    declared_ids = sorted(declared)
    if not declared_ids:
        raise ValueError("the frozen declarations layer carries no sources")

    prior = prior_catalog_reader(deployment.catalog_root, query)
    prior_ids = set(prior.source_ids)
    removed_ids = {row.source_id for row in config.removed_sources}
    unknown_removed = sorted(removed_ids - prior_ids)
    if unknown_removed:
        raise ValueError(f"removed sources the prior release never declared: {unknown_removed[:5]}")
    unplanned_removals = sorted(prior_ids - set(declared_ids) - removed_ids)
    if unplanned_removals:
        raise ValueError(
            "prior sources absent from the frozen layer without a planned removal: " + str(unplanned_removals[:5])
        )
    present_removals = sorted(removed_ids & set(declared_ids))
    if present_removals:
        raise ValueError(
            "the frozen layer still carries planned removals; land their pending-measurement"
            " staging first: " + str(present_removals[:5])
        )

    prior_pins = _verify_prior_pins(config)
    prior_declarations = _verify_prior_pin(prior_pins["declarations"], "prior_release.declarations")
    prior_curriculum = _verify_prior_pin(prior_pins["composed_release"], "prior_release.composed_release")
    prior_inventory = {canonical_source_id(row["id"]): row for row in prior_curriculum["inventory"]}
    replacements = {
        source_id
        for previous in (prior_declarations, prior_inventory)
        for source_id, row in previous.items()
        if source_id in declared and fingerprint_input(row) != fingerprint_input(declared[source_id])
    }
    targets = tuple(sorted((set(declared_ids) - prior_ids | replacements) - removed_ids))

    populations: dict[str, str] = {}
    task_counts: dict[str, int] = {}
    evidence_kinds: dict[str, str] = {}
    resolution_errors: list[str] = []
    for source_id in declared_ids:
        try:
            path, count, kind = _resolve_population(source_id, sources[source_id], deployment.region_prefix)
        except Exception as error:
            resolution_errors.append(f"{source_id}: {error}")
            continue
        populations[source_id] = path
        task_counts[source_id] = count
        evidence_kinds[source_id] = kind
    if resolution_errors:
        raise ValueError(
            f"{len(resolution_errors)} declared sources have no resolvable frozen population"
            " (land their census receipts or stage them for intake before the cut):\n  "
            + "\n  ".join(resolution_errors[:20])
        )

    expected_population_tasks = sum(task_counts.values())
    exclusion_ids = {row.source_id for row in config.qc_exclusions}
    unknown_exclusions = sorted(exclusion_ids - set(prior.review_panel_source_ids))
    if unknown_exclusions:
        raise ValueError(f"QC exclusions the prior load holds no panel for: {unknown_exclusions[:5]}")
    expected_panels = len([sid for sid in prior.review_panel_source_ids if sid not in exclusion_ids])

    registry_index = _read_registry_index(targets)
    measured = _measured_dependency(repo_root, config.freeze_source_revision)
    expected_contract_count, expected_contract_tasks = _measure_contract_counts(targets)
    grader_drift_reference = (drift_reference_fn or _measure_grader_drift_reference)(
        repo_root, config.freeze_source_revision, declared
    )

    plan_identity = {
        "release_id": config.release_id,
        "snapshot_id": f"assembly-{config.release_id}",
        "prior_load_id": prior.load_id,
        "artifact_version": artifact_version,
        "source_revision": config.freeze_source_revision,
    }
    plan = ReleasePlan(
        kind=PLAN_KIND,
        release_id=config.release_id,
        source_revision=config.freeze_source_revision,
        # One release prefix per freeze: the artifact version (the freeze commit's
        # own date) keeps re-derivations at the same freeze idempotent under the
        # same prefix while a moved freeze gets a fresh one.
        release_root=f"{deployment.region_prefix}preview-release/{config.release_id}-{artifact_version}",
        catalog_root=deployment.catalog_root,
        artifact_version=artifact_version,
        prior={
            "release_id": config.prior_release.release_id,
            "load_id": prior.load_id,
            "snapshot_uri": prior.snapshot_uri,
            "snapshot_sha256": prior.snapshot_sha256,
            "catalog_uri": deployment.catalog_root,
        },
        prior_snapshot_id=prior.snapshot_id,
        snapshot_id=plan_identity["snapshot_id"],
        targets=targets,
        expected_source_count=len(declared_ids),
        expected_browse_source_count=expected_panels,
        expected_contract_count=expected_contract_count,
        expected_contract_tasks=expected_contract_tasks,
        # The loads re-select the prior load's current-pin review panels; the QC
        # exclusions gate the browse copy, not the index replay.
        expected_selected_panels=len(prior.review_panel_source_ids),
        expected_prior_sources=len(prior.source_ids),
        expected_panels=expected_panels,
        expected_population_tasks=expected_population_tasks,
        region_prefix=deployment.region_prefix,
        populations=populations,
        scan_regions={deployment.region_label: deployment.region_prefix},
        pins={
            "prior_declarations": prior_pins["declarations"].model_dump(mode="json"),
            "prior_identity": prior_pins["identity"].model_dump(mode="json"),
            "prior_backend": prior_pins["backend"].model_dump(mode="json"),
            "prior_release": prior_pins["composed_release"].model_dump(mode="json"),
            "prior_coverage": prior_pins["composed_coverage"].model_dump(mode="json"),
            "registry_index": registry_index.model_dump(mode="json"),
            "source_counts": prior_pins["source_counts"].model_dump(mode="json"),
            "parity_receipt": config.parity_receipt.model_dump(mode="json") if config.parity_receipt else None,
            "exposure_input_check": (
                config.exposure_input_check.model_dump(mode="json") if config.exposure_input_check else None
            ),
        },
        service={
            "service_url": resolved_service.service_url,
            "applet_id": resolved_service.applet_id,
            "applet_origin": resolved_service.applet_origin,
            "public_root": resolved_service.public_root,
            "production_version": config.production_version,
            "preview_base_version": config.preview_base_version,
        },
        measured=measured,
        catalog_target_load_id=_derive_target_load_id("atlas-catalog-target-load", plan_identity),
        curriculum_target_load_id=_derive_target_load_id("atlas-curriculum-target-load", plan_identity),
        prior_curriculum_load_id=prior.curriculum_load_id,
        embed_cache_root=f"{deployment.region_prefix}embedding-cache/{config.release_id}",
        package_output_root=config.package_output_root or f"/tmp/{config.release_id}-package",
        gpu_runtime=config.gpu_runtime,
        previously_available=tuple(sorted(prior.available_source_ids)),
        qc_exclusions=tuple(sorted(exclusion_ids)),
        migrations=tuple(config.migrations),
        rebound_source=config.rebound_source,
        rebound_tasks=config.rebound_tasks,
        removed_sources=tuple(sorted(removed_ids)),
        main_source_sync=config.main_source_sync,
        strict_audit_local=config.strict_audit_local,
        grader_drift_reference=grader_drift_reference,
        validation_grading_environments={},
        validation_integrity_contracts={
            source_id: record
            for source_id in targets
            if (record := _target_integrity_contract(source_id, sources[source_id])) is not None
        },
        validation_runtime_defaults=VALIDATION_RUNTIME_DEFAULTS,
    )
    plan.validate_shape()

    receipt = {
        "kind": DERIVATION_RECEIPT_KIND,
        "release_id": config.release_id,
        "config_fingerprint": config_fingerprint(config),
        "freeze_source_revision": config.freeze_source_revision,
        "deriving_head": head,
        "artifact_version": artifact_version,
        "measured": {
            "declared_source_count": len(declared_ids),
            "prior_source_count": len(prior.source_ids),
            "prior_load_id": prior.load_id,
            "prior_curriculum_load_id": prior.curriculum_load_id,
            "prior_snapshot_id": prior.snapshot_id,
            "prior_review_panel_source_count": len(prior.review_panel_source_ids),
            "expected_panels": expected_panels,
            "expected_selected_panels": len(prior.review_panel_source_ids),
            "expected_population_tasks": expected_population_tasks,
            "expected_contract_count": expected_contract_count,
            "expected_contract_tasks": expected_contract_tasks,
            "population_task_counts": dict(sorted(task_counts.items())),
            "population_evidence_kinds": dict(sorted(evidence_kinds.items())),
            "targets": list(targets),
            "replacement_sources": sorted(replacements),
            "oracle_empty_sources": sorted(k for k, v in evidence_kinds.items() if v == "oracle-empty"),
            "integrity_contract_targets": sorted(plan.validation_integrity_contracts),
        },
        "target_load_ids": {
            "catalog": plan.catalog_target_load_id,
            "curriculum": plan.curriculum_target_load_id,
        },
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    return DerivedRelease(plan_document=plan.model_dump(mode="json"), receipt=receipt)


def seal_derivation(release_root: str, derived: DerivedRelease) -> dict:
    """Seal the derived plan and its receipt write-once under the release prefix.

    Re-deriving at the same freeze must produce byte-identical documents: a
    content mismatch refuses (a moved prior state or a changed registry is a
    decision, not a resume). Returns the sealed plan reference.
    """
    plan = ReleasePlan.model_validate(derived.plan_document)
    plan.validate_shape()
    plan_ref = write_plan(plan, release_root)
    receipt_path = StoragePath(f"{release_root}/{DERIVATION_RECEIPT_NAME}")
    raw = (json.dumps(derived.receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if receipt_path.read_bytes() != raw:
            raise ValueError(
                f"the derivation receipt already exists with different content at {receipt_path};"
                " the release inputs moved after the freeze — re-derive deliberately"
            )
    else:
        receipt_path.write_bytes(raw, auto_mkdir=True)
    if receipt_path.read_bytes() != raw:
        raise ValueError("derivation receipt readback mismatch")
    return plan_ref
