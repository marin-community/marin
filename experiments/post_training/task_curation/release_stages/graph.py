# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release artifact graph (DESIGN.md sections 1-3): one entry point builds a release.

Every stage is a lazy ``ArtifactStep`` whose build runs the stage module's ``run_*``
in write mode and records a typed :class:`StageReceipt` (the sealed receipt reference
plus the full proof). Downstream stages bind their typed inputs from ``ctx.resolved``
of their dependencies, so hashes and URIs flow through the graph and are never copied
by hand. Independent stages are sibling dependencies of their join step and run
concurrently under the runner; cluster stages dispatch through
``marin.execution.remote`` with resources from the measured sizing curves. Remote
workers cannot return values, so a cluster stage seals its proof into a write-once
graph-receipt document under the release prefix and the driver reads it back
digest-bound — the same handoff discipline the stage receipts use.

The release plan (``ReleasePlan``) is the typed configuration resolved once at the
CLI boundary — the only place a release id, load ID or snapshot name enters the
system. Stage sources carry none of them (``guard.py`` enforces it). A changed plan
is a changed recipe: bump ``artifact_version`` so the graph rebuilds under fresh
artifact identities instead of refusing a stale cached stage.

Graph shape (``A → B`` means B consumes A's receipt; join steps wait for all inputs):

    plan → census-inputs → census ─┬→ corpus-assembly → embed → compose-manifest
                                   │                                    → curriculum-compose
                                   │                                     ├→ text-scan per region → merge
                                   │                                     └→ curriculum-load
                                   ├→ admission-check ──────────────────┐
                                   └→ qc-browse plan → qc-browse copy ──┤
    identity-backend-compose ────────────────────────────────────────────┤
    history-staging ─────────────────────────────────────────────────────┼→ snapshot-publication
                                                                         ├→ catalog-staging → catalog-load
                                                                         └→ curriculum-load
    catalog-load ──┐
    curriculum-load┴→ strict-audit before → package-preview → release-check preflight
                                                   → release-check hosted → marina-release
                                                   → hosted-check, strict-audit after
                                                   → release-check postpublish → exposure-probe

Sequencing decisions wired here (flagged for the r39 builder's quirk review):

- ``release_check(hosted)`` runs BEFORE ``marina_release``: the landed
  ``marina_release`` contract requires a hosted-phase release-check receipt pinned
  to this release, so the graph runs the hosted check against the live preview
  revision (``published_revision = preview_base_version``) before publishing.
  ``hosted_check`` then verifies the new revision through its installed API, and
  ``release_check(postpublish)`` runs the full sweep at the new revision.

- The curriculum compose seals the load-facing handoff keys (inventory count,
  unchanged-value flags, rebound pins) that ``curriculum_load.verify_handoff``
  consumes. Rebound pins are typed plan fields; the maintained additions-mode
  compose refuses a declared rebound it cannot execute (the projection pass is not
  wired into the compose yet).
"""

import dataclasses
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, get_args, get_origin

from fray.types import GpuConfig, ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import secret_path

from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from experiments.post_training.task_curation.grader_drift import GraderDriftReference
from experiments.post_training.task_curation.release_stages import (
    admission_check as admission_check_stage,
)
from experiments.post_training.task_curation.release_stages import (
    backend_validation as backend_validation_stage,
)
from experiments.post_training.task_curation.release_stages import (
    catalog_load as catalog_load_stage,
)
from experiments.post_training.task_curation.release_stages import (
    catalog_stage as catalog_stage_stage,
)
from experiments.post_training.task_curation.release_stages import (
    census as census_stage,
)
from experiments.post_training.task_curation.release_stages import (
    contract_census as contract_census_stage,
)
from experiments.post_training.task_curation.release_stages import (
    corpus_assembly as corpus_assembly_stage,
)
from experiments.post_training.task_curation.release_stages import (
    curriculum_compose as curriculum_compose_stage,
)
from experiments.post_training.task_curation.release_stages import (
    curriculum_load as curriculum_load_stage,
)
from experiments.post_training.task_curation.release_stages import (
    declarations as declarations_stage,
)
from experiments.post_training.task_curation.release_stages import (
    embed as embed_stage,
)
from experiments.post_training.task_curation.release_stages import (
    exposure_probe as exposure_probe_stage,
)
from experiments.post_training.task_curation.release_stages import (
    fresh_qc_index as fresh_qc_index_stage,
)
from experiments.post_training.task_curation.release_stages import (
    history_staging as history_staging_stage,
)
from experiments.post_training.task_curation.release_stages import (
    hosted_check as hosted_check_stage,
)
from experiments.post_training.task_curation.release_stages import (
    identity_backend_compose as identity_backend_compose_stage,
)
from experiments.post_training.task_curation.release_stages import (
    main_grader_drift as main_grader_drift_stage,
)
from experiments.post_training.task_curation.release_stages import (
    main_sources as main_source_sync_stage,
)
from experiments.post_training.task_curation.release_stages import (
    marina_release as marina_release_stage,
)
from experiments.post_training.task_curation.release_stages import (
    package_preview as package_preview_stage,
)
from experiments.post_training.task_curation.release_stages import (
    qc_browse as qc_browse_stage,
)
from experiments.post_training.task_curation.release_stages import (
    release_check as release_check_stage,
)
from experiments.post_training.task_curation.release_stages import (
    snapshot_publication as snapshot_publication_stage,
)
from experiments.post_training.task_curation.release_stages import (
    strict_audit as strict_audit_stage,
)
from experiments.post_training.task_curation.release_stages import (
    text_scans as text_scans_stage,
)
from experiments.post_training.task_curation.release_stages.manifest import (
    ArtifactRef,
    PriorPin,
    read_ref,
    sealed_contract_rows,
)
from experiments.post_training.task_curation.release_stages.paced_query import (
    RetryControl,
    open_paced_query,
)
from experiments.post_training.task_curation.release_stages.prior_curriculum import prior_curriculum_pins
from experiments.post_training.task_curation.release_stages.sizing import (
    StageResources,
    census_sizing,
    corpus_sizing,
    embedding_sizing,
    qc_browse_sizing,
)
from infra.marina.apps.rl_data_catalog.server.content_access import content_is_withheld

PLAN_KIND = "atlas-release-plan"
GRAPH_RECEIPT_KIND = "atlas-release-graph-receipt"
# The repo checkout the driver-local stages verify against (graph.py sits five
# levels below the repo root: release_stages → task_curation → post_training →
# experiments → repo).
REPO_ROOT = Path(__file__).resolve().parents[4]
_CALVER = re.compile(r"^\d{4}\.\d{2}\.\d{2}(\.\d+)?$")
_LOAD_ID_HEX = set("0123456789abcdef")


class ReleasePins(BaseModel):
    """The external sealed documents a release consumes; every pin is digest-bound."""

    prior_declarations: ArtifactRef
    prior_identity: ArtifactRef
    prior_backend: ArtifactRef
    prior_release: ArtifactRef
    prior_coverage: ArtifactRef
    prior_manifest: ArtifactRef | None = None
    registry_index: ArtifactRef
    # Plans cut before the package derived its inputs (R40) carry the prior oracle
    # inventory. No stage reads it; it stays so their sealed plan documents reseal.
    oracle_inventory: ArtifactRef | None = None
    source_counts: ArtifactRef
    exposure_input_check: ArtifactRef | None = None
    parity_receipt: ArtifactRef | None = None


class ServicePins(BaseModel):
    """The served endpoints and live revisions a release verifies against."""

    service_url: str
    applet_id: str
    applet_origin: str
    public_root: str
    production_version: int
    preview_base_version: int

    def validate_shape(self) -> None:
        if not self.service_url or not self.applet_id or not self.applet_origin or not self.public_root:
            raise ValueError("service_url, applet_id, applet_origin and public_root are required")
        if self.production_version <= 0 or self.preview_base_version <= 0:
            raise ValueError("production_version and preview_base_version must be positive")


class MeasuredDependency(BaseModel):
    """The module/schema revision the composed surveys were measured under."""

    module: str
    revision: str
    sha256: str
    schema_sha256: str

    def validate_shape(self) -> None:
        if not self.module or not self.revision or not self.sha256 or not self.schema_sha256:
            raise ValueError("the measured dependency pins (module, revision, sha256, schema_sha256) are required")


DIFFICULTY_SNAPSHOTS = Path(__file__).with_name("difficulty_snapshots.json")


class ReleasePlan(BaseModel):
    """Typed release configuration resolved once at the CLI boundary.

    Every field is data the graph passes as typed inputs; no stage source carries
    these values as literals. The plan document is sealed under the release prefix
    so the guard's no-literal rule and the preflight receipts bind it.
    """

    kind: str = PLAN_KIND
    release_id: str
    source_revision: str
    release_root: str
    catalog_root: str
    artifact_version: str
    prior: PriorPin
    prior_snapshot_id: str
    snapshot_id: str
    targets: tuple[str, ...]
    expected_source_count: int
    expected_browse_source_count: int
    expected_contract_count: int
    expected_contract_tasks: int
    expected_selected_panels: int
    expected_prior_sources: int
    expected_panels: int
    expected_population_tasks: int
    region_prefix: str
    populations: dict[str, str]
    scan_regions: dict[str, str]
    pins: ReleasePins
    service: ServicePins
    measured: MeasuredDependency
    catalog_target_load_id: str
    curriculum_target_load_id: str
    prior_curriculum_load_id: str
    embed_cache_root: str
    package_output_root: str
    gpu_runtime: str | None = None
    previously_available: tuple[str, ...] = ()
    qc_exclusions: tuple[str, ...] = ()
    difficulty_snapshot: ArtifactRef | None = None
    migrations: tuple[str, ...] = ()
    rebound_source: str = ""
    rebound_tasks: int = 0
    recut: bool = False
    deliberate_exclusions: tuple[str, ...] = ()
    panel_copy_exclusions: tuple[str, ...] = ()
    removed_sources: tuple[str, ...] = ()
    main_source_sync: main_source_sync_stage.MainSourceSyncPlan | None = None
    strict_audit_local: bool = False
    slice_output_root: str = ""
    grader_drift_reference: GraderDriftReference | None = None
    """The derivation-measured main-recipe comparison basis; a plan predating it
    reseals byte-identically (the field is absent from the document when unset).
    The drift stage it feeds is advisory and runs only when the reference exists."""
    validation_grading_environments: dict[str, dict]
    validation_integrity_contracts: dict[str, dict]
    validation_runtime_defaults: dict[str, dict]

    def validate_shape(self) -> None:
        """Refuse a plan that cannot drive the graph."""
        if self.kind != PLAN_KIND:
            raise ValueError(f"release plan kind is {self.kind!r}, expected {PLAN_KIND!r}")
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if not self.release_root or not self.catalog_root:
            raise ValueError("release_root and catalog_root are required")
        if not _CALVER.match(self.artifact_version):
            raise ValueError(
                f"artifact_version {self.artifact_version!r} must be a calendar version YYYY.MM.DD(.N)"
                " so the lazy graph identities are fresh per cut"
            )
        for name in ("prior.load_id", "catalog_target_load_id", "curriculum_target_load_id", "prior_curriculum_load_id"):
            value = self.prior.load_id if name == "prior.load_id" else getattr(self, name)
            if len(value) != 32 or any(char not in _LOAD_ID_HEX for char in value):
                raise ValueError(f"{name} must be 32-hex, got {value!r}")
        if not self.targets or self.expected_source_count <= 0:
            raise ValueError("targets and a positive expected_source_count are required")
        for name in (
            "expected_contract_count",
            "expected_contract_tasks",
            "expected_prior_sources",
            "expected_population_tasks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("expected_browse_source_count", "expected_selected_panels", "expected_panels"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.region_prefix.endswith("/"):
            raise ValueError("region_prefix must end with '/' so region pins are unambiguous")
        if not self.populations:
            raise ValueError("populations must pin a frozen population artifact for every declared source")
        if not self.scan_regions:
            raise ValueError("scan_regions must label at least one storage region for the text scans")
        for label, prefix in self.scan_regions.items():
            if not label or not prefix or not prefix.endswith("/"):
                raise ValueError(f"scan region {label!r} needs a non-empty '/'-suffixed prefix")
        if not self.embed_cache_root or not self.package_output_root:
            raise ValueError("embed_cache_root and package_output_root are required")
        if self.main_source_sync is not None and self.main_source_sync.queue_path == "":
            raise ValueError("main_source_sync.queue_path must be a path or unset, not an empty string")
        self.service.validate_shape()
        self.measured.validate_shape()
        if len(self.previously_available) != len(set(self.previously_available)):
            raise ValueError("previously_available repeats a source id")
        if self.rebound_tasks < 0 or (bool(self.rebound_source) != (self.rebound_tasks > 0)):
            raise ValueError("rebound pins must be empty (no rebound) or a source with a positive task count")


def _plan_document_data(plan: ReleasePlan) -> dict:
    """The canonical plan document bytes' source: the model dump, with unset opt-ins absent.

    A plan that predates an optional opt-in reseals byte-identically: the field is
    absent from the document rather than serialized as null, so adding an opt-in to
    the model never moves a sealed plan or any fingerprint that binds it.
    """
    data = plan.model_dump(mode="json")
    if plan.main_source_sync is None:
        data.pop("main_source_sync", None)
    if plan.grader_drift_reference is None:
        data.pop("grader_drift_reference", None)
    if plan.pins.oracle_inventory is None:
        data["pins"].pop("oracle_inventory", None)
    return data


def load_plan(path: Path | str) -> ReleasePlan:
    """Load and validate the sealed release plan document.

    The document is either a repo file (the reviewed plan) or the sealed derived
    plan's storage URI under the release prefix; storage URIs read through the
    storage layer so an in-region bundle can run the sealed plan directly.
    """
    source = str(path)
    text = StoragePath(source).read_text() if "://" in source else Path(source).read_text(encoding="utf-8")
    plan = ReleasePlan.model_validate_json(text)
    registry = json.loads(DIFFICULTY_SNAPSHOTS.read_text())
    bound = registry.get(plan.release_id)
    if bound is not None:
        reference = ArtifactRef.model_validate(bound)
        if plan.difficulty_snapshot is not None and plan.difficulty_snapshot != reference:
            raise ValueError("Release plan differs from its registered difficulty seal")
        plan = plan.model_copy(update={"difficulty_snapshot": reference})
    plan.validate_shape()
    return plan


def plan_fingerprint(plan: ReleasePlan) -> str:
    """The digest of the canonical plan document; the identity its stages carry."""
    document = json.dumps(_plan_document_data(plan), sort_keys=True)
    return hashlib.sha256(document.encode()).hexdigest()


def _ref_dict(ref: ArtifactRef | None) -> dict | None:
    return None if ref is None else {"uri": ref.uri, "sha256": ref.sha256}


def plan_ref(release_root: str, relative: str, sha256: str | None = None) -> dict:
    """A {uri, sha256} reference into the release prefix (sha filled by the stage)."""
    uri = str(StoragePath(f"{release_root}/{relative}"))
    return {"uri": uri, "sha256": sha256} if sha256 else {"uri": uri, "sha256": ""}


# The derived package inputs (oracle inventory, source counts) seal under this stage dir.
PACKAGE_INPUTS_STAGE_DIR = "package-inputs"
COMPOSE_MANIFEST_VERSION_SUFFIX = "served-prior-v2"


def stage_output_root(plan: ReleasePlan, stage: str) -> str:
    """The deterministic output prefix of one stage's artifacts."""
    return f"{plan.release_root}/{stage}"


def write_plan(plan: ReleasePlan, release_root: str) -> dict:
    """Seal the release plan document write-once; the graph's first artifact."""
    raw = (json.dumps(_plan_document_data(plan), indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(f"{release_root}/release-plan.json")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"release plan already exists with different content at {path}")
    else:
        path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError("release plan readback mismatch")
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


# ---------------------------------------------------------------------------
# Graph artifact types
# ---------------------------------------------------------------------------


class PlanDocument(Artifact):
    """The sealed release plan document under the release prefix."""

    uri: str
    sha256: str


class CensusInputsDocument(Artifact):
    """The sealed census-inputs document: one row per declared source."""

    uri: str
    sha256: str
    source_count: int


class ComposeManifestDocument(Artifact):
    """The sealed composition manifest the curriculum compose consumes."""

    uri: str
    sha256: str
    targets: tuple[str, ...]


class StageReceipt(Artifact):
    """The lazy-graph record of one stage run: the sealed receipt plus its proof."""

    stage: str
    release_id: str
    receipt: ArtifactRef
    proof: dict


# ---------------------------------------------------------------------------
# Typed-input construction: JSON configs rebuilt into the stage dataclasses
# ---------------------------------------------------------------------------


def _coerce_ref(value: Any, hint: Any) -> Any:
    """Rebuild one typed-input value from its JSON form against a type hint."""
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is dict:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        value_hint = args[1] if len(args) == 2 else None
        if value_hint is ArtifactRef:
            return {key: ArtifactRef(**item) for key, item in value.items()}
        if value_hint is not None and value_hint is not dict and get_origin(value_hint) is None:
            # Plain typed values stay as-is; the JSON config already carries them.
            return {key: _coerce_ref(item, value_hint) for key, item in value.items()}
        return value
    if origin is not None:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        if origin is tuple:
            return tuple(value)
        if len(args) == 1:
            return _coerce_ref(value, args[0])
        raise ValueError(f"unsupported typed-input hint {hint!r}")
    if hint is ArtifactRef:
        return ArtifactRef(**value)
    return value


def typed_inputs(inputs_type: type, config: dict) -> Any:
    """Build one stage's typed inputs from its JSON config, coercing ArtifactRef fields."""
    import typing  # noqa: PLC0415 — only the graph's dispatch boundary needs the reflection

    hints = typing.get_type_hints(inputs_type)
    kwargs = {}
    for field in dataclasses.fields(inputs_type):
        if field.name not in config and field.default is not dataclasses.MISSING:
            # A phase-specific optional input the stage's config omits keeps its default.
            continue
        kwargs[field.name] = _coerce_ref(config[field.name], hints[field.name])
    return inputs_type(**kwargs)


def _receipt_ref(proof: dict, key: str) -> ArtifactRef:
    """Extract a sealed {uri, sha256} reference from a proof, following dotted keys."""
    value: Any = proof
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"stage proof carries no {key!r} sealed reference: {sorted(proof)}")
        value = value[part]
    if not isinstance(value, dict) or "uri" not in value or "sha256" not in value:
        raise ValueError(f"stage proof key {key!r} is not a sealed reference")
    return ArtifactRef(uri=value["uri"], sha256=value["sha256"])


def _stage_receipt(stage: str, proof: dict, receipt_key: str) -> StageReceipt:
    return StageReceipt(
        stage=stage,
        release_id=proof["release_id"],
        receipt=_receipt_ref(proof, receipt_key),
        proof=proof,
    )


# ---------------------------------------------------------------------------
# Per-stage executors: write-mode entry points, dispatchable on a worker
# ---------------------------------------------------------------------------


def _execute_census_inputs(config: dict) -> dict:
    declarations = read_ref(ArtifactRef(**config["declarations"]))
    populations = config["populations"]
    if config.get("recut"):
        # A status-only re-cut's census is incremental: the new admissions' rows
        # only. The retained population's census receipt is the prior
        # publication's own, so the retained populations are never re-read.
        targets = set(config["targets"])
        declarations = {source_id: row for source_id, row in declarations.items() if source_id in targets}
        populations = {source_id: path for source_id, path in populations.items() if source_id in targets}
    return _seal_census_inputs(
        declarations=declarations,
        populations=populations,
        release_root=config["release_root"],
        source_revision=config["source_revision"],
    )


def _execute_compose_manifest(config: dict) -> dict:
    contracts = sealed_contract_rows(read_ref(ArtifactRef(**config["contracts"])))
    prior_release, prior_coverage = config["prior_release"], config["prior_coverage"]
    if config.get("prior_curriculum"):
        # Older sealed re-cut plans refreshed catalog metadata but kept the
        # config's original curriculum. Recover from the load ID already frozen
        # in that plan, never from today's selected pointer.
        prior = config["prior_curriculum"]
        query = open_paced_query(prior["service_url"], prior["applet_id"], RetryControl())
        pins = prior_curriculum_pins(query, prior["load_id"])
        prior_release = pins["composed_release"].model_dump(mode="json")
        prior_coverage = pins["composed_coverage"].model_dump(mode="json")
    return _seal_compose_manifest(
        declarations=read_ref(ArtifactRef(**config["declarations"])),
        contract_receipts={row["source_id"]: DerivedContractReceipt.model_validate(row) for row in contracts},
        targets=tuple(config["targets"]),
        prior_release=prior_release,
        prior_coverage=prior_coverage,
        fresh_release=config["fresh_release"],
        fresh_coverage=config["fresh_coverage"],
        rebound_source=config["rebound_source"],
        rebound_tasks=config["rebound_tasks"],
        release_root=config["release_root"],
        removed_sources=tuple(config.get("removed_sources", ())),
    )


def _execute_declarations(config: dict) -> dict:
    inputs = typed_inputs(declarations_stage.DeclarationInputs, config)
    return declarations_stage.run_declarations(inputs, dry_run=False, repo_root=REPO_ROOT)


def _execute_admission_check(config: dict) -> dict:
    inputs = typed_inputs(admission_check_stage.AdmissionCheckInputs, config)
    return admission_check_stage.run_admission_check(inputs, dry_run=False)


def _execute_census(config: dict) -> dict:
    return census_stage.run_census(typed_inputs(census_stage.CensusInputs, config), dry_run=False)


def _execute_corpus_assembly(config: dict) -> dict:
    inputs = typed_inputs(corpus_assembly_stage.CorpusAssemblyInputs, config)
    return corpus_assembly_stage.run_corpus_assembly(inputs, dry_run=False)


def _execute_embed(config: dict) -> dict:
    return embed_stage.run_embed(typed_inputs(embed_stage.EmbedInputs, config), dry_run=False)


def _execute_curriculum_compose(config: dict) -> dict:
    inputs = typed_inputs(curriculum_compose_stage.CurriculumComposeInputs, config)
    return curriculum_compose_stage.run_compose(inputs, dry_run=False)


def _execute_identity_backend_compose(config: dict) -> dict:
    inputs = typed_inputs(identity_backend_compose_stage.IdentityBackendComposeInputs, config)
    return identity_backend_compose_stage.run_identity_backend_compose(inputs, dry_run=False)


def _execute_backend_validation(config: dict) -> dict:
    inputs = typed_inputs(backend_validation_stage.BackendValidationInputs, config)
    return backend_validation_stage.run_backend_validation(inputs, dry_run=False, repo_root=REPO_ROOT)


def _execute_fresh_qc_index(config: dict) -> dict:
    inputs = typed_inputs(fresh_qc_index_stage.FreshQcIndexInputs, config)
    return fresh_qc_index_stage.run_fresh_qc_index(inputs, dry_run=False)


def _execute_history_staging(config: dict) -> dict:
    inputs = typed_inputs(history_staging_stage.HistoryStagingInputs, config)
    return history_staging_stage.run_history_staging(inputs, dry_run=False)


def _execute_text_scan(config: dict) -> dict:
    inputs = typed_inputs(text_scans_stage.TextScanInputs, config)
    return text_scans_stage.regional_scan(inputs, config["scan_region"], dry_run=False)


def _execute_text_scan_merge(config: dict) -> dict:
    inputs = typed_inputs(text_scans_stage.TextScanInputs, config)
    receipts = tuple(ArtifactRef(**ref) for ref in config["region_receipts"])
    return text_scans_stage.merge_region_receipts(inputs, receipts, dry_run=False)


def _execute_qc_browse_plan(config: dict) -> dict:
    inputs = typed_inputs(qc_browse_stage.QcBrowseInputs, config)
    control = RetryControl()
    query = open_paced_query(inputs.service_url, inputs.applet_id, control)
    return qc_browse_stage.plan_qc_browse(inputs, query=query, dry_run=False)


def _execute_qc_browse_copy(config: dict) -> dict:
    inputs = typed_inputs(qc_browse_stage.QcBrowseInputs, config)
    return qc_browse_stage.run_qc_browse_copy(
        inputs,
        ArtifactRef(**config["browse_plan"]),
        query=None,
        query_factory=None,
    )


def _execute_snapshot_publication(config: dict) -> dict:
    inputs = typed_inputs(snapshot_publication_stage.SnapshotPublicationInputs, config)
    return snapshot_publication_stage.run_snapshot_publication(
        inputs,
        check_only=False,
        repo_root=REPO_ROOT,
        prior_declarations_uri=config["prior_declarations_uri"],
    )


def _execute_contract_census(config: dict) -> dict:
    inputs = typed_inputs(contract_census_stage.ContractCensusInputs, config)
    return contract_census_stage.run_contract_census(inputs, dry_run=False)


def _execute_catalog_stage(config: dict) -> dict:
    inputs = typed_inputs(catalog_stage_stage.CatalogStageInputs, config)
    return catalog_stage_stage.run_catalog_stage(inputs, dry_run=False)


def _execute_catalog_load(config: dict) -> dict:
    inputs = typed_inputs(catalog_load_stage.CatalogLoadInputs, config)
    return catalog_load_stage.run_catalog_load(inputs, dry_run=False)


def _execute_curriculum_load(config: dict) -> dict:
    inputs = typed_inputs(curriculum_load_stage.CurriculumLoadInputs, config)
    return curriculum_load_stage.run_curriculum_load(inputs, dry_run=False)


def _execute_strict_audit(config: dict) -> dict:
    inputs = typed_inputs(strict_audit_stage.StrictAuditInputs, config)
    return strict_audit_stage.run_strict_audit(inputs, dry_run=False)


def _execute_package_preview(config: dict) -> dict:
    inputs = typed_inputs(package_preview_stage.PackagePreviewInputs, config)
    return package_preview_stage.run_package_preview(inputs, dry_run=False, repo_root=REPO_ROOT)


def _execute_release_check(config: dict) -> dict:
    inputs = typed_inputs(release_check_stage.ReleaseCheckInputs, config)
    return release_check_stage.run_release_check(inputs, dry_run=False, repo_root=REPO_ROOT)


def _execute_marina_release(config: dict) -> dict:
    inputs = typed_inputs(marina_release_stage.MarinaReleaseInputs, config)
    return marina_release_stage.run_marina_release(inputs, dry_run=False, repo_root=REPO_ROOT)


def _execute_hosted_check(config: dict) -> dict:
    inputs = typed_inputs(hosted_check_stage.HostedCheckInputs, config)
    return hosted_check_stage.run_hosted_check(inputs, dry_run=False)


def _execute_exposure_probe(config: dict) -> dict:
    inputs = typed_inputs(exposure_probe_stage.ExposureProbeInputs, config)
    return exposure_probe_stage.run_exposure_probe(inputs, dry_run=False)


def _execute_main_source_sync(config: dict) -> dict:
    inputs = typed_inputs(main_source_sync_stage.MainSourceSyncInputs, config)
    return main_source_sync_stage.run_main_source_sync(inputs, dry_run=False)


def _execute_main_grader_drift(config: dict) -> dict:
    inputs = typed_inputs(main_grader_drift_stage.GraderDriftInputs, config)
    return main_grader_drift_stage.run_main_grader_drift(inputs, dry_run=False)


_STAGE_EXECUTORS: dict[str, Callable[[dict], dict]] = {
    "census_inputs": _execute_census_inputs,
    "compose_manifest": _execute_compose_manifest,
    "declarations": _execute_declarations,
    "admission_check": _execute_admission_check,
    "census": _execute_census,
    "corpus_assembly": _execute_corpus_assembly,
    "embed": _execute_embed,
    "curriculum_compose": _execute_curriculum_compose,
    "identity_backend_compose": _execute_identity_backend_compose,
    "backend_validation": _execute_backend_validation,
    "fresh_qc_index": _execute_fresh_qc_index,
    "history_staging": _execute_history_staging,
    "text_scan_region": _execute_text_scan,
    "text_scans": _execute_text_scan_merge,
    "qc_browse_plan": _execute_qc_browse_plan,
    "qc_browse": _execute_qc_browse_copy,
    "snapshot_publication": _execute_snapshot_publication,
    "contract_census": _execute_contract_census,
    "catalog_stage": _execute_catalog_stage,
    "catalog_load": _execute_catalog_load,
    "curriculum_load": _execute_curriculum_load,
    "strict_audit": _execute_strict_audit,
    "package_preview": _execute_package_preview,
    "release_check": _execute_release_check,
    "marina_release": _execute_marina_release,
    "hosted_check": _execute_hosted_check,
    "exposure_probe": _execute_exposure_probe,
    "main_source_sync": _execute_main_source_sync,
    "main_grader_drift": _execute_main_grader_drift,
}


def _execute_stage(stage: str, config: dict) -> dict:
    """Run one stage's write-mode entry point; the local and remote dispatch boundary."""
    executor = _STAGE_EXECUTORS.get(stage)
    if executor is None:
        raise ValueError(f"no write-mode executor for graph node {stage!r}")
    return executor(config)


# Iris children do not inherit the parent environment: the dispatched stage jobs
# receive the run's attestation and the credentials explicitly (the workspace rides
# by bundle inheritance).
# The uv extra carrying the CPU torch wheels the Harrier embedder imports.
EMBED_CPU_DEPENDENCY_GROUP = "cpu"
ADC_SECRET_NAME = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

STAGE_CHILD_ENV_KEYS = (
    "ATLAS_RELEASE_SOURCE_REVISION",
    "ATLAS_PUSH_LOG",
    "ATLAS_WATCHDOG_FEED",
    "CW_KEY_ID",
    "CW_KEY_SECRET",
    # The pinned Atlas registry archive is a private HF dataset; a node's shared HF
    # cache hides a missing token only until a pin moves to an uncached revision.
    "HF_TOKEN",
    "MARIN_IAP_CREDENTIAL_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _stage_child_env() -> dict[str, str]:
    """The parent environment a dispatched stage job carries."""
    return {key: value for key in STAGE_CHILD_ENV_KEYS if (value := os.environ.get(key))}


def materialize_job_credentials() -> None:
    """Configure the job's mounted ADC and submitter-passed Marin IAP record.

    Iris children inherit the workspace bundle but neither the parent environment nor
    its filesystem; the record travels as MARIN_IAP_CREDENTIAL_JSON and the marina
    client reads it from the credentials directory.
    """
    adc = secret_path(ADC_SECRET_NAME)
    if adc.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc)
    raw = os.environ.get("MARIN_IAP_CREDENTIAL_JSON")
    if not raw:
        return
    target = Path.home() / ".config/marin/credentials/marin.json"
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw)


def _execute_stage_node(node: str, stage: str, receipt_key: str, config: dict) -> None:
    """Worker-side dispatch for a cluster stage: run it, then seal its graph receipt."""
    materialize_job_credentials()
    proof = _execute_stage(stage, config)
    document = {
        "kind": GRAPH_RECEIPT_KIND,
        "node": node,
        "release_id": proof["release_id"],
        "receipt": _receipt_ref(proof, receipt_key).model_dump(mode="json"),
        "proof": proof,
    }
    _seal_document(f"{config['graph_receipt_root']}/{node}.json", document)


# ---------------------------------------------------------------------------
# Sealed graph documents (the configuration artifacts the stages consume)
# ---------------------------------------------------------------------------


def _seal_document(path: str, document: dict) -> dict:
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed graph document {path} exists with different content; use a new release prefix")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"sealed graph document readback mismatch at {path}")
    return {"uri": str(target), "sha256": hashlib.sha256(raw).hexdigest()}


def _seal_census_inputs(
    declarations: dict, populations: dict[str, str], release_root: str, source_revision: str
) -> dict:
    """Seal the census-inputs document: one row per declared source with its frozen population."""
    missing = sorted(set(declarations) - set(populations))
    if missing:
        raise ValueError(f"the plan pins no population for declared sources: {missing[:5]}")
    extra = sorted(set(populations) - set(declarations))
    if extra:
        raise ValueError(f"the plan pins populations outside the declared sources: {extra[:5]}")
    rows = [
        {"source_id": source_id, "declaration": declarations[source_id], "population_path": populations[source_id]}
        for source_id in sorted(declarations)
    ]
    document = {
        "kind": census_stage.INPUTS_KIND,
        "source_revision": source_revision,
        "rows": rows,
    }
    ref = _seal_document(f"{release_root}/census-inputs.json", document)
    return {"kind": "census-inputs", "release_root": release_root, **ref, "source_count": len(rows)}


def _seal_compose_manifest(
    declarations: dict,
    contract_receipts: dict[str, DerivedContractReceipt],
    targets: tuple[str, ...],
    prior_release: dict,
    prior_coverage: dict,
    fresh_release: dict,
    fresh_coverage: dict,
    rebound_source: str,
    rebound_tasks: int,
    release_root: str,
    removed_sources: tuple[str, ...] = (),
) -> dict:
    """Seal the composition manifest: prior + fresh alignment pins and the target declarations."""
    missing = sorted(set(targets) - set(declarations))
    if missing:
        raise ValueError(f"compose targets missing from the sealed declarations: {missing[:5]}")
    retained = read_ref(ArtifactRef.model_validate(prior_release))
    inventory = ({row["id"] for row in retained["inventory"]} | set(targets)) - set(removed_sources)
    if inventory != set(declarations):
        raise ValueError(
            "composition prior inventory does not conserve the frozen declarations:"
            f" missing={sorted(set(declarations) - inventory)}, extra={sorted(inventory - set(declarations))}"
        )
    for source_id in targets:
        receipt = contract_receipts.get(source_id)
        binding = declarations[source_id]["actor_census_input"]
        if (
            receipt is None
            or receipt.input.model_dump(mode="json") != binding
            or receipt.scanned_task_specs_sha256 != binding["task_specs_sha256"]
        ):
            raise ValueError(f"composition contract does not bind the frozen declaration: {source_id}")
    manifest_root = f"{release_root}/compose/{COMPOSE_MANIFEST_VERSION_SUFFIX}"
    target_declarations_ref = _seal_document(
        f"{manifest_root}/target-declarations.json",
        {source_id: declarations[source_id] for source_id in sorted(targets)},
    )
    target_contracts_ref = _seal_document(
        f"{manifest_root}/target-contracts.json",
        {source_id: contract_receipts[source_id].model_dump(mode="json") for source_id in sorted(targets)},
    )
    rebound = None if not rebound_source else {"source_id": rebound_source, "tasks": rebound_tasks}
    document = {
        "kind": "atlas-curriculum-composition-manifest",
        "old_release": dict(prior_release),
        "old_coverage": dict(prior_coverage),
        "fresh_release": dict(fresh_release),
        "fresh_coverage": dict(fresh_coverage),
        "current_target_declarations": target_declarations_ref,
        "target_contracts": target_contracts_ref,
        "rebound": rebound,
    }
    ref = _seal_document(f"{manifest_root}/composition-manifest.json", document)
    return {"kind": "compose-manifest", "release_root": release_root, **ref, "targets": list(targets)}


def _plan_document_step(plan: ReleasePlan) -> ArtifactStep[PlanDocument]:
    document = _plan_document_data(plan)

    def build_config(_ctx: StepContext) -> dict:
        return {"plan": document}

    def run(config: dict) -> PlanDocument:
        ref = write_plan(ReleasePlan.model_validate(config["plan"]), plan.release_root)
        return PlanDocument(uri=ref["uri"], sha256=ref["sha256"])

    step = ArtifactStep(
        name=f"atlas-release-{plan.release_id}-plan",
        version=plan.artifact_version,
        artifact_type=PlanDocument,
        run=run,
        build_config=build_config,
        deps=(),
    )
    return dataclasses.replace(step, expected_fingerprint=step.fingerprint())


def _census_inputs_step(plan: ReleasePlan, declarations: ArtifactStep) -> ArtifactStep[CensusInputsDocument]:
    config_identity = {
        "populations": plan.populations,
        "release_root": plan.release_root,
        "source_revision": plan.source_revision,
        "recut": plan.recut,
        "targets": list(plan.targets),
    }

    def build_config(ctx: StepContext) -> dict:
        if ctx.is_fingerprint:
            return {"node": "census_inputs", **config_identity}
        return {**config_identity, "declarations": _resolved_receipt(ctx, declarations)}

    def run(config: dict) -> CensusInputsDocument:
        proof = _execute_stage("census_inputs", config)
        return CensusInputsDocument(uri=proof["uri"], sha256=proof["sha256"], source_count=proof["source_count"])

    step = ArtifactStep(
        name=f"atlas-release-{plan.release_id}-census-inputs",
        version=plan.artifact_version,
        artifact_type=CensusInputsDocument,
        run=run,
        build_config=build_config,
        deps=(declarations,),
    )
    return dataclasses.replace(step, expected_fingerprint=step.fingerprint())


def _compose_manifest_step(
    plan: ReleasePlan, declarations: ArtifactStep, embed: ArtifactStep, contracts: ArtifactStep
) -> ArtifactStep[ComposeManifestDocument]:
    config_identity = {
        "targets": list(plan.targets),
        "prior_release": _ref_dict(plan.pins.prior_release),
        "prior_coverage": _ref_dict(plan.pins.prior_coverage),
        "prior_curriculum": (
            {
                "load_id": plan.prior_curriculum_load_id,
                "service_url": plan.service.service_url,
                "applet_id": plan.service.applet_id,
            }
            if plan.recut
            else None
        ),
        "removed_sources": list(plan.removed_sources),
        "rebound_source": plan.rebound_source,
        "rebound_tasks": plan.rebound_tasks,
        "release_root": plan.release_root,
    }

    def build_config(ctx: StepContext) -> dict:
        if ctx.is_fingerprint:
            return {"node": "compose_manifest", "contracts": ctx.artifact_path(contracts), **config_identity}
        embed_proof = _resolved_proof(ctx, embed)
        return {
            **config_identity,
            "declarations": _resolved_receipt(ctx, declarations),
            "contracts": _resolved_proof(ctx, contracts)["contracts"],
            "fresh_release": embed_proof["release_receipt"],
            "fresh_coverage": embed_proof["coverage_receipt"],
        }

    def run(config: dict) -> ComposeManifestDocument:
        proof = _execute_stage("compose_manifest", config)
        return ComposeManifestDocument(uri=proof["uri"], sha256=proof["sha256"], targets=tuple(proof["targets"]))

    step = ArtifactStep(
        name=f"atlas-release-{plan.release_id}-compose-manifest-{COMPOSE_MANIFEST_VERSION_SUFFIX}",
        version=plan.artifact_version,
        artifact_type=ComposeManifestDocument,
        run=run,
        build_config=build_config,
        deps=(declarations, embed, contracts),
    )
    return dataclasses.replace(step, expected_fingerprint=step.fingerprint())


# ---------------------------------------------------------------------------
# Stage steps
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One registered graph node: its module, executor key, edges and dispatch mode.

    ``node`` is the unique graph key (also the preflight-receipt and graph-receipt
    key); ``stage`` is the write-mode executor key in ``_STAGE_EXECUTORS`` (shared by
    phase variants such as the two audits and the three release checks).
    """

    node: str
    stage: str
    module: str
    depends_on: tuple[str, ...]
    cluster: bool
    receipt_key: str = "receipt"
    pip_packages: tuple[str, ...] = ()


_STAGE_SPECS: dict[str, StageSpec] = {
    "contract_census": StageSpec(
        "contract_census", "contract_census", contract_census_stage.__name__, ("census",), True
    ),
    "declarations": StageSpec(
        "declarations", "declarations", declarations_stage.__name__, (), False, receipt_key="declarations"
    ),
    "admission_check": StageSpec(
        "admission_check", "admission_check", admission_check_stage.__name__, ("declarations",), False
    ),
    "census": StageSpec("census", "census", census_stage.__name__, ("census_inputs",), True),
    "corpus_assembly": StageSpec(
        "corpus_assembly", "corpus_assembly", corpus_assembly_stage.__name__, ("census",), True
    ),
    "embed": StageSpec("embed", "embed", embed_stage.__name__, ("corpus_assembly",), True),
    # The compose stage seals its handoff document; there is no separate receipt.
    "curriculum_compose": StageSpec(
        "curriculum_compose",
        "curriculum_compose",
        curriculum_compose_stage.__name__,
        ("compose_manifest",),
        True,
        receipt_key="handoff",
    ),
    "identity_backend_compose": StageSpec(
        "identity_backend_compose",
        "identity_backend_compose",
        identity_backend_compose_stage.__name__,
        ("backend_validation",),
        True,
        receipt_key="documents.identity_receipt",
    ),
    "backend_validation": StageSpec(
        "backend_validation",
        "backend_validation",
        backend_validation_stage.__name__,
        ("census_inputs",),
        True,
        # the qualification's survey imports the shellbox daytona machines; the
        # package rides as a pip install (the pin follows lib/shellbox's extra)
        pip_packages=("daytona==0.200.2",),
    ),
    "fresh_qc_index": StageSpec(
        "fresh_qc_index",
        "fresh_qc_index",
        fresh_qc_index_stage.__name__,
        ("qc_browse", "declarations"),
        False,
        receipt_key="index",
    ),
    "history_staging": StageSpec("history_staging", "history_staging", history_staging_stage.__name__, (), True),
    "text_scan_region": StageSpec(
        "text_scan_region", "text_scan_region", text_scans_stage.__name__, ("curriculum_compose",), True
    ),
    "text_scans": StageSpec(
        "text_scans",
        "text_scans",
        text_scans_stage.__name__,
        ("curriculum_compose", "text_scan_region[*]"),
        False,
    ),
    "qc_browse_plan": StageSpec(
        "qc_browse_plan", "qc_browse_plan", qc_browse_stage.__name__, ("declarations",), True, receipt_key="plan_ref"
    ),
    "qc_browse": StageSpec(
        "qc_browse", "qc_browse", qc_browse_stage.__name__, ("qc_browse_plan",), True, receipt_key="completion"
    ),
    "snapshot_publication": StageSpec(
        "snapshot_publication",
        "snapshot_publication",
        snapshot_publication_stage.__name__,
        ("census", "identity_backend_compose", "admission_check"),
        False,
    ),
    "catalog_stage": StageSpec(
        "catalog_stage",
        "catalog_stage",
        catalog_stage_stage.__name__,
        ("declarations", "snapshot_publication", "history_staging", "qc_browse"),
        False,
    ),
    "catalog_load": StageSpec("catalog_load", "catalog_load", catalog_load_stage.__name__, ("catalog_stage",), True),
    "curriculum_load": StageSpec(
        "curriculum_load",
        "curriculum_load",
        curriculum_load_stage.__name__,
        ("curriculum_compose", "text_scans"),
        True,
    ),
    "strict_audit_before": StageSpec(
        "strict_audit_before",
        "strict_audit",
        strict_audit_stage.__name__,
        ("census", "catalog_load", "curriculum_load"),
        False,
    ),
    "strict_audit_after": StageSpec(
        "strict_audit_after", "strict_audit", strict_audit_stage.__name__, ("census", "marina_release"), False
    ),
    "package_preview": StageSpec(
        "package_preview", "package_preview", package_preview_stage.__name__, ("strict_audit_before", "census"), False
    ),
    "release_check_preflight": StageSpec(
        "release_check_preflight", "release_check", release_check_stage.__name__, ("package_preview",), False
    ),
    "release_check_hosted": StageSpec(
        "release_check_hosted",
        "release_check",
        release_check_stage.__name__,
        ("release_check_preflight", "marina_release"),
        False,
    ),
    "marina_release": StageSpec(
        "marina_release",
        "marina_release",
        marina_release_stage.__name__,
        ("package_preview", "release_check_preflight"),
        False,
        receipt_key="manifest",
    ),
    "hosted_check": StageSpec(
        "hosted_check",
        "hosted_check",
        hosted_check_stage.__name__,
        ("declarations", "identity_backend_compose", "census", "marina_release"),
        False,
    ),
    "release_check_postpublish": StageSpec(
        "release_check_postpublish",
        "release_check",
        release_check_stage.__name__,
        ("release_check_preflight", "release_check_hosted", "marina_release"),
        False,
    ),
    "exposure_probe": StageSpec(
        "exposure_probe",
        "exposure_probe",
        exposure_probe_stage.__name__,
        ("declarations", "marina_release", "hosted_check", "release_check_postpublish", "strict_audit_after"),
        False,
    ),
    # The main-source sync is a plan-gated leaf: it never joins another stage's
    # dependency edges, so plans that opt out keep every existing fingerprint. A
    # plan opts in through `main_source_sync`; without it the node refuses to run.
    "main_source_sync": StageSpec(
        "main_source_sync",
        "main_source_sync",
        main_source_sync_stage.__name__,
        ("declarations",),
        False,
        receipt_key="report",
    ),
    # The grader-drift leaf shares the main-source-sync shape: every plan carries
    # the node, only a plan whose derivation pinned a grader_drift_reference can
    # run it, nothing depends on it, and it is advisory — its findings never gate.
    "main_grader_drift": StageSpec(
        "main_grader_drift",
        "main_grader_drift",
        main_grader_drift_stage.__name__,
        ("declarations",),
        False,
    ),
}


@dataclasses.dataclass(frozen=True)
class PreflightPolicy:
    """Per-stage preflight gating wired into the graph's stage steps.

    ``ensure`` is called with the node key immediately before a stage executes
    (driver-local or cluster): it must guarantee a fresh preflight receipt for the
    node — running the small-slice runner when needed — or raise to refuse the
    stage before any submission (DESIGN.md §5-§6).
    """

    ensure: Callable[[str], None]


def _resources_config(resources: StageResources) -> ResourceConfig:
    return ResourceConfig(cpu=resources.cpu, ram=f"{resources.memory_gb}g", disk=f"{resources.disk_gb}g")


def _stage_step(
    plan: ReleasePlan,
    node: str,
    spec: StageSpec,
    deps: tuple[ArtifactStep, ...],
    *,
    build_inputs: Callable[[StepContext], dict],
    identity: Callable[[], dict],
    inputs_type: type,
    resources: Callable[[], ResourceConfig] | None = None,
    dependency_groups: list[str] | None = None,
    receipt_key: str | None = None,
    name_suffix: str = "",
) -> ArtifactStep[StageReceipt]:
    """One stage as a lazy step: typed inputs in, sealed :class:`StageReceipt` out.

    ``node`` is the graph key (unique, used for the graph-receipt document);
    ``spec.stage`` is the executor key (shared by phase variants). ``receipt_key``
    overrides the spec's sealed-reference key; it defaults to the spec's own so a
    custom spec key cannot drift out of the cluster dispatch. ``dependency_groups``
    names the uv extras the dispatched job (and the Zephyr workers it spawns, which
    reproduce its setup) installs; ``None`` infers them from the resources' device.
    """
    graph_receipt_root = f"{plan.release_root}/graph"
    receipt_key = spec.receipt_key if receipt_key is None else receipt_key
    dispatch_resources = resources() if resources is not None else None

    def build_config(ctx: StepContext) -> dict:
        if ctx.is_fingerprint:
            # The receipt key is the step's output contract: the sealed reference the
            # consumers read. A contract change must invalidate the stored record.
            return {"node": node, "stage": spec.stage, "receipt_key": receipt_key, **identity()}
        return {**build_inputs(ctx), "graph_receipt_root": graph_receipt_root}

    def run(config: dict) -> StageReceipt:
        path = f"{graph_receipt_root}/{node}.json"
        if spec.cluster:
            if not StoragePath(path).exists():
                if dispatch_resources is None:
                    raise ValueError(f"cluster stage {node!r} has no sizing-derived resources")
                dispatch = remote(
                    partial(_execute_stage_node, node, spec.stage, receipt_key),
                    resources=dispatch_resources,
                    env_vars=_stage_child_env(),
                    pip_dependency_groups=dependency_groups,
                    pip_packages=list(spec.pip_packages) or None,
                )
                dispatch(config)
            document = json.loads(StoragePath(path).read_text())
            if document["kind"] != GRAPH_RECEIPT_KIND or document["node"] != node:
                raise ValueError(f"graph receipt at {path} does not bind node {node!r}")
            return StageReceipt(
                stage=node,
                release_id=document["release_id"],
                receipt=ArtifactRef(**document["receipt"]),
                proof=document["proof"],
            )
        return _stage_receipt(node, _execute_stage(spec.stage, config), receipt_key)

    step = ArtifactStep(
        name=f"atlas-release-{plan.release_id}-{node}",
        version=plan.artifact_version,
        artifact_type=StageReceipt,
        run=run,
        build_config=build_config,
        deps=deps,
    )
    return dataclasses.replace(step, expected_fingerprint=step.fingerprint())


def _identity_fields(plan: ReleasePlan, *names: str) -> Callable[[], dict]:
    """An identity payload over the named plan fields, canonicalized to plain JSON."""

    def identity() -> dict:
        fields: dict[str, Any] = {"release_root": plan.release_root}
        for name in names:
            value = getattr(plan, name)
            if isinstance(value, ArtifactRef):
                fields[name] = _ref_dict(value)
            elif isinstance(value, BaseModel):
                fields[name] = value.model_dump(mode="json")
            else:
                fields[name] = value
        return fields

    return identity


def _resolved_receipt(ctx: StepContext, handle: ArtifactStep) -> dict:
    return ctx.resolved(handle).receipt.model_dump(mode="json")


def _resolved_proof(ctx: StepContext, handle: ArtifactStep) -> dict:
    return ctx.resolved(handle).proof


# --- declarations and the admission gate --------------------------------------


def _main_source_sync_step(plan: ReleasePlan, declarations: ArtifactStep) -> ArtifactStep[StageReceipt]:
    """The plan-gated main-source sync leaf: report main's new sources, queue them.

    Every plan carries the node so the graph shape is stable, but only a plan that
    declares ``main_source_sync`` can execute it: the inputs carry no pinned main
    revision otherwise, and the stage refuses. It depends on nothing but the sealed
    declarations and nothing depends on it, so opting in or out never moves another
    stage's fingerprint.
    """
    spec = _STAGE_SPECS["main_source_sync"]
    identity = _identity_fields(plan, "release_id", "source_revision", "main_source_sync")

    def build_inputs(ctx: StepContext) -> dict:
        if plan.main_source_sync is None:
            raise ValueError("plan declares no main_source_sync opt-in; the main-source sync stage cannot run")
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "main_source_revision": plan.main_source_sync.main_source_revision,
            "declarations": _resolved_receipt(ctx, declarations),
            "report_root": stage_output_root(plan, "main-source-sync"),
            "queue_path": plan.main_source_sync.queue_path,
        }

    return _stage_step(
        plan,
        "main_source_sync",
        spec,
        (declarations,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=main_source_sync_stage.MainSourceSyncInputs,
    )


def _main_grader_drift_step(plan: ReleasePlan, declarations: ArtifactStep) -> ArtifactStep[StageReceipt]:
    """The plan-gated grader-drift leaf: the advisory stale-grader queue.

    Every plan carries the node so the graph shape is stable, but only a plan
    whose derivation pinned a ``grader_drift_reference`` can execute it. It
    depends on nothing but the sealed declarations and nothing depends on it, so
    opting in or out never moves another stage's fingerprint, and the release
    tail never waits on an advisory instrument.
    """
    spec = _STAGE_SPECS["main_grader_drift"]
    identity = _identity_fields(plan, "release_id", "source_revision", "grader_drift_reference")

    def build_inputs(ctx: StepContext) -> dict:
        if plan.grader_drift_reference is None:
            raise ValueError(
                "plan pins no grader_drift_reference; the main-grader-drift stage cannot run"
                " (re-derive the plan at a freeze whose checkout reaches origin/main)"
            )
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "declarations": _resolved_receipt(ctx, declarations),
            "reference": plan.grader_drift_reference,
            "output_root": stage_output_root(plan, "grader-drift"),
        }

    return _stage_step(
        plan,
        "main_grader_drift",
        spec,
        (declarations,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=main_grader_drift_stage.GraderDriftInputs,
    )


def _declarations_step(plan: ReleasePlan, plan_handle: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["declarations"]
    identity = _identity_fields(plan, "release_id", "source_revision", "expected_source_count", "targets", "pins")

    def build_inputs(_ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "output_root": stage_output_root(plan, "declarations"),
            "expected_source_count": plan.expected_source_count,
            "targets": plan.targets,
            "prior_declarations": _ref_dict(plan.pins.prior_declarations),
            "removed_sources": plan.removed_sources,
            "deliberate_exclusions": plan.deliberate_exclusions,
        }

    return _stage_step(
        plan,
        "declarations",
        spec,
        (plan_handle,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=declarations_stage.DeclarationInputs,
    )


def _admission_check_step(
    plan: ReleasePlan, declarations: ArtifactStep, identity_backend: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["admission_check"]
    identity = _identity_fields(
        plan, "release_id", "source_revision", "snapshot_id", "catalog_root", "expected_source_count", "service"
    )

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "declarations": _resolved_receipt(ctx, declarations),
            "backend": _resolved_proof(ctx, identity_backend)["documents"]["backend_map"],
            "snapshot_id": plan.snapshot_id,
            "catalog_root": plan.catalog_root,
            "prior_load_id": plan.prior.load_id,
            "output_root": stage_output_root(plan, "admission-check"),
            "expected_source_count": plan.expected_source_count,
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
        }

    return _stage_step(
        plan,
        "admission_check",
        spec,
        (declarations, identity_backend),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=admission_check_stage.AdmissionCheckInputs,
    )


# --- census through curriculum compose ----------------------------------------


def _census_step(plan: ReleasePlan, census_inputs: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["census"]
    identity = _identity_fields(plan, "release_id", "source_revision", "region_prefix", "expected_source_count")

    def build_inputs(ctx: StepContext) -> dict:
        inputs_doc = ctx.resolved(census_inputs)
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "inputs": {"uri": inputs_doc.uri, "sha256": inputs_doc.sha256},
            "output_root": stage_output_root(plan, "census"),
            "region_prefix": plan.region_prefix,
            # A re-cut's census is incremental (the new admissions only); the
            # census-inputs document's own row count is the expected source count.
            "expected_sources": inputs_doc.source_count,
            # The per-source fan-out rides the job's measured core count.
            "workers": census_sizing(plan.expected_population_tasks).cpu,
            "slice_output_root": plan.slice_output_root or None,
        }

    return _stage_step(
        plan,
        "census",
        spec,
        (census_inputs,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=census_stage.CensusInputs,
        resources=lambda: _resources_config(census_sizing(plan.expected_population_tasks)),
    )


def _corpus_assembly_step(
    plan: ReleasePlan, census_inputs: ArtifactStep, census: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["corpus_assembly"]
    identity = _identity_fields(plan, "release_id", "source_revision", "expected_source_count", "targets")
    sizing = corpus_sizing(plan.expected_population_tasks)

    def build_inputs(ctx: StepContext) -> dict:
        inputs_doc = ctx.resolved(census_inputs)
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "inputs": {"uri": inputs_doc.uri, "sha256": inputs_doc.sha256},
            "census_receipt": _resolved_receipt(ctx, census),
            "output_root": stage_output_root(plan, "corpus"),
            # A re-cut's census-inputs document is scoped to the new admissions, and
            # the re-cut census receipt carries exactly those results — the expected
            # count flows from the document, not the release's full source count.
            "expected_sources": inputs_doc.source_count,
            "targets": plan.targets,
            "workers": sizing.workers,
            "slice_output_root": plan.slice_output_root or None,
        }

    return _stage_step(
        plan,
        "corpus_assembly",
        spec,
        (census_inputs, census),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=corpus_assembly_stage.CorpusAssemblyInputs,
        resources=lambda: _resources_config(sizing),
    )


def _fresh_corpus_expectation(plan: ReleasePlan, census_proof: dict) -> dict:
    """The shipped fresh corpus: the targets the census did not oracle-filter to empty."""
    results = {row["source_id"]: row for row in census_proof["results"]}
    missing = sorted(set(plan.targets) - set(results))
    if missing:
        raise ValueError(f"census results carry no rows for targets: {missing[:5]}")
    shipped = [results[source_id] for source_id in plan.targets if results[source_id].get("oracle_empty") is None]
    return {
        "expected_source_count": len(shipped),
        "expected_task_count": sum(row["inventory"]["retained_tasks"] for row in shipped),
    }


def _embed_step(plan: ReleasePlan, corpus: ArtifactStep, census: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["embed"]
    identity = _identity_fields(
        plan, "release_id", "source_revision", "expected_source_count", "expected_population_tasks", "gpu_runtime"
    )

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "corpus": _resolved_proof(ctx, corpus)["corpus"],
            "artifact_root": stage_output_root(plan, "embed"),
            "cache_root": plan.embed_cache_root,
            **_fresh_corpus_expectation(plan, _resolved_proof(ctx, census)),
            "parity_receipt": _ref_dict(plan.pins.parity_receipt),
            "gpu_runtime": plan.gpu_runtime,
        }

    cuda = plan.pins.parity_receipt is not None and bool(plan.gpu_runtime)

    def resources() -> ResourceConfig:
        config = _resources_config(embedding_sizing(plan.expected_population_tasks))
        if cuda:
            return dataclasses.replace(config, device=GpuConfig(variant=plan.gpu_runtime))
        return config

    return _stage_step(
        plan,
        "embed",
        spec,
        (corpus, census),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=embed_stage.EmbedInputs,
        resources=resources,
        # The Harrier embedder needs torch: the CPU wheel group on CPU, the inferred
        # GPU group on a CUDA run.
        dependency_groups=None if cuda else [EMBED_CPU_DEPENDENCY_GROUP],
    )


def _curriculum_compose_step(
    plan: ReleasePlan, compose_manifest: ArtifactStep, census: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["curriculum_compose"]
    identity = _identity_fields(
        plan, "release_id", "source_revision", "targets", "removed_sources", "expected_population_tasks"
    )

    def build_inputs(ctx: StepContext) -> dict:
        # The same shipped fresh set the embed stage expects (an oracle-empty source
        # carries no census inventory).
        fresh = _fresh_corpus_expectation(plan, _resolved_proof(ctx, census))
        inputs_doc = ctx.resolved(compose_manifest)
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "inputs": {"uri": inputs_doc.uri, "sha256": inputs_doc.sha256},
            "output_root": stage_output_root(plan, "compose"),
            "expected_fresh_task_count": fresh["expected_task_count"],
            "removed_sources": plan.removed_sources,
        }

    return _stage_step(
        plan,
        "curriculum_compose",
        spec,
        (compose_manifest, census),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=curriculum_compose_stage.CurriculumComposeInputs,
        resources=lambda: _resources_config(census_sizing(plan.expected_population_tasks)),
    )


# --- verification and SQL stages ----------------------------------------------


def _sql_stage_resources() -> ResourceConfig:
    """The shared job sizing for the paced SQL stages (history staging, loads, QC copy)."""
    return _resources_config(StageResources(cpu=4, memory_gb=16, disk_gb=64, timeout_s=14_400))


def _backend_validation_step(plan: ReleasePlan, census_inputs: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["backend_validation"]
    identity = _identity_fields(
        plan,
        "release_id",
        "source_revision",
        "snapshot_id",
        "catalog_root",
        "targets",
        "validation_grading_environments",
        "validation_integrity_contracts",
        "validation_runtime_defaults",
    )

    def build_inputs(ctx: StepContext) -> dict:
        census_document = ctx.resolved(census_inputs)
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "snapshot_id": plan.snapshot_id,
            "catalog_root": plan.catalog_root,
            "output_root": stage_output_root(plan, "backend-validation"),
            "targets": plan.targets,
            "census_inputs": {"uri": census_document.uri, "sha256": census_document.sha256},
            "grading_environments": plan.validation_grading_environments,
            "integrity_contracts": plan.validation_integrity_contracts,
            "runtime_defaults": plan.validation_runtime_defaults,
        }

    return _stage_step(
        plan,
        "backend_validation",
        spec,
        (census_inputs,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=backend_validation_stage.BackendValidationInputs,
        resources=_sql_stage_resources,
    )


def _identity_backend_compose_step(plan: ReleasePlan, backend_validation: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["identity_backend_compose"]
    identity = _identity_fields(plan, "release_id", "snapshot_id", "source_revision", "targets", "expected_source_count")

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": plan.source_revision,
            "output_root": stage_output_root(plan, "identity-backend"),
            "targets": plan.targets,
            "removed_sources": plan.removed_sources,
            "prior_identity": _ref_dict(plan.pins.prior_identity),
            "prior_backend": _ref_dict(plan.pins.prior_backend),
            "fresh_backend_receipt": _resolved_receipt(ctx, backend_validation),
            "registry_index": _ref_dict(plan.pins.registry_index),
            "expected_target_count": len(plan.targets),
        }

    return _stage_step(
        plan,
        "identity_backend_compose",
        spec,
        (backend_validation,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=identity_backend_compose_stage.IdentityBackendComposeInputs,
        resources=_sql_stage_resources,
    )


def _history_staging_step(plan: ReleasePlan) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["history_staging"]
    identity = _identity_fields(
        plan,
        "release_id",
        "prior",
        "catalog_target_load_id",
        "snapshot_id",
        "source_revision",
        "catalog_root",
        "expected_prior_sources",
    )

    def build_inputs(_ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "prior_load_id": plan.prior.load_id,
            "target_load_id": plan.catalog_target_load_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": plan.source_revision,
            "catalog_root": plan.catalog_root,
            "receipt_root": plan.release_root,
            "expected_prior_sources": plan.expected_prior_sources,
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
        }

    return _stage_step(
        plan,
        "history_staging",
        spec,
        (),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=history_staging_stage.HistoryStagingInputs,
        resources=_sql_stage_resources,
    )


def _text_scan_region_step(plan: ReleasePlan, compose: ArtifactStep, region: str) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["text_scan_region"]
    identity = _identity_fields(plan, "release_id", "source_revision", "scan_regions")

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "handoff": _resolved_proof(ctx, compose)["handoff"],
            "output_root": stage_output_root(plan, "text-scans"),
            "slice_output_root": None,
            "regions": tuple(plan.scan_regions),
            "region_prefixes": dict(plan.scan_regions),
            "max_files": 0,
            "scan_region": region,
        }

    return _stage_step(
        plan,
        f"text_scan_region[{region}]",
        spec,
        (compose,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=text_scans_stage.TextScanInputs,
        resources=_sql_stage_resources,
    )


def _text_scan_merge_step(
    plan: ReleasePlan, compose: ArtifactStep, regions: tuple[ArtifactStep, ...]
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["text_scans"]
    identity = _identity_fields(plan, "release_id", "source_revision", "scan_regions")

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "handoff": _resolved_proof(ctx, compose)["handoff"],
            "output_root": stage_output_root(plan, "text-scans"),
            "slice_output_root": None,
            "regions": tuple(plan.scan_regions),
            "region_prefixes": dict(plan.scan_regions),
            "max_files": 0,
            "region_receipts": [_resolved_receipt(ctx, handle) for handle in regions],
        }

    return _stage_step(
        plan,
        "text_scans",
        spec,
        (compose, *regions),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=text_scans_stage.TextScanInputs,
    )


def _qc_browse_inputs(plan: ReleasePlan, declarations_ref: dict) -> dict:
    # The derivation measured the copy exclusions against the served state (the
    # same pair the panel expectation used); recomputing them here against a
    # different document pair would desynchronize the panel count check.
    copy_exclusions = set(plan.panel_copy_exclusions)
    return {
        "release_id": plan.release_id,
        "prior_load_id": plan.prior.load_id,
        "target_load_id": plan.catalog_target_load_id,
        "source_revision": plan.source_revision,
        "declarations": declarations_ref,
        "output_root": stage_output_root(plan, "qc-browse"),
        "expected_panels": plan.expected_panels,
        # The copy fans panels out across paced SQL clients; the plan step stays serial.
        "workers": qc_browse_sizing(plan.expected_browse_source_count).workers,
        "exclusions": tuple(sorted(copy_exclusions)),
        "service_url": plan.service.service_url,
        "applet_id": plan.service.applet_id,
    }


def _qc_browse_plan_step(plan: ReleasePlan, declarations: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["qc_browse_plan"]
    identity = _identity_fields(
        plan, "release_id", "prior", "catalog_target_load_id", "source_revision", "expected_panels", "qc_exclusions"
    )

    def build_inputs(ctx: StepContext) -> dict:
        return _qc_browse_inputs(plan, _resolved_receipt(ctx, declarations))

    return _stage_step(
        plan,
        "qc_browse_plan",
        spec,
        (declarations,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=qc_browse_stage.QcBrowseInputs,
        resources=lambda: _resources_config(StageResources(cpu=2, memory_gb=8, disk_gb=32, timeout_s=14_400)),
    )


def _qc_browse_copy_step(plan: ReleasePlan, browse_plan: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["qc_browse"]
    identity = _identity_fields(
        plan, "release_id", "prior", "catalog_target_load_id", "source_revision", "expected_panels", "qc_exclusions"
    )

    def build_inputs(ctx: StepContext) -> dict:
        plan_proof = _resolved_proof(ctx, browse_plan)
        return {
            **_qc_browse_inputs(plan, plan_proof["plan"]["declarations_ref"]),
            "browse_plan": plan_proof["plan_ref"],
        }

    return _stage_step(
        plan,
        "qc_browse",
        spec,
        (browse_plan,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=qc_browse_stage.QcBrowseInputs,
        resources=lambda: _resources_config(qc_browse_sizing(plan.expected_browse_source_count)),
        receipt_key="completion",
    )


# --- snapshot publication through the loads ------------------------------------


def _snapshot_publication_step(
    plan: ReleasePlan,
    census: ArtifactStep,
    identity_backend: ArtifactStep,
    admission_check: ArtifactStep,
    contracts: ArtifactStep,
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["snapshot_publication"]
    identity = _identity_fields(
        plan,
        "release_id",
        "snapshot_id",
        "prior_snapshot_id",
        "source_revision",
        "catalog_root",
        "targets",
        "removed_sources",
        "expected_source_count",
        "expected_contract_count",
        "expected_contract_tasks",
    )

    def build_inputs(ctx: StepContext) -> dict:
        documents = _resolved_proof(ctx, identity_backend)["documents"]
        return {
            "release_id": plan.release_id,
            "snapshot_id": plan.snapshot_id,
            "prior_snapshot_id": plan.prior_snapshot_id,
            "source_revision": plan.source_revision,
            "catalog_root": plan.catalog_root,
            "release_root": plan.release_root,
            "targets": plan.targets,
            "deliberate_exclusions": plan.deliberate_exclusions,
            "removed_sources": plan.removed_sources,
            "expected_source_count": plan.expected_source_count,
            "expected_contract_count": plan.expected_contract_count,
            "expected_contract_tasks": _resolved_proof(ctx, contracts)["contract_tasks"],
            "contracts": _resolved_proof(ctx, contracts)["contracts"],
            "census_receipt": _resolved_receipt(ctx, census),
            "identity_receipt": documents["identity_survey"],
            "backend_receipt": documents["backend_map"],
            "prior_declarations_uri": plan.pins.prior_declarations.uri,
        }

    return _stage_step(
        plan,
        "snapshot_publication",
        spec,
        (census, identity_backend, admission_check, contracts),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=snapshot_publication_stage.SnapshotPublicationInputs,
    )


def _catalog_stage_step(
    plan: ReleasePlan,
    declarations: ArtifactStep,
    snapshot: ArtifactStep,
    history: ArtifactStep,
    qc_browse: ArtifactStep,
    fresh_qc_index: ArtifactStep,
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["catalog_stage"]
    identity = _identity_fields(
        plan,
        "release_id",
        "prior",
        "catalog_target_load_id",
        "snapshot_id",
        "source_revision",
        "catalog_root",
        "expected_source_count",
        "expected_browse_source_count",
        "expected_contract_count",
        "expected_contract_tasks",
        "migrations",
    )

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "prior_load_id": plan.prior.load_id,
            "target_load_id": plan.catalog_target_load_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": plan.source_revision,
            "catalog_root": plan.catalog_root,
            "output_root": stage_output_root(plan, "catalog-stage"),
            "expected_source_count": plan.expected_source_count,
            "expected_browse_source_count": plan.expected_browse_source_count,
            "expected_contract_count": plan.expected_contract_count,
            "expected_contract_tasks": _resolved_proof(ctx, snapshot)["contract_tasks"],
            "declarations": _resolved_receipt(ctx, declarations),
            "publication": _resolved_receipt(ctx, snapshot),
            "history_receipt": _resolved_receipt(ctx, history),
            "prior_qc_completion": _resolved_receipt(ctx, qc_browse),
            "fresh_qc_index": _resolved_receipt(ctx, fresh_qc_index),
            "contracts": _resolved_proof(ctx, snapshot)["contracts"],
            "migrations": plan.migrations,
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
        }

    return _stage_step(
        plan,
        "catalog_stage",
        spec,
        (declarations, snapshot, history, qc_browse, fresh_qc_index),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=catalog_stage_stage.CatalogStageInputs,
    )


def _fresh_qc_index_step(
    plan: ReleasePlan, qc_browse: ArtifactStep, declarations: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["fresh_qc_index"]
    identity = _identity_fields(
        plan,
        "release_id",
        "prior",
        "catalog_target_load_id",
        "snapshot_id",
        "source_revision",
        "removed_sources",
    )

    def build_inputs(ctx: StepContext) -> dict:
        declarations_ref = _resolved_receipt(ctx, declarations)
        frozen = read_ref(ArtifactRef(**declarations_ref))
        carried = {row["source_id"] for row in _resolved_proof(ctx, qc_browse)["results"]}
        fresh_panels = {}
        for source_id, declaration in sorted(frozen.items()):
            if source_id in carried or source_id in plan.qc_exclusions:
                continue
            if not (declaration.get("independent_qc") or {}).get("evidence_url"):
                continue
            if content_is_withheld(declaration):
                continue
            uri = qc_browse_stage.selected_uri(declaration)
            fresh_panels[source_id] = {"uri": uri, "sha256": hashlib.sha256(StoragePath(uri).read_bytes()).hexdigest()}
        return {
            "release_id": plan.release_id,
            "prior_load_id": plan.prior.load_id,
            "target_load_id": plan.catalog_target_load_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": plan.source_revision,
            "output_root": stage_output_root(plan, "qc-index"),
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
            "declarations": declarations_ref,
            "qc_completion": _resolved_receipt(ctx, qc_browse),
            "fresh_panels": fresh_panels,
            "removed_sources": plan.removed_sources,
        }

    return _stage_step(
        plan,
        "fresh_qc_index",
        spec,
        (qc_browse, declarations),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=fresh_qc_index_stage.FreshQcIndexInputs,
    )


def measured_module_path(module: str) -> str:
    """The measured module as a file path catalog_load can digest.

    A derived plan records the repo-relative path; the hand-written R40 plan names the
    dotted module. Normalized here, once, at the load's input boundary.
    """
    if "/" in module:
        return str(REPO_ROOT / module)
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError(f"measured module {module!r} is neither a repo path nor an importable module")
    return spec.origin


def _catalog_load_step(plan: ReleasePlan, catalog_stage: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["catalog_load"]
    identity = _identity_fields(
        plan,
        "release_id",
        "prior",
        "catalog_target_load_id",
        "snapshot_id",
        "source_revision",
        "catalog_root",
        "expected_source_count",
        "expected_contract_count",
        "expected_contract_tasks",
        "expected_selected_panels",
        "expected_browse_source_count",
        "previously_available",
        "migrations",
        "measured",
        "difficulty_snapshot",
    )

    def build_inputs(ctx: StepContext) -> dict:
        staged = _resolved_proof(ctx, catalog_stage)["refs"]
        return {
            "release_id": plan.release_id,
            "prior_load_id": plan.prior.load_id,
            "target_load_id": plan.catalog_target_load_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": plan.source_revision,
            "catalog_root": plan.catalog_root,
            "receipt_root": plan.release_root,
            "publication": staged["publication"],
            "staging": {
                name: staged[name] for name in ("prior_qc_completion", "fresh_qc_index", "contracts", "history_receipt")
            },
            "expected_source_count": plan.expected_source_count,
            "expected_contract_count": plan.expected_contract_count,
            "expected_contract_tasks": _resolved_proof(ctx, catalog_stage)["contract_tasks"],
            "expected_selected_panels": plan.expected_selected_panels,
            "expected_browse_source_count": plan.expected_browse_source_count,
            "measured_module": measured_module_path(plan.measured.module),
            "measured_revision": plan.measured.revision,
            "measured_sha256": plan.measured.sha256,
            "measured_schema_sha256": plan.measured.schema_sha256,
            "previously_available": plan.previously_available,
            "migrations": plan.migrations,
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
            "difficulty_snapshot": _ref_dict(plan.difficulty_snapshot),
        }

    return _stage_step(
        plan,
        "catalog_load",
        spec,
        (catalog_stage,),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=catalog_load_stage.CatalogLoadInputs,
        resources=_sql_stage_resources,
    )


def _curriculum_load_step(
    plan: ReleasePlan, compose: ArtifactStep, text_scans: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["curriculum_load"]
    identity = _identity_fields(
        plan,
        "release_id",
        "curriculum_target_load_id",
        "prior_curriculum_load_id",
        "prior",
        "catalog_target_load_id",
        "source_revision",
        "expected_source_count",
        "rebound_source",
        "rebound_tasks",
    )

    def build_inputs(ctx: StepContext) -> dict:
        compose_proof = _resolved_proof(ctx, compose)
        scan_proof = _resolved_proof(ctx, text_scans)
        return {
            "release_id": plan.release_id,
            "prior_curriculum_load_id": plan.prior_curriculum_load_id,
            "catalog_prior_load_id": plan.prior.load_id,
            "catalog_target_load_id": plan.catalog_target_load_id,
            "source_revision": plan.source_revision,
            "receipt_root": plan.release_root,
            "handoff": compose_proof["handoff"],
            "text_scan": scan_proof["receipt"],
            "expected_source_count": plan.expected_source_count,
            "expected_inventory_count": compose_proof["inventory_count"],
            "expected_aligned_sources": compose_proof["source_count"],
            "expected_task_count": compose_proof["task_count"],
            "expected_rebound_source": plan.rebound_source,
            "expected_rebound_tasks": plan.rebound_tasks,
            "expected_confidence_rows": compose_proof["confidence_validated_rows"],
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
        }

    return _stage_step(
        plan,
        "curriculum_load",
        spec,
        (compose, text_scans),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=curriculum_load_stage.CurriculumLoadInputs,
        resources=_sql_stage_resources,
    )


# --- release tail ---------------------------------------------------------------


def _strict_audit_step(
    plan: ReleasePlan,
    declarations: ArtifactStep,
    identity_backend: ArtifactStep,
    phase: str,
    census: ArtifactStep,
    catalog_load: ArtifactStep | None = None,
    curriculum_load: ArtifactStep | None = None,
    origin_handle: ArtifactStep | None = None,
) -> ArtifactStep[StageReceipt]:
    if phase == "before":
        if origin_handle is not None:
            raise ValueError("the before-phase audit reads the live applet origin, not a published revision")
        if catalog_load is None or curriculum_load is None:
            raise ValueError("the before-phase audit runs after both loads and needs their handles")
        deps = (declarations, identity_backend, census, catalog_load, curriculum_load)
    else:
        if origin_handle is None:
            raise ValueError("the after-phase audit needs the marina release for its published origin")
        deps = (declarations, identity_backend, census, origin_handle)
    spec = _STAGE_SPECS[f"strict_audit_{phase}"]
    identity = _identity_fields(plan, "release_id", "source_revision", "expected_source_count")

    def build_inputs(ctx: StepContext) -> dict:
        if phase == "before":
            origin = plan.service.applet_origin
        else:
            publish = _resolved_proof(ctx, origin_handle)["publish"]
            origin = publish["url"]
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "origin": origin,
            "output_root": stage_output_root(plan, f"strict-audit-{phase}"),
            "local": plan.strict_audit_local,
            "declarations": _resolved_receipt(ctx, declarations),
            "backend": _resolved_proof(ctx, identity_backend)["documents"]["backend_map"],
            "identity": _resolved_proof(ctx, identity_backend)["documents"]["identity_survey"],
            "expected_source_count": plan.expected_source_count,
            # The measured census total, not the plan's hand count (R40: 4,272,894 vs 4,363,214).
            "retained_tasks": _resolved_proof(ctx, census)["tasks"],
            "phase": phase,
        }

    return _stage_step(
        plan,
        f"strict_audit_{phase}",
        spec,
        deps,
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=strict_audit_stage.StrictAuditInputs,
    )


def _package_preview_step(plan: ReleasePlan, audit: ArtifactStep, census: ArtifactStep) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["package_preview"]
    identity = _identity_fields(
        plan, "release_id", "source_revision", "pins", "package_output_root", "expected_source_count"
    )

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "output_root": plan.package_output_root,
            "census": _resolved_receipt(ctx, census),
            # Only the ruling citations carry forward from the prior counts.
            "prior_source_counts": _ref_dict(plan.pins.source_counts),
            "expected_source_count": plan.expected_source_count,
            "derived_root": stage_output_root(plan, PACKAGE_INPUTS_STAGE_DIR),
        }

    return _stage_step(
        plan,
        "package_preview",
        spec,
        (audit, census),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=package_preview_stage.PackagePreviewInputs,
    )


def _release_check_step(
    plan: ReleasePlan,
    phase: str,
    package: ArtifactStep | None,
    preflight: ArtifactStep | None,
    hosted: ArtifactStep | None,
    marina: ArtifactStep | None,
) -> ArtifactStep[StageReceipt]:
    deps = tuple(handle for handle in (package, preflight, hosted, marina) if handle is not None)
    spec = _STAGE_SPECS[f"release_check_{phase}"]
    identity = _identity_fields(
        plan, "release_id", "source_revision", "expected_source_count", "service", "package_output_root"
    )

    def build_inputs(ctx: StepContext) -> dict:
        inputs: dict[str, Any] = {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "phase": phase,
            "output_root": stage_output_root(plan, f"release-check-{phase}"),
            # The checker runs on the workstation that holds the package.
            "work_root": plan.package_output_root,
            "service_url": plan.service.service_url,
            "applet_id": plan.service.applet_id,
            "expected_source_count": plan.expected_source_count,
            "production_version": plan.service.production_version,
        }
        if phase == "preflight":
            package_proof = _resolved_proof(ctx, package)
            inputs["package"] = _resolved_receipt(ctx, package)
            inputs["oracle_inventory"] = package_proof["oracle_inventory"]
            inputs["source_counts"] = package_proof["source_counts"]
        else:
            inputs["preflight_receipt"] = _resolved_receipt(ctx, preflight)
            inputs["published_revision"] = _resolved_proof(ctx, marina)["publish"]["version"]
            if phase == "postpublish":
                inputs["hosted_receipt"] = _resolved_receipt(ctx, hosted)
        return inputs

    return _stage_step(
        plan,
        f"release_check_{phase}",
        spec,
        deps,
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=release_check_stage.ReleaseCheckInputs,
    )


def _node_of(handle: ArtifactStep) -> str:
    return handle.name.split("atlas-release-", 1)[1]


def _marina_release_step(
    plan: ReleasePlan, package: ArtifactStep, preflight: ArtifactStep
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["marina_release"]
    identity = _identity_fields(plan, "release_id", "source_revision", "service", "pins")

    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "applet_id": plan.service.applet_id,
            "service_url": plan.service.service_url,
            "package": _resolved_receipt(ctx, package),
            "preflight_receipt": _resolved_receipt(ctx, preflight),
            "prior_manifest": _ref_dict(plan.pins.prior_manifest),
            # The live revision and mode the preflight check observed, not the plan's
            # cut-time preview_base_version (R40: planned 32, live 39). The hosted
            # check then verifies the published revision (RELEASE.md).
            "base_version": _resolved_proof(ctx, preflight)["live"]["version"],
            "access_mode": _resolved_proof(ctx, preflight)["live"]["mode"],
            "output_root": stage_output_root(plan, "marina-release"),
        }

    return _stage_step(
        plan,
        "marina_release",
        spec,
        (package, preflight),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=marina_release_stage.MarinaReleaseInputs,
    )


def _hosted_check_step(
    plan: ReleasePlan,
    declarations: ArtifactStep,
    identity_backend: ArtifactStep,
    census: ArtifactStep,
    marina: ArtifactStep,
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["hosted_check"]
    identity = _identity_fields(plan, "release_id", "source_revision", "expected_source_count", "service")

    def build_inputs(ctx: StepContext) -> dict:
        publish = _resolved_proof(ctx, marina)["publish"]
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "revision": publish["version"],
            "root": publish["url"],
            "declarations": _resolved_receipt(ctx, declarations),
            "backend": _resolved_proof(ctx, identity_backend)["documents"]["backend_map"],
            "identity": _resolved_proof(ctx, identity_backend)["documents"]["identity_survey"],
            "expected_source_count": plan.expected_source_count,
            # The measured census total, as the strict audit reads it.
            "retained_tasks": _resolved_proof(ctx, census)["tasks"],
            "release_manifest": _resolved_receipt(ctx, marina),
            "output_root": stage_output_root(plan, "hosted-check"),
        }

    return _stage_step(
        plan,
        "hosted_check",
        spec,
        (declarations, identity_backend, census, marina),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=hosted_check_stage.HostedCheckInputs,
    )


def _exposure_probe_step(
    plan: ReleasePlan,
    declarations: ArtifactStep,
    marina: ArtifactStep,
    hosted: ArtifactStep,
    postpublish: ArtifactStep,
    audit_after: ArtifactStep,
) -> ArtifactStep[StageReceipt]:
    spec = _STAGE_SPECS["exposure_probe"]
    identity = _identity_fields(plan, "release_id", "service", "pins")

    def build_inputs(ctx: StepContext) -> dict:
        # The withheld sources and grants come from the release's sealed declaration
        # cut; an exposure_input_check pin, when a plan carries one, is a cross-check.
        publish = _resolved_proof(ctx, marina)["publish"]
        withheld, grants = exposure_probe_stage.exposure_lists(
            read_ref(ArtifactRef(**_resolved_receipt(ctx, declarations)))
        )
        return {
            "release_id": plan.release_id,
            "revision": publish["version"],
            "root": plan.service.public_root,
            "withheld_sources": withheld,
            "publication_grants": grants,
            "input_check_receipt": _ref_dict(plan.pins.exposure_input_check),
            "output_root": stage_output_root(plan, "exposure-probe"),
        }

    return _stage_step(
        plan,
        "exposure_probe",
        spec,
        (declarations, marina, hosted, postpublish, audit_after),
        build_inputs=build_inputs,
        identity=identity,
        inputs_type=exposure_probe_stage.ExposureProbeInputs,
    )


# ---------------------------------------------------------------------------
# The registry and the full release graph
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReleaseGraph:
    """The lowered release graph: one handle per node plus the terminal stage."""

    plan: ReleasePlan
    plan_handle: ArtifactStep[PlanDocument]
    census_inputs_handle: ArtifactStep[CensusInputsDocument]
    compose_manifest_handle: ArtifactStep[ComposeManifestDocument]
    stages: dict[str, ArtifactStep]
    """Graph-keyed stage handles (tail phases keyed with their phase suffix)."""
    terminal: ArtifactStep[StageReceipt]
    """The exposure probe; running it transitively builds every stage."""

    def handle(self, stage: str) -> ArtifactStep:
        return self.stages[stage]


def _preflight_wrapped_run(handle: ArtifactStep, node: str, policy: PreflightPolicy) -> Callable[[Any], Artifact | None]:
    """Wrap a stage's run so its fresh preflight receipt is ensured before execution.

    ``run`` never enters a fingerprint, so wrapping it after construction keeps every
    artifact identity unchanged; the gate fires exactly once per stage, immediately
    before the stage does anything (DESIGN.md §5: refuse a stage whose receipt is
    missing or stale for the landed source SHA).
    """
    original = handle.run

    def run(config: dict) -> Artifact | None:
        policy.ensure(node)
        return original(config)

    return run


def _contract_census_step(plan: ReleasePlan, census: ArtifactStep) -> ArtifactStep[StageReceipt]:
    def build_inputs(ctx: StepContext) -> dict:
        return {
            "release_id": plan.release_id,
            "source_revision": plan.source_revision,
            "census": _resolved_receipt(ctx, census),
            "targets": plan.targets,
            "output_root": stage_output_root(plan, "contract-census"),
        }

    return _stage_step(
        plan,
        "contract_census",
        _STAGE_SPECS["contract_census"],
        (census,),
        build_inputs=build_inputs,
        identity=_identity_fields(plan, "release_id", "source_revision", "targets"),
        inputs_type=contract_census_stage.ContractCensusInputs,
        resources=lambda: _resources_config(census_sizing(plan.expected_contract_tasks)),
    )


def release_graph(plan: ReleasePlan, *, preflight: PreflightPolicy | None = None) -> ReleaseGraph:
    """Build the full release graph for one sealed plan; constructs handles only.

    With a :class:`PreflightPolicy`, every stage node gates its own execution on a
    fresh preflight receipt (auto-preflight at submission time).
    """
    plan.validate_shape()
    plan_handle = _plan_document_step(plan)
    stages: dict[str, ArtifactStep] = {}

    def stage[T: Artifact](node: str, handle: ArtifactStep[T]) -> ArtifactStep[T]:
        if preflight is not None:
            handle = dataclasses.replace(handle, run=_preflight_wrapped_run(handle, node, preflight))
        stages[node] = handle
        return handle

    declarations = stage("declarations", _declarations_step(plan, plan_handle))
    stage("main_source_sync", _main_source_sync_step(plan, declarations))
    stage("main_grader_drift", _main_grader_drift_step(plan, declarations))
    census_inputs = stage("census_inputs", _census_inputs_step(plan, declarations))
    backend_validation = stage("backend_validation", _backend_validation_step(plan, census_inputs))
    identity_backend = stage("identity_backend_compose", _identity_backend_compose_step(plan, backend_validation))
    admission_check = stage("admission_check", _admission_check_step(plan, declarations, identity_backend))
    census = stage("census", _census_step(plan, census_inputs))
    contracts = stage("contract_census", _contract_census_step(plan, census))
    corpus = stage("corpus_assembly", _corpus_assembly_step(plan, census_inputs, census))
    embed = stage("embed", _embed_step(plan, corpus, census))
    compose_manifest = stage("compose_manifest", _compose_manifest_step(plan, declarations, embed, contracts))
    compose = stage("curriculum_compose", _curriculum_compose_step(plan, compose_manifest, census))
    region_handles = tuple(
        stage(f"text_scan_region[{region}]", _text_scan_region_step(plan, compose, region))
        for region in sorted(plan.scan_regions)
    )
    text_scans = stage("text_scans", _text_scan_merge_step(plan, compose, region_handles))
    history = stage("history_staging", _history_staging_step(plan))
    qc_plan = stage("qc_browse_plan", _qc_browse_plan_step(plan, declarations))
    qc_browse = stage("qc_browse", _qc_browse_copy_step(plan, qc_plan))
    fresh_qc_index = stage("fresh_qc_index", _fresh_qc_index_step(plan, qc_browse, declarations))
    snapshot = stage(
        "snapshot_publication", _snapshot_publication_step(plan, census, identity_backend, admission_check, contracts)
    )
    catalog_stage = stage(
        "catalog_stage", _catalog_stage_step(plan, declarations, snapshot, history, qc_browse, fresh_qc_index)
    )
    catalog_load = stage("catalog_load", _catalog_load_step(plan, catalog_stage))
    curriculum_load = stage("curriculum_load", _curriculum_load_step(plan, compose, text_scans))
    audit_before = stage(
        "strict_audit_before",
        _strict_audit_step(plan, declarations, identity_backend, "before", census, catalog_load, curriculum_load),
    )
    package = stage("package_preview", _package_preview_step(plan, audit_before, census))
    check_preflight = stage("release_check_preflight", _release_check_step(plan, "preflight", package, None, None, None))
    marina = stage("marina_release", _marina_release_step(plan, package, check_preflight))
    check_hosted = stage(
        "release_check_hosted", _release_check_step(plan, "hosted", None, check_preflight, None, marina)
    )
    audit_after = stage(
        "strict_audit_after",
        _strict_audit_step(plan, declarations, identity_backend, "after", census, origin_handle=marina),
    )
    hosted = stage("hosted_check", _hosted_check_step(plan, declarations, identity_backend, census, marina))
    check_postpublish = stage(
        "release_check_postpublish",
        _release_check_step(plan, "postpublish", None, check_preflight, check_hosted, marina),
    )
    exposure = stage(
        "exposure_probe", _exposure_probe_step(plan, declarations, marina, hosted, check_postpublish, audit_after)
    )
    return ReleaseGraph(
        plan=plan,
        plan_handle=plan_handle,
        census_inputs_handle=census_inputs,
        compose_manifest_handle=compose_manifest,
        stages=stages,
        terminal=exposure,
    )


def graph_nodes(graph: ReleaseGraph) -> list[dict]:
    """The lowered graph as display data: one row per node with its dependencies.

    The plan-gated main-source sync leaf hangs outside the terminal's edges, so it
    is appended after the reachable nodes: every build sees the node it can opt
    into, whether or not its plan does.
    """
    nodes: list[dict] = []
    memo: set[int] = set()

    def visit(handle: ArtifactStep) -> None:
        if id(handle) in memo:
            return
        memo.add(id(handle))
        for dep in handle.deps:
            visit(dep)
        nodes.append(
            {
                "name": handle.name,
                "version": handle.version,
                "fingerprint": handle.fingerprint(),
                "deps": [dep.name for dep in handle.deps],
            }
        )

    visit(graph.terminal)
    sync = graph.stages.get("main_source_sync")
    if sync is not None:
        visit(sync)
    drift = graph.stages.get("main_grader_drift")
    if drift is not None:
        visit(drift)
    return nodes
