# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-stage preflight slices (DESIGN.md §6, BRIEF requirement 3).

Before a stage submits, it must hold a fresh preflight receipt: a type-check of its
module plus a small-slice run of its real code path on a bounded slice of real
inputs — one census source, one corpus preparation, the real prior-load SQL reads —
never synthetic fixtures. This module owns the slice selections and runners; the
``atlas-release preflight`` mode and the graph's per-stage submission gate
(``graph.PreflightPolicy``) both execute through it.

A slice runner is available only for stages whose real inputs exist without
write-mode upstream receipts. The remaining stages are ``blocked``: their preflight
waits until the real upstream stages have run (the submission gate refuses them
until then, with the reason recorded here).
"""

import dataclasses
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import (
    admission_check as admission_check_stage,
)
from experiments.post_training.task_curation.release_stages import (
    backend_validation as backend_validation_stage,
)
from experiments.post_training.task_curation.release_stages import (
    census as census_stage,
)
from experiments.post_training.task_curation.release_stages import (
    corpus_assembly as corpus_assembly_stage,
)
from experiments.post_training.task_curation.release_stages import (
    declarations as declarations_stage,
)
from experiments.post_training.task_curation.release_stages import (
    embed as embed_stage,
)
from experiments.post_training.task_curation.release_stages import (
    history_staging as history_staging_stage,
)
from experiments.post_training.task_curation.release_stages import (
    identity_backend_compose as identity_backend_compose_stage,
)
from experiments.post_training.task_curation.release_stages import paced_query
from experiments.post_training.task_curation.release_stages import (
    package_preview as package_preview_stage,
)
from experiments.post_training.task_curation.release_stages import (
    preflight as preflight_module,
)
from experiments.post_training.task_curation.release_stages import (
    qc_browse as qc_browse_stage,
)
from experiments.post_training.task_curation.release_stages.graph import (
    _STAGE_SPECS,
    REPO_ROOT,
    ReleasePlan,
    _seal_census_inputs,
    release_graph,
    stage_output_root,
)
from experiments.post_training.task_curation.release_stages.guard import repo_head
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

# Nodes whose small slice runs on real inputs available before any write-mode
# upstream receipt exists. Everything else is blocked until its upstreams complete.
RUNNABLE_NODES = frozenset(
    {
        "declarations",
        "census_inputs",
        "census",
        "corpus_assembly",
        "embed",
        "admission_check",
        "backend_validation",
        "history_staging",
        "identity_backend_compose",
        "qc_browse_plan",
        "package_preview",
    }
)

BLOCK_REASONS: dict[str, str] = {
    "compose_manifest": "needs the real embed release/coverage receipts",
    "curriculum_compose": "needs the sealed composition manifest and the real census receipt",
    "text_scan_region": "needs the composed handoff document",
    "text_scans": "needs the composed handoff and the regional scan receipts",
    "qc_browse": "writes SQL; needs the sealed browse plan receipt",
    "fresh_qc_index": "needs the sealed browse completion receipt",
    "snapshot_publication": "needs the real census receipt and the composed survey documents",
    "catalog_stage": "needs the publication, history-staging and QC-browse receipts",
    "catalog_load": "needs the sealed catalog staging plan",
    "curriculum_load": "needs the composed handoff and the merged text-scan receipt",
    "strict_audit_before": "needs the sealed declarations, backend and identity documents",
    "strict_audit_after": "needs the published revision and the sealed documents",
    "release_check_preflight": "needs the sealed package receipt and its inventory pins",
    "release_check_hosted": "needs the sealed preflight receipt and the published revision",
    "release_check_postpublish": "needs the preflight and hosted receipts and the published revision",
    "marina_release": "needs the package, preflight and hosted receipts",
    "hosted_check": "needs the sealed deployment manifest",
    "exposure_probe": "needs the published revision and the sealed input-check receipt",
}

# The bounded text-scan slice the (blocked) regional scan would use.
DEFAULT_MAX_FILES = 25


@dataclasses.dataclass(frozen=True)
class SliceSelection:
    """The bounded real-input selections the slice runners use."""

    source: str | None = None
    """The census/corpus slice source; the first declared source by default."""
    max_files: int = DEFAULT_MAX_FILES
    """The text-scan file bound (unused while the regional scan is blocked)."""

    def resolved_source(self, plan: ReleasePlan) -> str:
        if self.source is not None:
            if self.source not in plan.populations:
                raise ValueError(f"slice source {self.source!r} is not a declared population pin")
            return self.source
        return sorted(plan.populations)[0]


def slice_root(plan: ReleasePlan) -> str:
    """Where slice artifacts land: the plan's slice root, else the release prefix."""
    return plan.slice_output_root or f"{plan.release_root}/preflight-slices"


def source_slice_root(plan: ReleasePlan, source: str) -> str:
    """One bounded slice's root, keyed by the slice source so per-source
    documents cannot collide under the same release prefix."""
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    return f"{slice_root(plan)}/{key}"


def _ref_dict(ref: ArtifactRef) -> dict:
    return {"uri": ref.uri, "sha256": ref.sha256}


def _declarations_document() -> dict:
    """The current declaration set, read the way the census rows bind it.

    The declaration gate belongs to the declarations stage's own check; the
    slice's census-inputs document needs the registry's declarations as the
    census rows will see them.
    """
    from experiments.post_training.task_curation.sources import all_sources  # noqa: PLC0415
    from infra.marina.apps.rl_data_catalog.server.source_identity import (  # noqa: PLC0415 — the app import stays beside its only use
        canonical_source_id,
    )

    document = {}
    for source in all_sources().values():
        document[canonical_source_id(source.info.id)] = source.declaration.model_dump(mode="json")
    return document


def _seal_slice_document(path: str, document) -> ArtifactRef:
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"slice document drifted at {path}; use a fresh slice root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def _slice_declarations_ref(plan: ReleasePlan) -> ArtifactRef:
    """The slice's declarations document: the plan's re-bound served export.

    The plan's prior_declarations pin IS the current served state (the
    coordinator's re-bindings land there); the slices review that document, not
    a separate ungated read that would carry stale bindings.
    """
    return ArtifactRef(**plan.pins.prior_declarations.model_dump(mode="json"))


def _graph_receipt_document(plan: ReleasePlan, node: str, dotted_key: str) -> dict:
    """A sealed reference from an upstream stage's graph receipt under the release prefix."""
    path = StoragePath(f"{plan.release_root}/graph/{node}.json")
    if not path.exists():
        raise ValueError(f"blocked: {node} has not run; its graph receipt is missing at {path}")
    document = json.loads(path.read_text())
    value: dict = document
    for part in dotted_key.split("."):
        value = value[part]
    return {"uri": value["uri"], "sha256": value["sha256"]}


def _graph_receipt_receipt(plan: ReleasePlan, node: str) -> dict:
    """The sealed receipt reference from an upstream stage's graph receipt."""
    return _graph_receipt_document(plan, node, "receipt")


def _census_inputs_document(plan: ReleasePlan, source: str) -> tuple[str, str]:
    """Seal the one-source census-inputs document into the source's slice root."""
    document = _declarations_document()
    if source not in document:
        raise ValueError(f"declared source {source!r} is missing from the current declaration set")
    proof = _seal_census_inputs(
        declarations={source: document[source]},
        populations={source: plan.populations[source]},
        release_root=f"{source_slice_root(plan, source)}/census-inputs",
        source_revision=plan.source_revision,
    )
    return proof["uri"], proof["sha256"]


def _census_measure_receipt(plan: ReleasePlan, source: str) -> ArtifactRef:
    """Measure one real source and seal the slice census receipt its consumers read."""
    inputs_uri, _inputs_sha = _census_inputs_document(plan, source)
    row = census_stage.CensusSourceRow(**json.loads(StoragePath(inputs_uri).read_text())["rows"][0])
    measured = census_stage.measure_source(row, f"{slice_root(plan)}/census", plan.source_revision, plan.region_prefix)
    document = {
        "kind": f"atlas-{plan.release_id}-{census_stage.RECEIPT_SUFFIX}",
        "release_id": plan.release_id,
        "source_revision": plan.source_revision,
        "results": [measured],
    }
    return _seal_slice_document(f"{source_slice_root(plan, source)}/census-receipt.json", document)


def slice_inputs(node: str, plan: ReleasePlan, selection: SliceSelection) -> dict:
    """The plan-derived configuration a node's slice runs against (digest-stable)."""
    root = slice_root(plan)
    sliced = node in ("census_inputs", "census", "corpus_assembly", "embed")
    source = selection.resolved_source(plan) if sliced else None
    inputs: dict = {
        "release_id": plan.release_id,
        "source_revision": plan.source_revision,
        "slice_root": root,
        "region_prefix": plan.region_prefix,
        "expected_source_count": plan.expected_source_count,
        "prior_load_id": plan.prior.load_id,
        "service_url": plan.service.service_url,
        "applet_id": plan.service.applet_id,
        "applet_origin": plan.service.applet_origin,
    }
    if source is not None:
        inputs["only"] = source
        inputs["population"] = plan.populations[source]
    if node == "embed":
        inputs["cache_root"] = plan.embed_cache_root
        inputs["parity_receipt"] = _ref_dict(plan.pins.parity_receipt) if plan.pins.parity_receipt else None
        inputs["gpu_runtime"] = plan.gpu_runtime
    if node in ("admission_check", "qc_browse_plan", "history_staging", "identity_backend_compose"):
        inputs["snapshot_id"] = plan.snapshot_id
        inputs["catalog_root"] = plan.catalog_root
    if node == "backend_validation":
        inputs["target"] = plan.validation_target
        inputs["grading_environments"] = plan.validation_grading_environments
        inputs["integrity_contracts"] = plan.validation_integrity_contracts
        inputs["runtime_defaults"] = plan.validation_runtime_defaults
    if node == "identity_backend_compose":
        inputs["pins"] = {
            name: _ref_dict(getattr(plan.pins, name)) for name in ("prior_identity", "prior_backend", "registry_index")
        }
        inputs["targets"] = list(plan.targets)
        inputs["fresh_backend_receipt"] = _graph_receipt_receipt(plan, "backend_validation")
    if node == "admission_check":
        inputs["backend"] = _graph_receipt_document(plan, "identity_backend_compose", "documents.backend_map")
    if node == "qc_browse_plan":
        inputs["expected_panels"] = plan.expected_panels
        inputs["exclusions"] = list(plan.qc_exclusions)
        inputs["catalog_target_load_id"] = plan.catalog_target_load_id
    if node == "package_preview":
        inputs["package_output_root"] = plan.package_output_root
        inputs["oracle_inventory"] = _ref_dict(plan.pins.oracle_inventory)
        inputs["source_counts"] = _ref_dict(plan.pins.source_counts)
    return inputs


def slice_runner(node: str, plan: ReleasePlan, selection: SliceSelection) -> Callable[[], dict]:
    """The node's small-slice runner: its real code path on bounded real inputs."""
    root = slice_root(plan)
    source = selection.resolved_source(plan)
    source_root = source_slice_root(plan, source)

    if node == "declarations":

        def run() -> dict:
            proof = declarations_stage.run_declarations(
                declarations_stage.DeclarationInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    output_root=f"{root}/declarations",
                    expected_source_count=plan.expected_source_count,
                    targets=plan.targets,
                    prior_declarations=plan.pins.prior_declarations,
                ),
                dry_run=True,
                repo_root=REPO_ROOT,
            )
            return {"source_count": proof["source_count"], "targets": list(proof["targets"]), "mode": "dry-run"}

        return run

    if node == "census_inputs":

        def run() -> dict:
            uri, sha256 = _census_inputs_document(plan, source)
            return {"inputs": {"uri": uri, "sha256": sha256}, "only": source, "mode": "slice-seal"}

        return run

    if node == "census":

        def run() -> dict:
            uri, sha256 = _census_inputs_document(plan, source)
            proof = census_stage.run_census(
                census_stage.CensusInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    inputs=ArtifactRef(uri=uri, sha256=sha256),
                    output_root=stage_output_root(plan, "census"),
                    region_prefix=plan.region_prefix,
                    expected_sources=1,
                    slice_output_root=f"{source_root}/census",
                ),
                dry_run=True,
                only=source,
            )
            return {
                "only": source,
                "tasks": proof["tasks"],
                "oracle_tasks": proof["oracle_tasks"],
                "mode": "dry-run",
            }

        return run

    if node == "corpus_assembly":

        def run() -> dict:
            receipt = _census_measure_receipt(plan, source)
            uri, sha256 = _census_inputs_document(plan, source)
            proof = corpus_assembly_stage.run_corpus_assembly(
                corpus_assembly_stage.CorpusAssemblyInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    inputs=ArtifactRef(uri=uri, sha256=sha256),
                    census_receipt=receipt,
                    output_root=stage_output_root(plan, "corpus"),
                    expected_sources=1,
                    slice_output_root=f"{source_root}/corpus",
                ),
                dry_run=True,
                only=source,
            )
            return {"only": source, "tasks": proof["tasks"], "mode": "dry-run"}

        return run

    if node == "embed":

        def run() -> dict:
            corpus_path = StoragePath(f"{source_root}/corpus/corpus.json")
            if not corpus_path.exists():
                raise ValueError(
                    "the embed slice needs the corpus-assembly slice output; run the corpus_assembly preflight first"
                )
            payload = corpus_path.read_bytes()
            corpus = ArtifactRef(uri=str(corpus_path), sha256=hashlib.sha256(payload).hexdigest())
            tasks = sum(row["task_count"] for row in json.loads(payload)["inventory"])
            proof = embed_stage.run_embed(
                embed_stage.EmbedInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    corpus=corpus,
                    artifact_root=f"{source_root}/embed",
                    cache_root=plan.embed_cache_root,
                    expected_source_count=1,
                    expected_task_count=tasks,
                    parity_receipt=plan.pins.parity_receipt,
                    gpu_runtime=plan.gpu_runtime,
                ),
                dry_run=True,
            )
            return {"device": proof["device"], "tasks": proof["tasks"], "mode": "dry-run"}

        return run

    if node == "admission_check":

        def run() -> dict:
            proof = admission_check_stage.run_admission_check(
                admission_check_stage.AdmissionCheckInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    declarations=_slice_declarations_ref(plan),
                    backend=ArtifactRef(
                        **_graph_receipt_document(plan, "identity_backend_compose", "documents.backend_map")
                    ),
                    snapshot_id=plan.snapshot_id,
                    catalog_root=plan.catalog_root,
                    prior_load_id=plan.prior.load_id,
                    output_root=f"{root}/admission-check",
                    expected_source_count=plan.expected_source_count,
                    service_url=plan.service.service_url,
                    applet_id=plan.service.applet_id,
                ),
                dry_run=True,
            )
            return {
                "review_source_count": proof["review_source_count"],
                "previously_available": proof["previously_available"],
                "blocked": proof["blocked"],
                "mode": "dry-run",
            }

        return run

    if node == "backend_validation":

        def run() -> dict:
            uri, sha = _census_inputs_document(plan, plan.validation_target)
            proof = backend_validation_stage.run_backend_validation(
                backend_validation_stage.BackendValidationInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    snapshot_id=plan.snapshot_id,
                    catalog_root=plan.catalog_root,
                    output_root=f"{root}/backend-validation",
                    target=plan.validation_target,
                    census_inputs=ArtifactRef(uri=uri, sha256=sha),
                    grading_environments=plan.validation_grading_environments,
                    integrity_contracts=plan.validation_integrity_contracts,
                    runtime_defaults=plan.validation_runtime_defaults,
                ),
                dry_run=True,
            )
            return {"target": proof["target"], "task_count": proof["task_count"], "mode": "dry-run"}

        return run

    if node == "history_staging":

        def run() -> dict:
            proof = history_staging_stage.run_history_staging(
                history_staging_stage.HistoryStagingInputs(
                    release_id=plan.release_id,
                    prior_load_id=plan.prior.load_id,
                    target_load_id=plan.catalog_target_load_id,
                    snapshot_id=plan.snapshot_id,
                    source_revision=plan.source_revision,
                    catalog_root=plan.catalog_root,
                    receipt_root=root,
                    expected_prior_sources=plan.expected_prior_sources,
                    service_url=plan.service.service_url,
                    applet_id=plan.service.applet_id,
                ),
                dry_run=True,
            )
            return {
                "prior_source_count": proof["prior_source_count"],
                "prior_item_count": proof["prior_item_count"],
                "mode": "dry-run",
            }

        return run

    if node == "identity_backend_compose":

        def run() -> dict:
            proof = identity_backend_compose_stage.run_identity_backend_compose(
                identity_backend_compose_stage.IdentityBackendComposeInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    output_root=f"{root}/identity-backend",
                    targets=plan.targets,
                    prior_identity=plan.pins.prior_identity,
                    prior_backend=plan.pins.prior_backend,
                    fresh_backend_receipt=ArtifactRef(**_graph_receipt_receipt(plan, "backend_validation")),
                    registry_index=plan.pins.registry_index,
                    expected_target_count=len(plan.targets),
                ),
                dry_run=True,
            )
            return {
                "fingerprint_count": proof["identity"]["fingerprint_count"],
                "backend_source_count": proof["backend"]["source_count"],
                "mode": "dry-run",
            }

        return run

    if node == "qc_browse_plan":

        def run() -> dict:
            control = paced_query.RetryControl()
            query = paced_query.open_paced_query(plan.service.service_url, plan.service.applet_id, control)
            proof = qc_browse_stage.plan_qc_browse(
                qc_browse_stage.QcBrowseInputs(
                    release_id=plan.release_id,
                    prior_load_id=plan.prior.load_id,
                    target_load_id=plan.catalog_target_load_id,
                    source_revision=plan.source_revision,
                    declarations=_slice_declarations_ref(plan),
                    output_root=f"{root}/qc-browse",
                    expected_panels=plan.expected_panels,
                    exclusions=plan.qc_exclusions,
                    service_url=plan.service.service_url,
                    applet_id=plan.service.applet_id,
                ),
                query=query,
                dry_run=True,
            )
            return {"panels": proof["panels"], "excluded": proof["excluded"], "mode": "dry-run"}

        return run

    if node == "package_preview":

        def run() -> dict:
            proof = package_preview_stage.run_package_preview(
                package_preview_stage.PackagePreviewInputs(
                    release_id=plan.release_id,
                    source_revision=plan.source_revision,
                    output_root=plan.package_output_root,
                    oracle_inventory=plan.pins.oracle_inventory,
                    source_counts=plan.pins.source_counts,
                ),
                dry_run=True,
                repo_root=REPO_ROOT,
            )
            return {"packaged": proof["packaged"] is not None, "mode": "dry-run"}

        return run

    raise ValueError(f"node {node!r} has no slice runner: {BLOCK_REASONS.get(node, 'unregistered')}")


def _upstream_witness(node: str, plan: ReleasePlan) -> tuple[dict[str, str], dict] | None:
    """The node's upstream artifact records, digest-bound, once every upstream has run.

    A pipeline stage whose inputs are upstream release documents has no small real-input
    slice: its real gate is the landed module, the typed-input validation and the
    write-once seals. The witness receipt binds those upstream records so a changed
    upstream stales the gate exactly like a changed slice input.
    """
    from marin.execution.artifact import read_record  # noqa: PLC0415 — beside its only use
    from rigging.filesystem.cluster_config import marin_prefix  # noqa: PLC0415 — beside its only use

    handle = release_graph(plan).stages[node]
    digests: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for dep in handle.deps:
        path = dep.path(marin_prefix())
        record = read_record(path)
        if record is None or record.result is None:
            return None
        raw = StoragePath(path).read_bytes()
        digests[dep.name] = hashlib.sha256(raw).hexdigest()
        evidence[dep.name] = record.result_type or ""
    return digests, evidence


def ensure_preflight(
    node: str,
    plan: ReleasePlan,
    *,
    repo_root=REPO_ROOT,
    selection: SliceSelection | None = None,
    output_root: str | None = None,
) -> preflight_module.PreflightReceipt:
    """Guarantee a fresh preflight receipt for one node; raise to refuse the stage.

    A runnable node's receipt is sealed at the current landed sha over unchanged slice
    inputs (the slice runs, type-check included) and served as-is while fresh. A
    pipeline stage whose inputs are upstream release documents has no small real-input
    slice: it is gated on an ``upstream`` witness receipt binding its upstream stages'
    artifact records, and refuses while any upstream has not run.
    """
    selection = selection or SliceSelection()
    root = output_root or plan.release_root
    landed = repo_head(repo_root)
    existing = preflight_module.read_receipt(root, node)
    if node not in RUNNABLE_NODES:
        witness = _upstream_witness(node, plan)
        if witness is None:
            reason = BLOCK_REASONS.get(node, "no slice runner")
            state = "missing" if existing is None else ("failed" if not existing.ok else "stale")
            raise ValueError(
                f"stage {node} has no fresh preflight receipt ({state}) and its slice is blocked: {reason};"
                " run its upstream stages first, then preflight it"
            )
        digests, evidence = witness
        if preflight_module.receipt_is_fresh(existing, landed, digests):
            return existing
        receipt = preflight_module.PreflightReceipt(
            stage=node,
            mode="upstream",
            repo_sha=landed,
            input_digests=digests,
            ok=True,
            checked_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            slice_evidence=evidence,
        )
        preflight_module.write_receipt(root, receipt)
        return receipt
    inputs = slice_inputs(node, plan, selection)
    digests = preflight_module.input_digests(inputs)
    if preflight_module.receipt_is_fresh(existing, landed, digests):
        return existing
    return preflight_module.run_preflight(
        node,
        module_path(node),
        inputs,
        output_root=root,
        repo_root=repo_root,
        slice_runner=slice_runner(node, plan, selection),
    )


def module_path(node: str) -> str:
    """The module file path the preflight type-checks for one node."""
    if node in ("census_inputs", "compose_manifest"):
        # The configuration-document nodes seal through the graph's helpers.
        return "experiments/post_training/task_curation/release_stages/graph.py"
    return _STAGE_SPECS[node].module.replace(".", "/") + ".py"
