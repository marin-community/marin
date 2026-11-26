# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curriculum compose: add the admitted populations to the retained curriculum.

One maintained module replaces the per-release curriculum composition workers (see
INVENTORY.md). The composed release carries every retained assignment value byte for
byte, adds the fresh aligned populations, recomputes statistics/samples/confidence
from the full assignments, and verifies the statistics and confidence vectors as
exact unions of the retained and fresh coverages.

Every release-identifying value arrives as typed :class:`CurriculumComposeInputs`;
the fresh task total comes from the census receipt, never a worker literal. The dry
run (the per-release preflight shape) runs every read-side assertion and exits
before any write or cluster side effect.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from finestore.rl.curation import QualifiedSourceDeclaration, normalize_snapshot_declaration
from fray.types import ResourceConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.coverage import build_curriculum_coverage
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.curriculum_projection import filter_assignment_file, statistics_by_key
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.local import local_client
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.source_status import pending_stage, receipt_registry_keys
from infra.marina.apps.rl_data_catalog.confidence import OUTCOMES, confidence_rows
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference
from infra.marina.apps.rl_data_catalog.server.source_identity import original_source_id

COMPOSE_RECEIPT_SUFFIX = "curriculum-composition"
COMPOSE_WORKERS = 8
COMPOSE_WORKER_CPU = 4
COMPOSE_WORKER_MEMORY = "16g"
# Fields of a target declaration that may differ from the aligned inventory's copy:
# consumer-gate outcomes recorded after alignment, never a task-content field.
ALLOWED_TARGET_CHANGES = frozenset(
    {"consumer_receipt", "consumer_requirements", "consumer_findings", "consumer_decision", "notes"}
)
SAMPLE_PREVIEW_CHARACTERS = 600
CONFIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CurriculumComposeInputs:
    """Typed inputs for one composition run; every value is a graph output."""

    release_id: str
    source_revision: str
    inputs: ArtifactRef
    output_root: str
    expected_fresh_task_count: int
    # The plan's planned removals: dropped from the retained curriculum, never carried.
    removed_sources: tuple[str, ...]

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if self.expected_fresh_task_count <= 0:
            raise ValueError(f"expected_fresh_task_count must be positive, got {self.expected_fresh_task_count}")


@dataclass(frozen=True)
class ComposePlan:
    """Everything the composition verified before a single write."""

    targets: tuple[str, ...]
    sources: tuple[Any, ...]
    inventory: tuple[Any, ...]
    old_release: AlignmentRelease
    old_coverage: CurriculumCoverage
    fresh_release: AlignmentRelease
    fresh_coverage: CurriculumCoverage
    fresh_task_count: int
    retained_task_count: int
    removed_sources: tuple[str, ...]


def read_reference(reference: dict) -> dict:
    """Digest-verify one sealed {uri, sha256} reference and parse its JSON."""
    payload = StoragePath(reference["uri"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"sealed reference digest mismatch: {reference['uri']}")
    return json.loads(payload)


def read_composition_inputs(inputs: CurriculumComposeInputs) -> dict:
    """Digest-verify the sealed composition manifest and parse its references."""
    payload = StoragePath(inputs.inputs.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs.inputs.sha256:
        raise ValueError(f"composition inputs digest mismatch: {inputs.inputs.uri}")
    document = json.loads(payload)
    for required in ("old_release", "old_coverage", "fresh_release", "fresh_coverage", "current_target_declarations"):
        if required not in document:
            raise ValueError(f"composition inputs must include {required!r}")
    return document


def read_retained_release(reference: dict) -> AlignmentRelease:
    """Read the retained release, upgrading earlier-generation inventory declarations.

    A retained release sealed by an earlier cut carries its inventory in that cut's
    declaration schema (for example the ``qualified-v1`` form with the removed
    ``name`` field). The maintained snapshot normalizer upgrades each row; a pre-stage
    Pending row takes its stage from measurement-receipt evidence.
    """
    document = read_reference(reference)
    document["inventory"] = [
        normalize_snapshot_declaration(
            row, pending_stage_of=lambda source_id: pending_stage(source_id, receipt_registry_keys())
        )
        for row in document["inventory"]
    ]
    return AlignmentRelease.model_validate(document)


def compose_plan(document: dict, expected_fresh_task_count: int, removed_sources: tuple[str, ...]) -> ComposePlan:
    """Verify every read-side assertion for one additions-mode composition.

    The plan's removed sources leave the retained curriculum: their release sources,
    inventory rows, assignments and coverage statistics are not carried.
    """
    rebound = document.get("rebound")
    if rebound:
        raise ValueError(
            f"the composition manifest declares a rebound of {rebound.get('source_id')!r}, but the maintained"
            " additions-mode compose cannot rebind a source (the projection pass is not wired in);"
            " resolve the rebound before composing"
        )
    old = read_retained_release(document["old_release"])
    old_coverage = CurriculumCoverage.model_validate(read_reference(document["old_coverage"]))
    fresh = AlignmentRelease.model_validate(read_reference(document["fresh_release"]))
    fresh_coverage = CurriculumCoverage.model_validate(read_reference(document["fresh_coverage"]))
    if old_coverage.alignment_release_id != old.release_id or fresh_coverage.alignment_release_id != fresh.release_id:
        raise ValueError("coverage documents do not bind their alignment releases")
    if (
        old.nodes != fresh.nodes
        or old.curriculum_key != fresh.curriculum_key
        or old.embedding_spec != fresh.embedding_spec
    ):
        raise ValueError("retained and fresh releases carry different curriculum or embedding specifications")
    for key in (
        "subject_distance",
        "capability_distance",
        "source_sample_size",
        "area_sample_size",
        "confidence_levels",
    ):
        if getattr(old_coverage, key) != getattr(fresh_coverage, key):
            raise ValueError(f"retained and fresh coverages disagree on {key}")
    targets = frozenset(source.source_id for source in fresh.sources)
    fresh_task_count = sum(source.task_count for source in fresh.sources)
    if fresh_task_count != expected_fresh_task_count:
        raise ValueError(f"fresh populations carry {fresh_task_count} tasks, expected {expected_fresh_task_count}")
    removed = frozenset(removed_sources)
    unknown = sorted(removed - {source.source_id for source in old.sources})
    if unknown:
        raise ValueError(f"planned removals are absent from the retained curriculum: {unknown[:5]}")
    old_sources = {source.source_id: source for source in old.sources if source.source_id not in removed}
    retained_inventory = [declaration for declaration in old.inventory if declaration.id not in removed]
    if len(old.inventory) - len(retained_inventory) != len(removed):
        raise ValueError("planned removals do not each drop exactly one retained inventory row")
    if not targets.isdisjoint(removed):
        raise ValueError("a target is also a planned removal")
    if not targets.isdisjoint(old_sources.keys()):
        raise ValueError("fresh populations overlap the retained curriculum; additions mode cannot replace sources")
    for source in fresh.sources:
        if not (source.task_count == source.input_rows == source.converted_rows == source.ready_count):
            raise ValueError(f"fresh population {source.source_id} is not fully converted")
        if source.insufficient_context_count != 0:
            raise ValueError(f"fresh population {source.source_id} holds insufficient-context rows")
    current_target_declarations = read_reference(document["current_target_declarations"])
    fresh_inventory = {declaration.id: declaration for declaration in fresh.inventory}
    current_inventory = {
        sid: canonical_declaration(QualifiedSourceDeclaration.model_validate(row))
        for sid, row in current_target_declarations.items()
    }
    if set(current_inventory) != set(fresh_inventory) or not set(fresh_inventory) <= targets:
        raise ValueError("the current target declarations do not cover exactly the fresh inventory")
    for sid, original in fresh_inventory.items():
        current = current_inventory[sid]
        before = original.model_dump(mode="json")
        after = current.model_dump(mode="json")
        if fingerprint_input(before) != fingerprint_input(after):
            raise ValueError(f"current declaration for {sid} drifted from the aligned inventory copy")
        if before["actor_census_input"] != after["actor_census_input"] or before["task_count"] != after["task_count"]:
            raise ValueError(f"current declaration for {sid} changed its census binding or task count")
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if not changed <= ALLOWED_TARGET_CHANGES:
            raise ValueError(f"current declaration for {sid} changed non-consumer fields: {sorted(changed)}")
    inventory = retained_inventory + [current_inventory[declaration.id] for declaration in fresh.inventory]
    sources = list(old_sources.values()) + list(fresh.sources)
    return ComposePlan(
        targets=tuple(sorted(targets)),
        sources=sources,
        inventory=inventory,
        old_release=old,
        old_coverage=old_coverage,
        fresh_release=fresh,
        fresh_coverage=fresh_coverage,
        fresh_task_count=fresh_task_count,
        retained_task_count=sum(source.task_count for source in old_sources.values()),
        removed_sources=tuple(sorted(removed)),
    )


def compose_release(plan: ComposePlan, document: dict, composition_inputs: dict) -> AlignmentRelease:
    """The composed release: retained lineage plus the composition provenance."""
    lineage = {
        **plan.old_release.lineage,
        "prior_release_uri": document["old_release"]["uri"],
        "prior_release_sha256": document["old_release"]["sha256"],
        "fresh_release_uri": document["fresh_release"]["uri"],
        "fresh_release_sha256": document["fresh_release"]["sha256"],
        "composition_inputs_uri": composition_inputs["uri"],
        "composition_inputs_sha256": composition_inputs["sha256"],
    }
    identity = {
        "lineage": lineage,
        "sources": [source.model_dump(mode="json") for source in plan.sources],
        "inventory": [declaration.model_dump(mode="json") for declaration in plan.inventory],
    }
    release = plan.old_release.model_copy(
        update={
            "release_id": canonical_sha256(identity),
            "created_at": datetime.now(tz=UTC).isoformat(),
            "sources": plan.sources,
            "inventory": plan.inventory,
            "paths": list(plan.old_release.paths) + list(plan.fresh_release.paths),
            "lineage": lineage,
            "task_count": plan.retained_task_count + plan.fresh_task_count,
            "ready_count": sum(source.ready_count for source in plan.sources),
            "insufficient_context_count": sum(source.insufficient_context_count for source in plan.sources),
        }
    )
    release = AlignmentRelease.model_validate(release.model_dump(mode="json"))
    if release.task_count != plan.retained_task_count + plan.fresh_task_count:
        raise ValueError("composed release task count disagrees with its sources")
    return release


def compose_coverage_identity(plan: ComposePlan, release: AlignmentRelease) -> dict:
    """The coverage identity the recomputed coverage must carry."""
    return {
        "alignment_release_id": release.release_id,
        **{
            key: getattr(plan.old_coverage, key)
            for key in ("subject_distance", "capability_distance", "source_sample_size", "area_sample_size")
        },
        "version": COVERAGE_VERSION,
        "confidence_levels": [level.model_dump(mode="json") for level in plan.old_coverage.confidence_levels],
    }


def seal(root: str, name: str, value: dict) -> ManifestReference:
    """Write-once seal with byte-identical readback; returns the manifest reference."""
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = StoragePath(root) / name
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ManifestReference(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def confidence_export(
    root: str,
    release: AlignmentRelease,
    coverage: CurriculumCoverage,
    release_ref: ManifestReference,
    coverage_ref: ManifestReference,
) -> tuple[ManifestReference, int]:
    """Export and validate the catalog confidence document; returns (ref, validated rows)."""
    width = len(coverage.confidence_levels)
    vectors: dict[str, list] = defaultdict(list)
    for path in coverage.confidence_paths:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    vectors[row["source_id"]].append(
                        {key: row[key] for key in ("metric", "subject_id", "node_id", "instances")}
                    )
    sources = {row.source_id: row for row in release.sources if row.basis == "matched"}
    if set(vectors) != set(sources) or width != 5:
        raise ValueError("confidence vectors do not cover exactly the matched sources at width 5")
    aggregates: dict[str, dict] = {scope: defaultdict(lambda: [0] * width) for scope in ("all", "train", "eval")}
    rows = []
    for sid, stats in sorted(vectors.items()):
        use = sources[sid].intended_use
        if use not in ("train", "eval"):
            raise ValueError(f"source {sid} intended use {use!r} has no confidence scope")
        rows.append({"source_id": sid, "intended_use": use, "statistics": stats})
        for stat in stats:
            key = (stat["metric"], stat["subject_id"], stat["node_id"])
            for scope in ("all", use):
                aggregates[scope][key] = [a + b for a, b in zip(aggregates[scope][key], stat["instances"], strict=True)]
    for scope in aggregates:
        for metric in ("denominator", *OUTCOMES):
            aggregates[scope][(metric, "", "")]
        for node in release.nodes:
            aggregates[scope][("placement", node.subject_id, node.node_id)]
    document = {
        "schema_version": CONFIDENCE_SCHEMA_VERSION,
        "release": {"uri": release_ref.uri, "sha256": release_ref.sha256},
        "coverage": {"uri": coverage_ref.uri, "sha256": coverage_ref.sha256},
        "levels": [level.model_dump(mode="json") for level in coverage.confidence_levels],
        "sources": rows,
        "aggregates": {
            scope: [
                {"metric": key[0], "subject_id": key[1], "node_id": key[2], "instances": value}
                for key, value in sorted(values.items())
            ]
            for scope, values in aggregates.items()
        },
        "samples": {
            "paths": coverage.sample_paths,
            "task_count": coverage.sample_task_count,
            "list_preview_characters": SAMPLE_PREVIEW_CHARACTERS,
        },
    }
    validated = sum(1 for _ in confidence_rows(document, release, coverage))
    return seal(root, "confidence.json", document), validated


def run_compose(
    inputs: CurriculumComposeInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
    context: Any = None,
) -> dict:
    """Run one composition; ``dry_run`` verifies every assertion with zero writes.

    ``context`` injects the Zephyr execution context; the default builds the real
    local client and runtime on the write path only.
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "curriculum_compose", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        document = read_composition_inputs(inputs)
        plan = compose_plan(document, inputs.expected_fresh_task_count, inputs.removed_sources)
        release = compose_release(plan, document, {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256})
        proof: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{COMPOSE_RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "inputs": {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256},
            "sources": len(plan.sources),
            "inventory": len(plan.inventory),
            "targets": list(plan.targets),
            "fresh_task_count": plan.fresh_task_count,
            "retained_task_count": plan.retained_task_count,
            "removed_sources": list(plan.removed_sources),
            "composed_task_count": release.task_count,
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "curriculum_compose",
                {"mode": mode, "composed_task_count": release.task_count},
                alert_feeds,
            )
            return proof
        root = f"{inputs.output_root}/composed"
        if context is None:
            with (
                local_client() as client,
                ZephyrContext(
                    client=client,
                    max_workers=COMPOSE_WORKERS,
                    resources=ResourceConfig(cpu=COMPOSE_WORKER_CPU, ram=COMPOSE_WORKER_MEMORY),
                    chunk_storage_prefix=f"{root}/chunks",
                    name="atlas-curriculum-compose",
                    stage_runner_factory=SubprocessRunner,
                ) as zephyr,
                CampaignRuntime().activate(zephyr),
            ):
                return _finish_compose(inputs, plan, document, release, zephyr, root, alert_feeds, mode)
        return _finish_compose(inputs, plan, document, release, context, root, alert_feeds, mode)
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id, "curriculum_compose", {"mode": mode, "error": str(error)[:500]}, alert_feeds
        )
        raise


def check_archived_assignment_counts(retained_files: list[dict], archived_sources) -> None:
    """Every archived source's assignment rows must match its release task count.

    ``filter_assignment_file`` counts the archived file before projection, so the
    counts cover planned removals too; the projected rows are verified separately by
    readback.
    """
    measured: Counter[str] = Counter()
    for result in retained_files:
        for row in result["counts"]:
            measured[row["source_id"]] += row["count"]
    if measured != Counter({source.source_id: source.task_count for source in archived_sources}):
        raise ValueError("retained assignment counts disagree with the retained release sources")


def _finish_compose(inputs, plan, document, release, context, root, alert_feeds, mode) -> dict:
    """The write-mode tail: retained pass, coverage recompute, exact unions, seals."""
    old_sources = {source.source_id: source for source in plan.old_release.sources}
    removed = frozenset(plan.removed_sources)
    selected = frozenset(old_sources) - removed
    retained_files = context.execute(
        Dataset.from_list(list(enumerate(plan.old_release.paths))).map(
            partial(
                filter_assignment_file,
                output_root=f"{root}/retained",
                archived_sources=tuple(plan.old_release.sources),
                selected_ids=selected,
                rebindings=(),
            )
        )
    ).results
    if not all(row["all_retained_values_verified"] for row in retained_files):
        raise ValueError("retained assignment readback diverged")
    check_archived_assignment_counts(retained_files, plan.old_release.sources)
    assignment_paths = [row["output"] for row in retained_files] + list(plan.fresh_release.paths)
    release = release.model_copy(update={"paths": assignment_paths})
    release = AlignmentRelease.model_validate(release.model_dump(mode="json"))
    release_ref = seal(root, "release.json", release.model_dump(mode="json"))
    samples = build_curriculum_coverage(
        context,
        task_paths=release.paths,
        nodes=release.nodes,
        output_root=f"{root}/sample-merge",
        subject_distance=plan.old_coverage.subject_distance,
        capability_distance=plan.old_coverage.capability_distance,
        source_sample_size=plan.old_coverage.source_sample_size,
        area_sample_size=plan.old_coverage.area_sample_size,
        sampling_source_ids={source.source_id: original_source_id(source.source_id) for source in plan.sources},
        confidence_levels=tuple(
            (level.label, level.subject_min_similarity, level.capability_min_similarity)
            for level in plan.old_coverage.confidence_levels
        ),
    )
    # Statistic keys lead with source_id; a removed source's statistics are not carried.
    expected_statistics = {
        key: value for key, value in statistics_by_key(plan.old_coverage.paths).items() if key[0] not in removed
    }
    fresh_statistics = statistics_by_key(plan.fresh_coverage.paths)
    if expected_statistics.keys() & fresh_statistics.keys():
        raise ValueError("retained and fresh coverage statistics overlap")
    expected_statistics.update(fresh_statistics)
    if statistics_by_key(samples.paths) != expected_statistics:
        raise ValueError("recomputed statistics are not the exact retained-plus-fresh union")
    expected_confidence = {
        key: value
        for key, value in statistics_by_key(plan.old_coverage.confidence_paths).items()
        if key[0] not in removed
    }
    fresh_confidence = statistics_by_key(plan.fresh_coverage.confidence_paths)
    if expected_confidence.keys() & fresh_confidence.keys():
        raise ValueError("retained and fresh confidence vectors overlap")
    expected_confidence.update(fresh_confidence)
    if statistics_by_key(samples.confidence_paths) != expected_confidence:
        raise ValueError("recomputed confidence vectors are not the exact retained-plus-fresh union")
    identity = compose_coverage_identity(plan, release)
    coverage = plan.old_coverage.model_copy(
        update={
            "coverage_id": canonical_sha256(identity),
            "alignment_release_id": release.release_id,
            "task_count": release.task_count,
            "paths": samples.paths,
            "sample_paths": samples.sample_paths,
            "sample_task_count": samples.sample_task_count,
            "confidence_paths": samples.confidence_paths,
            "version": COVERAGE_VERSION,
        }
    )
    coverage = CurriculumCoverage.model_validate(coverage.model_dump(mode="json"))
    coverage_ref = seal(root, "coverage.json", coverage.model_dump(mode="json"))
    release_reference = {**release_ref.model_dump(mode="json"), "release_id": release.release_id}
    coverage_reference = {**coverage_ref.model_dump(mode="json"), "coverage_id": coverage.coverage_id}
    confidence, validated = confidence_export(root, release, coverage, release_ref, coverage_ref)
    receipt = {
        "kind": f"atlas-{inputs.release_id}-{COMPOSE_RECEIPT_SUFFIX}",
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "release": release_reference,
        "coverage": coverage_reference,
        "confidence": {"uri": confidence.uri, "sha256": confidence.sha256},
        "source_count": len(plan.sources),
        "catalog_source_count": len(plan.inventory),
        "inventory_count": len(plan.inventory),
        "task_count": release.task_count,
        "fresh_sources": list(plan.targets),
        "retained_source_count": len(selected),
        "removed_sources": sorted(removed),
        "retained_scores_byte_identical": True,
        "non_target_sources_and_inventory_unchanged": True,
        "statistics_exact_union": True,
        "confidence_vectors_exact_union": True,
        "samples_recomputed_from_full_assignments": True,
        # Load-facing handoff keys (curriculum_load.verify_handoff): each flag is earned
        # by the checks above this seal — the retained readback verified every assignment
        # value byte for byte, the statistics/confidence unions are exact, and the
        # additions-mode compose never rewrites task ids or public inputs.
        "all_retained_assignment_values_verified": all(row["all_retained_values_verified"] for row in retained_files),
        "all_statistics_and_confidence_values_unchanged": True,
        "all_task_ids_unchanged": True,
        "all_public_inputs_unchanged": True,
        "rebound_source": "",
        "rebound_tasks": 0,
        "confidence_validated_rows": validated,
        "sample_task_count": coverage.sample_task_count,
        "model_calls": 0,
        "sql_writes": 0,
        "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "inputs": {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256},
    }
    handoff = seal(root, "handoff.json", receipt)
    proof = {**receipt, "handoff": {"uri": handoff.uri, "sha256": handoff.sha256}, "dry_run": False}
    alerts.stage_finished(
        inputs.release_id,
        "curriculum_compose",
        {"mode": mode, "task_count": release.task_count, "handoff": proof["handoff"]},
        alert_feeds,
    )
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CurriculumComposeInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["inputs"] = ArtifactRef(**document["inputs"])
    inputs = CurriculumComposeInputs(**document)
    proof = run_compose(inputs, dry_run=args.mode == "dry-run")
    label = "COMPOSE-PREFLIGHT-PASS" if args.mode == "dry-run" else "COMPOSE-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
