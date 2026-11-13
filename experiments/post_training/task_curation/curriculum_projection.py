# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project an archived curriculum onto an exact current catalog without rescoring."""

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.curation import SourceDeclaration
from fray.types import ResourceConfig
from marin.execution.lazy import OUT, ArtifactStep, apply, run
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.coverage import STATISTIC_FIELDS, build_curriculum_coverage
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, AlignmentSource, CurriculumCoverage
from taskcompendium.pipeline.semantics import TaskSemanticInput
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION, SemanticArtifact
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from infra.marina.apps.rl_data_catalog.role_projection import (
    CurriculumManifests,
    ManifestReference,
    manifest_bytes,
    manifest_documents,
)
from infra.marina.apps.rl_data_catalog.server.quality import intended_use
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

LOGGER = logging.getLogger(__name__)
ASSIGNMENT_BATCH_SIZE = 65536
PROJECTION_VERSION = "2026.11.13.1"
METADATA_FIELDS = frozenset(
    {
        "grading_environment",
        "lexical_contamination",
        "quality_result",
        "quality_result_finalization",
        "quality_result_waiver",
        "independent_qc",
        "notes",
        "status",
    }
)


class SourceReplacement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    source_manifest: ManifestReference
    public_inputs: ManifestReference


class ProjectionInputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    archive: CurriculumManifests
    archive_verification: ManifestReference
    live_inventory: ManifestReference
    retirements: ManifestReference
    catalog_git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    replacements: list[SourceReplacement] = Field(default_factory=list)


@dataclass(frozen=True)
class TaskRebinding:
    source: AlignmentSource
    declaration: SourceDeclaration
    task_ids: tuple[str, ...]
    old_task_hashes: tuple[str, ...]
    new_task_hashes: tuple[str, ...]
    content_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSelection:
    sources: list[AlignmentSource]
    inventory: list[SourceDeclaration]
    metadata_changes: dict[str, list[str]]


def projection_selection(
    archive: AlignmentRelease,
    inventory: list[SourceDeclaration],
    retired: frozenset[str],
    rebindings: tuple[TaskRebinding, ...],
) -> ProjectionSelection:
    """Require the exact non-retired population and explicit proof for changed task inputs."""
    normalized = [canonical_declaration(row) for row in inventory]
    current = {row.id: row for row in normalized}
    old_inventory = {row.id: row for row in archive.inventory}
    old_sources = {row.source_id: row for row in archive.sources}
    replacements = {row.source.source_id: row for row in rebindings}
    if len(current) != len(inventory) or len(old_sources) != len(archive.sources):
        raise ValueError("Projection source IDs must be unique")
    if set(current) != set(old_sources) - retired or set(current) & retired:
        raise ValueError("Live inventory must equal the archived population minus retired sources")
    if replacements.keys() - current.keys() or len(replacements) != len(rebindings):
        raise ValueError("Rebindings must identify distinct live sources")
    sources, changes = [], {}
    for identifier, declaration in current.items():
        old = old_inventory[identifier].model_dump(mode="json")
        new = declaration.model_dump(mode="json")
        changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        source = old_sources[identifier]
        if source.basis != "matched" or source.intended_use != intended_use(new):
            raise ValueError(f"Projection changes a source's population or intended use: {identifier}")
        if identifier in replacements:
            replacement = replacements[identifier]
            prepared = replacement.declaration.model_dump(mode="json")
            if any(
                prepared.get(key) != new.get(key) for key in prepared.keys() | new.keys() if key not in METADATA_FIELDS
            ):
                raise ValueError(f"Replacement task inputs differ from the current pin: {identifier}")
            if replacement.declaration.task_count != declaration.task_count:
                raise ValueError(f"Replacement population differs from the current source: {identifier}")
            source = replacement.source
        elif set(changed) - METADATA_FIELDS or fingerprint_input(old) != fingerprint_input(new):
            raise ValueError(f"Task-producing inputs changed without a verified public-input rebind: {identifier}")
        if source.task_count != declaration.task_count:
            raise ValueError(f"Projection changes a source's task count: {identifier}")
        sources.append(source)
        if changed:
            changes[identifier] = changed
    return ProjectionSelection(sources, list(current.values()), changes)


def _public_tasks(reference: ManifestReference) -> tuple[dict, dict[str, TaskSemanticInput]]:
    document = json.loads(manifest_bytes(reference))
    tasks = {}
    for path in document["paths"]:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=1024):
                for raw in batch.to_pylist():
                    row = TaskSemanticInput.model_validate(raw)
                    if row.task_id in tasks:
                        raise ValueError(f"Repeated public task identity: {row.source_id}/{row.task_id}")
                    tasks[row.task_id] = row
    return document, tasks


def task_rebinding(archive: AlignmentSource, replacement: SourceReplacement) -> TaskRebinding:
    """Prove every public task input unchanged before adopting new TaskSpec provenance."""
    if archive.curation_root is None or archive.source_id != replacement.source_id:
        raise ValueError("A rebind requires the archived source's public-input lineage")
    old_root = StoragePath(archive.curation_root)
    old_manifest = json.loads((old_root / "manifest.json").read_bytes())
    old_public = old_root / "public-inputs.json"
    _, old = _public_tasks(ManifestReference(uri=str(old_public), sha256=file_sha256(old_public)))
    manifest = json.loads(manifest_bytes(replacement.source_manifest))
    public, new = _public_tasks(replacement.public_inputs)
    source_key = canonical_sha256(manifest)
    if (
        canonical_sha256(old_manifest) != archive.source_key
        or public["source_key"] != source_key
        or public["stored_stream_sha256"] != manifest["input"]["task_spec_scan_sha256"]
        or set(old) != set(new)
        or len(new) != archive.task_count
        or len(new) != manifest["converted_rows"]
    ):
        raise ValueError("Rebound public inputs do not match their frozen populations or scans")
    for identifier, before in old.items():
        after = new[identifier]
        fields = ("source_id", "task_id", "content_hash", "intended_use", "population", "public_input")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError(f"Public task content changed; existing scores cannot be reused: {identifier}")
        if before.source_key != archive.source_key or after.source_key != source_key:
            raise ValueError(f"Public task source key differs from its manifest: {identifier}")
    identifiers = tuple(sorted(new))
    declaration = canonical_declaration(SourceDeclaration.model_validate(manifest["input"]["declaration"]))
    if declaration.id != archive.source_id or manifest["intended_use"] != archive.intended_use:
        raise ValueError("Replacement manifest identifies a different source or intended use")
    source = archive.model_copy(
        update={
            "source_key": source_key,
            "curation_root": str(StoragePath(replacement.source_manifest.uri).parent),
            "input_rows": manifest["input_rows"],
            "converted_rows": manifest["converted_rows"],
            "rejections": manifest["rejections"],
        }
    )
    return TaskRebinding(
        source,
        declaration,
        identifiers,
        tuple(old[key].task_hash for key in identifiers),
        tuple(new[key].task_hash for key in identifiers),
        tuple(new[key].content_hash for key in identifiers),
    )


def project_assignment_table(
    table: pa.Table, selected_ids: frozenset[str], rebindings: tuple[TaskRebinding, ...]
) -> pa.Table:
    """Retain selected rows and change only independently verified TaskSpec provenance."""
    table = table.filter(pc.is_in(table["source_id"], value_set=pa.array(sorted(selected_ids))))
    for replacement in rebindings:
        mask = pc.equal(table["source_id"], replacement.source.source_id)
        indices = pc.index_in(table["task_id"], value_set=pa.array(replacement.task_ids))
        if pc.any(pc.is_null(pc.filter(indices, mask))).as_py():
            raise ValueError("Assignment contains a task outside the verified rebind")
        for column, expected in (
            ("task_hash", replacement.old_task_hashes),
            ("content_hash", replacement.content_hashes),
        ):
            actual = pc.filter(table[column], mask)
            desired = pc.filter(pc.take(pa.array(expected), indices), mask)
            if not actual.equals(desired):
                raise ValueError(f"Archived assignment differs from verified public input: {column}")
        for column, values in (
            ("source_key", pa.scalar(replacement.source.source_key)),
            ("task_hash", pc.take(pa.array(replacement.new_task_hashes), indices)),
        ):
            table = table.set_column(
                table.schema.get_field_index(column), table.schema.field(column), pc.if_else(mask, values, table[column])
            )
    return table


def filter_assignment_file(
    indexed_path: tuple[int, str],
    *,
    output_root: str,
    archived_sources: tuple[AlignmentSource, ...],
    selected_ids: frozenset[str],
    rebindings: tuple[TaskRebinding, ...],
) -> dict:
    """Write a selected assignment file and compare every retained value after readback."""
    index, source_path = indexed_path
    destination = StoragePath(output_root) / f"assignments/part-{index:05d}.parquet"
    if destination.exists():
        raise ValueError(f"Preserve prior assignment output: {destination}")
    expected = {row.source_id: row for row in archived_sources}
    counts = Counter()
    groups = 0
    source_fs, source_storage_key = filesystem_for(source_path)
    destination_fs, destination_storage_key = filesystem_for(str(destination))
    with source_fs.open(source_storage_key, "rb", cache_type="none") as stream:
        parquet = pq.ParquetFile(stream)
        with (
            destination.open("wb", auto_mkdir=True) as output,
            pq.ParquetWriter(output, parquet.schema_arrow, compression="zstd") as writer,
        ):
            for batch in parquet.iter_batches(batch_size=ASSIGNMENT_BATCH_SIZE):
                table = pa.Table.from_batches([batch])
                keys = ["source_id", "source_key", "population", "intended_use", "description_status"]
                for row in table.select([*keys, "task_id"]).group_by(keys).aggregate([("task_id", "count")]).to_pylist():
                    source = expected.get(row["source_id"])
                    if source is None or any(row[key] != getattr(source, key) for key in keys[1:4]):
                        raise ValueError("Archived assignment source lineage differs from its release")
                    counts[row["source_id"], row["description_status"]] += row["task_id_count"]
                projected = project_assignment_table(table, selected_ids, rebindings)
                if projected.num_rows:
                    writer.write_table(projected, row_group_size=projected.num_rows)
                    groups += 1
    with (
        source_fs.open(source_storage_key, "rb", cache_type="none") as old_stream,
        destination_fs.open(destination_storage_key, "rb", cache_type="none") as new_stream,
    ):
        written = pq.ParquetFile(new_stream)
        group = 0
        for batch in pq.ParquetFile(old_stream).iter_batches(batch_size=ASSIGNMENT_BATCH_SIZE):
            projected = project_assignment_table(pa.Table.from_batches([batch]), selected_ids, rebindings)
            if projected.num_rows:
                if not projected.equals(written.read_row_group(group), check_metadata=True):
                    raise ValueError("Projected assignment values changed during publication")
                group += 1
        if group != groups or group != written.num_row_groups:
            raise ValueError("Projected assignment row groups do not reconcile")
    return {
        "input": source_path,
        "output": str(destination),
        "sha256": file_sha256(destination),
        "counts": [{"source_id": key[0], "status": key[1], "count": count} for key, count in sorted(counts.items())],
        "all_retained_values_verified": True,
    }


def statistics_by_key(paths: list[str]) -> dict[tuple, int]:
    values = {}
    for path in paths:
        with StoragePath(path).open("rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                key = tuple(row[field] for field in STATISTIC_FIELDS)
                if key in values:
                    raise ValueError("Coverage contains duplicate statistic keys")
                values[key] = row["instances"]
    return values


def project_curriculum(
    runtime: CampaignRuntime, *, inputs: ProjectionInputs, inputs_reference: ManifestReference, output_root: str
) -> SemanticArtifact:
    """Publish a population projection and rebuild bounded examples from existing assignments."""
    if ProjectionInputs.model_validate_json(manifest_bytes(inputs_reference)) != inputs:
        raise ValueError("Projection inputs differ from their immutable reference")
    root = StoragePath(output_root)
    if (root / "release.json").exists() or (root / "coverage.json").exists():
        raise ValueError(f"Preserve previous projection outputs: {root}")
    archived = manifest_documents(inputs.archive)
    archive = AlignmentRelease.model_validate(archived.release)
    old_coverage = CurriculumCoverage.model_validate(archived.coverage)
    verification = json.loads(manifest_bytes(inputs.archive_verification))
    if (
        verification["release_id"] != archive.release_id
        or verification["coverage_id"] != old_coverage.coverage_id
        or verification["release_sha256"] != inputs.archive.release_sha256
        or verification["coverage_sha256"] != inputs.archive.coverage_sha256
        or not verification["all_frozen_pins_and_roles_match"]
        or not verification["coverage_denominators_and_outcomes_reconciled"]
    ):
        raise ValueError("Archive verification refers to different curriculum manifests")
    inventory = [
        canonical_declaration(SourceDeclaration.model_validate(row))
        for row in json.loads(manifest_bytes(inputs.live_inventory))
    ]
    ledger = json.loads(manifest_bytes(inputs.retirements))
    retired = frozenset(
        canonical_source_id(row["id"])
        for row in ledger["entries"]
        if row.get("disposition") == "retired_admission_check"
    )
    old_sources = {row.source_id: row for row in archive.sources}
    rebindings = tuple(task_rebinding(old_sources[row.source_id], row) for row in inputs.replacements)
    selected = projection_selection(archive, inventory, retired, rebindings)
    files = runtime.context.execute(
        Dataset.from_list(list(enumerate(archive.paths))).map(
            partial(
                filter_assignment_file,
                output_root=output_root,
                archived_sources=tuple(archive.sources),
                selected_ids=frozenset(row.source_id for row in selected.sources),
                rebindings=rebindings,
            )
        ),
        name="project-curriculum-assignments",
    ).results
    counts = Counter()
    for record in files:
        for row in record["counts"]:
            counts[row["source_id"], row["status"]] += row["count"]
    for source in archive.sources:
        if (
            sum(value for (identifier, _), value in counts.items() if identifier == source.source_id)
            != source.task_count
            or counts[source.source_id, "ready"] != source.ready_count
            or counts[source.source_id, "insufficient_context"] != source.insufficient_context_count
        ):
            raise ValueError(f"Full archived assignment population differs: {source.source_id}")
    lineage = {
        **archive.lineage,
        "archived_alignment_root": archive.lineage["alignment_root"],
        "alignment_root": str(StoragePath(output_root) / "assignments"),
        "projection_inputs_uri": inputs_reference.uri,
        "projection_inputs_sha256": inputs_reference.sha256,
        "archived_release_uri": inputs.archive.release_uri,
        "archived_release_sha256": inputs.archive.release_sha256,
    }
    identity = {
        "lineage": lineage,
        "sources": [row.model_dump(mode="json") for row in selected.sources],
        "inventory": [row.model_dump(mode="json") for row in selected.inventory],
    }
    release = archive.model_copy(
        update={
            "release_id": canonical_sha256(identity),
            "created_at": datetime.now(UTC).isoformat(),
            "sources": selected.sources,
            "inventory": selected.inventory,
            "lineage": lineage,
            "paths": [row["output"] for row in files],
            "task_count": sum(row.task_count for row in selected.sources),
            "ready_count": sum(row.ready_count for row in selected.sources),
            "insufficient_context_count": sum(row.insufficient_context_count for row in selected.sources),
        }
    )
    (root / "release.json").write_text(release.model_dump_json(indent=2))
    data = build_curriculum_coverage(
        runtime.context,
        task_paths=release.paths,
        nodes=release.nodes,
        output_root=output_root,
        subject_distance=old_coverage.subject_distance,
        capability_distance=old_coverage.capability_distance,
        source_sample_size=old_coverage.source_sample_size,
        area_sample_size=old_coverage.area_sample_size,
        sampling_source_ids={row.source_id: original_source_id(row.source_id) for row in selected.sources},
    )
    selected_ids = {row.source_id for row in selected.sources}
    expected_statistics = {
        key: value for key, value in statistics_by_key(old_coverage.paths).items() if key[0] in selected_ids
    }
    if statistics_by_key(data.paths) != expected_statistics:
        raise ValueError("Selected-source curriculum statistics changed during projection")
    coverage_identity = {
        "alignment_release_id": release.release_id,
        **{
            key: getattr(old_coverage, key)
            for key in ("subject_distance", "capability_distance", "source_sample_size", "area_sample_size")
        },
        "version": COVERAGE_VERSION,
    }
    coverage = old_coverage.model_copy(
        update={
            "coverage_id": canonical_sha256(coverage_identity),
            "alignment_release_id": release.release_id,
            "task_count": release.task_count,
            "paths": data.paths,
            "sample_paths": data.sample_paths,
            "sample_task_count": data.sample_task_count,
        }
    )
    (root / "coverage.json").write_text(coverage.model_dump_json(indent=2))
    corrected = CurriculumManifests(
        release_uri=str(root / "release.json"),
        release_sha256=file_sha256(root / "release.json"),
        release_id=release.release_id,
        coverage_uri=str(root / "coverage.json"),
        coverage_sha256=file_sha256(root / "coverage.json"),
        coverage_id=coverage.coverage_id,
    )
    manifest_documents(corrected)
    receipt = {
        "kind": "source-subset-projection",
        "phase": "retained",
        "representation": "direct",
        "inputs": inputs.model_dump(mode="json"),
        "inputs_reference": inputs_reference.model_dump(mode="json"),
        "corrected": corrected.model_dump(mode="json"),
        "source_count": len(release.sources),
        "task_count": release.task_count,
        "ready_count": release.ready_count,
        "insufficient_context_count": release.insufficient_context_count,
        "excluded_sources": sorted(set(old_sources) - {row.source_id for row in selected.sources}),
        "metadata_changes": selected.metadata_changes,
        "assignment_files": files,
        "rebound_sources": [
            {
                "source_id": row.source.source_id,
                "source_key": row.source.source_key,
                "task_count": len(row.task_ids),
                "all_public_inputs_unchanged": True,
            }
            for row in rebindings
        ],
        "vectors_recomputed": 0,
        "scores_recomputed": 0,
        "embedding_cache_reads": 0,
        "embedding_cache_misses": 0,
        "samples_rebuilt_from_selected_population": True,
        "all_selected_statistics_unchanged": True,
    }
    (root / "projection-receipt.json").write_text(json.dumps(receipt, indent=2))
    return SemanticArtifact(path=output_root, paths=data.paths)


def curriculum_projection_step(
    inputs: ProjectionInputs, reference: ManifestReference, runtime: CampaignRuntime
) -> ArtifactStep[SemanticArtifact]:
    return apply(
        f"rl-semantics/live-projection-{reference.sha256[:16]}",
        partial(project_curriculum, runtime),
        version=PROJECTION_VERSION,
        artifact_type=SemanticArtifact,
        inputs=inputs,
        inputs_reference=reference,
        output_root=OUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--worker-memory", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    reference = ManifestReference(uri=args.inputs, sha256=args.inputs_sha256)
    inputs = ProjectionInputs.model_validate_json(manifest_bytes(reference))
    runtime = CampaignRuntime()
    with artifact_prefix(args.artifact_root):
        step = curriculum_projection_step(inputs, reference, runtime)
        LOGGER.info("Projection output: %s", step.path())
        if not args.run:
            return
        with (
            local_client() as client,
            ZephyrContext(
                client=client,
                max_workers=args.workers,
                resources=ResourceConfig(cpu=4, ram=args.worker_memory),
                coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
                chunk_storage_prefix=str(StoragePath(args.artifact_root) / "chunks"),
                name="atlas-curriculum-projection",
            ) as context,
            runtime.activate(context),
        ):
            result = run(step, max_concurrent=1)[0]
            LOGGER.info("Completed projection at %s", result.path)


if __name__ == "__main__":
    main()
