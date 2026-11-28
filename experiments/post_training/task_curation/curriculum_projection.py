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
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.curation import LexicalUse, QualifiedSourceDeclaration, SourceDeclaration
from finestore.rl.status_taxonomy import DataRestriction
from fray.types import ResourceConfig
from marin.execution.lazy import OUT, ArtifactStep, apply, run
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.curriculum.coverage import STATISTIC_FIELDS, build_curriculum_coverage
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, AlignmentSource, CurriculumCoverage
from taskcompendium.pipeline.semantics import TaskSemanticInput
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION, SemanticArtifact
from experiments.post_training.task_curation.frozen_task_specs import FrozenTaskSpecStream, verified_task_spec_rows
from experiments.post_training.task_curation.lexical_policy import LexicalInputUse
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from infra.marina.apps.rl_data_catalog.role_projection import (
    CurriculumManifests,
    ManifestReference,
    manifest_bytes,
    manifest_documents,
)
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction
from infra.marina.apps.rl_data_catalog.server.source_identity import (
    canonical_source_id,
    canonical_source_map,
    original_source_id,
)

LOGGER = logging.getLogger(__name__)
ASSIGNMENT_BATCH_SIZE = 65536
PROJECTION_VERSION = "2026.11.13.1"
METADATA_FIELDS = frozenset(
    {
        "data_restriction",
        "license_receipt",
        "grading_environment",
        "lexical_contamination",
        "quality_result",
        "quality_result_finalization",
        "quality_result_waiver",
        "independent_qc",
        "intended_use",
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
    source_uses: ManifestReference
    retirements: ManifestReference
    catalog_git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    replacements: list[SourceReplacement] = Field(default_factory=list)


@dataclass(frozen=True)
class TaskRebinding:
    source: AlignmentSource
    declaration: SourceDeclaration
    task_ids: tuple[str, ...]
    new_task_ids: tuple[str, ...]
    old_task_hashes: tuple[str, ...]
    new_task_hashes: tuple[str, ...]
    content_hashes: tuple[str, ...]
    upstream_row_sha256: tuple[str | None, ...]
    removed_task_ids: tuple[str, ...]
    removed_task_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSelection:
    sources: list[AlignmentSource]
    inventory: list[SourceDeclaration]
    metadata_changes: dict[str, list[str]]


class ProjectionSourceMissing(ValueError):
    """An archived population has no declaration in the selected catalog."""

    code = "projection_source_missing"

    def __init__(self, source_id: str):
        self.source_id = source_id
        super().__init__(f"{self.code}: {source_id}")


def projection_inventory(source_ids: list[str], inventory: list[SourceDeclaration]) -> list[SourceDeclaration]:
    """Select current declarations by canonical ID without dropping archived populations."""
    raw = {row.id: row for row in inventory}
    if len(raw) != len(inventory):
        raise ValueError("Projection catalog contains duplicate source IDs")
    current = canonical_source_map(raw)
    selected = []
    for source_id in source_ids:
        identifier = canonical_source_id(source_id)
        if identifier not in current:
            raise ProjectionSourceMissing(identifier)
        selected.append(canonical_declaration(current[identifier]))
    return selected


def projection_selection(
    archive: AlignmentRelease,
    inventory: list[SourceDeclaration],
    retired: frozenset[str],
    rebindings: tuple[TaskRebinding, ...],
    *,
    source_uses: list[LexicalInputUse],
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
    uses = {row.source_id: row for row in source_uses}
    if set(uses) != set(current) or len(uses) != len(source_uses):
        raise ValueError("Registered Use bindings must cover every live source exactly once")
    sources, changes = [], {}
    for identifier, declaration in current.items():
        old = old_inventory[identifier].model_dump(mode="json")
        new = declaration.model_dump(mode="json")
        changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        source = old_sources[identifier]
        binding = uses[identifier]
        if declaration.pipeline is None or (
            binding.dataset_id,
            binding.dataset_revision,
            binding.converter_revision,
        ) != (declaration.dataset_id, declaration.dataset_revision, declaration.pipeline.version):
            raise ValueError(f"Registered Use belongs to a different converter or dataset pin: {identifier}")
        expected_use = "train" if binding.use is LexicalUse.TRAIN else "eval"
        if declaration.intended_use is not None and declaration.intended_use.value != expected_use:
            raise ValueError(f"Declared Use differs from the registered converter: {identifier}")
        if source.basis != "matched" or source.intended_use != expected_use:
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


def public_task_correspondence(
    old: dict[str, TaskSemanticInput], new: dict[str, TaskSemanticInput], upstream_rows: dict[str, str]
) -> dict[str, str]:
    """Match stable IDs and unique public content, using verified upstream rows for repeats."""
    matches = {identifier: identifier for identifier in old.keys() & new.keys()}
    unmatched_old = old.keys() - matches.keys()
    unmatched_new = new.keys() - matches.values()
    upstream_new_ids = set(upstream_rows.values())
    by_content = {}
    for identifier in unmatched_new:
        if identifier in upstream_new_ids:
            continue
        content_hash = new[identifier].content_hash
        if content_hash in by_content:
            raise ValueError("Changed task IDs have ambiguous public inputs; existing scores cannot be reused")
        by_content[content_hash] = identifier
    for identifier in unmatched_old:
        if identifier in upstream_rows:
            matches[identifier] = upstream_rows[identifier]
            continue
        candidate = by_content.pop(old[identifier].content_hash, None)
        if candidate is not None:
            matches[identifier] = candidate
    if by_content:
        raise ValueError("Changed task IDs leave unmatched public inputs; existing scores cannot be reused")
    if len(matches) != len(new) or len(set(matches.values())) != len(new):
        raise ValueError("Current public inputs lack an injective archived match; existing scores cannot be reused")
    return matches


def pinned_upstream_rows(
    manifest: dict, semantic_tasks: dict[str, TaskSemanticInput], selected: set[str]
) -> dict[tuple[str, str, str], tuple[str, str]]:
    """Bind upstream row identities to exact TaskSpecs from the complete frozen stream."""
    frozen = manifest["input"]
    stream = FrozenTaskSpecStream.model_validate(
        {
            "source_id": frozen["declaration"]["id"],
            **{key: frozen.get(key) for key in FrozenTaskSpecStream.model_fields if key != "source_id"},
        }
    )
    rows = {}
    upstream_counts = Counter()
    for encoded in verified_task_spec_rows(stream):
        document = json.loads(encoded)
        source = document["source"]
        key = (source["dataset"], source["revision"], source["row"])
        upstream_counts[key] += 1
        if document["id"] not in selected:
            continue
        task = TaskSpec.model_validate_json(json.dumps(document))
        if task_spec_sha256(task) != semantic_tasks[task.id].task_hash:
            raise ValueError(f"Upstream identity TaskSpec differs from its public manifest: {task.id}")
        if key in rows:
            raise ValueError("Repeated public input has no unique pinned upstream row")
        rows[key] = (task.id, canonical_sha256(key))
    if {row[0] for row in rows.values()} != selected or any(upstream_counts[key] != 1 for key in rows):
        raise ValueError("Upstream rows do not uniquely cover the selected frozen tasks")
    return rows


def ambiguous_upstream_correspondence(
    old_manifest: dict, new_manifest: dict, old: dict[str, TaskSemanticInput], new: dict[str, TaskSemanticInput]
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve repeated public inputs only through a unique immutable upstream row identity."""
    old_unmatched = old.keys() - new.keys()
    new_unmatched = new.keys() - old.keys()
    old_counts = Counter(old[key].content_hash for key in old_unmatched)
    new_counts = Counter(new[key].content_hash for key in new_unmatched)
    repeated = {key for key in old_counts.keys() | new_counts.keys() if old_counts[key] > 1 or new_counts[key] > 1}
    if not repeated:
        return {}, {}
    before = pinned_upstream_rows(old_manifest, old, {key for key in old_unmatched if old[key].content_hash in repeated})
    after = pinned_upstream_rows(new_manifest, new, {key for key in new_unmatched if new[key].content_hash in repeated})
    if not after.keys() <= before.keys():
        raise ValueError("Repeated public inputs lack complete pinned upstream-row matches")
    return (
        {before[key][0]: after[key][0] for key in after},
        {before[key][0]: before[key][1] for key in after},
    )


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
        or not new
        or len(new) > archive.task_count
        or len(old) != archive.task_count
        or len(new) != manifest["converted_rows"]
    ):
        raise ValueError("Rebound public inputs do not match their frozen populations or scans")
    upstream_rows, upstream_digests = ambiguous_upstream_correspondence(old_manifest, manifest, old, new)
    matches = public_task_correspondence(old, new, upstream_rows)
    for identifier in matches:
        before = old[identifier]
        after = new[matches[identifier]]
        fields = ("source_id", "content_hash", "intended_use", "population", "public_input")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError(f"Public task content changed; existing scores cannot be reused: {identifier}")
        if before.source_key != archive.source_key or after.source_key != source_key:
            raise ValueError(f"Public task source key differs from its manifest: {identifier}")
    identifiers = tuple(sorted(matches))
    removed = tuple(sorted(old.keys() - matches.keys()))
    if removed and archive.ready_count not in (0, archive.task_count):
        raise ValueError("A retained subset with mixed readiness requires fresh alignment")
    declaration = canonical_declaration(QualifiedSourceDeclaration.model_validate(manifest["input"]["declaration"]))
    if (
        declaration.id != archive.source_id
        or declaration.task_count != len(new)
        or manifest["intended_use"] != archive.intended_use
    ):
        raise ValueError("Replacement manifest identifies a different source, population count or intended use")
    source = archive.model_copy(
        update={
            "source_key": source_key,
            "curation_root": str(StoragePath(replacement.source_manifest.uri).parent),
            "input_rows": manifest["input_rows"],
            "converted_rows": manifest["converted_rows"],
            "rejections": manifest["rejections"],
            "task_count": len(new),
            "ready_count": (len(new) if archive.ready_count else 0) if removed else archive.ready_count,
            "insufficient_context_count": (
                (0 if archive.ready_count else len(new)) if removed else archive.insufficient_context_count
            ),
        }
    )
    return TaskRebinding(
        source,
        declaration,
        identifiers,
        tuple(matches[key] for key in identifiers),
        tuple(old[key].task_hash for key in identifiers),
        tuple(new[matches[key]].task_hash for key in identifiers),
        tuple(new[matches[key]].content_hash for key in identifiers),
        tuple(upstream_digests.get(key) for key in identifiers),
        removed,
        tuple(old[key].task_hash for key in removed),
    )


def project_assignment_table(
    table: pa.Table, selected_ids: frozenset[str], rebindings: tuple[TaskRebinding, ...]
) -> pa.Table:
    """Retain selected rows and change only independently verified TaskSpec provenance."""
    table = table.filter(pc.is_in(table["source_id"], value_set=pa.array(sorted(selected_ids))))
    for replacement in rebindings:
        if replacement.removed_task_ids:
            removed_indices = pc.index_in(table["task_id"], value_set=pa.array(replacement.removed_task_ids))
            removed = pc.and_(pc.equal(table["source_id"], replacement.source.source_id), pc.is_valid(removed_indices))
            expected = pc.filter(pc.take(pa.array(replacement.removed_task_hashes), removed_indices), removed)
            if not pc.filter(table["task_hash"], removed).equals(expected):
                raise ValueError("Removed assignment differs from its archived public input")
            table = table.filter(pc.invert(removed))
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
            ("task_id", pc.take(pa.array(replacement.new_task_ids), indices)),
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
    destination = StoragePath(output_root) / f"assignments/part-{index:05d}-{uuid4().hex}.parquet"
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


def statistics_by_key(paths: list[str]) -> dict[tuple, Any]:
    values = {}
    for path in paths:
        with StoragePath(path).open("rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                key = tuple(row[field] for field in STATISTIC_FIELDS)
                if key in values:
                    raise ValueError("Coverage contains duplicate statistic keys")
                values[key] = row["instances"]
    return values


def verify_retained_statistics(archived: dict, projected: dict, reduced_counts: dict[str, int]) -> None:
    """Check exact unaffected values and subset bounds for reduced populations."""
    if {key: value for key, value in archived.items() if key[0] not in reduced_counts} != {
        key: value for key, value in projected.items() if key[0] not in reduced_counts
    }:
        raise ValueError("Unaffected curriculum statistics changed during projection")
    for source_id, count in reduced_counts.items():
        denominator = np.asarray(projected.get((source_id, "denominator", "", ""), []))
        if not denominator.size or np.any(denominator != count):
            raise ValueError(f"Reduced population denominator differs from its retained tasks: {source_id}")
    for key, value in projected.items():
        if key[0] not in reduced_counts:
            continue
        current = np.asarray(value)
        before = np.asarray(archived.get(key, np.zeros_like(current)))
        if current.shape != before.shape or np.any(current < 0) or np.any(current > before):
            raise ValueError(f"Reduced population statistic exceeds its archived value: {key}")


def write_task_identity_mappings(
    root: StoragePath, rebindings: tuple[TaskRebinding, ...]
) -> dict[str, ManifestReference]:
    """Publish immutable old/new task identities with the shared public-content digest."""
    task_mappings = {}
    for row in rebindings:
        removed = (None,) * len(row.removed_task_ids)
        path = root / "task-identity-mappings" / f"{canonical_sha256(row.source.source_id)}.parquet"
        with path.open("wb", auto_mkdir=True) as stream:
            pq.write_table(
                pa.table(
                    {
                        "old_task_id": row.task_ids + row.removed_task_ids,
                        "new_task_id": row.new_task_ids + removed,
                        "old_task_sha256": row.old_task_hashes + row.removed_task_hashes,
                        "new_task_sha256": row.new_task_hashes + removed,
                        "public_content_sha256": row.content_hashes + removed,
                        "upstream_row_sha256": pa.array(row.upstream_row_sha256 + removed, type=pa.string()),
                    }
                ),
                stream,
                compression="zstd",
            )
        task_mappings[row.source.source_id] = ManifestReference(uri=str(path), sha256=file_sha256(path))
    return task_mappings


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
        canonical_declaration(QualifiedSourceDeclaration.model_validate(row))
        for row in json.loads(manifest_bytes(inputs.live_inventory))
    ]
    private = sorted(
        row.id
        for row in inventory
        if declaration_restriction(row.model_dump(mode="json")) is DataRestriction.PRIVATE_ALWAYS
    )
    if private:
        # A release carries task previews and details for every source: a
        # PRIVATE_ALWAYS source is refused at this shared export path, and the
        # refusal also catches a flipped or dropped tag on the release input.
        raise ValueError(f"Curriculum release contains a PRIVATE_ALWAYS source: {private}")
    ledger = json.loads(manifest_bytes(inputs.retirements))
    retired = frozenset(
        canonical_source_id(row["id"])
        for row in ledger["entries"]
        if row.get("disposition") == "retired_admission_check"
    )
    old_sources = {row.source_id: row for row in archive.sources}
    rebindings = tuple(task_rebinding(old_sources[row.source_id], row) for row in inputs.replacements)
    task_mappings = write_task_identity_mappings(root, rebindings)
    source_uses = [LexicalInputUse.model_validate(row) for row in json.loads(manifest_bytes(inputs.source_uses))]
    selected = projection_selection(archive, inventory, retired, rebindings, source_uses=source_uses)
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
        confidence_levels=tuple(
            (level.label, level.subject_min_similarity, level.capability_min_similarity)
            for level in old_coverage.confidence_levels
        ),
    )
    selected_ids = {row.source_id for row in selected.sources}
    expected_statistics = {
        key: value for key, value in statistics_by_key(old_coverage.paths).items() if key[0] in selected_ids
    }
    reduced_counts = {row.source.source_id: len(row.task_ids) for row in rebindings if row.removed_task_ids}
    verify_retained_statistics(expected_statistics, statistics_by_key(data.paths), reduced_counts)
    coverage_identity = {
        "alignment_release_id": release.release_id,
        **{
            key: getattr(old_coverage, key)
            for key in ("subject_distance", "capability_distance", "source_sample_size", "area_sample_size")
        },
        "version": COVERAGE_VERSION,
        **(
            {"confidence_levels": [level.model_dump(mode="json") for level in old_coverage.confidence_levels]}
            if old_coverage.confidence_levels
            else {}
        ),
    }
    coverage = old_coverage.model_copy(
        update={
            "coverage_id": canonical_sha256(coverage_identity),
            "alignment_release_id": release.release_id,
            "task_count": release.task_count,
            "paths": data.paths,
            "sample_paths": data.sample_paths,
            "sample_task_count": data.sample_task_count,
            "confidence_paths": data.confidence_paths,
            "version": COVERAGE_VERSION,
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
                "removed_task_count": len(row.removed_task_ids),
                "all_public_inputs_unchanged": True,
                "changed_task_id_count": sum(
                    old != new for old, new in zip(row.task_ids, row.new_task_ids, strict=True)
                ),
                "task_identity_mapping": task_mappings[row.source.source_id].model_dump(mode="json"),
                "pinned_upstream_row_matches": sum(value is not None for value in row.upstream_row_sha256),
            }
            for row in rebindings
        ],
        "vectors_recomputed": 0,
        "scores_recomputed": 0,
        "embedding_cache_reads": 0,
        "embedding_cache_misses": 0,
        "samples_rebuilt_from_selected_population": True,
        "all_selected_statistics_unchanged": not reduced_counts,
        "retained_subset_statistics_verified": True,
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
