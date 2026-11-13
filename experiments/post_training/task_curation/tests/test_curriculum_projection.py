# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import SourceDeclaration, SourceOrigin, source_identifier
from finestore.schema import arrow_schema
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, AlignmentSource, CurriculumCoverage
from taskcompendium.pipeline.curriculum.task_alignment import TaskAlignment
from taskcompendium.pipeline.semantics import TaskSemanticInput
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.curriculum_projection import (
    ProjectionInputs,
    SourceReplacement,
    project_curriculum,
    projection_selection,
    task_rebinding,
)
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.local import local_client
from infra.marina.apps.rl_data_catalog.role_projection import CurriculumManifests, ManifestReference
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def declaration(identifier, revision="1"):
    return SourceDeclaration(
        id=source_identifier(SourceOrigin.MARIN, identifier, None, None),
        name=identifier,
        display_name=identifier,
        origin="Marin",
        build_path="fixture",
        upstream_dataset=identifier,
        upstream_name=identifier,
        current_release={"repo": "fixture", "revision": revision, "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=1,
        within_source_dedup=deduplication_evidence("fixture", revision, "1"),
        notes="",
        dataset_id="fixture",
        dataset_revision=revision,
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": revision, "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
    )


def reference(path):
    return ManifestReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))


def public_source(root, declaration, *, public="same-public-instruction", task_hash="old-task"):
    root.mkdir()
    manifest = {
        "intended_use": "train",
        "input_rows": 1,
        "converted_rows": 1,
        "rejections": {},
        "input": {"declaration": declaration.model_dump(mode="json"), "task_spec_scan_sha256": "a" * 64},
    }
    source_key = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    row = TaskSemanticInput(
        source_id=declaration.id,
        source_key=source_key,
        task_id="same-id",
        task_hash=task_hash,
        content_hash=canonical_sha256(public),
        intended_use="train",
        population="converted",
        public_input=public,
    )
    pq.write_table(
        pa.Table.from_pylist([row.model_dump()], schema=arrow_schema(TaskSemanticInput)), root / "tasks.parquet"
    )
    (root / "public-inputs.json").write_text(
        json.dumps({"source_key": source_key, "stored_stream_sha256": "a" * 64, "paths": [str(root / "tasks.parquet")]})
    )
    source = AlignmentSource(
        source_id=declaration.id,
        source_key=source_key,
        population="converted",
        intended_use="train",
        task_count=1,
        ready_count=1,
        insufficient_context_count=0,
        basis="matched",
        curation_root=str(root),
        input_rows=1,
        converted_rows=1,
    )
    return source, row


@pytest.fixture
def archived(tmp_path):
    declarations = [declaration(identifier) for identifier in ("Marin:kept", "Marin:rebound", "Marin:retired")]
    prepared = [public_source(tmp_path / f"source-{index}", row) for index, row in enumerate(declarations)]
    rows = []
    for _, public in prepared:
        rows.append(
            {
                **{
                    key: public.model_dump()[key]
                    for key in (
                        "source_id",
                        "source_key",
                        "task_id",
                        "task_hash",
                        "content_hash",
                        "intended_use",
                        "population",
                    )
                },
                "description_status": "ready",
                "reason": "",
                "description": "same-public-instruction",
                "task_preview": "same-public-instruction",
                "instruction_characters": 23,
                "preview_truncated": False,
                "matches": {
                    "subjects": [
                        {
                            "subject_id": "math",
                            "facet": "subject_domain",
                            "membership_similarity": 0.875,
                            "membership_anchor_id": "subject-anchor",
                            "capability_id": "add",
                            "margin": 0.125,
                            "candidates": [
                                {
                                    "subject_id": "math",
                                    "capability_id": "add",
                                    "similarity": 0.9375,
                                    "anchor_id": "add-anchor",
                                }
                            ],
                            "subject_candidates": [
                                {"subject_id": "math", "similarity": 0.875, "anchor_id": "subject-anchor"}
                            ],
                        }
                    ],
                    "capabilities": [],
                },
            }
        )
    assignments = tmp_path / "archive-assignments.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema(TaskAlignment)), assignments)
    lineage = {
        "alignment_root": str(tmp_path),
        "task_vectors_root": "unchanged-vectors",
        "curriculum_vectors_root": "unchanged-anchors",
    }
    sources = [source for source, _ in prepared]
    release = AlignmentRelease(
        release_id=canonical_sha256(
            {
                "lineage": lineage,
                "sources": [row.model_dump() for row in sources],
                "inventory": [row.model_dump() for row in declarations],
            }
        ),
        created_at="2026-11-13T00:00:00Z",
        curriculum_key="same-curriculum",
        population="converted",
        sample_size=None,
        task_count=3,
        ready_count=3,
        insufficient_context_count=0,
        sources=sources,
        inventory=declarations,
        nodes=[
            {
                "subject_id": "math",
                "node_id": "subject:math",
                "parent_id": None,
                "kind": "subject",
                "name": "Math",
                "description": "Mathematics",
            },
            {
                "subject_id": "math",
                "node_id": "add",
                "parent_id": "subject:math",
                "kind": "capability",
                "name": "Addition",
                "description": "Add numbers",
            },
        ],
        paths=[str(assignments)],
        lineage=lineage,
        embedding_spec={
            "model": "unused",
            "revision": "1",
            "tokenizer_revision": "1",
            "dimensions": 2,
            "max_tokens": 16,
            "compute_dtype": "float32",
        },
    )
    coverage_identity = {
        "alignment_release_id": release.release_id,
        "subject_distance": 0.45,
        "capability_distance": 0.25,
        "source_sample_size": 1,
        "area_sample_size": 1,
        "version": COVERAGE_VERSION,
    }
    coverage = CurriculumCoverage(
        **{key: value for key, value in coverage_identity.items() if key != "version"},
        coverage_id=canonical_sha256(coverage_identity),
        curriculum_key=release.curriculum_key,
        task_count=3,
        paths=[],
        sample_paths=[],
        sample_task_count=0,
    )
    statistics = [
        {"source_id": source.source_id, "metric": metric, "subject_id": subject, "node_id": node, "instances": 1}
        for source in sources
        for metric, subject, node in (
            ("denominator", "", ""),
            ("assigned", "", ""),
            ("placement", "math", "subject:math"),
            ("placement", "math", "add"),
            ("capability", "math", "subject:math"),
        )
    ]
    pq.write_table(pa.Table.from_pylist(statistics), tmp_path / "archive-statistics.parquet")
    coverage = coverage.model_copy(update={"paths": [str(tmp_path / "archive-statistics.parquet")]})
    (tmp_path / "archive-release.json").write_text(release.model_dump_json())
    (tmp_path / "archive-coverage.json").write_text(coverage.model_dump_json())
    manifests = CurriculumManifests(
        release_uri=str(tmp_path / "archive-release.json"),
        release_sha256=file_sha256(StoragePath(str(tmp_path / "archive-release.json"))),
        release_id=release.release_id,
        coverage_uri=str(tmp_path / "archive-coverage.json"),
        coverage_sha256=file_sha256(StoragePath(str(tmp_path / "archive-coverage.json"))),
        coverage_id=coverage.coverage_id,
    )
    return release, declarations, rows, manifests


def test_projection_rebinds_only_provenance_and_rebuilds_selected_samples(tmp_path, archived):
    archive, declarations, original_rows, manifests = archived
    changed = declaration("Marin:rebound", "2")
    new_source, _ = public_source(tmp_path / "repaired", changed, task_hash="repaired-task")
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "repaired/manifest.json"),
        public_inputs=reference(tmp_path / "repaired/public-inputs.json"),
    )
    current = [declarations[0].model_copy(update={"notes": "New content review"}), changed]
    (tmp_path / "live.json").write_text(json.dumps([row.model_dump(mode="json") for row in current]))
    (tmp_path / "retirements.json").write_text(
        json.dumps({"entries": [{"id": declarations[2].id, "disposition": "retired_admission_check"}]})
    )
    (tmp_path / "verification.json").write_text(
        json.dumps(
            {
                "release_id": archive.release_id,
                "coverage_id": manifests.coverage_id,
                "release_sha256": manifests.release_sha256,
                "coverage_sha256": manifests.coverage_sha256,
                "all_frozen_pins_and_roles_match": True,
                "coverage_denominators_and_outcomes_reconciled": True,
            }
        )
    )
    inputs = ProjectionInputs(
        archive=manifests,
        archive_verification=reference(tmp_path / "verification.json"),
        live_inventory=reference(tmp_path / "live.json"),
        retirements=reference(tmp_path / "retirements.json"),
        catalog_git_revision="a" * 40,
        replacements=[replacement],
    )
    (tmp_path / "inputs.json").write_text(inputs.model_dump_json())
    saved = (tmp_path / "archive-assignments.parquet").read_bytes()
    runtime = CampaignRuntime()
    with (
        local_client() as client,
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        project_curriculum(
            runtime,
            inputs=inputs,
            inputs_reference=reference(tmp_path / "inputs.json"),
            output_root=str(tmp_path / "projected"),
        )
    release = AlignmentRelease.model_validate_json((tmp_path / "projected/release.json").read_text())
    coverage = CurriculumCoverage.model_validate_json((tmp_path / "projected/coverage.json").read_text())
    projected = [row for path in release.paths for row in load_parquet(path)]
    expected = [dict(row) for row in original_rows[:2]]
    expected[1].update(source_key=new_source.source_key, task_hash="repaired-task")
    assert projected == expected
    assert release.task_count == coverage.task_count == 2
    assert [row.id for row in release.inventory] == [row.id for row in current]
    assert {row["source_id"] for path in coverage.paths for row in load_parquet(path)} == {row.id for row in current}
    assert {row["source_id"] for path in coverage.sample_paths for row in load_parquet(path)} == {
        row.id for row in current
    }
    assert coverage.sample_task_count == 2
    assert release.lineage["task_vectors_root"] == "unchanged-vectors"
    assert (tmp_path / "archive-assignments.parquet").read_bytes() == saved


def test_changed_pin_requires_rebinding_and_changed_public_content_rejects_reuse(tmp_path, archived):
    archive, declarations, _, _ = archived
    changed = declaration("Marin:rebound", "2")
    with pytest.raises(ValueError, match="without a verified public-input rebind"):
        projection_selection(archive, [declarations[0], changed], frozenset({declarations[2].id}), ())
    public_source(tmp_path / "changed-public", changed, public="changed instructions", task_hash="new-task")
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "changed-public/manifest.json"),
        public_inputs=reference(tmp_path / "changed-public/public-inputs.json"),
    )
    with pytest.raises(ValueError, match="Public task content changed"):
        task_rebinding(archive.sources[1], replacement)


def test_projection_cannot_silently_drop_a_live_source_or_restore_a_retired_source(archived):
    archive, declarations, _, _ = archived
    for inventory in ([declarations[0]], declarations):
        with pytest.raises(ValueError, match="minus retired sources"):
            projection_selection(archive, inventory, frozenset({declarations[2].id}), ())
