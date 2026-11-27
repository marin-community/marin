# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import LexicalUse, QualifiedSourceDeclaration, SourceOrigin, SourceUse, source_identifier
from finestore.rl.licenses import LicenseReceipt
from finestore.schema import arrow_schema
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.curriculum.coverage import CONFIDENCE_LEVELS, build_curriculum_coverage
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, AlignmentSource, CurriculumCoverage
from taskcompendium.pipeline.curriculum.task_alignment import TaskAlignment
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.semantics import TaskInputFile, TaskSemanticInput, task_semantic_input
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.curriculum_projection import (
    ProjectionInputs,
    ProjectionSourceMissing,
    SourceReplacement,
    curriculum_projection_step,
    filter_assignment_file,
    projection_inventory,
    projection_selection,
    task_rebinding,
)
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.lexical_policy import LexicalInputUse
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.release_stages import curriculum_compose as compose
from experiments.post_training.task_curation.release_stages import curriculum_load
from experiments.post_training.task_curation.release_stages import public_input_reuse as reuse
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.role_projection import CurriculumManifests, ManifestReference
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def declaration(identifier, revision="1"):
    return QualifiedSourceDeclaration(
        id=source_identifier(SourceOrigin.MARIN, identifier, None, None),
        display_name=f"Marin · {identifier}",
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


def use_bindings(declarations):
    return [
        LexicalInputUse(
            source_id=row.id,
            dataset_id=row.dataset_id,
            dataset_revision=row.dataset_revision,
            converter_revision=row.pipeline.version,
            use=LexicalUse.TRAIN,
        )
        for row in declarations
    ]


def test_projection_inventory_resolves_live_alias_and_names_missing_source():
    current = declaration("gpqa").model_copy(update={"origin": "GPQA", "id": "MarinSkyRL:gpqa"})
    selected = projection_inventory(["GPQA:gpqa"], [current])
    assert len(selected) == 1
    assert selected[0].id == "GPQA:gpqa"
    assert selected[0].dataset_revision == current.dataset_revision
    assert selected[0].task_count == current.task_count
    with pytest.raises(ProjectionSourceMissing) as failure:
        projection_inventory(["GPQA:gpqa"], [])
    assert failure.value.source_id == "GPQA:gpqa"
    assert failure.value.code == "projection_source_missing"


def reference(path):
    return ManifestReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))


def public_source(
    root, declaration, *, public="same-public-instruction", task_hash="old-task", task_id="same-id", additional_public=()
):
    root.mkdir()
    manifest = {
        "intended_use": "train",
        "input_rows": 1 + len(additional_public),
        "converted_rows": 1 + len(additional_public),
        "rejections": {},
        "input": {"declaration": declaration.model_dump(mode="json"), "task_spec_scan_sha256": "a" * 64},
    }
    source_key = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    row = TaskSemanticInput(
        source_id=declaration.id,
        source_key=source_key,
        task_id=task_id,
        task_hash=task_hash,
        content_hash=canonical_sha256(public),
        intended_use="train",
        population="converted",
        public_input=public,
    )
    rows = [row] + [
        row.model_copy(
            update={
                "task_id": f"{task_id}-extra-{index}",
                "task_hash": f"{task_hash}-extra-{index}",
                "content_hash": canonical_sha256(text),
                "public_input": text,
            }
        )
        for index, text in enumerate(additional_public)
    ]
    pq.write_table(
        pa.Table.from_pylist([r.model_dump() for r in rows], schema=arrow_schema(TaskSemanticInput)),
        root / "tasks.parquet",
    )
    (root / "public-inputs.json").write_text(
        json.dumps({"source_key": source_key, "stored_stream_sha256": "a" * 64, "paths": [str(root / "tasks.parquet")]})
    )
    source = AlignmentSource(
        source_id=declaration.id,
        source_key=source_key,
        population="converted",
        intended_use="train",
        task_count=len(rows),
        ready_count=len(rows),
        insufficient_context_count=0,
        basis="matched",
        curation_root=str(root),
        input_rows=len(rows),
        converted_rows=len(rows),
    )
    return source, row


@pytest.fixture
def archived(tmp_path):
    declarations = [declaration(identifier) for identifier in ("Marin:kept", "Marin:rebound", "Marin:retired")]
    declarations[0] = declarations[0].model_copy(update={"is_benchmark": True})
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
        "confidence_levels": [
            {"label": label, "subject_min_similarity": subject, "capability_min_similarity": capability}
            for label, subject, capability in CONFIDENCE_LEVELS
        ],
    }
    coverage = CurriculumCoverage(
        **coverage_identity,
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


@pytest.mark.parametrize("public", ["same-public-instruction", "changed-public-instruction"])
def test_reuse_receipt_requires_complete_public_equivalence(tmp_path, archived, public, monkeypatch, capsys):
    _, _, _, manifests = archived
    changed = declaration("Marin:rebound", "2")
    public_source(tmp_path / "current", changed, public=public, task_hash="new-task", task_id="current-task-id")
    census = {
        "kind": "atlas-test-admission-population-census",
        "source_revision": "a" * 40,
        "results": [
            {
                "source_id": changed.id,
                "declaration": changed.model_dump(mode="json"),
                "inventory": {"scanned_task_specs_sha256": "a" * 64, "retained_tasks": 1},
                "old_task_specs_or_count_bindings_inherited": False,
                "qa_bindings_inherited": False,
                "public_preparation": {
                    "manifest": reference(tmp_path / "current/manifest.json").model_dump(mode="json"),
                    "public_inputs": reference(tmp_path / "current/public-inputs.json").model_dump(mode="json"),
                },
            }
        ],
    }
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    inputs = reuse.PublicInputReuseInputs(
        release_id="test",
        source_revision="a" * 40,
        archive_release=ArtifactRef(uri=manifests.release_uri, sha256=manifests.release_sha256),
        census_receipt=ArtifactRef(**reference(census_path).model_dump(mode="json")),
        targets=(changed.id,),
        output_root=str(tmp_path / "reuse"),
    )
    feeds = (tmp_path / "alerts.jsonl",)
    if public != "same-public-instruction":
        with pytest.raises(ValueError):
            reuse.run_public_input_reuse(inputs, dry_run=False, feeds=feeds)
        assert not (tmp_path / "reuse").exists()
        return
    dry_run = reuse.run_public_input_reuse(inputs, dry_run=True, feeds=feeds)
    assert dry_run["tasks"] == 1 and not (tmp_path / "reuse").exists()
    cli_inputs = tmp_path / "reuse-inputs.json"
    cli_inputs.write_text(
        json.dumps(
            {
                "release_id": inputs.release_id,
                "source_revision": inputs.source_revision,
                "archive_release": inputs.archive_release.model_dump(mode="json"),
                "census_receipt": inputs.census_receipt.model_dump(mode="json"),
                "targets": inputs.targets,
                "output_root": inputs.output_root,
            }
        )
    )
    monkeypatch.setattr(reuse.alerts, "WATCHDOG_FEED", feeds[0])
    monkeypatch.setattr(sys, "argv", ["public_input_reuse", "--inputs", str(cli_inputs), "--mode", "write"])
    reuse.main()
    result = json.loads(capsys.readouterr().out)
    receipt = json.loads(StoragePath(result["receipt"]["uri"]).read_text())
    assert receipt["status"] == "PASS" and receipt["results"][0]["changed_task_ids"] == 1
    mapping = receipt["task_identity_mappings"][changed.id]
    assert file_sha256(StoragePath(mapping["uri"])) == mapping["sha256"]
    rows = pq.read_table(mapping["uri"]).to_pylist()
    assert [(row["old_task_id"], row["new_task_id"]) for row in rows] == [("same-id", "current-task-id")]
    assert [(row["old_task_sha256"], row["new_task_sha256"]) for row in rows] == [("old-task", "new-task")]


@pytest.mark.parametrize("task_id", ["same-id", "current-task-id"])
def test_projection_rebinds_only_provenance_and_rebuilds_selected_samples(tmp_path, archived, task_id):
    archive, declarations, original_rows, manifests = archived
    changed = declaration("Marin:rebound", "2")
    replacement_hash = "repaired-task"
    new_source, _ = public_source(tmp_path / "repaired", changed, task_hash=replacement_hash, task_id=task_id)
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "repaired/manifest.json"),
        public_inputs=reference(tmp_path / "repaired/public-inputs.json"),
    )
    metadata_receipt = LicenseReceipt(
        source_id=declarations[0].id,
        dataset_id=declarations[0].dataset_id,
        revision=declarations[0].dataset_revision,
        resolved_at="2026-11-19T00:00:00+00:00",
        resolver_version="fixture",
        evidence=[],
        bundled_third_party=False,
        model_generated_outputs=False,
    )
    current = [
        declarations[0].model_copy(
            update={"notes": "New content review", "intended_use": SourceUse.TRAIN, "license_receipt": metadata_receipt}
        ),
        changed,
    ]
    (tmp_path / "uses.json").write_text(json.dumps([row.model_dump(mode="json") for row in use_bindings(current)]))
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
        source_uses=reference(tmp_path / "uses.json"),
        retirements=reference(tmp_path / "retirements.json"),
        catalog_git_revision="a" * 40,
        replacements=[replacement],
    )
    (tmp_path / "inputs.json").write_text(inputs.model_dump_json())
    saved = (tmp_path / "archive-assignments.parquet").read_bytes()
    runtime = CampaignRuntime()
    with (
        artifact_prefix(str(tmp_path / "artifacts")),
        local_client() as client,
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        step = curriculum_projection_step(inputs, reference(tmp_path / "inputs.json"), runtime)
        result = run(step, max_concurrent=1)[0]
    projected_root = Path(result.path)
    release = AlignmentRelease.model_validate_json((projected_root / "release.json").read_text())
    coverage = CurriculumCoverage.model_validate_json((projected_root / "coverage.json").read_text())
    projected = [row for path in release.paths for row in load_parquet(path)]
    expected = [dict(row) for row in original_rows[:2]]
    expected[1].update(source_key=new_source.source_key, task_hash=replacement_hash, task_id=task_id)
    assert projected == expected
    assert release.task_count == coverage.task_count == 2
    assert [row.id for row in release.inventory] == [row.id for row in current]
    assert {row["source_id"] for path in coverage.paths for row in load_parquet(path)} == {row.id for row in current}
    assert {row["source_id"] for path in coverage.sample_paths for row in load_parquet(path)} == {
        row.id for row in current
    }
    assert coverage.sample_task_count == 2
    confidence = [row for path in coverage.confidence_paths for row in load_parquet(path)]
    assert {row["source_id"] for row in confidence} == {row.id for row in current}
    assert {row["source_id"]: row["instances"] for row in confidence if row["metric"] == "assigned"} == {
        row.id: [1] * len(CONFIDENCE_LEVELS) for row in current
    }
    assert release.lineage["task_vectors_root"] == "unchanged-vectors"
    assert (tmp_path / "archive-assignments.parquet").read_bytes() == saved
    receipt = json.loads((projected_root / "projection-receipt.json").read_text())
    mapping = receipt["rebound_sources"][0]["task_identity_mapping"]
    assert file_sha256(StoragePath(mapping["uri"])) == mapping["sha256"]
    assert pq.read_table(mapping["uri"]).to_pylist() == [
        {
            "old_task_id": "same-id",
            "new_task_id": task_id,
            "old_task_sha256": "old-task",
            "new_task_sha256": replacement_hash,
            "public_content_sha256": canonical_sha256("same-public-instruction"),
            "upstream_row_sha256": None,
        }
    ]


def test_changed_pin_requires_rebinding_and_changed_public_content_rejects_reuse(tmp_path, archived):
    archive, declarations, _, _ = archived
    changed = declaration("Marin:rebound", "2")
    with pytest.raises(ValueError, match="without a verified public-input rebind"):
        projection_selection(
            archive,
            [declarations[0], changed],
            frozenset({declarations[2].id}),
            (),
            source_uses=use_bindings([declarations[0], changed]),
        )
    public_source(tmp_path / "changed-public", changed, public="changed instructions", task_hash="new-task")
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "changed-public/manifest.json"),
        public_inputs=reference(tmp_path / "changed-public/public-inputs.json"),
    )
    with pytest.raises(ValueError, match="Public task content changed"):
        task_rebinding(archive.sources[1], replacement)


def test_changed_task_ids_require_full_public_input_equality_even_with_matching_hashes(tmp_path, archived):
    archive, _, _, _ = archived
    changed = declaration("Marin:rebound", "2")
    public_source(tmp_path / "repaired", changed, task_id="new-task-id", task_hash="new-task")
    path = tmp_path / "repaired/tasks.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["public_input"] = "different public instructions behind the same claimed hash"
    pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema(TaskSemanticInput)), path)
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "repaired/manifest.json"),
        public_inputs=reference(tmp_path / "repaired/public-inputs.json"),
    )
    with pytest.raises(ValueError):
        task_rebinding(archive.sources[1], replacement)


def test_changed_task_ids_with_ambiguous_public_inputs_refuse_score_reuse(tmp_path, archived):
    archive, _, _, _ = archived
    changed = declaration("Marin:rebound", "2")
    public_source(tmp_path / "repaired", changed, task_id="new-task-id", task_hash="new-task")
    for root, second_id in (
        (Path(archive.sources[1].curation_root), "old-second"),
        (tmp_path / "repaired", "new-second"),
    ):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["input_rows"] = manifest["converted_rows"] = 2
        manifest_path.write_text(json.dumps(manifest))
        source_key = canonical_sha256(manifest)
        rows = pq.read_table(root / "tasks.parquet").to_pylist()
        rows[0]["source_key"] = source_key
        rows.append({**rows[0], "task_id": second_id})
        pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema(TaskSemanticInput)), root / "tasks.parquet")
        public_path = root / "public-inputs.json"
        public = json.loads(public_path.read_text())
        public["source_key"] = source_key
        public_path.write_text(json.dumps(public))
    original = archive.sources[1].model_copy(
        update={
            "source_key": canonical_sha256(
                json.loads((Path(archive.sources[1].curation_root) / "manifest.json").read_text())
            ),
            "task_count": 2,
        }
    )
    replacement = SourceReplacement(
        source_id=changed.id,
        source_manifest=reference(tmp_path / "repaired/manifest.json"),
        public_inputs=reference(tmp_path / "repaired/public-inputs.json"),
    )
    with pytest.raises(ValueError):
        task_rebinding(original, replacement)


@pytest.mark.parametrize("change", ["none", "row", "task_hash", "stream"])
def test_repeated_public_inputs_require_pinned_upstream_rows_and_exact_frozen_tasks(tmp_path, change):
    manifests = {}
    source = None
    for side, revision in (("old", "1"), ("new", "2")):
        root = tmp_path / side
        root.mkdir()
        declared = declaration("Marin:rebound", revision).model_copy(
            update={"task_count": 2, "within_source_dedup": deduplication_evidence("fixture", revision, "1", 2)}
        )
        tasks = [
            numeric_answer_task(
                RawRow(
                    id=f"{side}-{index}",
                    source=Source(
                        dataset="pinned-upstream",
                        revision="immutable-revision",
                        importer_revision=revision,
                        row=f"row-{index}" if not (side == "new" and change == "row" and index == 1) else "other-row",
                    ),
                    data={},
                ),
                prompt="Add two and three.",
                answer="5",
                tolerance_abs=0,
                tolerance_rel=0,
            )
            for index in range(2)
        ]
        encoded = [task.model_dump_json() for task in tasks]
        specs = root / "specs"
        specs.mkdir()
        pq.write_table(pa.Table.from_pylist([{"task_spec": value} for value in encoded]), specs / "part.parquet")
        stream_hash = hashlib.sha256("".join(encoded).encode()).hexdigest()
        manifest = {
            "intended_use": "train",
            "input_rows": 2,
            "converted_rows": 2,
            "rejections": {},
            "input": {
                "declaration": declared.model_dump(mode="json"),
                "specs_dir": str(specs),
                "task_column": "task_spec",
                "raw_count": 2,
                "task_spec_scan_sha256": stream_hash,
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        key = canonical_sha256(manifest)
        rows = [task_semantic_input(task, TaskInputFile(declared.id, key, "", "train", "converted")) for task in tasks]
        if side == "new" and change == "task_hash":
            rows[0] = rows[0].model_copy(update={"task_hash": "not-the-frozen-task"})
        pq.write_table(
            pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema(TaskSemanticInput)),
            root / "tasks.parquet",
        )
        (root / "public-inputs.json").write_text(
            json.dumps({"source_key": key, "stored_stream_sha256": stream_hash, "paths": [str(root / "tasks.parquet")]})
        )
        if side == "new" and change == "stream":
            pq.write_table(
                pa.Table.from_pylist([{"task_spec": value} for value in reversed(encoded)]), specs / "part.parquet"
            )
        manifests[side] = root
        if side == "old":
            source = AlignmentSource(
                source_id=declared.id,
                source_key=key,
                population="converted",
                intended_use="train",
                task_count=2,
                ready_count=2,
                insufficient_context_count=0,
                basis="matched",
                curation_root=str(root),
            )
    replacement = SourceReplacement(
        source_id=declared.id,
        source_manifest=reference(manifests["new"] / "manifest.json"),
        public_inputs=reference(manifests["new"] / "public-inputs.json"),
    )
    if change != "none":
        with pytest.raises(ValueError):
            task_rebinding(source, replacement)
        return
    rebound = task_rebinding(source, replacement)
    assert dict(zip(rebound.task_ids, rebound.new_task_ids, strict=True)) == {"old-0": "new-0", "old-1": "new-1"}
    assert rebound.upstream_row_sha256 == tuple(
        canonical_sha256(("pinned-upstream", "immutable-revision", f"row-{index}")) for index in range(2)
    )


def test_projection_cannot_silently_drop_a_live_source_or_restore_a_retired_source(archived):
    archive, declarations, _, _ = archived
    for inventory in ([declarations[0]], declarations):
        with pytest.raises(ValueError, match="minus retired sources"):
            projection_selection(
                archive, inventory, frozenset({declarations[2].id}), (), source_uses=use_bindings(inventory)
            )


@pytest.mark.parametrize("changed", ["pin", "use", "declaration"])
def test_registered_use_must_match_the_current_pin_and_archived_role(archived, changed):
    archive, declarations, _, _ = archived
    inventory = declarations[:2]
    bindings = use_bindings(inventory)
    if changed == "declaration":
        inventory[0] = inventory[0].model_copy(update={"intended_use": SourceUse.EVAL})
    else:
        bindings[0] = bindings[0].model_copy(
            update={"dataset_revision": "different"} if changed == "pin" else {"use": LexicalUse.EVAL}
        )
    with pytest.raises(ValueError, match=r"different converter or dataset pin|population or intended use|Declared Use"):
        projection_selection(archive, inventory, frozenset({declarations[2].id}), (), source_uses=bindings)


def test_retried_assignment_publication_preserves_prior_attempt(tmp_path, archived):
    archive, declarations, rows, _ = archived
    options = dict(
        output_root=str(tmp_path / "retry-output"),
        archived_sources=tuple(archive.sources),
        selected_ids=frozenset(row.id for row in declarations[:2]),
        rebindings=(),
    )
    first = filter_assignment_file((0, archive.paths[0]), **options)
    saved = Path(first["output"]).read_bytes()
    retried = filter_assignment_file((0, archive.paths[0]), **options)
    assert first["output"] != retried["output"]
    assert Path(first["output"]).read_bytes() == saved
    assert list(load_parquet(first["output"])) == list(load_parquet(retried["output"])) == rows[:2]


@pytest.mark.parametrize("retained_task_id", [None, "same-id", "current-retained-id"])
def test_release_compose_replaces_archived_assignments_with_fresh_alignment(tmp_path, archived, retained_task_id):
    archive, _declarations, old_rows, manifests = archived
    rebound = None
    if retained_task_id is not None:
        prior = _declarations[2].model_copy(
            update={"task_count": 2, "within_source_dedup": deduplication_evidence("fixture", "1", "1", 2)}
        )
        old_source, _ = public_source(tmp_path / "old-two-tasks", prior, additional_public=("removed instruction",))
        public = pq.read_table(tmp_path / "old-two-tasks/tasks.parquet").to_pylist()
        old_rows = old_rows[:2] + [
            {**old_rows[2], **{key: task[key] for key in ("source_key", "task_id", "task_hash", "content_hash")}}
            for task in public
        ]
        pq.write_table(
            pa.Table.from_pylist(old_rows, schema=arrow_schema(TaskAlignment)), tmp_path / "archive-assignments.parquet"
        )
        archive = archive.model_copy(
            update={
                "sources": [*archive.sources[:2], old_source],
                "inventory": [*archive.inventory[:2], prior],
                "task_count": 4,
                "ready_count": 4,
            }
        )
        rebound = declaration("Marin:retired", "2")
        rebound_source, _ = public_source(
            tmp_path / "retained-subset", rebound, task_id=retained_task_id, task_hash="current-retained-hash"
        )
    changed = declaration("Marin:rebound", "2")
    source, fresh_public = public_source(
        tmp_path / "replacement", changed, public="changed public instructions", task_hash="new-hash", task_id="new-id"
    )
    fresh_rows = [
        {
            **row,
            "source_key": source.source_key,
            "task_hash": "new-hash",
            "task_id": "new-id",
            "content_hash": fresh_public.content_hash,
            "matches": {
                **row["matches"],
                "subjects": [{**row["matches"]["subjects"][0], "membership_similarity": 0.5}],
            },
        }
        for row in old_rows
        if row["source_id"] == changed.id
    ]
    path = tmp_path / "fresh-assignments.parquet"
    pq.write_table(pa.Table.from_pylist(fresh_rows, schema=arrow_schema(TaskAlignment)), path)
    fresh = archive.model_copy(
        update={
            "release_id": "fresh",
            "sources": [source],
            "inventory": [changed],
            "paths": [str(path)],
            "task_count": 1,
            "ready_count": 1,
        }
    )
    runtime = CampaignRuntime()
    with (
        local_client() as client,
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        old_coverage = CurriculumCoverage.model_validate_json(Path(manifests.coverage_uri).read_text())
        coverages = []
        for label, release in (("old", archive), ("fresh", fresh)):
            results = build_curriculum_coverage(
                context,
                task_paths=release.paths,
                nodes=release.nodes,
                output_root=str(tmp_path / label),
                subject_distance=old_coverage.subject_distance,
                capability_distance=old_coverage.capability_distance,
                source_sample_size=old_coverage.source_sample_size,
                area_sample_size=old_coverage.area_sample_size,
                confidence_levels=CONFIDENCE_LEVELS,
                sampling_source_ids={row.source_id: row.source_id for row in release.sources},
            )
            coverages.append(
                old_coverage.model_copy(
                    update={
                        "alignment_release_id": release.release_id,
                        "task_count": release.task_count,
                        "paths": results.paths,
                        "confidence_paths": results.confidence_paths,
                        "sample_paths": results.sample_paths,
                        "sample_task_count": results.sample_task_count,
                    }
                )
            )
        documents = {
            "old_release": archive.model_dump(mode="json"),
            "old_coverage": coverages[0].model_dump(mode="json"),
            "fresh_release": fresh.model_dump(mode="json"),
            "fresh_coverage": coverages[1].model_dump(mode="json"),
            "current_target_declarations": {changed.id: changed.model_dump(mode="json")},
        }
        if rebound is not None:
            documents["current_target_declarations"][rebound.id] = rebound.model_dump(mode="json")
        inputs_document = {}
        for name, document in documents.items():
            location = tmp_path / f"{name}.json"
            location.write_text(json.dumps(document))
            inputs_document[name] = reference(location).model_dump()
        if rebound is not None:
            inputs_document["public_input_rebindings"] = [
                SourceReplacement(
                    source_id=rebound.id,
                    source_manifest=reference(tmp_path / "retained-subset/manifest.json"),
                    public_inputs=reference(tmp_path / "retained-subset/public-inputs.json"),
                ).model_dump(mode="json")
            ]
        inputs_path = tmp_path / "compose-inputs.json"
        inputs_path.write_text(json.dumps(inputs_document))
        proof = compose.run_compose(
            compose.CurriculumComposeInputs(
                release_id="calendar-test",
                source_revision="a" * 40,
                inputs=compose.ArtifactRef(**reference(inputs_path).model_dump()),
                output_root=str(tmp_path / "composed"),
                expected_fresh_task_count=1,
                removed_sources=(),
            ),
            dry_run=False,
            context=context,
            feeds=(tmp_path / "alerts.jsonl",),
        )
    release = AlignmentRelease.model_validate_json(Path(proof["release"]["uri"]).read_text())
    rows = [row for path in release.paths for row in pq.read_table(path).to_pylist()]
    assert [row for row in rows if row["source_id"] == changed.id] == fresh_rows
    rewritten = {changed.id} | ({rebound.id} if rebound is not None else set())
    assert [row for row in rows if row["source_id"] not in rewritten] == [
        row for row in old_rows if row["source_id"] not in rewritten
    ]
    assert proof["replacement_sources"] == sorted(rewritten)
    assert proof["task_count"] == 3
    if rebound is not None:
        kept = next(row for row in rows if row["source_id"] == rebound.id)
        assert (kept["task_id"], kept["task_hash"], kept["source_key"]) == (
            retained_task_id,
            "current-retained-hash",
            rebound_source.source_key,
        )
        assert kept["matches"] == old_rows[2]["matches"]
        assert [row["task_id"] for row in rows if row["source_id"] == rebound.id] == [retained_task_id]
        coverage = CurriculumCoverage.model_validate_json(Path(proof["coverage"]["uri"]).read_text())
        statistics = [row for path in coverage.paths for row in load_parquet(path)]
        assert (
            next(
                row["instances"]
                for row in statistics
                if row["source_id"] == rebound.id and row["metric"] == "denominator"
            )
            == 1
        )
        samples = [row for path in coverage.sample_paths for row in load_parquet(path)]
        assert {row["task_id"] for row in samples if row["source_id"] == rebound.id} == {retained_task_id}
        identity = proof["public_input_rebindings"][0]
        mapping = pq.read_table(identity["task_identity_mapping"]["uri"]).to_pylist()
        assert [(row["old_task_id"], row["new_task_id"]) for row in mapping] == [
            ("same-id", retained_task_id),
            ("same-id-extra-0", None),
        ]
        curriculum_load.verify_public_input_rebindings(proof)
        bad_mapping = tmp_path / "duplicate-mapping.parquet"
        mapping[1]["old_task_id"] = mapping[0]["old_task_id"]
        pq.write_table(pa.Table.from_pylist(mapping), bad_mapping)
        identity["task_identity_mapping"] = reference(bad_mapping).model_dump(mode="json")
        with pytest.raises(ValueError, match="does not cover"):
            curriculum_load.verify_public_input_rebindings(proof)
    assert pq.read_table(tmp_path / "archive-assignments.parquet").to_pylist() == old_rows
