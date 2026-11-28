# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PRIVATE_ALWAYS fixture sources with canary strings for the private data tier.

The fixture carries no real vendor data. The canary is unique nonsense text planted in
every content-bearing field of the restricted fixture — task packets, reviewer text,
curriculum previews, archived reviews and audit notes — so the end-to-end leak test can
assert it appears in no response, artifact or log.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import QualifiedSourceDeclaration as SourceDeclaration
from finestore.rl.curation import source_display_name
from finestore.rl.status_taxonomy import DataRestriction
from sqlalchemy import text
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage

from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin
from experiments.post_training.task_curation.tests.admission_fixtures import (
    actor_census_input,
    grading_evidence,
)
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

PRIVATE_SOURCE_ID = "Marin:synthetic-private-v1"
OPEN_SOURCE_ID = "Marin:synthetic-open-v1"

PRIVATE_CANARY = "CANARY-PRIVATE-TIER-20261117-b7e4d2a1c9f3"
"""Unique marker text planted in every content field of the restricted fixture."""

FIXTURE_DATASET = "open-athena/private-tier-fixture"
FIXTURE_REVISION = "c" * 40

_MARKER_FIELDS = ("prompt", "answer", "instructions")


def private_canary_text(field: str = "prompt") -> str:
    """One synthetic task text that embeds the canary; never real vendor content."""
    if field not in _MARKER_FIELDS:
        raise ValueError(f"Fixture canary text covers only {(_MARKER_FIELDS)} fields: {field}")
    return f"Synthetic {field} for the private tier fixture: {PRIVATE_CANARY} and nothing else."


def restricted_declaration(
    source_id: str = PRIVATE_SOURCE_ID,
    restriction: DataRestriction = DataRestriction.PRIVATE_ALWAYS,
    *,
    notes: str = "",
) -> dict:
    """A complete valid synthetic declaration; ``restriction`` selects the tier under test."""
    count = 3
    lexical = decontamination_evidence(FIXTURE_DATASET, FIXTURE_REVISION, count)
    return {
        "license_receipt": fixture_license(source_id, FIXTURE_DATASET, FIXTURE_REVISION).model_dump(mode="json"),
        "id": source_id,
        "display_name": source_display_name(source_id),
        "data_restriction": restriction.value,
        "origin": "Marin",
        "build_path": "test",
        "upstream_dataset": source_id.partition(":")[2],
        "upstream_name": source_id.upper(),
        "current_release": {"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        "daytona_safety": "not_applicable",
        "daytona_snapshot_count": 0,
        "daytona_machine_slots": 0,
        "daytona_execution_failures": 0,
        "daytona_count_basis": "Fixture tasks contain no actor environment.",
        "grading_environment": grading_evidence(count).model_dump(mode="json"),
        "family": source_id,
        "tags": [],
        "task_count": count,
        "within_source_dedup": (
            deduplication_evidence(FIXTURE_DATASET, FIXTURE_REVISION, "1", count).model_dump(mode="json")
        ),
        "notes": notes,
        "dataset_id": FIXTURE_DATASET,
        "dataset_revision": FIXTURE_REVISION,
        "lexical_contamination": lexical.model_dump(mode="json"),
        "url": None,
        "type": "RLVR",
        "turns": "Single-turn",
        "is_benchmark": False,
        "intended_use": "train",
        "kind": "Dataset",
        "status": "Available",
        "pipeline": {
            "name": "fixture",
            "version": "1",
            "source": FIXTURE_DATASET,
            "revision": FIXTURE_REVISION,
            "files": [],
        },
        "actor_census_input": actor_census_input(FIXTURE_DATASET, FIXTURE_REVISION, "1", count).model_dump(mode="json"),
    }


def restricted_info(restriction: DataRestriction = DataRestriction.PRIVATE_ALWAYS, source_id: str = PRIVATE_SOURCE_ID):
    """A typed SourceInfo fixture carrying the tier explicitly."""
    return SourceInfo(
        data_restriction=restriction,
        origin=SourceOrigin.MARIN,
        upstream_dataset=source_id.partition(":")[2],
        upstream_name=source_id.upper(),
        build_path="test",
        id=source_id,
    )


def curriculum_release_fixture(
    directory: Path,
    declarations: list[dict],
    *,
    preview: str = "Count public objects.",
    instructions: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """A minimal valid curriculum release whose task text embeds ``instructions`` when given.

    Returns ``(release, coverage, tasks, statistics)`` paths in ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for declaration in declarations:
        for index, (task_id, score) in enumerate((("one", 0.9), ("two", 0.65), ("three", None)), start=1):
            records.append(
                {
                    "source_id": declaration["id"],
                    "source_key": declaration["id"] + "-key",
                    "task_id": task_id,
                    "task_hash": declaration["id"] + task_id,
                    "content_hash": declaration["id"] + str(index),
                    "intended_use": "train",
                    "population": "converted",
                    "description_status": "ready" if score is not None else "insufficient_context",
                    "description": (instructions or preview) if score is not None else "",
                    "reason": "",
                    "task_preview": instructions or preview,
                    "preview_truncated": False,
                    "instruction_characters": len(instructions or preview),
                    "matches": (
                        {
                            "subjects": [
                                {
                                    "subject_id": "D01",
                                    "facet": "subject_domain",
                                    "membership_similarity": 0.9,
                                    "capability_id": "add",
                                    "candidates": [
                                        {
                                            "subject_id": "D01",
                                            "capability_id": "add",
                                            "similarity": score,
                                            "anchor_id": "addition",
                                        }
                                    ],
                                }
                            ],
                            "capabilities": [],
                        }
                        if score is not None
                        else None
                    ),
                }
            )
    tasks_path = directory / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(records), tasks_path)

    count = 3 * len(declarations)
    ready = 2 * len(declarations)
    release = AlignmentRelease(
        release_id="private-tier-fixture",
        created_at="2026-11-17T00:00:00Z",
        curriculum_key="curriculum-private-tier",
        population="converted",
        sample_size=None,
        task_count=count,
        ready_count=ready,
        insufficient_context_count=count - ready,
        sources=[
            {
                "source_id": declaration["id"],
                "source_key": declaration["id"] + "-key",
                "population": "converted",
                "intended_use": "train",
                "basis": "matched",
                "task_count": 3,
                "ready_count": 2,
                "insufficient_context_count": 1,
            }
            for declaration in declarations
        ],
        inventory=[SourceDeclaration.model_validate(declaration) for declaration in declarations],
        nodes=[
            {
                "node_id": "subject:D01",
                "subject_id": "D01",
                "parent_id": None,
                "kind": "subject",
                "name": "Mathematics",
                "description": "Numbers",
            },
            {
                "node_id": "arithmetic",
                "subject_id": "D01",
                "parent_id": "subject:D01",
                "kind": "group",
                "name": "Arithmetic",
                "description": "Calculate",
            },
            {
                "node_id": "add",
                "subject_id": "D01",
                "parent_id": "arithmetic",
                "kind": "capability",
                "name": "Addition",
                "description": "Add numbers",
            },
        ],
        paths=[str(directory / "full-corpus-not-loaded-by-atlas.parquet")],
        lineage={"alignment_root": "fixture"},
        embedding_spec={
            "model": "fixture",
            "revision": "fixture",
            "tokenizer_revision": "fixture",
            "dimensions": 2,
            "max_tokens": 512,
            "compute_dtype": "float32",
        },
    )
    release_path = directory / "release.json"
    release_path.write_text(release.model_dump_json())

    statistics = []
    for declaration in declarations:
        for metric, instances in (("denominator", 3), ("assigned", 2), ("insufficient_context", 1)):
            statistics.append(
                dict(source_id=declaration["id"], metric=metric, subject_id="", node_id="", instances=instances)
            )
        for node_id, metric, instances in (
            ("subject:D01", "placement", 2),
            ("arithmetic", "placement", 2),
            ("add", "placement", 2),
        ):
            statistics.append(
                dict(source_id=declaration["id"], metric=metric, subject_id="D01", node_id=node_id, instances=instances)
            )
        statistics.append(
            dict(source_id=declaration["id"], metric="capability", subject_id="D01", node_id="subject:D01", instances=2)
        )
    statistics_path = directory / "statistics.parquet"
    pq.write_table(pa.Table.from_pylist(statistics), statistics_path)

    coverage = CurriculumCoverage(
        coverage_id="private-tier-fixture",
        alignment_release_id="private-tier-fixture",
        curriculum_key="curriculum-private-tier",
        subject_distance=0.45,
        capability_distance=0.25,
        task_count=count,
        paths=[str(statistics_path)],
        sample_paths=[str(tasks_path)],
        sample_task_count=len(records),
        source_sample_size=100,
        area_sample_size=10,
    )
    coverage_path = directory / "coverage.json"
    coverage_path.write_text(coverage.model_dump_json())
    return release_path, coverage_path, tasks_path, statistics_path


def write_review_rows(connection, source_id: str, review_id: str, collection: dict, artifact: str) -> None:
    """Insert one archived review and its raw artifact for the tier tests."""
    connection.execute(
        text(
            "INSERT INTO catalog_reviews (id, source_id, collection, updated_at) "
            "VALUES (:id, :source_id, :collection, NOW())"
        ),
        {"id": review_id, "source_id": source_id, "collection": json.dumps(collection)},
    )
    connection.execute(
        text(
            "INSERT INTO review_artifacts (review_id, path, content, sha256) "
            "VALUES (:review_id, :path, :content, :sha256)"
        ),
        {
            "review_id": review_id,
            "path": "review/notes.txt",
            "content": artifact,
            "sha256": hashlib.sha256(artifact.encode()).hexdigest(),
        },
    )
