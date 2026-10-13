# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.reader import ReadView
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.catalog_snapshot import current_catalog, publish_catalog
from experiments.post_training.task_curation.count_inputs import parquet_counts
from experiments.post_training.task_curation.curation_catalog import append_curation_catalog
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import DataSourceReview, RlDataSource, SourceInfo
from experiments.post_training.task_curation.source_metadata import source_row
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def test_parquet_count_export_identifies_the_counted_input(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(pa.table({"question": ["one", "two", "three"]}), snapshot / "tasks.parquet")
    metadata = snapshot / ".cache/huggingface/download/tasks.parquet.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("pinned-input\nfile-etag\n0\n")
    count = parquet_counts(snapshot, "pinned-input", "*.parquet")["tasks.parquet"]
    pipeline = replace(
        cast(CurationRecipe, next(source.config for source in skyrl_math.sources() if source.name == "math500")),
        source=HfSource("local/questions", "pinned-input", ("tasks.parquet",), SourceFormat.PARQUET),
    )
    source = RlDataSource(
        pipeline=process_rows,
        info=SourceInfo(id="local:questions", title="Questions", origin="local", count=count),
        config=pipeline,
    )
    row = source_row(source)
    assert row["task_count"] == 3
    assert row["dataset_revision"] == row["pipeline"]["revision"] == "pinned-input"
    assert row["url"] == "https://huggingface.co/datasets/local/questions/tree/pinned-input"
    assert row["pipeline"]["files"] == ("tasks.parquet",)
    assert source_row(replace(source, info=replace(source.info, count=None)))["task_count"] is None
    with pytest.raises(ValueError, match="expected revision"):
        parquet_counts(snapshot, "another-revision", "*.parquet")


def test_source_metadata_keeps_inventory_only_entries_without_authored_grades():
    reviewed = RlDataSource(
        info=SourceInfo(id="local:reviewed", title="Reviewed", origin="local", count=7),
        review=DataSourceReview(grade="good", evidence_url="https://example.org/review", reviewed_at="2026-10-08"),
    )
    row = source_row(reviewed)
    assert row["id"] == "local:reviewed" and row["task_count"] == 7 and row["pipeline"] is None
    assert "quality" not in row and "review_url" not in row


def test_dataset_catalog_identity_is_available_without_execution(tmp_path):
    source = number_source(tmp_path / "not-staged")
    row = source_row(source)
    assert row["dataset_id"] == "local-numbers"
    assert row["dataset_revision"] == "pinned"
    assert row["pipeline"] == {
        "name": "numbers",
        "version": "1",
        "source": "local-numbers",
        "revision": "pinned",
        "files": (),
    }
    assert row["url"] == "https://example.org/numbers"


def test_published_catalog_freezes_evidence_and_selects_only_matching_pipeline_grades(tmp_path):
    source = number_source(tmp_path / "numbers.txt")
    inventory_only = RlDataSource(
        info=SourceInfo(id="native:only", title="Native", origin="native", count=50),
        review=DataSourceReview(grade="good", evidence_url="https://example.org/old-review"),
    )
    root = str(tmp_path / "catalog")
    inputs, judgments, curated = (tmp_path / name for name in ("inputs", "judgments", "curated"))
    for path in (inputs, judgments, curated):
        path.mkdir()
    task_id = "numbers:one"
    packets = inputs / "packets.parquet"
    grades = judgments / "grades.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task_id,
                    "cohort": "admitted",
                    "packet_json": json.dumps({"task": {"id": task_id}}),
                    "omissions": [],
                }
            ]
        ),
        packets,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task_id,
                    "cohort": "admitted",
                    "status": "reviewed",
                    "verdict": "good",
                    "explanation": "Inspected fixture",
                    "evidence_ids": ["task"],
                }
            ]
        ),
        grades,
    )
    (inputs / "manifest.json").write_text(
        json.dumps({"packet_paths": [str(packets)], "sample": [{"task_id": task_id, "cohort": "admitted"}]})
    )
    (judgments / "manifest.json").write_text(json.dumps({"grade_paths": [str(grades)]}))
    curation = {
        "source": "numbers",
        "status": "completed",
        "mode": "full",
        "raw_population_count": 2,
        "processed_rows": 2,
        "unprocessed_rows": 0,
        "admission_counts": {"admitted": 2},
        "verification": {
            "status": "passed",
            "counts": {"checked": 2, "passed": 2},
            "results": [{"task_id": task_id, "stdout": "Detailed control output"}],
            "evidence": [{"task_id": task_id, "reference": "Private control reference"}],
        },
    }
    (curated / "manifest.json").write_text(json.dumps(curation))
    report = {
        "scope": "full",
        "basis": "sampled_model_opinions",
        "grade_rule": "fixture",
        "model": {"revision": "fixture"},
        "source_evidence": {"reader_declared": False, "rows": 0, "unavailable_rows": 0},
        "curation": curation,
        "assessment": {
            "admitted": {
                "grade": "good",
                "sample_count": 1,
                "examples": {"good": [{"task_id": task_id, "explanation": "Consistent reference"}]},
            },
            "excluded": {},
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    evidence = {
        "manifest": str(curated / "manifest.json"),
        "assessment_inputs": str(inputs),
        "assessment_judgments": str(judgments),
        "assessment_report": str(report_path),
    }

    def append(run_id, key, status):
        outcome = {
            "name": "numbers",
            "path": str(curated),
            "status": status,
            "pipeline_key": key,
            "error": "transport unavailable" if status == "failed" else None,
            "planned_evidence": evidence,
        }
        if status == "completed":
            outcome["result"] = {"evidence": evidence, "outputs": {}}
        campaign = {
            "status": status,
            "mode": "full",
            "run_id": run_id,
            "updated_at": "2026-10-12T10:00:00+00:00",
            "counts": {status: 1},
            "sources": [outcome],
        }
        append_curation_catalog(
            root,
            json.dumps(campaign).encode(),
            run_id=run_id,
            report_path="campaign.json",
            source_ids={"numbers": source.info.id},
        )

    append("first", "pipeline@1:key", "completed")
    first = publish_catalog(root, "first", [source, inventory_only], legacy_archive_url="https://example.org/archive")
    assert current_catalog(root) == first
    by_id = {row["id"]: row for row in first["sources"]}
    assert by_id[source.info.id]["task_count"] == 2
    verification = by_id[source.info.id]["curation"]["verification"]
    assert verification["status"] == "passed" and verification["counts"] == {"checked": 2, "passed": 2}
    assert "results" not in verification and "evidence" not in verification
    archived = json.loads(ReadView(root).resolve(by_id[source.info.id]["assessment"]["report_uri"]))
    assert archived["curation"]["verification"] == curation["verification"]
    assert "examples" not in by_id[source.info.id]["assessment"]["admitted"]
    assert archived["assessment"]["admitted"]["examples"] == report["assessment"]["admitted"]["examples"]
    native = by_id[inventory_only.info.id]
    assert native["task_count"] == 50 and native["assessment"]["status"] == "skipped"
    assert native["quality"] is None and "review_url" not in native
    uri = by_id[source.info.id]["assessment"]["samples"][0]["evidence_uri"]
    assert json.loads(ReadView(root).resolve(uri))["judgment"]["verdict"] == "good"

    packets.unlink()
    grades.unlink()
    report_path.write_text(json.dumps({**report, "assessment": {"admitted": {"grade": "bad"}, "excluded": {}}}))
    assert publish_catalog(root, "first", [source, inventory_only]) == first
    append("retry", "pipeline@1:key", "failed")
    retry = publish_catalog(root, "retry", [source, inventory_only])
    row = next(row for row in retry["sources"] if row["id"] == source.info.id)
    assert row["quality"] == "good" and row["assessment"]["latest_attempt_error"] == "transport unavailable"
    assert retry["legacy_archive_url"] == "https://example.org/archive"
    assert json.loads(ReadView(root).resolve(uri))["packet"]["task"]["id"] == task_id

    append("changed", "pipeline@2:new-key", "failed")
    changed = publish_catalog(root, "changed", [source, inventory_only])
    row = next(row for row in changed["sources"] if row["id"] == source.info.id)
    assert row["quality"] is None and row["assessment"]["status"] == "failed"
    assert current_catalog(root)["snapshot_id"] == "changed"
