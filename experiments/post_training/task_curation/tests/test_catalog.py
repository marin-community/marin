# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.count_inputs import parquet_counts
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.export_catalog import catalog_document, source_row
from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, recipe_source
from experiments.post_training.task_curation.source import DataSourceReview, RlDataSource, SourceInfo
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
        cast(RlDataPipeline, next(source.pipeline for source in skyrl_math.sources() if source.name == "math500")),
        source=HfSource("local/questions", "pinned-input", ("tasks.parquet",), SourceFormat.PARQUET),
    )
    source = recipe_source(
        info=SourceInfo(id="local:questions", title="Questions", origin="local", count=count),
        pipeline=pipeline,
    )
    row = source_row(source)
    assert row["task_count"] == 3
    assert row["dataset_revision"] == row["pipeline"]["revision"] == "pinned-input"
    assert row["url"] == "https://huggingface.co/datasets/local/questions/tree/pinned-input"
    assert row["pipeline"]["files"] == ("tasks.parquet",)
    assert source_row(replace(source, info=replace(source.info, count=None)))["task_count"] is None
    with pytest.raises(ValueError, match="expected revision"):
        parquet_counts(snapshot, "another-revision", "*.parquet")


def test_catalog_export_is_deterministic_and_keeps_metadata_only_sources():
    reviewed = RlDataSource(
        info=SourceInfo(id="local:reviewed", title="Reviewed", origin="local", count=7),
        review=DataSourceReview(grade="good", evidence_url="https://example.org/review", reviewed_at="2026-10-08"),
    )
    excluded = RlDataSource(info=SourceInfo(id="local:excluded", title="Excluded", origin="local", tags=("excluded",)))
    document = catalog_document([reviewed, excluded])
    assert document == catalog_document([excluded, reviewed])
    rows = json.loads(json.dumps(document))["sources"]
    assert [(row["id"], row["status"], row["pipeline"]) for row in rows] == [
        ("local:excluded", "Excluded", None),
        ("local:reviewed", "Available", None),
    ]
    assert rows[1]["quality"] == "good"
    assert rows[1]["review_url"] == "https://example.org/review"
    assert (
        document["revision"]
        == hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    changed = replace(reviewed, info=replace(reviewed.info, count=8))
    assert catalog_document([changed, excluded])["revision"] != document["revision"]


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
