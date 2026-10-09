# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

from experiments.post_training.task_curation.export_catalog import catalog_document, source_row
from experiments.post_training.task_curation.source import DataSourceMetadata, DataSourceReview, RlDataSource
from experiments.post_training.task_curation.sources import all_pipelines, all_sources
from experiments.post_training.task_curation.tests import (
    test_arc,
    test_nemotron_ultra,
    test_reasoning_gym,
    test_skyrl,
    test_tasktrove_code,
    test_tasktrove_text,
)

FAMILY_TESTS = (test_arc, test_nemotron_ultra, test_reasoning_gym, test_skyrl, test_tasktrove_code, test_tasktrove_text)


def test_every_declaration_has_a_fixture_row():
    covered = [name for module in FAMILY_TESTS for name in module.ROWS]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(all_pipelines())


def test_catalog_preserves_release_population_and_conversion_input():
    source = all_sources()["Task Trove:DCAgent__code-contests-noblock"]
    row = source_row(source)
    assert row["dataset_id"] == "open-athena/task-trove"
    assert row["pipeline"]["source"] == "open-thoughts/TaskTrove"
    assert row["dataset_revision"] != row["pipeline"]["revision"]
    assert row["task_count"] == 8222
    assert row["quality"] is None


def test_catalog_export_is_deterministic_and_keeps_metadata_only_sources():
    reviewed = RlDataSource(
        metadata=DataSourceMetadata(id="local:reviewed", name="reviewed", origin="local", task_count=7),
        review=DataSourceReview(grade="good", evidence_url="https://example.org/review", reviewed_at="2026-10-08"),
    )
    excluded = RlDataSource(
        metadata=DataSourceMetadata(id="local:excluded", name="excluded", origin="local", status="Excluded")
    )
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
    changed = replace(reviewed, metadata=replace(reviewed.metadata, task_count=8))
    assert catalog_document([changed, excluded])["revision"] != document["revision"]
