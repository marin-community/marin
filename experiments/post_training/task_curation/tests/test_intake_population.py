# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completed QUICK inputs enter the existing exact-contract population writer."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.intake_population import QuickPopulationConfig, normalize_quick
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


def test_quick_population_deduplicates_contracts_and_rejects_changed_frozen_inputs(tmp_path):
    path = tmp_path / "normalized.parquet"
    documents = [
        json.loads(prebuilt_task(identifier)["lowered_task_spec"])["task"] for identifier in ("first", "second")
    ]
    pq.write_table(pa.Table.from_pylist([{"task_json": json.dumps(task)} for task in documents]), path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"converted_rows": 2}')
    config = QuickPopulationConfig(
        source_id="Marin:fixture",
        dataset_id="fixture",
        revision="pin",
        version="1",
        converter_revision="a" * 40,
        manifest_path=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        inputs=((str(path), hashlib.sha256(path.read_bytes()).hexdigest()),),
        converted_rows=2,
        rejections={},
        output_path=str(tmp_path / "population"),
    )
    population = normalize_quick(config)
    assert population.task_count == 1
    assert population.within_source_dedup["raw_count"] == 2
    assert population.within_source_dedup["within_source_duplicates"] == 1
    stored = pq.read_table(tmp_path / "population/specs/part-000000.parquet").to_pylist()
    assert json.loads(stored[0]["task_spec"]) == documents[0]
    pq.write_table(pa.Table.from_pylist([{"task_json": json.dumps(documents[1])}]), path)
    with pytest.raises(ValueError):
        normalize_quick(replace(config, output_path=str(tmp_path / "changed-input")))
