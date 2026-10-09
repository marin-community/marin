# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import ScriptGrader, TaskSpec
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignFailed
from experiments.post_training.task_curation.datasets.tasktrove import calendar
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.quick import run_local_sources


def test_local_campaign_continues_after_missing_source_and_converts_calendar(tmp_path: Path):
    source = calendar.sources()[0]
    assert source.pipeline is not None
    assert isinstance(source.pipeline.source, HfSource)
    missing = replace(
        source,
        pipeline=replace(
            source.pipeline, name="missing", source=replace(source.pipeline.source, files=("missing.parquet",))
        ),
    )
    input_root = tmp_path / "input"
    staged = input_root / source.pipeline.source.files[0]
    staged.parent.mkdir(parents=True)
    blob = (Path(__file__).parent / "fixtures" / "calendar.tar.gz").read_bytes()
    pq.write_table(pa.Table.from_pylist([{"path": "calendar-fixture.tar.gz", "task_binary": blob}]), staged)
    output = tmp_path / "output"
    with pytest.raises(CampaignFailed):
        run_local_sources({missing.name: missing, source.name: source}, input_root, output, inputs={}, max_workers=1)
    report = json.loads((output / "campaign.json").read_text())
    assert report["status"] == "failed"
    assert [row["status"] for row in report["sources"]] == ["failed", "completed"]
    assert "FileNotFoundError" in report["sources"][0]["error"]
    records = [
        row for shard in (output / source.name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))
    ]
    assert len(records) == 1
    assert records[0]["original_path"] == "calendar-fixture.tar.gz"
    task = TaskSpec.model_validate_json(records[0]["task_json"])
    assert isinstance(task.grader, ScriptGrader)
    assert source.pipeline.grader is not None and source.pipeline.grader.lock is not None
    assert task.grader.environment.packages_lock == str(source.pipeline.grader.lock.resolve())
    assert task.grader.answer_path == "/app/answer.txt"
