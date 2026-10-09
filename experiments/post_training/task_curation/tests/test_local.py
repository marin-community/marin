# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.convert.answers import answer_task, exact_answer_task
from taskcompendium.models import EnvironmentRequirements, TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.inputs import SourceFormat, required_grader_environment
from verifyit.spec import ExactSpec
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import pipeline as pipeline_module
from experiments.post_training.task_curation.campaign import CampaignFailed
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.local import run_local_sources
from experiments.post_training.task_curation.sources import all_pipelines
from experiments.post_training.task_curation.tasktrove.compare import source_file_path


def convert_local_answer(row, context):
    task = answer_task(row, prompt=row.data["prompt"], spec=ExactSpec((row.data["answer"],), ignore_case=False))
    return task.model_copy(
        update={"grader": task.grader.model_copy(update={"environment": required_grader_environment(context)})}
    )


@pytest.fixture
def local_source():
    source = all_pipelines()["tasktrove-calendar"]
    return replace(
        source,
        source=HfSource("fixture/questions", "a" * 40, ("rows.parquet",), SourceFormat.PARQUET),
        convert=convert_local_answer,
    )


def test_local_campaign_continues_after_missing_input(local_source, tmp_path):
    source = local_source
    missing = replace(source, name="missing", source=replace(source.source, files=("missing.parquet",)))
    input_root = tmp_path / "input"
    staged = input_root / source.source.files[0]
    staged.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"path": "original-row", "prompt": "One plus one?", "answer": "two"}]), staged)
    output = tmp_path / "output"
    with pytest.raises(CampaignFailed):
        run_local_sources(
            {missing.name: missing, source.name: source},
            input_root,
            output,
            inputs={},
            max_workers=1,
            download_cache=tmp_path / "downloads",
        )
    report = json.loads((output / "campaign.json").read_text())
    assert report["status"] == "failed"
    assert [row["status"] for row in report["sources"]] == ["failed", "completed"]
    assert "FileNotFoundError" in report["sources"][0]["error"]
    records = [
        row for shard in (output / source.name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))
    ]
    assert len(records) == 1
    assert records[0]["original_path"] == "original-row"
    task = TaskSpec.model_validate_json(records[0]["task_json"])
    grader = cast(VerifyitGrader, task.grader)
    assert verifyit_spec(grader) == ExactSpec(("two",), ignore_case=False)
    environment = cast(EnvironmentRequirements, grader.environment)
    assert environment.packages_lock == str(source.grader.lock.resolve())


def answer_from_auxiliary(row, context):
    answer = (context.inputs["answers"] / "answer.txt").read_text()
    return exact_answer_task(row, prompt=row.data["prompt"], answers=(answer,), ignore_case=False)


def converted_task(output: Path, name: str) -> TaskSpec:
    records = [row for shard in (output / name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))]
    assert len(records) == 1
    return TaskSpec.model_validate_json(records[0]["task_json"])


def test_local_campaign_stages_pinned_inputs_and_reuses_downloads_offline(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    revision = "a" * 40
    primary = remote / "fixture/questions" / revision
    primary.mkdir(parents=True)
    (primary / "rows.jsonl").write_text('{"prompt": "What is one plus one?"}\n')
    # The selected file patterns must not pull unrelated files into the cache.
    (primary / "unselected.jsonl").write_text('{"prompt": "unselected"}\n')
    auxiliary = remote / "fixture/answers" / revision
    auxiliary.mkdir(parents=True)
    (auxiliary / "answer.txt").write_text("two")
    source = all_pipelines()["tasktrove-calendar"]
    source = replace(
        source,
        source=HfSource("fixture/questions", revision, ("rows.jsonl",), SourceFormat.JSONL),
        inputs={"answers": HfSource("fixture/answers", revision, ("answer.txt",), SourceFormat.JSONL)},
        convert=answer_from_auxiliary,
        grader=None,
    )
    plan_download = pipeline_module.plan_download

    def local_plan(config):
        # Replace the remote I/O boundary; retain the real transfer, Zephyr, and artifact cache.
        return plan_download(replace(config, source_url_override=str(remote / config.hf_dataset_id / config.revision)))

    monkeypatch.setattr(pipeline_module, "plan_download", local_plan)
    cache = tmp_path / "downloads"
    cold = tmp_path / "cold"
    run_local_sources({source.name: source}, None, cold, inputs={}, max_workers=1, download_cache=cache)
    task = converted_task(cold, source.name)
    assert verifyit_spec(cast(VerifyitGrader, task.grader)) == ExactSpec(("two",), ignore_case=False)
    assert not list(cache.rglob("unselected.jsonl"))
    manifest = json.loads((cold / source.name / "manifest.json").read_text())
    staged = source_file_path(manifest, "rows.jsonl")
    assert staged.is_relative_to(cache)
    assert json.loads(staged.read_text()) == {"prompt": "What is one plus one?"}

    shutil.rmtree(remote)
    warm = tmp_path / "warm"
    run_local_sources({source.name: source}, None, warm, inputs={}, max_workers=1, download_cache=cache)
    assert converted_task(warm, source.name) == task

    override = tmp_path / "override"
    override.mkdir()
    (override / "answer.txt").write_text("explicit answer")
    overridden = tmp_path / "overridden"
    run_local_sources(
        {source.name: source},
        None,
        overridden,
        inputs={"answers": str(override)},
        max_workers=1,
        download_cache=cache,
    )
    override_task = converted_task(overridden, source.name)
    assert verifyit_spec(cast(VerifyitGrader, override_task.grader)) == ExactSpec(
        ("explicit answer",), ignore_case=False
    )

    next_revision = "b" * 40
    updated = remote / "fixture/questions" / next_revision
    updated.mkdir(parents=True)
    (updated / "rows.jsonl").write_text('{"prompt": "The next pinned question"}\n')
    source = replace(source, source=replace(cast(HfSource, source.source), revision=next_revision))
    next_output = tmp_path / "next"
    run_local_sources({source.name: source}, None, next_output, inputs={}, max_workers=1, download_cache=cache)
    next_task = converted_task(next_output, source.name)
    assert next_task.source.revision == next_revision
    assert cast(TextMessage, next_task.context.events[0]).content == "The next pinned question"


def test_explicit_local_file_preserves_logical_identity_and_records_actual_bytes(local_source, tmp_path):
    source = local_source
    logical_path = source.source.files[0]
    local_file = tmp_path / "different-name.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"path": "original-row", "prompt": "One plus one?", "answer": "two"}]), local_file
    )
    output = tmp_path / "output"
    run_local_sources(
        {source.name: source},
        None,
        output,
        inputs={},
        max_workers=1,
        download_cache=tmp_path / "downloads",
        source_files_override={logical_path: local_file},
    )
    records = [row for file in (output / source.name / "normalize").glob("*.parquet") for row in load_parquet(str(file))]
    assert records[0]["source_row"] == f"{logical_path}:0"
    assert records[0]["original_path"] == "original-row"
    manifest = json.loads((output / source.name / "manifest.json").read_text())
    assert manifest["source_file_overrides"] == {
        logical_path: {"path": str(local_file.resolve()), "sha256": hashlib.sha256(local_file.read_bytes()).hexdigest()}
    }
    # A per-file override remains authoritative even when another staging root is supplied.
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    assert source_file_path(manifest, logical_path) == local_file
    assert source_file_path(manifest, logical_path, alternate) == local_file
    assert not (tmp_path / "downloads").exists()
