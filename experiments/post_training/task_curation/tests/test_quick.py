# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.models import ScriptGrader, TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.inputs import SourceFormat
from verifyit.spec import ExactSpec
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import pipeline as pipeline_module
from experiments.post_training.task_curation.campaign import CampaignFailed
from experiments.post_training.task_curation.datasets.tasktrove import calendar, code
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.quick import run_local_sources
from experiments.post_training.task_curation.tests.conversion import tasktrove_row


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
    assert records[0]["original_path"] == "calendar-fixture.tar.gz"
    task = TaskSpec.model_validate_json(records[0]["task_json"])
    assert isinstance(task.grader, ScriptGrader)
    assert source.pipeline.grader is not None and source.pipeline.grader.lock is not None
    assert task.grader.environment.packages_lock == str(source.pipeline.grader.lock.resolve())
    assert task.grader.answer_path == "/app/answer.txt"


def test_quick_retains_reviewed_defects_and_sample_only_rejections(tmp_path):
    source = next(source for source in code.sources() if source.name == "tasktrove-competitive_coding")
    assert source.pipeline is not None
    prompt = "Read two integers and print their sum. Write `/app/solution.py`."
    files = {
        "instruction.md": prompt.encode(),
        "environment/Dockerfile": b"FROM python:3.12-slim\nWORKDIR /app\n",
        "tests/verifier_data.json": json.dumps({"inputs": ["3 4\n"], "outputs": ["7\n"]}).encode(),
    }
    rows = [
        tasktrove_row(files, path="comp-coding-dd2a13b32896.tar.gz"),
        tasktrove_row(files, path="hidden-case"),
        # This path is rejected only in code-contests, not in this source.
        tasktrove_row(files, path="code_contests-4395"),
        tasktrove_row({**files, "instruction.md": (prompt + "\nExample: 3 4 gives 7.").encode()}, path="sample-only"),
    ]
    staged = tmp_path / "input" / source.pipeline.source.files[0]
    staged.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), staged)
    output = tmp_path / "output"
    run_local_sources(
        {source.name: source},
        tmp_path / "input",
        output,
        inputs={},
        max_workers=1,
        download_cache=tmp_path / "downloads",
    )
    records = {
        row["original_path"]: row
        for shard in (output / source.name / "normalize").glob("*.parquet")
        for row in load_parquet(str(shard))
    }
    assert set(records) == {row["path"] for row in rows}
    assert records["hidden-case"]["task_json"]
    assert records["code_contests-4395"]["task_json"]
    for path, reason in (
        ("comp-coding-dd2a13b32896.tar.gz", "reviewed_defect"),
        ("sample-only", "gold_in_instruction"),
    ):
        assert records[path]["task_json"] is None
        assert records[path]["normalization_kind"] == "source_defect"
        assert records[path]["normalization_reason"] == reason


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
    source = calendar.sources()[0]
    assert source.pipeline is not None
    source = replace(
        source,
        pipeline=replace(
            source.pipeline,
            source=HfSource("fixture/questions", revision, ("rows.jsonl",), SourceFormat.JSONL),
            inputs={"answers": HfSource("fixture/answers", revision, ("answer.txt",), SourceFormat.JSONL)},
            convert=answer_from_auxiliary,
            grader=None,
        ),
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
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == ExactSpec(("two",), ignore_case=False)
    assert not list(cache.rglob("unselected.jsonl"))

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
    assert isinstance(override_task.grader, VerifyitGrader)
    assert verifyit_spec(override_task.grader) == ExactSpec(("explicit answer",), ignore_case=False)

    next_revision = "b" * 40
    updated = remote / "fixture/questions" / next_revision
    updated.mkdir(parents=True)
    (updated / "rows.jsonl").write_text('{"prompt": "The next pinned question"}\n')
    assert source.pipeline is not None
    assert isinstance(source.pipeline.source, HfSource)
    source = replace(
        source, pipeline=replace(source.pipeline, source=replace(source.pipeline.source, revision=next_revision))
    )
    next_output = tmp_path / "next"
    run_local_sources({source.name: source}, None, next_output, inputs={}, max_workers=1, download_cache=cache)
    next_task = converted_task(next_output, source.name)
    assert next_task.source.revision == next_revision
    assert isinstance(next_task.context.events[0], TextMessage)
    assert next_task.context.events[0].content == "The next pinned question"
