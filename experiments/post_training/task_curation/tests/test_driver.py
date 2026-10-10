# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import HostImage, NetworkPolicy
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.runtime.local import LocalRuntime, local_runtime
from zephyr.readers import load_parquet

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation import pipeline as processor
from experiments.post_training.task_curation.campaign import CampaignPool
from experiments.post_training.task_curation.config import ImageGraderPlacement
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    HfSource,
    campaign_machines,
    environment_requirements,
    process_rows,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.tests.image_builds import (
    REPOSITORY,
    install_fake_build_tools,
    tracked_lock,
)
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source

PINNED_WORKER = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64
CONTROLLER_URL = "http://controller.invalid"


def math500() -> CurationRecipe:
    return cast(CurationRecipe, next(source.config for source in skyrl_math.sources() if source.name == "math500"))


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    pipelines = {name: replace(math500(), name=name) for name in ("first", "second", "third")}
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.runnable_sources",
        lambda: {
            **{
                name: RlDataSource(
                    pipeline=process_rows,
                    info=SourceInfo(id=f"fixture:{name}", title=name, origin="fixture"),
                    config=pipeline,
                )
                for name, pipeline in pipelines.items()
            },
            "numbers": number_source(tmp_path / "not-yet-generated"),
        },
    )
    return pipelines


def arguments(tmp_path) -> list[str]:
    options = {
        "--mode": "sample",
        "--review-mode": "chat",
        "--model-revision": "fixture-revision",
        "--review-cache": str(tmp_path / "cache"),
        "--max-workers": "1",
        "--coordinator-memory": "16g",
        "--normalized-shards": "1",
        "--worker-image": "fixture-image",
        "--image-grader-placement": "gvisor",
        "--report-path": str(tmp_path / "report.json"),
    }
    return [item for pair in options.items() for item in pair]


def test_source_option_selects_catalog_order_without_changing_identity(tmp_path, catalog):
    runner = CliRunner()
    full = runner.invoke(main, arguments(tmp_path))
    subset = runner.invoke(main, [*arguments(tmp_path), "--source", "third", "--source", "first"])
    assert full.exit_code == 0, full.output
    assert subset.exit_code == 0, subset.output
    planned = json.loads(full.output)["sources"]
    assert len(planned) == 4 and planned[-1]["name"] == "data/rl/numbers"
    assert json.loads(subset.output)["sources"] == [planned[0], planned[2]]

    unknown = runner.invoke(main, [*arguments(tmp_path), "--source", "unknown", "--run"])
    assert unknown.exit_code == 2
    assert "Unknown source: unknown" in unknown.output


def test_local_environments_grade_in_the_worker_with_the_runtime_built_from_their_lock(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    install_fake_build_tools(tmp_path, monkeypatch)
    # The factory probes for a working bubblewrap; CI hosts cannot build sandboxes.
    monkeypatch.setattr("shellbox.backends.local.machine._working_bwrap", lambda candidates: Path("/usr/bin/bwrap"))
    environment = Environment(lock=tracked_lock(tmp_path), data=("nltk:punkt_tab",))
    (artifact,) = run(environment_artifact(environment, REPOSITORY))
    built = []
    monkeypatch.setattr(
        LocalRuntime, "ensure_built", lambda self: (self.root.mkdir(parents=True), built.append(self.root))
    )
    machines = campaign_machines(ImageGraderPlacement.IRIS, PINNED_WORKER, CONTROLLER_URL)
    factory, spec = machines.machine(environment_requirements(environment, artifact), 2048)
    runtime = local_runtime(artifact.lock_url)
    request.addfinalizer(lambda: shutil.rmtree(runtime.root, ignore_errors=True))
    assert runtime.lock_sha256 == artifact.lock_sha256 and runtime.data == ("nltk:punkt_tab",)
    assert built == [runtime.root]
    assert factory.read_only == (runtime.root,) and factory.bin_dirs == (runtime.root / "env" / "venv" / "bin",)
    assert spec.source == HostImage() and spec.network == NetworkPolicy.DENY and spec.workdir == "/app"
    assert spec.env == {"NLTK_DATA": str(runtime.root / "share" / "nltk_data")}


def iris_arguments(tmp_path) -> list[str]:
    """The fixture options with the verification backend left at its default, Iris."""
    options = arguments(tmp_path)
    index = options.index("--image-grader-placement")
    return options[:index] + options[index + 2 :]


def test_iris_backend_planning_does_not_require_a_controller(tmp_path, catalog):
    result = CliRunner().invoke(main, iris_arguments(tmp_path))
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "report.json").exists()


def test_quick_cli_plans_then_converts_once_in_request_order(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BULK_TOKEN", raising=False)
    output, cache = tmp_path / "output", tmp_path / "downloads"
    options = {
        "--mode": "quick",
        "--output-root": str(output),
        "--download-cache": str(cache),
        "--max-workers": "1",
    }
    args = [item for pair in options.items() for item in pair]
    args += ["--source", "math500", "--source", "aime24", "--source", "math500"]
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    assert [source["name"] for source in json.loads(result.output)["sources"]] == ["math500", "aime24"]
    assert not output.exists()
    assert not cache.exists()

    staged = tmp_path / "input"
    (staged / "data").mkdir(parents=True)
    (staged / "test.jsonl").write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    pq.write_table(
        pa.Table.from_pylist([{"problem": "One plus one?", "answer": "2"}]), staged / "data/train-0000.parquet"
    )
    result = runner.invoke(main, [*args, "--input-root", str(staged), "--run"])
    assert result.exit_code == 0, result.output
    report = json.loads((output / "campaign.json").read_text())
    assert [source["name"] for source in report["sources"]] == ["math500", "aime24"]
    assert report["counts"] == {"completed": 2}
    for name, source_row in (("math500", "test.jsonl:0"), ("aime24", "data/train-0000.parquet:0")):
        rows = [row for shard in (output / name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))]
        assert [row["source_row"] for row in rows] == [source_row]


@pytest.mark.parametrize("option", ["--input-root", "--input-file", "--input", "--output-root", "--download-cache"])
def test_reviewed_cli_rejects_local_input_and_output_options(tmp_path, catalog, option):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    file = inputs / "fixture.jsonl"
    file.write_text('{"prompt": "fixture"}\n')
    values = {
        "--input-root": [str(inputs)],
        "--input-file": ["fixture.jsonl", str(file)],
        "--input": ["answers", str(inputs)],
        "--output-root": [str(tmp_path / "output")],
        "--download-cache": [str(tmp_path / "downloads")],
    }
    result = CliRunner().invoke(main, [*arguments(tmp_path), option, *values[option]])
    assert result.exit_code == 2
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "downloads").exists()


@pytest.mark.parametrize("mode, expected", [("quick", [2, 4, 6]), ("sample", [2, 4]), ("full", [2, 4, 6])])
def test_dataset_cli_runs_own_ingestion_without_review_controller_or_build_settings(
    tmp_path, monkeypatch, mode, expected
):
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    input_file = tmp_path / "numbers.txt"
    input_file.write_text("1\n2\n3\n")
    source = number_source(input_file)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    options = ["--mode", mode, "--source", source.name, "--max-workers", "1"]
    if mode == "quick":
        output_root = tmp_path / "output"
        report = output_root / "campaign.json"
        options += ["--output-root", str(output_root), "--download-cache", str(tmp_path / "downloads")]
    else:
        report = tmp_path / "report.json"
        options += ["--coordinator-memory", "1g", "--worker-image", "fixture-image", "--report-path", str(report)]
    runner = CliRunner()
    planned = runner.invoke(main, options)
    assert planned.exit_code == 0, planned.output
    assert len(json.loads(planned.output)["sources"]) == 1
    assert not report.exists()
    assert not (tmp_path / "downloads").exists()
    assert not (tmp_path / "artifacts").exists()
    client = LocalClient()
    try:
        with set_current_client(client):
            executed = runner.invoke(main, [*options, "--run"])
        assert executed.exit_code == 0, executed.output
        outcome = json.loads(report.read_text())["sources"][0]
        result = outcome["result"]
        assert json.loads(StoragePath(result["outputs"]["numbers"]).read_text()) == expected
        evidence = json.loads(StoragePath(result["evidence"]["ingestion"]).read_text())
        assert evidence["mode"] == mode
        assert evidence["catalog_id"] == source.info.id
        assert result["stages"] == ["ingest", "multiply"]
        if mode == "quick":
            # The generated pipeline really consumes its upstream artifact.
            ingested = list((tmp_path / "downloads").rglob("numbers.txt"))
            assert len(ingested) == 1
            assert ingested[0].read_text() == "1\n2\n3\n"
            input_file.unlink()
            StoragePath(result["outputs"]["numbers"]).write_text("[]")
            repeated = runner.invoke(main, [*options, "--run"])
            assert repeated.exit_code == 0, repeated.output
            assert json.loads(StoragePath(result["outputs"]["numbers"]).read_text()) == expected
            assert Path(os.environ["MARIN_PREFIX"]) == tmp_path / "artifacts"
    finally:
        client.shutdown()


@pytest.mark.parametrize("mode", ["sample", "full"])
@pytest.mark.parametrize("rubric", [None, "Check the reference answer."])
def test_recipe_cli_requires_model_settings_only_when_review_is_chosen(tmp_path, monkeypatch, mode, rubric):
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.CampaignPool",
        partial(CampaignPool, chunk_storage_prefix=str(tmp_path / "chunks")),
    )
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    source = replace(
        declared,
        config=replace(
            math500(),
            source=HfSource("fixture/questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
            rubric=rubric,
            controls=None,
            grader=None,
        ),
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "rows.jsonl").write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    original = processor.plan_download
    monkeypatch.setattr(
        processor, "plan_download", lambda request: original(replace(request, source_url_override=str(remote)))
    )
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    report = tmp_path / "report.json"
    options = [
        "--mode",
        mode,
        "--max-workers",
        "1",
        "--coordinator-memory",
        "1g",
        "--worker-image",
        "fixture-image",
        "--normalized-shards",
        "1",
        "--report-path",
        str(report),
    ]
    runner = CliRunner()
    planned = runner.invoke(main, options)
    assert planned.exit_code == (2 if rubric is not None else 0), planned.output
    assert not (tmp_path / "artifacts").exists()
    assert not report.exists()
    client = LocalClient()
    try:
        with set_current_client(client):
            executed = runner.invoke(main, [*options, "--run"])
        if rubric is not None:
            assert executed.exit_code == 2, executed.output
            assert not (tmp_path / "artifacts").exists()
            assert not report.exists()
            return
        assert executed.exit_code == 0, executed.output
        result = json.loads(report.read_text())["sources"][0]["result"]
        rows = [
            row
            for shard in (StoragePath(result["outputs"]["final"]) / "*.parquet").glob()
            for row in load_parquet(str(shard))
        ]
        assert [row["source_row"] for row in rows] == ["rows.jsonl:0"]
        assert result["stages"] == ["download", "normalize", "final"]
    finally:
        client.shutdown()
