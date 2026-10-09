# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.lazy import run
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import HostImage, NetworkPolicy
from taskcompendium.convert.environment import grading_environment
from taskcompendium.runtime.local import LocalRuntime, local_runtime
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import (
    VerificationBackend,
    campaign_machines,
    job_controller_url,
    main,
)
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.pipeline import environment_requirements
from experiments.post_training.task_curation.tests.image_builds import (
    REPOSITORY,
    install_fake_build_tools,
    tracked_lock,
)

PINNED_WORKER = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64
CONTROLLER_URL = "http://controller.invalid"
GRADER = "ghcr.io/marin-community/task-curation-grader@sha256:" + "b" * 64


def math500():
    return next(
        source.pipeline for source in skyrl_math.sources() if source.name == "math500" and source.pipeline is not None
    )


@pytest.fixture
def catalog(monkeypatch):
    pipelines = {name: replace(math500(), name=name) for name in ("first", "second", "third")}
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_pipelines", lambda: pipelines)
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
        "--verification-backend": "gvisor",
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
    assert json.loads(subset.output)["sources"] == [planned[0], planned[2]]

    unknown = runner.invoke(main, [*arguments(tmp_path), "--source", "unknown", "--run"])
    assert unknown.exit_code == 2
    assert "Unknown source: unknown" in unknown.output


def test_iris_schedules_each_grader_image_on_the_controller_without_network():
    machines = campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, CONTROLLER_URL)
    factory, spec = machines.machine(grading_environment(GRADER), 2048)
    assert isinstance(factory, IrisMachineFactory)
    assert spec.source == RegistryImage(GRADER)
    assert spec.network == NetworkPolicy.DENY
    assert spec.memory_mb == 2048


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
    machines = campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, CONTROLLER_URL)
    factory, spec = machines.machine(environment_requirements(environment, artifact), 2048)
    runtime = local_runtime(artifact.lock_url)
    request.addfinalizer(lambda: shutil.rmtree(runtime.root, ignore_errors=True))
    assert runtime.lock_sha256 == artifact.lock_sha256 and runtime.data == ("nltk:punkt_tab",)
    assert isinstance(factory, LocalMachineFactory)
    assert built == [runtime.root]
    assert factory.read_only == (runtime.root,) and factory.bin_dirs == (runtime.root / "env" / "venv" / "bin",)
    assert spec.source == HostImage() and spec.network == NetworkPolicy.DENY and spec.workdir == "/app"
    assert spec.env == {"NLTK_DATA": str(runtime.root / "share" / "nltk_data")}


def test_iris_verification_requires_a_controller():
    with pytest.raises(ValueError, match="controller URL"):
        campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, None)


def test_machine_identity_records_the_backend_and_controller_presence():
    iris = campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, CONTROLLER_URL).identity()
    gvisor = campaign_machines(VerificationBackend.GVISOR, PINNED_WORKER, None).identity()
    assert (iris["backend"], iris["controller"]) == ("iris", True)
    assert (gvisor["backend"], gvisor["controller"]) == ("gvisor", False)
    other_controller = campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, "http://other.invalid").identity()
    assert other_controller == iris


@pytest.fixture
def iris_job():
    set_job_info(JobInfo(task_id=JobName.from_string("/fixture/driver/0"), controller_address=CONTROLLER_URL))
    yield
    set_job_info(None)


def iris_arguments(tmp_path) -> list[str]:
    """The fixture options with the verification backend left at its default, Iris."""
    options = arguments(tmp_path)
    index = options.index("--verification-backend")
    return options[:index] + options[index + 2 :]


def test_iris_backend_outside_a_job_requires_a_controller_url(tmp_path, catalog):
    result = CliRunner().invoke(main, iris_arguments(tmp_path))
    assert result.exit_code == 2
    assert "requires --controller-url" in result.output
    explicit = CliRunner().invoke(main, [*iris_arguments(tmp_path), "--controller-url", CONTROLLER_URL])
    assert explicit.exit_code == 0, explicit.output


def test_iris_backend_inside_a_job_uses_the_job_controller(tmp_path, catalog, iris_job):
    assert job_controller_url() == CONTROLLER_URL
    result = CliRunner().invoke(main, iris_arguments(tmp_path))
    assert result.exit_code == 0, result.output


def test_quick_cli_converts_selected_sources_once_in_request_order(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BULK_TOKEN", raising=False)
    staged = tmp_path / "input"
    (staged / "data").mkdir(parents=True)
    (staged / "test.jsonl").write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    pq.write_table(
        pa.Table.from_pylist([{"problem": "One plus one?", "answer": "2"}]), staged / "data/train-0000.parquet"
    )
    output = tmp_path / "output"
    result = CliRunner().invoke(
        main,
        [
            "--mode",
            "quick",
            "--run",
            "--source",
            "math500",
            "--source",
            "aime24",
            "--source",
            "math500",
            "--input-root",
            str(staged),
            "--output-root",
            str(output),
            "--max-workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output / "campaign.json").read_text())
    assert [source["name"] for source in report["sources"]] == ["math500", "aime24"]
    assert report["counts"] == {"completed": 2}
    for name, source_row in (("math500", "test.jsonl:0"), ("aime24", "data/train-0000.parquet:0")):
        rows = [row for shard in (output / name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))]
        assert [row["source_row"] for row in rows] == [source_row]

    unknown = CliRunner().invoke(
        main, ["--mode", "quick", "--source", "unknown", "--output-root", str(tmp_path / "unknown"), "--run"]
    )
    assert unknown.exit_code == 2
    assert not (tmp_path / "unknown").exists()


def test_quick_cli_without_run_plans_in_request_order_without_staging(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BULK_TOKEN", raising=False)
    output, cache = tmp_path / "output", tmp_path / "downloads"
    result = CliRunner().invoke(
        main,
        [
            "--mode",
            "quick",
            "--source",
            "math500",
            "--source",
            "aime24",
            "--source",
            "math500",
            "--output-root",
            str(output),
            "--download-cache",
            str(cache),
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["mode"] == "quick"
    assert [source["name"] for source in plan["sources"]] == ["math500", "aime24"]
    assert not output.exists()
    assert not cache.exists()


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
