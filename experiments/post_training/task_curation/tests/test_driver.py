# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import replace

import pytest
from click.testing import CliRunner
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.lazy import StepContext, run
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import HostImage, NetworkPolicy
from taskcompendium.convert.environment import grading_environment

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import (
    VerificationBackend,
    campaign_machines,
    job_controller_url,
    main,
)
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.environment_runtime import LocalRuntime, local_runtime
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
    return next(pipeline for pipeline in skyrl_math.pipelines() if pipeline.name == "math500")


@pytest.fixture
def catalog(monkeypatch):
    pipelines = {name: replace(math500(), name=name) for name in ("first", "second", "third")}
    monkeypatch.setattr("experiments.post_training.task_curation.driver.all_pipelines", lambda: pipelines)
    return pipelines


def arguments(tmp_path) -> list[str]:
    options = {
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


def test_full_run_depends_on_the_source_sample_and_reuses_its_trials(tmp_path, monkeypatch, catalog):
    captured = {}
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.run_campaign",
        lambda steps, **kwargs: captured.update(steps=steps, **kwargs),
    )
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    runner = CliRunner()
    sampled = runner.invoke(main, [*arguments(tmp_path), "--source", "first"])
    assert sampled.exit_code == 0, sampled.output
    (sample,) = json.loads(sampled.output)["sources"]
    result = runner.invoke(
        main,
        [*arguments(tmp_path), "--source", "first", "--mode", "full", "--base-url", "https://fixture.invalid", "--run"],
    )
    assert result.exit_code == 0, result.output
    (full,) = captured["steps"]
    assert captured["mode"] == "full"
    (previous,) = [dep for dep in full.deps if dep.name.startswith("data/rl/")]
    assert (previous.name, previous.fingerprint()) == (sample["name"], sample["fingerprint"])
    run = full.build_config(
        StepContext.for_run(str(tmp_path / "full-source"), str(tmp_path / "artifacts"), deps=full.deps)
    )
    assert run.previous_verification_report == previous.path() + "/verify/report.json"


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
