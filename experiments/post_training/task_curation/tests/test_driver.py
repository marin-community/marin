# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest
from click.testing import CliRunner
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import NetworkPolicy

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import (
    VerificationBackend,
    campaign_machines,
    job_controller_url,
    main,
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
        "--review-transport": "direct-chat",
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


def test_full_run_reuses_only_admitted_sample_outputs(tmp_path, monkeypatch, catalog):
    captured = {}
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.run_campaign",
        lambda steps, **kwargs: captured.update(steps=steps, **kwargs),
    )
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    runner = CliRunner()
    planned = runner.invoke(main, arguments(tmp_path))
    assert planned.exit_code == 0, planned.output
    sources = json.loads(planned.output)["sources"]
    identity = hashlib.sha256(
        canonical_json(
            {
                "sources": sorted((s["name"], s["version"], s["fingerprint"]) for s in sources),
                "worker_image": "fixture-image",
            }
        ).encode()
    ).hexdigest()
    statuses = ("sampled", "failed", "gated")
    outcomes = [
        {
            "name": source["name"],
            "path": str(tmp_path / "artifacts" / source["name"] / source["version"]),
            "status": status,
            "error": None,
        }
        for source, status in zip(sources, statuses, strict=True)
    ]
    sample_report = tmp_path / "sample.json"
    sample_report.write_text(
        json.dumps(
            {
                "mode": "sample",
                "status": "failed",
                "sample_identity": identity,
                "counts": dict(Counter(statuses)),
                "sources": outcomes,
            }
        )
    )
    result = runner.invoke(
        main,
        [
            *arguments(tmp_path),
            "--mode",
            "full",
            "--sample-report",
            str(sample_report),
            "--base-url",
            "https://fixture.invalid",
            "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    admitted, failed, gated = captured["steps"]
    samples = [dep for dep in admitted.deps if dep.name.startswith("task-curation/sample/first-")]
    assert len(samples) == 1
    assert samples[0].adopt_source == outcomes[0]["path"]
    assert samples[0].adopt_config == {
        "campaign_report": str(sample_report),
        "sample_identity": identity,
        "sample_source": sources[0]["name"],
        "sample_fingerprint": sources[0]["fingerprint"],
        "sample_path": outcomes[0]["path"],
    }
    run = admitted.build_config(
        StepContext.for_run(str(tmp_path / "full-source"), str(tmp_path / "artifacts"), deps=admitted.deps)
    )
    assert run.previous_verification_report == outcomes[0]["path"] + "/verify/report.json"
    for step in (failed, gated):
        assert not any(dep.name.startswith("task-curation/sample/") for dep in step.deps)
    assert [captured["sample_outcomes"][step.name].status for step in captured["steps"]] == list(statuses)
    assert captured["sample_identity"] == identity


def test_iris_schedules_each_grader_image_on_the_controller_without_network():
    factory, spec = campaign_machines(VerificationBackend.IRIS, PINNED_WORKER, CONTROLLER_URL).machine(GRADER, 2048)
    assert isinstance(factory, IrisMachineFactory)
    assert spec.source == RegistryImage(GRADER)
    assert spec.network == NetworkPolicy.DENY
    assert spec.memory_mb == 2048


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
