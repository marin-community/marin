# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext
from shellbox.machine import NetworkPolicy, QemuBundle, UnsupportedMachineSpec

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import (
    CampaignMachines,
    VerificationBackend,
    main,
    qemu_bundles,
)
from experiments.post_training.task_curation.images import APPS_IMAGE, ARC_IMAGE

PINNED_WORKER = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64


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


def test_qemu_runs_committed_images_from_their_worker_bundles():
    qemu = CampaignMachines(VerificationBackend.QEMU, PINNED_WORKER, None, qemu_bundles())
    factory, spec = qemu.machine(APPS_IMAGE.reference, 2048)
    assert factory.backend.value == "qemu"
    assert spec.source == QemuBundle(Path(APPS_IMAGE.qemu_bundle))
    assert spec.network == NetworkPolicy.DENY
    assert spec.memory_mb == 2048
    with pytest.raises(UnsupportedMachineSpec):
        qemu.machine(ARC_IMAGE.reference, 2048)
