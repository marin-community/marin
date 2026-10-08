# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections import Counter
from dataclasses import replace

import pytest
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.campaign import (
    CampaignArtifact,
    CampaignFailed,
    CampaignPool,
    CampaignRuntime,
    SourceOutcome,
    campaign_identity,
    require_matching_sample,
    run_campaign,
)


class CampaignResult(CampaignArtifact):
    status: str = "completed"
    rows: list[int]


def test_full_admission_rejects_changed_source_model_and_runtime():
    def source(model):
        return ArtifactStep(
            "source",
            "2026.10.06",
            Artifact,
            lambda config: None,
            lambda ctx: {"revision": "pinned-input", "model": model},
        )

    sample = [source("model-revision-1")]
    identity = campaign_identity(sample, "image@sha256:original")
    report = {
        "status": "completed",
        "mode": "sample",
        "sample_identity": identity,
        "counts": {"sampled": 1},
        "sources": [{"name": sample[0].name, "path": sample[0].path(), "status": "sampled"}],
    }
    require_matching_sample(report, campaign_identity(sample, "image@sha256:original"), sample)
    for changed in (
        campaign_identity([source("model-revision-2")], "image@sha256:original"),
        campaign_identity(sample, "image@sha256:changed"),
        campaign_identity([], "image@sha256:original"),
    ):
        with pytest.raises(ValueError, match="matching"):
            require_matching_sample(report, changed, sample)


def test_campaign_shares_pool_preserves_peers_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    runtime = CampaignRuntime()
    client = LocalClient(max_threads=16)
    barrier = threading.Barrier(2)
    contexts = []
    calls = []
    failed = {"source-1"}

    def step(name):
        def config(ctx: StepContext):
            return {"output": ctx.output_path, "name": name}

        def execute(config):
            assert current_client() is client
            contexts.append(runtime.context)
            calls.append(name)
            if len(calls) <= 2:
                barrier.wait(timeout=10)
            rows = runtime.context.execute(Dataset.from_list([1, 2]).map(lambda value: value * 3)).results
            if name in failed:
                raise RuntimeError("fixture source failed")
            return CampaignResult(path=config["output"], rows=rows)

        return ArtifactStep(name, "2026.10.06", CampaignResult, execute, config)

    steps = [step(f"source-{index}") for index in range(4)]
    pool = CampaignPool(
        2,
        concurrent_sources=2,
        coordinator_resources=ResourceConfig(cpu=1, ram="16g", preemptible=False),
        chunk_storage_prefix=str(tmp_path / "chunks"),
    )
    report = str(tmp_path / "report.json")
    identity = campaign_identity(steps, "fixture-image")
    try:
        with set_current_client(client):
            with pytest.raises(CampaignFailed):
                run_campaign(steps, runtime=runtime, pool=pool, report_path=report, sample_identity=identity)
            with pytest.raises(RuntimeError, match="active campaign"):
                _ = runtime.context
            first = json.loads((tmp_path / "report.json").read_text())
            assert first["status"] == "failed"
            assert [item["status"] for item in first["sources"]] == ["completed", "failed", "completed", "completed"]
            assert "RuntimeError: fixture source failed" in first["sources"][1]["error"]
            assert len({id(context) for context in contexts}) == 1
            failed.clear()
            outcomes = run_campaign(steps, runtime=runtime, pool=replace(pool, concurrent_sources=1), report_path=report)
            with pytest.raises(RuntimeError, match="active campaign"):
                _ = runtime.context
            assert all(outcome.status == "completed" for outcome in outcomes)
            assert calls.count("source-0") == 1
            assert calls.count("source-1") == 2
            assert calls.count("source-2") == 1
            assert calls.count("source-3") == 1
            assert CampaignResult.raw_load(outcomes[1].path).rows == [3, 6]
            sample = require_matching_sample(first, identity, steps)
            full_steps = [replace(step, name=f"full/{step.name}") for step in steps]
            full = run_campaign(
                full_steps,
                runtime=runtime,
                pool=pool,
                report_path=report,
                sample_identity=identity,
                mode="full",
                sample_outcomes={step.name: outcome for step, outcome in zip(full_steps, sample, strict=True)},
            )
            assert [outcome.status for outcome in full] == ["completed", "not_admitted", "completed", "completed"]
            assert calls.count("source-1") == 2
            final = json.loads((tmp_path / "report.json").read_text())
            assert final["counts"] == {"completed": 3, "not_admitted": 1}
            assert final["sample_outcomes"]["full/source-1"] == first["sources"][1]
    finally:
        client.shutdown()


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "unmatched", "path", "running", "counts"])
def test_full_admission_cannot_use_inconsistent_source_coverage(corruption):
    steps = [
        ArtifactStep(name, "2026.10.06", Artifact, lambda config: None, lambda ctx: {}) for name in ("healthy", "failed")
    ]
    identity = campaign_identity(steps, "fixture-image")
    sources = [
        {"name": step.name, "path": step.path(), "status": "sampled" if index == 0 else "failed"}
        for index, step in enumerate(steps)
    ]
    if corruption == "duplicate":
        sources[1] = sources[0]
    elif corruption == "missing":
        sources.pop()
    elif corruption == "unmatched":
        sources[1]["name"] = "different-source"
    elif corruption == "path":
        sources[0]["path"] = "different-input-artifact"
    elif corruption == "running":
        sources[1]["status"] = "running"
    report = {
        "mode": "sample",
        "status": "failed",
        "sample_identity": identity,
        "sources": sources,
        "counts": dict(Counter(source["status"] for source in sources)),
    }
    if corruption == "counts":
        report["counts"] = {"sampled": 2}
    with pytest.raises(ValueError):
        require_matching_sample(report, identity, steps)


def test_full_campaign_records_gated_and_incomplete_samples_without_dispatch(tmp_path):
    def unexpected_execution(_config):
        raise AssertionError("A gated or incomplete source must not execute")

    steps = [
        ArtifactStep(name, "2026.10.06", CampaignArtifact, unexpected_execution, lambda ctx: {})
        for name in ("gated", "incomplete")
    ]
    samples = {
        step.name: SourceOutcome(f"sample/{step.name}", f"sample/{step.name}/2026.10.06", step.name) for step in steps
    }
    path = tmp_path / "full.json"
    outcomes = run_campaign(
        steps,
        runtime=CampaignRuntime(),
        pool=CampaignPool(1, coordinator_resources=ResourceConfig(cpu=1, ram="16g", preemptible=False)),
        report_path=str(path),
        mode="full",
        sample_outcomes=samples,
    )
    assert [outcome.status for outcome in outcomes] == ["not_admitted", "not_admitted"]
    report = json.loads(path.read_text())
    assert report["status"] == "completed"
    assert report["counts"] == {"not_admitted": 2}
    assert report["sample_outcomes"]["gated"]["status"] == "gated"
    assert report["sample_outcomes"]["incomplete"]["status"] == "incomplete"
