# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.binding import pipeline_step
from experiments.post_training.task_curation.campaign import (
    CampaignArtifact,
    CampaignFailed,
    CampaignPool,
    CampaignRuntime,
    run_campaign,
)
from experiments.post_training.task_curation.invocation import PipelineResult
from experiments.post_training.task_curation.tests.custom_pipeline import number_source


@pytest.mark.parametrize(
    "mode, expected", [(SourceProcessingMode.SAMPLE, [3, 6]), (SourceProcessingMode.FULL, [3, 6, 9])]
)
def test_custom_campaign_retains_named_products_without_standard_dependencies(tmp_path, monkeypatch, mode, expected):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    primary = tmp_path / "numbers.txt"
    primary.write_text("1\n2\n3\n")
    unused = tmp_path / "unused.txt"
    factor = tmp_path / "factor.txt"
    factor.write_text("3")
    runtime = CampaignRuntime()
    step = pipeline_step(
        number_source(unused), mode=mode, campaign=runtime, source_input=str(primary), inputs={"factor": str(factor)}
    )
    client = LocalClient()
    try:
        with set_current_client(client):
            outcomes = run_campaign(
                [step],
                runtime=runtime,
                pool=CampaignPool(
                    1,
                    coordinator_resources=ResourceConfig(cpu=1, ram="1g"),
                    chunk_storage_prefix=str(tmp_path / "chunks"),
                ),
                report_path=str(tmp_path / "report.json"),
                mode=mode,
            )
        result = cast(PipelineResult, outcomes[0].result)
        assert json.loads(StoragePath(result.outputs["numbers"]).read_text()) == expected
        assert json.loads(StoragePath(result.evidence["ingestion"]).read_text()) == {
            "source": "numbers",
            "mode": mode,
            "input": str(primary),
        }
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["sources"][0]["result"]["stages"] == ["ingest", "multiply"]
        assert report["sources"][0]["result"]["outputs"] == result.outputs
        assert not (Path(outcomes[0].path) / "final").exists()
    finally:
        client.shutdown()


class CampaignResult(CampaignArtifact):
    status: str = "completed"
    rows: list[int]


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
    try:
        with set_current_client(client):
            with pytest.raises(CampaignFailed):
                run_campaign(steps, runtime=runtime, pool=pool, report_path=report)
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
    finally:
        client.shutdown()
