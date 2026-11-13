# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskcompendium.pipeline.reviewer_spec import JudgeTier

from experiments.post_training.task_curation import driver, registry
from experiments.post_training.task_curation.judges import JUDGE_TIERS, HostedJudgeEndpoint
from experiments.post_training.task_curation.tests.local_grader import LocalGraderMachines, local_grader_machines
from experiments.post_training.task_curation.tests.review_server import running_review_server


@pytest.fixture(autouse=True)
def offline_registry(request, monkeypatch):
    if request.node.get_closest_marker("data_integration") is not None:
        return

    def unavailable(**kwargs):
        raise AssertionError("Unit tests must inject local registry documents at the download boundary")

    monkeypatch.setattr(registry, "hf_hub_download", unavailable)
    monkeypatch.setattr(driver, "all_sources", lambda: {})


@pytest.fixture(scope="session")
def machines() -> LocalGraderMachines:
    """Docker machines running the local grader image; requesting tests skip without Docker or the image."""
    return local_grader_machines()


@pytest.fixture
def review_server():
    with running_review_server() as server:
        yield server


@pytest.fixture
def qa_reviewer(review_server, monkeypatch):
    definition = JUDGE_TIERS[JudgeTier.HEAVY]
    assert definition.qa_reviewer is not None
    spec = replace(
        definition.qa_reviewer,
        endpoint=HostedJudgeEndpoint("glm-5.3", f"http://127.0.0.1:{review_server.server_port}/v1", "fixture"),
    )
    review_server.model_ids.append(spec.model)
    monkeypatch.setattr(driver, "JUDGE_TIERS", {**JUDGE_TIERS, JudgeTier.HEAVY: replace(definition, qa_reviewer=spec)})
    return spec
