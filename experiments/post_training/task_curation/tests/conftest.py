# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from dataclasses import replace

import pytest
from taskcompendium.pipeline.reviewer_spec import JudgeTier

from experiments.post_training.task_curation import driver
from experiments.post_training.task_curation.judges import JUDGE_TIERS, HostedJudgeEndpoint
from experiments.post_training.task_curation.tests.local_grader import LocalGraderMachines, local_grader_machines
from experiments.post_training.task_curation.tests.review_server import ReviewHTTPServer


@pytest.fixture(scope="session")
def machines() -> LocalGraderMachines:
    """Docker machines running the local grader image; requesting tests skip without Docker or the image."""
    return local_grader_machines()


@pytest.fixture
def review_server():
    server = ReviewHTTPServer()
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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
