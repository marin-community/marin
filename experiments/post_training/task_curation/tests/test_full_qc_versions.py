# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.inference.openai_chat import OpenAIChatClient
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


def test_v3_generates_separately_from_v2_and_reuses_its_own_valid_cache(review_server, tmp_path):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("versioned")["lowered_task_spec"])["task"])
    client = OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2)
    reviewers = [
        ChatReviewer(client, local_reviewer(version), query_cache_root=str(tmp_path / "cache"))
        for version in ReviewerVersion
    ]
    for reviewer in reviewers:
        result = reviewer.review([task], REVIEW_RUBRIC)
        assert result.reviews[0].status is ReviewStatus.REVIEWED
    assert [body["max_tokens"] for body in review_server.requests] == [4096, 8192]
    for reviewer in reviewers:
        reviewer.review([task], REVIEW_RUBRIC)
    assert len(review_server.requests) == 2


def test_v3_retries_truncation_with_the_doubled_generation_budget(review_server):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("truncated")["lowered_task_spec"])["task"])
    client = OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2)
    budgets = []

    class TruncatedFirstResponse:
        def complete(self, body):
            budgets.append(body["max_tokens"])
            if len(budgets) == 1:
                return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            return client.complete(body)

    reviewer = ChatReviewer(TruncatedFirstResponse(), local_reviewer(ReviewerVersion.V3))
    result = reviewer.review([task], REVIEW_RUBRIC)
    assert result.reviews[0].status is ReviewStatus.REVIEWED
    assert budgets == [8192, 16384]
