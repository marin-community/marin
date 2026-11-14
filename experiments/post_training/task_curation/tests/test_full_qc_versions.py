# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from marin.inference.config import EffectiveServing
from marin.inference.openai_chat import OpenAIChatClient
from marin.inference.types import OpenAIEndpoint, RunningModel
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation import full_qc, judges
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
    assert "top_p" not in review_server.requests[0] and "top_k" not in review_server.requests[0]
    assert review_server.requests[1]["top_p"] == 0.95
    assert review_server.requests[1]["top_k"] == 0
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


def test_v3_serves_and_reviews_its_literal_pin_after_shared_presets_change(review_server, monkeypatch):
    spec = local_reviewer(ReviewerVersion.V3)
    identity = spec.identity
    definition = judges.JUDGE_TIERS[spec.tier]
    changed = replace(
        definition,
        model_identity="changed/model",
        model_revision="changed-revision",
        context_window_tokens=128,
        spec=replace(definition.spec, model="changed-api"),
        qa_reviewer=replace(
            definition.qa_reviewer,
            model="changed-api",
            model_revision="changed-revision",
            temperature=0.2,
            reasoning_effort="high",
            max_tokens=10,
            retry_max_tokens=11,
            max_attempts=1,
            prompt_budget=128,
            retry_prompt_budget=128,
        ),
    )
    presets = {**judges.JUDGE_TIERS, spec.tier: changed}
    monkeypatch.setattr(judges, "JUDGE_TIERS", presets)
    monkeypatch.setattr(full_qc, "JUDGE_TIERS", presets)
    spec = local_reviewer(ReviewerVersion.V3)
    assert spec.identity == identity
    base_url = f"http://127.0.0.1:{review_server.server_port}/v1"
    review_server.model_ids = ["glm-5.3"]
    configurations = []

    @contextmanager
    def inference(config):
        configurations.append(config)
        session = SimpleNamespace(
            model=RunningModel(OpenAIEndpoint(base_url, "glm-5.3")),
            effective_serving=EffectiveServing(
                tensor_parallel_size=4,
                data_parallel_size=1,
                pipeline_parallel_size=2,
                task_count=2,
                max_model_len=524288,
            ),
            jobs=(),
        )
        session.ready_direct_session = lambda: session
        yield session

    monkeypatch.setattr(judges, "remote_inference", inference)
    task = TaskSpec.model_validate(json.loads(prebuilt_task("literal")["lowered_task_spec"])["task"])
    with spec.endpoint.connect() as connection:
        reviewer = ChatReviewer(OpenAIChatClient(connection.base_url, connection.api_key, timeout=2), spec)
        result = reviewer.review([task], REVIEW_RUBRIC)
    assert result.reviews[0].status is ReviewStatus.REVIEWED
    assert configurations[0].model.model_id == "glm-5.3"
    assert configurations[0].model.max_model_len == 524288
    assert review_server.requests[0]["temperature"] == 0.7
    assert review_server.requests[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}
