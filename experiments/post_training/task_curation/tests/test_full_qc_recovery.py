# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

from marin.inference import iris as inference
from marin.inference.iris import RemoteInferenceSession
from marin.inference.openai_chat import OpenAIChatClient
from marin.inference.types import OpenAIEndpoint, RunningModel
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import ReviewRubric, ReviewStatus
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.full_qc import ReadyChatReviewer, local_reviewer
from experiments.post_training.task_curation.tests.review_server import running_review_server


@dataclass
class EndpointDirectory:
    address: str

    def resolve_endpoint(self, endpoint_name):
        return self.address


def test_review_reconnects_after_serving_moves_and_preserves_valid_cache(review_server, monkeypatch, tmp_path):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    spec = local_reviewer()
    first_url = f"http://127.0.0.1:{review_server.server_port}"
    directory = EndpointDirectory(first_url)
    monkeypatch.setattr(inference, "iris_ctx", lambda: SimpleNamespace(client=directory))
    session = RemoteInferenceSession(
        model=RunningModel(OpenAIEndpoint(first_url + "/v1", spec.model)),
        jobs=(),
        endpoint_name="/fixture/qa",
        endpoint_health_timeout_seconds=30,
        streaming=True,
        tensor_parallel_size=4,
        backend_name="fixture",
    )
    first = TaskSpec(
        id="apples",
        context=ConversationInput(events=(TextMessage(role="user", content="Count two apples."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Review fixture"),
        source=Source(dataset="fixture", revision="pin", row="apples", importer_revision="1"),
    )
    second = first.model_copy(
        update={
            "id": "pears",
            "context": ConversationInput(events=(TextMessage(role="user", content="Count three pears."),)),
        }
    )
    rubric = ReviewRubric("fixture", "1", ("Review the supplied task.",))
    cache = str(tmp_path / "cache")
    original = ChatReviewer(OpenAIChatClient(first_url + "/v1", "unused", timeout=2), spec, query_cache_root=cache)
    saved = original.review([first], rubric).reviews[0]
    review_server.review_available = False

    with running_review_server() as replacement:
        directory.address = f"http://127.0.0.1:{replacement.server_port}"
        recovered = ReadyChatReviewer(session, spec, cache, max_concurrent=1)
        result = recovered.review([first, second], rubric)

    assert result.reviews[0] == saved
    assert [record.status for record in result.reviews] == [ReviewStatus.REVIEWED, ReviewStatus.REVIEWED]
    assert len(review_server.requests) == 1
    assert len(replacement.requests) == 1
    assert session.model.endpoint.base_url == first_url + "/v1"
