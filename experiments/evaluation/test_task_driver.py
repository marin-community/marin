# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise driver recovery through HTTP, FineStore, and Iris RPC boundaries."""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import openai
import pytest
from finestore.reader import ReadView
from fray.local_backend import LocalClient
from iris.client.client import IrisClient
from iris.cluster.client import remote_client
from iris.cluster.types import ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation.task_driver import (
    ATTEMPTS_TABLE,
    EvaluationInfrastructureError,
    EvaluationRunConfig,
    ModelIdentity,
)
from rolloutengine.models import RolloutLimits
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.evaluation import EvaluationProtocol, Metric, MetricKind, Sampling, TrialPolicy
from taskcompendium.pipeline.evaluation import chat_item, publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from zephyr.context import ZephyrContext

from experiments.evaluation import task_driver
from experiments.evaluation.task_driver import SOLVER_API_KEY, SOLVER_ENDPOINT, execute_with_retries, submit_evaluation


@pytest.fixture
def driver_config():
    return EvaluationRunConfig(
        model=ModelIdentity(
            name="solver",
            revision="model-pin",
            tokenizer="solver",
            tokenizer_revision="tokenizer-pin",
            context_window=1024,
            serving={"backend": "fixture"},
        ),
        max_tokens=16,
        concurrency=1,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=10,
            tool_turn_timeout=10,
            total_turn_timeout=20,
            attempt_timeout=30,
            verifier_timeout=10,
            cleanup_timeout=10,
        ),
    )


def recovery_task(row, _context):
    task = exact_answer_task(row, prompt=row.data["question"], answers=("red",), ignore_case=False)
    return task.model_copy(update={"tags": ("eval",)})


@pytest.fixture
def recovery_corpora(tmp_path):
    client = LocalClient()
    roots = []
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            for name in ("colors", "paints"):
                source = tmp_path / name
                source.mkdir()
                (source / "rows.jsonl").write_text(
                    "".join(json.dumps({"question": f"{name}/{index}"}) + "\n" for index in range(2))
                )
                recipe = SourceRecipe(
                    name=name,
                    version="1",
                    source=SourceFiles(name, "pinned", ("rows.jsonl",), SourceFormat.JSONL),
                    convert=recovery_task,
                    rubric=None,
                    controls=None,
                    intended_use=IntendedUse.EVAL,
                )
                protocol = EvaluationProtocol(
                    benchmark=name,
                    version="1",
                    operation="chat_generation",
                    sampling=Sampling(
                        temperature=0, top_p=None, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={}
                    ),
                    trials=TrialPolicy(count=2, seed=7),
                    metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
                    primary_metric="accuracy",
                    reducer="mean_v1",
                    expected_population=2,
                )
                output = tmp_path / (name + "-published")
                publish_evaluation_corpus(recipe, protocol, chat_item, context, str(source), str(output))
                roots.append(str(output))
        return tuple(roots)
    finally:
        client.shutdown()


@dataclass
class RecoveryEndpoint:
    failures_remaining: int
    status: int = 503
    requests: list[dict] = field(default_factory=list)

    def handle(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        question = payload["messages"][-1]["content"]
        if question == "colors/0" and payload["seed"] == 7 and self.failures_remaining:
            self.failures_remaining -= 1
            return httpx.Response(self.status, json={"error": {"message": "temporary fixture failure"}})
        content = "blue" if question == "paints/1" else "red"
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "solver",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )


@pytest.fixture
def recovery_endpoint(monkeypatch):
    endpoint = RecoveryEndpoint(failures_remaining=1)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[BOS]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        bos_token="[BOS]",
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
    )
    monkeypatch.setattr(task_driver.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)

    def client(**kwargs):
        return openai.AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handle))
        )

    monkeypatch.setattr(task_driver, "AsyncOpenAI", client)
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _delay: None)
    return endpoint


def test_retry_drains_later_corpora_and_resumes_only_failed_attempt_with_same_seed(
    tmp_path, recovery_corpora, recovery_endpoint, driver_config
):
    output = str(tmp_path / "results")
    execute_with_retries(recovery_corpora, output, driver_config, "http://fixture.test/v1", "unused", None)
    terminal = [row for row in ReadView(output).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    failed = [row for row in terminal if row["outcome"] == "infra_error"]
    graded = [row for row in terminal if row["outcome"] == "graded"]
    assert len(failed) == 1 and len(graded) == 8
    assert [(row["task_id"], row["retry"], row["seed"]) for row in graded if row["retry"]] == [
        (failed[0]["task_id"], 1, 7)
    ]
    assert recovery_endpoint.requests[-1] == recovery_endpoint.requests[0]
    assert {(r["messages"][-1]["content"], r["seed"]) for r in recovery_endpoint.requests[:-1]} == {
        (f"{name}/{index}", seed) for name in ("colors", "paints") for index in range(2) for seed in (7, 8)
    }
    reader = ReadView(output)
    assert sum(dict(row["metrics"]).get("accuracy", 0) for row in reader.iter_rows("samples")) == 6
    assert (
        json.loads(reader.read_blob(failed[0]["result_blob"]))["failure_evidence"]["provider_error"]["message"]
        == "temporary fixture failure"
    )


def test_retry_exhaustion_preserves_all_failed_physical_attempts_and_later_grades(
    tmp_path, recovery_corpora, recovery_endpoint, driver_config
):
    recovery_endpoint.failures_remaining = 10
    output = str(tmp_path / "results")
    with pytest.raises(EvaluationInfrastructureError):
        execute_with_retries(recovery_corpora, output, driver_config, "http://fixture.test/v1", "unused", None)
    terminal = [row for row in ReadView(output).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert sorted(row["retry"] for row in terminal if row["outcome"] == "infra_error") == [0, 1, 2]
    assert len([row for row in terminal if row["outcome"] == "graded"]) == 7


@pytest.mark.parametrize(
    "status,error_type", [(400, openai.BadRequestError), (401, openai.AuthenticationError), (404, openai.NotFoundError)]
)
def test_permanent_endpoint_errors_propagate_without_retry(
    tmp_path, recovery_corpora, recovery_endpoint, driver_config, status, error_type
):
    recovery_endpoint.failures_remaining = 10
    recovery_endpoint.status = status
    output = str(tmp_path / "results")
    with pytest.raises(ExceptionGroup) as error:
        execute_with_retries(recovery_corpora, output, driver_config, "http://fixture.test/v1", "unused", None)
    assert error.value.subgroup(error_type) is not None
    terminal = [row for row in ReadView(output).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert [(row["retry"], row["outcome"]) for row in terminal] == [(0, "infra_error")]


def test_submission_serializes_iris_timeout_and_separates_connection_secrets(monkeypatch, driver_config):
    requests = []

    def launch(request, **_kwargs):
        requests.append(request)
        return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

    monkeypatch.setattr(
        remote_client,
        "ControllerServiceClientSync",
        lambda **_kwargs: SimpleNamespace(launch_job=launch, close=lambda: None),
    )
    connections = {SOLVER_ENDPOINT: "https://solver.invalid/scoped/secret-token", SOLVER_API_KEY: "secret-solver-key"}
    with IrisClient.remote("http://controller.invalid") as client:
        job = submit_evaluation(
            client,
            ("s3://fixture/corpus",),
            "s3://fixture/results",
            driver_config,
            name="evaluation",
            resources=ResourceSpec(cpu=2, memory="4g", disk="8g"),
            timeout=73,
            secrets=connections,
        )
    (request,) = requests
    assert job.job_id.to_wire() == request.name
    assert request.timeout.milliseconds == 73_000
    assert request.container_profile == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert not request.resources.HasField("device")
    for name, value in connections.items():
        assert request.environment.secrets[name] == value
        assert value.encode() not in request.entrypoint.SerializeToString()
