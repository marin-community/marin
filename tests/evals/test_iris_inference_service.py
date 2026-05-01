# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import pytest
import requests
from fray.local_backend import LocalClient
from marin.evaluation.lm_eval import LmEvalRun, build_lm_eval_model_args
from marin.inference.iris_service import (
    MARIN_REQUEST_ID_HEADER,
    BrokerRequestStatus,
    IrisInferenceBroker,
    IrisInferenceLauncher,
    IrisInferenceLauncherConfig,
    OpenAIEndpointKind,
    OpenAIHttpRequestEnvelope,
    OpenAIHttpResponseEnvelope,
)
from marin.inference.types import ModelDeployment, RunningModel

from tests.evals.openai_stub import (
    DeterministicOpenAIStub,
    assert_chat_generation_contract,
    assert_completions_scoring_contract,
    serve_deterministic_openai_stub,
)


@dataclass
class ManualClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, amount: float) -> None:
        self.now += amount


def _completion_request(request_id: str = "request-1") -> OpenAIHttpRequestEnvelope:
    return OpenAIHttpRequestEnvelope(
        request_id=request_id,
        endpoint=OpenAIEndpointKind.COMPLETIONS,
        payload_json='{"model":"gpt2","prompt":"A B","echo":true,"logprobs":1}',
    )


def _response(text: str = "ok") -> OpenAIHttpResponseEnvelope:
    return OpenAIHttpResponseEnvelope(status_code=200, payload_json=f'{{"result":"{text}"}}')


def test_broker_submit_lease_complete_wait_returns_exact_response() -> None:
    clock = ManualClock()
    broker = IrisInferenceBroker(lease_timeout=10.0, now=clock)
    request = _completion_request()
    response = _response()

    assert broker.submit(request) is True
    lease = broker.lease("worker-1", wait_timeout=0).lease
    assert lease is not None
    assert lease.request == request
    assert broker.status(request.request_id) == BrokerRequestStatus.LEASED

    assert broker.complete(request.request_id, response) is True
    assert broker.wait(request.request_id, timeout=0) == response
    assert broker.poll(request.request_id) == response
    assert broker.status(request.request_id) == BrokerRequestStatus.SUCCEEDED


def test_broker_releases_expired_work_again() -> None:
    clock = ManualClock()
    broker = IrisInferenceBroker(lease_timeout=5.0, now=clock)
    request = _completion_request()

    broker.submit(request)
    first = broker.lease("worker-1", wait_timeout=0).lease
    assert first is not None
    assert broker.lease("worker-2", wait_timeout=0).lease is None

    clock.advance(5.1)
    second = broker.lease("worker-2", wait_timeout=0).lease
    assert second is not None
    assert second.request == request
    assert second.lease_id != first.lease_id
    assert second.worker_id == "worker-2"


def test_broker_duplicate_terminal_result_keeps_first_response() -> None:
    clock = ManualClock()
    broker = IrisInferenceBroker(lease_timeout=10.0, now=clock)
    request = _completion_request()
    first = _response("first")
    duplicate = _response("duplicate")

    broker.submit(request)
    assert broker.lease("worker-1", wait_timeout=0).lease is not None

    assert broker.complete(request.request_id, first) is True
    assert broker.fail(request.request_id, duplicate) is False
    assert broker.complete(request.request_id, duplicate) is False
    assert broker.wait(request.request_id, timeout=0) == first


def test_proxy_worker_end_to_end_routes_completions_and_chat_to_engine() -> None:
    client = LocalClient()
    try:
        with serve_deterministic_openai_stub(model="gpt2") as stub:
            with _launch_local_service(client, stub) as running_model:
                assert_completions_scoring_contract(running_model.endpoint.base_url, stub.model)
                assert_chat_generation_contract(running_model.endpoint.base_url, stub.model)

            assert len(stub.requests_for("/v1/completions")) == 1
            assert len(stub.requests_for("/v1/chat/completions")) == 1
    finally:
        client.shutdown(wait=True)


def test_proxy_rejects_unsafe_request_id_header() -> None:
    client = LocalClient()
    try:
        with serve_deterministic_openai_stub(model="gpt2") as stub:
            with _launch_local_service(client, stub) as running_model:
                response = requests.post(
                    running_model.endpoint.url("completions"),
                    json={"model": stub.model, "prompt": "A"},
                    headers={MARIN_REQUEST_ID_HEADER: "unsafe/request/id"},
                    timeout=5,
                )

            assert response.status_code == 400
            assert MARIN_REQUEST_ID_HEADER not in response.headers
            assert stub.requests_for("/v1/completions") == []
    finally:
        client.shutdown(wait=True)


def test_real_lm_eval_local_completions_scoring_through_proxy_worker(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tiktoken")
    pytest.importorskip("lm_eval")

    from lm_eval.api.instance import Instance
    from lm_eval.api.task import Task, TaskConfig
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.loggers import EvaluationTracker

    class TinyScoringTask(Task):
        OUTPUT_TYPE = "loglikelihood"

        def __init__(self) -> None:
            super().__init__()
            self._config = TaskConfig(
                task="marin_tiny_scoring",
                output_type="loglikelihood",
                num_fewshot=0,
                repeats=1,
            )
            self.task_name = "marin_tiny_scoring"

        def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
            self.dataset = {"test": [{"ctx": "A", "target": " B"}]}

        def has_training_docs(self) -> bool:
            return False

        def has_validation_docs(self) -> bool:
            return False

        def has_test_docs(self) -> bool:
            return True

        def test_docs(self) -> list[dict[str, str]]:
            return self.dataset["test"]

        def doc_to_text(self, doc: dict[str, str]) -> str:
            return doc["ctx"]

        def doc_to_target(self, doc: dict[str, str]) -> str:
            return doc["target"]

        def construct_requests(self, doc: dict[str, str], ctx: str, **kwargs: object) -> Instance:
            return Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, self.doc_to_target(doc)),
                idx=0,
                metadata=kwargs["metadata"],
            )

        def process_results(self, doc: dict[str, str], results: list[tuple[float, bool]]) -> dict[str, float]:
            loglikelihood, is_greedy = results[0]
            return {"loglikelihood": loglikelihood, "greedy": float(is_greedy)}

        def aggregation(self) -> dict[str, object]:
            return {"loglikelihood": mean, "greedy": mean}

        def higher_is_better(self) -> dict[str, bool]:
            return {"loglikelihood": True, "greedy": True}

    client = LocalClient()
    try:
        with serve_deterministic_openai_stub(model="gpt2") as stub:
            with _launch_local_service(client, stub) as running_model:
                run = LmEvalRun(
                    tasks=["marin_tiny_scoring"],
                    output_path=str(tmp_path / "lm_eval_results"),
                    extra_model_args={
                        "tokenizer_backend": "tiktoken",
                        "tokenized_requests": False,
                        "max_retries": 1,
                        "timeout": 5,
                    },
                )
                evaluation_tracker = EvaluationTracker(output_path=run.output_path)
                results = simple_evaluate(
                    model="local-completions",
                    tasks=[TinyScoringTask()],
                    model_args=build_lm_eval_model_args(running_model, run),
                    batch_size=1,
                    bootstrap_iters=0,
                    evaluation_tracker=evaluation_tracker,
                    log_samples=True,
                    random_seed=0,
                    numpy_random_seed=0,
                    torch_random_seed=0,
                    fewshot_random_seed=0,
                )

            assert results is not None
            samples = results.pop("samples")
            evaluation_tracker.save_results_aggregated(results=results, samples=samples)
            for task_name in results["configs"].keys():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

            completion_requests = stub.requests_for("/v1/completions")
            assert len(completion_requests) == 1
            assert completion_requests[0].payload == {
                "model": "gpt2",
                "prompt": "A B",
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": 1234,
                "echo": True,
            }
            assert results["results"]["marin_tiny_scoring"]["loglikelihood,none"] == pytest.approx(-0.1)
            assert results["results"]["marin_tiny_scoring"]["greedy,none"] == 1.0
    finally:
        client.shutdown(wait=True)


def _launch_local_service(client: LocalClient, stub: DeterministicOpenAIStub) -> AbstractContextManager[RunningModel]:
    deployment = ModelDeployment(model_name=stub.model, model_path="deterministic-openai-stub")
    launcher = IrisInferenceLauncher(
        client=client,
        config=IrisInferenceLauncherConfig(
            engine_base_url=stub.base_url,
            request_timeout=5.0,
            worker_lease_wait_timeout=0.05,
        ),
    )
    return launcher.launch(deployment)
