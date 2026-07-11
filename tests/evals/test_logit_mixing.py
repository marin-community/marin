# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping
from time import monotonic
from types import MappingProxyType

import httpx
import pytest
from marin.inference.broker import InferenceBroker
from marin.inference.logit_mixing import LogitMixingInferenceHandler
from marin.inference.proxy import serve_inference_proxy
from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.worker import InferenceWorker, run_inference_worker
from rigging.timing import ExponentialBackoff

from tests.evals.openai_stub import serve_deterministic_openai_stub

REQUEST_TIMEOUT = 5
EVALCHEMY_REQUEST = MappingProxyType(
    {
        "model": "mixed",
        "prompt": "A",
        "max_tokens": 256,
        "max_new_tokens": 1,
        "temperature": 0,
        "top_p": 1.0,
        "stop": ["<eos>"],
        "seed": 1234,
    }
)


def test_logit_mixing_evalchemy_request_selects_mixed_token() -> None:
    response = _mixed_completion(
        teacher={"A": {" T": -0.1, " C": -0.2, " S": -5.0}},
        student={"A": {" S": -0.1, " C": -0.2, " T": -5.0}},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice == {"text": " C", "index": 0, "logprobs": None, "finish_reason": "length"}


def test_logit_mixing_detects_stop_across_generated_tokens() -> None:
    top_logprobs = {
        "A": {" B": -0.1, " X": -2.0},
        "A B": {" C": -0.1, " Y": -2.0},
    }
    response = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**EVALCHEMY_REQUEST, "max_new_tokens": 2, "stop": [" B C"]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == ""
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_logit_mixing_alpha_zero_does_not_call_teacher() -> None:
    response = _mixed_completion(
        teacher={},
        student={"A": {" B": -0.1}},
        alpha=0,
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " B"


def test_logit_mixing_samples_from_mixed_distribution_with_evalchemy_seed() -> None:
    top_logprobs = {"A": {" B": -0.1, " C": -0.2}}
    response = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**EVALCHEMY_REQUEST, "temperature": 1},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " C"


def test_logit_mixing_enforces_end_to_end_timeout() -> None:
    now = [0.0]

    def expire_deadline() -> None:
        now[0] = REQUEST_TIMEOUT + 1

    response = _mixed_completion(
        teacher={"A": {" B": -0.1}},
        student={"A": {" B": -0.1}},
        teacher_callbacks={"A": expire_deadline},
        clock=lambda: now[0],
    )

    assert response.status_code == 504


@pytest.mark.parametrize(("field", "value"), [("echo", True), ("logprobs", 1), ("stream", True)])
def test_logit_mixing_rejects_non_evalchemy_generation_fields(field: str, value: object) -> None:
    response = _mixed_completion(
        teacher={"A": {" B": -0.1}},
        student={"A": {" B": -0.1}},
        payload={**EVALCHEMY_REQUEST, field: value},
    )

    assert response.status_code == 400


def _mixed_completion(
    *,
    teacher: Mapping[str, Mapping[str, float]],
    student: Mapping[str, Mapping[str, float]],
    payload: Mapping[str, object] = EVALCHEMY_REQUEST,
    alpha: float = 0.5,
    teacher_callbacks: Mapping[str, Callable[[], None]] | None = None,
    clock: Callable[[], float] = monotonic,
) -> httpx.Response:
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_callbacks=teacher_callbacks,
            completion_top_logprobs=teacher,
        ) as teacher_stub,
        serve_deterministic_openai_stub(model="student", completion_top_logprobs=student) as student_stub,
    ):
        broker = InferenceBroker(request_lease_timeout_seconds=30)
        handler = LogitMixingInferenceHandler(
            teacher=RunningModel(endpoint=OpenAIEndpoint(teacher_stub.base_url, teacher_stub.model)),
            student=RunningModel(endpoint=OpenAIEndpoint(student_stub.base_url, student_stub.model)),
            model="mixed",
            alpha=alpha,
            request_timeout_seconds=REQUEST_TIMEOUT,
            top_logprobs=8,
            clock=clock,
        )
        worker = InferenceWorker(broker=broker, handler=handler, request_timeout_seconds=REQUEST_TIMEOUT)
        with (
            serve_inference_proxy(
                broker=broker,
                model="mixed",
                request_timeout_seconds=REQUEST_TIMEOUT,
                readiness_timeout_seconds=REQUEST_TIMEOUT,
                max_pending_requests=8,
                response_fetch_batch_size=8,
                server_start_timeout_seconds=REQUEST_TIMEOUT,
            ) as running_model,
            run_inference_worker(
                worker,
                max_in_flight=2,
                backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
            ),
        ):
            return httpx.post(running_model.endpoint.url("completions"), json=dict(payload), timeout=REQUEST_TIMEOUT)
