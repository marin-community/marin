# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic

import httpx
import pytest
from marin.inference.broker import InferenceBroker
from marin.inference.logit_mixing import LogitMixingInferenceWorker, run_logit_mixing_worker
from marin.inference.proxy import serve_inference_proxy
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.timing import ExponentialBackoff

from tests.evals.openai_stub import DeterministicOpenAIStub, serve_deterministic_openai_stub

REQUEST_TIMEOUT = 5


def test_logit_mixing_worker_returns_mixed_completion() -> None:
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_top_logprobs={"A": {" T": -0.1, " S": -2.0}},
        ) as teacher,
        serve_deterministic_openai_stub(
            model="student",
            completion_top_logprobs={"A": {" T": -2.0, " S": -0.1}},
        ) as student,
        _serve_logit_mixing_proxy(teacher=teacher, student=student, alpha=0.75) as mixed,
    ):
        response = httpx.post(
            f"{mixed.endpoint.base_url}/completions",
            json={"model": mixed.endpoint.model, "prompt": "A", "max_tokens": 1, "logprobs": 2},
            timeout=REQUEST_TIMEOUT,
        )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["text"] == " T"
    assert choice["logprobs"]["tokens"] == [" T"]
    assert choice["logprobs"]["top_logprobs"][0] == pytest.approx({" T": -0.326956432, " S": -1.276956432})


def test_logit_mixing_worker_omits_unrequested_logprobs() -> None:
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_top_logprobs={"A": {" T": -0.1, " S": -2.0}},
        ) as teacher,
        serve_deterministic_openai_stub(
            model="student",
            completion_top_logprobs={"A": {" T": -2.0, " S": -0.1}},
        ) as student,
        _serve_logit_mixing_proxy(teacher=teacher, student=student, alpha=0.75) as mixed,
    ):
        response = httpx.post(
            f"{mixed.endpoint.base_url}/completions",
            json={"model": mixed.endpoint.model, "prompt": "A", "max_tokens": 1},
            timeout=REQUEST_TIMEOUT,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["logprobs"] is None


def test_logit_mixing_worker_generates_autoregressive_steps() -> None:
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_top_logprobs={
                "A": {" B": -0.1, " X": -2.0},
                "A B": {" C": -0.1, " Y": -2.0},
            },
        ) as teacher,
        serve_deterministic_openai_stub(
            model="student",
            completion_top_logprobs={
                "A": {" B": -0.2, " X": -2.0},
                "A B": {" C": -0.2, " Y": -2.0},
            },
        ) as student,
        _serve_logit_mixing_proxy(teacher=teacher, student=student, alpha=0.5) as mixed,
    ):
        response = httpx.post(
            f"{mixed.endpoint.base_url}/completions",
            json={"model": mixed.endpoint.model, "prompt": "A", "max_tokens": 2, "logprobs": 2},
            timeout=REQUEST_TIMEOUT,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " B C"


def test_logit_mixing_worker_ignores_inactive_teacher_stop() -> None:
    top_logprobs = {
        "A": {" B": -0.1, " X": -2.0},
        "A B": {" C": -0.1, " Y": -2.0},
    }
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_top_logprobs=top_logprobs,
            completion_finish_reasons={"A": "stop"},
        ) as teacher,
        serve_deterministic_openai_stub(model="student", completion_top_logprobs=top_logprobs) as student,
        _serve_logit_mixing_proxy(teacher=teacher, student=student, alpha=0) as mixed,
    ):
        response = httpx.post(
            f"{mixed.endpoint.base_url}/completions",
            json={"model": mixed.endpoint.model, "prompt": "A", "max_tokens": 2},
            timeout=REQUEST_TIMEOUT,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " B C"


def test_logit_mixing_worker_enforces_end_to_end_timeout() -> None:
    times = iter([0.0, 0.0, 6.0])
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_top_logprobs={"A": {" B": -0.1}},
        ) as teacher,
        serve_deterministic_openai_stub(
            model="student",
            completion_top_logprobs={"A": {" B": -0.1}},
        ) as student,
        _serve_logit_mixing_proxy(
            teacher=teacher,
            student=student,
            alpha=0.5,
            request_timeout_seconds=REQUEST_TIMEOUT,
            clock=lambda: next(times),
        ) as mixed,
    ):
        response = httpx.post(
            f"{mixed.endpoint.base_url}/completions",
            json={"model": mixed.endpoint.model, "prompt": "A", "max_tokens": 2},
            timeout=REQUEST_TIMEOUT,
        )

    assert response.status_code == 504


def test_logit_mixing_proxy_models_returns_mixed_model() -> None:
    with (
        serve_deterministic_openai_stub(model="teacher", completion_top_logprobs={}) as teacher,
        serve_deterministic_openai_stub(model="student", completion_top_logprobs={}) as student,
        _serve_logit_mixing_proxy(teacher=teacher, student=student, alpha=0.5) as mixed,
    ):
        response = httpx.get(f"{mixed.endpoint.base_url}/models", timeout=REQUEST_TIMEOUT)

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "mixed"


@contextmanager
def _serve_logit_mixing_proxy(
    *,
    teacher: DeterministicOpenAIStub,
    student: DeterministicOpenAIStub,
    alpha: float,
    request_timeout_seconds: float = REQUEST_TIMEOUT,
    clock: Callable[[], float] = monotonic,
) -> Iterator[RunningModel]:
    broker = InferenceBroker(request_lease_timeout_seconds=30)
    teacher_model = RunningModel(endpoint=OpenAIEndpoint(base_url=teacher.base_url, model=teacher.model))
    student_model = RunningModel(endpoint=OpenAIEndpoint(base_url=student.base_url, model=student.model))
    worker = LogitMixingInferenceWorker(
        broker=broker,
        teacher=teacher_model,
        student=student_model,
        model="mixed",
        alpha=alpha,
        request_timeout_seconds=request_timeout_seconds,
        top_logprobs=8,
        clock=clock,
    )
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
        run_logit_mixing_worker(
            worker,
            max_in_flight=2,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
        ),
    ):
        yield running_model
