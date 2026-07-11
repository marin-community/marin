# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import Callable, Mapping
from time import monotonic
from types import MappingProxyType

import httpx
import pytest
from marin.inference.logit_mixing import serve_logit_mixing_model
from marin.inference.types import OpenAIEndpoint, RunningModel

from tests.evals.openai_stub import serve_deterministic_openai_stub

REQUEST_TIMEOUT = 5
MIXED_MODEL = "mixed"
GENERATION_REQUEST = MappingProxyType(
    {
        "model": MIXED_MODEL,
        "prompt": "A",
        "max_tokens": 256,
        "max_new_tokens": 1,
        "temperature": 0,
        "top_p": 1.0,
        "stop": ["<eos>"],
        "seed": 1234,
    }
)


def test_logit_mixing_generation_selects_mixed_token() -> None:
    response = _mixed_completion(
        teacher={"A": {" T": -0.1, " C": -0.2, " S": -5.0}},
        student={"A": {" S": -0.1, " C": -0.2, " T": -5.0}},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice == {"text": " C", "index": 0, "logprobs": None, "finish_reason": "length"}


def test_logit_mixing_considers_tokens_returned_by_only_one_model() -> None:
    response = _mixed_completion(
        teacher={"A": {" T": 0, " C": -10}},
        student={"A": {" S": 0, " C": -20}},
        alpha=0.8,
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " T"


def test_logit_mixing_handles_disjoint_top_logprobs() -> None:
    response = _mixed_completion(
        teacher={"A": {" T": -0.1}},
        student={"A": {" S": -0.1}},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " S"


def test_logit_mixing_queries_models_concurrently() -> None:
    both_started = threading.Barrier(2)

    def wait_for_other_model() -> None:
        both_started.wait(timeout=REQUEST_TIMEOUT / 2)

    response = _mixed_completion(
        teacher={"A": {" B": -0.1}},
        student={"A": {" B": -0.1}},
        teacher_callbacks={"A": wait_for_other_model},
        student_callbacks={"A": wait_for_other_model},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " B"


def test_logit_mixing_detects_stop_across_generated_tokens() -> None:
    top_logprobs = {
        "A": {" B": -0.1, " X": -2.0},
        "A B": {" C": -0.1, " Y": -2.0},
    }
    response = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**GENERATION_REQUEST, "max_new_tokens": 2, "stop": [" B C"]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == ""
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_logit_mixing_alpha_zero_does_not_call_teacher() -> None:
    response = _mixed_completion(
        teacher={},
        student={"A": {" B": -0.1}},
        payload={**GENERATION_REQUEST, "stop": None},
        alpha=0,
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == " B"


def test_logit_mixing_samples_from_mixed_distribution_with_seed() -> None:
    top_logprobs = {"A": {" B": -0.1, " C": -0.2}}
    first = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**GENERATION_REQUEST, "temperature": 1},
    )
    second = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**GENERATION_REQUEST, "temperature": 1},
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice == second.json()["choices"][0]
    assert first_choice["text"] == " C"


def test_logit_mixing_seeded_sampling_breaks_probability_ties_by_token() -> None:
    top_logprobs = {"A": {" C": -0.1, " B": -0.1}}
    response = _mixed_completion(
        teacher=top_logprobs,
        student=top_logprobs,
        payload={**GENERATION_REQUEST, "temperature": 1},
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
def test_logit_mixing_rejects_unsupported_generation_fields(field: str, value: object) -> None:
    response = _mixed_completion(
        teacher={"A": {" B": -0.1}},
        student={"A": {" B": -0.1}},
        payload={**GENERATION_REQUEST, field: value},
    )

    assert response.status_code == 400


def _mixed_completion(
    *,
    teacher: Mapping[str, Mapping[str, float]],
    student: Mapping[str, Mapping[str, float]],
    payload: Mapping[str, object] = GENERATION_REQUEST,
    alpha: float = 0.5,
    teacher_callbacks: Mapping[str, Callable[[], None]] | None = None,
    student_callbacks: Mapping[str, Callable[[], None]] | None = None,
    clock: Callable[[], float] = monotonic,
) -> httpx.Response:
    with (
        serve_deterministic_openai_stub(
            model="teacher",
            completion_callbacks=teacher_callbacks,
            completion_top_logprobs=teacher,
        ) as teacher_stub,
        serve_deterministic_openai_stub(
            model="student",
            completion_callbacks=student_callbacks,
            completion_top_logprobs=student,
        ) as student_stub,
        serve_logit_mixing_model(
            teacher=RunningModel(endpoint=OpenAIEndpoint(teacher_stub.base_url, teacher_stub.model)),
            student=RunningModel(endpoint=OpenAIEndpoint(student_stub.base_url, student_stub.model)),
            model=MIXED_MODEL,
            tokenizer=None,
            alpha=alpha,
            request_timeout_seconds=REQUEST_TIMEOUT,
            top_logprobs=8,
            clock=clock,
        ) as running_model,
    ):
        return httpx.post(running_model.endpoint.url("completions"), json=dict(payload), timeout=REQUEST_TIMEOUT)
