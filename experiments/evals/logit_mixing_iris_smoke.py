# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import replace
from typing import Any

import requests
from fray.types import ResourceConfig
from marin.inference.logit_mixing import (
    LogitMixingBrokeredVllmConfig,
    LogitMixingWorkerConfig,
    start_iris_logit_mixing_brokered_vllm,
)
from marin.inference.vllm import BrokeredVllmSystemConfig, InferenceWorkerConfig, VllmProxyConfig
from rigging.log_setup import configure_logging

TEACHER_MODEL = "Qwen/Qwen3-0.6B-Base"
STUDENT_MODEL = "Qwen/Qwen3-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-0.6B"
REGIONS = ["us-east5", "us-central1"]
TPU_TYPE = "v6e-4"
TIMEOUT_SECONDS = 2400
ALPHA = 0.5
PROMPT = "The capital of France is"
TOP_LOGPROBS = 16


def brokered_config(model: str) -> BrokeredVllmSystemConfig:
    config = BrokeredVllmSystemConfig(
        model=model,
        tokenizer=TOKENIZER,
        request_lease_timeout_seconds=1000,
        proxy=VllmProxyConfig(request_timeout_seconds=1200, readiness_timeout_seconds=TIMEOUT_SECONDS),
        workers=InferenceWorkerConfig(count=1, max_in_flight_per_worker=1, request_timeout_seconds=900),
        worker_resources=ResourceConfig.with_tpu(
            TPU_TYPE,
            regions=REGIONS,
            cpu=64,
            ram="128g",
            disk="100g",
        ),
        worker_env_vars={
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
            "VLLM_TPU_SKIP_PRECOMPILE": "1",
        },
        priority=0,
    )
    return replace(
        config,
        server=replace(
            config.server,
            port=8000,
            timeout_seconds=TIMEOUT_SECONDS,
            max_model_len=1024,
            max_num_batched_tokens=512,
        ),
        broker_ready_timeout_seconds=TIMEOUT_SECONDS,
    )


def main() -> None:
    configure_logging()
    teacher = brokered_config(TEACHER_MODEL)
    student = brokered_config(STUDENT_MODEL)
    config = LogitMixingBrokeredVllmConfig(
        teacher=teacher,
        student=student,
        model="qwen3-logit-mix-smoke",
        alpha=ALPHA,
        tokenizer=TOKENIZER,
        request_lease_timeout_seconds=1000,
        proxy=VllmProxyConfig(request_timeout_seconds=1200, readiness_timeout_seconds=TIMEOUT_SECONDS),
        worker=LogitMixingWorkerConfig(max_in_flight=1, request_timeout_seconds=900, top_logprobs=TOP_LOGPROBS),
    )
    with start_iris_logit_mixing_brokered_vllm(config) as running_model:
        print(
            json.dumps(
                {
                    "alpha": ALPHA,
                    "event": "mixed_model_ready",
                    "base_url": running_model.endpoint.base_url,
                    "model": running_model.endpoint.model,
                    "student_model": running_model.student.endpoint.model,
                    "teacher_model": running_model.teacher.endpoint.model,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        response = requests.post(
            running_model.endpoint.url("completions"),
            json={
                "model": running_model.endpoint.model,
                "prompt": PROMPT,
                "max_tokens": 1,
                "logprobs": TOP_LOGPROBS,
            },
            timeout=1200,
        )
        response.raise_for_status()
        teacher_response = completion(running_model.teacher.endpoint.base_url, running_model.teacher.endpoint.model)
        student_response = completion(running_model.student.endpoint.base_url, running_model.student.endpoint.model)
        verification = verify_mixed_logits(
            teacher_response=teacher_response,
            student_response=student_response,
            mixed_response=response.json(),
        )
        print(
            json.dumps(
                {
                    "event": "mixed_completion_response",
                    "status_code": response.status_code,
                    "payload": response.json(),
                    "verification": verification,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def completion(base_url: str, model: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/completions",
        json={
            "model": model,
            "prompt": PROMPT,
            "max_tokens": 1,
            "echo": False,
            "logprobs": TOP_LOGPROBS,
            "temperature": 0,
        },
        timeout=1200,
    )
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def verify_mixed_logits(
    *,
    teacher_response: dict[str, Any],
    student_response: dict[str, Any],
    mixed_response: dict[str, Any],
) -> dict[str, Any]:
    teacher_top = completion_top_logprobs(teacher_response)
    student_top = completion_top_logprobs(student_response)
    mixed_top = completion_top_logprobs(mixed_response)
    expected_top = expected_mixed_logprobs(teacher_top, student_top)
    assert expected_top

    expected_token, expected_logprob = next(iter(expected_top.items()))
    mixed_choice = completion_choice(mixed_response)
    mixed_token = mixed_choice["text"]
    mixed_logprob = mixed_choice["logprobs"]["token_logprobs"][0]
    assert mixed_token == expected_token
    assert math.isclose(mixed_logprob, expected_logprob, abs_tol=1e-5)

    max_delta = 0.0
    for token, expected in expected_top.items():
        assert token in mixed_top
        delta = abs(mixed_top[token] - expected)
        assert delta < 1e-5
        max_delta = max(max_delta, delta)

    return {
        "alpha": ALPHA,
        "expected_token": expected_token,
        "expected_logprob": expected_logprob,
        "max_abs_logprob_delta": max_delta,
        "mixed_token": mixed_token,
        "teacher_token": completion_choice(teacher_response)["text"],
        "student_token": completion_choice(student_response)["text"],
        "shared_top_token_count": len(expected_top),
    }


def completion_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload["choices"]
    assert isinstance(choices, list)
    choice = choices[0]
    assert isinstance(choice, dict)
    return choice


def completion_top_logprobs(payload: dict[str, Any]) -> dict[str, float]:
    choice = completion_choice(payload)
    logprobs = choice["logprobs"]
    assert isinstance(logprobs, dict)
    top_logprobs = logprobs["top_logprobs"]
    assert isinstance(top_logprobs, list)
    return {str(token): float(logprob) for token, logprob in top_logprobs[-1].items()}


def expected_mixed_logprobs(teacher: dict[str, float], student: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for token in teacher.keys() & student.keys():
        scores[token] = ALPHA * teacher[token] + (1 - ALPHA) * student[token]
    normalizer = logsumexp(list(scores.values()))
    return dict(
        sorted(((token, score - normalizer) for token, score in scores.items()), key=lambda item: item[1], reverse=True)
    )


def logsumexp(values: list[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


if __name__ == "__main__":
    main()
