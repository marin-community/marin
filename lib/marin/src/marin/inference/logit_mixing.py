# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import itertools
import logging
import math
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import httpx
from rigging.timing import ExponentialBackoff

from marin.inference.broker import InferenceBroker
from marin.inference.proxy import serve_inference_proxy
from marin.inference.types import (
    InferenceRequest,
    InferenceRequestProvider,
    InferenceResponse,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    OpenAIEndpoint,
    RunningModel,
    pack_json_payload,
    unpack_json_payload,
)
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT,
    DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT,
    BrokeredVllmSystemConfig,
    VllmProxyConfig,
    _wait_for_brokered_vllm_ready,
    start_iris_brokered_vllm,
)
from marin.inference.worker import run_brokered_inference_loop

logger = logging.getLogger(__name__)

DEFAULT_LOGIT_MIXING_TOP_LOGPROBS = 64


@dataclass(frozen=True)
class LogitMixingWorkerConfig:
    """Configure mixed-completion request processing."""

    max_in_flight: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER
    request_timeout_seconds: float = DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT.total_seconds()
    top_logprobs: int = DEFAULT_LOGIT_MIXING_TOP_LOGPROBS


@dataclass(frozen=True)
class LogitMixingBrokeredVllmConfig:
    """Configure teacher, student, and mixed brokered endpoints."""

    teacher: BrokeredVllmSystemConfig
    student: BrokeredVllmSystemConfig
    model: str = "logit-mix"
    alpha: float = 0.5
    tokenizer: str | None = None
    request_lease_timeout_seconds: float = DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT.total_seconds()
    proxy: VllmProxyConfig = field(default_factory=VllmProxyConfig)
    worker: LogitMixingWorkerConfig = field(default_factory=LogitMixingWorkerConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}.")
        if self.worker.max_in_flight < 1:
            raise ValueError("worker.max_in_flight must be at least 1.")
        if self.worker.top_logprobs < 1:
            raise ValueError("worker.top_logprobs must be at least 1.")
        if not 0 < self.worker.request_timeout_seconds < self.request_lease_timeout_seconds:
            raise ValueError(
                "Logit mixing timeouts must satisfy "
                "0 < worker.request_timeout_seconds < request_lease_timeout_seconds."
            )
        if self.request_lease_timeout_seconds >= self.proxy.request_timeout_seconds:
            raise ValueError("request_lease_timeout_seconds must be lower than proxy.request_timeout_seconds.")


@dataclass(frozen=True, kw_only=True)
class RunningLogitMixingModel(RunningModel):
    """A mixed endpoint and its running component models."""

    teacher: RunningModel
    student: RunningModel


class LogitMixingInferenceWorker:
    """Generate completions from weighted teacher and student logprobs."""

    def __init__(
        self,
        *,
        broker: InferenceRequestProvider,
        teacher: RunningModel,
        student: RunningModel,
        model: str,
        alpha: float,
        request_timeout_seconds: float,
        top_logprobs: int,
    ) -> None:
        self._broker = broker
        self._teacher = teacher
        self._student = student
        self._model = model
        self._alpha = alpha
        self._request_timeout_seconds = request_timeout_seconds
        self._top_logprobs = top_logprobs

    def run_forever(
        self,
        *,
        max_in_flight: int,
        stop_event: threading.Event | None = None,
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        run_brokered_inference_loop(
            broker=self._broker,
            handle_request=self._forward_one,
            max_in_flight=max_in_flight,
            request_timeout_seconds=self._request_timeout_seconds,
            stop_event=stop_event,
            backoff=backoff,
        )

    def _forward_one(self, client: httpx.Client, leased_request: LeasedInferenceRequest) -> LeasedInferenceResponse:
        request = leased_request.request
        try:
            if request.method.upper() == "GET" and request.path == "/v1/models":
                response = _models_response(request, self._model)
            elif request.method.upper() == "POST" and request.path == "/v1/completions":
                response = self._mixed_completion_response(client, request)
            else:
                response = _error_response(
                    request,
                    405,
                    "logit mixing inference only supports GET /v1/models and POST /v1/completions",
                )
        except httpx.TimeoutException as exc:
            response = _error_response(
                request,
                504,
                "logit mixing request timed out",
                detail=repr(exc),
            )
        except Exception as exc:
            response = _error_response(
                request,
                502,
                "unexpected logit mixing worker failure",
                detail=repr(exc),
                exc_info=True,
            )
        return LeasedInferenceResponse(lease_id=leased_request.lease_id, response=response)

    def _mixed_completion_response(self, client: httpx.Client, request: InferenceRequest) -> InferenceResponse:
        payload = unpack_json_payload(request.payload)
        if not isinstance(payload.get("prompt"), str):
            return _error_response(request, 400, "logit mixing completions require a string prompt")
        if payload.get("echo", False):
            return _error_response(request, 400, "logit mixing completions do not support echo=true")
        if int(payload.get("n", 1)) != 1:
            return _error_response(request, 400, "logit mixing completions require n=1")
        requested_logprobs = payload.get("logprobs")
        if requested_logprobs is not None and (
            not isinstance(requested_logprobs, int) or isinstance(requested_logprobs, bool) or requested_logprobs < 0
        ):
            return _error_response(request, 400, "logprobs must be a non-negative integer or null")
        max_tokens = int(payload.get("max_tokens", 16))
        if max_tokens < 0:
            return _error_response(request, 400, "max_tokens must be non-negative")

        generated_tokens: list[str] = []
        token_logprobs: list[float] = []
        top_logprobs: list[dict[str, float]] = []
        current_text = str(payload["prompt"])
        finish_reason = "length"
        deadline = monotonic() + self._request_timeout_seconds

        for _ in range(max_tokens):
            step_payload = _step_payload(payload, current_text, self._top_logprobs)
            teacher = self._post_completion(client, self._teacher.endpoint, step_payload, deadline=deadline)
            student = self._post_completion(client, self._student.endpoint, step_payload, deadline=deadline)
            upstream_error = _upstream_error(request, teacher, student)
            if upstream_error is not None:
                return upstream_error

            teacher_top = _completion_top_logprobs(teacher)
            student_top = _completion_top_logprobs(student)
            mixed = _mix_top_logprobs(teacher_top, student_top, alpha=self._alpha)
            if not mixed:
                return _error_response(request, 502, "upstream completions did not return top logprobs")
            token, logprob = next(iter(mixed.items()))
            generated_tokens.append(token)
            token_logprobs.append(logprob)
            top_logprobs.append(mixed)
            current_text += token

            mixed_finish_reason = _mixed_finish_reason(teacher, student, token=token, alpha=self._alpha)
            if mixed_finish_reason is not None:
                finish_reason = mixed_finish_reason
                break

        text = "".join(generated_tokens)
        response_logprobs = None
        if requested_logprobs is not None:
            response_logprobs = {
                "tokens": generated_tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": [
                    dict(itertools.islice(logprobs.items(), requested_logprobs)) for logprobs in top_logprobs
                ],
                "text_offset": _text_offsets(str(payload["prompt"]), generated_tokens),
            }
        response_payload = {
            "id": "cmpl-logit-mix",
            "object": "text_completion",
            "model": self._model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": response_logprobs,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return InferenceResponse(
            request_id=request.request_id,
            status_code=200,
            payload=pack_json_payload(response_payload),
        )

    def _post_completion(
        self,
        client: httpx.Client,
        endpoint: OpenAIEndpoint,
        payload: Mapping[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        upstream_payload = dict(payload)
        upstream_payload["model"] = endpoint.model
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise httpx.TimeoutException("logit mixing request deadline exceeded")
        response = client.post(endpoint.url("completions"), json=upstream_payload, timeout=remaining)
        try:
            parsed = response.json()
        except ValueError:
            return {
                "status_code": response.status_code,
                "error": {
                    "message": "upstream endpoint returned a non-JSON response",
                    "body": response.content[:1000].decode(errors="replace"),
                },
            }
        if not isinstance(parsed, dict):
            return {
                "status_code": response.status_code,
                "error": {"message": "upstream endpoint returned a non-object JSON response"},
            }
        parsed["status_code"] = response.status_code
        return parsed


@contextlib.contextmanager
def run_logit_mixing_worker(
    worker: LogitMixingInferenceWorker,
    *,
    max_in_flight: int,
    backoff: ExponentialBackoff | None = None,
) -> Iterator[None]:
    """Run a logit-mixing worker in a background thread."""

    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: worker.run_forever(stop_event=stop_event, max_in_flight=max_in_flight, backoff=backoff),
        name="logit-mixing-worker",
    )
    logger.info("Starting LogitMixingInferenceWorker thread max_in_flight=%d", max_in_flight)
    thread.start()
    try:
        yield
    finally:
        logger.info("Stopping LogitMixingInferenceWorker thread")
        stop_event.set()
        thread.join()


@contextlib.contextmanager
def start_iris_logit_mixing_brokered_vllm(config: LogitMixingBrokeredVllmConfig) -> Iterator[RunningLogitMixingModel]:
    """Start teacher, student, and mixed endpoints inside an Iris job."""

    with (
        start_iris_brokered_vllm(config.teacher) as teacher,
        start_iris_brokered_vllm(config.student) as student,
    ):
        broker = InferenceBroker(request_lease_timeout_seconds=config.request_lease_timeout_seconds)
        worker = LogitMixingInferenceWorker(
            broker=broker,
            teacher=teacher,
            student=student,
            model=config.model,
            alpha=config.alpha,
            request_timeout_seconds=config.worker.request_timeout_seconds,
            top_logprobs=config.worker.top_logprobs,
        )
        with (
            _start_logit_mixing_proxy(config, broker) as running_model,
            run_logit_mixing_worker(worker, max_in_flight=config.worker.max_in_flight),
        ):
            _wait_for_brokered_vllm_ready(running_model, timeout_seconds=config.proxy.readiness_timeout_seconds)
            yield RunningLogitMixingModel(
                endpoint=running_model.endpoint,
                tokenizer=config.tokenizer or student.tokenizer or teacher.tokenizer,
                teacher=teacher,
                student=student,
            )


@contextlib.contextmanager
def _start_logit_mixing_proxy(
    config: LogitMixingBrokeredVllmConfig,
    broker: InferenceBroker,
) -> Iterator[RunningModel]:
    proxy_config = config.proxy
    with serve_inference_proxy(
        broker=broker,
        model=config.model,
        host=proxy_config.host,
        port=proxy_config.port,
        request_timeout_seconds=proxy_config.request_timeout_seconds,
        readiness_timeout_seconds=proxy_config.readiness_timeout_seconds,
        max_pending_requests=proxy_config.max_pending_requests,
        response_fetch_batch_size=proxy_config.response_fetch_batch_size,
        server_start_timeout_seconds=proxy_config.server_start_timeout_seconds,
    ) as running_model:
        yield running_model


def _models_response(request: InferenceRequest, model: str) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        status_code=200,
        payload=pack_json_payload({"object": "list", "data": [{"id": model, "object": "model"}]}),
    )


def _step_payload(payload: Mapping[str, Any], prompt: str, top_logprobs: int) -> dict[str, Any]:
    step_payload = dict(payload)
    step_payload["prompt"] = prompt
    step_payload["max_tokens"] = 1
    step_payload["echo"] = False
    step_payload["logprobs"] = max(int(step_payload.get("logprobs") or 0), top_logprobs)
    step_payload["temperature"] = 0
    return step_payload


def _completion_choice(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0]


def _completion_top_logprobs(payload: Mapping[str, Any]) -> Mapping[str, float]:
    choice = _completion_choice(payload)
    if choice is None:
        return {}
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return {}
    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(top_logprobs, list) or not top_logprobs or not isinstance(top_logprobs[-1], dict):
        return {}
    return {str(token): float(logprob) for token, logprob in top_logprobs[-1].items()}


def _completion_finish_reason(payload: Mapping[str, Any]) -> str | None:
    choice = _completion_choice(payload)
    if choice is None:
        return None
    finish_reason = choice.get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def _mixed_finish_reason(
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    *,
    token: str,
    alpha: float,
) -> str | None:
    teacher_reason = _selected_token_finish_reason(teacher, token)
    student_reason = _selected_token_finish_reason(student, token)
    if alpha == 1:
        return teacher_reason
    if alpha == 0:
        return student_reason
    return teacher_reason or student_reason


def _selected_token_finish_reason(payload: Mapping[str, Any], token: str) -> str | None:
    choice = _completion_choice(payload)
    reason = _completion_finish_reason(payload)
    if choice is None or choice.get("text") != token or reason in {None, "length"}:
        return None
    return reason


def _mix_top_logprobs(
    teacher: Mapping[str, float],
    student: Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    scores = {
        token: _mixed_logit_score(teacher.get(token), student.get(token), alpha=alpha)
        for token in teacher.keys() | student.keys()
    }
    finite_scores = {token: score for token, score in scores.items() if math.isfinite(score)}
    if not finite_scores:
        return {}
    normalizer = _logsumexp(finite_scores.values())
    return dict(
        sorted(
            ((token, score - normalizer) for token, score in finite_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def _mixed_logit_score(teacher: float | None, student: float | None, *, alpha: float) -> float:
    if alpha == 1:
        return -math.inf if teacher is None else teacher
    if alpha == 0:
        return -math.inf if student is None else student
    if teacher is None or student is None:
        return -math.inf
    return alpha * teacher + (1 - alpha) * student


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _text_offsets(prompt: str, tokens: list[str]) -> list[int]:
    offsets: list[int] = []
    offset = len(prompt)
    for token in tokens:
        offsets.append(offset)
        offset += len(token)
    return offsets


def _upstream_error(
    request: InferenceRequest,
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
) -> InferenceResponse | None:
    for name, payload in [("teacher", teacher), ("student", student)]:
        status_code = int(payload.get("status_code", 200))
        if status_code >= 400:
            return _error_response(
                request,
                502,
                f"{name} upstream returned an error",
                detail=str(payload.get("error", payload))[:1000],
            )
    return None


def _error_response(
    request: InferenceRequest,
    status_code: int,
    message: str,
    *,
    detail: str | None = None,
    exc_info: bool = False,
) -> InferenceResponse:
    logger.warning(
        "LogitMixingInferenceWorker returning error request_id=%s method=%s path=%s status_code=%d error=%s "
        "detail=%s",
        request.request_id,
        request.method,
        request.path,
        status_code,
        message,
        detail or "-",
        exc_info=exc_info,
    )
    error: dict[str, Any] = {"message": message}
    if detail is not None:
        error["detail"] = detail
    return InferenceResponse(
        request_id=request.request_id,
        status_code=status_code,
        payload=pack_json_payload({"error": error}),
    )
