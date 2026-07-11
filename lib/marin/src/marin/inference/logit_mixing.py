# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from time import monotonic

import httpx
from levanter.inference.openai_protocol import CompletionRequest
from pydantic import ValidationError

from marin.inference.broker import InferenceBroker
from marin.inference.types import (
    InferenceRequest,
    InferenceResponse,
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
    start_brokered_inference_proxy,
    start_iris_brokered_vllm,
)
from marin.inference.worker import InferenceWorker, inference_error_response, run_inference_worker

DEFAULT_LOGIT_MIXING_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class LogitMixingBrokeredVllmConfig:
    """Configure teacher, student, and mixed brokered endpoints."""

    teacher: BrokeredVllmSystemConfig
    student: BrokeredVllmSystemConfig
    alpha: float
    tokenizer: str
    model: str = "logit-mix"
    max_in_flight: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER
    request_timeout_seconds: float = DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT.total_seconds()
    top_logprobs: int = DEFAULT_LOGIT_MIXING_TOP_LOGPROBS
    request_lease_timeout_seconds: float = DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT.total_seconds()
    proxy: VllmProxyConfig = field(default_factory=VllmProxyConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}.")
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1.")
        if not 1 <= self.top_logprobs <= DEFAULT_LOGIT_MIXING_TOP_LOGPROBS:
            raise ValueError(f"top_logprobs must be in [1, {DEFAULT_LOGIT_MIXING_TOP_LOGPROBS}].")
        if not 0 < self.request_timeout_seconds < self.request_lease_timeout_seconds:
            raise ValueError(
                "Logit mixing timeouts must satisfy 0 < request_timeout_seconds < request_lease_timeout_seconds."
            )
        if self.request_lease_timeout_seconds >= self.proxy.request_timeout_seconds:
            raise ValueError("request_lease_timeout_seconds must be lower than proxy.request_timeout_seconds.")
        component_tokenizers = {tokenizer for tokenizer in (self.teacher.tokenizer, self.student.tokenizer) if tokenizer}
        if component_tokenizers - {self.tokenizer}:
            raise ValueError("tokenizer must match the teacher and student tokenizer.")


@dataclass(frozen=True)
class _GenerationRequest:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class _UpstreamStep:
    top_logprobs: Mapping[str, float]


class LogitMixingInferenceHandler:
    """Handle text generation by mixing two model distributions."""

    def __init__(
        self,
        *,
        teacher: RunningModel,
        student: RunningModel,
        model: str,
        alpha: float,
        request_timeout_seconds: float,
        top_logprobs: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._model = model
        self._alpha = alpha
        self._request_timeout_seconds = request_timeout_seconds
        self._top_logprobs = top_logprobs
        self._clock = clock

    def __call__(self, client: httpx.Client, request: InferenceRequest) -> InferenceResponse:
        if request.method.upper() == "GET" and request.path == "/v1/models":
            return _models_response(request, self._model)
        if request.method.upper() != "POST" or request.path != "/v1/completions":
            return inference_error_response(
                request,
                405,
                "logit mixing supports GET /v1/models and POST /v1/completions",
            )

        try:
            generation = _generation_request(request, self._model)
        except ValueError as exc:
            return inference_error_response(request, 400, str(exc))
        text, finish_reason = self._generate(client, generation)
        return _completion_response(request, self._model, text, finish_reason)

    def _generate(self, client: httpx.Client, request: _GenerationRequest) -> tuple[str, str]:
        generated_text = ""
        current_text = request.prompt
        deadline = self._clock() + self._request_timeout_seconds
        rng = random.Random(request.seed)

        for _ in range(request.max_tokens):
            teacher = (
                self._upstream_step(client, self._teacher.endpoint, current_text, deadline) if self._alpha else None
            )
            student = (
                self._upstream_step(client, self._student.endpoint, current_text, deadline) if self._alpha < 1 else None
            )
            mixed = _mix_top_logprobs(
                {} if teacher is None else teacher.top_logprobs,
                {} if student is None else student.top_logprobs,
                alpha=self._alpha,
            )
            if not mixed:
                raise ValueError("upstream completions returned no shared top logprobs")
            token = _sample_token(mixed, temperature=request.temperature, top_p=request.top_p, rng=rng)
            generated_text += token
            current_text += token
            stopped_text = _text_before_stop(generated_text, request.stop)
            if stopped_text is not None:
                return stopped_text, "stop"

        return generated_text, "length"

    def _upstream_step(
        self,
        client: httpx.Client,
        endpoint: OpenAIEndpoint,
        prompt: str,
        deadline: float,
    ) -> _UpstreamStep:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise httpx.TimeoutException("logit mixing request deadline exceeded")
        response = client.post(
            endpoint.url("completions"),
            json={
                "model": endpoint.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": self._top_logprobs,
                "echo": False,
                "stop": None,
                "ignore_eos": True,
            },
            timeout=remaining,
        )
        response.raise_for_status()
        return _upstream_step(response.json())


@contextlib.contextmanager
def start_iris_logit_mixing_brokered_vllm(config: LogitMixingBrokeredVllmConfig) -> Iterator[RunningModel]:
    """Start teacher, student, and mixed endpoints inside an Iris job."""

    with (
        start_iris_brokered_vllm(config.teacher) as teacher,
        start_iris_brokered_vllm(config.student) as student,
    ):
        broker = InferenceBroker(request_lease_timeout_seconds=config.request_lease_timeout_seconds)
        handler = LogitMixingInferenceHandler(
            teacher=teacher,
            student=student,
            model=config.model,
            alpha=config.alpha,
            request_timeout_seconds=config.request_timeout_seconds,
            top_logprobs=config.top_logprobs,
        )
        worker = InferenceWorker(
            broker=broker,
            handler=handler,
            request_timeout_seconds=config.request_timeout_seconds,
        )
        with (
            start_brokered_inference_proxy(
                config.proxy,
                model=config.model,
                broker=broker,
                tokenizer=config.tokenizer,
            ) as running_model,
            run_inference_worker(worker, max_in_flight=config.max_in_flight),
        ):
            _wait_for_brokered_vllm_ready(running_model, timeout_seconds=config.proxy.readiness_timeout_seconds)
            yield running_model


def _generation_request(request: InferenceRequest, model: str) -> _GenerationRequest:
    raw_payload = unpack_json_payload(request.payload)
    try:
        payload = CompletionRequest.model_validate(raw_payload)
    except ValidationError as exc:
        raise ValueError("invalid completion request") from exc

    if payload.model != model:
        raise ValueError(f"completion model must be {model!r}")
    if not isinstance(payload.prompt, str):
        raise ValueError("completion prompt must be a string")
    if payload.echo:
        raise ValueError("logit mixing generation does not support echo=true")
    if payload.n not in {None, 1}:
        raise ValueError("logit mixing generation requires n=1")
    if payload.stream:
        raise ValueError("logit mixing generation does not support streaming")
    if payload.logprobs is not None:
        raise ValueError("logit mixing generation does not return response logprobs")
    supported_fields = {
        "echo",
        "logprobs",
        "max_new_tokens",
        "max_tokens",
        "model",
        "n",
        "prompt",
        "seed",
        "stop",
        "stream",
        "temperature",
        "top_p",
    }
    unsupported = sorted(raw_payload.keys() - supported_fields)
    if unsupported:
        raise ValueError(f"unsupported completion fields: {', '.join(unsupported)}")
    max_tokens = raw_payload.get("max_new_tokens", payload.max_tokens)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if payload.temperature < 0:
        raise ValueError("temperature must be non-negative")
    top_p = 1.0 if payload.top_p is None else payload.top_p
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    stop = (payload.stop,) if isinstance(payload.stop, str) else tuple(payload.stop or ())
    stop = tuple(item for item in stop if item)
    if not stop:
        raise ValueError("logit mixing generation requires a stop sequence")

    return _GenerationRequest(
        prompt=payload.prompt,
        max_tokens=max_tokens,
        temperature=payload.temperature,
        top_p=top_p,
        stop=stop,
        seed=0 if payload.seed is None else payload.seed,
    )


def _upstream_step(payload: object) -> _UpstreamStep:
    if not isinstance(payload, dict):
        raise ValueError("upstream completion response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("upstream completion response must contain a choice")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("upstream completion choice must contain logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(top_logprobs, list) or not top_logprobs or not isinstance(top_logprobs[-1], dict):
        raise ValueError("upstream completion choice must contain top_logprobs")
    values = top_logprobs[-1]
    if any(not isinstance(value, int | float) or isinstance(value, bool) for value in values.values()):
        raise ValueError("upstream top_logprobs must be numeric")
    return _UpstreamStep({str(token): float(value) for token, value in values.items()})


def _models_response(request: InferenceRequest, model: str) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        status_code=200,
        payload=pack_json_payload({"object": "list", "data": [{"id": model, "object": "model"}]}),
    )


def _completion_response(request: InferenceRequest, model: str, text: str, finish_reason: str) -> InferenceResponse:
    payload = {
        "id": "cmpl-logit-mix",
        "object": "text_completion",
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }
    return InferenceResponse(request_id=request.request_id, status_code=200, payload=pack_json_payload(payload))


def _mix_top_logprobs(
    teacher: Mapping[str, float],
    student: Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    if alpha == 1:
        scores = dict(teacher)
    elif alpha == 0:
        scores = dict(student)
    else:
        scores = {
            token: alpha * teacher[token] + (1 - alpha) * student[token] for token in teacher.keys() & student.keys()
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


def _sample_token(
    logprobs: Mapping[str, float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> str:
    if temperature == 0:
        return next(iter(logprobs))
    scaled = {token: logprob / temperature for token, logprob in logprobs.items()}
    normalizer = _logsumexp(scaled.values())
    candidates: list[tuple[str, float]] = []
    cumulative = 0.0
    for token, score in sorted(scaled.items(), key=lambda item: item[1], reverse=True):
        probability = math.exp(score - normalizer)
        candidates.append((token, probability))
        cumulative += probability
        if cumulative >= top_p:
            break
    draw = rng.random() * sum(probability for _, probability in candidates)
    cumulative = 0.0
    for token, probability in candidates:
        cumulative += probability
        if draw <= cumulative:
            return token
    return candidates[-1][0]


def _text_before_stop(text: str, stop: tuple[str, ...]) -> str | None:
    positions = [position for item in stop if (position := text.find(item)) >= 0]
    return text[: min(positions)] if positions else None


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))
