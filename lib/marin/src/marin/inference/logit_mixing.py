# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from time import monotonic

import anyio
import httpx
from levanter.inference.openai_protocol import CompletionRequest
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from marin.inference.quick_serve_dashboard import bind_serving_socket, serve_app_background
from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.vllm import (
    DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT,
    BrokeredVllmSystemConfig,
    start_iris_brokered_vllm,
)

DEFAULT_LOGIT_MIXING_TOP_LOGPROBS = 20
DEFAULT_LOGIT_MIXING_SERVER_START_TIMEOUT = 10.0


@dataclass(frozen=True)
class LogitMixingServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    start_timeout_seconds: float = DEFAULT_LOGIT_MIXING_SERVER_START_TIMEOUT


@dataclass(frozen=True)
class LogitMixingBrokeredVllmConfig:
    """Configure teacher, student, and mixed brokered endpoints."""

    teacher: BrokeredVllmSystemConfig
    student: BrokeredVllmSystemConfig
    alpha: float
    tokenizer: str
    model: str = "logit-mix"
    request_timeout_seconds: float = DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT.total_seconds()
    top_logprobs: int = DEFAULT_LOGIT_MIXING_TOP_LOGPROBS
    server: LogitMixingServerConfig = field(default_factory=LogitMixingServerConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}.")
        if not 1 <= self.top_logprobs <= DEFAULT_LOGIT_MIXING_TOP_LOGPROBS:
            raise ValueError(f"top_logprobs must be in [1, {DEFAULT_LOGIT_MIXING_TOP_LOGPROBS}].")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if self.server.start_timeout_seconds <= 0:
            raise ValueError("server.start_timeout_seconds must be positive.")
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


class LogitMixingService:
    """Serve text completions mixed from two OpenAI-compatible models."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        teacher: RunningModel,
        student: RunningModel,
        model: str,
        alpha: float,
        request_timeout_seconds: float,
        top_logprobs: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._client = client
        self._teacher = teacher
        self._student = student
        self._model = model
        self._alpha = alpha
        self._request_timeout_seconds = request_timeout_seconds
        self._top_logprobs = top_logprobs
        self._clock = clock
        self.app = Starlette(
            routes=[
                Route("/v1/models", self._models),
                Route("/v1/completions", self._completions, methods=["POST"]),
            ]
        )

    def _models(self, _request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": self._model, "object": "model"}]})

    def _completions(self, request: Request) -> Response:
        raw_payload = anyio.from_thread.run(request.json)
        try:
            generation = _generation_request(raw_payload, self._model)
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        try:
            text, finish_reason = self._generate(generation)
        except httpx.TimeoutException:
            return JSONResponse({"error": {"message": "logit mixing request timed out"}}, status_code=504)
        except (httpx.HTTPError, ValueError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=502)
        return JSONResponse(_completion_payload(self._model, text, finish_reason))

    def _generate(self, request: _GenerationRequest) -> tuple[str, str]:
        generated_text = ""
        current_text = request.prompt
        deadline = self._clock() + self._request_timeout_seconds
        rng = random.Random(request.seed)

        for _ in range(request.max_tokens):
            teacher = self._upstream_step(self._teacher.endpoint, current_text, deadline) if self._alpha else None
            student = self._upstream_step(self._student.endpoint, current_text, deadline) if self._alpha < 1 else None
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
        endpoint: OpenAIEndpoint,
        prompt: str,
        deadline: float,
    ) -> _UpstreamStep:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise httpx.TimeoutException("logit mixing request deadline exceeded")
        response = self._client.post(
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
def serve_logit_mixing_model(
    *,
    teacher: RunningModel,
    student: RunningModel,
    model: str,
    tokenizer: str | None,
    alpha: float,
    request_timeout_seconds: float,
    top_logprobs: int = DEFAULT_LOGIT_MIXING_TOP_LOGPROBS,
    server: LogitMixingServerConfig = LogitMixingServerConfig(),
    clock: Callable[[], float] = monotonic,
) -> Iterator[RunningModel]:
    """Serve an OpenAI-compatible logit-mixed completion model."""

    with (
        bind_serving_socket(server.host, server.port) as sock,
        httpx.Client(timeout=request_timeout_seconds) as client,
    ):
        service = LogitMixingService(
            client=client,
            teacher=teacher,
            student=student,
            model=model,
            alpha=alpha,
            request_timeout_seconds=request_timeout_seconds,
            top_logprobs=top_logprobs,
            clock=clock,
        )
        with serve_app_background(service.app, sock, start_timeout_seconds=server.start_timeout_seconds):
            host, port = sock.getsockname()[:2]
            yield RunningModel(
                endpoint=OpenAIEndpoint(base_url=f"http://{host}:{port}/v1", model=model),
                tokenizer=tokenizer,
            )


@contextlib.contextmanager
def start_iris_logit_mixing_brokered_vllm(config: LogitMixingBrokeredVllmConfig) -> Iterator[RunningModel]:
    """Start brokered teacher and student models with a mixed completion endpoint."""

    with (
        start_iris_brokered_vllm(config.teacher) as teacher,
        start_iris_brokered_vllm(config.student) as student,
        serve_logit_mixing_model(
            teacher=teacher,
            student=student,
            model=config.model,
            tokenizer=config.tokenizer,
            alpha=config.alpha,
            request_timeout_seconds=config.request_timeout_seconds,
            top_logprobs=config.top_logprobs,
            server=config.server,
        ) as running_model,
    ):
        yield running_model


def _generation_request(raw_payload: object, model: str) -> _GenerationRequest:
    if not isinstance(raw_payload, dict):
        raise ValueError("completion request must be an object")
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


def _completion_payload(model: str, text: str, finish_reason: str) -> dict[str, object]:
    return {
        "id": "cmpl-logit-mix",
        "object": "text_completion",
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }


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
