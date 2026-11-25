# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI-compatible inference API for Levanter models.

This module provides FastAPI-based endpoints that are compatible with OpenAI's
completions and chat completions APIs, allowing Levanter models to be used as
drop-in replacements for OpenAI models.
"""

import asyncio
import base64
import collections
import collections.abc
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import haliax as hax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import jax.random as jrandom
import numpy as np
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, HTTPException, Request as HttpRequest, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from openai.types import Completion, CompletionUsage, Model
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import Choice as ChatCompletionChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceLogprobs as ChunkChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import CompletionChoice, Logprobs
from levanter.inference.binary import (
    BEHAVIOR_LOGPROBS,
    EXPERT_IDS,
    EXPERT_VALID,
    GENERATE_PATH,
    LOGPROB_MODE,
    MEDIA_TYPE,
    MAX_POLICY_ID_LENGTH,
    POLICY_ID,
    POLICY_LOGPROBS,
    PROMPT_TOKEN_IDS,
    TOKEN_IDS,
    GenerateOptions,
    choice_tensor_key,
    decode_frame,
    encode_frame,
)
from levanter.inference.engine import (
    GenerationResult,
    GeneratedExpertRoutes,
    InferenceEngine,
    InferenceEngineConfig,
    Request,
    TokenSequenceLogprobs,
)
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams, format_finish_reason
from levanter.layers.sampler import LogprobMode
from levanter.inference.openai_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    TokenList,
    TokensRequest,
    TokensResponse,
)
from levanter.inference.utils import INVALID
from levanter.models.lm_model import LmHeadModel
from levanter.tokenizers import HfMarinTokenizer, MarinTokenizer
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)

POLICY_MISMATCH = "The requested policy is no longer active"
CONTROL_SUCCESS = {"status": "ok"}


DEFAULT_MODEL_NAME = "levanter"
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_EVENT_STREAM = "text/event-stream"
_CHAT_CHUNK = "chat.completion.chunk"
RESERVED_CHAT_TEMPLATE_KWARGS = frozenset({"add_generation_prompt", "chat_template", "return_dict", "tokenize"})


@dataclass
class InferenceServerConfig:
    """Configuration for OpenAI-compatible inference server."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(4096))

    model_name: str = DEFAULT_MODEL_NAME
    """Model id this server advertises from ``/v1/models``."""
    initial_policy_id: str | None = None
    """Immutable weight identity supplied by the publisher."""

    # Default generation parameters for API
    temperature: float = 0.7
    seed: int = 42

    batch_timeout: float = 0.1  # seconds to wait for more requests before processing batch

    host: str = "localhost"
    port: int = 0  # auto-assign port


@dataclass
class InferenceRequest:
    """Internal request structure for the inference thread"""

    request_id: str
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float
    top_p: float | None
    stop_tokens: List[List[int]] | None
    seed: int | None
    future: asyncio.Future
    abort_event: threading.Event = field(default_factory=threading.Event)
    n_generations: int = 1
    echo_logprobs_top_k: int | None = None
    return_route_captures: bool = False
    expected_policy_id: str | None = None
    on_tokens: collections.abc.Callable[[int, list[int]], None] | None = None


@dataclass
class InferenceResponse:
    """Internal response structure for the inference thread"""

    request_id: str
    text: str
    tokens: List[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason
    logprobs: Optional[List[float]] = None
    echo_token_ids: List[int] | None = None
    echo_logprobs: TokenSequenceLogprobs | None = None
    expert_routes: GeneratedExpertRoutes | None = None
    model_version: int = 0
    policy_id: str | None = None


class PauseGenerationRequest(BaseModel):
    mode: Literal["abort"]
    clear_cache: Literal[True]


class AbortRequest(BaseModel):
    request_ids: list[str] = Field(min_length=1)


class PolicyMismatchError(ValueError):
    """The requested policy ID differs from the active policy ID."""


class InferenceBatch(list):
    def num_seqs(self) -> int:
        return sum(req.n_generations for req in self)

    def total_tokens(self) -> int:
        return sum(len(req.prompt_tokens) + req.max_tokens for req in self)


# A callback which replaces the current model.
WeightSource = collections.abc.Callable[[LmHeadModel], LmHeadModel]


def _fetch_all_from_queue(q: queue.Queue, timeout: float) -> List:
    """Fetch all items from `q` which arrive within `timeout` seconds."""
    deadline = time.time() + timeout
    items = []
    while time.time() < deadline:
        try:
            item = q.get(timeout=max(0, deadline - time.time()))
            items.append(item)
        except queue.Empty:
            break
    return items


def _encode_stop_tokens(stop: Union[str, List[str], None], tokenizer: MarinTokenizer) -> List[List[int]] | None:
    """Tokenize each stop string as a separate sequence."""
    if not stop:
        return None
    stop_list = [stop] if isinstance(stop, str) else stop
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in stop_list]
    return [seq for seq in sequences if seq] or None


class InferenceContext:
    """Background thread that manages the InferenceEngine and processes requests"""

    def __init__(
        self,
        model: LmHeadModel,
        tokenizer,
        engine: InferenceEngine,
        config: InferenceServerConfig,
        *,
        mesh: Mesh | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.config = config
        self.mesh = config.trainer.device_mesh if mesh is None else mesh
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.batch_queue: queue.Queue[InferenceBatch] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._publication_pending = threading.Event()
        self._paused = threading.Event()
        self._model_version = 0
        self._policy_id = config.initial_policy_id
        self._published_policy_ids = {self._policy_id} if self._policy_id is not None else set()
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self._next_request_id = 0
        self._requests_lock = threading.Lock()
        self._http_requests: dict[str, threading.Event] = {}
        self._queued_requests: dict[str, InferenceRequest] = {}

    def start(self):
        """Start the inference and batch processing threads"""
        logger.info("Starting inference context...")
        self.inference_thread.start()
        self.batch_thread.start()

    def shutdown(self):
        """Signal shutdown and wait for the serving workers to exit."""
        logger.info("Shutting down inference context.")
        self.shutdown_event.set()
        self.inference_thread.join()
        self.batch_thread.join()

    def unload(self):
        """Unload the inference model to free up resources."""
        logger.info("Unloading inference model...")
        with self._publication_lock, self.model_lock:
            # pyrefly: ignore[bad-assignment]  # unload() deliberately clears the model; reload() restores it
            self.model = None
            self.engine = None  # type: ignore[assignment]
        logger.info("Inference model unloaded.")

    def pause_generation(self) -> None:
        """Abort outstanding generation and wait for the serving cache to clear."""
        with self._publication_lock:
            with self._requests_lock:
                self._paused.set()
                for request in self._queued_requests.values():
                    request.abort_event.set()
                    self._abort_queued_request(request)
                self._queued_requests.clear()
            with (
                self.model_lock,
                hax.partitioning.set_mesh(self.mesh),
                hax.axis_mapping(self.config.trainer.compute_axis_mapping),
            ):
                self.engine.reset()
                jax.block_until_ready(self.engine.gen_state)

    def resume_generation(self) -> None:
        """Reopen admission after any in-flight pause or publication completes."""
        with self._publication_lock, self._requests_lock:
            self._paused.clear()

    def _check_policy_update(self, expected_version: int | None, policy_id: str | None) -> None:
        if expected_version is not None and expected_version != self._model_version:
            raise ValueError(f"Expected model version {expected_version}, serving {self._model_version}")
        if policy_id is not None and (
            not 0 < len(policy_id) <= MAX_POLICY_ID_LENGTH or policy_id in self._published_policy_ids
        ):
            raise ValueError("A weight update requires a new, nonempty policy ID")

    @contextmanager
    def _publication_barrier(self) -> collections.abc.Iterator[None]:
        with self._publication_lock:
            self._publication_pending.set()
            try:
                with (
                    self.model_lock,
                    hax.partitioning.set_mesh(self.mesh),
                    hax.axis_mapping(self.config.trainer.compute_axis_mapping),
                ):
                    yield
            finally:
                self._publication_pending.clear()

    def publish(self, restore: WeightSource, *, expected_version: int, policy_id: str) -> int:
        """Stage compatible weights while serving, then install at the expected version."""
        with self._publication_lock:
            self._check_policy_update(expected_version, policy_id)
            active, engine = self.model, self.engine
        with (
            hax.partitioning.set_mesh(self.mesh),
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            candidate = restore(active)
            if not engine.can_reuse_cache(candidate):
                raise ValueError("Staged weights must preserve the model structure and array layouts")
            jax.block_until_ready(candidate)

        with self._publication_barrier():
            self._check_policy_update(expected_version, policy_id)
            if engine is not self.engine:
                raise ValueError("Serving engine changed while staging weights")
            self._reset_engine_for_model(engine, candidate)
            return self._commit_policy(candidate, engine, policy_id)

    def reload(
        self, weight_callback: WeightSource, *, expected_version: int | None = None, policy_id: str | None = None
    ) -> int:
        """Install new weights with empty KV state and return the new serving version."""
        with self._publication_barrier():
            self._check_policy_update(expected_version, policy_id)
            model = weight_callback(self.model)
            engine = self.engine
            if engine.can_reuse_cache(model):
                self._reset_engine_for_model(engine, model)
            else:
                engine = InferenceEngine.from_model_with_config(
                    model=model,
                    tokenizer=self.tokenizer,
                    config=self.config.service,
                    axis_resources=self.config.trainer.compute_axis_mapping,
                )
            return self._commit_policy(model, engine, policy_id)

    @staticmethod
    def _reset_engine_for_model(engine: InferenceEngine, model: LmHeadModel) -> None:
        engine.reset()
        jax.block_until_ready(engine.gen_state)
        engine.model = model

    def _commit_policy(self, model: LmHeadModel, engine: InferenceEngine, policy_id: str | None) -> int:
        self.model, self.engine = model, engine
        with self._requests_lock:
            self._model_version += 1
            self._policy_id = policy_id
            if policy_id is not None:
                self._published_policy_ids.add(policy_id)
            return self._model_version

    def submit_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        stop_tokens: List[List[int]] | None,
        seed: int | None,
        future: asyncio.Future,
        n_generations: int = 1,
        echo_logprobs_top_k: int | None = None,
        *,
        abort_event: threading.Event | None = None,
        return_route_captures: bool = False,
        expected_policy_id: str | None = None,
        on_tokens: collections.abc.Callable[[int, list[int]], None] | None = None,
    ) -> str:
        """Submit a request to the inference queue"""
        assert self.shutdown_event.is_set() is False, "InferenceContext is shut down"
        with self._requests_lock:
            request_id = f"req_{self._next_request_id}"
            self._next_request_id += 1

            request = InferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_tokens=stop_tokens,
                seed=seed,
                future=future,
                abort_event=abort_event if abort_event is not None else threading.Event(),
                n_generations=n_generations,
                echo_logprobs_top_k=echo_logprobs_top_k,
                return_route_captures=return_route_captures,
                expected_policy_id=expected_policy_id,
                on_tokens=on_tokens,
            )
            if self._paused.is_set() or request.abort_event.is_set():
                self._abort_queued_request(request)
            else:
                self._queued_requests[request_id] = request
                self.request_queue.put(request)
            return request_id

    def _register_http_request(self, request: HttpRequest) -> tuple[str, threading.Event]:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        event = threading.Event()
        with self._requests_lock:
            if request_id in self._http_requests:
                raise HTTPException(status_code=409, detail="Request ID is already active")
            self._http_requests[request_id] = event
        return request_id, event

    def _unregister_http_request(self, request_id: str, event: threading.Event) -> None:
        with self._requests_lock:
            if self._http_requests.get(request_id) is event:
                del self._http_requests[request_id]

    async def _watch_disconnect(self, request: HttpRequest, request_id: str) -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass
        self.abort([request_id])

    @asynccontextmanager
    async def http_generation(self, request: HttpRequest) -> collections.abc.AsyncIterator[threading.Event]:
        request_id, event = self._register_http_request(request)
        disconnect = asyncio.create_task(self._watch_disconnect(request, request_id))
        try:
            yield event
        except BaseException:
            self.abort([request_id])
            raise
        finally:
            disconnect.cancel()
            self._unregister_http_request(request_id, event)
            with suppress(asyncio.CancelledError):
                await disconnect

    def abort(self, request_ids: collections.abc.Sequence[str]) -> None:
        """Abort queued or running generation for the supplied HTTP request IDs."""
        with self._requests_lock:
            for request_id in request_ids:
                event = self._http_requests.get(request_id)
                if event is not None:
                    event.set()
            cancelled = [request for request in self._queued_requests.values() if request.abort_event.is_set()]
            for request in cancelled:
                del self._queued_requests[request.request_id]
                self._abort_queued_request(request)

    def _abort_queued_request(self, request: InferenceRequest) -> None:
        if request.expected_policy_id is not None and request.expected_policy_id != self._policy_id:
            self._resolve_request(request, PolicyMismatchError(POLICY_MISMATCH))
            return
        self._resolve_request(
            request,
            [
                InferenceResponse(
                    request_id=request.request_id,
                    text="",
                    tokens=[],
                    logprobs=[],
                    prompt_tokens=len(request.prompt_tokens),
                    completion_tokens=0,
                    finish_reason=FinishReason.ABORT,
                    model_version=self._model_version,
                    policy_id=self._policy_id,
                )
                for _ in range(request.n_generations)
            ],
        )

    def _inference_loop(self) -> None:
        """Collect requests from the serving and batch them into batches of appropriate size for inference."""
        logger.info("Inference thread started")

        while not self.shutdown_event.is_set():
            requests: list[InferenceRequest] = _fetch_all_from_queue(self.request_queue, self.config.batch_timeout)
            if not requests:
                continue

            batch = InferenceBatch()
            max_tokens_per_seq = self.engine.config.max_seq_len
            max_tokens_per_batch = self.engine.config.page_size * self.engine.config.max_pages  # type: ignore
            logger.debug("Max tokens per sequence %d and batch %d", max_tokens_per_seq, max_tokens_per_batch)

            for r in requests:
                if r.abort_event.is_set():
                    continue
                if len(r.prompt_tokens) > max_tokens_per_seq:
                    # slice down requests that are too long
                    logger.warning(
                        "Request %s prompt too long (%d tokens), truncating to last %d tokens",
                        r.request_id,
                        len(r.prompt_tokens),
                        max_tokens_per_seq,
                    )
                    r.prompt_tokens = r.prompt_tokens[-max_tokens_per_seq:]

                if r.n_generations > self.engine.config.max_seqs:
                    # fail requests that are too large
                    error_msg = (
                        f"Request {r.request_id} has n={r.n_generations} which exceeds "
                        f"the maximum allowed {self.engine.config.max_seqs}"
                    )
                    logger.error(error_msg)
                    with self._requests_lock:
                        if self._queued_requests.pop(r.request_id, None) is not None:
                            self._resolve_request(r, ValueError(error_msg))
                    continue

                if (
                    batch.num_seqs() + r.n_generations <= self.engine.config.max_seqs
                    and batch.total_tokens() + (len(r.prompt_tokens) + r.max_tokens) <= max_tokens_per_batch
                ):
                    batch.append(r)
                else:
                    if batch:
                        self.batch_queue.put(batch)
                    batch = InferenceBatch([r])

            if batch:
                self.batch_queue.put(batch)

        logger.info("Inference thread shutting down")

    def _batch_processing_loop(self) -> None:
        waiting: collections.deque[InferenceRequest] = collections.deque()
        # Retained-prefix admission reserves the shared KV budget across batches.
        reserve_pages = self.config.service.max_retained_prompts > 0

        def poll(capacity: int, capture_routes: bool) -> InferenceBatch:
            """Take ready requests in order, leaving a blocked head in the pending queue."""
            admitted = InferenceBatch()
            while capacity > 0:
                if not waiting:
                    if admitted and not reserve_pages:
                        break
                    try:
                        waiting.extend(self.batch_queue.get_nowait())
                    except queue.Empty:
                        break
                request = waiting[0]
                if request.n_generations > capacity or (request.return_route_captures and not capture_routes):
                    break
                admitted.append(waiting.popleft())
                capacity -= request.n_generations
            return admitted

        while not self.shutdown_event.is_set():
            batch = poll(self.config.service.max_seqs, capture_routes=True)
            if not batch:
                try:
                    batch = self.batch_queue.get(timeout=1)
                except queue.Empty:
                    continue
            try:
                # A waiting publication gets a chance to acquire the model lock
                # before the next generation window starts.
                with self._publication_lock:
                    pass
                with (
                    self.model_lock,
                    hax.partitioning.set_mesh(self.mesh),
                    hax.axis_mapping(self.config.trainer.compute_axis_mapping),
                ):
                    self._execute_batch(batch, poll)
            except Exception as exc:
                logger.exception("Inference window failed")
                for request in batch:
                    self._resolve_request(request, exc)

    def _filter_policy_requests(self, requests: InferenceBatch) -> InferenceBatch:
        admitted = InferenceBatch()
        for request in requests:
            with self._requests_lock:
                queued = self._queued_requests.pop(request.request_id, None)
                if queued is not None and request.abort_event.is_set():
                    self._abort_queued_request(request)
                    continue
            if queued is None or request.future.cancelled():
                continue
            if request.expected_policy_id is not None and request.expected_policy_id != self._policy_id:
                self._resolve_request(request, PolicyMismatchError(POLICY_MISMATCH))
            else:
                admitted.append(request)
        return admitted

    @staticmethod
    def _resolve_request(request: InferenceRequest, outcome: list[InferenceResponse] | Exception) -> None:
        def resolve() -> None:
            if request.future.done():
                return
            if isinstance(outcome, Exception):
                request.future.set_exception(outcome)
            else:
                request.future.set_result(outcome)

        request.future.get_loop().call_soon_threadsafe(resolve)

    @staticmethod
    def _engine_request(request: InferenceRequest, request_id: int) -> Request:
        stop_ids = None
        if request.stop_tokens:
            width = max(map(len, request.stop_tokens))
            padded = [[INVALID] * (width - len(seq)) + seq for seq in request.stop_tokens]
            stop_ids = hax.named(jnp.asarray(padded, dtype=jnp.int32), ("stop_seq", "position"))
        seed = request.seed
        if seed is None:
            seed = np.random.default_rng().integers(np.iinfo(np.uint32).max, endpoint=True, dtype=np.uint32)
        params = SeqDecodingParams(
            max_num_tokens=jnp.asarray(len(request.prompt_tokens) + request.max_tokens, dtype=jnp.int32),
            stop_tokens=stop_ids,
            temperature=jnp.asarray(request.temperature, dtype=jnp.float32),
            top_p=jnp.asarray(1.0 if request.top_p is None else request.top_p, dtype=jnp.float32),
            key=jrandom.PRNGKey(seed),
        )
        return Request(request.prompt_tokens, request_id, params, request.n_generations)

    def _execute_batch(
        self, requests: InferenceBatch, poll: collections.abc.Callable[[int, bool], InferenceBatch]
    ) -> None:
        if not self.engine:
            raise RuntimeError("Inference engine is not initialized.")
        capture_routes = any(request.return_route_captures for request in requests)
        outstanding: dict[int, InferenceRequest] = {}
        next_id = 0

        def admit(batch: InferenceBatch) -> list[Request]:
            nonlocal next_id
            admitted = []
            for request in self._filter_policy_requests(batch):
                try:
                    engine_request = self._engine_request(request, next_id)
                except Exception as exc:
                    self._resolve_request(request, exc)
                    continue
                outstanding[next_id] = request
                admitted.append(engine_request)
                next_id += 1
            return admitted

        def receive(capacity: int) -> list[Request]:
            if self._paused.is_set() or self._publication_pending.is_set() or self.shutdown_event.is_set():
                return []
            return admit(poll(capacity, capture_routes))

        def complete(request_id: int, result: GenerationResult) -> None:
            request = outstanding.pop(request_id)
            if request.future.cancelled():
                return
            try:
                self._complete_request(request, result)
            except Exception as exc:
                self._resolve_request(request, exc)

        def should_abort(request_id: int) -> bool:
            request = outstanding[request_id]
            return self._paused.is_set() or request.abort_event.is_set() or request.future.cancelled()

        def stream_tokens(request_id: int, choice: int, tokens: list[int]) -> None:
            request = outstanding[request_id]
            if request.on_tokens is not None and not request.future.cancelled():
                request.on_tokens(choice, tokens)

        try:
            initial = admit(requests)
            self.engine.generate_continuously(
                initial,
                on_complete=complete,
                poll_requests=receive,
                on_tokens=stream_tokens,
                should_abort=should_abort,
                capture_routes=capture_routes,
            )
        except Exception as exc:
            for request in outstanding.values():
                self._resolve_request(request, exc)
            raise

    def _complete_request(self, request: InferenceRequest, result: GenerationResult) -> None:
        outputs = []
        for index, tokens in enumerate(result.tokens):
            echo_token_ids = echo_logprobs = None
            if request.echo_logprobs_top_k is not None and (
                tokens or result.finish_reasons[index] != FinishReason.ABORT
            ):
                echo_token_ids = request.prompt_tokens + tokens
                echo_logprobs = self.engine.score_token_logprobs(echo_token_ids, request.echo_logprobs_top_k)
            outputs.append(
                InferenceResponse(
                    request_id=request.request_id,
                    text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    tokens=tokens,
                    prompt_tokens=len(request.prompt_tokens),
                    completion_tokens=len(tokens),
                    finish_reason=result.finish_reasons[index],
                    logprobs=result.logprobs[index] if result.logprobs is not None else None,
                    model_version=self._model_version,
                    policy_id=self._policy_id,
                    echo_token_ids=echo_token_ids,
                    echo_logprobs=echo_logprobs,
                    expert_routes=(
                        result.expert_routes[index]
                        if request.return_route_captures and result.expert_routes is not None
                        else None
                    ),
                )
            )

        self._resolve_request(request, outputs)


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


def _chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def _sse_event(payload: str) -> str:
    return f"data: {payload}\n\n"


def _chat_completion_events(
    completion: ChatCompletion,
    *,
    emitted_text: dict[int, str] | None = None,
    include_usage: bool = False,
    route_frame: str | None = None,
) -> collections.abc.Iterator[str]:
    """Finish each choice and optionally deliver final token usage."""
    final_choice = completion.choices[-1].index
    for choice in completion.choices:
        logprobs = (
            ChunkChoiceLogprobs(content=choice.logprobs.content, refusal=choice.logprobs.refusal)
            if choice.logprobs is not None
            else None
        )
        prefix = emitted_text.get(choice.index, "") if emitted_text is not None else ""
        text = choice.message.content or ""
        if not text.startswith(prefix):
            raise ValueError("Incremental decoding differs from the final completion")
        remainder = text[len(prefix) :]
        chunks = []
        if not prefix or remainder:
            chunks.append(
                ChatCompletionChunkChoice(
                    index=choice.index,
                    delta=ChoiceDelta(role=None if prefix else "assistant", content=remainder),
                    logprobs=logprobs,
                )
            )
        chunks.append(
            ChatCompletionChunkChoice(index=choice.index, delta=ChoiceDelta()).model_copy(
                update={"finish_reason": choice.finish_reason}
            )
        )
        for chunk_choice in chunks:
            chunk = ChatCompletionChunk(
                id=completion.id,
                object=_CHAT_CHUNK,
                created=completion.created,
                model=completion.model,
                choices=[chunk_choice],
            )
            payload = chunk.model_dump_json()
            if route_frame is not None and choice.index == final_choice and chunk_choice.finish_reason is not None:
                payload = payload[:-1] + ',"levanter_route_frame":"' + route_frame + '"}'
            yield _sse_event(payload)
    if include_usage:
        usage = ChatCompletionChunk(
            id=completion.id,
            object=_CHAT_CHUNK,
            created=completion.created,
            model=completion.model,
            choices=[],
            usage=completion.usage,
        )
        yield _sse_event(usage.model_dump_json())
    yield _sse_event("[DONE]")


def _completion_events(completion: Completion) -> collections.abc.Iterator[str]:
    """Render a finished text completion as an OpenAI server-sent-event stream.

    The streaming and non-streaming completion payloads share a shape, so each choice rides
    its own event. As with chat, the events are well-formed but not incremental.
    """
    for choice in completion.choices:
        chunk = Completion(
            id=completion.id,
            object="text_completion",
            created=completion.created,
            model=completion.model,
            choices=[choice],
        )
        yield _sse_event(chunk.model_dump_json())
    yield _sse_event("[DONE]")


def _decoded_token_pieces(tokenizer: MarinTokenizer, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def _token_text_offsets(tokens: List[str]) -> List[int]:
    offsets = []
    offset = 0
    for token in tokens:
        offsets.append(offset)
        offset += len(token)
    return offsets


@dataclass(frozen=True)
class CompletionLogprobData:
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[dict[str, float]]
    text_offset: List[int]


def _completion_logprobs(
    tokenizer: MarinTokenizer,
    token_ids: List[int],
    sequence_logprobs: TokenSequenceLogprobs,
) -> CompletionLogprobData:
    tokens = _decoded_token_pieces(tokenizer, token_ids)
    if len(tokens) != len(sequence_logprobs.token_logprobs):
        raise ValueError(f"Expected {len(tokens)} token logprobs, got {len(sequence_logprobs.token_logprobs)}")
    if len(tokens) != len(sequence_logprobs.top_token_logprobs):
        raise ValueError(
            f"Expected {len(tokens)} top-logprob entries, got {len(sequence_logprobs.top_token_logprobs)}"
        )

    top_logprobs = [
        {
            tokenizer.decode([token_id], skip_special_tokens=False): logprob
            for token_id, logprob in token_logprobs.items()
        }
        for token_logprobs in sequence_logprobs.top_token_logprobs
    ]
    return CompletionLogprobData(
        tokens=tokens,
        token_logprobs=sequence_logprobs.token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=_token_text_offsets(tokens),
    )


async def _create_completion(
    ctx: InferenceContext, request: CompletionRequest, abort_event: threading.Event | None = None
) -> Completion:
    """Create a text completion using OpenAI API format."""
    try:
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        stop_tokens = _encode_stop_tokens(request.stop, ctx.tokenizer)

        # Create futures for all prompts
        futures = []
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        prompt_token_lists: List[List[int]] = []
        echo_logprobs_top_k = int(request.logprobs) if request.echo and request.logprobs else None

        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = ctx.tokenizer.encode(prompt, add_special_tokens=False)
            scored_token_budget = len(prompt_tokens) + request.max_tokens
            if echo_logprobs_top_k is not None and scored_token_budget > ctx.config.service.max_seq_len:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "echo logprobs are not supported when prompt tokens plus max_tokens exceeds "
                        f"max_seq_len={ctx.config.service.max_seq_len}"
                    ),
                )
            prompt_token_lists.append(prompt_tokens)
            total_prompt_tokens += len(prompt_tokens)

        for prompt_tokens in prompt_token_lists:
            # Create future for this request
            future: asyncio.Future = asyncio.Future()
            futures.append(future)

            # Submit to inference thread
            ctx.submit_request(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_tokens=stop_tokens,
                seed=request.seed,
                future=future,
                n_generations=request.n or 1,
                echo_logprobs_top_k=echo_logprobs_top_k,
                abort_event=abort_event,
            )

        # Wait for all results
        results: List[List[InferenceResponse]] = await asyncio.gather(*futures)

        # Format responses
        choice_idx = 0
        for prompt, result in zip(prompts, results, strict=True):
            for generation in result:
                choice_text = f"{prompt}{generation.text}" if request.echo else generation.text

                # Format logprobs if available
                logprobs = None
                if request.logprobs and (generation.tokens or generation.finish_reason != FinishReason.ABORT):
                    if request.echo:
                        if generation.echo_token_ids is None or generation.echo_logprobs is None:
                            raise RuntimeError("Echo logprobs requested but missing from generation result.")
                        echo_logprobs = _completion_logprobs(
                            ctx.tokenizer,
                            generation.echo_token_ids,
                            generation.echo_logprobs,
                        )
                        logprobs = Logprobs(
                            tokens=echo_logprobs.tokens,
                            token_logprobs=echo_logprobs.token_logprobs,
                            text_offset=echo_logprobs.text_offset,
                            top_logprobs=echo_logprobs.top_logprobs,
                        )
                    else:
                        # Convert logprobs to API format
                        generated_tokens = generation.tokens

                        # Create token logprobs in OpenAI format
                        tokens = []
                        token_logprobs = []
                        if generation.logprobs:
                            for token_id, lp in zip(generated_tokens, generation.logprobs):
                                # Use convert_ids_to_tokens to preserve BPE format
                                token_str = ctx.tokenizer.convert_ids_to_tokens(token_id)
                                tokens.append(token_str)
                                token_logprobs.append(float(lp))

                        logprobs = Logprobs(
                            tokens=tokens,
                            token_logprobs=token_logprobs,
                            text_offset=None,
                            top_logprobs=None,
                        )

                choices.append(
                    CompletionChoice.model_construct(
                        text=choice_text,
                        index=choice_idx,
                        finish_reason=format_finish_reason(generation.finish_reason),
                        logprobs=logprobs,
                    )
                )
                total_completion_tokens += generation.completion_tokens
                choice_idx += 1

        return Completion(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in completion.", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _compute_tokens(
    messages: list[ChatMessage],
    tokenizer: MarinTokenizer,
    tools: list[dict[str, object]] | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
) -> List[int]:
    """Encode a conversation with the tokenizer's chat template.

    A model with no chat template cannot represent a conversation, so a chat request against one
    is rejected rather than rendered into some invented format the model never saw. Base and
    midtrained checkpoints are served through ``/v1/completions`` instead.
    """
    if tokenizer.chat_template is None:
        raise HTTPException(
            status_code=400,
            detail="This model has no chat template; use /v1/completions, or serve it with a chat template.",
        )
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    template_kwargs = dict(chat_template_kwargs or {})
    overridden = sorted(RESERVED_CHAT_TEMPLATE_KWARGS.intersection(template_kwargs))
    if overridden:
        names = ", ".join(overridden)
        raise HTTPException(status_code=400, detail=f"chat_template_kwargs may not override: {names}")
    if tools is not None:
        template_kwargs["tools"] = tools
    # return_dict=False pins the token ids to a flat list; tokenizers otherwise hand back a
    # BatchEncoding here, which is the shape the rest of this module cannot use.
    result = tokenizer.apply_chat_template(
        dict_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **template_kwargs,
    )
    assert isinstance(result, list)
    return result


async def _fetch_tokens(ctx: InferenceContext, request: TokensRequest) -> TokensResponse:
    """Fetch tokenized prompts after system prompt injection and encoding."""
    try:
        results = []
        for messages in request.message_list:
            token_ids = _compute_tokens(messages, ctx.tokenizer)
            results.append(TokenList(tokens=token_ids))

        return TokensResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in tokenization.", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _submit_chat_completion(
    ctx: InferenceContext,
    request: ChatCompletionRequest,
    on_tokens: collections.abc.Callable[[int, list[int]], None] | None = None,
    *,
    abort_event: threading.Event | None = None,
) -> tuple[list[int], asyncio.Future[list[InferenceResponse]]]:
    prompt_tokens = _compute_tokens(request.messages, ctx.tokenizer, request.tools, request.chat_template_kwargs)
    future = asyncio.get_running_loop().create_future()
    ctx.submit_request(
        prompt_tokens=prompt_tokens,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_tokens=_encode_stop_tokens(request.stop, ctx.tokenizer),
        seed=request.seed,
        future=future,
        n_generations=request.n or 1,
        on_tokens=on_tokens,
        abort_event=abort_event,
        return_route_captures=request.return_route_captures,
        expected_policy_id=request.expected_policy_id,
    )
    return prompt_tokens, future


def _format_chat_completion(
    ctx: InferenceContext,
    request: ChatCompletionRequest,
    prompt_tokens: list[int],
    results: list[InferenceResponse],
    *,
    completion_id: str,
    created: int,
) -> ChatCompletion:
    # Format response
    choices = []
    total_completion_tokens = 0

    for i, generation in enumerate(results):
        # Format logprobs if available
        logprobs = None
        if request.logprobs:
            generated_tokens = generation.tokens

            # Create content logprobs in OpenAI format
            content_logprobs = []
            assert generation.logprobs is not None, "Logprobs requested but missing in generation result"
            for token_id, lp in zip(generated_tokens, generation.logprobs, strict=True):
                # Use convert_ids_to_tokens to preserve BPE format (e.g., Ġ for spaces)
                # This allows the client to round-trip: convert_tokens_to_ids(token_str) == token_id
                token_str = ctx.tokenizer.convert_ids_to_tokens(token_id)
                content_logprobs.append(
                    ChatCompletionTokenLogprob(
                        token=token_str,
                        logprob=float(lp),
                        bytes=list(token_str.encode("utf-8")),
                        top_logprobs=[],
                    )
                )

            logprobs = ChoiceLogprobs(content=content_logprobs)

        choices.append(
            ChatCompletionChoice.model_construct(
                index=i,
                message=ChatCompletionMessage(role="assistant", content=generation.text),
                finish_reason=format_finish_reason(generation.finish_reason),
                logprobs=logprobs,
            )
        )
        total_completion_tokens += generation.completion_tokens

    response = ChatCompletion(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=total_completion_tokens,
            total_tokens=len(prompt_tokens) + total_completion_tokens,
        ),
    )
    return response


async def _create_chat_completion(
    ctx: InferenceContext, request: ChatCompletionRequest, abort_event: threading.Event | None = None
) -> ChatCompletion:
    try:
        prompt, future = _submit_chat_completion(ctx, request, abort_event=abort_event)
        return _format_chat_completion(
            ctx,
            request,
            prompt,
            await future,
            completion_id=_chat_completion_id(),
            created=int(time.time()),
        )
    except HTTPException:
        raise
    except PolicyMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _stream_chat_completion(
    ctx: InferenceContext,
    request: ChatCompletionRequest,
    tokenizer: HfMarinTokenizer,
    raw_request: HttpRequest,
    *,
    include_usage: bool,
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[tuple[int, list[int]] | None] = asyncio.Queue()
    closed = threading.Event()

    def on_tokens(choice: int, tokens: list[int]) -> None:
        if not closed.is_set():
            loop.call_soon_threadsafe(updates.put_nowait, (choice, tokens))

    def on_done(future: asyncio.Future[list[InferenceResponse]]) -> None:
        if not future.cancelled():
            future.exception()
        if not closed.is_set():
            updates.put_nowait(None)

    request_id, abort_event = ctx._register_http_request(raw_request)
    try:
        prompt, future = _submit_chat_completion(ctx, request, on_tokens, abort_event=abort_event)
    except Exception:
        ctx._unregister_http_request(request_id, abort_event)
        raise
    future.add_done_callback(on_done)
    header = ChatCompletionChunk(
        id=_chat_completion_id(),
        object=_CHAT_CHUNK,
        created=int(time.time()),
        model=request.model,
        choices=[],
    )
    disconnect = asyncio.create_task(ctx._watch_disconnect(raw_request, request_id))
    try:
        first = await updates.get()
        if first is None:
            future.result()
    except asyncio.CancelledError:
        closed.set()
        ctx.abort([request_id])
        future.cancel()
        ctx._unregister_http_request(request_id, abort_event)
        raise
    except PolicyMismatchError as exc:
        closed.set()
        ctx._unregister_http_request(request_id, abort_event)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        closed.set()
        ctx._unregister_http_request(request_id, abort_event)
        logger.exception("Error before chat streaming started")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        disconnect.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect

    async def events() -> collections.abc.AsyncIterator[str]:
        decoders: dict[int, collections.abc.Callable[[list[int]], str | None]] = {}
        emitted: dict[int, list[str]] = {}
        update = first
        try:
            while update is not None:
                choice, tokens = update
                if choice not in decoders:
                    decoders[choice] = tokenizer.streaming_decoder()
                text = decoders[choice](tokens)
                if text:
                    pieces = emitted.setdefault(choice, [])
                    delta = ChoiceDelta(role=None if pieces else "assistant", content=text)
                    pieces.append(text)
                    chunk = header.model_copy(
                        update={"choices": [ChatCompletionChunkChoice(index=choice, delta=delta)]}
                    )
                    yield _sse_event(chunk.model_dump_json())
                update = await updates.get()
            results = future.result()
            completion = _format_chat_completion(
                ctx, request, prompt, results, completion_id=header.id, created=header.created
            )
            frame = None
            if request.return_route_captures:
                frame = await asyncio.to_thread(
                    _encode_chat_route_frame, prompt, results, LogprobMode(ctx.config.service.logprob_mode)
                )
            yield_events = _chat_completion_events(
                completion,
                emitted_text={choice: "".join(pieces) for choice, pieces in emitted.items()},
                include_usage=include_usage,
                route_frame=frame,
            )
            for event in yield_events:
                yield event
        except Exception:
            logger.exception("Error during chat streaming")
            yield _sse_event('{"error":{"message":"Chat generation failed","type":"server_error"}}')
        finally:
            closed.set()
            if future.cancel():
                ctx.abort([request_id])
            ctx._unregister_http_request(request_id, abort_event)

    return StreamingResponse(
        events(), media_type=_EVENT_STREAM,
        background=BackgroundTask(ctx._unregister_http_request, request_id, abort_event),
    )


def _encode_chat_route_frame(prompt: list[int], results: list[InferenceResponse], score_mode: LogprobMode) -> str:
    frame = _encode_binary_outputs(np.asarray(prompt, dtype=np.int32), results, score_mode)
    return base64.b64encode(frame).decode("ascii")


async def _read_generate_request(request: HttpRequest) -> tuple[np.ndarray, GenerateOptions]:
    """Read a bounded generation frame with exactly one int32 token vector."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != MEDIA_TYPE:
        raise HTTPException(status_code=415, detail=f"Expected {MEDIA_TYPE}")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Generation frame exceeds the request size limit")
        body.extend(chunk)

    try:
        metadata, tensors = decode_frame(bytes(body))
        options = GenerateOptions.model_validate(metadata)
        if tensors.keys() != {PROMPT_TOKEN_IDS}:
            raise ValueError("Generation requires exactly one prompt_token_ids tensor")
        prompt = tensors[PROMPT_TOKEN_IDS]
        if prompt.ndim != 1 or prompt.dtype != np.int32 or prompt.size == 0:
            raise ValueError("prompt_token_ids must be a nonempty one-dimensional int32 tensor")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prompt, options


def _validate_binary_options(ctx: InferenceContext, prompt: np.ndarray, options: GenerateOptions) -> None:
    if options.return_route_captures and not ctx.engine.supports_route_capture:
        raise HTTPException(status_code=400, detail="Expert route capture requires a Snowball model")
    config = ctx.engine.config
    if options.expected_logprob_mode is not None and options.expected_logprob_mode != config.logprob_mode:
        raise HTTPException(status_code=400, detail="The server is configured for a different logprob mode")
    max_prefill_size = config.max_prefill_size or config.max_seq_len
    if prompt.size + options.max_tokens > config.max_seq_len or prompt.size > max_prefill_size:
        raise HTTPException(status_code=400, detail="Prompt and generation must fit the configured context")
    if options.n > config.max_seqs:
        raise HTTPException(status_code=400, detail="Requested choices exceed the configured sequence capacity")
    vocab_size = ctx.model.Vocab.size
    if np.any((prompt < 0) | (prompt >= vocab_size)):
        raise HTTPException(status_code=400, detail="Prompt contains an out-of-vocabulary token")
    stops = options.stop_tokens
    if stops and (
        len(stops) > config.max_stop_seqs
        or any(not 0 < len(seq) <= config.max_stop_tokens for seq in stops)
        or any(not 0 <= token < vocab_size for seq in stops for token in seq)
    ):
        raise HTTPException(status_code=400, detail="Stop tokens exceed the configured capacity or vocabulary")


async def _generate_binary(ctx: InferenceContext, request: HttpRequest) -> Response:
    prompt, options = await _read_generate_request(request)
    _validate_binary_options(ctx, prompt, options)
    async with ctx.http_generation(request) as abort_event:
        future = asyncio.get_running_loop().create_future()
        ctx.submit_request(
            prompt.tolist(),
            options.max_tokens,
            options.temperature,
            options.top_p,
            options.stop_tokens,
            options.seed,
            future,
            n_generations=options.n,
            abort_event=abort_event,
            return_route_captures=options.return_route_captures,
            expected_policy_id=options.expected_policy_id,
        )
        try:
            outputs = await future
        except PolicyMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    score_mode = LogprobMode(ctx.config.service.logprob_mode)
    if options.return_route_captures:
        content = await asyncio.to_thread(_encode_binary_outputs, prompt, outputs, score_mode)
    else:
        content = _encode_binary_outputs(prompt, outputs, score_mode)
    return Response(content=content, media_type=MEDIA_TYPE)


def _encode_binary_outputs(prompt: np.ndarray, outputs: list[InferenceResponse], score_mode: LogprobMode) -> bytes:
    score_field = POLICY_LOGPROBS if score_mode == LogprobMode.POLICY else BEHAVIOR_LOGPROBS
    tensors = {PROMPT_TOKEN_IDS: prompt}
    choices = []
    for choice, output in enumerate(outputs):
        metadata: dict[str, int | str] = {
            "completion_tokens": output.completion_tokens,
            "finish_reason": format_finish_reason(output.finish_reason),
            "model_version": output.model_version,
        }
        if output.policy_id is not None:
            metadata[POLICY_ID] = output.policy_id
        choices.append(metadata)
        tensors[choice_tensor_key(choice, TOKEN_IDS)] = np.asarray(output.tokens, dtype=np.int32)
        if output.logprobs is not None:
            tensors[choice_tensor_key(choice, score_field)] = np.asarray(output.logprobs, dtype=np.float32)
        if output.expert_routes is not None:
            tensors[choice_tensor_key(choice, EXPERT_IDS)] = np.ascontiguousarray(output.expert_routes.expert_ids)
            tensors[choice_tensor_key(choice, EXPERT_VALID)] = output.expert_routes.valid
    return encode_frame({"choices": choices, LOGPROB_MODE: score_mode}, tensors)


class InferenceServer:
    """Wraps a FastAPI server around the inference context.

    Provides OpenAI compatible endpoints for text and chat completions.
    """

    _server: uvicorn.Server | None
    app: FastAPI
    config: InferenceServerConfig
    inference_context: InferenceContext

    def __init__(self, config: InferenceServerConfig, inference_context: InferenceContext, app: FastAPI):
        """Initialize the inference server with pre-built components.

        Use InferenceServer.create() to build a new server instance.
        """
        self.config = config
        self.inference_context = inference_context
        self.app = app
        self._server = None

    @staticmethod
    def create(
        config: InferenceServerConfig,
        model: LmHeadModel,
        tokenizer: MarinTokenizer,
        *,
        mesh: Mesh | None = None,
    ) -> "InferenceServer":
        """Create and initialize a new InferenceServer.

        This factory method loads the model, tokenizer, and creates all necessary
        components for the inference server.
        """
        mesh = config.trainer.device_mesh if mesh is None else mesh
        with hax.partitioning.set_mesh(mesh):
            service = InferenceEngine.from_model_with_config(
                model=model,
                tokenizer=tokenizer,
                config=config.service,
                axis_resources=config.trainer.compute_axis_mapping,
            )

        # Create and start inference thread
        inference_context = InferenceContext(model, tokenizer, service, config, mesh=mesh)
        inference_context.start()

        # Create FastAPI app with initialized context
        app = InferenceServer._create_app(inference_context)
        return InferenceServer(config, inference_context, app)

    @staticmethod
    def _create_app(inference_context: InferenceContext) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title="Levanter Inference Service", version="1.0.0")
        model_name = inference_context.config.model_name

        # Register routes with thin wrappers that call helper functions
        @app.get("/health")
        async def health_check():
            return _health_check()

        @app.post("/pause_generation")
        async def pause_generation(_request: PauseGenerationRequest):
            await asyncio.to_thread(inference_context.pause_generation)
            return CONTROL_SUCCESS

        @app.post("/resume_generation")
        async def resume_generation():
            await asyncio.to_thread(inference_context.resume_generation)
            return CONTROL_SUCCESS

        @app.post("/abort")
        async def abort(request: AbortRequest):
            inference_context.abort(request.request_ids)
            return CONTROL_SUCCESS

        @app.get("/v1/models")
        async def list_models() -> dict:
            model = Model(id=model_name, object="model", created=int(time.time()), owned_by="levanter")
            return {
                "object": "list",
                "data": [
                    {**model.model_dump(mode="json"), "max_model_len": inference_context.engine.config.max_seq_len}
                ],
            }

        # A streaming request returns a StreamingResponse, which FastAPI passes through
        # untouched; `response_model` still describes the non-streaming body.
        @app.post("/v1/chat/completions", response_model=ChatCompletion)
        async def create_chat_completion(request: ChatCompletionRequest, raw_request: HttpRequest):
            include_usage = (request.stream_options or {}).get("include_usage") is True
            tokenizer = inference_context.tokenizer
            if request.return_route_captures and (
                not request.stream
                or request.logprobs
                or not isinstance(tokenizer, HfMarinTokenizer)
                or not inference_context.engine.supports_route_capture
            ):
                raise HTTPException(status_code=400, detail="Route frames require native Snowball streaming")
            if request.stream and not request.logprobs and isinstance(tokenizer, HfMarinTokenizer):
                return await _stream_chat_completion(
                    inference_context, request, tokenizer, raw_request, include_usage=include_usage
                )
            async with inference_context.http_generation(raw_request) as abort_event:
                completion = await _create_chat_completion(inference_context, request, abort_event)
            if not request.stream:
                return completion
            events = _chat_completion_events(completion, include_usage=include_usage)
            return StreamingResponse(events, media_type=_EVENT_STREAM)

        @app.post("/v1/completions", response_model=Completion)
        async def create_completion(request: CompletionRequest, raw_request: HttpRequest):
            async with inference_context.http_generation(raw_request) as abort_event:
                completion = await _create_completion(inference_context, request, abort_event)
            if not request.stream:
                return completion
            return StreamingResponse(_completion_events(completion), media_type=_EVENT_STREAM)

        @app.post("/v1/tokens", response_model=TokensResponse)
        async def fetch_tokens(request: TokensRequest) -> TokensResponse:
            return await _fetch_tokens(inference_context, request)

        @app.post(GENERATE_PATH)
        async def generate_binary(request: HttpRequest) -> Response:
            return await _generate_binary(inference_context, request)

        return app

    def unload(self):
        """Unload the inference model to free up resources."""
        self.inference_context.unload()

    def publish(self, restore: WeightSource, *, expected_version: int, policy_id: str) -> int:
        """Prepare and install a complete policy snapshot."""
        return self.inference_context.publish(restore, expected_version=expected_version, policy_id=policy_id)

    def reload(
        self, weight_callback: WeightSource, *, expected_version: int | None = None, policy_id: str | None = None
    ) -> int:
        """Install the weights and immutable policy identity in one serving version."""
        return self.inference_context.reload(weight_callback, expected_version=expected_version, policy_id=policy_id)

    def address(self):
        """Get the full address the server is running on."""
        for server in self._server.servers:
            for sock in server.sockets:
                addr = sock.getsockname()
                host, port = addr[0], addr[1]
                # handle weird ipv6 localhost address which confuses clients
                if host == "::1" or host == ":1":
                    host = "localhost"
                return f"{host}:{port}"
        return None

    def port(self):
        """Get the port the server is running on."""
        if self.config.port > 0:
            return self.config.port

        # query the uvicorn server socket list for the port
        for server in self._server.servers:
            for sock in server.sockets:
                addr = sock.getsockname()
                return addr[1]

        return None

    def serve(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            self._server = uvicorn.Server(uvicorn.Config(self.app, host=self.config.host, port=self.config.port))
            self._server.run()
        finally:
            self.shutdown()

    async def serve_async(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            config = uvicorn.Config(self.app, host=self.config.host, port=self.config.port)
            self._server = uvicorn.Server(config)
            await self._server.serve()
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the inference context."""
        self.inference_context.shutdown()
