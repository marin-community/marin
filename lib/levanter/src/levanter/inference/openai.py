# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI-compatible inference API for Levanter models.

This module provides FastAPI-based endpoints that are compatible with OpenAI's
completions and chat completions APIs, allowing Levanter models to be used as
drop-in replacements for OpenAI models.
"""

import asyncio
import collections
import collections.abc
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Union, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send
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
from levanter.inference.engine import (
    DecodeResult,
    InferenceEngine,
    InferenceEngineConfig,
    Request,
    TokenSequenceLogprobs,
)
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.utils import INVALID
from levanter.inference.weight_reload import WeightTransferConfig
from levanter.inference.weight_reload_http import add_weight_reload_routes
from levanter.inference.openai_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ChatTokenizeRequest,
    ChatTokenizeResponse,
    CompletionRequest,
    PauseGenerationRequest,
    TokenList,
    TokensRequest,
    TokensResponse,
)
from levanter.models.lm_model import LmHeadModel
from levanter.tokenizers import MarinTokenizer
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "levanter"
TOKEN_ID_PREFIX = "token_id:"
RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {"add_generation_prompt", "continue_final_message", "chat_template", "return_dict", "tokenize"}
)


@dataclass
class InferenceServerConfig:
    """Configuration for OpenAI-compatible inference server."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(4096))

    model_name: str = DEFAULT_MODEL_NAME
    """Model id this server advertises from ``/v1/models``."""

    # Default generation parameters for API
    temperature: float = 0.7
    seed: int = 42

    weight_transfer: WeightTransferConfig | None = None

    batch_timeout: float = 0.1  # seconds to wait for more requests before processing batch

    host: str = "localhost"
    port: int = 0  # auto-assign port


@dataclass(frozen=True)
class InferenceDelta:
    """New output for one choice, delivered at a host decode boundary."""

    index: int
    text: str
    tokens: list[int]
    logprobs: list[float]
    finish_reason: FinishReason
    model_version: int
    prompt_tokens: list[int]


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
    cancel_event: threading.Event | None = None
    on_delta: collections.abc.Callable[[InferenceDelta], None] | None = None
    admission_epoch: int = 0
    n_generations: int = 1
    echo_logprobs_top_k: int | None = None


@dataclass
class InferenceResponse:
    """Internal response structure for the inference thread"""

    request_id: str
    text: str
    tokens: List[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason
    model_version: int
    logprobs: Optional[List[float]] = None
    echo_token_ids: List[int] | None = None
    echo_logprobs: TokenSequenceLogprobs | None = None


def _complete_future(future: asyncio.Future, outcome: list[InferenceResponse] | Exception) -> None:
    if future.done():
        return
    if isinstance(outcome, Exception):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)


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


def _encode_stop_tokens(stop: Union[str, List[str], None], tokenizer: MarinTokenizer) -> Optional[List[List[int]]]:
    """Tokenize each stop string as an independent sequence."""
    if not stop:
        return None
    stop_list = [stop] if isinstance(stop, str) else stop
    stop_tokens: List[List[int]] = []
    for s in stop_list:
        stop_ids = tokenizer.encode(s, add_special_tokens=False)
        if stop_ids:
            stop_tokens.append(stop_ids)
    return stop_tokens


class InferenceContext:
    """Background thread that manages the InferenceEngine and processes requests"""

    def __init__(self, model: LmHeadModel, tokenizer, engine: InferenceEngine, config: InferenceServerConfig):
        self.model = model
        self.model_version = 0
        self.tokenizer = tokenizer
        self.engine = engine
        self.config = config
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.batch_queue: queue.Queue[InferenceBatch] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
        self.admission_lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.pause_event = threading.Event()
        self.admission_epoch = 0
        self.active_requests: dict[str, threading.Event] = {}
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self._next_request_id = 0

    def start(self):
        """Start the inference and batch processing threads"""
        logger.info("Starting inference context...")
        self.inference_thread.start()
        self.batch_thread.start()

    def shutdown(self):
        """Signal shutdown and wait for threads to finish"""
        logger.info("Shutting down inference context.")
        self.shutdown_event.set()
        self.inference_thread.join(timeout=1)
        self.batch_thread.join(timeout=1)

    def unload(self):
        """Unload the inference model to free up resources."""
        logger.info("Unloading inference model...")
        with self.model_lock:
            # pyrefly: ignore[bad-assignment]  # unload() deliberately clears the model; reload() restores it
            self.model = None
            self.engine = None  # type: ignore[assignment]
        logger.info("Inference model unloaded.")

    def pause_generation(self) -> None:
        """Abort unfinished requests and clear serving state before returning."""
        with self.lifecycle_lock:
            with self.admission_lock:
                self.pause_event.set()
                self.admission_epoch += 1
            # The active batch observes pause_event between device decode rounds.
            with self.model_lock, self.config.trainer.use_device_mesh():
                self.engine.reset()

    def resume_generation(self) -> None:
        """Allow new requests after a completed pause or weight replacement."""
        with self.lifecycle_lock, self.admission_lock:
            self.pause_event.clear()

    def reset_prefix_cache(self) -> None:
        """Abort active requests and clear cached state without changing model weights."""
        with self.lifecycle_lock:
            was_paused = self.pause_event.is_set()
            self.pause_generation()
            if not was_paused:
                self.resume_generation()

    def reload(self, weight_callback: WeightSource, *, expected_version: int) -> int:
        """Stage same-architecture weights, then atomically install a new serving version.

        The callback must return new weights without donating or mutating the current model.
        Staging failures preserve the current model, version, and admission state.
        An already paused context remains paused after installation.
        """
        with self.admission_lock:
            if expected_version != self.model_version:
                raise ValueError(f"Expected model version {expected_version}, serving {self.model_version}")
            current_model = self.model
        with (
            self.config.trainer.use_device_mesh(),
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            candidate = weight_callback(current_model)
            current_arrays = eqx.filter(current_model, eqx.is_array)
            candidate_arrays = eqx.filter(candidate, eqx.is_array)
            if jax.tree.structure(current_arrays) != jax.tree.structure(candidate_arrays):
                raise ValueError("Replacement weights must preserve model architecture")
            for old, new in zip(jax.tree.leaves(current_arrays), jax.tree.leaves(candidate_arrays), strict=True):
                if (old.shape, old.dtype, old.sharding) != (new.shape, new.dtype, new.sharding):
                    raise ValueError("Replacement weights must preserve shape, dtype, and sharding")
            jax.block_until_ready(candidate)

        with self.lifecycle_lock:
            with self.admission_lock:
                if expected_version != self.model_version:
                    raise ValueError(f"Expected model version {expected_version}, serving {self.model_version}")
            was_paused = self.pause_event.is_set()
            self.pause_generation()
            try:
                with self.model_lock, self.admission_lock:
                    jax.block_until_ready(self.engine.gen_state)
                    self.model = candidate
                    self.engine.model = candidate
                    self.model_version += 1
                    installed_version = self.model_version
            finally:
                if not was_paused:
                    self.resume_generation()
        return installed_version

    def abort(self, request_ids: collections.abc.Sequence[str]) -> None:
        """Cancel only the named HTTP request groups at the next host decode boundary."""
        with self.admission_lock:
            for request_id in request_ids:
                event = self.active_requests.get(request_id)
                if event is not None:
                    event.set()

    def submit_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        stop_tokens: Optional[List[List[int]]],
        seed: int | None,
        future: asyncio.Future,
        n_generations: int = 1,
        echo_logprobs_top_k: int | None = None,
        cancel_event: threading.Event | None = None,
        on_delta: collections.abc.Callable[[InferenceDelta], None] | None = None,
    ) -> str:
        """Submit a request to the inference queue"""
        assert self.shutdown_event.is_set() is False, "InferenceContext is shut down"
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
            n_generations=n_generations,
            echo_logprobs_top_k=echo_logprobs_top_k,
            cancel_event=cancel_event,
            on_delta=on_delta,
        )

        logger.info("Enqueuing request %s", request)
        with self.admission_lock:
            request.admission_epoch = self.admission_epoch
            if self.pause_event.is_set():
                self._abort_request(request)
            else:
                self.request_queue.put(request)
        return request_id

    def _abort_request(self, request: InferenceRequest) -> None:
        if request.future.cancelled():
            return
        responses = [
            InferenceResponse(
                request_id=request.request_id,
                text="",
                tokens=[],
                prompt_tokens=len(request.prompt_tokens),
                completion_tokens=0,
                finish_reason=FinishReason.ABORT,
                model_version=self.model_version,
                logprobs=[],
            )
            for _ in range(request.n_generations)
        ]
        if request.on_delta is not None:
            for index in range(request.n_generations):
                delta = InferenceDelta(
                    index, "", [], [], FinishReason.ABORT, self.model_version, request.prompt_tokens
                )
                request.future.get_loop().call_soon_threadsafe(request.on_delta, delta)
        request.future.get_loop().call_soon_threadsafe(_complete_future, request.future, responses)

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
            logger.info(f"Max tokens per seq: {max_tokens_per_seq}, per batch: {max_tokens_per_batch}")

            for r in requests:
                if len(r.prompt_tokens) > max_tokens_per_seq:
                    r.future.get_loop().call_soon_threadsafe(
                        _complete_future, r.future, ValueError("Prompt exceeds the serving context limit")
                    )
                    continue

                if r.n_generations > self.engine.config.max_seqs:
                    # fail requests that are too large
                    error_msg = (
                        f"Request {r.request_id} has n={r.n_generations} which exceeds "
                        f"the maximum allowed {self.engine.config.max_seqs}"
                    )
                    logger.error(error_msg)
                    r.future.get_loop().call_soon_threadsafe(_complete_future, r.future, ValueError(error_msg))
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

    def _batch_processing_loop(self):
        """Batch processing loop running in background thread - waits for batches and executes them"""
        logger.info("Batch processing thread started")

        while not self.shutdown_event.is_set():
            try:
                batch = self.batch_queue.get(timeout=1)
                with (
                    self.model_lock,
                    hax.partitioning.set_mesh(self.config.trainer.device_mesh),
                    hax.axis_mapping(self.config.trainer.compute_axis_mapping),
                ):
                    self._execute_batch(batch)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error executing batch: {e}", exc_info=True)
                # Set exceptions on all futures in the batch
                for req in batch:
                    try:
                        req.future.get_loop().call_soon_threadsafe(_complete_future, req.future, e)
                    except Exception:
                        pass

        logger.info("Batch processing thread shutting down")

    def _execute_batch(self, requests: InferenceBatch):
        """Execute a batch of inference requests"""
        admitted = InferenceBatch()
        for request in requests:
            if (
                self.pause_event.is_set()
                or request.admission_epoch != self.admission_epoch
                or (request.cancel_event is not None and request.cancel_event.is_set())
            ):
                self._abort_request(request)
            else:
                admitted.append(request)
        requests = admitted
        if not requests:
            return
        service_requests = []

        if not self.engine:
            raise RuntimeError("Inference engine is not initialized.")

        for i, req in enumerate(requests):
            # Create stop tokens if specified
            stop_ids = None
            if req.stop_tokens:
                max_stop_length = max(map(len, req.stop_tokens))
                padded_stops = np.full((len(req.stop_tokens), max_stop_length), INVALID, dtype=np.int32)
                for index, stop in enumerate(req.stop_tokens):
                    padded_stops[index, -len(stop) :] = stop
                stop_ids = hax.named(jnp.asarray(padded_stops), axis=("stop_seq", "position"))

            # dumb fallback seed if none provided
            if req.seed is None:
                req.seed = np.random.default_rng().integers(0, 2**32 - 1)

            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(req.prompt_tokens) + req.max_tokens, dtype=jnp.int32),
                stop_tokens=stop_ids,
                temperature=jnp.array(req.temperature, dtype=jnp.float32),
                top_p=jnp.array(1.0 if req.top_p is None else req.top_p, dtype=jnp.float32),
                key=jrandom.PRNGKey(req.seed if req.seed is not None else i),
            )

            service_req = Request(
                prompt_tokens=req.prompt_tokens,
                request_id=i,  # Use batch index as service request id
                decode_params=seq_params,
                n_generations=req.n_generations,
            )
            service_requests.append(service_req)

        # Generate responses
        start_time = time.time()

        def should_abort(index: int) -> bool:
            event = requests[index].cancel_event
            return self.pause_event.is_set() or (event is not None and event.is_set())

        published: dict[tuple[int, int], tuple[int, str, FinishReason]] = {}
        completed: set[int] = set()

        def publish(index: int, choices: list[DecodeResult]) -> None:
            if index in completed:
                return
            req = requests[index]
            if req.future.cancelled():
                completed.add(index)
                return
            try:
                for choice in choices if req.on_delta is not None else []:
                    count, previous_text, reason = published.get((index, choice.choice), (0, "", FinishReason.RUNNING))
                    if count == len(choice.token_list) and reason == choice.finish_reason:
                        continue
                    text = self.tokenizer.decode(choice.token_list, skip_special_tokens=True)
                    # Byte-level tokenizers may end an unfinished Unicode character with U+FFFD.
                    stable_text = text if choice.done else text.rstrip("\ufffd")
                    if not stable_text.startswith(previous_text):
                        raise ValueError("Tokenizer changed already-emitted text during incremental decoding")
                    delta = InferenceDelta(
                        choice.choice,
                        stable_text[len(previous_text) :],
                        choice.token_list[count:].copy(),
                        choice.logprobs[count:].copy(),
                        choice.finish_reason,
                        self.model_version,
                        req.prompt_tokens,
                    )
                    req.future.get_loop().call_soon_threadsafe(req.on_delta, delta)
                    published[index, choice.choice] = (len(choice.token_list), stable_text, choice.finish_reason)
                if not all(choice.done for choice in choices):
                    return
                responses = []
                for choice in choices:
                    tokens = choice.token_list.copy()
                    echo_tokens = req.prompt_tokens + tokens if req.echo_logprobs_top_k is not None else None
                    echo_logprobs = (
                        self.engine.score_token_logprobs(echo_tokens, req.echo_logprobs_top_k)
                        if echo_tokens is not None
                        else None
                    )
                    responses.append(
                        InferenceResponse(
                            request_id=req.request_id,
                            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                            tokens=tokens,
                            logprobs=choice.logprobs.copy(),
                            prompt_tokens=len(req.prompt_tokens),
                            completion_tokens=len(tokens),
                            finish_reason=choice.finish_reason,
                            model_version=self.model_version,
                            echo_token_ids=echo_tokens,
                            echo_logprobs=echo_logprobs,
                        )
                    )
                completed.add(index)
                req.future.get_loop().call_soon_threadsafe(_complete_future, req.future, responses)
            except Exception as error:
                completed.add(index)
                if req.cancel_event is not None:
                    req.cancel_event.set()
                logger.exception("Error publishing output for request %s", req.request_id)
                req.future.get_loop().call_soon_threadsafe(_complete_future, req.future, error)

        result = self.engine.generate(service_requests, should_abort=should_abort, output_callback=publish)
        duration = time.time() - start_time
        logger.info(f"Batch completed in {duration:.2f}s, generated {result.total_generated} tokens")


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


def _sse_event(payload: str) -> str:
    return f"data: {payload}\n\n"


def _completion_events(completion: Completion) -> collections.abc.Iterator[str]:
    """Render a finished text completion as OpenAI server-sent events."""
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
    *,
    return_tokens_as_token_ids: bool = False,
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
            (
                f"{TOKEN_ID_PREFIX}{token_id}"
                if return_tokens_as_token_ids
                else tokenizer.decode([token_id], skip_special_tokens=False)
            ): logprob
            for token_id, logprob in token_logprobs.items()
        }
        for token_logprobs in sequence_logprobs.top_token_logprobs
    ]
    return CompletionLogprobData(
        tokens=[f"{TOKEN_ID_PREFIX}{token_id}" for token_id in token_ids] if return_tokens_as_token_ids else tokens,
        token_logprobs=sequence_logprobs.token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=_token_text_offsets(tokens),
    )


def _completion_prompt_ids(
    prompt: str | list[str] | list[int] | list[list[int]],
    tokenizer: MarinTokenizer,
) -> list[list[int]]:
    if isinstance(prompt, str):
        return [tokenizer.encode(prompt, add_special_tokens=False)]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must contain at least one token")
    if isinstance(prompt[0], int):
        return [cast(list[int], prompt)]
    if isinstance(prompt[0], str):
        return [tokenizer.encode(row, add_special_tokens=False) for row in cast(list[str], prompt)]
    return cast(list[list[int]], prompt)


def _validate_prompt_ids(ctx: InferenceContext, tokens: list[int]) -> None:
    if not tokens or len(tokens) >= ctx.config.service.max_seq_len:
        raise HTTPException(status_code=400, detail="Prompt must be nonempty and leave room within the context limit")
    if min(tokens) < 0 or max(tokens) >= ctx.model.Vocab.size:
        raise HTTPException(status_code=400, detail="Prompt contains a token ID outside the model vocabulary")


async def _create_completion(
    ctx: InferenceContext,
    request: CompletionRequest,
    cancel_event: threading.Event | None = None,
    updates: asyncio.Queue[InferenceDelta] | None = None,
) -> Completion:
    """Create a text completion using OpenAI API format."""
    try:
        if request.echo and request.logprobs == 0:
            raise HTTPException(status_code=400, detail="Echo logprobs require a positive logprobs count")
        prompt_token_lists = _completion_prompt_ids(request.prompt, ctx.tokenizer)

        stop_tokens = _encode_stop_tokens(request.stop, ctx.tokenizer)
        if request.stop_token_ids:
            stop_tokens = [*(stop_tokens or []), *([token] for token in request.stop_token_ids)]

        # Create futures for all prompts
        futures = []
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        echo_logprobs_top_k = int(request.logprobs) if request.echo and request.logprobs else None

        for prompt_tokens in prompt_token_lists:
            scored_token_budget = len(prompt_tokens) + request.max_tokens
            if echo_logprobs_top_k is not None and scored_token_budget > ctx.config.service.max_seq_len:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "echo logprobs are not supported when prompt tokens plus max_tokens exceeds "
                        f"max_seq_len={ctx.config.service.max_seq_len}"
                    ),
                )
            _validate_prompt_ids(ctx, prompt_tokens)
            total_prompt_tokens += len(prompt_tokens)

        for prompt_index, prompt_tokens in enumerate(prompt_token_lists):
            on_delta = None
            if updates is not None:
                offset = prompt_index * (request.n or 1)

                def on_delta(delta: InferenceDelta, offset=offset) -> None:
                    updates.put_nowait(replace(delta, index=offset + delta.index))

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
                cancel_event=cancel_event,
                on_delta=on_delta,
            )

        # Wait for all results
        results: List[List[InferenceResponse]] = await asyncio.gather(*futures)

        # Format responses
        choice_idx = 0
        for prompt_index, (prompt_tokens, result) in enumerate(zip(prompt_token_lists, results, strict=True)):
            if isinstance(request.prompt, str):
                prompt_text = request.prompt
            elif isinstance(request.prompt[0], str):
                prompt_text = cast(list[str], request.prompt)[prompt_index]
            else:
                prompt_text = ctx.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            for generation in result:
                choice_text = prompt_text + generation.text if request.echo else generation.text

                # Format logprobs if available
                logprobs = None
                echo_unstarted = (
                    request.echo and generation.finish_reason == FinishReason.ABORT and not generation.tokens
                )
                if request.logprobs is not None and not echo_unstarted:
                    if request.echo:
                        if generation.echo_token_ids is None or generation.echo_logprobs is None:
                            raise RuntimeError("Echo logprobs requested but missing from generation result.")
                        echo_logprobs = _completion_logprobs(
                            ctx.tokenizer,
                            generation.echo_token_ids,
                            generation.echo_logprobs,
                            return_tokens_as_token_ids=request.return_tokens_as_token_ids,
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
                                token_str = (
                                    f"{TOKEN_ID_PREFIX}{token_id}"
                                    if request.return_tokens_as_token_ids
                                    else ctx.tokenizer.convert_ids_to_tokens(token_id)
                                )
                                tokens.append(token_str)
                                token_logprobs.append(float(lp))

                        logprobs = Logprobs(
                            tokens=tokens,
                            token_logprobs=token_logprobs,
                            text_offset=None,
                            top_logprobs=None,
                        )

                choices.append(
                    CompletionChoice(
                        text=choice_text,
                        index=choice_idx,
                        finish_reason="stop" if generation.finish_reason == FinishReason.STOP else "length",
                        logprobs=logprobs,
                    )
                )
                choices[-1] = choices[-1].model_copy(update={"model_version": generation.model_version})
                if generation.finish_reason == FinishReason.ABORT:
                    choices[-1] = choices[-1].model_copy(update={"finish_reason": "abort"})
                if request.return_token_ids:
                    choices[-1] = choices[-1].model_copy(
                        update={
                            "token_ids": generation.tokens,
                            "prompt_token_ids": prompt_tokens,
                        }
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
    *,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
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
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
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


def _chat_token_logprobs(
    tokenizer: MarinTokenizer,
    tokens: list[int],
    logprobs: list[float],
    return_tokens_as_token_ids: bool,
) -> list[ChatCompletionTokenLogprob]:
    content = []
    for token_id, logprob in zip(tokens, logprobs, strict=True):
        token = (
            f"{TOKEN_ID_PREFIX}{token_id}" if return_tokens_as_token_ids else tokenizer.convert_ids_to_tokens(token_id)
        )
        content.append(
            ChatCompletionTokenLogprob(
                token=token,
                logprob=float(logprob),
                bytes=list(token.encode("utf-8")),
                top_logprobs=[],
            )
        )
    return content


async def _create_chat_completion(
    ctx: InferenceContext,
    request: ChatCompletionRequest,
    cancel_event: threading.Event | None = None,
    updates: asyncio.Queue[InferenceDelta] | None = None,
) -> ChatCompletion:
    """Create a chat completion using OpenAI API format."""
    try:
        # Convert Pydantic models to dicts for tokenizer
        if request.top_logprobs:
            raise HTTPException(status_code=400, detail="Generated top-logprob candidates are not supported")
        prompt_tokens = request.exact_prompt_token_ids
        if prompt_tokens is None:
            prompt_tokens = _compute_tokens(
                request.messages,
                ctx.tokenizer,
                request.tools,
                request.chat_template_kwargs,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
            )
        _validate_prompt_ids(ctx, prompt_tokens)

        stop_tokens = _encode_stop_tokens(request.stop, ctx.tokenizer)
        if request.stop_token_ids:
            stop_tokens = [*(stop_tokens or []), *([token] for token in request.stop_token_ids)]

        # Create future and submit request
        future: asyncio.Future = asyncio.Future()
        ctx.submit_request(
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_tokens=stop_tokens,
            seed=request.seed,
            future=future,
            n_generations=request.n or 1,
            cancel_event=cancel_event,
            on_delta=updates.put_nowait if updates is not None else None,
        )

        # Wait for result
        results: List[InferenceResponse] = await future

        # Format response
        choices = []
        total_completion_tokens = 0

        for i, generation in enumerate(results):
            # Format logprobs if available
            logprobs = None
            if request.logprobs:
                assert generation.logprobs is not None, "Logprobs requested but missing in generation result"
                logprobs = ChoiceLogprobs(
                    content=_chat_token_logprobs(
                        ctx.tokenizer,
                        generation.tokens,
                        generation.logprobs,
                        request.return_tokens_as_token_ids,
                    )
                )

            choices.append(
                ChatCompletionChoice(
                    index=i,
                    message=ChatCompletionMessage(role="assistant", content=generation.text),
                    finish_reason="stop" if generation.finish_reason == FinishReason.STOP else "length",
                    logprobs=logprobs,
                )
            )
            if generation.finish_reason == FinishReason.ABORT:
                choices[-1] = choices[-1].model_copy(update={"finish_reason": "abort"})
            if request.return_token_ids:
                choices[-1] = choices[-1].model_copy(update={"token_ids": generation.tokens})
            total_completion_tokens += generation.completion_tokens

        response = ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=total_completion_tokens,
                total_tokens=len(prompt_tokens) + total_completion_tokens,
            ),
        )
        response = response.model_copy(update={"model_version": results[0].model_version})
        if request.return_token_ids:
            response = response.model_copy(update={"prompt_token_ids": prompt_tokens})
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _delta_events(
    delta: InferenceDelta,
    request: CompletionRequest | ChatCompletionRequest,
    tokenizer: MarinTokenizer,
    response_id: str,
    created: int,
    first: bool,
) -> collections.abc.Iterator[str]:
    finish_reason = {
        FinishReason.RUNNING: None,
        FinishReason.STOP: "stop",
        FinishReason.LENGTH: "length",
        FinishReason.ABORT: "abort",
    }[delta.finish_reason]
    extra: dict[str, Any] = {"model_version": delta.model_version}
    if request.return_token_ids and first:
        extra["prompt_token_ids"] = delta.prompt_tokens
    if isinstance(request, ChatCompletionRequest):
        logprobs = None
        if request.logprobs:
            logprobs = ChunkChoiceLogprobs(
                content=_chat_token_logprobs(
                    tokenizer,
                    delta.tokens,
                    delta.logprobs,
                    request.return_tokens_as_token_ids,
                )
            )
        content = ChatCompletionChunkChoice(
            index=delta.index,
            delta=ChoiceDelta(role="assistant" if first else None, content=delta.text),
            logprobs=logprobs,
        )
        if request.return_token_ids:
            content = content.model_copy(update={"token_ids": delta.tokens})
        choices = [content] if first or delta.tokens or delta.text else []
        if finish_reason is not None:
            choices.append(
                ChatCompletionChunkChoice(index=delta.index, delta=ChoiceDelta()).model_copy(
                    update={"finish_reason": finish_reason}
                )
            )
        for choice in choices:
            chunk = ChatCompletionChunk(
                id=response_id,
                object="chat.completion.chunk",
                created=created,
                model=request.model,
                choices=[choice],
            ).model_copy(update=extra)
            yield _sse_event(chunk.model_dump_json())
    else:
        logprobs = None
        if request.logprobs is not None:
            token_text = [
                (
                    f"{TOKEN_ID_PREFIX}{token}"
                    if request.return_tokens_as_token_ids
                    else tokenizer.convert_ids_to_tokens(token)
                )
                for token in delta.tokens
            ]
            logprobs = Logprobs(tokens=token_text, token_logprobs=delta.logprobs, text_offset=None, top_logprobs=None)
        content = CompletionChoice(index=delta.index, text=delta.text, finish_reason="length", logprobs=logprobs)
        choice_extra: dict[str, Any] = {**extra, "finish_reason": finish_reason}
        if request.return_token_ids:
            choice_extra["token_ids"] = delta.tokens
        content = content.model_copy(update=choice_extra)
        chunk = Completion(
            id=response_id, object="text_completion", created=created, model=request.model, choices=[content]
        )
        yield _sse_event(chunk.model_dump_json())


class _GenerationStreamingResponse(StreamingResponse):
    def __init__(self, content, cleanup: collections.abc.Callable[[], collections.abc.Awaitable[None]]):
        super().__init__(content, media_type="text/event-stream")
        self.cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Own cleanup even if sending headers fails before the body iterator starts.
            await self.cleanup()


async def _http_generation[RequestT: (
    CompletionRequest,
    PauseGenerationRequest,
    ChatCompletionRequest,
), ResponseT: (Completion, ChatCompletion)](
    ctx: InferenceContext,
    raw_request: HttpRequest,
    create_completion: collections.abc.Callable[
        [InferenceContext, RequestT, threading.Event | None, asyncio.Queue[InferenceDelta] | None],
        collections.abc.Coroutine[Any, Any, ResponseT],
    ],
    request: RequestT,
) -> (ResponseT | StreamingResponse):
    request_id = raw_request.headers.get("x-request-id") or uuid.uuid4().hex
    event = threading.Event()
    with ctx.admission_lock:
        if request_id in ctx.active_requests:
            raise HTTPException(status_code=409, detail="Request ID is already active")
        ctx.active_requests[request_id] = event

    async def wait_for_disconnect() -> None:
        while (await raw_request.receive())["type"] != "http.disconnect":
            pass
        event.set()

    # Echo logprobs rescore the complete sequence and retain their completed-response path.
    incremental = request.stream and not (isinstance(request, CompletionRequest) and request.echo)
    updates: asyncio.Queue[InferenceDelta] = asyncio.Queue()
    generation = asyncio.create_task(create_completion(ctx, request, event, updates if incremental else None))
    disconnect = asyncio.create_task(wait_for_disconnect())

    async def cleanup() -> None:
        event.set()
        disconnect.cancel()
        generation.cancel()
        await asyncio.gather(generation, disconnect, return_exceptions=True)
        with ctx.admission_lock:
            del ctx.active_requests[request_id]

    async def stream(first_delta: InferenceDelta | None) -> collections.abc.AsyncIterator[str]:
        response_id = ("chatcmpl-" if isinstance(request, ChatCompletionRequest) else "cmpl-") + uuid.uuid4().hex[:8]
        created = int(time.time())
        seen: set[int] = set()
        next_update = None
        try:
            delta = first_delta
            while True:
                if delta is not None:
                    for chunk in _delta_events(
                        delta, request, ctx.tokenizer, response_id, created, delta.index not in seen
                    ):
                        yield chunk
                    seen.add(delta.index)
                if not updates.empty():
                    delta = updates.get_nowait()
                    continue
                if generation.done():
                    await generation
                    break
                next_update = asyncio.create_task(updates.get())
                done, _ = await asyncio.wait((next_update, generation), return_when=asyncio.FIRST_COMPLETED)
                if next_update in done:
                    delta = next_update.result()
                else:
                    next_update.cancel()
                    await asyncio.gather(next_update, return_exceptions=True)
                    delta = None if next_update.cancelled() else next_update.result()
            yield _sse_event("[DONE]")
        finally:
            if next_update is not None:
                next_update.cancel()
                await asyncio.gather(next_update, return_exceptions=True)

    first_update = asyncio.create_task(updates.get()) if incremental else None
    try:
        waiting = [generation, disconnect]
        if first_update is not None:
            waiting.append(first_update)
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            await disconnect
            raise asyncio.CancelledError
        if incremental:
            first_delta = first_update.result() if first_update in done else None
            if generation in done:
                await generation  # Preserve validation errors before sending HTTP headers.
            if first_delta is None:
                first_update.cancel()
                await asyncio.gather(first_update, return_exceptions=True)
                first_delta = None if first_update.cancelled() else first_update.result()
            # StreamingResponse owns disconnect handling once response headers are sent.
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)
            return _GenerationStreamingResponse(stream(first_delta), cleanup)
        completion = await generation
        await cleanup()
        if request.stream:
            return StreamingResponse(_completion_events(cast(Completion, completion)), media_type="text/event-stream")
        return completion
    except BaseException:
        if first_update is not None:
            first_update.cancel()
            await asyncio.gather(first_update, return_exceptions=True)
        await cleanup()
        raise


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
    def create(config: InferenceServerConfig, model: LmHeadModel, tokenizer: MarinTokenizer) -> "InferenceServer":
        """Create and initialize a new InferenceServer.

        This factory method loads the model, tokenizer, and creates all necessary
        components for the inference server.
        """
        service = InferenceEngine.from_model_with_config(
            model=model,
            tokenizer=tokenizer,
            config=config.service,
            axis_resources=config.trainer.compute_axis_mapping,
        )

        # Create and start inference thread
        inference_context = InferenceContext(model, tokenizer, service, config)
        inference_context.start()

        # Create FastAPI app with initialized context
        app = InferenceServer._create_app(inference_context)
        return InferenceServer(config, inference_context, app)

    @staticmethod
    def _create_app(inference_context: InferenceContext) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title="Levanter Inference Service", version="1.0.0")
        model_name = inference_context.config.model_name
        if inference_context.config.weight_transfer is not None:
            add_weight_reload_routes(
                app, inference_context, inference_context.config.trainer, inference_context.config.weight_transfer
            )

        # Register routes with thin wrappers that call helper functions
        @app.get("/health")
        async def health_check():
            return _health_check()

        @app.post("/pause_generation")
        async def pause_generation(request: PauseGenerationRequest):
            if request.mode != "abort" or not request.clear_cache:
                raise HTTPException(400, "Native serving requires mode=abort and clear_cache=true")
            await asyncio.to_thread(inference_context.pause_generation)
            return {"status": "ok"}

        @app.post("/resume_generation")
        async def resume_generation():
            await asyncio.to_thread(inference_context.resume_generation)
            return {"status": "ok"}

        @app.get("/v1/models")
        async def list_models() -> dict:
            model = Model(id=model_name, object="model", created=int(time.time()), owned_by="levanter")
            return {"object": "list", "data": [model.model_dump(mode="json")]}

        # A streaming request returns a StreamingResponse, which FastAPI passes through
        # untouched; `response_model` still describes the non-streaming body.
        @app.post("/v1/chat/completions", response_model=ChatCompletion)
        async def create_chat_completion(request: ChatCompletionRequest, raw_request: HttpRequest):
            return await _http_generation(inference_context, raw_request, _create_chat_completion, request)

        @app.post("/v1/completions", response_model=Completion)
        async def create_completion(request: CompletionRequest, raw_request: HttpRequest):
            return await _http_generation(inference_context, raw_request, _create_completion, request)

        @app.post("/tokenize", response_model=ChatTokenizeResponse)
        async def tokenize_chat(request: ChatTokenizeRequest) -> ChatTokenizeResponse:
            tokens = _compute_tokens(
                request.messages,
                inference_context.tokenizer,
                request.tools,
                request.chat_template_kwargs,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
            )
            return ChatTokenizeResponse(
                tokens=tokens, count=len(tokens), max_model_len=inference_context.config.service.max_seq_len
            )

        @app.post("/v1/tokens", response_model=TokensResponse)
        async def fetch_tokens(request: TokensRequest) -> TokensResponse:
            return await _fetch_tokens(inference_context, request)

        return app

    def abort(self, request_ids: collections.abc.Sequence[str]) -> None:
        """Cancel named requests while allowing unrelated requests to finish."""
        self.inference_context.abort(request_ids)

    def pause_generation(self) -> None:
        """Abort current requests and clear cache state before weight replacement."""
        self.inference_context.pause_generation()

    def resume_generation(self) -> None:
        """Resume request admission after the pause barrier."""
        self.inference_context.resume_generation()

    def unload(self):
        """Unload the inference model to free up resources."""
        self.inference_context.unload()

    @property
    def model_version(self) -> int:
        return self.inference_context.model_version

    def reload(self, weight_callback: WeightSource, *, expected_version: int) -> int:
        """Install staged weights and return the new version after clearing serving state."""
        return self.inference_context.reload(weight_callback, expected_version=expected_version)

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
