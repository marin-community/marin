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
from collections import deque
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Union, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
from tokenizers.decoders import DecodeStream
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from levanter.inference.engine import (
    GenerationResult,
    InferenceEngine,
    InferenceEngineConfig,
    Request,
    TokenSequenceLogprobs,
)
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.page_table import kv_pages_for_request
from levanter.inference.utils import INVALID
from levanter.inference.openai_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ChatTokenizeRequest,
    ChatTokenizeResponse,
    CompletionRequest,
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

    batch_timeout: float = 0.1  # seconds to wait for more requests before processing batch

    host: str = "localhost"
    port: int = 0  # auto-assign port


ChatTokenCallback = collections.abc.Callable[[int, list[int], int], None]


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
    admission_epoch: int = 0
    n_generations: int = 1
    echo_logprobs_top_k: int | None = None
    on_tokens: ChatTokenCallback | None = None


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


class InferenceBatch(list):
    def num_seqs(self) -> int:
        return sum(req.n_generations for req in self)


# A callback which replaces the current model.
WeightSource = collections.abc.Callable[[LmHeadModel], LmHeadModel]


def _fetch_from_queue(q: queue.Queue[InferenceRequest], timeout: float, max_requests: int) -> list[InferenceRequest]:
    """Collect requests until the batch fills or the timeout expires."""
    deadline = time.monotonic() + timeout
    items = []
    while len(items) < max_requests and (remaining := deadline - time.monotonic()) > 0:
        try:
            item = q.get(timeout=remaining)
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
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self._next_request_id = 0

    def start(self):
        """Start the inference and batch processing threads"""
        logger.info("Starting inference context...")
        self.inference_thread.start()
        self.batch_thread.start()

    def shutdown(self):
        """Stop accepting batches and wait for active device work to drain."""
        logger.info("Shutting down inference context.")
        self.shutdown_event.set()
        self.inference_thread.join()
        self.batch_thread.join()

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
        on_tokens: ChatTokenCallback | None = None,
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
            on_tokens=on_tokens,
        )

        logger.info("Enqueuing request %s with %d prompt tokens", request_id, len(prompt_tokens))
        with self.admission_lock:
            request.admission_epoch = self.admission_epoch
            if self.pause_event.is_set():
                self._abort_request(request)
            else:
                self.request_queue.put(request)
        return request_id

    def _abort_request(self, request: InferenceRequest) -> None:
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
        request.future.get_loop().call_soon_threadsafe(request.future.set_result, responses)

    def _inference_loop(self) -> None:
        """Collect requests from the serving and batch them into batches of appropriate size for inference."""
        logger.info("Inference thread started")

        while not self.shutdown_event.is_set():
            requests = _fetch_from_queue(self.request_queue, self.config.batch_timeout, self.config.service.max_seqs)
            if not requests:
                continue

            batch = InferenceBatch()
            max_tokens_per_seq = self.engine.config.max_seq_len
            page_size = self.engine.config.page_size
            max_pages = self.engine.config.max_pages
            assert max_pages is not None
            batch_pages = 0
            logger.info("Max tokens per sequence: %d, pages per batch: %d", max_tokens_per_seq, max_pages)

            for r in requests:
                if len(r.prompt_tokens) >= max_tokens_per_seq:
                    r.future.get_loop().call_soon_threadsafe(
                        r.future.set_exception,
                        ValueError("Prompt must leave room for generation within the context limit"),
                    )
                    continue

                if r.n_generations > self.engine.config.max_seqs:
                    # fail requests that are too large
                    error_msg = (
                        f"Request {r.request_id} has n={r.n_generations} which exceeds "
                        f"the maximum allowed {self.engine.config.max_seqs}"
                    )
                    logger.error(error_msg)
                    r.future.get_loop().call_soon_threadsafe(r.future.set_exception, ValueError(error_msg))
                    continue

                max_tokens = min(len(r.prompt_tokens) + r.max_tokens, max_tokens_per_seq)
                request_pages = kv_pages_for_request(len(r.prompt_tokens), max_tokens, r.n_generations, page_size)
                if request_pages > max_pages:
                    r.future.get_loop().call_soon_threadsafe(
                        r.future.set_exception, ValueError("Request exceeds the serving KV page budget")
                    )
                    continue

                if batch.num_seqs() + r.n_generations > self.engine.config.max_seqs or (
                    batch_pages + request_pages > max_pages
                ):
                    self.batch_queue.put(batch)
                    batch = InferenceBatch()
                    batch_pages = 0
                batch.append(r)
                batch_pages += request_pages

            if batch:
                self.batch_queue.put(batch)

        logger.info("Inference thread shutting down")

    def _batch_processing_loop(self):
        """Batch processing loop running in background thread - waits for batches and executes them"""
        logger.info("Batch processing thread started")

        waiting: deque[InferenceRequest] = deque()

        def receive(limit: int) -> InferenceBatch:
            while len(waiting) < limit:
                try:
                    waiting.extend(self.batch_queue.get_nowait())
                except queue.Empty:
                    break
            return InferenceBatch(waiting.popleft() for _ in range(min(limit, len(waiting))))

        while not self.shutdown_event.is_set():
            try:
                batch = receive(self.config.service.max_seqs) or self.batch_queue.get(timeout=1)
                with (
                    self.model_lock,
                    hax.partitioning.set_mesh(self.config.trainer.device_mesh),
                    hax.axis_mapping(self.config.trainer.compute_axis_mapping),
                ):
                    self._execute_batch(batch, receive if self.config.service.max_seqs > 1 else None)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error executing batch: {e}", exc_info=True)
                # Set exceptions on all futures in the batch
                for req in batch:
                    try:
                        req.future.get_loop().call_soon_threadsafe(_fail_request, req.future, e)
                    except Exception:
                        pass

        logger.info("Batch processing thread shutting down")

    def _execute_batch(
        self,
        requests: InferenceBatch,
        receive: collections.abc.Callable[[int], InferenceBatch] | None = None,
    ) -> None:
        """Run queued requests with one serving model version."""
        if not self.engine:
            raise RuntimeError("Inference engine is not initialized.")

        in_flight: dict[int, InferenceRequest] = {}
        next_id = 0

        def admit(queued: InferenceBatch) -> list[Request]:
            nonlocal next_id
            admitted = []
            for req in queued:
                if self.pause_event.is_set() or req.admission_epoch != self.admission_epoch:
                    self._abort_request(req)
                    continue
                in_flight[next_id] = req
                admitted.append(self._engine_request(next_id, req))
                next_id += 1
            return admitted

        def on_tokens(request_index: int, choice: int, tokens: list[int]) -> None:
            callback = in_flight[request_index].on_tokens
            if callback is not None:
                callback(choice, tokens, self.model_version)

        def on_complete(request_index: int, result: GenerationResult) -> None:
            self._complete_request(in_flight.pop(request_index), result)

        def should_abort() -> bool:
            return self.shutdown_event.is_set() or self.pause_event.is_set()

        try:
            initial = admit(requests)
            if receive is None:
                self.engine.generate(initial, should_abort=should_abort, on_tokens=on_tokens, on_complete=on_complete)
            else:
                self.engine.generate_continuously(
                    initial,
                    lambda limit: admit(receive(limit)),
                    on_complete,
                    should_abort=should_abort,
                    on_tokens=on_tokens,
                )
        except Exception as error:
            for req in in_flight.values():
                req.future.get_loop().call_soon_threadsafe(_fail_request, req.future, error)
            raise

    def _engine_request(self, request_index: int, req: InferenceRequest) -> Request:
        stop_ids = None
        if req.stop_tokens:
            max_stop_length = max(map(len, req.stop_tokens))
            padded_stops = np.full((len(req.stop_tokens), max_stop_length), INVALID, dtype=np.int32)
            for index, stop in enumerate(req.stop_tokens):
                padded_stops[index, -len(stop) :] = stop
            stop_ids = hax.named(jnp.asarray(padded_stops), axis=("stop_seq", "position"))
        seed = req.seed if req.seed is not None else np.random.default_rng().integers(0, 2**32 - 1)
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(req.prompt_tokens) + req.max_tokens, dtype=jnp.int32),
            stop_tokens=stop_ids,
            temperature=jnp.array(req.temperature, dtype=jnp.float32),
            top_p=jnp.array(1.0 if req.top_p is None else req.top_p, dtype=jnp.float32),
            key=jrandom.PRNGKey(seed),
        )
        return Request(req.prompt_tokens, request_index, seq_params, req.n_generations)

    def _complete_request(self, req: InferenceRequest, result: GenerationResult) -> None:
        try:
            req_outputs = []
            for index, generated_tokens in enumerate(result.tokens):
                echo_token_ids = None
                echo_logprobs = None
                if req.echo_logprobs_top_k is not None:
                    echo_token_ids = req.prompt_tokens + generated_tokens
                    echo_logprobs = self.engine.score_token_logprobs(echo_token_ids, req.echo_logprobs_top_k)
                req_outputs.append(
                    InferenceResponse(
                        text=self.tokenizer.decode(generated_tokens, skip_special_tokens=True),
                        tokens=generated_tokens,
                        logprobs=result.logprobs[index] if result.logprobs is not None else None,
                        prompt_tokens=len(req.prompt_tokens),
                        completion_tokens=len(generated_tokens),
                        finish_reason=result.finish_reasons[index],
                        model_version=self.model_version,
                        request_id=req.request_id,
                        echo_token_ids=echo_token_ids,
                        echo_logprobs=echo_logprobs,
                    )
                )
            req.future.get_loop().call_soon_threadsafe(_finish_request, req.future, req_outputs)
        except Exception as e:
            logger.exception("Error processing result for %s", req.request_id)
            req.future.get_loop().call_soon_threadsafe(_fail_request, req.future, e)


def _finish_request(future: asyncio.Future, outputs: list[InferenceResponse]) -> None:
    if not future.done():
        future.set_result(outputs)


def _fail_request(future: asyncio.Future, error: Exception) -> None:
    if not future.done():
        future.set_exception(error)


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


def _sse_event(payload: str) -> str:
    return f"data: {payload}\n\n"


def _chat_completion_events(
    completion: ChatCompletion,
    *,
    include_usage: bool,
    emitted_text: dict[int, list[str]] | None = None,
) -> collections.abc.Iterator[str]:
    """Emit any remaining text, the finish reasons, and optional usage."""
    for choice in completion.choices:
        prefix = "" if emitted_text is None else "".join(emitted_text.get(choice.index, ()))
        text = choice.message.content or ""
        if not text.startswith(prefix):
            raise ValueError("Final completion does not match the streamed text")
        suffix = text[len(prefix) :]
        logprobs = (
            ChunkChoiceLogprobs(content=choice.logprobs.content, refusal=choice.logprobs.refusal)
            if choice.logprobs is not None
            else None
        )
        content = ChatCompletionChunkChoice(
            index=choice.index,
            delta=ChoiceDelta(role=None if prefix else "assistant", content=suffix),
            logprobs=logprobs,
        )
        if choice.model_extra and "token_ids" in choice.model_extra:
            content = content.model_copy(update={"token_ids": choice.model_extra["token_ids"]})
        finish = ChatCompletionChunkChoice(index=choice.index, delta=ChoiceDelta()).model_copy(
            update={"finish_reason": choice.finish_reason}
        )
        for chunk_choice in (content, finish) if suffix or not prefix else (finish,):
            chunk = ChatCompletionChunk(
                id=completion.id,
                object="chat.completion.chunk",
                created=completion.created,
                model=completion.model,
                choices=[chunk_choice],
            )
            if completion.model_extra and "model_version" in completion.model_extra:
                chunk = chunk.model_copy(update={"model_version": completion.model_extra["model_version"]})
            if completion.model_extra and "prompt_token_ids" in completion.model_extra:
                chunk = chunk.model_copy(update={"prompt_token_ids": completion.model_extra["prompt_token_ids"]})
            yield _sse_event(chunk.model_dump_json())
    if include_usage:
        chunk = ChatCompletionChunk(
            id=completion.id,
            object="chat.completion.chunk",
            created=completion.created,
            model=completion.model,
            choices=[],
            usage=completion.usage,
        )
        yield _sse_event(chunk.model_dump_json())
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


async def _create_completion(ctx: InferenceContext, request: CompletionRequest) -> Completion:
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


def _submit_chat_completion(
    ctx: InferenceContext, request: ChatCompletionRequest, *, on_tokens: ChatTokenCallback | None = None
) -> tuple[list[int], asyncio.Future[list[InferenceResponse]]]:
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

    future: asyncio.Future[list[InferenceResponse]] = asyncio.get_running_loop().create_future()
    ctx.submit_request(
        prompt_tokens=prompt_tokens,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_tokens=stop_tokens,
        seed=request.seed,
        future=future,
        n_generations=request.n or 1,
        on_tokens=on_tokens,
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
                token_str = (
                    f"{TOKEN_ID_PREFIX}{token_id}"
                    if request.return_tokens_as_token_ids
                    else ctx.tokenizer.convert_ids_to_tokens(token_id)
                )
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
    response = response.model_copy(update={"model_version": results[0].model_version})
    if request.return_token_ids:
        response = response.model_copy(update={"prompt_token_ids": prompt_tokens})
    return response


async def _create_chat_completion(ctx: InferenceContext, request: ChatCompletionRequest) -> ChatCompletion:
    try:
        prompt_tokens, future = _submit_chat_completion(ctx, request)
        return _format_chat_completion(
            ctx,
            request,
            prompt_tokens,
            await future,
            completion_id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _chat_stream_tokenizer(tokenizer: MarinTokenizer) -> PreTrainedTokenizerFast | None:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    if not tokenizer.clean_up_tokenization_spaces:
        return tokenizer
    # Transformers skips the whitespace cleanup for BPE unless explicitly enabled.
    if isinstance(tokenizer.backend_tokenizer.model, BPE) and not getattr(
        tokenizer, "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output", True
    ):
        return tokenizer
    return None


async def _stream_chat_completion(
    ctx: InferenceContext, request: ChatCompletionRequest, tokenizer: PreTrainedTokenizerFast, *, include_usage: bool
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[tuple[int, list[int], int] | None] = asyncio.Queue()
    closed = threading.Event()

    def on_tokens(choice: int, tokens: list[int], model_version: int) -> None:
        if not closed.is_set():
            loop.call_soon_threadsafe(updates.put_nowait, (choice, tokens, model_version))

    def on_done(future: asyncio.Future[list[InferenceResponse]]) -> None:
        if not future.cancelled():
            future.exception()  # The client may have disconnected before awaiting the result.
        if not closed.is_set():
            updates.put_nowait(None)

    prompt_tokens, future = _submit_chat_completion(ctx, request, on_tokens=on_tokens)
    future.add_done_callback(on_done)
    header = ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        choices=[],
        created=int(time.time()),
        model=request.model,
        object="chat.completion.chunk",
    )
    try:
        first = await updates.get()
        if first is None:
            future.result()  # Preserve the HTTP error status if generation failed before any tokens.
    except asyncio.CancelledError:
        closed.set()
        raise
    except Exception as exc:
        closed.set()
        logger.exception("Error before chat streaming started")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def events() -> collections.abc.AsyncIterator[str]:
        decoders: dict[int, DecodeStream] = {}
        emitted: dict[int, list[str]] = {}
        update = first
        try:
            while update is not None:
                choice, tokens, version = update
                if choice not in decoders:
                    decoders[choice] = DecodeStream(skip_special_tokens=True)
                text = decoders[choice].step(tokenizer.backend_tokenizer, tokens)
                if text:
                    pieces = emitted.setdefault(choice, [])
                    delta = ChoiceDelta(role=None if pieces else "assistant", content=text)
                    pieces.append(text)
                    chunk = header.model_copy(
                        update={
                            "choices": [ChatCompletionChunkChoice(index=choice, delta=delta)],
                            "model_version": version,
                        }
                    )
                    yield _sse_event(chunk.model_dump_json())
                update = await updates.get()
            completion = _format_chat_completion(
                ctx, request, prompt_tokens, future.result(), completion_id=header.id, created=header.created
            )
            for event in _chat_completion_events(completion, include_usage=include_usage, emitted_text=emitted):
                yield event
        except Exception:
            logger.exception("Error during chat streaming")
            yield _sse_event('{"error":{"message":"Chat generation failed","type":"server_error"}}')
        finally:
            closed.set()

    return StreamingResponse(events(), media_type="text/event-stream")


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

        # Register routes with thin wrappers that call helper functions
        @app.get("/health")
        async def health_check():
            return _health_check()

        @app.get("/v1/models")
        async def list_models() -> dict:
            model = Model(id=model_name, object="model", created=int(time.time()), owned_by="levanter")
            return {"object": "list", "data": [model.model_dump(mode="json")]}

        # A streaming request returns a StreamingResponse, which FastAPI passes through
        # untouched; `response_model` still describes the non-streaming body.
        @app.post("/v1/chat/completions", response_model=ChatCompletion)
        async def create_chat_completion(request: ChatCompletionRequest):
            include_usage = (request.stream_options or {}).get("include_usage") is True
            if request.stream and not request.logprobs and not request.return_token_ids:
                tokenizer = _chat_stream_tokenizer(inference_context.tokenizer)
                if tokenizer is not None:
                    return await _stream_chat_completion(
                        inference_context, request, tokenizer, include_usage=include_usage
                    )
            completion = await _create_chat_completion(inference_context, request)
            if not request.stream:
                return completion
            return StreamingResponse(
                _chat_completion_events(completion, include_usage=include_usage), media_type="text/event-stream"
            )

        @app.post("/v1/completions", response_model=Completion)
        async def create_completion(request: CompletionRequest):
            completion = await _create_completion(inference_context, request)
            if not request.stream:
                return completion
            return StreamingResponse(_completion_events(completion), media_type="text/event-stream")

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
