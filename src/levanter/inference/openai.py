# Copyright 2025 The Levanter Authors
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
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from openai.types import Completion, CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import CompletionChoice, Logprobs
from pydantic import BaseModel, Field

from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


# OpenAI requests are all defined as TypedDicts, which FastAPI struggles to
# encode in a useful way. Since we control this half of the API, we'll define
# our own equivalent Pydantic models here.


class ChatMessage(BaseModel):
    """A single chat message in the conversation."""

    role: Literal["system", "user", "assistant", "tool", "function", "developer"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict] = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions endpoint."""

    model: str
    messages: List[ChatMessage]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: bool = Field(default=False, description="Whether to include logprobs in the response")
    top_logprobs: Optional[int] = None
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    stream_options: Optional[Dict] = None
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """Request model for text completions endpoint."""

    model: str
    prompt: Union[str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[int] = None
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    stream_options: Optional[Dict] = None
    suffix: Optional[str] = None
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = None
    user: Optional[str] = None


@dataclass
class InferenceServerConfig:
    """Configuration for OpenAI-compatible inference server."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LmConfig)

    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)

    # Default generation parameters for API
    temperature: float = 0.7
    seed: int = 42

    batch_timeout: float = 0.1  # seconds to wait for more requests before processing batch

    host: str = "localhost"
    port: int = 10243


@dataclass
class InferenceRequest:
    """Internal request structure for the inference thread"""

    request_id: str
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float
    stop_tokens: List[int] | None
    seed: int | None
    future: asyncio.Future
    n_generations: int = 1
    enable_logprobs: bool = False


@dataclass
class InferenceResponse:
    """Internal response structure for the inference thread"""

    request_id: str
    text: str
    tokens: List[int]
    prompt_tokens: int
    completion_tokens: int
    logprobs: Optional[List[float]] = None


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


class InferenceContext:
    """Background thread that manages the InferenceEngine and processes requests"""

    def __init__(self, model: LmHeadModel, tokenizer, engine: InferenceEngine, config: InferenceServerConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.config = config
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.batch_queue: queue.Queue[InferenceBatch] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
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
            self.model = None
            self.engine = None
        logger.info("Inference model unloaded.")

    def reload(self, weight_callback: WeightSource):
        """Reload the inference model using the given weight callback.

        If new weights are found, new requests are paused, existing requests are
        allowed to complete, and the new weights are loaded.
        """
        logger.info("New weights available, waiting for model lock...")
        lock_start_time = time.time()
        with self.model_lock:
            lock_wait_time = time.time() - lock_start_time
            logger.info(f"Acquired model lock after {lock_wait_time}, reloading weights...")

            start = time.time()
            with (
                self.config.trainer.device_mesh,
                hax.axis_mapping(self.config.trainer.compute_axis_mapping),
            ):
                self.model = weight_callback(self.model)
                self.engine = InferenceEngine.from_model_with_config(
                    model=self.model, tokenizer=self.tokenizer, config=self.config.service
                )
                elapsed = time.time() - start
            logger.info(f"Model reloaded in {elapsed:.2f}s")

    def submit_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[int]],
        seed: int | None,
        future: asyncio.Future,
        n_generations: int = 1,
        enable_logprobs: bool = False,
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
            stop_tokens=stop_tokens,
            seed=seed,
            future=future,
            n_generations=n_generations,
            enable_logprobs=enable_logprobs,
        )

        logger.info("Enqueuing request %s", request)
        self.request_queue.put(request)
        return request_id

    def _inference_loop(self):
        """Main inference loop running in background thread - collects requests into batches"""
        logger.info("Inference thread started")

        while not self.shutdown_event.is_set():
            requests: list[InferenceRequest] = _fetch_all_from_queue(self.request_queue, self.config.batch_timeout)
            if not requests:
                continue

            batch = InferenceBatch()
            max_tokens_per_seq = self.engine.config.max_pages_per_seq * self.engine.config.page_size
            max_tokens_per_batch = self.engine.config.imputed_max_pages * self.engine.config.page_size
            logger.info(f"Max tokens per seq: {max_tokens_per_seq}, per batch: {max_tokens_per_batch}")

            for r in requests:
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
                    r.future.get_loop().call_soon_threadsafe(r.future.set_exception, ValueError(error_msg))
                elif (
                    batch.num_seqs() + r.n_generations <= self.engine.config.max_seqs
                    and batch.total_tokens() + (len(r.prompt_tokens) + r.max_tokens) <= max_tokens_per_batch
                ):
                    batch.append(r)
                else:
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
                    self.config.trainer.device_mesh,
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
                        req.future.get_loop().call_soon_threadsafe(req.future.set_exception, e)
                    except Exception:
                        pass

        logger.info("Batch processing thread shutting down")

    def _execute_batch(self, requests: InferenceBatch):
        """Execute a batch of inference requests"""
        service_requests = []

        if not self.engine:
            raise RuntimeError("Inference engine is not initialized.")

        for i, req in enumerate(requests):
            # Create stop tokens if specified
            stop_ids = None
            if req.stop_tokens:
                stop_ids = hax.named(jnp.asarray(req.stop_tokens, dtype=jnp.int32), axis="position").broadcast_axis(
                    {"stop_seq": 1}
                )

            # dumb fallback seed if none provided
            if req.seed is None:
                req.seed = np.random.default_rng().integers(0, 2**32 - 1)

            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(req.prompt_tokens) + req.max_tokens, dtype=jnp.int32),
                stop_tokens=stop_ids,
                temperature=jnp.array(req.temperature, dtype=jnp.float32),
                key=jrandom.PRNGKey(req.seed if req.seed is not None else i),
            )

            service_req = Request(
                prompt_tokens=req.prompt_tokens,
                request_id=i,  # Use batch index as service request id
                decode_params=seq_params,
                n_generations=req.n_generations,
                enable_logprobs=req.enable_logprobs,
            )
            service_requests.append(service_req)

        # Generate responses
        start_time = time.time()
        result = self.engine.generate(service_requests)
        duration = time.time() - start_time
        logger.info(f"Batch completed in {duration:.2f}s, generated {result.total_generated} tokens")

        # Return results to futures
        output_idx = 0
        for req in requests:
            try:
                req_outputs = []
                for _ in range(req.n_generations):
                    if output_idx < len(result.tokens):
                        generated_tokens = result.tokens[output_idx]
                        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                        # Extract logprobs if requested, logprobs are already only for generated tokens
                        result_logprobs = None
                        if req.enable_logprobs:
                            assert result.logprobs is not None
                            result_logprobs = result.logprobs[output_idx]

                        req_outputs.append(
                            InferenceResponse(
                                text=text,
                                tokens=result.tokens[output_idx],
                                logprobs=result_logprobs,
                                prompt_tokens=len(req.prompt_tokens),
                                completion_tokens=len(generated_tokens),
                                request_id=req.request_id,
                            )
                        )
                        output_idx += 1
                    else:
                        logger.error(f"Missing output for request {req.request_id}")
                        req_outputs.append(
                            InferenceResponse(
                                text="<error while generating>",
                                tokens=[],
                                logprobs=None,
                                prompt_tokens=0,
                                completion_tokens=0,
                                request_id=req.request_id,
                            )
                        )

                # Set the future result
                req.future.get_loop().call_soon_threadsafe(req.future.set_result, req_outputs)
            except Exception as e:
                logger.error(f"Error processing result for {req.request_id}: {e}")
                req.future.get_loop().call_soon_threadsafe(req.future.set_exception, e)


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


async def _create_completion(ctx: InferenceContext, request: CompletionRequest) -> Completion:
    """Create a text completion using OpenAI API format."""
    try:
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        # Process stop sequences
        stop_tokens = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_list = [request.stop]
            else:
                stop_list = request.stop

            # Tokenize stop sequences
            stop_tokens = []
            for stop in stop_list:
                stop_ids = ctx.tokenizer(stop, add_special_tokens=False)["input_ids"]
                if stop_ids:
                    stop_tokens.extend(stop_ids)

        # Create futures for all prompts
        futures = []
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            prompt_tokens = ctx.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            total_prompt_tokens += len(prompt_tokens)

            # Create future for this request
            future: asyncio.Future = asyncio.Future()
            futures.append(future)

            # Submit to inference thread
            ctx.submit_request(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_tokens=stop_tokens,
                seed=request.seed,
                future=future,
                n_generations=request.n or 1,
                enable_logprobs=bool(request.logprobs),
            )

        # Wait for all results
        results: List[List[InferenceResponse]] = await asyncio.gather(*futures)

        # Format responses
        choice_idx = 0
        for result in results:
            for generation in result:
                # Format logprobs if available
                logprobs = None
                if request.logprobs:
                    # Convert logprobs to API format
                    prompt_len = generation.prompt_tokens
                    generated_tokens = generation.tokens[prompt_len:]

                    # Create token logprobs in OpenAI format
                    tokens = []
                    token_logprobs = []
                    if generation.logprobs:
                        for token_id, lp in zip(generated_tokens, generation.logprobs):
                            token_str = ctx.tokenizer.decode([token_id], skip_special_tokens=False)
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
                        text=generation.text,
                        index=choice_idx,
                        finish_reason="stop",
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

    except Exception as e:
        logger.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _create_chat_completion(ctx: InferenceContext, request: ChatCompletionRequest) -> ChatCompletion:
    """Create a chat completion using OpenAI API format."""
    try:
        # Convert Pydantic models to dicts for tokenizer
        messages_dict = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # Apply chat template
        try:
            prompt_tokens = ctx.tokenizer.apply_chat_template(messages_dict, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            # Fallback: simple concatenation if template fails
            logger.warning(f"Chat template failed, using fallback: {e}", exc_info=True)
            prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            prompt_tokens = ctx.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Process stop sequences
        stop_tokens = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_list = [request.stop]
            else:
                stop_list = request.stop

            stop_tokens = []
            for stop in stop_list:
                stop_ids = ctx.tokenizer(stop, add_special_tokens=False)["input_ids"]
                if stop_ids:
                    stop_tokens.extend(stop_ids)

        # Create future and submit request
        future: asyncio.Future = asyncio.Future()
        ctx.submit_request(
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_tokens=stop_tokens,
            seed=request.seed,
            future=future,
            n_generations=request.n or 1,
            enable_logprobs=request.logprobs,
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
                # Convert logprobs to API format
                prompt_len = generation.prompt_tokens
                generated_tokens = generation.tokens[prompt_len:]

                # Create content logprobs in OpenAI format
                content_logprobs = []
                if generation.logprobs:
                    for token_id, lp in zip(generated_tokens, generation.logprobs):
                        token_str = ctx.tokenizer.decode([token_id], skip_special_tokens=False)
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
                    finish_reason="stop",
                    logprobs=logprobs,
                )
            )
            total_completion_tokens += generation.completion_tokens

        return ChatCompletion(
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

    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class InferenceServer:
    """Wraps a FastAPI server around the inference context.

    Provides OpenAI compatible endpoints for text and chat completions.
    """

    def __init__(self, config: InferenceServerConfig, inference_context: InferenceContext, app: FastAPI):
        """Initialize the inference server with pre-built components.

        Use InferenceServer.create() to build a new server instance.
        """
        self.config = config
        self.inference_context = inference_context
        self.app = app

    @staticmethod
    def create(config: InferenceServerConfig) -> "InferenceServer":
        """Create and initialize a new InferenceServer.

        This factory method loads the model, tokenizer, and creates all necessary
        components for the inference server.
        """
        tokenizer_path: str | None = config.tokenizer
        if config.tokenizer is None:
            if config.hf_checkpoint is not None:
                tokenizer_path = config.hf_checkpoint.model_name_or_path

        if tokenizer_path is None:
            raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")

        tokenizer = load_tokenizer(tokenizer_path)
        key = jrandom.PRNGKey(config.seed)
        vocab_size = len(tokenizer)
        logger.info(f"Loaded tokenizer with vocab size {vocab_size} from {tokenizer_path}")

        # N.B. I don't know what the difference between compute and parameter axis mapping is
        # `compute_axis_mapping` shards across tensor-parallel dimensions (?)
        with (
            config.trainer.device_mesh,
            hax.axis_mapping(config.trainer.compute_axis_mapping),
        ):
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
            model = _load_model(config, Vocab, key=key)

        service = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.service)

        # Create and start inference thread
        inference_context = InferenceContext(model, tokenizer, service, config)
        inference_context.start()

        logger.info("Inference service initialized and ready")

        # Create FastAPI app with initialized context
        app = InferenceServer._create_app(inference_context)

        return InferenceServer(config, inference_context, app)

    @staticmethod
    def _create_app(inference_context: InferenceContext) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title="Levanter Inference Service", version="1.0.0")

        # Register routes with thin wrappers that call helper functions
        @app.get("/health")
        async def health_check():
            return _health_check()

        @app.post("/v1/completions", response_model=Completion)
        async def create_completion(request: CompletionRequest) -> Completion:
            return await _create_completion(inference_context, request)

        @app.post("/v1/chat/completions", response_model=ChatCompletion)
        async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletion:
            return await _create_chat_completion(inference_context, request)

        return app

    def unload(self):
        """Unload the inference model to free up resources."""
        self.inference_context.unload()

    def reload(self, weight_callback: WeightSource):
        """Reload the model weights using the provided callback.

        Args:
            weight_callback: Function that takes the current model and returns new model
        """
        self.inference_context.reload(weight_callback)

    def serve(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        finally:
            self.shutdown()

    async def serve_async(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            config = uvicorn.Config(self.app, host=self.config.host, port=self.config.port)
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the inference context."""
        self.inference_context.shutdown()


def _load_model(config: InferenceServerConfig, Vocab: Axis, *, key) -> LmHeadModel:
    """Load a model either from a checkpoint or HF repo."""

    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    mp = config.trainer.mp

    if config.checkpoint_path is not None:
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = mp.cast_to_compute(model)
        return model
    else:
        assert config.hf_checkpoint
        logger.info(
            f"Loading model from HF checkpoint {config.hf_checkpoint} "
            + f"type={config.model.model_type}, "
            + f"vocab_size={Vocab.size}, "
            + f"dtype={config.trainer.mp.compute_dtype}"
        )
        converter: HFCheckpointConverter = HFCheckpointConverter.from_hf(config.hf_checkpoint)
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint)
        if config.tokenizer is not None:
            converter = converter.replaced(tokenizer=load_tokenizer(config.tokenizer))

        logger.info(f"Param mapping: {config.trainer.parameter_axis_mapping}")
        logger.info(f"Compute mapping: {config.trainer.compute_axis_mapping}")

        model = converter.load_pretrained(
            config.model.model_type,
            ref=config.hf_checkpoint,
            dtype=config.trainer.mp.compute_dtype,
            axis_mapping=config.trainer.parameter_axis_mapping,
        )
        return model
