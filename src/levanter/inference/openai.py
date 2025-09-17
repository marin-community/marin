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
from typing import List, Optional, Union

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import uvicorn
from fastapi import FastAPI, HTTPException
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from pydantic import BaseModel

from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass
class InferenceServerConfig:
    """Configuration for OpenAI-compatible inference server."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)

    # Default generation parameters for API
    max_new_tokens: int = 16
    temperature: float = 0.7
    seed: int = 42


# Pydantic models for OpenAI API compatibility


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 16
    temperature: float = 1.0
    stop: Optional[Union[List[str], str]] = None
    n: int = 1
    seed: Optional[int] = None
    logprobs: bool = False


class LogProb(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"
    logprobs: Optional[List[LogProb]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    stop: Optional[Union[List[str], str]] = None
    n: int = 1
    seed: Optional[int] = None
    logprobs: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Optional[List[LogProb]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


# Internal data structures


@dataclass
class InferenceRequest:
    """Internal request structure for the inference thread"""

    request_id: str
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float
    stop_tokens: Optional[List[int]]
    seed: int
    future: asyncio.Future
    n_generations: int = 1
    original_prompt: str = ""
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


# A callback which replaces the current model.
WeightSource = collections.abc.Callable[[LmHeadModel], LmHeadModel]


class InferenceContext:
    """Background thread that manages the GenerationService and processes requests"""

    def __init__(self, model: LmHeadModel, tokenizer, service: InferenceEngine, config):
        self.model = model
        self.tokenizer = tokenizer
        self.service = service
        self.config = config
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._next_request_id = 0

    def start(self):
        """Start the inference thread"""
        logger.info("Starting inference context...")
        self.thread.start()

    def shutdown(self):
        """Signal shutdown and wait for thread to finish"""
        logger.info("Shutting down inference context...")
        self.shutdown_event.set()
        self.thread.join(timeout=1)

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
            self.model = weight_callback(self.model)
            elapsed = time.time() - start
            logger.info(f"Model reloaded in {elapsed:.2f}s")

    def submit_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[int]],
        seed: int,
        future: asyncio.Future,
        n_generations: int = 1,
        original_prompt: str = "",
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
            original_prompt=original_prompt,
            enable_logprobs=enable_logprobs,
        )

        logger.info("Enqueuing request %s", request)

        self.request_queue.put(request)
        return request_id

    def _inference_loop(self):
        """Main inference loop running in background thread"""
        logger.info("Inference thread started")

        while not self.shutdown_event.is_set():
            try:
                # Enqueue a batch of requests. Wait up to 0.1s to batch requests together.
                requests = []
                try:
                    req = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                requests.append(req)
                deadline = time.time() + 0.1
                while time.time() < deadline and len(requests) < self.config.service.max_seqs:
                    try:
                        requests.append(self.request_queue.get(timeout=deadline - time.time()))
                    except queue.Empty:
                        break

                with self.model_lock:
                    self._process_batch(requests)
            except Exception as e:
                logger.error(f"Error in inference loop: {e}", exc_info=True)

        logger.info("Inference thread shutting down")

    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of inference requests"""
        try:
            service_requests = []

            for i, req in enumerate(requests):
                # Create stop tokens if specified
                stop_ids = None
                if req.stop_tokens:
                    stop_ids = hax.named(
                        jnp.asarray(req.stop_tokens, dtype=jnp.int32), axis="position"
                    ).broadcast_axis({"stop_seq": 1})

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
            result = self.service.generate(service_requests)
            duration = time.time() - start_time
            logger.info(f"Batch completed in {duration:.2f}s, generated {result.total_generated} tokens")

            # Return results to futures
            output_idx = 0
            for req in requests:
                try:
                    req_outputs = []
                    for _ in range(req.n_generations):
                        if output_idx < len(result.tokens):
                            # Decode tokens to text, excluding prompt
                            prompt_len = len(req.prompt_tokens)
                            generated_tokens = result.tokens[output_idx][prompt_len:]
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
                                    prompt_tokens=prompt_len,
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

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            # Set exceptions on all futures
            for req in requests:
                try:
                    req.future.get_loop().call_soon_threadsafe(req.future.set_exception, e)
                except Exception:
                    pass


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


async def _create_completion(ctx: InferenceContext, request: CompletionRequest) -> CompletionResponse:
    """Create a text completion using OpenAI API format."""
    try:
        # Handle both string and list prompts
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

        logger.info("Here?")

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
                seed=request.seed if request.seed is not None else 42,
                future=future,
                n_generations=request.n,
                original_prompt=prompt,
                enable_logprobs=request.logprobs,
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
                    logprobs = [
                        LogProb(
                            token=ctx.tokenizer.decode([token_id], skip_special_tokens=False),
                            logprob=float(lp),
                            bytes=list(ctx.tokenizer.decode([token_id], skip_special_tokens=False).encode("utf-8")),
                        )
                        for token_id, lp in zip(generated_tokens, generation.logprobs or [])
                    ]

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

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _create_chat_completion(ctx: InferenceContext, request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Create a chat completion using OpenAI API format."""
    try:
        # Convert messages to prompt using tokenizer's chat template
        messages = [msg.model_dump() for msg in request.messages]

        # Apply chat template
        try:
            prompt_tokens = ctx.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            # Fallback: simple concatenation if template fails
            logger.warning(f"Chat template failed, using fallback: {e}")
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
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
            seed=request.seed if request.seed is not None else 42,
            future=future,
            n_generations=request.n,
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
                logprobs = [
                    LogProb(
                        token=ctx.tokenizer.decode([token_id], skip_special_tokens=False),
                        logprob=float(lp),
                        bytes=list(ctx.tokenizer.decode([token_id], skip_special_tokens=False).encode("utf-8")),
                    )
                    for token_id, lp in zip(generated_tokens, generation.logprobs or [])
                ]

            choices.append(
                ChatChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=generation.text),
                    finish_reason="stop",
                    logprobs=logprobs,
                )
            )
            total_completion_tokens += generation.completion_tokens

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
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

        with (
            config.trainer.device_mesh,
            hax.axis_mapping(config.trainer.compute_axis_mapping),
        ):
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
            model = _load_model(config, Vocab, key=key)
            assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

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

        @app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest) -> CompletionResponse:
            return await _create_completion(inference_context, request)

        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
            return await _create_chat_completion(inference_context, request)

        return app

    def reload(self, weight_callback: WeightSource):
        """Reload the model weights using the provided callback.

        Args:
            weight_callback: Function that takes the current model and returns new model
        """
        self.inference_context.reload(weight_callback)

    def serve(self, host: str, port: int):
        try:
            logger.info(f"Starting Levanter inference server on {host}:{port}")
            uvicorn.run(self.app, host=host, port=port)
        finally:
            self.shutdown()

    async def serve_async(self, host: str, port: int):
        try:
            logger.info(f"Starting Levanter inference server on {host}:{port}")
            config = uvicorn.Config(self.app, host=host, port=port)
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
        assert hasattr(config.model, "hf_checkpoint_converter"), "model config lacks HF loader"
        converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(
            reference_checkpoint=config.hf_checkpoint, tokenizer=load_tokenizer(config.tokenizer)
        )
        model = converter.load_pretrained(
            config.model.model_type, ref=config.hf_checkpoint, dtype=config.trainer.mp.compute_dtype
        )
        return model
