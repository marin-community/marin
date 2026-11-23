# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve one model and its paged engine behind protocol-independent admission."""

import asyncio
import collections.abc
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from levanter.inference.engine import (
    GenerationResult,
    InferenceEngine,
    InferenceEngineConfig,
    Request,
    TokenSequenceLogprobs,
)
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.page_table import kv_pages_for_request
from levanter.inference.snowball_engine_replay import SnowballReplaySession
from levanter.inference.snowball_replay_payload import SnowballReplayPayload
from levanter.inference.utils import INVALID
from levanter.models.lm_model import LmHeadModel
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "levanter"


@dataclass
class InferenceServerConfig:
    """Model mesh, engine limits and capabilities shared by serving adapters."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(4096))

    model_name: str = DEFAULT_MODEL_NAME
    """Model id this server advertises from ``/v1/models``."""

    capture_routes: bool = False
    """Capture the Snowball routes for completed single-choice sequences."""

    initial_policy_id: str | None = None
    """Immutable identity of the initial snapshot, or a server-scoped identity if omitted."""

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
    prompt_tokens: list[int]
    max_tokens: int
    temperature: float
    top_p: float | None
    stop_tokens: list[list[int]] | None
    seed: int | None
    future: asyncio.Future
    admission_epoch: int = 0
    model_version: int = 0
    policy_id: str = ""
    return_route_captures: bool = False
    n_generations: int = 1
    echo_logprobs_top_k: int | None = None
    on_tokens: ChatTokenCallback | None = None


@dataclass
class InferenceResponse:
    """Internal response structure for the inference thread"""

    request_id: str
    text: str
    tokens: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason
    model_version: int
    logprobs: list[float] | None = None
    echo_token_ids: list[int] | None = None
    echo_logprobs: TokenSequenceLogprobs | None = None
    policy_id: str = ""
    behavior_logprobs: list[float] | None = None
    policy_logprobs: list[float] | None = None
    route_capture: SnowballReplayPayload | None = None


class PolicyVersionMismatch(ValueError):
    """The caller requested a policy version that is no longer available."""


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


class InferenceContext:
    """Background thread that manages the InferenceEngine and processes requests"""

    def __init__(self, model: LmHeadModel, tokenizer, engine: InferenceEngine, config: InferenceServerConfig):
        self.model = model
        self.model_version = 0
        self._server_id = uuid.uuid4().hex
        self.policy_id = config.initial_policy_id or f"{self._server_id}:0"
        self._installed_policy_ids = {self.policy_id}
        if config.capture_routes:
            SnowballReplaySession.validate_engine(engine)
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

    def policy_identity(self) -> tuple[int, str]:
        """Read the version and policy identity from one admission epoch."""
        with self.admission_lock:
            return self.model_version, self.policy_id

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

    def reload(self, weight_callback: WeightSource, *, expected_version: int, policy_id: str | None = None) -> int:
        """Stage same-architecture weights, then atomically install a new serving version.

        The callback must return new weights without donating or mutating the current model.
        Staging failures preserve the current model, version, and admission state.
        An already paused context remains paused after installation.
        """
        with self.admission_lock:
            if expected_version != self.model_version:
                raise PolicyVersionMismatch(f"Expected model version {expected_version}, serving {self.model_version}")
            installed_policy_id = policy_id or f"{self._server_id}:{expected_version + 1}"
            if installed_policy_id in self._installed_policy_ids:
                raise ValueError("A policy identity can only be installed once")
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
                    raise PolicyVersionMismatch(
                        f"Expected model version {expected_version}, serving {self.model_version}"
                    )
            was_paused = self.pause_event.is_set()
            self.pause_generation()
            try:
                with self.model_lock, self.admission_lock:
                    jax.block_until_ready(self.engine.gen_state)
                    self.model = candidate
                    self.engine.model = candidate
                    self.model_version += 1
                    self.policy_id = installed_policy_id
                    self._installed_policy_ids.add(installed_policy_id)
                    installed_version = self.model_version
            finally:
                if not was_paused:
                    self.resume_generation()
        return installed_version

    def submit_request(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        stop_tokens: list[list[int]] | None,
        seed: int | None,
        future: asyncio.Future,
        n_generations: int = 1,
        echo_logprobs_top_k: int | None = None,
        on_tokens: ChatTokenCallback | None = None,
        expected_model_version: int | None = None,
        expected_policy_id: str | None = None,
        return_route_captures: bool = False,
    ) -> str:
        """Validate token inputs and enqueue the request for the active policy."""
        if self.shutdown_event.is_set():
            raise RuntimeError("InferenceContext is shut down")
        service = self.engine.config
        assert service.max_prefill_size is not None
        if (
            not prompt_tokens
            or len(prompt_tokens) >= service.max_seq_len
            or len(prompt_tokens) > service.max_prefill_size
        ):
            raise ValueError("Prompt must be nonempty, fit prefill and leave room within the context limit")
        if min(prompt_tokens) < 0 or max(prompt_tokens) >= self.model.Vocab.size:
            raise ValueError("Prompt contains a token ID outside the model vocabulary")
        if max_tokens < 1 or not 1 <= n_generations <= service.max_seqs:
            raise ValueError("Generation length and choice count must be positive and fit serving limits")
        if not np.isfinite(temperature) or temperature < 0 or (top_p is not None and not 0 < top_p <= 1):
            raise ValueError("Sampling requires finite temperature >= 0 and top_p in (0, 1]")
        if seed is not None and not 0 <= seed < 2**32:
            raise ValueError("Seed must be a uint32")
        if stop_tokens and (
            len(stop_tokens) > service.max_stop_seqs
            or any(
                not 0 < len(stop) <= service.max_stop_tokens or min(stop) < 0 or max(stop) >= self.model.Vocab.size
                for stop in stop_tokens
            )
        ):
            raise ValueError("Stop sequences must contain vocabulary IDs and fit the configured limits")
        if return_route_captures and not self.config.capture_routes:
            raise ValueError("This server does not have route capture enabled")
        if self.config.capture_routes and n_generations != 1:
            raise ValueError("Route capture requires one generation per prompt")

        with self.admission_lock:
            if expected_model_version is not None and expected_model_version != self.model_version:
                raise PolicyVersionMismatch(
                    f"Expected model version {expected_model_version}, serving {self.model_version}"
                )
            if expected_policy_id is not None and expected_policy_id != self.policy_id:
                raise PolicyVersionMismatch("The expected policy snapshot is no longer serving")
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
                admission_epoch=self.admission_epoch,
                model_version=self.model_version,
                policy_id=self.policy_id,
                return_route_captures=return_route_captures,
                n_generations=n_generations,
                echo_logprobs_top_k=echo_logprobs_top_k,
                on_tokens=on_tokens,
            )
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
                model_version=request.model_version,
                policy_id=request.policy_id,
                logprobs=[],
                behavior_logprobs=[],
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
        replay = SnowballReplaySession(self.engine, self.policy_id) if self.config.capture_routes else None
        next_id = 0

        def admit(queued: InferenceBatch) -> list[Request]:
            nonlocal next_id
            admitted = []
            for req in queued:
                if self.pause_event.is_set() or req.admission_epoch != self.admission_epoch:
                    self._abort_request(req)
                    continue
                engine_request = self._engine_request(next_id, req)
                if replay is not None:
                    replay.admit(engine_request, retain_payload=req.return_route_captures)
                in_flight[next_id] = req
                admitted.append(engine_request)
                next_id += 1
            return admitted

        def on_tokens(request_index: int, choice: int, tokens: list[int]) -> None:
            callback = in_flight[request_index].on_tokens
            if callback is not None:
                callback(choice, tokens, in_flight[request_index].model_version)

        def on_complete(request_index: int, result: GenerationResult) -> None:
            req = in_flight.pop(request_index)
            capture = None if replay is None else replay.complete(request_index, result)
            self._complete_request(req, result, capture)

        def should_abort() -> bool:
            return self.shutdown_event.is_set() or self.pause_event.is_set()

        try:
            initial = admit(requests)
            if replay is not None:
                replay.generate(
                    initial,
                    on_complete=on_complete,
                    should_abort=should_abort,
                    on_tokens=on_tokens,
                    poll_requests=None if receive is None else lambda limit: admit(receive(limit)),
                )
            elif receive is None:
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

    def _complete_request(
        self, req: InferenceRequest, result: GenerationResult, capture: SnowballReplayPayload | None = None
    ) -> None:
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
                        model_version=req.model_version,
                        policy_id=req.policy_id,
                        behavior_logprobs=(
                            [0.0] * len(generated_tokens)
                            if req.temperature == 0
                            else (None if result.logprobs is None else result.logprobs[index])
                        ),
                        policy_logprobs=None if result.policy_logprobs is None else result.policy_logprobs[index],
                        route_capture=capture,
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
