# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Run identical materialized inference batches on every JAX process.

Only process zero accepts HTTP requests. Other processes call ``worker_loop``.
The runtime must terminate the entire job if any process fails during a collective.
"""

import json
import logging
import os
import threading
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

import haliax as hax

from levanter.inference.engine import GenerationResult, InferenceEngine, Request, TokenSequenceLogprobs
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.snowball import SnowballCache
from levanter.inference.snowball_scoring import score_cached


logger = logging.getLogger(__name__)


def _broadcast_command(command):
    source = jax.process_index() == 0
    payload = json.dumps(command).encode() if source else b""
    try:
        length = np.array([len(payload)], dtype=np.int32)
        length = int(multihost_utils.broadcast_one_to_all(length, is_source=source)[0])
        data = np.frombuffer(payload, dtype=np.uint8).copy() if source else np.zeros(length, dtype=np.uint8)
        data = multihost_utils.broadcast_one_to_all(data, is_source=source)
        return json.loads(data.tobytes())
    except Exception:
        logger.exception("Distributed inference command broadcast failed; terminating this rank")
        os._exit(1)


def _encode_request(request):
    params = request.decode_params
    values = (params.max_num_tokens, params.temperature, params.top_p, params.key)
    if params.stop_tokens is not None:
        values += (params.stop_tokens.array,)
    if any(isinstance(value, jax.Array) for value in values):
        raise ValueError("Multi-host request metadata must remain host-backed until broadcast")
    return dict(
        prompt_tokens=request.prompt_tokens,
        request_id=request.request_id,
        n_generations=request.n_generations,
        max_num_tokens=int(params.max_num_tokens),
        temperature=float(params.temperature),
        top_p=float(params.top_p),
        key=np.asarray(params.key).tolist(),
        stop_tokens=None if params.stop_tokens is None else np.asarray(params.stop_tokens.array).tolist(),
    )


def _decode_request(value):
    stops = value["stop_tokens"]
    params = SeqDecodingParams(
        max_num_tokens=jnp.array(value["max_num_tokens"], jnp.int32),
        temperature=jnp.array(value["temperature"], jnp.float32),
        top_p=jnp.array(value["top_p"], jnp.float32),
        key=jnp.array(value["key"], jnp.uint32),
        stop_tokens=None if stops is None else hax.named(jnp.array(stops, jnp.int32), ("stop_seq", "position")),
    )
    return Request(value["prompt_tokens"], value["request_id"], params, value["n_generations"])


class MultiHostInferenceEngine(InferenceEngine):
    """Broadcast after tokenization, seed resolution, stop encoding, and batching."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._command_lock = threading.Lock()

    def _dispatch(self, command):
        if command["op"] == "generate":
            result = super().generate([_decode_request(r) for r in command["requests"]])
            cache = self.gen_state.cache
            # Snowball caches expose routing drops; a dropped assignment invalidates the result.
            if isinstance(cache, SnowballCache) and np.any(np.asarray(cache.dropped_assignments)):
                raise RuntimeError(f"Inference dropped expert assignments: {np.asarray(cache.dropped_assignments)}")
            return result
        if command["op"] == "score":
            if isinstance(self.gen_state.cache, SnowballCache):
                return score_cached(self, command["tokens"], command["top_k"])
            return super().score_token_logprobs(command["tokens"], command["top_k"])
        raise ValueError(f"Unknown inference operation: {command['op']}")

    def _execute_or_abort(self, command):
        try:
            return self._dispatch(command)
        except Exception:
            # A rank may have failed before its peers entered a collective. Exiting
            # triggers JAX heartbeat fate sharing and Iris gang failure; an HTTP
            # handler must not keep a partly failed distributed world alive.
            logger.exception("Distributed inference execution failed; terminating this rank")
            os._exit(1)

    def generate(self, requests: Sequence[Request], step_callback=None) -> GenerationResult:
        if step_callback is not None:
            raise ValueError("Multi-host generation does not support process-local step callbacks")
        if jax.process_index() != 0:
            raise RuntimeError("Only process zero may submit inference batches")
        if not requests:
            raise ValueError("Generation batch must not be empty")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Request IDs must be unique within a batch")
        for request in requests:
            if request.n_generations < 1 or request.n_generations > self.config.max_seqs:
                raise ValueError("Generation count exceeds configured sequence slots")
            if not request.prompt_tokens or len(request.prompt_tokens) > self.config.max_seq_len:
                raise ValueError("Prompt must be nonempty and fit the configured sequence length")
            stops = request.decode_params.stop_tokens
            if stops is not None:
                if (
                    stops.axis_size("stop_seq") > self.config.max_stop_seqs
                    or stops.axis_size("position") > self.config.max_stop_tokens
                ):
                    raise ValueError("Stop-token configuration exceeds service capacity")
            if any(t < 0 or t >= self.model.Vocab.size for t in request.prompt_tokens):
                raise ValueError("Prompt contains a token outside the model vocabulary")
        with self._command_lock:
            encoded = [_encode_request(r) for r in requests]
            logger.info("Broadcasting %d host-materialized inference requests", len(encoded))
            command = _broadcast_command(dict(op="generate", requests=encoded))
            result = self._execute_or_abort(command)
            assert isinstance(result, GenerationResult)
            return result

    def score_token_logprobs(self, token_ids: Sequence[int], top_k: int) -> TokenSequenceLogprobs:
        if jax.process_index() != 0:
            raise RuntimeError("Only process zero may submit scoring batches")
        token_ids = list(token_ids)
        if top_k < 1 or len(token_ids) > self.config.max_seq_len:
            raise ValueError("Scoring requires positive top_k and a sequence within the configured length")
        if any(t < 0 or t >= self.model.Vocab.size for t in token_ids):
            raise ValueError("Scoring contains a token outside the model vocabulary")
        with self._command_lock:
            command = _broadcast_command(dict(op="score", tokens=token_ids, top_k=top_k))
            result = self._execute_or_abort(command)
            assert isinstance(result, TokenSequenceLogprobs)
            return result

    def worker_loop(self):
        if jax.process_index() == 0:
            raise RuntimeError("Process zero must run the HTTP server")
        while True:
            command = _broadcast_command(None)
            if command["op"] == "shutdown":
                return
            self._execute_or_abort(command)

    def shutdown_workers(self):
        with self._command_lock:
            _broadcast_command(dict(op="shutdown"))
