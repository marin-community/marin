# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in production engine capture and learner consumption of Snowball routes."""

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import haliax as hax
import jax.numpy as jnp
import numpy as np

from levanter.inference.engine import GenerationResult, InferenceEngine, Request
from levanter.inference.snowball_replay_payload import (
    SnowballReplayPayload,
    SnowballReplaySampling,
    _SnowballReplayCollector,
)
from levanter.layers.sampler import Sampler
from levanter.models.snowball import SnowballLMHeadModel


@dataclass(frozen=True)
class SnowballEngineReplayResult:
    """Original generation outputs and payloads in the same request order."""

    generation: GenerationResult
    payloads: tuple[SnowballReplayPayload, ...]


def generate_snowball_replay(
    engine: InferenceEngine, requests: Sequence[Request], *, policy_id: str
) -> SnowballEngineReplayResult:
    """Run the production scheduler, retaining its actual Snowball serving routes.

    Supports heterogeneous contiguous causal prompts, one choice per request,
    STOP/LENGTH completion, pending admission and slot reuse. Callbacks, abort,
    clones and packed-document segments are outside this API. Validation occurs
    before resetting the engine. The caller supplies an immutable weight-policy
    identity and keeps the serving mesh and arithmetic context active.
    """
    if not isinstance(engine.model, SnowballLMHeadModel) or type(engine.sampler) is not Sampler:
        raise ValueError("Snowball replay requires a Snowball model and the standard engine sampler")
    if not policy_id:
        raise ValueError("Snowball replay requires an immutable policy identity")
    model = engine.model
    if engine.config.max_seq_len > model.config.max_seq_len or engine.config.max_rounds <= 0:
        raise ValueError("engine sequence/round limits exceed the replay model contract")
    pages = []
    for request in requests:
        if request.n_generations != 1:
            raise ValueError("Snowball replay supports n_generations=1; clones are unsupported")
        tokens = np.asarray(request.prompt_tokens)
        if (
            tokens.ndim != 1
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0)
            or np.any(tokens >= model.Vocab.size)
        ):
            raise ValueError("Snowball replay prompts must contain vocabulary token IDs")
        params = request.decode_params
        if params.stop_tokens is not None:
            stops = params.stop_tokens
            if (
                stops.ndim != 2
                or stops.axis_size("stop_seq") > engine.config.max_stop_seqs
                or stops.axis_size("position") > engine.config.max_stop_tokens
            ):
                raise ValueError("replay stop sequences exceed the engine's configured capacity")
        length = min(int(params.max_num_tokens), engine.config.max_seq_len)
        if length <= len(tokens):
            raise ValueError("Snowball replay requires room for at least one generated target")
        if (
            not np.isfinite(float(params.temperature))
            or float(params.temperature) < 0
            or not 0 < float(params.top_p) <= 1
        ):
            raise ValueError("unsupported replay sampling parameters")
        pages.append((length + engine.config.page_size - 1) // engine.config.page_size)
    if sum(sorted(pages, reverse=True)[: engine.config.max_seqs]) > engine.gen_state.decode_state.page_table.num_pages:
        raise ValueError("Snowball replay needs sufficient pages for terminal inputs before releasing finished slots")
    collector = _SnowballReplayCollector()
    generation = engine._generate(requests, replay=collector)
    assert generation.logprobs is not None
    payloads = []
    for index, request in enumerate(requests):
        params = request.decode_params
        stops = (
            ()
            if params.stop_tokens is None
            else tuple(tuple(int(t) for t in row) for row in np.asarray(params.stop_tokens.array))
        )
        sampling = SnowballReplaySampling(
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            max_num_tokens=min(int(params.max_num_tokens), engine.config.max_seq_len),
            stop_tokens=stops,
            key=tuple(int(k) for k in np.asarray(params.key)),
        )
        payloads.append(
            collector.payload(
                request_id=int(request.request_id),
                tokens=[*request.prompt_tokens, *generation.tokens[index]],
                prompt_length=len(request.prompt_tokens),
                sampling=sampling,
                logprobs=generation.logprobs[index],
                finish_reason=generation.finish_reasons[index].name,
                policy_id=policy_id,
                model=model,
            )
        )
    return SnowballEngineReplayResult(generation, tuple(payloads))


def replay_snowball_logits(
    model: SnowballLMHeadModel, payload: SnowballReplayPayload, *, policy_id: str
) -> hax.NamedArray:
    """Compute untransformed learner logits with the transported serving routes."""
    example, capture = payload.training_inputs(model, policy_id=policy_id)
    return model(example.tokens, example.attn_mask, route_captures=capture)


def replay_snowball_loss(
    model: SnowballLMHeadModel,
    payload: SnowballReplayPayload,
    *,
    policy_id: str,
    target_weights: hax.NamedArray | None = None,
) -> jnp.ndarray:
    """Compute weighted next-token loss over every generated target, including EOS.

    Optional weights have one axis named ``target`` and one entry per generated
    token. The public model loss retains its combine-weight and gradient rules.
    The caller supplies the learner mesh and Haliax axis mapping.
    No teacher capture or persistent optimizer state is created.
    """
    example, capture = payload.training_inputs(model, policy_id=policy_id)
    if target_weights is not None:
        if target_weights.axes != (hax.Axis("target", len(payload.sampled_logprobs)),):
            raise ValueError("target_weights must have one 'target' entry per generated token")
        weights = jnp.zeros((len(payload.tokens),), dtype=target_weights.dtype)
        weights = weights.at[payload.prompt_length - 1 : -1].set(target_weights.array)
        example = dataclasses.replace(example, loss_weight=hax.named(weights, example.tokens.axes))
    loss = model.compute_next_token_loss(example, route_captures=capture)
    return loss.scalar() if isinstance(loss, hax.NamedArray) else loss
