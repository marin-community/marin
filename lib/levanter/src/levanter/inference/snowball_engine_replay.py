# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in production engine capture and learner consumption of Snowball routes."""

import dataclasses
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import haliax as hax
import jax.numpy as jnp
import numpy as np

from levanter.inference.engine import (
    CompletionCallback,
    GenerationResult,
    InferenceEngine,
    Request,
    RequestSource,
    TokenCallback,
)
from levanter.inference.jit_scheduler import FinishReason
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


class SnowballReplaySession:
    """Capture the routes used by a single engine generation call."""

    @staticmethod
    def validate_engine(engine: InferenceEngine) -> None:
        if not isinstance(engine.model, SnowballLMHeadModel) or type(engine.sampler) is not Sampler:
            raise ValueError("Snowball replay requires Snowball and the standard sampler")
        if engine.config.max_seq_len > engine.model.config.max_seq_len or engine.config.max_rounds <= 0:
            raise ValueError("Engine sequence or round limits exceed the replay model contract")

    def __init__(self, engine: InferenceEngine, policy_id: str):
        self.validate_engine(engine)
        if not policy_id:
            raise ValueError("Snowball replay requires an immutable policy identity")
        self.engine = engine
        self.policy_id = policy_id
        self._requests: dict[int, Request] = {}
        self._collector = _SnowballReplayCollector()

    def admit(self, request: Request, *, retain_payload: bool) -> None:
        """Validate the replay scope before the engine mutates state."""
        if request.n_generations != 1:
            raise ValueError("Snowball replay supports n_generations=1; clones are unsupported")
        tokens = np.asarray(request.prompt_tokens)
        if (
            tokens.ndim != 1
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0)
            or np.any(tokens >= self.engine.model.Vocab.size)
        ):
            raise ValueError("Snowball replay prompts must contain vocabulary token IDs")
        params = request.decode_params
        length = min(int(params.max_num_tokens), self.engine.config.max_seq_len)
        if length <= len(tokens):
            raise ValueError("Snowball replay requires room for at least one generated target")
        if (
            not np.isfinite(float(params.temperature))
            or float(params.temperature) < 0
            or not 0 < float(params.top_p) <= 1
        ):
            raise ValueError("Unsupported replay sampling parameters")
        pages = self.engine.gen_state.decode_state.page_table.num_pages
        if self.engine._request_pages(request) > pages:
            raise ValueError(
                "Snowball replay needs sufficient pages for terminal inputs before releasing finished slots"
            )
        if retain_payload:
            self._requests[request.request_id] = request
        if retain_payload or self.engine.config.prefix_cache != "disabled":
            self._collector.retain.add(request.request_id)

    def complete(self, request_id: int, result: GenerationResult) -> SnowballReplayPayload | None:
        """Package one completed request before its slot is reused."""
        request = self._requests.pop(request_id, None)
        self._collector.retain.discard(request_id)
        if request is None or result.finish_reasons == [FinishReason.ABORT]:
            self._collector.rows.pop((request_id, 0), None)
            return None
        assert result.logprobs is not None
        assert isinstance(self.engine.model, SnowballLMHeadModel)
        params = request.decode_params
        stops = (
            ()
            if params.stop_tokens is None
            else tuple(tuple(int(t) for t in row) for row in np.asarray(params.stop_tokens.array))
        )
        sampling = SnowballReplaySampling(
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            max_num_tokens=min(int(params.max_num_tokens), self.engine.config.max_seq_len),
            stop_tokens=stops,
            key=tuple(int(k) for k in np.asarray(params.key)),
        )
        return self._collector.payload(
            request_id=request_id,
            tokens=[*request.prompt_tokens, *result.tokens[0]],
            prompt_length=len(request.prompt_tokens),
            sampling=sampling,
            logprobs=result.logprobs[0],
            finish_reason=result.finish_reasons[0].name,
            policy_id=self.policy_id,
            model=self.engine.model,
        )

    def generate(
        self,
        requests: Sequence[Request],
        *,
        on_complete: CompletionCallback,
        should_abort: Callable[[], bool] | None = None,
        on_tokens: TokenCallback | None = None,
        poll_requests: RequestSource | None = None,
    ) -> GenerationResult:
        return self.engine._generate(
            requests,
            replay=self._collector,
            on_complete=on_complete,
            should_abort=should_abort,
            on_tokens=on_tokens,
            poll_requests=poll_requests,
        )


def generate_snowball_replay(
    engine: InferenceEngine, requests: Sequence[Request], *, policy_id: str
) -> SnowballEngineReplayResult:
    """Run the production scheduler, returning one completed payload per request."""
    session = SnowballReplaySession(engine, policy_id)
    payloads = {}
    for request in requests:
        session.admit(request, retain_payload=True)

    def complete(request_id, result):
        payload = session.complete(request_id, result)
        assert payload is not None
        payloads[request_id] = payload

    generation = session.generate(requests, on_complete=complete)
    return SnowballEngineReplayResult(generation, tuple(payloads[request.request_id] for request in requests))


def align_snowball_policy_routes(
    model: SnowballLMHeadModel,
    payloads: Sequence[SnowballReplayPayload],
    *,
    policy_id: str,
    tokens: np.ndarray,
    token_valid: np.ndarray,
    position_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align one full emitted route payload to each padded trajectory row."""
    if tokens.ndim != 2 or token_valid.shape != tokens.shape or position_ids.shape != tokens.shape:
        raise ValueError("Route replay requires matching [batch, position] inputs")
    if len(payloads) != tokens.shape[0]:
        raise ValueError("Route replay requires one payload per trajectory")
    cfg = model.config
    selected = np.empty((*tokens.shape, cfg.num_layers, cfg.num_experts_per_token), dtype=np.int32)
    selected[...] = np.arange(cfg.num_experts_per_token, dtype=np.int32)
    alpha = np.zeros((*tokens.shape, cfg.num_layers), dtype=np.float32)
    for row, payload in enumerate(payloads):
        payload.validate_model(model, policy_id=policy_id)
        valid = np.asarray(token_valid[row], dtype=np.bool_)
        if not np.array_equal(tokens[row, valid], payload.tokens):
            raise ValueError(f"Rollout route token identity differs at trajectory {row}")
        if not np.array_equal(position_ids[row, valid], payload.positions):
            raise ValueError(f"Rollout route positions differ at trajectory {row}")
        for layer, routes in enumerate(payload.layers):
            selected[row, valid, layer] = routes.selected_experts
            alpha[row, valid, layer] = routes.qb_alpha[:, 0]
    return selected, alpha


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
