# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker: the inference primitive used by the per-shard pipeline body.

The vLLM engine is cached at module scope so it survives across shards on the
same Zephyr worker actor. `InlineRunner` re-uses the worker process across
shards, and a module-level binding survives the per-shard ``_InProcessWorkerContext``
reset that Zephyr does between shards.

`infer_records` is the single inference primitive: a batch of input records
in, a batch of `ResponseRecord` out. `pipeline.py` composes it with file
I/O and skip-existing checks to form the actual map_shard callable.

Tests inject a fake engine via `set_engine_factory`; the production factory
(`_load_vllm_engine`) is the only place ``import vllm`` happens, so importing
this module does not require vLLM to be installed.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ModelSpec, SamplingParams
from .input import PAYLOAD_KIND_MESSAGES, PAYLOAD_KIND_TEXT
from .output import ResponseRecord

logger = logging.getLogger(__name__)


class InferenceEngine(Protocol):
    """Subset of vLLM's `LLM` API that the worker actually uses.

    Defined as a Protocol so tests can inject a stub without depending on vLLM.
    Both methods take a list of inputs and a sampling-params object and return
    a list of objects whose ``.outputs[0].text`` is the generated string —
    mirroring vLLM's `RequestOutput` shape.
    """

    def generate(self, prompts: Sequence[str], sampling_params: Any) -> Sequence[Any]: ...

    def chat(self, messages: Sequence[Sequence[Mapping[str, Any]]], sampling_params: Any) -> Sequence[Any]: ...


EngineFactory = Callable[[str, Mapping[str, object]], InferenceEngine]
SamplingFactory = Callable[[SamplingParams], Any]


@dataclass
class _EngineCache:
    """Mutable module-level cache for the loaded vLLM engine.

    Wrapped in a dataclass so tests can replace `_CACHE` cleanly without
    fiddling with multiple module globals.
    """

    engine: InferenceEngine | None = None
    key: str | None = None
    engine_factory: EngineFactory | None = None
    sampling_factory: SamplingFactory | None = None


_CACHE = _EngineCache()


def set_engine_factory(factory: EngineFactory | None) -> None:
    """Override the engine constructor. Used by tests; None restores the default."""
    _CACHE.engine_factory = factory
    # Invalidate any previously-cached engine so the next ensure call rebuilds it.
    _CACHE.engine = None
    _CACHE.key = None


def set_sampling_factory(factory: SamplingFactory | None) -> None:
    """Override the SamplingParams converter. Used by tests."""
    _CACHE.sampling_factory = factory


def reset_engine_cache() -> None:
    """Drop any cached engine. Useful between tests."""
    _CACHE.engine = None
    _CACHE.key = None


def _engine_key(model_spec: ModelSpec, region: str) -> str:
    # The resolved model identifier already includes the region for `marin://`
    # paths; engine_kwargs are part of the cache identity because two specs
    # with different engine kwargs are not interchangeable.
    return f"{model_spec.resolve_for_region(region)}|{sorted(model_spec.engine_kwargs.items())}"


def _load_vllm_engine(model: str, engine_kwargs: Mapping[str, object]) -> InferenceEngine:
    """Default factory: import vLLM and construct an `LLM` engine."""
    import vllm  # local import: vLLM is optional at import time

    return vllm.LLM(model=model, **dict(engine_kwargs))  # type: ignore[return-value]


def _default_sampling(sampling: SamplingParams) -> Any:
    """Default converter: build a vllm.SamplingParams from our dataclass."""
    import vllm  # local import: vLLM is optional at import time

    kwargs: dict[str, Any] = {
        "temperature": sampling.temperature,
        "max_tokens": sampling.max_tokens,
        "min_tokens": sampling.min_tokens,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "repetition_penalty": sampling.repetition_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "presence_penalty": sampling.presence_penalty,
        "n": sampling.n,
        "skip_special_tokens": sampling.skip_special_tokens,
    }
    if sampling.stop:
        kwargs["stop"] = list(sampling.stop)
    if sampling.stop_token_ids:
        kwargs["stop_token_ids"] = list(sampling.stop_token_ids)
    if sampling.seed is not None:
        kwargs["seed"] = sampling.seed
    if sampling.logprobs is not None:
        kwargs["logprobs"] = sampling.logprobs
    if sampling.prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = sampling.prompt_logprobs
    kwargs.update(sampling.extra)
    return vllm.SamplingParams(**kwargs)


def ensure_engine(model_spec: ModelSpec, region: str) -> InferenceEngine:
    """Return a cached engine, constructing one on first call.

    The engine survives across shards in the same worker process because it
    lives in module scope; Zephyr's per-shard context reset does not touch
    module globals.
    """
    key = _engine_key(model_spec, region)
    if _CACHE.engine is not None and _CACHE.key == key:
        return _CACHE.engine
    factory = _CACHE.engine_factory or _load_vllm_engine
    resolved = model_spec.resolve_for_region(region)
    logger.info("Loading inference engine for model=%s in region=%s", resolved, region)
    _CACHE.engine = factory(resolved, model_spec.engine_kwargs)
    _CACHE.key = key
    return _CACHE.engine


def _extract_response(output: Any) -> tuple[str, dict[str, Any]]:
    """Extract the generated text and accompanying metadata from a vLLM RequestOutput.

    The text is returned **verbatim**: we do not strip ``<think>`` /
    ``<reasoning>`` blocks, special tokens, or any other model-emitted markers.
    Downstream callers post-process as they see fit (SFT pipelines strip
    thinking traces; reward-model pipelines keep them; evals parse them).

    Field names mirror vLLM's `CompletionOutput` / `RequestOutput`
    (verified against vllm-project/vllm@main: ``vllm/outputs.py``). Extras
    captured (omitted from the dict when the source field is None):

    Always-captured (scalar):
      - ``finish_reason``       — ``"stop"`` / ``"length"`` / ``"abort"`` / ...
      - ``stop_reason``         — stop string or token id that triggered the stop
      - ``cumulative_logprob``  — single float, sum of token logprobs
      - ``num_cached_tokens``   — prefix-cache hits on the request

    Caller-opt-in (returned by vLLM only when the corresponding
    `SamplingParams` knob is set):
      - ``logprobs``        — per-position top-k logprobs (from
        ``SamplingParams.logprobs``)
      - ``prompt_logprobs`` — per-prompt-token logprobs (from
        ``SamplingParams.prompt_logprobs``)

    Intentionally **not** captured (large or internal):
      - ``token_ids``, ``prompt_token_ids`` — re-tokenize ``text`` if needed;
        add ``capture_token_ids`` to `InferenceConfig` as a follow-up if a
        workflow demands it.
      - ``metrics``, ``routed_experts``, ``lora_request``,
        ``kv_transfer_params``, ``encoder_prompt``,
        ``encoder_prompt_token_ids`` — internal or niche.
    """
    completion = output.outputs[0]
    text = completion.text
    extras: dict[str, Any] = {}
    for attr in ("finish_reason", "stop_reason", "cumulative_logprob"):
        value = getattr(completion, attr, None)
        if value is not None:
            extras[attr] = value
    num_cached_tokens = getattr(output, "num_cached_tokens", None)
    if num_cached_tokens is not None:
        extras["num_cached_tokens"] = num_cached_tokens
    completion_logprobs = getattr(completion, "logprobs", None)
    if completion_logprobs is not None:
        extras["logprobs"] = _serialize_logprobs(completion_logprobs)
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        extras["prompt_logprobs"] = _serialize_logprobs(prompt_logprobs)
    return text, extras


def _serialize_logprobs(logprobs: Any) -> Any:
    """Reduce vLLM's per-position ``dict[int, Logprob]`` to JSON-serializable form.

    vLLM's `Logprob` is a small dataclass; ``vars()`` works on it. We keep
    the per-position dicts as ``{token_id: {"logprob": ..., "rank": ..., ...}}``.
    Falls back to ``repr`` if a position is not a mapping (defensive).
    """
    if logprobs is None:
        return None
    serialized: list[Any] = []
    for position in logprobs:
        if position is None:
            serialized.append(None)
        elif isinstance(position, dict):
            serialized.append(
                {str(token_id): vars(lp) if hasattr(lp, "__dict__") else lp for token_id, lp in position.items()}
            )
        else:
            serialized.append(repr(position))
    return serialized


def _convert_sampling(sampling: SamplingParams) -> Any:
    factory = _CACHE.sampling_factory or _default_sampling
    return factory(sampling)


def infer_records(
    records: Sequence[dict[str, Any]],
    *,
    model_spec: ModelSpec,
    sampling: SamplingParams,
    region: str,
    shard_idx: int,
) -> list[ResponseRecord]:
    """Run inference over a homogeneous batch of input records.

    All records must share the same ``payload['kind']``; the function
    dispatches on it to call either ``engine.generate`` (text) or
    ``engine.chat`` (messages). Returns one ResponseRecord per input record.
    The engine is loaded (or reused from cache) on first call within the
    worker process.
    """
    if not records:
        return []
    if sampling.n != 1:
        raise NotImplementedError(
            "SamplingParams.n > 1 is not supported in v1 — `_extract_response` only "
            "reads outputs[0], which would silently drop the remaining completions. "
            "Track support as a follow-up; until then set n=1."
        )
    kinds = {r["payload"]["kind"] for r in records}
    if len(kinds) > 1:
        raise ValueError(
            f"infer_records was given mixed payload kinds {sorted(kinds)}; "
            "callers must split inputs into homogeneous batches."
        )
    kind = kinds.pop()
    engine = ensure_engine(model_spec, region)
    vllm_sampling = _convert_sampling(sampling)
    if kind == PAYLOAD_KIND_TEXT:
        prompts = [r["payload"]["prompt"] for r in records]
        outputs = engine.generate(prompts, vllm_sampling)
    elif kind == PAYLOAD_KIND_MESSAGES:
        conversations = [r["payload"]["messages"] for r in records]
        outputs = engine.chat(conversations, vllm_sampling)
    else:
        raise ValueError(f"Unknown payload kind {kind!r}.")
    response_records: list[ResponseRecord] = []
    for record, output in zip(records, outputs, strict=True):
        text, extras = _extract_response(output)
        response_records.append(ResponseRecord(id=record["id"], shard=shard_idx, response=text, extra=extras))
    return response_records
