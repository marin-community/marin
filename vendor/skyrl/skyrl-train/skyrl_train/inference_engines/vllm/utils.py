import json
import errno
from typing import Any, Dict, Protocol

from skyrl_train.config.behavior_logprobs import (
    ROLLOUT_LOGPROB_VALIDATION_KEY,
    validate_behavior_logprob_sampling,
)


class PrefixCacheStatsLike(Protocol):
    """The token counters vLLM's `PrefixCacheStats` carries for one scheduler iteration."""

    queries: int
    hits: int


def is_port_collision(exc: BaseException) -> bool:
    """Return whether an exception chain identifies an address-in-use failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "eaddrinuse" in message or "address already in use" in message:
            return True
        if isinstance(current, OSError) and current.errno == errno.EADDRINUSE:
            return True
        current = current.__cause__ or current.__context__
    return False


def prefix_cache_hit_rate_percent(stats: PrefixCacheStatsLike) -> float | None:
    """Share of queried prefix tokens that were already cached, as a percentage.

    `PrefixCacheStats` counts tokens, not requests: `queries` is how many were looked up and
    `hits` how many were served from cache.

    Returns None for an iteration that queried nothing, which has no rate to report rather
    than a rate of zero.
    """
    if stats.queries == 0:
        return None
    return stats.hits / stats.queries * 100.0


class PrefixCacheHitRateAccumulator:
    """Peak and per-iteration samples of the prefix cache hit rate for one engine.

    Owns which scheduler iterations are sampled at all. Only iterations that queried prefix
    tokens are: `prefix_cache_stats` is a delta that admission writes, decode-only iterations
    outnumber admissions heavily, and scoring those as zeroes would pin the median near 0.0.
    vLLM's `CachingMetrics.observe` skips the same iterations, keyed on `requests`, which
    `PrefixCacheStats.record` bumps alongside `queries`.
    """

    def __init__(self) -> None:
        self.peak: float = 0.0
        self.samples: list[float] = []

    def observe(self, stats: PrefixCacheStatsLike | None, is_active: bool) -> None:
        """Fold one scheduler iteration in, where `is_active` means it had queued or running work."""
        if stats is None:
            return
        rate = prefix_cache_hit_rate_percent(stats)
        if rate is None:
            return
        self.peak = max(self.peak, rate)
        if is_active:
            self.samples.append(rate)


def pop_vllm_wrapper_kwargs(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove SkyRL serving options before passing engine_kwargs to vLLM.
    """
    wrapper_kwargs: Dict[str, Any] = {}

    enable_auto_tools = engine_kwargs.pop("enable_auto_tools", engine_kwargs.pop("enable_auto_tool_choice", None))
    if enable_auto_tools is not None:
        wrapper_kwargs["enable_auto_tools"] = bool(enable_auto_tools)

    tool_parser = engine_kwargs.pop("tool_parser", engine_kwargs.pop("tool_call_parser", None))
    if tool_parser is not None:
        wrapper_kwargs["tool_parser"] = tool_parser

    # Sampling params for OpenAI-style requests (Harbor terminal-bench rollouts)
    openai_sampling = engine_kwargs.pop("openai_sampling_params", None)
    if openai_sampling is not None:
        wrapper_kwargs["openai_sampling_params"] = openai_sampling

    if ROLLOUT_LOGPROB_VALIDATION_KEY in engine_kwargs:
        wrapper_kwargs[ROLLOUT_LOGPROB_VALIDATION_KEY] = engine_kwargs.pop(ROLLOUT_LOGPROB_VALIDATION_KEY)

    return wrapper_kwargs


def ensure_token_ids_in_sse_chunk(sse_chunk: str) -> str:
    """Copy per-chunk token IDs to harbor's canonical streaming location.

    vLLM 0.20.2 puts delta token IDs flat on the choice at
    ``choices[0].token_ids`` (sibling to ``delta``, gated by
    ``return_token_ids=True``).  Harbor's ``_chunk_completion_token_ids``
    checks ``choices[0].delta.provider_specific_fields.token_ids`` first
    (canonical streaming path), then falls back to ``choices[0].token_ids``.

    This function copies the flat field into the delta's
    ``provider_specific_fields`` so harbor finds it on the FIRST path —
    avoiding edge cases where the fallback misses control-token chunks
    (empty-delta chunks that still carry token IDs).
    """
    if not sse_chunk.startswith("data: "):
        return sse_chunk
    payload = sse_chunk[len("data: ") :]
    if payload.strip() == "[DONE]":
        return sse_chunk
    try:
        data = json.loads(payload)
        choices = data.get("choices")
        if not choices:
            return sse_chunk
        choice = choices[0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list):
            return sse_chunk
        delta = choice.setdefault("delta", {})
        psf = delta.setdefault("provider_specific_fields", {})
        if "token_ids" not in psf:
            psf["token_ids"] = token_ids
            return f"data: {json.dumps(data)}\n\n"
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return sse_chunk


def apply_openai_sampling(
    body: Dict[str, Any], sampling_params: Dict[str, Any], validate_rollout_logprob_sampling: bool
) -> None:
    """Apply generator sampling overrides and validate training probabilities."""
    body.update(
        {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "top_k": sampling_params.get("top_k", -1),
            "min_p": sampling_params.get("min_p", 0.0),
        }
    )
    # Completion logprobs=0 requests sampled-token probabilities; False disables chat logprobs.
    logprobs = body.get("logprobs")
    if validate_rollout_logprob_sampling and logprobs is not None and logprobs is not False:
        validate_behavior_logprob_sampling(body)
