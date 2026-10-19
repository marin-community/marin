"""One resolved evaluation-limit contract for every Evalchemy execution path.

``lm-eval`` calls its limits ``max_length`` (model/context window) and
``max_gen_toks`` (generation).  Evalchemy's chat benchmarks historically used
``max_new_tokens`` while OpenAI-compatible endpoints use ``max_tokens``.  The
different spellings are adapter details, not independent user controls.

This module is deliberately dependency-free: resolve the two user-facing
limits once at CLI ingress, materialize the backend-specific representations,
then pass the same resolved values to native and custom benchmarks.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

MAX_OUTPUT_ALIASES = ("max_tokens", "max_new_tokens", "max_gen_toks")
MODEL_LENGTH_ALIASES = ("max_length", "max_model_len")

# The caller's ``max_length`` is the served model's total context window. Leave
# more than the requested ~50-character wiggle room: tokenization and endpoint
# templates are not guaranteed to be byte-identical across every provider.
DEFAULT_CONTEXT_SAFETY_TOKENS = 64


class MissingContextLengthError(ValueError):
    """Raised when a task cannot select inputs without a context length."""


class ContextWindowExceededError(ValueError):
    """Raised when a prompt and its required output cannot fit a context window."""

    def __init__(self, *, context_length: int, required_tokens: int):
        self.context_length = context_length
        self.required_tokens = required_tokens
        super().__init__(f"request requires {required_tokens} tokens but the context window is {context_length} tokens")


def require_context_length(context_length: int | None, *, task_name: str) -> int:
    """Return an explicit context length or raise a typed configuration error."""
    if context_length is None:
        raise MissingContextLengthError(
            f"{task_name} requires an explicit max_length so unsupported examples can be excluded"
        )
    return _as_positive_int(context_length, "max_length")


def encoded_token_count(encode: Callable[[str], Sequence[Any]], text: str) -> int:
    """Count text tokens with a caller-supplied encoder."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return len(encode(text))


def message_content_token_count(encode: Callable[[str], Sequence[Any]], messages: Sequence[Mapping[str, Any]]) -> int:
    """Count message content without provider-specific chat framing."""
    total = 0
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise TypeError("each message must be a mapping with string content")
        total += encoded_token_count(encode, message["content"])
    return total


def ensure_context_window(
    *,
    context_length: int,
    prompt_tokens: int,
    output_tokens: int,
    safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS,
) -> None:
    """Raise a typed error unless the complete request fits its context window."""
    for name, value in (
        ("context_length", context_length),
        ("prompt_tokens", prompt_tokens),
        ("output_tokens", output_tokens),
        ("safety_tokens", safety_tokens),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    required_tokens = prompt_tokens + output_tokens + safety_tokens
    if required_tokens > context_length:
        raise ContextWindowExceededError(context_length=context_length, required_tokens=required_tokens)


def parse_key_value_args(value: Optional[str | Mapping[str, Any]]) -> dict[str, Any]:
    """Parse lm-eval's comma-separated ``key=value`` notation without coercion."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    parsed: dict[str, Any] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value in argument string, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in argument string item {item!r}")
        parsed[key] = raw_value.strip()
    return parsed


def format_key_value_args(values: Mapping[str, Any]) -> str:
    """Render a deterministic lm-eval ``key=value`` argument string."""
    return ",".join(f"{key}={value}" for key, value in values.items())


def safe_generation_cap(
    *,
    context_length: int,
    prompt_tokens: int,
    requested_max_tokens: int,
    safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS,
) -> int:
    """Return the largest safe output cap for one fully rendered endpoint request.

    ``context_length`` is the server's actual total window, not lm-eval's
    prompt-only bookkeeping limit. A request whose prompt already consumes the
    usable window fails locally and clearly rather than retrying a deterministic
    endpoint 400.
    """
    for name, value in (
        ("context_length", context_length),
        ("prompt_tokens", prompt_tokens),
        ("requested_max_tokens", requested_max_tokens),
        ("safety_tokens", safety_tokens),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    ensure_context_window(
        context_length=context_length,
        prompt_tokens=prompt_tokens,
        output_tokens=1,
        safety_tokens=safety_tokens,
    )
    available = context_length - prompt_tokens - safety_tokens
    return min(requested_max_tokens, available)


def endpoint_prompt_token_count(
    tokenizer: Any,
    payload: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Count one final endpoint payload with the tokenizer/template it will use.

    Text completions are encoded directly. Chat payloads are rendered with the
    configured tokenizer's chat template and an assistant-generation marker.
    ``JsonChatStr`` is intentionally handled duck-typed to avoid importing an
    lm-eval-private class into this dependency-light module.
    """
    if isinstance(payload, (list, tuple)) and all(isinstance(token, int) for token in payload):
        return len(payload)
    if hasattr(payload, "prompt"):
        payload = json.loads(payload.prompt)
    if isinstance(payload, str):
        return len(tokenizer.encode(payload, add_special_tokens=False))
    if isinstance(payload, Sequence) and all(isinstance(message, Mapping) for message in payload):
        encoded = tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], Sequence):
            encoded = encoded[0]
        return len(encoded)
    raise TypeError(f"unsupported endpoint payload for preflight tokenization: {type(payload).__name__}")


def preflight_endpoint_generation(
    *,
    tokenizer: Any,
    payloads: Sequence[Any],
    gen_kwargs: Mapping[str, Any] | None,
    context_length: int | None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, int | None, int | None]:
    """Tokenize a transport batch and lower its output cap if necessary.

    Returns ``(new_gen_kwargs, largest_prompt, effective_max_tokens)``. Calls
    without an explicit context or generation cap are a strict no-op. A batch
    shares one endpoint generation kwargs dictionary, so it uses the largest
    rendered prompt in that batch. Without a client tokenizer, text and chat
    prompts cannot be counted accurately; their requested cap is left intact.
    Token-ID payloads remain countable without a tokenizer.
    """
    if context_length is None or gen_kwargs is None:
        return (dict(gen_kwargs) if gen_kwargs is not None else None, None, None)
    present = [(key, gen_kwargs[key]) for key in MAX_OUTPUT_ALIASES if key in gen_kwargs]
    if not present:
        return (dict(gen_kwargs), None, None)
    requested = resolve_limit("max_tokens", present)
    assert requested is not None
    if tokenizer is None and any(
        not (isinstance(payload, (list, tuple)) and all(isinstance(token, int) for token in payload))
        for payload in payloads
    ):
        return dict(gen_kwargs), None, None
    largest_prompt = max(
        endpoint_prompt_token_count(tokenizer, payload, chat_template_kwargs=chat_template_kwargs)
        for payload in payloads
    )
    cap = safe_generation_cap(
        context_length=context_length,
        prompt_tokens=largest_prompt,
        requested_max_tokens=requested,
    )
    adjusted = copy.deepcopy(dict(gen_kwargs))
    for key, _ in present:
        adjusted[key] = cap
    return adjusted, largest_prompt, cap


def _as_positive_int(value: Any, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{source} must be a positive integer, got {parsed}")
    return parsed


def resolve_limit(name: str, values: Iterable[tuple[str, Any]]) -> Optional[int]:
    seen: list[tuple[str, int]] = []
    for source, value in values:
        if value is not None:
            seen.append((source, _as_positive_int(value, source)))
    if not seen:
        return None
    distinct = {value for _, value in seen}
    if len(distinct) != 1:
        rendered = ", ".join(f"{source}={value}" for source, value in seen)
        raise ValueError(f"conflicting {name} values: {rendered}; supply one consistent limit")
    return seen[0][1]


@dataclass(frozen=True)
class EvaluationLimits:
    """Resolved model-context and model-output limits for one evaluation run."""

    max_length: Optional[int] = None
    max_tokens: Optional[int] = None

    def apply(self, args: Any) -> None:
        """Materialize this contract in the representations each adapter consumes.

        ``model_args.max_length`` reaches every lm-eval model adapter.  Native
        lm-eval tasks consume ``gen_kwargs.max_gen_toks``; custom benchmarks
        consume ``args.max_tokens`` and are additionally protected by
        :class:`eval.task.BaseBenchmark` at request construction time.
        """
        model_args = parse_key_value_args(getattr(args, "model_args", None))
        gen_kwargs = parse_key_value_args(getattr(args, "gen_kwargs", None))

        if self.max_length is not None:
            model_args["max_length"] = self.max_length
            # vLLM's LM adapter historically calls this spelling.  Keep it in
            # sync only when a legacy caller already selected that adapter key.
            if "max_model_len" in model_args:
                model_args["max_model_len"] = self.max_length

        if self.max_tokens is not None:
            for alias in MAX_OUTPUT_ALIASES:
                gen_kwargs.pop(alias, None)
            # lm-eval's native task API expects this spelling.  The custom
            # benchmark layer converts it to its endpoint-specific spelling.
            gen_kwargs["max_gen_toks"] = self.max_tokens

        args.model_args = format_key_value_args(model_args)
        args.gen_kwargs = format_key_value_args(gen_kwargs) if gen_kwargs else None
        args.max_length = self.max_length
        args.max_tokens = self.max_tokens


def resolve_evaluation_limits(args: Any) -> EvaluationLimits:
    """Resolve and materialize Evalchemy's sole max-length/max-output contract.

    ``--max_length`` and ``--max_tokens`` are the canonical public inputs.
    Existing ``model_args`` / ``gen_kwargs`` spellings remain supported as
    compatibility aliases, but disagreement is a hard error rather than a
    benchmark-path-dependent result.
    """
    model_args = parse_key_value_args(getattr(args, "model_args", None))
    gen_kwargs = parse_key_value_args(getattr(args, "gen_kwargs", None))

    max_length = resolve_limit(
        "max_length",
        [("--max_length", getattr(args, "max_length", None))]
        + [(f"model_args.{alias}", model_args.get(alias)) for alias in MODEL_LENGTH_ALIASES],
    )
    max_tokens = resolve_limit(
        "max_tokens",
        [("--max_tokens", getattr(args, "max_tokens", None))]
        + [(f"gen_kwargs.{alias}", gen_kwargs.get(alias)) for alias in MAX_OUTPUT_ALIASES],
    )
    limits = EvaluationLimits(max_length=max_length, max_tokens=max_tokens)
    limits.apply(args)
    return limits
