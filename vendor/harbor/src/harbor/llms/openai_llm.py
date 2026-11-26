import asyncio
import functools
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import httpx
import openai
from openai import AsyncOpenAI
from upath import UPath

try:
    # orjson is an RL-image speedup (3-10x faster than stdlib json on the large
    # vLLM bodies TIS produces). It is an OPTIONAL dependency: images that do not
    # install it (non-RL eval/datagen) fall back transparently to stdlib json in
    # _loads_raw_response below, so this import must never hard-fail harbor.
    import orjson
except ImportError:
    orjson = None

from harbor.literal import extract_literal_from_response
from harbor.llms.base import (
    BaseLLM,
    ContextBudgetExceededError,
    ContextLengthExceededError,
    LLMRequestTimeoutError,
    LLMResponse,
    OpenAITransportConnectTimeoutError,
    OutputLengthExceededError,
)
from harbor.llms.utils import (
    add_anthropic_caching,
    add_vllm_reasoning,
    validate_hosted_vllm_model_config,
)
from harbor.llms.vllm_tokenization import tokenize_chat
from harbor.models.metric import UsageInfo
from harbor.telemetry.observer import FAILURE_KIND_LLM_REQUEST, record_failure
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.utils.logger import logger
from harbor_config.model_name import split_provider_model_name
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_DEFAULT_OPENAI_CONNECT_TIMEOUT_SEC = 20.0
_OPENAI_CONNECT_TIMEOUT_ENV = "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC"


def _openai_connect_timeout_sec(env: Mapping[str, str] | None = None) -> float:
    """Return the positive, finite OpenAI transport connect timeout."""
    raw = (os.environ if env is None else env).get(_OPENAI_CONNECT_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_OPENAI_CONNECT_TIMEOUT_SEC
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{_OPENAI_CONNECT_TIMEOUT_ENV} must be a positive finite number, "
            f"got {raw!r}"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{_OPENAI_CONNECT_TIMEOUT_ENV} must be a positive finite number, "
            f"got {raw!r}"
        )
    return value


def _openai_transport_timeout(
    env: Mapping[str, str] | None = None,
) -> httpx.Timeout:
    return httpx.Timeout(
        connect=_openai_connect_timeout_sec(env),
        read=None,
        write=600.0,
        pool=600.0,
    )


_OPENAI_TRANSPORT_TIMEOUT = _openai_transport_timeout()


@dataclass(frozen=True)
class LLMRequestMetadata:
    request_id: str
    session_id: str | None
    request_sequence: int
    model_name: str
    connect_timeout_sec: float | None
    read_timeout_sec: float | None
    write_timeout_sec: float | None
    pool_timeout_sec: float | None

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "request_sequence": self.request_sequence,
            "model_name": self.model_name,
            "connect_timeout_sec": self.connect_timeout_sec,
            "read_timeout_sec": self.read_timeout_sec,
            "write_timeout_sec": self.write_timeout_sec,
            "pool_timeout_sec": self.pool_timeout_sec,
        }

    def to_timeout_error(self, timeout_phase: str) -> LLMRequestTimeoutError:
        error_type = (
            OpenAITransportConnectTimeoutError
            if timeout_phase == "connect"
            else LLMRequestTimeoutError
        )
        return error_type(
            request_id=self.request_id,
            session_id=self.session_id,
            request_sequence=self.request_sequence,
            model_name=self.model_name,
            timeout_phase=timeout_phase,
            connect_timeout_sec=self.connect_timeout_sec,
            read_timeout_sec=self.read_timeout_sec,
            write_timeout_sec=self.write_timeout_sec,
            pool_timeout_sec=self.pool_timeout_sec,
        )


_TIMEOUT_PHASE_BY_CLASS_NAME: dict[str, str] = {
    "ConnectTimeout": "connect",
    "ReadTimeout": "read",
    "WriteTimeout": "write",
    "PoolTimeout": "pool",
}
_HTTP_TIMEOUT_MODULE_ROOTS = frozenset({"httpx", "httpx2", "httpcore", "httpcore2"})


def _timeout_phase(error: openai.APITimeoutError) -> str:
    cause = error.__cause__
    if isinstance(cause, httpx.ConnectTimeout):
        return "connect"
    if isinstance(cause, httpx.ReadTimeout):
        return "read"
    if isinstance(cause, httpx.WriteTimeout):
        return "write"
    if isinstance(cause, httpx.PoolTimeout):
        return "pool"
    cause_type = type(cause)
    module_root = cause_type.__module__.split(".", 1)[0]
    if module_root in _HTTP_TIMEOUT_MODULE_ROOTS:
        return _TIMEOUT_PHASE_BY_CLASS_NAME.get(cause_type.__name__, "unknown")
    return "unknown"


def _is_retryable_llm_error(error: BaseException) -> bool:
    """Retry policy for ``OpenAILLM.call``.

    Only the dedicated OpenAI connection-timeout subtype is retried among request
    timeouts. It represents a failure before request bytes reached the server.
    """
    if not isinstance(error, Exception):
        return False
    if isinstance(error, OpenAITransportConnectTimeoutError):
        return True
    if isinstance(
        error,
        (
            ContextLengthExceededError,
            ContextBudgetExceededError,
            LLMRequestTimeoutError,
            OutputLengthExceededError,
            openai.AuthenticationError,
        ),
    ):
        return False
    if isinstance(error, LLMRequestTimeoutError):
        return error.timeout_phase == "connect"
    return True


def _loads_raw_response(content: bytes, text: str) -> dict[str, Any]:
    """Parse a raw vLLM JSON body, preferring orjson (3-10x faster than stdlib).

    Called via ``asyncio.to_thread`` on the AsyncOpenAI raw path: the parse is
    GIL-bound (neither orjson nor stdlib json releases the GIL), and on the async
    RolloutCoordinator loop a large body — vLLM echoes ``prompt_token_ids`` every
    turn once TIS requests ``logprobs`` + ``return_token_ids`` — otherwise stalls
    every co-resident trial's dispatch (the batch-of-8 rollout-supply sawtooth).
    orjson rejects non-finite floats (e.g. ``-Infinity`` logprobs) that stdlib
    accepts, so on ANY orjson failure fall back to stdlib json, which is the
    correctness arbiter. When orjson is absent this is a transparent stdlib no-op.
    """
    if orjson is not None:
        try:
            return orjson.loads(content)
        except Exception:
            pass  # non-finite floats / any orjson quirk -> stdlib is authoritative
    return json.loads(text)


_OPENAI_CHAT_PARAMS = frozenset(
    {
        "messages",
        "model",
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
        # openai SDK passthrough params (not part of the request body schema).
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
)
_OPENAI_COMPLETION_PARAMS = frozenset(
    {
        "model",
        "prompt",
        "best_of",
        "echo",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "suffix",
        "temperature",
        "top_p",
        "user",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
)
_OPENAI_RESPONSES_PARAMS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
)
# _llm_kwargs (constructor extras) that configure the CLIENT, not a request —
# stripped before they can reach a completion/responses call.
_CLIENT_ONLY_LLM_KWARGS = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "drop_params",
        "logger_fn",
        "custom_llm_provider",
        "num_retries",
        "max_retries",
        "mock_response",
    }
)


# --- libuv-adjacent boundary fix: vLLM context safety buffer ---
# The proactive context guard computes free_tokens = context_limit -
# current_tokens, but current_tokens comes from a token counter that can
# disagree with vLLM's BPE by +/-1 at the boundary. When the count undercounts
# by 1, a prompt that the guard believes fits at exactly max_model_len renders
# to max_model_len + 1 and vLLM rejects it ("32769 input tokens"). We shave a
# few tokens off the vLLM-reported max_model_len so the effective limit sits
# just below the hard wall and a +/-1 boundary prompt can never render past it.
# Default 8 tokens; override via HARBOR_VLLM_CONTEXT_SAFETY_BUFFER. Only ever
# applied to the live vLLM limit (the hard wall), never to the configured value.
try:
    _VLLM_CONTEXT_SAFETY_BUFFER = int(
        os.environ.get("HARBOR_VLLM_CONTEXT_SAFETY_BUFFER", "8")
    )
except (TypeError, ValueError):
    _VLLM_CONTEXT_SAFETY_BUFFER = 8
if _VLLM_CONTEXT_SAFETY_BUFFER < 0:
    _VLLM_CONTEXT_SAFETY_BUFFER = 0


_VLLM_MAX_MODEL_LEN_CACHE: dict[str, int | None] = {}


# Reasoning markers emitted inline by models served through text completions,
# where vLLM's chat-serving reasoning parser does not run.
_COMPLETION_REASONING_MARKERS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|start_think|>", "<|end_think|>"),
)

_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE = [
    {"role": "user", "content": "I am a user."},
    {
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "thinking",
    },
]

# Same conversation with different assistant content. Rendering both and taking
# the longest common tail locates where assistant content ends without assuming
# how the content itself tokenizes.
_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE_ALT = [
    {"role": "user", "content": "I am a user."},
    {
        "role": "assistant",
        "content": "different",
        "reasoning_content": "thinking",
    },
]
_TOKEN_IN_TOKEN_OUT_TEMPLATE_FOLLOWUP = {
    "role": "user",
    "content": "I am another user.",
}


@dataclass(frozen=True)
class _TokenContinuationTemplate:
    base_ids: list[int]
    assistant_content_end: int


def _common_suffix(left: list[int], right: list[int]) -> list[int]:
    """Return the longest shared tail of two token sequences."""
    n = 0
    while n < min(len(left), len(right)) and left[-1 - n] == right[-1 - n]:
        n += 1
    return left[len(left) - n :] if n else []


def _suffix_not_already_generated(
    template_suffix: list[int], completion_token_ids: list[int]
) -> list[int]:
    """Return the template suffix after its overlap with generated tokens."""
    for overlap in range(min(len(template_suffix), len(completion_token_ids)), 0, -1):
        if completion_token_ids[-overlap:] == template_suffix[:overlap]:
            return template_suffix[overlap:]
    return list(template_suffix)


def _is_openai_compatible_route(
    provider_prefix: str | None, api_base: str | None
) -> bool:
    """Identify OpenAI-compatible endpoints and unprefixed OpenAI models."""
    if provider_prefix in (None, "hosted_vllm", "openai"):
        return True
    if api_base:
        low = api_base.rstrip("/").lower()
        if low.endswith("/v1") or "/v1/" in low:
            return True
    return False


class OpenAILLM(BaseLLM):
    # One-shot guard for the Task-A rollout-logprobs-empty diagnostic (see call()).
    # Class-level so the warning fires at most once per worker process under
    # high-concurrency rollout generation.
    _logprob_diag_emitted: bool = False

    def __init__(
        self,
        model_name: str,
        temperature: float | None = None,
        api_base: str | None = None,
        session_id: str | None = None,
        collect_rollout_details: bool = False,
        max_thinking_tokens: int | None = None,
        reasoning_effort: str | None = None,
        model_info: dict[str, Any] | None = None,
        use_responses_api: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._model_name = model_name
        self._llm_kwargs = kwargs
        self._temperature = temperature
        self._model_info = model_info
        self._logger = logger.getChild(__name__)
        self._openai_client: AsyncOpenAI | None = None
        self._request_sequence = 0

        hosted_vllm_validation: tuple[str, dict[str, Any]] | None = None
        if "hosted_vllm" in model_name.lower():
            hosted_vllm_validation = validate_hosted_vllm_model_config(
                model_name, self._model_info
            )

        (
            self._provider_prefix,
            self._canonical_model_name,
        ) = split_provider_model_name(model_name)

        if hosted_vllm_validation is not None:
            self._provider_prefix = "hosted_vllm"
            self._canonical_model_name, self._model_info = hosted_vllm_validation

        self._api_base = api_base
        self._use_openai_transport = _is_openai_compatible_route(
            self._provider_prefix, api_base
        )
        self._session_id = session_id
        self._collect_rollout_details = collect_rollout_details
        self._max_thinking_tokens = max_thinking_tokens
        self._reasoning_effort = reasoning_effort
        self._token_in_token_out_templates: dict[str, _TokenContinuationTemplate] = {}

        self._use_responses_api = use_responses_api

    @property
    def _display_name(self) -> str:
        """Get a display name for logging that shows both canonical and original names if different."""
        lookup_name = self._canonical_model_name
        if lookup_name != self._model_name:
            return f"{lookup_name} (from '{self._model_name}')"
        return lookup_name

    async def build_continuation_prompt_token_ids(
        self,
        prompt: str,
        previous_prompt_token_ids: list[int],
        previous_completion_token_ids: list[int],
        **kwargs,
    ) -> list[int] | None:
        """Append a newly rendered user turn without re-tokenizing model output."""
        if (
            not self._collect_rollout_details
            or not self._api_base
            or not self._use_openai_transport
        ):
            return None

        template_options: dict[str, Any] = {}
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            for key in ("chat_template", "chat_template_kwargs", "tools"):
                if key in extra_body:
                    template_options[key] = extra_body[key]
        if "tools" in kwargs:
            template_options["tools"] = kwargs["tools"]

        template_key = json.dumps(template_options, sort_keys=True, default=str)
        template = self._token_in_token_out_templates.get(template_key)
        if template is None:
            base_ids = await tokenize_chat(
                api_base=self._api_base,
                model=self._canonical_model_name,
                messages=[
                    *_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE,
                    _TOKEN_IN_TOKEN_OUT_TEMPLATE_FOLLOWUP,
                ],
                add_generation_prompt=True,
                template_options=template_options,
            )
            alt_ids = await tokenize_chat(
                api_base=self._api_base,
                model=self._canonical_model_name,
                messages=[
                    *_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE_ALT,
                    _TOKEN_IN_TOKEN_OUT_TEMPLATE_FOLLOWUP,
                ],
                add_generation_prompt=True,
                template_options=template_options,
            )
            template = _TokenContinuationTemplate(
                base_ids=base_ids,
                assistant_content_end=len(base_ids)
                - len(_common_suffix(base_ids, alt_ids)),
            )
            self._token_in_token_out_templates[template_key] = template
        continued_ids = await tokenize_chat(
            api_base=self._api_base,
            model=self._canonical_model_name,
            messages=[
                *_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE,
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            template_options=template_options,
        )
        if (
            continued_ids[: template.assistant_content_end]
            != template.base_ids[: template.assistant_content_end]
        ):
            raise ValueError(
                "The vLLM chat template does not have a stable continuation prefix"
            )

        # Take the continuation from the end of assistant content in the extended
        # render. This includes the template's actual between-turn suffix, which
        # can differ from its end-of-conversation suffix. The model normally emits
        # the leading stop token itself, so drop only that overlapping prefix.
        continuation_ids = _suffix_not_already_generated(
            continued_ids[template.assistant_content_end :],
            list(previous_completion_token_ids),
        )
        return [
            *previous_prompt_token_ids,
            *previous_completion_token_ids,
            *continuation_ids,
        ]

    def _get_openai_client(self) -> AsyncOpenAI:
        """Reuse one OpenAI client; Harbor owns retries and request timeouts."""
        if not _is_openai_compatible_route(self._provider_prefix, self._api_base):
            raise ValueError(
                "Configure an OpenAI-compatible api_base for this provider"
            )
        if self._openai_client is None:
            api_key = (
                self._llm_kwargs.get("api_key")
                or os.environ.get("OPENAI_API_KEY")
                # Self-hosted vLLM ignores the key, but the SDK requires a
                # non-empty one.
                or "EMPTY"
            )
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "max_retries": 0,
                "timeout": _OPENAI_TRANSPORT_TIMEOUT,
            }
            if self._api_base:
                client_kwargs["base_url"] = self._api_base
            self._openai_client = AsyncOpenAI(**client_kwargs)
        return self._openai_client

    def _prepare_request_metadata(
        self, request_kwargs: dict[str, Any]
    ) -> LLMRequestMetadata:
        """Attach a correlation ID and capture the effective read deadline."""
        headers = dict(request_kwargs.get("extra_headers") or {})
        request_id = next(
            (
                str(value)
                for key, value in headers.items()
                if key.lower() == "x-request-id"
            ),
            None,
        )
        if request_id is None:
            request_id = f"harbor-{uuid.uuid4()}"
            headers["X-Request-Id"] = request_id
        request_kwargs["extra_headers"] = headers

        self._request_sequence += 1
        configured_timeout = request_kwargs.get("timeout", _OPENAI_TRANSPORT_TIMEOUT)
        if isinstance(configured_timeout, httpx.Timeout):
            timeout_config = configured_timeout
        elif configured_timeout is None:
            timeout_config = httpx.Timeout(None)
        else:
            timeout_config = httpx.Timeout(float(configured_timeout))
        return LLMRequestMetadata(
            request_id=request_id,
            session_id=self._session_id,
            request_sequence=self._request_sequence,
            model_name=self._model_name,
            connect_timeout_sec=timeout_config.connect,
            read_timeout_sec=timeout_config.read,
            write_timeout_sec=timeout_config.write,
            pool_timeout_sec=timeout_config.pool,
        )

    def _get_vllm_max_model_len(self) -> int | None:
        """Query the live vLLM server's `max_model_len` via `/v1/models`.

        The server's max_model_len is authoritative for served checkpoints.
        Use it to bound explicit model_info limits and avoid oversized requests
        when model_info is unavailable.
        Cached per instance AND process-globally (see _VLLM_MAX_MODEL_LEN_CACHE);
        returns None when no vLLM api_base is set or the probe fails.
        """
        if not self._api_base:
            return None
        if hasattr(self, "_cached_vllm_max_model_len"):
            return self._cached_vllm_max_model_len

        # Process-global cache keyed by endpoint: identical for every OpenAILLM
        # instance pointing at this vLLM server, so a fresh instance never
        # re-probes (no inline /v1/models GET on the coordinator loop). See the
        # _VLLM_MAX_MODEL_LEN_CACHE comment above.
        cache_key = self._api_base.rstrip("/")
        if cache_key in _VLLM_MAX_MODEL_LEN_CACHE:
            limit = _VLLM_MAX_MODEL_LEN_CACHE[cache_key]
            self._cached_vllm_max_model_len = limit
            return limit

        limit: int | None = None
        try:
            import httpx

            base = self._api_base.rstrip("/")
            # /v1/models is an OpenAI-compatible endpoint under /v1 (unlike
            # /tokenize, which lives at the server root).
            url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                for entry in resp.json().get("data", []):
                    mml = entry.get("max_model_len")
                    if mml:
                        limit = int(mml)
                        break
        except Exception:
            limit = None

        _VLLM_MAX_MODEL_LEN_CACHE[cache_key] = limit
        self._cached_vllm_max_model_len = limit
        return limit

    def get_model_context_limit(self) -> int:
        """Get the context limit (max input tokens) for the current model.

        Prefers the live vLLM server's `max_model_len` (the value vLLM actually
        enforces), falling back to model_info, then to a large constant.
        When both are available we take the smaller (most conservative).

        Returns:
            int: The maximum input tokens the model can accept, or a fallback value if unavailable.
        """
        # The context limit is CONSTANT for the life of this OpenAILLM instance
        # (fixed model + fixed vLLM server). Memoize the final resolved limit so
        # every subsequent per-turn check is a pure attribute read and issues NO
        # HTTP.
        if getattr(self, "_cached_context_limit", None) is not None:
            return self._cached_context_limit

        fallback_context_limit = 1000000

        model_info = self._model_info or {}
        configured_limit = model_info.get("max_input_tokens") or model_info.get(
            "max_tokens"
        )

        vllm_limit = self._get_vllm_max_model_len()
        # Shave the safety buffer off the live vLLM wall so a +/-1 BPE boundary
        # prompt cannot render to max_model_len + 1. Registry limit is left as-is.
        if vllm_limit is not None and _VLLM_CONTEXT_SAFETY_BUFFER:
            vllm_limit = max(1, vllm_limit - _VLLM_CONTEXT_SAFETY_BUFFER)

        candidates = [c for c in (configured_limit, vllm_limit) if c]
        if candidates:
            self._cached_context_limit = min(candidates)
            return self._cached_context_limit

        self._logger.warning(
            f"No context limit available for '{self._display_name}' from the vLLM "
            f"server or model_info. Using fallback context limit: {fallback_context_limit}"
        )
        # Do NOT memoize the fallback: it means both the vLLM probe and the
        # configured limit were unavailable this call (e.g. server not up yet). Leaving
        # it uncached lets a later call resolve the real limit once ready.
        return fallback_context_limit

    def get_model_output_limit(self) -> int | None:
        """Get the output limit (max output tokens) for the current model.

        Returns:
            int | None: The maximum output tokens the model can generate, or None if unavailable.
        """
        return (self._model_info or {}).get("max_output_tokens")

    def _clean_value(self, value):
        match value:
            case _ if callable(value):
                return None
            case dict():
                return {
                    k: v
                    for k, v in {
                        k: self._clean_value(v) for k, v in value.items()
                    }.items()
                    if v is not None
                }
            case list():
                return [
                    self._clean_value(v)
                    for v in value
                    if self._clean_value(v) is not None
                ]
            case str() | int() | float() | bool():
                return value
            case _:
                return str(value)

    def _format_episode_log(
        self,
        request_kwargs: dict[str, Any],
        response: dict[str, Any],
    ) -> str:
        """Serialize the raw request+response (with logprobs/token_ids).

        Called after an API response: only reached when a
        ``logging_path`` is given (Terminus-2 passes None by default via
        ``enable_episode_logging=False``, so the RL hot path never writes). Any
        api key present in the request is hashed, never persisted in the clear.
        The JSON pass is multi-MB; never call this on the event loop.
        """
        payload = {"request": request_kwargs, "response": response}
        clean_dict = self._clean_value(payload)
        if isinstance(clean_dict, dict):
            req = clean_dict.get("request")
            if isinstance(req, dict):
                for key in ("api_key", "x-api-key"):
                    if key in req:
                        req[f"{key}_sha256"] = hashlib.sha256(
                            str(req[key]).encode()
                        ).hexdigest()
                        del req[key]
        return json.dumps(clean_dict, indent=4)

    def _enqueue_episode_log(
        self,
        logging_path: Path | UPath,
        request_kwargs: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Enqueue the episode debug log on the background artifact writer.

        Snapshots the mutable request containers (kwargs dict, messages list)
        on the caller so a chat history that grows after this call does not
        leak later turns into this episode's log; the expensive serialization
        and the (possibly remote) write run on the writer's worker pool.
        """
        request_snapshot = dict(request_kwargs)
        messages = request_snapshot.get("messages")
        if isinstance(messages, list):
            request_snapshot["messages"] = list(messages)
        input_items = request_snapshot.get("input")
        if isinstance(input_items, list):
            request_snapshot["input"] = list(input_items)
        get_artifact_writer().submit_optional(
            logging_path,
            functools.partial(self._format_episode_log, request_snapshot, response),
            description="episode debug log",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        retry=retry_if_exception(_is_retryable_llm_error),
        reraise=True,
    )
    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | UPath | None = None,
        **kwargs,
    ) -> LLMResponse:
        prompt_token_ids = kwargs.pop("prompt_token_ids", None)
        if prompt_token_ids is not None:
            return await self._call_token_completion(
                prompt_token_ids=prompt_token_ids,
                logging_path=logging_path,
                response_format=response_format,
                **kwargs,
            )
        if self._use_responses_api:
            return await self._call_responses(
                prompt, message_history, response_format, logging_path, **kwargs
            )

        kwargs.pop("previous_response_id", None)
        completion_kwargs: dict[str, Any] | None = None
        request_metadata: LLMRequestMetadata | None = None

        # Prepare messages with caching for Anthropic models
        messages = message_history + [{"role": "user", "content": prompt}]
        messages = add_anthropic_caching(messages, self._model_name)
        if self._provider_prefix == "hosted_vllm":
            messages = add_vllm_reasoning(messages)

        try:
            # Build completion_kwargs with all parameters
            completion_kwargs = {
                **self._build_base_kwargs(),
                "messages": messages,
                "response_format": response_format,
                "reasoning_effort": self._reasoning_effort,
            }
            if self._temperature is not None:
                completion_kwargs["temperature"] = self._temperature

            # Add logprobs and return_token_ids if rollout details collection is enabled
            if self._collect_rollout_details:
                completion_kwargs["logprobs"] = True
                # Explicitly request top_logprobs=0 alongside logprobs=True.
                #
                # vLLM's OpenAI chat-completion serving gates logprobs emission on
                # `request.logprobs AND request.top_logprobs is not None`. Sending
                # top_logprobs=0 keeps the sampled-token logprob (no top-k
                # overhead) while making the field unambiguously present.
                completion_kwargs["top_logprobs"] = 0
                # Request token IDs from provider (supported by vLLM). Some
                # providers (e.g. OpenAI) reject this; we catch+retry without it.
                if "extra_body" not in completion_kwargs:
                    completion_kwargs["extra_body"] = {}
                extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                extra_body["return_token_ids"] = True

            # Add any additional kwargs, deep-merging extra_body to preserve
            # internally-set fields (e.g., return_token_ids) when caller also
            # passes extra_body via llm_call_kwargs.
            if "extra_body" in completion_kwargs and "extra_body" in kwargs:
                existing_extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                new_extra_body: dict[str, Any] = kwargs.pop("extra_body")
                completion_kwargs["extra_body"] = {
                    **existing_extra_body,
                    **new_extra_body,
                }
            elif "extra_body" in kwargs:
                kwargs["extra_body"] = {**kwargs["extra_body"]}
            completion_kwargs.update(kwargs)
            request_metadata = self._prepare_request_metadata(completion_kwargs)

            # Reject an invalid output budget before dispatch. Clamping it to one
            # token hides the budgeting defect and produces a predictably unusable
            # completion instead of a truthful typed failure.
            if completion_kwargs.get("max_tokens") is not None:
                if completion_kwargs["max_tokens"] < 1:
                    raise ContextBudgetExceededError(
                        f"max_tokens must be positive, got "
                        f"{completion_kwargs['max_tokens']}"
                    )

            # Add thinking parameter for Anthropic models if max_thinking_tokens
            # is set. `thinking` is not an openai chat param, so it rides in
            # extra_body (the openai SDK forwards extra_body into the JSON body).
            if self._max_thinking_tokens is not None and (
                "anthropic" in self._model_name.lower()
                or "claude" in self._model_name.lower()
            ):
                budget = self._max_thinking_tokens
                if budget < 1024:
                    self._logger.warning(
                        f"max_thinking_tokens={budget} is below minimum of 1024. "
                        "Using minimum value of 1024."
                    )
                    budget = 1024
                if "extra_body" not in completion_kwargs:
                    completion_kwargs["extra_body"] = {}
                extra_body = completion_kwargs["extra_body"]  # type: ignore[assignment]
                extra_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

            # Add session_id to extra_body if available
            if self._session_id is not None:
                if "extra_body" not in completion_kwargs:
                    completion_kwargs["extra_body"] = {}
                extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                extra_body["session_id"] = self._session_id

            try:
                response = await self._dispatch_chat(completion_kwargs)
            except openai.BadRequestError as e:
                # If provider (e.g., OpenAI) rejects extra_body parameters, retry without them.
                error_msg = str(e)
                if (
                    "Unrecognized request argument" in error_msg
                    and "extra_body" in completion_kwargs
                ):
                    rejected_params = []
                    if "return_token_ids" in error_msg:
                        rejected_params.append("return_token_ids")
                    if "session_id" in error_msg:
                        rejected_params.append("session_id")

                    if rejected_params:
                        self._logger.warning(
                            f"Provider {self._model_name} rejected extra_body parameters: {', '.join(rejected_params)}. "
                            f"Retrying without them. Token IDs will not be available."
                        )
                        extra_body_val = completion_kwargs.get("extra_body")
                        if extra_body_val and isinstance(extra_body_val, dict):
                            extra_body = extra_body_val
                            for param in rejected_params:
                                if param in extra_body:
                                    del extra_body[param]

                            if not extra_body:
                                del completion_kwargs["extra_body"]

                        response = await self._dispatch_chat(completion_kwargs)
                    else:
                        raise e
                else:
                    raise e
        except Exception as e:
            self._handle_llm_error(
                e,
                request_metadata=request_metadata,
                logging_path=logging_path,
                request_kwargs=completion_kwargs,
            )

        # Optionally persist the raw request+response (with logprobs/token_ids).
        # Best-effort: serialization/write errors are logged by the writer.
        if logging_path is not None:
            self._enqueue_episode_log(logging_path, completion_kwargs, response)

        # Extract usage information
        usage_info = self._extract_usage_info(response)

        # Extract and process token IDs if rollout details collection is enabled
        prompt_token_ids = None
        completion_token_ids = None
        logprobs = None
        extra = None

        if self._collect_rollout_details:
            literal = extract_literal_from_response(response)
            prompt_token_ids = literal["prompt_token_ids"]
            completion_token_ids = literal["completion_token_ids"]
            logprobs = literal["logprobs"]
            extra = self._extract_provider_extra(response)
            # One-shot diagnostic (Task A: 80B R3+TIS all-None rollout logprobs).
            # Emit the actual response-choice shape ONCE per process so the next
            # run reveals whether vLLM dropped `logprobs` vs the engine genuinely
            # not emitting them. Free at 300-concurrent (gated + one-shot).
            if not logprobs and not OpenAILLM._logprob_diag_emitted:
                OpenAILLM._logprob_diag_emitted = True
                try:
                    choices = response.get("choices", []) or []
                    ch0 = choices[0] if choices else None
                    lp = ch0.get("logprobs") if isinstance(ch0, dict) else None
                    psf = (
                        ch0.get("provider_specific_fields")
                        if isinstance(ch0, dict)
                        else None
                    )
                    self._logger.warning(
                        "[TIS-diag] rollout logprobs EMPTY despite collect_rollout_details. "
                        "choice0_keys=%s logprobs_type=%s logprobs_repr=%.300s "
                        "completion_token_ids_present=%s provider_specific_fields_keys=%s",
                        (
                            sorted(ch0.keys())
                            if isinstance(ch0, dict)
                            else type(ch0).__name__
                        ),
                        type(lp).__name__,
                        repr(lp),
                        completion_token_ids is not None,
                        sorted(psf.keys()) if isinstance(psf, dict) else psf,
                    )
                except Exception as diag_exc:  # pragma: no cover - diagnostic only
                    self._logger.warning(
                        f"[TIS-diag] failed to introspect response: {diag_exc}"
                    )

        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls") or None

        llm_response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=response.get("model"),
            usage=usage_info,
            response_id=response.get("id"),
            tool_calls=tool_calls,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs=logprobs,
            extra=extra,
        )

        # Sometimes the LLM returns a response with a finish reason of "length"
        # This typically means we hit the max_tokens limit, not the context window
        if choice.get("finish_reason") == "length":
            # Create exception with truncated response attached
            exc = OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. "
                f"Response was truncated. Consider increasing max_tokens if possible.",
                partial_response=llm_response,
            )
            raise exc

        return llm_response

    async def _call_token_completion(
        self,
        *,
        prompt_token_ids: list[int],
        logging_path: Path | UPath | None,
        **kwargs,
    ) -> LLMResponse:
        """Generate from an exact token prompt through vLLM's completion API."""
        if not self._use_openai_transport:
            raise ValueError(
                "Exact token prompts require an OpenAI-compatible endpoint"
            )
        kwargs.pop("previous_response_id", None)
        if kwargs.pop("response_format", None) is not None:
            raise ValueError(
                "response_format is not supported with exact token prompts"
            )

        completion_kwargs = {
            **self._build_base_kwargs(),
            "prompt": prompt_token_ids,
        }
        if self._temperature is not None:
            completion_kwargs["temperature"] = self._temperature
        completion_kwargs.update(kwargs)

        extra_body = dict(completion_kwargs.get("extra_body") or {})
        configured_template_kwargs = extra_body.pop("chat_template_kwargs", {})
        template_kwargs = (
            configured_template_kwargs
            if isinstance(configured_template_kwargs, dict)
            else {}
        )
        extra_body.pop("chat_template", None)
        extra_body.pop("tools", None)
        extra_body["return_token_ids"] = True
        extra_body["add_special_tokens"] = False
        if self._session_id is not None:
            extra_body["session_id"] = self._session_id
        completion_kwargs["extra_body"] = extra_body
        completion_kwargs["logprobs"] = 0
        completion_kwargs.pop("top_logprobs", None)
        completion_kwargs.pop("messages", None)
        completion_kwargs.pop("reasoning_effort", None)

        context_limit = self.get_model_context_limit()
        remaining_tokens = context_limit - len(prompt_token_ids)
        if remaining_tokens < 1:
            raise ContextBudgetExceededError(
                "The exact token prompt leaves no room for generation"
            )
        # Keep the requested output allowance separate from the context room so
        # a length stop can be routed to the appropriate recovery path.
        output_budget = completion_kwargs.get("max_tokens")
        if output_budget is None:
            output_budget = self.get_model_output_limit() or None
        completion_kwargs["max_tokens"] = (
            remaining_tokens
            if output_budget is None
            else min(output_budget, remaining_tokens)
        )

        request_metadata = self._prepare_request_metadata(completion_kwargs)
        try:
            response = await self._dispatch_completion(completion_kwargs)
        except Exception as error:
            self._handle_llm_error(
                error,
                request_metadata=request_metadata,
                logging_path=logging_path,
                request_kwargs=completion_kwargs,
            )

        if logging_path is not None:
            self._enqueue_episode_log(logging_path, completion_kwargs, response)

        literal = extract_literal_from_response(response)
        choice = response["choices"][0]
        raw_content = choice.get("text") or ""
        reasoning_content, content = self._split_completion_reasoning(
            raw_content, template_kwargs
        )
        llm_response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=response.get("model"),
            usage=self._extract_usage_info(response),
            prompt_token_ids=literal["prompt_token_ids"],
            completion_token_ids=literal["completion_token_ids"],
            logprobs=literal["logprobs"],
            extra=self._extract_provider_extra(response),
            response_id=response.get("id"),
        )
        if choice.get("finish_reason") == "length":
            # A full context requires context recovery even when the generation
            # also consumed its output allowance: another continuation cannot fit.
            completion_ids = literal["completion_token_ids"]
            completion_length = (
                len(completion_ids)
                if completion_ids is not None
                else (llm_response.usage.completion_tokens if llm_response.usage else 0)
            )
            if len(prompt_token_ids) + completion_length >= context_limit:
                raise ContextLengthExceededError(
                    f"Model {self._model_name} reached its context length: "
                    f"{len(prompt_token_ids)} prompt + {completion_length} "
                    f"completion tokens against a {context_limit}-token limit. "
                    "The generation was cut by the context window, not by the "
                    "output-token limit."
                )
            raise OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. "
                "Response was truncated. Consider increasing max_tokens if possible.",
                partial_response=llm_response,
            )
        return llm_response

    def _split_completion_reasoning(
        self,
        raw_content: str,
        template_kwargs: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Split inline reasoning markers returned by text completions."""
        thinking_enabled = template_kwargs.get("enable_thinking")
        if thinking_enabled is False:
            return None, raw_content

        for open_marker, close_marker in _COMPLETION_REASONING_MARKERS:
            before_start, marker, reasoning_and_content = raw_content.partition(
                open_marker
            )
            if marker and not before_start.strip():
                reasoning, end_marker, content = reasoning_and_content.partition(
                    close_marker
                )
                return (reasoning, content) if end_marker else (reasoning, "")

        if thinking_enabled is True or "qwen3" in self._model_name.lower():
            for _, close_marker in _COMPLETION_REASONING_MARKERS:
                reasoning, marker, content = raw_content.partition(close_marker)
                if marker:
                    return reasoning, content
            return raw_content, ""
        return None, raw_content

    async def _acreate_chat_raw(
        self, completion_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch a chat completion and return the byte-exact raw vLLM JSON.

        Streaming stays disabled on this path. Filters kwargs down
        to the params the openai SDK accepts (drop_params replacement) and reads
        the raw response body via ``with_raw_response`` so vLLM's vendor fields
        (``token_ids`` / ``prompt_token_ids``) reach the extractor un-coerced.
        """
        if completion_kwargs.get("stream"):
            raise NotImplementedError("Streaming is not supported for T bench yet")
        call_kwargs = {
            k: v
            for k, v in completion_kwargs.items()
            if k in _OPENAI_CHAT_PARAMS and v is not None
        }
        client = self._get_openai_client()
        raw_response = await client.chat.completions.with_raw_response.create(
            **call_kwargs
        )
        # Offload the JSON parse off the coordinator's asyncio loop. The body is
        # large (per-turn ``prompt_token_ids`` echo under TIS) and the parse is
        # GIL-bound; running it inline pins the loop ~10-30ms/turn and starves
        # every co-resident trial's dispatch (root cause of the generation
        # sawtooth). orjson + to_thread shrink and interleave the hold.
        return await asyncio.to_thread(
            _loads_raw_response, raw_response.content, raw_response.text
        )

    async def _acreate_completion_raw(
        self, completion_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch a token prompt and retain vLLM's literal response fields."""
        if completion_kwargs.get("stream"):
            raise NotImplementedError("Streaming is not supported for T bench yet")
        call_kwargs = {
            key: value
            for key, value in completion_kwargs.items()
            if key in _OPENAI_COMPLETION_PARAMS and value is not None
        }
        client = self._get_openai_client()
        raw_response = await client.completions.with_raw_response.create(**call_kwargs)
        return await asyncio.to_thread(
            _loads_raw_response, raw_response.content, raw_response.text
        )

    async def _dispatch_completion(
        self, completion_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return await self._acreate_completion_raw(completion_kwargs)
        except Exception as error:
            record_failure(FAILURE_KIND_LLM_REQUEST, error)
            raise

    async def _dispatch_chat(self, completion_kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._acreate_chat_raw(completion_kwargs)
        except Exception as error:
            record_failure(FAILURE_KIND_LLM_REQUEST, error)
            raise

    async def _acreate_response(self, responses_kwargs: dict[str, Any]):
        """Dispatch a Responses API call, returning the parsed openai Response."""
        call_kwargs = {
            k: v
            for k, v in responses_kwargs.items()
            if k in _OPENAI_RESPONSES_PARAMS and v is not None
        }
        client = self._get_openai_client()
        try:
            return await client.responses.create(**call_kwargs)
        except Exception as error:
            record_failure(FAILURE_KIND_LLM_REQUEST, error)
            raise

    def _extract_provider_extra(self, response) -> dict[str, Any] | None:
        """Extract non-token_ids fields from a choice's provider_specific_fields.

        Returns all fields except 'token_ids' (handled by the literal extractor)
        — e.g. router expert indices for MoE models. Operates on the raw response
        dict.

        Returns:
            Dictionary of extra provider-specific data, or None if not available.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None

            choice = choices[0]
            psf = (
                choice.get("provider_specific_fields")
                if isinstance(choice, dict)
                else None
            )
            extra = (
                {key: value for key, value in psf.items() if key != "token_ids"}
                if isinstance(psf, dict)
                else {}
            )
            for key in ("routed_experts", "router_logits"):
                if key in choice:
                    extra[key] = choice[key]
            return extra if extra else None

        except Exception as e:
            self._logger.debug(f"Error extracting provider extra fields: {e}")
            return None

    def _extract_usage_info(self, response) -> UsageInfo | None:
        """Extract token usage from a raw chat-completion response dict.

        Cost is hardcoded to 0.0: the completions served here are self-hosted
        vLLM (zero marginal cost), and the raw response carries no cost field.
        """
        try:
            usage = response.get("usage") if isinstance(response, dict) else None
            if not usage:
                return None

            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0

            cache_tokens = 0
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cache_tokens = details.get("cached_tokens", 0) or 0

            return UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_tokens=cache_tokens,
                cost_usd=0.0,
            )
        except (AttributeError, TypeError):
            return None

    def _extract_responses_usage_info(self, response) -> UsageInfo | None:
        """Extract token usage from a responses API response object.

        Cost is hardcoded to 0.0 (self-hosted vLLM).

        Returns:
            UsageInfo with token counts, or None if not available.
        """
        if not hasattr(response, "usage") or response.usage is None:
            return None

        usage = response.usage
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_tokens=0,
            cost_usd=0.0,
        )

    def _handle_llm_error(
        self,
        e: Exception,
        *,
        request_metadata: LLMRequestMetadata | None,
        logging_path: Path | UPath | None,
        request_kwargs: dict[str, Any] | None,
    ) -> NoReturn:
        """Translate openai exceptions into harbor exceptions.

        Always re-raises; never returns normally.
        """
        if isinstance(e, openai.APITimeoutError):
            if request_metadata is None:
                raise RuntimeError(
                    "OpenAI transport timed out before request metadata was prepared"
                ) from e
            timeout_error = request_metadata.to_timeout_error(_timeout_phase(e))
            self._logger.error("%s: %s", type(timeout_error).__name__, timeout_error)
            if logging_path is not None and request_kwargs is not None:
                self._enqueue_episode_log(
                    logging_path,
                    request_kwargs,
                    {
                        "error": {
                            "exception_type": type(timeout_error).__name__,
                            "message": str(timeout_error),
                            "timeout_phase": timeout_error.timeout_phase,
                            **request_metadata.as_dict(),
                        }
                    },
                )
            raise timeout_error from e
        if isinstance(e, openai.AuthenticationError):
            raise e
        if isinstance(e, openai.BadRequestError):
            if self._is_context_length_error(e):
                raise ContextLengthExceededError from e
            if self._is_context_budget_exhausted_error(e):
                raise ContextBudgetExceededError from e
        # openai surfaces vLLM's context-window rejections as a 400 BadRequest
        # (handled above); other APIStatusError subclasses may also carry a
        # context-length body, so string-check them too before re-raising.
        if isinstance(e, openai.APIStatusError):
            if self._is_context_length_error(e):
                raise ContextLengthExceededError from e
        raise e

    def _build_base_kwargs(self) -> dict[str, Any]:
        """Build the base kwargs shared by both completion and responses calls.

        Spreads the constructor's extra ``_llm_kwargs`` EXCEPT client-config keys
        (api_key/api_base/etc., which configure the AsyncOpenAI client, not a
        request), and pins the served ``model`` to the canonical (provider-prefix
        stripped) name the vLLM/OpenAI endpoint expects.
        """
        request_kwargs = {
            k: v
            for k, v in self._llm_kwargs.items()
            if k not in _CLIENT_ONLY_LLM_KWARGS
        }
        request_kwargs["model"] = self._canonical_model_name
        return request_kwargs

    def _is_context_length_error(self, error: Exception) -> bool:
        """Check provider error payloads for context-length overflow signals."""

        parts = [
            str(error),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
            str(getattr(error, "error", "")),
        ]

        combined = " ".join(part.lower() for part in parts if part)
        phrases = (
            "context length exceeded",
            "context_length_exceeded",
            "maximum context length",
            "`inputs` tokens + `max_new_tokens`",
            "model's context length",  # vllm 0.16.0 error message
            "prompt is too long",  # Anthropic via OpenAI-compatible proxies
            "input is too long for requested model",  # Bedrock via proxy
        )
        return any(phrase in combined for phrase in phrases)

    # vLLM rejects a request whose effective output allowance is zero with a
    # structured 400. The distinctive, narrow signal is `param=max_tokens` with
    # `value=0` ("max_tokens must be at least 1, got 0"). Matching ONLY this
    # payload keeps a genuine misconfiguration (an unrelated 400) as
    # BadRequestError rather than mislabelling it as context exhaustion.
    _MAX_TOKENS_ZERO_PHRASES = (
        "max_tokens must be at least 1, got 0",
        "max_tokens must be at least 1, got -",  # negative budget variants
    )

    def _is_context_budget_exhausted_error(self, error: Exception) -> bool:
        """True iff the provider rejected the request because the derived
        ``max_tokens`` was non-positive (the context-budget-exhausted signal)."""
        parts = [
            str(error),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
            str(getattr(error, "error", "")),
        ]
        combined = " ".join(part.lower() for part in parts if part)
        return any(phrase in combined for phrase in self._MAX_TOKENS_ZERO_PHRASES)

    async def _call_responses(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | UPath | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call the OpenAI Responses API via AsyncOpenAI.responses.create().

        When previous_response_id is provided (via kwargs), uses server-side
        state chaining — only the new user prompt is sent. Otherwise, builds
        the full input from message_history + prompt.
        """
        previous_response_id = kwargs.pop("previous_response_id", None)
        responses_kwargs: dict[str, Any] | None = None
        request_metadata: LLMRequestMetadata | None = None

        try:
            # Build responses_kwargs
            responses_kwargs: dict[str, Any] = self._build_base_kwargs()

            if self._reasoning_effort is not None:
                responses_kwargs["reasoning"] = {
                    "effort": self._reasoning_effort,
                }

            # Get max_output_tokens from model info
            max_output_tokens = self.get_model_output_limit()
            if max_output_tokens is not None:
                responses_kwargs["max_output_tokens"] = max_output_tokens

            if response_format is not None:
                responses_kwargs["response_format"] = response_format

            if self._temperature is not None:
                responses_kwargs["temperature"] = self._temperature
            responses_kwargs.update(kwargs)

            if previous_response_id is not None:
                # Server-side state chaining: only send the new prompt
                responses_kwargs["previous_response_id"] = previous_response_id
                responses_kwargs["input"] = prompt
            else:
                # Build full input from message history + new prompt
                input_items = []
                for msg in message_history:
                    role = (
                        msg.get("role", "user")
                        if isinstance(msg, dict)
                        else getattr(msg, "role", "user")
                    )
                    content = (
                        msg.get("content", "")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                    )
                    input_items.append({"role": role, "content": content})
                input_items.append({"role": "user", "content": prompt})
                responses_kwargs["input"] = input_items

            request_metadata = self._prepare_request_metadata(responses_kwargs)
            response = await self._acreate_response(responses_kwargs)

        except Exception as e:
            self._handle_llm_error(
                e,
                request_metadata=request_metadata,
                logging_path=logging_path,
                request_kwargs=responses_kwargs,
            )

        # Optionally persist the raw request+response.
        # Best-effort: serialization/write errors are logged by the writer.
        if logging_path is not None:
            dumped = (
                response.model_dump() if isinstance(response, BaseModel) else response
            )
            self._enqueue_episode_log(logging_path, responses_kwargs, dumped)

        # Extract text content from response.output
        content = ""
        reasoning_content = None
        for output_item in response.output:
            if getattr(output_item, "type", None) == "message":
                for content_part in getattr(output_item, "content", []):
                    if getattr(content_part, "type", None) == "output_text":
                        content += getattr(content_part, "text", "")

        # Extract usage information
        usage_info = self._extract_responses_usage_info(response)
        response_id = getattr(response, "id", None)
        llm_response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=getattr(response, "model", None),
            usage=usage_info,
            response_id=response_id,
        )

        # Check for truncation via response status
        response_status = getattr(response, "status", None)
        if response_status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            reason = (
                getattr(incomplete_details, "reason", "unknown")
                if incomplete_details
                else "unknown"
            )
            if reason == "max_output_tokens":
                raise OutputLengthExceededError(
                    f"Model {self._model_name} hit max_tokens limit. "
                    f"Response was truncated.",
                    partial_response=llm_response,
                )

        return llm_response
