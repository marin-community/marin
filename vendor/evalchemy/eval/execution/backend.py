"""Execute normalized model requests against local or HTTP model backends."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
import numpy as np

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientSession, ClientTimeout, TCPConnector
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from transformers import AutoTokenizer

from eval.completion_response import FailedGeneration
from eval.contracts.eval_spec import ModelRequest, RequestKind
from eval.limits import ensure_context_window, preflight_endpoint_generation
from eval.contracts.preflight import validate_model_request
from eval.contracts.failures import FailureCategory
from eval.serve_eval.local_api import RetryingLocalChatCompletions, RetryingLocalCompletions
from eval.runtime.pool import RetryPolicy, retry
from eval.execution.run_spec import RunSpec
from eval.execution.generation import canonical_generation_kwargs
from eval.execution.prompts import clean_fastchat_response, fastchat_prompt
from eval.robust_api import configured_generation_overrides

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
GPT4O_MAX_OUTPUT_TOKENS = 16384


def prompt_encoder(model: LM, *, tokenizer_defaults: bool = False) -> Callable[[str], list[int]]:
    """Resolve token IDs for source prompt budgets without changing transport encoding."""
    if not tokenizer_defaults and not isinstance(model, LocalChatCompletion):
        return model.tok_encode

    # LocalChatCompletion.tok_encode passes text through even with a tokenizer.
    # Load only when a source needs sizing; zero-shot chat needs no local tokenizer.
    tokenizer = None

    def encode(text: str) -> list[int]:
        nonlocal tokenizer
        if tokenizer is None:
            tokenizer = model.tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model.model, trust_remote_code=True)
        return tokenizer.encode(text)

    return encode


def retryable_transport(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in RETRYABLE_STATUSES
    return isinstance(error, (ClientConnectionError, ClientPayloadError, ConnectionError, TimeoutError))


def transport_retry_delay(error: Exception) -> float:
    """Respect a server's Retry-After response within the operation deadline."""
    if not isinstance(error, ClientResponseError) or error.headers is None:
        return 0.0
    value = error.headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return 0.0


class _Admitted:
    async def acquire(self) -> bool:
        return True

    def release(self) -> None:
        pass


def _local_call(model, kind, instances, seeds, batch_setting):
    if batch_setting is not None:
        attribute, size = batch_setting
        setattr(model, attribute, size)
    if seeds is not None:
        for seed, initialize in zip(seeds, (random.seed, np.random.seed, utils.set_torch_seed), strict=True):
            if seed is not None:
                initialize(seed)
    return getattr(model, kind)(instances)


@dataclass
class LMBackend:
    """One initialized LM, used as a transport or local inference implementation."""

    model: LM
    session: ClientSession | None = None
    seeds: tuple[int | None, ...] = RunSpec.seeds
    supports_request_seed: bool = False
    batch_sizes: dict[str, int | str] = field(default_factory=dict)
    batch_size_attribute: str = "batch_size"
    _pending: list = field(default_factory=list, init=False, repr=False)
    _batch: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> LMBackend:
        if self.model.world_size != 1:
            raise ValueError("the shared runtime requires one client process; serve distributed models behind an endpoint")
        if isinstance(self.model, TemplateAPI):
            self.session = ClientSession(
                connector=TCPConnector(limit=0, ssl=self.model.verify_certificate), timeout=ClientTimeout(total=None),
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._batch is not None:
            await self._batch
        if self.session is not None:
            await self.session.close()

    async def execute(self, request: ModelRequest) -> Any:
        validate_model_request(request)
        prompt = request.prompt
        conversation = None
        if request.prompt_format is not None:
            if request.prompt_format != "fastchat":
                raise ValueError(f"unknown prompt format: {request.prompt_format}")
            prompt, conversation = fastchat_prompt(self.model, prompt)
        if isinstance(prompt, list):
            prompt = self.model.apply_chat_template(prompt, **request.chat_template)
        parameters = canonical_generation_kwargs(request.generation)
        do_sample = parameters.pop("do_sample", None)
        if isinstance(self.model, TemplateAPI):
            if "source_do_sample" in request.metadata:
                if request.metadata["source_do_sample"] is False:
                    parameters["temperature"] = 0.0
                    parameters.pop("seed", None)
                overrides = configured_generation_overrides(self.model)
                if overrides.get("do_sample") is True and "temperature" not in overrides:
                    parameters.pop("temperature", None)
                parameters.update(overrides)
                parameters.pop("do_sample", None)
            output = await self._api_request(request, prompt, parameters)
            return clean_fastchat_response(self.model, conversation, output)
        if do_sample is False:
            parameters["temperature"] = 0.0
        args = (prompt, parameters)
        if request.kind is RequestKind.LIKELIHOOD:
            args = (prompt, request.continuation)
        elif request.kind is RequestKind.ROLLING_LIKELIHOOD:
            args = (prompt,)
        instance = Instance(request.kind.value, {}, args, 0, metadata=(
            request.metadata.get("task"), request.metadata.get("ordinal"), 1,
        ))
        instance.repeat_idx = request.metadata.get("trial") or 0
        rng = None
        if request.kind is RequestKind.GENERATE and "seed" in parameters and not self.supports_request_seed:
            # HF-style generation uses process RNGs rather than a seed keyword.
            # Keep distinct trial RNGs in distinct local batches.
            rng = (parameters.pop("seed"), *(
                None if value is None else value + instance.repeat_idx for value in self.seeds[1:3]
            ))
        future = asyncio.get_running_loop().create_future()
        self._pending.append((instance, future, rng))
        if self._batch is None or self._batch.done():
            self._batch = asyncio.create_task(self._local_batches())
        return clean_fastchat_response(self.model, conversation, await future)

    async def _local_batches(self):
        # Coalesce already admitted operations for local engines. The shared
        # pool bounds this queue; no source or second dispatcher admits work.
        await asyncio.sleep(0)
        while self._pending:
            pending, self._pending = self._pending, []
            for kind, rng, task in dict.fromkeys((instance.request_type, rng, instance.metadata[0]) for instance, _, rng in pending):
                batch = [(instance, future) for instance, future, item_rng in pending
                         if instance.request_type == kind and item_rng == rng and instance.metadata[0] == task and not future.cancelled()]
                if not batch:
                    continue
                size = self.batch_sizes.get(task)
                setting = (self.batch_size_attribute, size) if size is not None else None
                try:
                    outputs = await asyncio.to_thread(_local_call, self.model, kind, [item for item, _ in batch], rng, setting)
                    if len(outputs) != len(batch):
                        raise ValueError(f"model returned {len(outputs)} outputs for {len(batch)} requests")
                except Exception as error:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(error)
                else:
                    for (_, future), output in zip(batch, outputs, strict=True):
                        if not future.done():
                            future.set_result(output)

    async def _api_request(self, request: ModelRequest, prompt: Any, parameters: dict[str, Any]) -> Any:
        model = self.model
        if request.kind is RequestKind.GENERATE:
            if "source_do_sample" in request.metadata and isinstance(model, LocalCompletionsAPI) and "4o" in model.model:
                # Local endpoints still apply explicit caller limits when building the payload.
                parameters["max_gen_toks"] = min(parameters.get("max_gen_toks", model._max_gen_toks), GPT4O_MAX_OUTPUT_TOKENS)
            if model.tokenized_requests:
                # Match TemplateAPI.generate_until before calling its transport.
                prompt = model.tok_encode(prompt, add_special_tokens=model.add_bos_token)
                context_length = model.max_length + 1
                if request.context_length is not None:
                    context_length = min(context_length, request.context_length)
                output_tokens = parameters.get("max_gen_toks", model._max_gen_toks)
                ensure_context_window(context_length=context_length, prompt_tokens=1, output_tokens=output_tokens, safety_tokens=1)
                prompt = prompt[-(context_length - output_tokens - 1):]
                return await self._call([prompt], parameters, generate=True)
            bounded, _, cap = preflight_endpoint_generation(
                tokenizer=model.tokenizer,
                payloads=[prompt],
                gen_kwargs=parameters,
                context_length=request.context_length or (model.max_length + 1 if model.max_length is not None else None),
                chat_template_kwargs=getattr(model, "_evalchemy_chat_template_kwargs", None),
            )
            if bounded is not None:
                parameters = bounded
                if cap is not None:
                    parameters["_evalchemy_context_output_cap"] = cap
            return await self._call([prompt], parameters, generate=True)
        if request.kind is RequestKind.LIKELIHOOD:
            if prompt:
                context, continuation = model._encode_pair(prompt, request.continuation)
            else:
                context, continuation = [model.prefix_token_id], model.tok_encode(request.continuation)
            return await self._likelihood(context, continuation)
        windows = map(
            utils.make_disjoint_window,
            utils.get_rolling_token_windows(
                token_list=model.tok_encode(prompt), prefix_token=model.prefix_token_id,
                max_seq_len=model.max_length - 1, context_len=1,
            ),
        )
        total = 0.0
        for context, continuation in windows:
            score, _ = await self._likelihood(context, continuation)
            total += score
        return total

    async def _likelihood(self, context: list[int], continuation: list[int]) -> Any:
        tokens = (context + continuation)[-self.model.max_length:]
        context_length = len(context) - max(0, len(context) + len(continuation) - self.model.max_length)
        return await self._call([tokens], {}, generate=False, ctxlens=[context_length])

    async def _call(self, messages: list, parameters: dict, *, generate: bool, ctxlens=None) -> Any:
        model = self.model
        bounded_transport = isinstance(model, (RetryingLocalChatCompletions, RetryingLocalCompletions))
        budget = model.transport_retry_budget if bounded_transport else None
        policy = RetryPolicy(
            budget=float(model.timeout if budget is None else budget),
            attempt_timeout=float(model.transport_attempt_timeout if bounded_transport else model.timeout),
            max_attempts=None if budget is not None else model.max_retries,
        )

        async def call():
            [output] = await model.amodel_call(
                session=self.session, sem=_Admitted(), messages=messages,
                generate=generate, gen_kwargs=parameters, ctxlens=ctxlens,
            )
            return output

        try:
            return await retry(call, policy, retryable_transport, transport_retry_delay)
        except Exception as exc:
            if not generate or not retryable_transport(exc):
                raise
            category = FailureCategory.AGENT_TIMEOUT if isinstance(exc, TimeoutError) else FailureCategory.MODEL_TRANSPORT
            return FailedGeneration(category.value)
