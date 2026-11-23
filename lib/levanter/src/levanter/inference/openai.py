# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI-compatible inference API for Levanter models.

This module provides FastAPI-based endpoints that are compatible with OpenAI's
completions and chat completions APIs, allowing Levanter models to be used as
drop-in replacements for OpenAI models.
"""

import asyncio
import base64
import collections
import collections.abc
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Union, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai.types import Completion, CompletionUsage, Model
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import Choice as ChatCompletionChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceLogprobs as ChunkChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import CompletionChoice, Logprobs
from pydantic import TypeAdapter
from tokenizers.decoders import DecodeStream
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from levanter.inference.binary import GENERATE_PATH, register_binary_routes
from levanter.inference.engine import InferenceEngine, TokenSequenceLogprobs
from levanter.inference.jit_scheduler import FinishReason
from levanter.inference.runtime import (
    ChatTokenCallback,
    DEFAULT_MODEL_NAME as DEFAULT_MODEL_NAME,
    GenerationRequestError,
    InferenceBatch as InferenceBatch,
    InferenceContext as InferenceContext,
    InferenceResponse as InferenceResponse,
    InferenceServerConfig as InferenceServerConfig,
    PolicyVersionMismatch,
    WeightSource,
)
from levanter.inference.openai_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ChatTokenizeRequest,
    ChatTokenizeResponse,
    CompletionRequest,
    TokenList,
    TokensRequest,
    TokensResponse,
)
from levanter.inference.snowball_replay_payload import SnowballReplayPayload
from levanter.models.lm_model import LmHeadModel
from levanter.tokenizers import MarinTokenizer

logger = logging.getLogger(__name__)


TOKEN_ID_PREFIX = "token_id:"
_CHAT_CHUNK_ENCODER = TypeAdapter(ChatCompletionChunk)
RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {"add_generation_prompt", "continue_final_message", "chat_template", "return_dict", "tokenize"}
)


def _encode_stop_tokens(stop: Union[str, List[str], None], tokenizer: MarinTokenizer) -> Optional[List[List[int]]]:
    """Tokenize each stop string as an independent sequence."""
    if not stop:
        return None
    stop_list = [stop] if isinstance(stop, str) else stop
    stop_tokens: List[List[int]] = []
    for s in stop_list:
        stop_ids = tokenizer.encode(s, add_special_tokens=False)
        if stop_ids:
            stop_tokens.append(stop_ids)
    return stop_tokens


def _health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "levanter-inference"}


def _sse_event(payload: str) -> str:
    return f"data: {payload}\n\n"


def _chat_completion_events(
    completion: ChatCompletion,
    *,
    include_usage: bool,
    emitted_text: dict[int, list[str]] | None = None,
) -> collections.abc.Iterator[str | bytes]:
    """Emit any remaining text, the finish reasons, and optional usage."""
    for choice in completion.choices:
        prefix = "" if emitted_text is None else "".join(emitted_text.get(choice.index, ()))
        text = choice.message.content or ""
        if not text.startswith(prefix):
            raise ValueError("Final completion does not match the streamed text")
        suffix = text[len(prefix) :]
        logprobs = (
            ChunkChoiceLogprobs(content=choice.logprobs.content, refusal=choice.logprobs.refusal)
            if choice.logprobs is not None
            else None
        )
        content = ChatCompletionChunkChoice(
            index=choice.index,
            delta=ChoiceDelta(role=None if prefix else "assistant", content=suffix),
            logprobs=logprobs,
        )
        if choice.model_extra and "token_ids" in choice.model_extra:
            content = content.model_copy(update={"token_ids": choice.model_extra["token_ids"]})
        finish_fields = {"finish_reason": choice.finish_reason}
        for key in ("behavior_logprobs", "policy_logprobs", "route_capture"):
            if choice.model_extra and key in choice.model_extra:
                finish_fields[key] = choice.model_extra[key]
        finish = ChatCompletionChunkChoice(index=choice.index, delta=ChoiceDelta()).model_copy(update=finish_fields)
        for chunk_choice in (content, finish) if suffix or not prefix else (finish,):
            chunk = ChatCompletionChunk(
                id=completion.id,
                object="chat.completion.chunk",
                created=completion.created,
                model=completion.model,
                choices=[chunk_choice],
            )
            if completion.model_extra:
                chunk = chunk.model_copy(
                    update={
                        key: completion.model_extra[key]
                        for key in ("model_version", "policy_id", "prompt_token_ids")
                        if key in completion.model_extra
                    }
                )
            if chunk_choice is finish and "route_capture" in finish_fields:
                # A full-context route capture can take tens of MiB. Keep the JSON in bytes
                # and send the SSE framing separately to avoid copying the entire event.
                yield b"data: "
                yield _CHAT_CHUNK_ENCODER.dump_json(chunk)
                yield b"\n\n"
            else:
                yield _sse_event(chunk.model_dump_json())
    if include_usage:
        chunk = ChatCompletionChunk(
            id=completion.id,
            object="chat.completion.chunk",
            created=completion.created,
            model=completion.model,
            choices=[],
            usage=completion.usage,
        )
        yield _sse_event(chunk.model_dump_json())
    yield _sse_event("[DONE]")


def _completion_events(completion: Completion) -> collections.abc.Iterator[str]:
    """Render a finished text completion as an OpenAI server-sent-event stream.

    The streaming and non-streaming completion payloads share a shape, so each choice rides
    its own event. As with chat, the events are well-formed but not incremental.
    """
    for choice in completion.choices:
        chunk = Completion(
            id=completion.id,
            object="text_completion",
            created=completion.created,
            model=completion.model,
            choices=[choice],
        )
        yield _sse_event(chunk.model_dump_json())
    yield _sse_event("[DONE]")


def _decoded_token_pieces(tokenizer: MarinTokenizer, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def _token_text_offsets(tokens: List[str]) -> List[int]:
    offsets = []
    offset = 0
    for token in tokens:
        offsets.append(offset)
        offset += len(token)
    return offsets


@dataclass(frozen=True)
class CompletionLogprobData:
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[dict[str, float]]
    text_offset: List[int]


def _completion_logprobs(
    tokenizer: MarinTokenizer,
    token_ids: List[int],
    sequence_logprobs: TokenSequenceLogprobs,
    *,
    return_tokens_as_token_ids: bool = False,
) -> CompletionLogprobData:
    tokens = _decoded_token_pieces(tokenizer, token_ids)
    if len(tokens) != len(sequence_logprobs.token_logprobs):
        raise ValueError(f"Expected {len(tokens)} token logprobs, got {len(sequence_logprobs.token_logprobs)}")
    if len(tokens) != len(sequence_logprobs.top_token_logprobs):
        raise ValueError(
            f"Expected {len(tokens)} top-logprob entries, got {len(sequence_logprobs.top_token_logprobs)}"
        )

    top_logprobs = [
        {
            (
                f"{TOKEN_ID_PREFIX}{token_id}"
                if return_tokens_as_token_ids
                else tokenizer.decode([token_id], skip_special_tokens=False)
            ): logprob
            for token_id, logprob in token_logprobs.items()
        }
        for token_logprobs in sequence_logprobs.top_token_logprobs
    ]
    return CompletionLogprobData(
        tokens=[f"{TOKEN_ID_PREFIX}{token_id}" for token_id in token_ids] if return_tokens_as_token_ids else tokens,
        token_logprobs=sequence_logprobs.token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=_token_text_offsets(tokens),
    )


def _completion_prompt_ids(
    prompt: str | list[str] | list[int] | list[list[int]],
    tokenizer: MarinTokenizer,
) -> list[list[int]]:
    if isinstance(prompt, str):
        return [tokenizer.encode(prompt, add_special_tokens=False)]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must contain at least one token")
    if isinstance(prompt[0], int):
        return [cast(list[int], prompt)]
    if isinstance(prompt[0], str):
        return [tokenizer.encode(row, add_special_tokens=False) for row in cast(list[str], prompt)]
    return cast(list[list[int]], prompt)


def _validate_prompt_ids(ctx: InferenceContext, tokens: list[int]) -> None:
    if not tokens or len(tokens) >= ctx.config.service.max_seq_len:
        raise HTTPException(status_code=400, detail="Prompt must be nonempty and leave room within the context limit")
    if min(tokens) < 0 or max(tokens) >= ctx.model.Vocab.size:
        raise HTTPException(status_code=400, detail="Prompt contains a token ID outside the model vocabulary")


def _generation_options(ctx: InferenceContext, request: CompletionRequest | ChatCompletionRequest) -> dict:
    if request.return_policy_logprobs and not ctx.config.service.return_policy_logprobs:
        raise HTTPException(status_code=400, detail="This server does not have raw policy scoring enabled")
    options: dict[str, Any] = {}
    if request.expected_model_version is not None:
        options["expected_model_version"] = request.expected_model_version
    if request.expected_policy_id is not None:
        options["expected_policy_id"] = request.expected_policy_id
    if request.return_route_captures:
        options["return_route_captures"] = True
    return options


def _encode_route_capture(payload: SnowballReplayPayload) -> str:
    return base64.b64encode(payload.to_bytes()).decode("ascii")


async def _choice_metadata(request: CompletionRequest | ChatCompletionRequest, generation: InferenceResponse) -> dict:
    metadata: dict[str, Any] = {"model_version": generation.model_version, "policy_id": generation.policy_id}
    if request.return_policy_logprobs:
        metadata["policy_logprobs"] = generation.policy_logprobs
    if request.logprobs or request.return_policy_logprobs:
        metadata["behavior_logprobs"] = generation.behavior_logprobs
    if request.return_route_captures and generation.route_capture is not None:
        metadata["route_capture"] = await asyncio.to_thread(_encode_route_capture, generation.route_capture)
    return metadata


async def _create_completion(ctx: InferenceContext, request: CompletionRequest) -> Completion:
    """Create a text completion using OpenAI API format."""
    try:
        if request.echo and request.logprobs == 0:
            raise HTTPException(status_code=400, detail="Echo logprobs require a positive logprobs count")
        options = _generation_options(ctx, request)
        prompt_token_lists = _completion_prompt_ids(request.prompt, ctx.tokenizer)

        stop_tokens = _encode_stop_tokens(request.stop, ctx.tokenizer)
        if request.stop_token_ids:
            stop_tokens = [*(stop_tokens or []), *([token] for token in request.stop_token_ids)]

        # Create futures for all prompts
        futures = []
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        echo_logprobs_top_k = int(request.logprobs) if request.echo and request.logprobs else None

        for prompt_tokens in prompt_token_lists:
            scored_token_budget = len(prompt_tokens) + request.max_tokens
            if echo_logprobs_top_k is not None and scored_token_budget > ctx.config.service.max_seq_len:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "echo logprobs are not supported when prompt tokens plus max_tokens exceeds "
                        f"max_seq_len={ctx.config.service.max_seq_len}"
                    ),
                )
            _validate_prompt_ids(ctx, prompt_tokens)
            total_prompt_tokens += len(prompt_tokens)

        for prompt_tokens in prompt_token_lists:
            # Create future for this request
            future: asyncio.Future = asyncio.Future()
            futures.append(future)

            # Submit to inference thread
            ctx.submit_request(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_tokens=stop_tokens,
                seed=request.seed,
                future=future,
                n_generations=request.n or 1,
                echo_logprobs_top_k=echo_logprobs_top_k,
                **options,
            )

        # Wait for all results
        results: List[List[InferenceResponse]] = await asyncio.gather(*futures)

        # Format responses
        choice_idx = 0
        for prompt_index, (prompt_tokens, result) in enumerate(zip(prompt_token_lists, results, strict=True)):
            if isinstance(request.prompt, str):
                prompt_text = request.prompt
            elif isinstance(request.prompt[0], str):
                prompt_text = cast(list[str], request.prompt)[prompt_index]
            else:
                prompt_text = ctx.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            for generation in result:
                choice_text = prompt_text + generation.text if request.echo else generation.text

                # Format logprobs if available
                logprobs = None
                echo_unstarted = (
                    request.echo and generation.finish_reason == FinishReason.ABORT and not generation.tokens
                )
                if request.logprobs is not None and not echo_unstarted:
                    if request.echo:
                        if generation.echo_token_ids is None or generation.echo_logprobs is None:
                            raise RuntimeError("Echo logprobs requested but missing from generation result.")
                        echo_logprobs = _completion_logprobs(
                            ctx.tokenizer,
                            generation.echo_token_ids,
                            generation.echo_logprobs,
                            return_tokens_as_token_ids=request.return_tokens_as_token_ids,
                        )
                        logprobs = Logprobs(
                            tokens=echo_logprobs.tokens,
                            token_logprobs=echo_logprobs.token_logprobs,
                            text_offset=echo_logprobs.text_offset,
                            top_logprobs=echo_logprobs.top_logprobs,
                        )
                    else:
                        # Convert logprobs to API format
                        generated_tokens = generation.tokens

                        # Create token logprobs in OpenAI format
                        tokens = []
                        token_logprobs = []
                        if generation.logprobs:
                            for token_id, lp in zip(generated_tokens, generation.logprobs):
                                # Use convert_ids_to_tokens to preserve BPE format
                                token_str = (
                                    f"{TOKEN_ID_PREFIX}{token_id}"
                                    if request.return_tokens_as_token_ids
                                    else ctx.tokenizer.convert_ids_to_tokens(token_id)
                                )
                                tokens.append(token_str)
                                token_logprobs.append(float(lp))

                        logprobs = Logprobs(
                            tokens=tokens,
                            token_logprobs=token_logprobs,
                            text_offset=None,
                            top_logprobs=None,
                        )

                choices.append(
                    CompletionChoice(
                        text=choice_text,
                        index=choice_idx,
                        finish_reason="stop" if generation.finish_reason == FinishReason.STOP else "length",
                        logprobs=logprobs,
                    )
                )
                choices[-1] = choices[-1].model_copy(update=await _choice_metadata(request, generation))
                if generation.finish_reason == FinishReason.ABORT:
                    choices[-1] = choices[-1].model_copy(update={"finish_reason": "abort"})
                if request.return_token_ids:
                    choices[-1] = choices[-1].model_copy(
                        update={
                            "token_ids": generation.tokens,
                            "prompt_token_ids": prompt_tokens,
                        }
                    )
                total_completion_tokens += generation.completion_tokens
                choice_idx += 1

        return Completion(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    except HTTPException:
        raise
    except PolicyVersionMismatch as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except GenerationRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Error in completion.", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _compute_tokens(
    messages: list[ChatMessage],
    tokenizer: MarinTokenizer,
    tools: list[dict[str, object]] | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
    *,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> List[int]:
    """Encode a conversation with the tokenizer's chat template.

    A model with no chat template cannot represent a conversation, so a chat request against one
    is rejected rather than rendered into some invented format the model never saw. Base and
    midtrained checkpoints are served through ``/v1/completions`` instead.
    """
    if tokenizer.chat_template is None:
        raise HTTPException(
            status_code=400,
            detail="This model has no chat template; use /v1/completions, or serve it with a chat template.",
        )
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    template_kwargs = dict(chat_template_kwargs or {})
    overridden = sorted(RESERVED_CHAT_TEMPLATE_KWARGS.intersection(template_kwargs))
    if overridden:
        names = ", ".join(overridden)
        raise HTTPException(status_code=400, detail=f"chat_template_kwargs may not override: {names}")
    if tools is not None:
        template_kwargs["tools"] = tools
    # return_dict=False pins the token ids to a flat list; tokenizers otherwise hand back a
    # BatchEncoding here, which is the shape the rest of this module cannot use.
    result = tokenizer.apply_chat_template(
        dict_messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        return_dict=False,
        **template_kwargs,
    )
    assert isinstance(result, list)
    return result


async def _fetch_tokens(ctx: InferenceContext, request: TokensRequest) -> TokensResponse:
    """Fetch tokenized prompts after system prompt injection and encoding."""
    try:
        results = []
        for messages in request.message_list:
            token_ids = _compute_tokens(messages, ctx.tokenizer)
            results.append(TokenList(tokens=token_ids))

        return TokensResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in tokenization.", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _submit_chat_completion(
    ctx: InferenceContext, request: ChatCompletionRequest, *, on_tokens: ChatTokenCallback | None = None
) -> tuple[list[int], asyncio.Future[list[InferenceResponse]]]:
    if request.top_logprobs:
        raise HTTPException(status_code=400, detail="Generated top-logprob candidates are not supported")
    prompt_tokens = request.exact_prompt_token_ids
    if prompt_tokens is None:
        prompt_tokens = _compute_tokens(
            request.messages,
            ctx.tokenizer,
            request.tools,
            request.chat_template_kwargs,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
        )
    _validate_prompt_ids(ctx, prompt_tokens)

    stop_tokens = _encode_stop_tokens(request.stop, ctx.tokenizer)
    if request.stop_token_ids:
        stop_tokens = [*(stop_tokens or []), *([token] for token in request.stop_token_ids)]

    future: asyncio.Future[list[InferenceResponse]] = asyncio.get_running_loop().create_future()
    try:
        ctx.submit_request(
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_tokens=stop_tokens,
            seed=request.seed,
            future=future,
            n_generations=request.n or 1,
            on_tokens=on_tokens,
            **_generation_options(ctx, request),
        )
    except PolicyVersionMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GenerationRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prompt_tokens, future


async def _format_chat_completion(
    ctx: InferenceContext,
    request: ChatCompletionRequest,
    prompt_tokens: list[int],
    results: list[InferenceResponse],
    *,
    completion_id: str,
    created: int,
) -> ChatCompletion:
    # Format response
    choices = []
    total_completion_tokens = 0

    for i, generation in enumerate(results):
        # Format logprobs if available
        logprobs = None
        if request.logprobs:
            generated_tokens = generation.tokens

            # Create content logprobs in OpenAI format
            content_logprobs = []
            assert generation.logprobs is not None, "Logprobs requested but missing in generation result"
            for token_id, lp in zip(generated_tokens, generation.logprobs, strict=True):
                token_str = (
                    f"{TOKEN_ID_PREFIX}{token_id}"
                    if request.return_tokens_as_token_ids
                    else ctx.tokenizer.convert_ids_to_tokens(token_id)
                )
                content_logprobs.append(
                    ChatCompletionTokenLogprob(
                        token=token_str,
                        logprob=float(lp),
                        bytes=list(token_str.encode("utf-8")),
                        top_logprobs=[],
                    )
                )

            logprobs = ChoiceLogprobs(content=content_logprobs)

        choices.append(
            ChatCompletionChoice(
                index=i,
                message=ChatCompletionMessage(role="assistant", content=generation.text),
                finish_reason="stop" if generation.finish_reason == FinishReason.STOP else "length",
                logprobs=logprobs,
            )
        )
        choices[-1] = choices[-1].model_copy(update=await _choice_metadata(request, generation))
        if generation.finish_reason == FinishReason.ABORT:
            choices[-1] = choices[-1].model_copy(update={"finish_reason": "abort"})
        if request.return_token_ids:
            choices[-1] = choices[-1].model_copy(update={"token_ids": generation.tokens})
        total_completion_tokens += generation.completion_tokens

    response = ChatCompletion(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=total_completion_tokens,
            total_tokens=len(prompt_tokens) + total_completion_tokens,
        ),
    )
    response = response.model_copy(
        update={"model_version": results[0].model_version, "policy_id": results[0].policy_id}
    )
    if request.return_token_ids:
        response = response.model_copy(update={"prompt_token_ids": prompt_tokens})
    return response


async def _create_chat_completion(ctx: InferenceContext, request: ChatCompletionRequest) -> ChatCompletion:
    try:
        prompt_tokens, future = _submit_chat_completion(ctx, request)
        return await _format_chat_completion(
            ctx,
            request,
            prompt_tokens,
            await future,
            completion_id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
        )
    except HTTPException:
        raise
    except GenerationRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _chat_stream_tokenizer(tokenizer: MarinTokenizer) -> PreTrainedTokenizerFast | None:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    if not tokenizer.clean_up_tokenization_spaces:
        return tokenizer
    # Transformers skips the whitespace cleanup for BPE unless explicitly enabled.
    if isinstance(tokenizer.backend_tokenizer.model, BPE) and not getattr(
        tokenizer, "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output", True
    ):
        return tokenizer
    return None


async def _stream_chat_completion(
    ctx: InferenceContext, request: ChatCompletionRequest, tokenizer: PreTrainedTokenizerFast, *, include_usage: bool
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[tuple[int, list[int], int] | None] = asyncio.Queue()
    closed = threading.Event()

    def on_tokens(choice: int, tokens: list[int], model_version: int) -> None:
        if not closed.is_set():
            loop.call_soon_threadsafe(updates.put_nowait, (choice, tokens, model_version))

    def on_done(future: asyncio.Future[list[InferenceResponse]]) -> None:
        if not future.cancelled():
            future.exception()  # The client may have disconnected before awaiting the result.
        if not closed.is_set():
            updates.put_nowait(None)

    prompt_tokens, future = _submit_chat_completion(ctx, request, on_tokens=on_tokens)
    future.add_done_callback(on_done)
    header = ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        choices=[],
        created=int(time.time()),
        model=request.model,
        object="chat.completion.chunk",
    )
    try:
        first = await updates.get()
        if first is None:
            future.result()  # Preserve the HTTP error status if generation failed before any tokens.
    except asyncio.CancelledError:
        closed.set()
        raise
    except GenerationRequestError as exc:
        closed.set()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        closed.set()
        logger.exception("Error before chat streaming started")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def events() -> collections.abc.AsyncIterator[str | bytes]:
        decoders: dict[int, DecodeStream] = {}
        emitted: dict[int, list[str]] = {}
        update = first
        try:
            while update is not None:
                choice, tokens, version = update
                if choice not in decoders:
                    decoders[choice] = DecodeStream(skip_special_tokens=True)
                text = decoders[choice].step(tokenizer.backend_tokenizer, tokens)
                if text:
                    pieces = emitted.setdefault(choice, [])
                    delta = ChoiceDelta(role=None if pieces else "assistant", content=text)
                    pieces.append(text)
                    chunk = header.model_copy(
                        update={
                            "choices": [ChatCompletionChunkChoice(index=choice, delta=delta)],
                            "model_version": version,
                        }
                    )
                    yield _sse_event(chunk.model_dump_json())
                update = await updates.get()
            completion = await _format_chat_completion(
                ctx, request, prompt_tokens, future.result(), completion_id=header.id, created=header.created
            )
            for event in _chat_completion_events(completion, include_usage=include_usage, emitted_text=emitted):
                yield event
        except Exception:
            logger.exception("Error during chat streaming")
            yield _sse_event('{"error":{"message":"Chat generation failed","type":"server_error"}}')
        finally:
            closed.set()

    return StreamingResponse(events(), media_type="text/event-stream")


class InferenceServer:
    """Wraps a FastAPI server around the inference context.

    Provides OpenAI compatible endpoints for text and chat completions.
    """

    _server: uvicorn.Server | None
    app: FastAPI
    config: InferenceServerConfig
    inference_context: InferenceContext

    def __init__(self, config: InferenceServerConfig, inference_context: InferenceContext, app: FastAPI):
        """Initialize the inference server with pre-built components.

        Use InferenceServer.create() to build a new server instance.
        """
        self.config = config
        self.inference_context = inference_context
        self.app = app
        self._server = None

    @staticmethod
    def create(config: InferenceServerConfig, model: LmHeadModel, tokenizer: MarinTokenizer) -> "InferenceServer":
        """Create and initialize a new InferenceServer.

        This factory method loads the model, tokenizer, and creates all necessary
        components for the inference server.
        """
        service = InferenceEngine.from_model_with_config(
            model=model,
            tokenizer=tokenizer,
            config=config.service,
            axis_resources=config.trainer.compute_axis_mapping,
        )

        # Create and start inference thread
        inference_context = InferenceContext(model, tokenizer, service, config)
        inference_context.start()

        # Create FastAPI app with initialized context
        app = InferenceServer._create_app(inference_context)
        return InferenceServer(config, inference_context, app)

    @staticmethod
    def _create_app(inference_context: InferenceContext) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title="Levanter Inference Service", version="1.0.0")
        model_name = inference_context.config.model_name
        register_binary_routes(app, inference_context)

        # Register routes with thin wrappers that call helper functions
        @app.get("/health")
        async def health_check():
            return _health_check()

        @app.get("/v1/models")
        async def list_models() -> dict:
            model = Model(id=model_name, object="model", created=int(time.time()), owned_by="levanter")
            version, policy_id = inference_context.policy_identity()
            return {
                "object": "list",
                "data": [
                    {
                        **model.model_dump(mode="json"),
                        "max_model_len": inference_context.config.service.max_seq_len,
                        "model_version": version,
                        "policy_id": policy_id,
                        "levanter": {
                            "binary_generate": GENERATE_PATH,
                            "policy_logprobs": inference_context.config.service.return_policy_logprobs,
                            "route_captures": inference_context.config.capture_routes,
                        },
                    }
                ],
            }

        # A streaming request returns a StreamingResponse, which FastAPI passes through
        # untouched; `response_model` still describes the non-streaming body.
        @app.post("/v1/chat/completions", response_model=ChatCompletion)
        async def create_chat_completion(request: ChatCompletionRequest):
            include_usage = (request.stream_options or {}).get("include_usage") is True
            if request.stream and not request.logprobs and not request.return_token_ids:
                tokenizer = _chat_stream_tokenizer(inference_context.tokenizer)
                if tokenizer is not None:
                    return await _stream_chat_completion(
                        inference_context, request, tokenizer, include_usage=include_usage
                    )
            completion = await _create_chat_completion(inference_context, request)
            if not request.stream:
                return completion
            return StreamingResponse(
                _chat_completion_events(completion, include_usage=include_usage), media_type="text/event-stream"
            )

        @app.post("/v1/completions", response_model=Completion)
        async def create_completion(request: CompletionRequest):
            completion = await _create_completion(inference_context, request)
            if not request.stream:
                return completion
            return StreamingResponse(_completion_events(completion), media_type="text/event-stream")

        @app.post("/tokenize", response_model=ChatTokenizeResponse)
        async def tokenize_chat(request: ChatTokenizeRequest) -> ChatTokenizeResponse:
            tokens = _compute_tokens(
                request.messages,
                inference_context.tokenizer,
                request.tools,
                request.chat_template_kwargs,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
            )
            return ChatTokenizeResponse(
                tokens=tokens, count=len(tokens), max_model_len=inference_context.config.service.max_seq_len
            )

        @app.post("/v1/tokens", response_model=TokensResponse)
        async def fetch_tokens(request: TokensRequest) -> TokensResponse:
            return await _fetch_tokens(inference_context, request)

        return app

    def pause_generation(self) -> None:
        """Abort current requests and clear cache state before weight replacement."""
        self.inference_context.pause_generation()

    def resume_generation(self) -> None:
        """Resume request admission after the pause barrier."""
        self.inference_context.resume_generation()

    def unload(self):
        """Unload the inference model to free up resources."""
        self.inference_context.unload()

    @property
    def model_version(self) -> int:
        return self.inference_context.model_version

    def reload(self, weight_callback: WeightSource, *, expected_version: int, policy_id: str | None = None) -> int:
        """Install staged weights and return the new version after clearing serving state."""
        return self.inference_context.reload(weight_callback, expected_version=expected_version, policy_id=policy_id)

    def address(self):
        """Get the full address the server is running on."""
        for server in self._server.servers:
            for sock in server.sockets:
                addr = sock.getsockname()
                host, port = addr[0], addr[1]
                # handle weird ipv6 localhost address which confuses clients
                if host == "::1" or host == ":1":
                    host = "localhost"
                return f"{host}:{port}"
        return None

    def port(self):
        """Get the port the server is running on."""
        if self.config.port > 0:
            return self.config.port

        # query the uvicorn server socket list for the port
        for server in self._server.servers:
            for sock in server.sockets:
                addr = sock.getsockname()
                return addr[1]

        return None

    def serve(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            self._server = uvicorn.Server(uvicorn.Config(self.app, host=self.config.host, port=self.config.port))
            self._server.run()
        finally:
            self.shutdown()

    async def serve_async(self):
        try:
            logger.info(f"Starting Levanter inference server on {self.config.host}:{self.config.port}")
            config = uvicorn.Config(self.app, host=self.config.host, port=self.config.port)
            self._server = uvicorn.Server(config)
            await self._server.serve()
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the inference context."""
        self.inference_context.shutdown()
