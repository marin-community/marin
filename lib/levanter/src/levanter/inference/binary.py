# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned tensor generation API over the same queue as OpenAI serving."""

import asyncio
import json
import struct

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from safetensors import SafetensorError
from safetensors.numpy import load, save

from levanter.inference.runtime import (
    GenerationRequestError,
    InferenceContext,
    InferenceResponse,
    PolicyVersionMismatch,
)

MEDIA_TYPE = "application/vnd.levanter.generate.v1"
GENERATE_PATH = "/v1/levanter/generate"
_MAGIC = b"LEVANTER-GENERATE\x01"
_MAX_METADATA_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class GenerateOptions(BaseModel):
    """Tensor generation options; the prompt is carried as an int32 vector."""

    model_config = ConfigDict(extra="forbid")
    max_tokens: StrictInt = Field(gt=0)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: StrictInt | None = Field(default=None, ge=0, lt=2**32)
    stop_tokens: list[list[StrictInt]] | None = None
    n: StrictInt = Field(default=1, ge=1)
    expected_model_version: StrictInt | None = Field(default=None, ge=0)
    expected_policy_id: str | None = None
    return_route_captures: bool = False
    return_policy_logprobs: bool = False


def encode_frame(metadata: dict, tensors: dict[str, np.ndarray]) -> bytes:
    """Encode metadata and named CPU tensors, with no object deserialization."""
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > _MAX_METADATA_BYTES:
        raise ValueError("Binary frame metadata exceeds its size limit")
    return _MAGIC + struct.pack(">I", len(header)) + header + save(tensors)


def decode_frame(frame: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Read the versioned frame and its tensor table."""
    prefix = len(_MAGIC)
    if not frame.startswith(_MAGIC) or len(frame) < prefix + 4:
        raise ValueError("Invalid or unsupported generation frame")
    (header_size,) = struct.unpack_from(">I", frame, prefix)
    start = prefix + 4
    stop = start + header_size
    if not 0 < header_size <= _MAX_METADATA_BYTES or stop >= len(frame):
        raise ValueError("Invalid generation frame lengths")
    try:
        metadata = json.loads(frame[start:stop])
        tensors = load(frame[stop:])
    except (ValueError, RecursionError, SafetensorError) as exc:
        raise ValueError("Invalid generation frame metadata or tensors") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Generation frame metadata must be an object")
    return metadata, tensors


def _encode_response(prompt: np.ndarray, outputs: list[InferenceResponse], *, policy_scores: bool) -> bytes:
    tensors = {"prompt_token_ids": prompt}
    choices = []
    for choice, output in enumerate(outputs):
        choices.append(
            {
                "finish_reason": output.finish_reason.name.lower(),
                "model_version": output.model_version,
                "policy_id": output.policy_id,
            }
        )
        prefix = f"choice.{choice}."
        tensors[prefix + "token_ids"] = np.asarray(output.tokens, dtype=np.int32)
        tensors[prefix + "behavior_logprobs"] = np.asarray(output.behavior_logprobs, dtype=np.float32)
        if policy_scores and output.policy_logprobs is not None:
            tensors[prefix + "policy_logprobs"] = np.asarray(output.policy_logprobs, dtype=np.float32)
        if output.route_capture is not None:
            tensors[prefix + "route_capture"] = np.frombuffer(output.route_capture.to_bytes(), dtype=np.uint8)
    return encode_frame({"choices": choices}, tensors)


def register_binary_routes(app: FastAPI, runtime: InferenceContext) -> None:
    """Register the request-response adapter on the shared inference runtime."""

    @app.post(GENERATE_PATH)
    async def generate(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";")[0] != MEDIA_TYPE:
            raise HTTPException(status_code=415, detail=f"Expected {MEDIA_TYPE}")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Generation frame exceeds the request size limit")
            body.extend(chunk)
        try:
            metadata, tensors = decode_frame(bytes(body))
            options = GenerateOptions.model_validate(metadata)
            if tensors.keys() != {"prompt_token_ids"}:
                raise ValueError("Generation requires exactly one prompt_token_ids tensor")
            prompt = tensors["prompt_token_ids"]
            if prompt.ndim != 1 or prompt.dtype != np.int32:
                raise ValueError("prompt_token_ids must be a one-dimensional int32 tensor")
            if options.return_policy_logprobs and not runtime.config.service.return_policy_logprobs:
                raise ValueError("This server does not have raw policy scoring enabled")
            future = asyncio.get_running_loop().create_future()
            runtime.submit_request(
                prompt.tolist(),
                options.max_tokens,
                options.temperature,
                options.top_p,
                options.stop_tokens,
                options.seed,
                future,
                n_generations=options.n,
                expected_model_version=options.expected_model_version,
                expected_policy_id=options.expected_policy_id,
                return_route_captures=options.return_route_captures,
            )
        except PolicyVersionMismatch as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            outputs = await future
        except GenerationRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if options.return_route_captures:
            body = await asyncio.to_thread(
                _encode_response, prompt, outputs, policy_scores=options.return_policy_logprobs
            )
        else:
            body = _encode_response(prompt, outputs, policy_scores=options.return_policy_logprobs)
        return Response(body, media_type=MEDIA_TYPE)
