# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned tensor frames for exact-token generation."""

import json
import struct

import numpy as np
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from safetensors import SafetensorError
from safetensors.numpy import load, save

MEDIA_TYPE = "application/vnd.levanter.generate.v1"
GENERATE_PATH = "/v1/levanter/generate"
_MAGIC = b"LEVANTER-GENERATE\x01"
_MAX_METADATA_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class GenerateOptions(BaseModel):
    """Sampling options for an exact int32 prompt."""

    model_config = ConfigDict(extra="forbid")
    max_tokens: StrictInt = Field(gt=0)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: StrictInt | None = Field(default=None, ge=0, lt=2**32)
    stop_tokens: list[list[StrictInt]] | None = None
    n: StrictInt = Field(default=1, ge=1)


def encode_frame(metadata: dict, tensors: dict[str, np.ndarray]) -> bytes:
    """Encode JSON metadata and CPU tensors in a single frame."""
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > _MAX_METADATA_BYTES:
        raise ValueError("Binary frame metadata exceeds its size limit")
    return _MAGIC + struct.pack(">I", len(header)) + header + save(tensors)


def decode_frame(frame: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a versioned frame and validate the metadata and tensor boundaries."""
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


async def read_generate_request(request: Request) -> tuple[np.ndarray, GenerateOptions]:
    """Read a bounded generation frame with exactly one int32 token vector."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != MEDIA_TYPE:
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
        if prompt.ndim != 1 or prompt.dtype != np.int32 or prompt.size == 0:
            raise ValueError("prompt_token_ids must be a nonempty one-dimensional int32 tensor")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prompt, options
