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

from levanter.layers.sampler import LogprobMode

MEDIA_TYPE = "application/vnd.levanter.generate.v1"
GENERATE_PATH = "/v1/levanter/generate"
PROMPT_TOKEN_IDS = "prompt_token_ids"
TOKEN_IDS = "token_ids"
BEHAVIOR_LOGPROBS = "behavior_logprobs"
POLICY_LOGPROBS = "policy_logprobs"
EXPERT_IDS = "expert_ids"
EXPERT_VALID = "expert_valid"
LOGPROB_MODE = "logprob_mode"
MAX_POLICY_ID_LENGTH = 256
_MAGIC = b"LEVANTER-GENERATE\x01"
_HEADER_LENGTH = struct.Struct(">I")
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
    return_route_captures: bool = False
    expected_policy_id: str | None = Field(default=None, min_length=1, max_length=MAX_POLICY_ID_LENGTH)
    expected_logprob_mode: LogprobMode | None = None


def encode_frame(metadata: dict, tensors: dict[str, np.ndarray]) -> bytes:
    """Encode JSON metadata and CPU tensors in a single frame."""
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > _MAX_METADATA_BYTES:
        raise ValueError("Binary frame metadata exceeds its size limit")
    return _MAGIC + _HEADER_LENGTH.pack(len(header)) + header + save(tensors)


def decode_frame(frame: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a versioned frame and validate the metadata and tensor boundaries."""
    prefix = len(_MAGIC)
    if not frame.startswith(_MAGIC) or len(frame) < prefix + _HEADER_LENGTH.size:
        raise ValueError("Invalid or unsupported generation frame")
    (header_size,) = _HEADER_LENGTH.unpack_from(frame, prefix)
    start = prefix + _HEADER_LENGTH.size
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
        if tensors.keys() != {PROMPT_TOKEN_IDS}:
            raise ValueError("Generation requires exactly one prompt_token_ids tensor")
        prompt = tensors[PROMPT_TOKEN_IDS]
        if prompt.ndim != 1 or prompt.dtype != np.int32 or prompt.size == 0:
            raise ValueError("prompt_token_ids must be a nonempty one-dimensional int32 tensor")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prompt, options


def choice_tensor_key(choice: int, field: str) -> str:
    return f"choice.{choice}.{field}"


def choice_expert_routes(
    tensors: dict[str, np.ndarray], choice: int, input_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return captured IDs and validity when the frame tokens match the learner input."""
    try:
        prompt = tensors[PROMPT_TOKEN_IDS]
        generated = tensors[choice_tensor_key(choice, TOKEN_IDS)]
        expert_ids = tensors[choice_tensor_key(choice, EXPERT_IDS)]
        valid = tensors[choice_tensor_key(choice, EXPERT_VALID)]
    except KeyError as exc:
        raise ValueError("Response has no captured expert routes for this choice") from exc
    if prompt.ndim != 1 or generated.ndim != 1 or prompt.dtype != np.int32 or generated.dtype != np.int32:
        raise ValueError("Replay requires flat int32 prompt and generated tokens")
    if not np.array_equal(np.concatenate((prompt, generated)), input_ids):
        raise ValueError("Captured token IDs differ from the learner input")
    if (
        expert_ids.ndim != 3
        or expert_ids.shape[1] != input_ids.size
        or expert_ids.dtype not in (np.uint8, np.int32)
        or valid.shape != input_ids.shape
        or valid.dtype != bool
    ):
        raise ValueError("Invalid expert routes or token validity in generation frame")
    return expert_ids, valid
