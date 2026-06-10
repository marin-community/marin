# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for staging remote vLLM metadata locally before inflight startup."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import tempfile
from urllib.parse import urlparse

from rigging.filesystem import url_to_fs

from .vllm import vLLMInferenceContextConfig

logger = logging.getLogger(__name__)

_VLLM_METADATA_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "marin-rl-vllm-metadata")
_VLLM_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "params.json",
)


def prepare_vllm_inference_config_for_inflight(
    inference_config: vLLMInferenceContextConfig,
) -> vLLMInferenceContextConfig:
    """Stage remote model metadata locally for inflight vLLM startup."""
    model_path = inference_config.engine.model_name
    if urlparse(model_path).scheme not in {"gs", "s3"}:
        return inference_config

    local_model_path = stage_vllm_metadata_locally(model_path)
    logger.info(
        "Using local staged vLLM metadata for inflight rollout startup: %s -> %s",
        model_path,
        local_model_path,
    )
    return dataclasses.replace(
        inference_config,
        engine=dataclasses.replace(
            inference_config.engine,
            model_name=local_model_path,
            load_format="dummy",
        ),
    )


def stage_vllm_metadata_locally(model_path: str) -> str:
    """Copy the minimum vLLM metadata set for a remote model into a local cache."""
    fs, fs_path = url_to_fs(model_path)
    cache_key = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    local_dir = os.path.join(_VLLM_METADATA_CACHE_ROOT, cache_key)
    os.makedirs(local_dir, exist_ok=True)

    for filename in _VLLM_METADATA_FILES:
        remote_path = os.path.join(fs_path, filename)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            continue
        if not fs.exists(remote_path):
            continue
        fs.get(remote_path, local_path)

    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Failed to stage config.json for vLLM metadata from {model_path}")

    return local_dir
