# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model path resolution and automatic inference sharding."""

import json

from levanter.model_cache import resolve_cached_model_path
from rigging.filesystem import StoragePath
from transformers import AutoConfig

from marin.inference.vllm_server import _is_object_store_path

_MODEL_CACHE_PREFIX = "quick-serve-models"


def select_tensor_parallel_size(
    num_attention_heads: int,
    num_chips: int,
    num_key_value_heads: int | None = None,
) -> int:
    """Pick the largest valid power-of-two tensor-parallel size."""

    if num_chips < 1:
        return 1
    best = 1
    candidate = 1
    while candidate <= num_chips:
        if num_attention_heads % candidate == 0 and _kv_heads_compatible(num_key_value_heads, candidate):
            best = candidate
        candidate *= 2
    return best


def _kv_heads_compatible(num_key_value_heads: int | None, tensor_parallel_size: int) -> bool:
    if not num_key_value_heads:
        return True
    return num_key_value_heads % tensor_parallel_size == 0 or tensor_parallel_size % num_key_value_heads == 0


def read_attention_heads(model: str) -> tuple[int, int | None]:
    """Return attention and KV head counts from an HF or object-store config."""

    config_dict = _read_model_config_dict(model)
    for scope in (config_dict, config_dict.get("text_config"), config_dict.get("llm_config")):
        if not isinstance(scope, dict):
            continue
        heads = scope.get("num_attention_heads")
        if heads:
            kv_heads = scope.get("num_key_value_heads")
            return int(heads), (int(kv_heads) if kv_heads else None)
    raise ValueError(f"Could not find num_attention_heads in the model config for {model!r}.")


def _read_model_config_dict(model: str) -> dict:
    if _is_object_store_path(model):
        return json.loads((StoragePath(model) / "config.json").read_text())
    return AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()


def resolve_model_path(model: str, cache_ttl_days: int) -> str:
    """Resolve and optionally mirror an HF model to the region-local cache."""

    return resolve_cached_model_path(model, cache_ttl_days=cache_ttl_days, cache_prefix=_MODEL_CACHE_PREFIX)
