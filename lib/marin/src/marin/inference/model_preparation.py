# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model path resolution and automatic inference sharding."""

import hashlib
import json
import logging
import os
import posixpath
import tempfile
from pathlib import Path, PurePosixPath

from levanter.model_cache import resolve_cached_model_path
from rigging.filesystem import StoragePath, url_to_fs
from transformers import AutoConfig

from marin.inference.vllm_server import _is_object_store_path

_MODEL_CACHE_PREFIX = "quick-serve-models"
_LOCAL_MODEL_CACHE_DIR = "marin-models"
_MODEL_CONFIG_FILENAME = "config.json"

logger = logging.getLogger(__name__)


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


def read_attention_heads(model: str, revision: str | None = None) -> tuple[int, int | None]:
    """Return attention and KV head counts from an HF or object-store config."""

    config_dict = _read_model_config_dict(model, revision)
    for scope in (config_dict, config_dict.get("text_config"), config_dict.get("llm_config")):
        if not isinstance(scope, dict):
            continue
        heads = scope.get("num_attention_heads")
        if heads:
            kv_heads = scope.get("num_key_value_heads")
            return int(heads), (int(kv_heads) if kv_heads else None)
    raise ValueError(f"Could not find num_attention_heads in the model config for {model!r}.")


def _read_model_config_dict(model: str, revision: str | None = None) -> dict:
    if _is_object_store_path(model):
        return json.loads((StoragePath(model) / _MODEL_CONFIG_FILENAME).read_text())
    return AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=True).to_dict()


def resolve_model_path(model: str, cache_ttl_days: int, revision: str | None = None) -> str:
    """Resolve and optionally mirror an HF model to the region-local cache."""

    if revision is None or _is_object_store_path(model):
        return resolve_cached_model_path(model, cache_ttl_days=cache_ttl_days, cache_prefix=_MODEL_CACHE_PREFIX)
    pinned_model = f"{model}@{revision}"
    resolved = resolve_cached_model_path(
        pinned_model,
        cache_ttl_days=cache_ttl_days,
        cache_prefix=_MODEL_CACHE_PREFIX,
    )
    # With caching disabled, vLLM receives the bare model plus its separate revision argument.
    return model if resolved == pinned_model else resolved


def stage_object_store_model_locally(model: str, staging_root: Path | None = None) -> str:
    """Copy an object-store model snapshot to deterministic worker-local storage."""
    if not _is_object_store_path(model):
        return model

    filesystem, remote_root = url_to_fs(model)
    if not remote_root:
        raise ValueError(f"Object-store model path must include a checkpoint prefix: {model}")

    remote_entries = filesystem.find(remote_root, detail=True)
    if not remote_entries:
        raise FileNotFoundError(f"No files found under object-store model path: {model}")
    expected_config = posixpath.join(remote_root, _MODEL_CONFIG_FILENAME)
    if expected_config not in remote_entries:
        raise FileNotFoundError(f"Object-store model path has no {_MODEL_CONFIG_FILENAME}: {model}")

    cache_root = staging_root or Path(tempfile.gettempdir())
    destination = cache_root / _LOCAL_MODEL_CACHE_DIR / hashlib.sha256(model.encode()).hexdigest()[:16]
    logger.info("Staging %d object-store model entries from %s to %s", len(remote_entries), model, destination)
    copied = 0
    remote_root_path = PurePosixPath(remote_root)
    for remote_file, info in sorted(remote_entries.items()):
        if info.get("type") == "directory":
            continue
        try:
            relative = PurePosixPath(remote_file).relative_to(remote_root_path)
        except ValueError as exc:
            raise ValueError(f"Object-store listing escaped model prefix: {remote_file}") from exc
        if ".." in relative.parts:
            raise ValueError(f"Object-store listing escaped model prefix: {remote_file}")
        local_file = destination.joinpath(*relative.parts)
        expected_size = info.get("size")
        if expected_size is not None and local_file.is_file() and local_file.stat().st_size == expected_size:
            continue
        local_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = local_file.with_suffix(f"{local_file.suffix}.partial")
        filesystem.get_file(remote_file, str(temporary_file))
        os.replace(temporary_file, local_file)
        copied += 1

    logger.info("Staged object-store model at %s (%d files copied)", destination, copied)
    return str(destination)
