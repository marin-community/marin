# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional, cast

import jax
from rigging.filesystem import url_to_fs

from levanter.utils.fsspec_utils import join_path

_AUTOTUNE_CACHE_SUBDIR = "levanter_kernel_autotune"


def is_enabled_from_env(env_var: str, default: bool = True) -> bool:
    """Read a boolean-ish env var used to gate autotuning behavior."""
    import os

    value = os.environ.get(env_var)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def get_jax_compilation_cache_dir() -> str | None:
    """Return the configured JAX compilation cache directory, if present."""
    cache_dir: str | None = None
    read = getattr(jax.config, "read", None)
    if callable(read):
        try:
            cache_dir = cast(Optional[str], read("jax_compilation_cache_dir"))
        except Exception:
            cache_dir = None
    if not cache_dir:
        values = getattr(jax.config, "values", None)
        if isinstance(values, dict):
            value = values.get("jax_compilation_cache_dir")
            if isinstance(value, str) and value:
                cache_dir = value
    return cache_dir


def kernel_autotune_cache_url(*, kernel_name: str, filename: str) -> str | None:
    """Build a kernel-specific autotune cache URL under the JAX compilation cache root."""
    cache_dir = get_jax_compilation_cache_dir()
    if not cache_dir:
        return None
    return join_path(join_path(join_path(cache_dir, _AUTOTUNE_CACHE_SUBDIR), kernel_name), filename)


def load_json(url: str) -> dict[str, Any]:
    """Load JSON payload from a local or remote URL. Returns empty dict on missing path."""
    fs, path = url_to_fs(url)
    if not fs.exists(path):
        return {}
    with fs.open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, Any], payload)


def write_json(url: str, payload: dict[str, Any]) -> None:
    """Write JSON payload to a local or remote URL."""
    fs, path = url_to_fs(url)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as f:
        json.dump(payload, f, sort_keys=True)
