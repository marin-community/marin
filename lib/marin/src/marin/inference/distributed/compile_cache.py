# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM / JAX XLA compile-cache configuration.

JAX and vLLM both read ``JAX_COMPILATION_CACHE_DIR`` and ``VLLM_XLA_CACHE_PATH``
at engine-construction time and transparently fall back to a fresh compile
when nothing is cached. The simplest way to share compiled artifacts across
workers in the same region is to point those env vars at a region-local TTL
GCS prefix. JAX handles the read/write paths without our help.

The env vars must be set on the **worker** process (where vLLM compiles),
not on the regional CPU coordinator that submits the worker job. The
regional coordinator threads them through Fray's ``EnvironmentConfig.env_vars``
to the worker submission; see ``regional_job._build_context``.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import MutableMapping

from marin.rl.placement import marin_prefix_for_region

from .config import ModelSpec

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = "{region_prefix}/tmp/ttl=30d/vllm-cache/{model_hash}"


def model_cache_hash(model_spec: ModelSpec, region: str) -> str:
    """Stable short hash identifying (resolved model, engine_kwargs).

    A different engine_kwargs (e.g. different `tensor_parallel_size`) compiles
    different XLA programs, so the cache key includes them.
    """
    resolved = model_spec.resolve_for_region(region)
    payload = repr((resolved, sorted(model_spec.engine_kwargs.items())))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def resolve_cache_uri(
    model_spec: ModelSpec,
    region: str,
    template: str | None,
) -> str | None:
    """Return the regional compile-cache URI for a (model, region), or None.

    ``template`` supports two placeholders: ``{region_prefix}`` (the worker's
    ``gs://marin-{region}`` path) and ``{model_hash}``. When ``template`` is
    None, the function returns the canonical default location under the
    region's 30-day TTL scratch prefix. An empty string template explicitly
    disables the compile cache and returns None.
    """
    if template == "":
        return None
    if template is None:
        template = _DEFAULT_TEMPLATE
    region_prefix = marin_prefix_for_region(region)
    return template.format(region_prefix=region_prefix, model_hash=model_cache_hash(model_spec, region))


def configure_env(cache_uri: str | None, env: MutableMapping[str, str] | None = None) -> None:
    """Set ``JAX_COMPILATION_CACHE_DIR`` and ``VLLM_XLA_CACHE_PATH`` to ``cache_uri``.

    No-op when ``cache_uri`` is None. Uses ``setdefault`` semantics on the
    supplied ``env`` mapping (or ``os.environ``): callers can pre-set the env
    vars themselves and we will not override them.
    """
    if cache_uri is None:
        logger.debug("No compile-cache URI; leaving env unchanged.")
        return
    target_env = env if env is not None else os.environ
    target_env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    target_env.setdefault("JAX_COMPILATION_CACHE_DIR", cache_uri)
    target_env.setdefault("VLLM_XLA_CACHE_PATH", cache_uri)
    logger.info("Configured XLA compile cache at %s", cache_uri)
