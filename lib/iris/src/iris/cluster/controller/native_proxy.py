# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle boundary for the Iris native proxy companion wheel."""

import json
from dataclasses import dataclass

import iris_native as _native  # pyrefly: ignore[missing-import]

DECISION_SECRET_HEADER = _native.DECISION_SECRET_HEADER
DEFAULT_PROXY_TIMEOUT_SECONDS = _native.DEFAULT_PROXY_TIMEOUT_SECONDS
PROXY_DECISION_PATH = _native.PROXY_DECISION_PATH
PROXY_METHODS = tuple(_native.PROXY_METHODS)
PROXY_PREFIX_HEADER = _native.PROXY_PREFIX_HEADER
PROXY_TIMEOUT_HEADER = _native.PROXY_TIMEOUT_HEADER
UPSTREAM_AUTHORIZATION_HEADER = _native.UPSTREAM_AUTHORIZATION_HEADER
UPSTREAM_URL_HEADER = _native.UPSTREAM_URL_HEADER
NativeProxy = _native.NativeProxy


@dataclass(frozen=True, slots=True)
class NativeProxyStats:
    """Native endpoint-registry and JWT-cache counters."""

    registry_generation: int
    endpoint_count: int
    jwt_cache_hits: int
    jwt_cache_misses: int

    @classmethod
    def from_json(cls, payload: str) -> "NativeProxyStats":
        return cls(**json.loads(payload))
