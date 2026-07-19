# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers."""

from config import BridgeConfig


def bridge_config(cache_ttl: float = 20.0) -> BridgeConfig:
    return BridgeConfig(
        max_rows=1000,
        cache_ttl=cache_ttl,
        query_timeout_ms=5000,
        iris_cache_ttl=15.0,
        github_cache_ttl=60.0,
        k8s_cache_ttl=30.0,
        http_timeout=5.0,
        github_token=None,
        cw_read_token=None,
    )
