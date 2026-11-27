# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child environment defaults for distributed RunAI checkpoint loading."""

import json
from collections.abc import Mapping, Sequence

_LOADER_CONFIG_FLAG = "--model-loader-extra-config"
_DISTRIBUTED_DEFAULTS = (
    ("concurrency", "RUNAI_STREAMER_CONCURRENCY", "4"),
    ("memory_limit", "RUNAI_STREAMER_MEMORY_LIMIT", str(16 << 30)),
)


def distributed_streamer_environment(extra_args: Sequence[str], environment: Mapping[str, str]) -> dict[str, str]:
    """Default distributed loader buffers without rewriting argv or explicit settings."""
    value = None
    for position, token in enumerate(extra_args):
        flag, separator, inline_value = token.partition("=")
        if flag != _LOADER_CONFIG_FLAG:
            continue
        if separator:
            value = inline_value
        else:
            if position + 1 == len(extra_args):
                raise ValueError(f"{_LOADER_CONFIG_FLAG} requires a JSON value")
            value = extra_args[position + 1]

    child_environment = dict(environment)
    if value is None:
        return child_environment
    config = json.loads(value)
    if not isinstance(config, dict) or config.get("distributed") is not True:
        return child_environment
    for config_key, environment_key, default in _DISTRIBUTED_DEFAULTS:
        if config_key not in config:
            child_environment.setdefault(environment_key, default)
    return child_environment
