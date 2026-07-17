# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any, cast

from rigging.filesystem import StoragePath, url_to_fs


def is_enabled_from_env(env_var: str, default: bool = True) -> bool:
    """Read a boolean-ish env var used to gate autotuning behavior."""
    value = os.environ.get(env_var)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_json(url: str) -> dict[str, Any]:
    """Load JSON payload from a local or remote URL. Returns empty dict on missing path."""
    path = StoragePath(url)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
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
