# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for GLM structured tool responses."""

from __future__ import annotations

import json
from typing import Any


def tool_arguments(response_body: dict[str, Any], function_name: str) -> dict[str, Any]:
    """Return the arguments from one required tool call."""

    calls = response_body["choices"][0]["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != function_name:
        raise ValueError(f"expected exactly one {function_name} tool call")
    return json.loads(calls[0]["function"]["arguments"])
