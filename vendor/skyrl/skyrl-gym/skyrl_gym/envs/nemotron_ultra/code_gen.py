# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code-format rules for the released NVIDIA row schema."""

from __future__ import annotations

from typing import Any

DEFAULT_PER_TEST_TIMEOUT_SECONDS = 10


def has_reasoning_format_violation(text: str, assistant_message: dict[str, Any] | None) -> bool:
    """Match NVIDIA's malformed ``<think>``-tag penalty."""
    if "<think>" in text or "</think>" in text:
        return True
    reasoning = (assistant_message or {}).get("reasoning_content", "")
    return isinstance(reasoning, str) and (reasoning.count("<think>") > 1 or reasoning.count("</think>") > 1)
