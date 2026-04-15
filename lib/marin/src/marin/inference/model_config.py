# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for launching or querying an inference model."""

    name: str
    path: str | None
    engine_kwargs: dict[str, Any]
    generation_params: dict | None = None
    apply_chat_template: bool = False
    base_eval_run_name: str | None = None
