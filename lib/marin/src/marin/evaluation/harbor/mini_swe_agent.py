# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent  # pyrefly: ignore[missing-import]
from harbor.environments.base import BaseEnvironment  # pyrefly: ignore[missing-import]

from marin.evaluation.harbor.agent_context import MAX_INPUT_TOKENS_KEY, MAX_OUTPUT_TOKENS_KEY

MODEL_DIRECTORY = "/opt/marin-mini-swe"
MODEL_FILES = ("mini_swe_model.py", "mini_swe_request.py")
HARNESS_VERSION = "2.1.0"


class ContextLimitedMiniSweAgent(MiniSweAgent):
    """Install the native Mini-SWE harness with a rendered-context budget guard."""

    def __init__(
        self,
        *args: Any,
        max_context_tokens: int,
        model_info: dict[str, Any],
        api_base: str,
        version: str,
        config_file: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if version != HARNESS_VERSION or config_file is not None:
            raise ValueError("The context-limited Mini-SWE adapter requires the native 2.1.0 configuration")
        if min(max_context_tokens, model_info[MAX_INPUT_TOKENS_KEY], model_info[MAX_OUTPUT_TOKENS_KEY]) < 1:
            raise ValueError("Mini-SWE token budgets must be positive")
        super().__init__(
            *args,
            api_base=api_base,
            version=version,
            extra_env={**(extra_env or {}), "PYTHONPATH": MODEL_DIRECTORY},
            **kwargs,
        )
        self.max_context_tokens = max_context_tokens
        self.max_input_tokens = model_info[MAX_INPUT_TOKENS_KEY]
        self.max_output_tokens = model_info[MAX_OUTPUT_TOKENS_KEY]
        self.api_base = api_base

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await self.exec_as_root(environment, command=f"mkdir -p {shlex.quote(MODEL_DIRECTORY)}")
        for filename in MODEL_FILES:
            source = Path(__file__).with_name(filename)
            await environment.upload_file(source, f"{MODEL_DIRECTORY}/{filename}")

    def build_cli_flags(self) -> str:
        overrides = {
            "model.model_class": "mini_swe_model.ContextLimitedModel",
            "model.api_base": self.api_base,
            "model.max_context_tokens": self.max_context_tokens,
            "model.max_input_tokens": self.max_input_tokens,
            "model.max_output_tokens": self.max_output_tokens,
        }
        flags = " ".join(f"-c {shlex.quote(f'{key}={value}')}" for key, value in overrides.items())
        return f"{super().build_cli_flags()} -c mini.yaml {flags}"
