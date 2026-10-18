# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's mini-swe-agent context-limited model, with its API base read from the environment.

Uploaded into the sandbox next to Harbor's ``mini_swe_model.py``. The API base is the Iris capability
URL (a secret). Harbor passes it as ``-c model.api_base=...``, which puts it in the process command line,
in mini-swe-agent's setup log (``/logs/agent/mini-swe-agent.txt``, which the agent can read) and in the
saved trajectory. This class takes it from ``HOSTED_VLLM_API_BASE``, which Harbor already sets and
litellm already uses for the completion itself, so the flag can be dropped.
"""

import os
from typing import Any

from mini_swe_model import ContextLimitedModel

API_BASE_ENV = "HOSTED_VLLM_API_BASE"


class EnvApiBaseModel(ContextLimitedModel):
    def __init__(self, *, api_base: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_base=api_base or os.environ[API_BASE_ENV], **kwargs)

    def serialize(self) -> dict[str, Any]:
        result = super().serialize()
        result["info"]["config"]["model"]["api_base"] = f"${API_BASE_ENV}"
        return result
